"""
LLM-based caption rewriting for Phase 5 (Pipeline B + control set).

Generates linguistically varied captions for each approved training image,
using an open-source instruction-tuned LLM. The prompt is engineered to:

  * Produce SHORT, GROUNDED captions (no invented context like "on a sunny
    afternoon" — Stable Diffusion 1.5 was trained on LAION-style brief
    captions, not narrative prose).
  * Vary WORDING and STRUCTURE between captions, not just substitute words.
  * Keep the binding signal explicit: the (object, color) pair must be
    syntactically bound in every caption (a "red apple", not "apple, red color").
  * Avoid hallucinated objects or settings not in the image.

Deterministic given a seed: re-running with the same seed produces the same
captions, important for reproducibility.

The default model is Qwen2.5-7B-Instruct (~14 GB VRAM in bfloat16). Cab in
an A100; tight on L4. The pipeline is generic and accepts any HF causal LM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

@dataclass(frozen=True)
class CaptionResult:
    """Output of a single LLM call: 1+ generated captions for one image."""
    raw_output: str
    captions: list[str]


def build_prompt(color: str, obj: str) -> str:
    """
    Build the instruction prompt for the LLM. Few-shot with a different
    object/color to discourage verbatim copying of the examples.

    Captions are short and grounded — SD 1.5 was trained on LAION captions
    of similar length and style, so matching that distribution helps
    finetuning rather than confusing it.
    """
    a = "an" if obj[0].lower() in "aeiou" else "a"
    a_c = "an" if color[0].lower() in "aeiou" else "a"
    return (
        "You are generating short captions for a Stable Diffusion training "
        "set. Each caption must clearly describe an object and its color, "
        "matching what is visible in the image. Captions should be SHORT "
        "(under 12 words) and GROUNDED: do not invent extra objects, "
        "settings, lighting, weather, or any details that aren't stated. "
        "Vary the wording between captions, but always keep the color "
        "and object clearly bound together.\n\n"
        "Example captions for 'a red apple':\n"
        "A red apple.\n"
        "Red apple on a plain background.\n"
        "Photo of a red apple, close up.\n\n"
        f"Now generate exactly 3 different short captions for '{a_c} {color} {obj}'. "
        "Output one caption per line, no numbering, no quotes, no extra text:"
    )

_CAPTION_LINE_RE = re.compile(r"^\s*(?:[-*\d.)]+\s*)?(.+?)\s*$")
_QUOTE_RE = re.compile(r'^[\'"]+(.*?)[\'"]+$')

def parse_captions(raw_text: str, expected: int = 3) -> list[str]:
    """
    Extract clean caption lines from the LLM output.

    Models sometimes prefix with "Sure, here are 3 captions:", number lines,
    or wrap in quotes. We strip all of that and keep only the textual content.

    If we get fewer than `expected` clean lines, return what we have — the
    caller decides whether to retry or accept the partial result.
    """
    captions = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        
        if line.endswith(":") or line.lower().startswith(("sure", "here", "okay", "of course")):
            continue
        
        m = _CAPTION_LINE_RE.match(line)
        if not m:
            continue
        text = m.group(1).strip()
        qm = _QUOTE_RE.match(text)
        if qm:
            text = qm.group(1).strip()
        
        if len(text.split()) > 18:
            continue
        
        if len(text.split()) < 2:
            continue
        captions.append(text)
        if len(captions) >= expected:
            break
    return captions

def validate_caption(caption: str, obj: str, color: str) -> bool:
    """
    Sanity check: the caption must mention BOTH the object and the color.

    A model that hallucinates ("a red car driving fast") for a banana×blue
    request would silently poison the training set. This check is the cheap
    filter before the caption goes into train.csv.
    """
    text = caption.lower()
    obj_present = obj.lower() in text
    color_present = color.lower() in text
    return obj_present and color_present

class CaptionGenerator:
    """
    Wraps a HF causal LM for caption generation.

    Lazy imports — heavy dependencies (torch, transformers) loaded only on
    first call, so this module can be imported on a CPU-only machine for
    testing the prompt/parse utilities.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype_map.get(self.dtype, torch.bfloat16),
            device_map=self.device,
        )
        self._model.eval()

    def generate(self, obj: str, color: str, seed: int = 0) -> CaptionResult:
        """
        Generate 3 captions for the given pair. Returns CaptionResult with
        the raw model output and the parsed clean captions.
        """
        self._ensure_loaded()
        torch = self._torch

        prompt = build_prompt(color, obj)
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)

        torch.manual_seed(seed)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        gen_tokens = output[0, inputs.input_ids.shape[1]:]
        raw = self._tokenizer.decode(gen_tokens, skip_special_tokens=True)
        captions = parse_captions(raw, expected=3)
        return CaptionResult(raw_output=raw, captions=captions)
