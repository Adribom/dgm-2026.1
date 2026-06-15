"""
Stable Diffusion image generation pipeline.

Thin wrapper around `diffusers` that:
  - loads SD 1.5 with the scheduler specified in the config,
  - generates images with per-image deterministic seeds,
  - exposes a single `generate_one` function that the experiment
    script can call for each (object, color, seed) triple.

The class is intentionally minimal: it does NOT decide what to
generate, where to save, or how many. Those decisions live in
`experiments/exp3_generate_eval.py`, keeping the generator a pure
"give me an image for this prompt + seed" service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass(frozen=True)
class GenerationParams:
    """Generation hyperparameters that must stay constant across before/after runs."""

    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    negative_prompt: str = ""


class SDGenerator:
    """
    Loads a Stable Diffusion pipeline once and serves per-image generations.

    Why a class and not a function: pipeline loading is expensive (~10s and
    several GB of GPU memory). The script creates one `SDGenerator` and
    reuses it for all 3600 images.

    Args:
        model_id: HuggingFace repo id, e.g. "runwayml/stable-diffusion-v1-5".
        scheduler_name: class name from diffusers.schedulers,
            e.g. "DPMSolverMultistepScheduler".
        dtype: "float16" or "float32". fp16 is required to fit on a Colab T4.
        revision: optional git revision / commit hash for strict pinning.
        device: "cuda" or "cpu". If None, auto-detects.
    """

    def __init__(
        self,
        model_id: str,
        scheduler_name: str = "DPMSolverMultistepScheduler",
        dtype: str = "float16",
        revision: str | None = None,
        device: str | None = None,
    ) -> None:
        # Lazy imports keep this module importable in environments without
        # heavy deps (e.g. when only running tests of pure-Python modules).
        import torch
        from diffusers import StableDiffusionPipeline
        from diffusers import schedulers as diffusers_schedulers

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # fp16 only makes sense on GPU; CPU silently falls back to fp32.
        torch_dtype = torch.float16 if (dtype == "float16" and device == "cuda") else torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            revision=revision,
            safety_checker=None,        # disabled: we want to see actual outputs
            requires_safety_checker=False,
        )

        # Swap in the requested scheduler. The default scheduler in SD 1.5
        # (PNDM) gives noisier results at low step counts; DPM-Solver++ is
        # the modern standard and gives clean images in 25-30 steps.
        scheduler_cls = getattr(diffusers_schedulers, scheduler_name)
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)

        pipe = pipe.to(device)
        # Reduce console noise during the 3600-image run.
        pipe.set_progress_bar_config(disable=True)

        self.pipe = pipe
        self.device = device
        self.dtype = torch_dtype

    def generate_one(
        self,
        prompt: str,
        seed: int,
        params: GenerationParams,
    ) -> "Image":
        """
        Generate a single image deterministically from (prompt, seed, params).

        Two calls with the same arguments on the same hardware produce
        byte-identical images. Across different GPUs the pixel values may
        differ by a tiny amount due to cuDNN nondeterminism in fp16, which
        is why we record per-image hashes downstream — to know when bytes
        match.

        Args:
            prompt: text to encode.
            seed: integer seed for this specific image.
            params: generation hyperparameters (width, steps, etc.).

        Returns:
            A PIL.Image at (params.width, params.height).
        """
        from binding.seeds import seed_generator

        generator = seed_generator(seed)
        # The generator must live on the same device as the pipeline for
        # diffusers to actually use it for noise sampling.
        if self.device == "cuda":
            generator = generator  # CPU generator works fine with diffusers' API

        output = self.pipe(
            prompt=prompt,
            negative_prompt=params.negative_prompt or None,
            width=params.width,
            height=params.height,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            generator=generator,
        )
        return output.images[0]


def params_from_config(generation_cfg: dict[str, Any]) -> GenerationParams:
    """Build a GenerationParams from the `generation:` section of the YAML config."""
    return GenerationParams(
        width=int(generation_cfg["width"]),
        height=int(generation_cfg["height"]),
        num_inference_steps=int(generation_cfg["num_inference_steps"]),
        guidance_scale=float(generation_cfg["guidance_scale"]),
        negative_prompt=generation_cfg.get("negative_prompt", ""),
    )
