# Adapted from FakeVLM_extended/collator.py — removed frequency-domain
# preprocessing (no freq_extractor, no freq_pixel_values, 576 CLIP tokens).

import logging
import re
from inspect import isfunction
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import PIL
import torch
from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizer

logging.getLogger("transformers.processing_utils").setLevel(logging.ERROR)
from transformers.image_utils import get_image_size, to_numpy_array
from transformers.models.llava.processing_llava import LlavaProcessorKwargs
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import TensorType, get_json_schema, logging
from transformers.utils.chat_template_utils import (
    _compile_jinja_template,
    _render_with_assistant_indices,
)


logger = logging.get_logger(__name__)

IGNORE_TOKEN_ID = -100


# ---------------------------------------------------------------------------
# Chat template monkey patch (from FakeVLM/collators/chat_template_monkey_patch.py)
# Sets add_special_tokens=True in the tokenizer call.
# ---------------------------------------------------------------------------

def _apply_chat_template(
    self,
    conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
    tools: Optional[List[Dict]] = None,
    documents: Optional[List[Dict[str, str]]] = None,
    chat_template: Optional[str] = None,
    add_generation_prompt: bool = False,
    continue_final_message: bool = False,
    tokenize: bool = True,
    padding: bool = False,
    truncation: bool = False,
    max_length: Optional[int] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    return_dict: bool = False,
    return_assistant_tokens_mask: bool = False,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[str, List[int], List[str], List[List[int]], BatchEncoding]:
    if return_dict and not tokenize:
        raise ValueError(
            "`return_dict=True` is incompatible with `tokenize=False`."
        )
    if return_assistant_tokens_mask and not return_dict:
        raise ValueError(
            "`return_assistant_tokens_mask=True` requires `return_dict=True`."
        )
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    chat_template = self.get_chat_template(chat_template, tools)

    if return_assistant_tokens_mask and not re.search(
        r"\{\%-?\s*generation\s*-?\%\}", chat_template
    ):
        logger.warning_once(
            "return_assistant_tokens_mask==True but chat template has no "
            "`{% generation %}` keyword."
        )

    compiled_template = _compile_jinja_template(chat_template)

    if isinstance(conversation, (list, tuple)) and (
        isinstance(conversation[0], (list, tuple))
        or hasattr(conversation[0], "messages")
    ):
        conversations = conversation
        is_batched = True
    else:
        conversations = [conversation]
        is_batched = False

    if continue_final_message:
        if add_generation_prompt:
            raise ValueError(
                "continue_final_message and add_generation_prompt are "
                "mutually exclusive."
            )
        if return_assistant_tokens_mask:
            raise ValueError(
                "continue_final_message is not compatible with "
                "return_assistant_tokens_mask."
            )

    if tools is not None:
        tool_schemas = []
        for tool in tools:
            if isinstance(tool, dict):
                tool_schemas.append(tool)
            elif isfunction(tool):
                tool_schemas.append(get_json_schema(tool))
            else:
                raise ValueError(
                    "Tools should be a JSON schema or a callable with type "
                    "hints and a docstring."
                )
    else:
        tool_schemas = None

    if documents is not None:
        for document in documents:
            if not isinstance(document, dict):
                raise TypeError(
                    "Documents should be a list of dicts with 'title' and "
                    "'text' keys."
                )

    rendered = []
    all_generation_indices = []
    template_kwargs = {**self.special_tokens_map, **kwargs}
    for chat in conversations:
        if hasattr(chat, "messages"):
            chat = chat.messages
        if return_assistant_tokens_mask:
            rendered_chat, generation_indices = _render_with_assistant_indices(
                compiled_template=compiled_template,
                messages=chat,
                tools=tool_schemas,
                documents=documents,
                add_generation_prompt=add_generation_prompt,
                **template_kwargs,
            )
            all_generation_indices.append(generation_indices)
        else:
            rendered_chat = compiled_template.render(
                messages=chat,
                tools=tool_schemas,
                documents=documents,
                add_generation_prompt=add_generation_prompt,
                **template_kwargs,
            )
        if continue_final_message:
            final_message = chat[-1]["content"].strip()
            rendered_chat = rendered_chat[
                : rendered_chat.rindex(final_message) + len(final_message)
            ].rstrip()
        rendered.append(rendered_chat)

    if not is_batched:
        rendered = rendered[0]

    if tokenize:
        out = self(
            rendered,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=True,
            return_tensors=return_tensors,
            **tokenizer_kwargs,
        )
        if return_dict:
            if return_assistant_tokens_mask:
                assistant_masks = []
                if is_batched or return_tensors:
                    input_ids = out["input_ids"]
                else:
                    input_ids = [out["input_ids"]]
                for i in range(len(input_ids)):
                    current_mask = [0] * len(input_ids[i])
                    for start_char, end_char in all_generation_indices[i]:
                        start_token = out.char_to_token(i, start_char)
                        end_token = out.char_to_token(i, end_char - 1)
                        if start_token is None:
                            break
                        for token_id in range(
                            start_token,
                            end_token + 1 if end_token else len(input_ids[i]),
                        ):
                            current_mask[token_id] = 1
                    assistant_masks.append(current_mask)
                out["assistant_masks"] = (
                    assistant_masks if is_batched else assistant_masks[0]
                )
            return out
        else:
            return out["input_ids"]
    else:
        return rendered


# ---------------------------------------------------------------------------
# LLaVA collator (no frequency features)
# ---------------------------------------------------------------------------


class LlavaCollator:
    """Data collator for vanilla LLaVA 1.5 / FakeVLM fine-tuning.

    Reimplements the LLaVA 1.5 collator from FakeVLM without frequency-domain
    support. Image token expansion uses 576 CLIP tokens only.
    """

    def __init__(
        self,
        config: AutoConfig,
        tokenizer: PreTrainedTokenizer,
        processor: AutoProcessor,
        mask_question_tokens: bool = True,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.mask_question_tokens = mask_question_tokens

    def __call__(
        self, instances: Sequence[Dict]
    ) -> Dict[str, torch.Tensor]:
        self.tokenizer.apply_chat_template = _apply_chat_template.__get__(
            self.tokenizer
        )

        output_kwargs = self.processor._merge_kwargs(
            LlavaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        )

        all_raw_images: List[PIL.Image.Image] = [
            img
            for instance in instances
            for img in instance["images"]
        ]

        vision_inputs = dict()
        if len(all_raw_images) > 0:
            vision_inputs.update(
                **self.processor.image_processor(
                    all_raw_images,
                    return_tensors="pt",
                    **output_kwargs["images_kwargs"],
                )
            )

        images_per_instance: List[List[PIL.Image.Image]] = [
            instance["images"] for instance in instances
        ]
        system_prompts: List[Optional[str]] = [
            instance["system_prompt"] for instance in instances
        ]
        conversations: List[List] = [
            instance["conversations"] for instance in instances
        ]

        max_len = self.tokenizer.model_max_length
        image_token_id = self.config.image_token_index
        patch_size = self.processor.patch_size
        vision_feature_select_strategy = (
            self.processor.vision_feature_select_strategy
        )
        pad_token_id = self.tokenizer.pad_token_id

        input_ids = []
        labels = []

        for system_prompt, cur_images, cur_convs in zip(
            system_prompts, images_per_instance, conversations
        ):
            cur_num_images = 0
            cur_text = []

            if system_prompt is not None:
                cur_text.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    }
                )

            for i, text in enumerate(cur_convs):
                if i % 2 == 0:
                    num_images = len(
                        [m.start() for m in re.finditer("<image>", text)]
                    )
                    cur_num_images += num_images
                    text = text.replace("<image>", "").strip()
                    cur_text.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": text}]
                            + [{"type": "image"}] * num_images,
                        }
                    )
                else:
                    cur_text.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": text}],
                        }
                    )

            assert len(cur_images) == cur_num_images

            temp = self.processor.apply_chat_template(
                cur_text,
                add_generation_prompt=False,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False,
            )
            cur_input_ids = temp["input_ids"]
            cur_assistant_masks = torch.as_tensor(
                temp["assistant_masks"], dtype=torch.bool
            )
            if cur_assistant_masks.dim() == 1:
                cur_assistant_masks = cur_assistant_masks.unsqueeze(0)

            # Expand image tokens (576 CLIP tokens)
            temp_vision_inputs = self.processor.image_processor(
                cur_images, return_tensors="pt"
            )
            if temp_vision_inputs.get("pixel_values") is not None:
                if (
                    patch_size is not None
                    and vision_feature_select_strategy is not None
                ):
                    pixel_values = temp_vision_inputs["pixel_values"]
                    height, width = get_image_size(
                        to_numpy_array(pixel_values[0])
                    )
                    num_image_tokens = (
                        (height // patch_size) * (width // patch_size) + 1
                    )
                    if vision_feature_select_strategy == "default":
                        num_image_tokens -= 1

                    repeat = torch.where(
                        cur_input_ids == image_token_id,
                        num_image_tokens,
                        1,
                    ).squeeze()
                    cur_input_ids = cur_input_ids.repeat_interleave(
                        repeat, dim=1
                    )
                    cur_assistant_masks = cur_assistant_masks.repeat_interleave(
                        repeat, dim=1
                    )

            # Include EOS as part of labels
            cur_assistant_masks[0, -1] = True

            # Truncation
            if cur_input_ids.shape[1] > max_len:
                cur_input_ids = cur_input_ids[:, :max_len]
                cur_assistant_masks = cur_assistant_masks[:, :max_len]

            cur_labels = cur_input_ids.clone()
            if self.mask_question_tokens:
                cur_labels = torch.where(
                    cur_assistant_masks, cur_labels, IGNORE_TOKEN_ID
                )

            # Padding
            if cur_input_ids.shape[1] < max_len:
                pad_len = max_len - cur_input_ids.shape[1]
                cur_input_ids = torch.cat(
                    [
                        cur_input_ids,
                        torch.full(
                            (cur_input_ids.shape[0], pad_len),
                            pad_token_id,
                            dtype=cur_input_ids.dtype,
                        ),
                    ],
                    dim=1,
                )
                cur_labels = torch.cat(
                    [
                        cur_labels,
                        torch.full(
                            (cur_labels.shape[0], pad_len),
                            IGNORE_TOKEN_ID,
                            dtype=cur_labels.dtype,
                        ),
                    ],
                    dim=1,
                )

            input_ids.append(cur_input_ids)
            labels.append(cur_labels)

        input_ids = torch.cat(input_ids)
        labels = torch.cat(labels)

        return dict(
            **vision_inputs,
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(pad_token_id),
        )
