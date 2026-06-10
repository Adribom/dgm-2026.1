# Adapted from FakeVLM/loaders/base.py and FakeVLM/loaders/llava_1_5.py.

from typing import Dict, Optional, Tuple, Union

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    PreTrainedTokenizer,
)


def load_model(
    model_local_path: str,
    model_hf_path: str,
    compute_dtype: torch.dtype = torch.bfloat16,
    bnb_config: Optional[BitsAndBytesConfig] = None,
    use_flash_attn: bool = False,
    device_map: Optional[Union[Dict, str]] = None,
) -> Tuple[LlavaForConditionalGeneration, PreTrainedTokenizer, AutoProcessor, AutoConfig]:
    loading_kwargs = dict(
        torch_dtype=compute_dtype,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    if use_flash_attn:
        loading_kwargs["attn_implementation"] = "flash_attention_2"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_local_path, **loading_kwargs
    )
    # DeepSpeed compatibility
    model.config.hidden_size = model.model.language_model.config.hidden_size

    processor = AutoProcessor.from_pretrained(model_hf_path, add_eos_token=True)
    tokenizer = processor.tokenizer
    config = AutoConfig.from_pretrained(model_local_path)

    return model, tokenizer, processor, config
