# Adapted from FakeVLM/arguments.py — simplified ModelArguments, added FreqArguments.

from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_hf_path: str = field(
        default="llava-hf/llava-1.5-7b-hf",
        metadata={"help": "HuggingFace model path (for processor/tokenizer)."},
    )
    model_local_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path to model weights. Defaults to model_hf_path."},
    )

    def __post_init__(self):
        if self.model_local_path is None:
            self.model_local_path = self.model_hf_path


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data JSON file."}
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation data JSON file."}
    )
    image_folder: Optional[str] = field(default=None)
    user_key: Optional[str] = field(default="human")
    assistant_key: Optional[str] = field(default="gpt")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(default=1024)
    use_flash_attn: bool = field(default=False)
    mask_question_tokens: bool = field(default=True)

    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False


@dataclass
class LoraArguments:
    use_lora: bool = field(default=True)
    q_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = "none"


@dataclass
class FreqArguments:
    freq_extractor_name: str = field(
        default="fft",
        metadata={"help": "Frequency extractor to use. See extractors/ for options."},
    )
    freq_input_size: int = field(default=224)
    freq_pool_size: int = field(default=32)
    num_freq_tokens: int = field(default=1)
    freq_projector_hidden_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Hidden dim for freq projector MLP. Defaults to extractor output_dim."},
    )
    training_stage: int = field(
        default=1,
        metadata={"help": "1 = train freq projector only, 2 = LoRA on LLM + freq projector."},
    )
    freq_projector_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to freq_projector weights from Stage 1 (for Stage 2)."},
    )
