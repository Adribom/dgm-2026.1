# Adapted from FakeVLM_extended/arguments.py — removed FreqArguments,
# added AblationArguments for vanilla FakeVLM fine-tuning.

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
    eval_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Fraction of training data to hold out for eval (ignored if eval_data_path is set)."},
    )


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
class AblationArguments:
    train_projector: bool = field(
        default=False,
        metadata={"help": "Unfreeze the CLIP projector MLP during LoRA training."},
    )
