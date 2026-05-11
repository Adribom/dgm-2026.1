import os
os.environ["WANDB_PROJECT"] = "fakevlm-extended"

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import yaml
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
from transformers import BitsAndBytesConfig
from transformers.integrations import is_deepspeed_zero3_enabled

from .arguments import (
    DataArguments,
    FreqArguments,
    LoraArguments,
    ModelArguments,
    TrainingArguments,
)
from .collator import ExtendedLlavaCollator
from .dataset import LazySupervisedDataset
from .extractors import get_extractor
from .frequency_projector import FrequencyProjector
from .loader import load_model
from .model import extend_model
from .utils import (
    TrainerWithCustomSampler,
    find_all_linear_names,
    rank0_print,
    safe_save_model_for_hf_trainer,
)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments, FreqArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        freq_args,
    ) = parser.parse_args_into_dataclasses()

    # Dump arguments
    output_dir = training_args.output_dir
    args_dir = Path(output_dir) / "arguments"
    args_dir.mkdir(parents=True, exist_ok=True)
    yaml.dump(asdict(model_args), open(args_dir / "model.yaml", "w"))
    yaml.dump(asdict(data_args), open(args_dir / "data.yaml", "w"))
    yaml.dump(asdict(training_args), open(args_dir / "training.yaml", "w"))
    yaml.dump(asdict(lora_args), open(args_dir / "lora.yaml", "w"))
    yaml.dump(asdict(freq_args), open(args_dir / "freq.yaml", "w"))

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    if getattr(training_args, "deepspeed", None) and lora_args.q_lora:
        training_args.distributed_state.distributed_type = (
            DistributedType.DEEPSPEED
        )

    device_map = None
    if lora_args.q_lora:
        device_map = (
            {"": int(os.environ.get("LOCAL_RANK") or 0)}
            if int(os.environ.get("WORLD_SIZE", 1)) != 1
            else None
        )
        if len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled():
            raise ValueError("FSDP/ZeRO3 incompatible with QLoRA.")

    bnb_config = None
    if lora_args.use_lora and lora_args.q_lora:
        rank0_print("Quantization for LLM enabled...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
        )

    # --- Load base model ---
    rank0_print("Loading model, tokenizer, processor...")
    model, tokenizer, processor, config = load_model(
        model_local_path=model_args.model_local_path,
        model_hf_path=model_args.model_hf_path,
        compute_dtype=compute_dtype,
        bnb_config=bnb_config,
        use_flash_attn=training_args.use_flash_attn,
        device_map=device_map,
    )
    tokenizer.model_max_length = training_args.model_max_length

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # --- Create frequency components ---
    rank0_print(
        f"Creating frequency extractor: {freq_args.freq_extractor_name}"
    )
    freq_extractor = get_extractor(
        freq_args.freq_extractor_name,
        input_size=freq_args.freq_input_size,
        pool_size=freq_args.freq_pool_size,
    )

    llm_hidden_size = config.text_config.hidden_size
    freq_projector = FrequencyProjector(
        input_dim=freq_extractor.output_dim,
        output_dim=llm_hidden_size,
        num_tokens=freq_args.num_freq_tokens,
        hidden_dim=freq_args.freq_projector_hidden_dim,
    )

    # Load freq projector checkpoint (Stage 2)
    if freq_args.freq_projector_checkpoint is not None:
        rank0_print(
            f"Loading freq projector from {freq_args.freq_projector_checkpoint}"
        )
        state = torch.load(
            freq_args.freq_projector_checkpoint, map_location="cpu"
        )
        freq_projector.load_state_dict(state)

    # --- Extend model ---
    rank0_print("Extending model with frequency features...")
    model = extend_model(
        model, freq_extractor, freq_projector, freq_args.num_freq_tokens
    )

    # --- Freeze / unfreeze ---
    rank0_print("Freezing all parameters...")
    model.requires_grad_(False)

    rank0_print("Unfreezing frequency projector...")
    model.model.multi_modal_projector.freq_projector.requires_grad_(True)

    # --- LoRA (Stage 2 only) ---
    if freq_args.training_stage == 2 and lora_args.use_lora:
        rank0_print("Stage 2: applying LoRA to language model...")
        named_modules = {n: m for n, m in model.named_modules()}
        lora_modules = find_all_linear_names(
            named_modules, ["language_model"]
        )

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=training_args.gradient_checkpointing,
            )

        model = get_peft_model(model, lora_config)

        # Re-unfreeze freq projector after PEFT wrapping (PEFT freezes all
        # non-LoRA params).
        for p in (
            model.base_model.model.model.multi_modal_projector.freq_projector.parameters()
        ):
            p.requires_grad_(True)
    elif freq_args.training_stage == 1:
        rank0_print("Stage 1: training frequency projector only (no LoRA).")
    else:
        rank0_print(
            f"Stage {freq_args.training_stage}: no LoRA applied "
            f"(use_lora={lora_args.use_lora})."
        )

    # Print trainable parameters
    rank0_print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(f"\t{name}")

    # --- Data ---
    rank0_print("Loading data...")
    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        user_key=data_args.user_key,
        assistant_key=data_args.assistant_key,
    )
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = LazySupervisedDataset(
            data_path=data_args.eval_data_path,
            image_folder=data_args.image_folder,
            user_key=data_args.user_key,
            assistant_key=data_args.assistant_key,
        )
    else:
        training_args.eval_strategy = "no"

    data_collator = ExtendedLlavaCollator(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        freq_extractor=freq_extractor,
        num_freq_tokens=freq_args.num_freq_tokens,
        mask_question_tokens=training_args.mask_question_tokens,
    )

    # --- Train ---
    trainer = TrainerWithCustomSampler(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_state()

    # Save freq projector weights separately
    freq_proj_path = Path(output_dir) / "freq_projector.pt"
    if freq_args.training_stage == 2 and lora_args.use_lora:
        freq_proj_state = (
            model.base_model.model.model.multi_modal_projector.freq_projector.state_dict()
        )
    else:
        freq_proj_state = (
            model.model.multi_modal_projector.freq_projector.state_dict()
        )
    if trainer.args.should_save:
        torch.save(freq_proj_state, freq_proj_path)
        rank0_print(f"Saved freq_projector to {freq_proj_path}")

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_dir)


if __name__ == "__main__":
    train()
