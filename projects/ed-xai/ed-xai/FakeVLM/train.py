import json
import os
os.environ["WANDB_PROJECT"] = "fakevlm-ablation"

from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
from transformers import BitsAndBytesConfig
from transformers.integrations import is_deepspeed_zero3_enabled

from .arguments import (
    AblationArguments,
    DataArguments,
    LoraArguments,
    ModelArguments,
    TrainingArguments,
)
from .collator import LlavaCollator
from .dataset import LazySupervisedDataset
from .loader import load_model
from .utils import (
    TrainerWithCustomSampler,
    find_all_linear_names,
    rank0_print,
    safe_save_model_for_hf_trainer,
)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments, AblationArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        ablation_args,
    ) = parser.parse_args_into_dataclasses()

    output_dir = training_args.output_dir
    args_dir = Path(output_dir) / "arguments"
    args_dir.mkdir(parents=True, exist_ok=True)
    yaml.dump(asdict(model_args), open(args_dir / "model.yaml", "w"))
    yaml.dump(asdict(data_args), open(args_dir / "data.yaml", "w"))
    yaml.dump(asdict(training_args), open(args_dir / "training.yaml", "w"))
    yaml.dump(asdict(lora_args), open(args_dir / "lora.yaml", "w"))
    yaml.dump(asdict(ablation_args), open(args_dir / "ablation.yaml", "w"))

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

    # --- Freeze all parameters ---
    rank0_print("Freezing all parameters...")
    model.requires_grad_(False)

    # --- LoRA ---
    if lora_args.use_lora:
        rank0_print("Applying LoRA to language model...")
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

        if ablation_args.train_projector:
            rank0_print("Unfreezing CLIP projector (multi_modal_projector)...")
            for p in (
                model.base_model.model.model.multi_modal_projector.parameters()
            ):
                p.requires_grad_(True)
    else:
        rank0_print("No LoRA applied (use_lora=False).")
        if ablation_args.train_projector:
            rank0_print("Unfreezing CLIP projector (multi_modal_projector)...")
            model.model.multi_modal_projector.requires_grad_(True)

    # Print trainable parameters
    rank0_print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(f"\t{name}")

    # --- Data ---
    rank0_print("Loading data...")
    dataset_kwargs = dict(
        image_folder=data_args.image_folder,
        user_key=data_args.user_key,
        assistant_key=data_args.assistant_key,
    )
    eval_dataset = None

    if data_args.eval_data_path:
        train_dataset = LazySupervisedDataset(
            data_path=data_args.data_path, **dataset_kwargs
        )
        eval_dataset = LazySupervisedDataset(
            data_path=data_args.eval_data_path, **dataset_kwargs
        )
    elif data_args.eval_split_ratio > 0:
        with open(data_args.data_path) as f:
            all_data = json.load(f)
        n_eval = int(len(all_data) * data_args.eval_split_ratio)
        gen = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(all_data), generator=gen).tolist()
        eval_data = [all_data[i] for i in indices[:n_eval]]
        train_data = [all_data[i] for i in indices[n_eval:]]
        rank0_print(f"Split: {len(train_data)} train, {len(eval_data)} eval")
        train_dataset = LazySupervisedDataset(data=train_data, **dataset_kwargs)
        eval_dataset = LazySupervisedDataset(data=eval_data, **dataset_kwargs)
    else:
        train_dataset = LazySupervisedDataset(
            data_path=data_args.data_path, **dataset_kwargs
        )

    if eval_dataset is None:
        training_args.eval_strategy = "no"

    data_collator = LlavaCollator(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
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

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_dir)


if __name__ == "__main__":
    train()
