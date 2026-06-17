"""
LoRA training utilities for Stable Diffusion 1.5.

Reads (image_path, caption) pairs from train.csv, applies low-rank
adaptation to UNet AND text encoder, and trains with the standard
diffusion denoising objective (predict noise added to latents).

Design choices:

  * LoRA on UNet + text encoder. The Experiment 2 finding (CLIP binding
    blindness) suggests part of the failure is upstream of the U-Net —
    the text encoder itself loses binding information. Adapting both
    catches that.

  * peft library for LoRA. Saves us re-implementing rank decomposition,
    target_modules selection, and safetensors serialization. The trained
    LoRA loads back into diffusers without custom code.

  * Resolution preserved via Resize(shortest=512) + CenterCrop(512). Forced
    stretch to 512x512 distorts object geometry; the chosen transform is
    in-distribution for SD 1.5 (its training used this same crop).

  * Gradient checkpointing on the U-Net. Trades ~30% compute for memory
    headroom — important when both U-Net and text encoder are trainable.

Tests cover the dataset code path that doesn't need GPU. Training itself
is validated empirically through the periodic snapshot generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch  
    from torch.utils.data import Dataset 

@dataclass
class TrainConfig:
    """All hyperparameters in one place — easy to inspect, easy to log."""
    train_csv: Path = Path("data/finetuning/train.csv")
    resolution: int = 512

    base_model_id: str = "runwayml/stable-diffusion-v1-5"

    lora_rank: int = 16
    lora_alpha: int = 16              
    lora_dropout: float = 0.0
    unet_target_modules: tuple = ("to_q", "to_k", "to_v", "to_out.0")
    text_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "out_proj")

    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    epochs: int = 10
    batch_size: int = 4
    grad_accum_steps: int = 2
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"        
    gradient_checkpointing: bool = True

    val_every_steps: int = 100
    val_prompts: tuple = (
        "a blue banana",
        "a brown apple",
        "a purple chalkboard",
        "an orange polar bear",
        "a red apple",
        "a green frog",
    )
    val_seed: int = 0
    val_inference_steps: int = 25
    val_guidance: float = 7.5

    seed: int = 42

    output_dir: Path = Path("/content/lora_output")

def make_train_dataset(csv_path: Path, resolution: int):
    """
    Build the torch.utils.data.Dataset that reads the recaption manifest.

    Each item is (pixel_values_tensor, caption_string). Caption tokenization
    happens in the training loop, not here, so the dataset stays portable
    if the tokenizer changes.

    The dataset preserves aspect ratio: short side becomes `resolution`,
    then center crop. This matches SD 1.5's training distribution.

    Image errors (corrupt files, unreadable formats) are caught and the
    sample is replaced with a re-draw from the dataset. Important for
    LAION which occasionally has CMYK or unusual formats.
    """
    
    import csv
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),   
    ])

    class TrainCSVDataset(Dataset):
        def __init__(self, csv_path: Path):
            with Path(csv_path).open(newline="", encoding="utf-8") as f:
                self.rows = list(csv.DictReader(f))
            if not self.rows:
                raise ValueError(f"train.csv at {csv_path} is empty")

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            row = self.rows[idx]
            try:
                img = Image.open(row["image_path"]).convert("RGB")
                pixel_values = transform(img)
            except Exception:
                fallback_idx = (idx + 1) % len(self.rows)
                row = self.rows[fallback_idx]
                img = Image.open(row["image_path"]).convert("RGB")
                pixel_values = transform(img)
            return {
                "pixel_values": pixel_values,
                "caption": row["caption"],
                "role": row.get("role", ""),
                "object": row.get("object", ""),
                "color": row.get("color", ""),
            }

    return TrainCSVDataset(csv_path)

def make_collate_fn(tokenizer):
    """
    Returns a collate_fn that tokenizes captions and stacks pixel values.

    Token IDs are padded to the tokenizer's max length (77 for CLIP),
    which is what SD 1.5's text encoder expects.
    """
    import torch

    def collate(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        captions = [b["caption"] for b in batch]
        tokens = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
        }

    return collate

def apply_lora_to_pipeline(pipe, config: TrainConfig):
    """
    Inject LoRA adapters into the pipeline's UNet AND text encoder.

    After this call:
      - pipe.unet has LoRA layers, original weights frozen
      - pipe.text_encoder has LoRA layers, original weights frozen
      - pipe.vae remains fully frozen (we don't adapt the autoencoder)

    Returns the list of trainable parameters for the optimizer.
    """
    from peft import LoraConfig, get_peft_model

    unet_lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.unet_target_modules),
        init_lora_weights="gaussian",
    )
    pipe.unet = get_peft_model(pipe.unet, unet_lora_config)

    text_lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.text_target_modules),
        init_lora_weights="gaussian",
    )
    pipe.text_encoder = get_peft_model(pipe.text_encoder, text_lora_config)

    for p in pipe.vae.parameters():
        p.requires_grad = False

    trainable_params = []
    n_trainable = 0
    n_total = 0
    for p in pipe.unet.parameters():
        n_total += p.numel()
        if p.requires_grad:
            trainable_params.append(p)
            n_trainable += p.numel()
    for p in pipe.text_encoder.parameters():
        n_total += p.numel()
        if p.requires_grad:
            trainable_params.append(p)
            n_trainable += p.numel()

    return trainable_params, n_trainable, n_total
