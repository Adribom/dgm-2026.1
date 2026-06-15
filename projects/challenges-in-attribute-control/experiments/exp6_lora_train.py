"""
Phase 6: train LoRA on Stable Diffusion 1.5 with the corrective dataset.

Reads train.csv produced by Phase 5, applies LoRA to UNet + text encoder,
trains for `epochs` with periodic validation snapshots, and saves the
final LoRA weights ready to be loaded back into a SD 1.5 pipeline.

Validation snapshots: every `val_every_steps` steps, generates 6 images
(4 treated prompts + 2 control prompts) with the partially trained model.
Save as a grid for visual inspection — the most informative single signal
about whether training is working.

Output structure:
    <output-dir>/
        lora_unet/                       (final UNet LoRA, peft format)
        lora_text_encoder/               (final text encoder LoRA)
        snapshots/
            step_0100_grid.png
            step_0200_grid.png
            ...
        train_log.csv                    (loss per step)
        config.json                      (full hyperparameter snapshot)

Usage (Colab Pro A100):
    python experiments/exp6_lora_train.py \\
        --train-csv /content/train.csv \\
        --output-dir /content/lora_output
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from binding.lora_train import (  
    TrainConfig,
    apply_lora_to_pipeline,
    make_collate_fn,
    make_train_dataset,
)
from binding.seeds import set_all_seeds  

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("/content/lora_output"))
    p.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def make_validation_grid(images, prompts, save_path: Path):
    """Compose 6 generated images into a 2x3 grid with prompt labels."""
    from PIL import Image, ImageDraw, ImageFont

    W = images[0].width
    H = images[0].height
    label_h = 30
    cols, rows = 3, 2
    grid = Image.new("RGB", (cols * W, rows * (H + label_h)), color="white")
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, (img, prompt) in enumerate(zip(images, prompts)):
        r, c = i // cols, i % cols
        x = c * W
        y = r * (H + label_h)
        draw.text((x + 6, y + 4), prompt[:32], fill=(0, 0, 0), font=font)
        grid.paste(img, (x, y + label_h))

    grid.save(save_path)

def main() -> int:
    args = parse_args()
    set_all_seeds(args.seed)

    import torch
    from accelerate import Accelerator
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        DPMSolverMultistepScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from diffusers.optimization import get_scheduler
    from torch.utils.data import DataLoader
    from transformers import CLIPTextModel, CLIPTokenizer
    from tqdm.auto import tqdm

    cfg = TrainConfig(
        train_csv=args.train_csv,
        base_model_id=args.base_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_rank,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        val_every_steps=args.val_every,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "snapshots").mkdir(exist_ok=True)
    print(f"[train] config: {asdict(cfg)}")

    cfg_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    with (cfg.output_dir / "config.json").open("w") as f:
        json.dump(cfg_dict, f, indent=2)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.grad_accum_steps,
    )
    print(f"[train] accelerator device: {accelerator.device}")
    print(f"[train] loading {cfg.base_model_id}")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.base_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.base_model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.base_model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(cfg.base_model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.base_model_id, subfolder="scheduler")

    class _PipeShim:
        pass
    shim = _PipeShim()
    shim.unet = unet
    shim.text_encoder = text_encoder
    shim.vae = vae

    trainable_params, n_trainable, n_total = apply_lora_to_pipeline(shim, cfg)
    unet = shim.unet
    text_encoder = shim.text_encoder
    print(f"[train] LoRA parameters: {n_trainable:,} trainable / {n_total:,} total "
          f"({100 * n_trainable / n_total:.2f}%)")

    if cfg.gradient_checkpointing:
        def _enable_unet_gc(model):
            inner = model
            for attr in ("base_model", "model"):
                if hasattr(inner, attr):
                    candidate = getattr(inner, attr)
                    if hasattr(candidate, "enable_gradient_checkpointing"):
                        candidate.enable_gradient_checkpointing()
                        return True
                    inner = candidate
            if hasattr(inner, "enable_gradient_checkpointing"):
                inner.enable_gradient_checkpointing()
                return True
            return False

        def _enable_text_gc(model):
            inner = model
            for attr in ("base_model", "model"):
                if hasattr(inner, attr):
                    candidate = getattr(inner, attr)
                    if hasattr(candidate, "gradient_checkpointing_enable"):
                        candidate.gradient_checkpointing_enable()
                        return True
                    inner = candidate
            if hasattr(inner, "gradient_checkpointing_enable"):
                inner.gradient_checkpointing_enable()
                return True
            return False

        unet_gc_ok = _enable_unet_gc(unet)
        text_gc_ok = _enable_text_gc(text_encoder)
        print(f"[train] gradient checkpointing: unet={unet_gc_ok}, text_encoder={text_gc_ok}")
        if not (unet_gc_ok and text_gc_ok):
            print("[train] WARNING: gradient checkpointing partially disabled; OOM risk if batch is large")

    dataset = make_train_dataset(cfg.train_csv, cfg.resolution)
    collate = make_collate_fn(tokenizer)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate, num_workers=2, drop_last=True,
    )
    print(f"[train] {len(dataset)} training samples / {len(loader)} steps per epoch")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_steps = len(loader) // cfg.grad_accum_steps * cfg.epochs
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps * cfg.grad_accum_steps,
        num_training_steps=total_steps * cfg.grad_accum_steps,
    )

    unet, text_encoder, optimizer, loader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, loader, lr_scheduler
    )
    
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float32)
    vae.eval()

    log_path = cfg.output_dir / "train_log.csv"
    log_f = log_path.open("w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["step", "epoch", "loss", "lr"])

    def run_validation(step: int):
        unet.eval()
        text_encoder.eval()
        try:
            val_scheduler = DPMSolverMultistepScheduler.from_pretrained(
                cfg.base_model_id, subfolder="scheduler"
            )
            unet_unwrapped = accelerator.unwrap_model(unet)
            text_encoder_unwrapped = accelerator.unwrap_model(text_encoder)
            val_pipe = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_encoder_unwrapped,
                tokenizer=tokenizer,
                unet=unet_unwrapped,
                scheduler=val_scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
            val_pipe.to(accelerator.device)
            images = []
            generator = torch.Generator(device=accelerator.device).manual_seed(cfg.val_seed)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if cfg.mixed_precision == "bf16"
                else torch.autocast(device_type="cuda", dtype=torch.float16)
                if cfg.mixed_precision == "fp16"
                else torch.cuda.amp.autocast(enabled=False)
            )
            with autocast_ctx:
                for prompt in cfg.val_prompts:
                    img = val_pipe(
                        prompt=prompt,
                        num_inference_steps=cfg.val_inference_steps,
                        guidance_scale=cfg.val_guidance,
                        generator=generator,
                    ).images[0]
                    images.append(img)
            grid_path = cfg.output_dir / "snapshots" / f"step_{step:05d}_grid.png"
            make_validation_grid(images, list(cfg.val_prompts), grid_path)
            print(f"  [val] snapshot saved → {grid_path.name}")
        finally:
            unet.train()
            text_encoder.train()

    global_step = 0
    progress = tqdm(total=total_steps, disable=not accelerator.is_local_main_process)
    t0 = time.time()
    for epoch in range(cfg.epochs):
        for batch in loader:
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    latents = vae.encode(
                        batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                    ).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=accelerator.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                text_embeddings = text_encoder(batch["input_ids"].to(accelerator.device))[0]

                noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample

                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                log_w.writerow([global_step, epoch, float(loss.item()),
                                float(lr_scheduler.get_last_lr()[0])])
                log_f.flush()
                progress.set_description(f"epoch {epoch + 1}/{cfg.epochs} loss={loss.item():.4f}")

                if global_step % cfg.val_every_steps == 0:
                    run_validation(global_step)

    log_f.close()
    progress.close()
    elapsed = time.time() - t0
    print(f"\n[train] training done in {elapsed/60:.1f} min ({global_step} steps)")

    if global_step % cfg.val_every_steps != 0:
        run_validation(global_step)

    unet_save_dir = cfg.output_dir / "lora_unet"
    text_save_dir = cfg.output_dir / "lora_text_encoder"
    accelerator.unwrap_model(unet).save_pretrained(unet_save_dir)
    accelerator.unwrap_model(text_encoder).save_pretrained(text_save_dir)
    print(f"[train] saved LoRA UNet → {unet_save_dir}")
    print(f"[train] saved LoRA text encoder → {text_save_dir}")
    print(f"[train] train log → {log_path}")
    print(f"[train] snapshots → {cfg.output_dir / 'snapshots'}/")

    return 0

if __name__ == "__main__":
    sys.exit(main())
