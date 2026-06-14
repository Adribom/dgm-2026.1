"""
Translates ALL original images using the mask-guided CycleGAN.

    data/healthy_images/   --[G_H2P]--> data/generated/healthy_to_pneumonia/
    data/pnemonia_images/  --[G_P2H]--> data/generated/pneumonia_to_healthy/

Mask lookup (fast path): reuses any mask already in data/processed/*_masks/
that shares the same filename.  For images with no pre-computed mask, generates
one on-the-fly with TorchXRayVision PSPNet.

Hard masking is applied after generation:
    output = generated * mask + original * (1 - mask)
so pixels outside the lung are identical to the source image.

Script is resumable: images already present in the output folder are skipped.

Usage (PowerShell — execute a partir de projects/ecgpcx-ray/models/CycleGAN/):
    Set-Location "C:/Users/User/Documents/0Unicamp/IA376N/Projeto/dgm-2026.1/projects/ecgpcx-ray/models/CycleGAN"
    C:/Users/User/anaconda3/envs/ia376_env/python.exe mask/generate_all.py `
        --checkpoint mask/checkpoints/epoch_199.pt `
        --healthy_dir   ../../data/healthy_images `
        --pneumonia_dir ../../data/pnemonia_images `
        --processed_dir ../../data/processed `
        --output_dir    ../../data/generated `
        --batch_size 16 `
        --device cuda
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # ecgpcx-ray/

from model_utils.cyclegan import CycleGAN
from utils.xrv_lung_segmentation import TorchXRayVisionLungSegmenter

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Mask cache
# ---------------------------------------------------------------------------

def build_mask_cache(processed_dir: Path) -> dict:
    """Build {filename: mask_path} from all *_masks/ subdirs under processed_dir."""
    cache: dict[str, Path] = {}
    for mask_dir in processed_dir.glob("*/*_masks"):
        if mask_dir.is_dir():
            for p in mask_dir.iterdir():
                if p.suffix.lower() in EXTS and p.name not in cache:
                    cache[p.name] = p
    return cache


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def load_image(path: Path, image_size: int):
    """Return (image_tensor [1,H,W] in [-1,1], pil_original_L)."""
    img = Image.open(path).convert("L")
    img_r = img.resize((image_size, image_size), Image.LANCZOS)
    t = TF.normalize(TF.to_tensor(img_r), [0.5], [0.5])
    return t, img  # tensor + original PIL for on-the-fly mask gen


def get_mask_tensor(
    path: Path,
    orig_pil: Image.Image,
    image_size: int,
    mask_cache: dict,
    segmenter: "TorchXRayVisionLungSegmenter | None",
) -> torch.Tensor:
    """Return binary mask tensor [1, H, W] in {0, 1}."""
    if path.name in mask_cache:
        mask_pil = Image.open(mask_cache[path.name]).convert("L")
        mask_pil = mask_pil.resize((image_size, image_size), Image.NEAREST)
    else:
        # Generate on-the-fly (uses original resolution, then resize)
        mask_pil = segmenter.mask_image(orig_pil)
        mask_pil = mask_pil.resize((image_size, image_size), Image.NEAREST)
    return (TF.to_tensor(mask_pil) > 0.5).float()


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert a [1, H, W] tensor in [-1, 1] to a uint8 grayscale PIL image."""
    arr = np.clip((t.cpu().squeeze(0).numpy() * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def process_domain(
    img_paths: list,
    generator: torch.nn.Module,
    mask_cache: dict,
    segmenter,
    output_dir: Path,
    image_size: int,
    batch_size: int,
    device: torch.device,
    domain_label: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    already_done = sum(1 for p in img_paths if (output_dir / p.name).exists())
    if already_done:
        print(f"  {already_done}/{len(img_paths)} already generated — skipping those.")

    n_saved = 0
    todo = [p for p in img_paths if not (output_dir / p.name).exists()]

    for batch_start in tqdm(range(0, len(todo), batch_size),
                             desc=domain_label, unit="batch"):
        batch_paths = todo[batch_start: batch_start + batch_size]

        imgs_t, masks_t, out_paths = [], [], []
        for p in batch_paths:
            img_t, orig_pil = load_image(p, image_size)
            mask_t = get_mask_tensor(p, orig_pil, image_size, mask_cache, segmenter)
            imgs_t.append(img_t)
            masks_t.append(mask_t)
            out_paths.append(output_dir / p.name)

        imgs_batch  = torch.stack(imgs_t).to(device)
        masks_batch = torch.stack(masks_t).to(device)

        with torch.no_grad():
            generated = generator(torch.cat([imgs_batch, masks_batch], dim=1))
            generated = generated * masks_batch + imgs_batch * (1 - masks_batch)

        for gen_t, out_path in zip(generated, out_paths):
            tensor_to_pil(gen_t).save(out_path)
            n_saved += 1

    return n_saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Translate all original images with the mask-guided CycleGAN."
    )
    p.add_argument("--checkpoint",    required=True,
                   help="Path to .pt checkpoint file (e.g. mask/checkpoints/epoch_199.pt)")
    p.add_argument("--healthy_dir",   required=True,
                   help="Folder with original healthy X-rays")
    p.add_argument("--pneumonia_dir", required=True,
                   help="Folder with original pneumonia X-rays")
    p.add_argument("--processed_dir", required=True,
                   help="data/processed root — used to reuse pre-computed masks")
    p.add_argument("--output_dir",    default="../../data/generated",
                   help="Root output folder (default: ../../data/generated)")
    p.add_argument("--image_size",    type=int, default=128)
    p.add_argument("--batch_size",    type=int, default=16)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    # -- Load model -----------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = CycleGAN(image_channels=1, use_mask=True)
    model.G_H2P.load_state_dict(ckpt["G_H2P"])
    model.G_P2H.load_state_dict(ckpt["G_P2H"])
    G_H2P = model.G_H2P.to(device).eval()
    G_P2H = model.G_P2H.to(device).eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')} checkpoint.\n")

    # -- Mask cache (pre-computed masks from data/processed/) -----------------
    processed_dir = Path(args.processed_dir)
    print(f"Building mask cache from: {processed_dir}")
    mask_cache = build_mask_cache(processed_dir)
    print(f"  {len(mask_cache)} pre-computed masks found.\n")

    # -- Segmenter for on-the-fly mask generation -----------------------------
    print(f"Loading PSPNet segmenter on {args.device}...")
    segmenter = TorchXRayVisionLungSegmenter(device=args.device)
    print("  Segmenter ready.\n")

    # -- Collect input paths --------------------------------------------------
    healthy_paths   = sorted(p for p in Path(args.healthy_dir).iterdir()
                              if p.suffix.lower() in EXTS)
    pneumonia_paths = sorted(p for p in Path(args.pneumonia_dir).iterdir()
                              if p.suffix.lower() in EXTS)

    print(f"Healthy images   : {len(healthy_paths)}")
    print(f"Pneumonia images : {len(pneumonia_paths)}\n")

    output_root = Path(args.output_dir)
    h2p_dir     = output_root / "healthy_to_pneumonia"
    p2h_dir     = output_root / "pneumonia_to_healthy"

    # -- Run ------------------------------------------------------------------
    print(f"Generating healthy -> pneumonia  (output: {h2p_dir})")
    n_h = process_domain(
        healthy_paths, G_H2P, mask_cache, segmenter,
        h2p_dir, args.image_size, args.batch_size, device,
        domain_label="H->P",
    )

    print(f"\nGenerating pneumonia -> healthy  (output: {p2h_dir})")
    n_p = process_domain(
        pneumonia_paths, G_P2H, mask_cache, segmenter,
        p2h_dir, args.image_size, args.batch_size, device,
        domain_label="P->H",
    )

    print(f"\nDone. {n_h + n_p} images saved total ({n_h} H->P, {n_p} P->H).")


if __name__ == "__main__":
    main()
