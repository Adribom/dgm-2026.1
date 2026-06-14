"""
Visualization of mask-guided CycleGAN image generation.

Each example shows three columns:
    [Input (Real) | Generated | Lung Mask Overlay]

Two rows:
    Row 0 — Healthy  → Pneumonia
    Row 1 — Pneumonia → Healthy

Usage:
    python mask/visualize_generation.py \\
        --checkpoint mask/checkpoints/epoch_199.pt \\
        --healthy_dir          ../../data/processed/test/healthy \\
        --pneumonia_dir        ../../data/processed/test/pneumonia \\
        --healthy_masks_dir    ../../data/processed/test/healthy_masks \\
        --pneumonia_masks_dir  ../../data/processed/test/pneumonia_masks \\
        [--n_examples 4] [--image_size 128] [--seed 42] [--device cpu]
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

sys.path.insert(0, str(Path(__file__).parent.parent))
from model_utils.cyclegan import CycleGAN

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

INPUT_COLOR = "#2c7bb6"
GEN_COLOR   = "#d7191c"
MASK_COLOR  = "#4dac26"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_masked_images(img_dir, masks_dir, n, image_size, seed):
    paths = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in EXTS)
    if not paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    rng = random.Random(seed)
    selected = rng.sample(paths, min(n, len(paths)))
    masks_dir = Path(masks_dir)

    imgs, masks = [], []
    for p in selected:
        img  = TF.resize(Image.open(p).convert("L"), (image_size, image_size))
        mask = TF.resize(
            Image.open(masks_dir / p.name).convert("L"),
            (image_size, image_size),
            interpolation=InterpolationMode.NEAREST,
        )
        img_t  = TF.normalize(TF.to_tensor(img), [0.5], [0.5])
        mask_t = (TF.to_tensor(mask) > 0.5).float()
        imgs.append(img_t)
        masks.append(mask_t)

    return torch.stack(imgs), torch.stack(masks)   # (N,1,H,W), (N,1,H,W)


def to_disp(tensor):
    """(1, H, W) in [-1, 1] → (H, W) float in [0, 1]."""
    return np.clip(tensor.squeeze(0).numpy() * 0.5 + 0.5, 0, 1)


def mask_overlay(image_hw, mask_hw, color=(0.2, 0.8, 0.4), alpha=0.35):
    """Overlay a semi-transparent tint on the lung region. Returns (H, W, 3)."""
    rgb     = np.stack([image_hw, image_hw, image_hw], axis=-1)
    overlay = np.array(color, dtype=np.float32)
    blend   = rgb * (1 - alpha * mask_hw[..., None]) + overlay * alpha * mask_hw[..., None]
    return np.clip(blend, 0, 1)


def style_axis(ax, spine_color, lw=2.5):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)
        spine.set_linewidth(lw)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generation visualization for mask-guided CycleGAN."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--healthy_dir", required=True)
    p.add_argument("--pneumonia_dir", required=True)
    p.add_argument("--healthy_masks_dir", required=True)
    p.add_argument("--pneumonia_masks_dir", required=True)
    p.add_argument("--n_examples",  type=int, default=4)
    p.add_argument("--image_size",  type=int, default=128)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--output_dir",  default="mask/outputs/img_generation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    n      = args.n_examples

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = CycleGAN(image_channels=1, use_mask=True)
    model.G_H2P.load_state_dict(ckpt["G_H2P"])
    model.G_P2H.load_state_dict(ckpt["G_P2H"])
    G_H2P = model.G_H2P.to(device).eval()
    G_P2H = model.G_P2H.to(device).eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')} checkpoint.")

    # Load images + masks
    h_imgs, h_masks = load_masked_images(
        args.healthy_dir,   args.healthy_masks_dir,   n, args.image_size, args.seed
    )
    p_imgs, p_masks = load_masked_images(
        args.pneumonia_dir, args.pneumonia_masks_dir, n, args.image_size, args.seed
    )

    # Generate translations with hard masking (background preserved from real image)
    with torch.no_grad():
        h_input = torch.cat([h_imgs, h_masks], dim=1).to(device)
        p_input = torch.cat([p_imgs, p_masks], dim=1).to(device)
        gen_P = G_H2P(h_input)
        gen_P = (gen_P * h_masks.to(device) + h_imgs.to(device) * (1 - h_masks.to(device))).cpu()
        gen_H = G_P2H(p_input)
        gen_H = (gen_H * p_masks.to(device) + p_imgs.to(device) * (1 - p_masks.to(device))).cpu()

    h_imgs, p_imgs = h_imgs.cpu(), p_imgs.cpu()
    h_masks, p_masks = h_masks.cpu(), p_masks.cpu()

    # ------------------------------------------------------------------
    # Build figure
    # Layout: 2 rows × (3 * n) columns
    #   Columns per example: Input | Generated | Mask Overlay
    # ------------------------------------------------------------------

    col_titles  = ["Input", "Generated", "Mask Overlay"]
    col_colors  = [INPUT_COLOR, GEN_COLOR, MASK_COLOR]
    row_labels  = [
        ("Healthy → Pneumonia",  INPUT_COLOR),
        ("Pneumonia → Healthy",  GEN_COLOR),
    ]

    n_cols  = 3 * n
    fig_w   = 3.5 * n_cols + 0.5
    fig, axes = plt.subplots(2, n_cols, figsize=(fig_w, 8))

    fig.subplots_adjust(
        left=0.08, right=0.98,
        top=0.91, bottom=0.03,
        wspace=0.05, hspace=0.12,
    )

    for row_idx, (real_imgs, real_masks, gen_imgs) in enumerate([
        (h_imgs, h_masks, gen_P),
        (p_imgs, p_masks, gen_H),
    ]):
        label, label_color = row_labels[row_idx]

        for i in range(n):
            real_np    = to_disp(real_imgs[i])
            gen_np     = to_disp(gen_imgs[i])
            mask_np    = real_masks[i].squeeze(0).numpy()
            overlay_np = mask_overlay(real_np, mask_np)

            axes[row_idx, 3 * i    ].imshow(real_np,    cmap="gray", vmin=0, vmax=1)
            axes[row_idx, 3 * i + 1].imshow(gen_np,     cmap="gray", vmin=0, vmax=1)
            axes[row_idx, 3 * i + 2].imshow(overlay_np)

            style_axis(axes[row_idx, 3 * i    ], INPUT_COLOR)
            style_axis(axes[row_idx, 3 * i + 1], GEN_COLOR)
            style_axis(axes[row_idx, 3 * i + 2], MASK_COLOR)

            # Column headers on row 0 only
            if row_idx == 0:
                for k, (title, color) in enumerate(zip(col_titles, col_colors)):
                    axes[0, 3 * i + k].set_title(
                        title, fontsize=9, fontweight="bold", color=color, pad=5
                    )

        axes[row_idx, 0].set_ylabel(
            label, fontsize=11, fontweight="bold",
            color=label_color, labelpad=10, rotation=90, va="center",
        )

    # Legend
    patches = [
        mpatches.Patch(color=INPUT_COLOR, label="Input (Real)"),
        mpatches.Patch(color=GEN_COLOR,   label="Generated (CycleGAN + Mask)"),
        mpatches.Patch(color=MASK_COLOR,  label="Lung Mask Overlay"),
    ]
    fig.legend(handles=patches, loc="upper right", fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.98, 0.97))

    fig.suptitle(
        "CycleGAN (Mask-Guided) — Counterfactual Image Generation",
        fontsize=13, fontweight="bold", y=0.97,
    )

    # Save
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "generation_examples_mask.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved to: {save_path}")


if __name__ == "__main__":
    main()
