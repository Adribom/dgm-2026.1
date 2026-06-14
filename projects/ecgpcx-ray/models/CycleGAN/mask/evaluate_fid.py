"""
FID evaluation script for mask-guided CycleGAN counterfactual generation.

Computes Fréchet Inception Distance (FID) between:
  - Real pneumonia (test) vs. Fake pneumonia generated from healthy (H→P)
  - Real healthy  (test)  vs. Fake healthy  generated from pneumonia (P→H)

Each image is loaded together with its lung mask; the 2-channel (image+mask)
tensor is fed to the generator which outputs a 1-channel image used for FID.

Usage:
    python mask/evaluate_fid.py \\
        --checkpoint mask/checkpoints/epoch_199.pt \\
        --healthy_dir          ../../data/processed/test/healthy \\
        --pneumonia_dir        ../../data/processed/test/pneumonia \\
        --healthy_masks_dir    ../../data/processed/test/healthy_masks \\
        --pneumonia_masks_dir  ../../data/processed/test/pneumonia_masks \\
        [--batch_size 16] [--image_size 128] [--device cuda]
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms.functional import InterpolationMode

sys.path.insert(0, str(Path(__file__).parent.parent))
from model_utils.cyclegan import CycleGAN

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MaskedImageDataset(Dataset):
    """Returns 2-channel (image + mask) tensors for a directory of X-rays."""

    def __init__(self, img_dir, masks_dir, image_size=128):
        self.paths = sorted(
            p for p in Path(img_dir).iterdir() if p.suffix.lower() in EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {img_dir}")
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]

        img = TF.resize(Image.open(p).convert("L"), (self.image_size, self.image_size))
        mask = TF.resize(
            Image.open(self.masks_dir / p.name).convert("L"),
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.NEAREST,
        )

        img_t = TF.normalize(TF.to_tensor(img), [0.5], [0.5])
        mask_t = (TF.to_tensor(mask) > 0.5).float()

        return torch.cat([img_t, mask_t], dim=0)   # (2, H, W)


# ---------------------------------------------------------------------------
# Inception feature extractor
# ---------------------------------------------------------------------------

class InceptionFeatureExtractor(torch.nn.Module):
    """InceptionV3 truncated at the pool3 (2048-d) layer."""

    def __init__(self, device):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        inception.eval()
        self.layers = torch.nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.to(device)

    @torch.no_grad()
    def forward(self, x):
        return self.layers(x).squeeze(-1).squeeze(-1)   # (N, 2048)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def to_inception_input(tensor, device):
    """Convert 1-channel CycleGAN output ([-1,1]) to 3-channel InceptionV3 input."""
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0.0, 1.0)
    tensor = tensor.repeat(1, 3, 1, 1)
    tensor = F.interpolate(tensor, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


# ---------------------------------------------------------------------------
# Feature collection
# ---------------------------------------------------------------------------

def collect_features(loader, extractor, device, generator=None, desc=""):
    """
    Collect 2048-d Inception features.

    If generator is provided the 2-channel (image+mask) batch is passed through
    it to produce counterfactuals (1 channel); otherwise only ch 0 (the real
    image) is used.
    """
    all_features = []
    print(f"  Extracting features — {desc} ({len(loader.dataset)} images)...")

    for batch in loader:
        batch = batch.to(device)

        if generator is not None:
            with torch.no_grad():
                mask   = batch[:, 1:2, :, :]
                image  = batch[:, 0:1, :, :]
                generated = generator(batch)        # (B, 1, H, W)
                # Hard mask: keep background identical to the input image
                images = generated * mask + image * (1 - mask)
        else:
            images = batch[:, 0:1, :, :]            # real image channel only

        features = extractor(to_inception_input(images, device))
        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)     # (N, 2048)


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------

def compute_statistics(features):
    return features.mean(axis=0), np.cov(features, rowvar=False)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("sqrtm produced large imaginary component")
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="FID evaluation for mask-guided CycleGAN."
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint (e.g. mask/checkpoints/epoch_199.pt).")
    p.add_argument("--healthy_dir", required=True,
                   help="Directory with test healthy X-ray images.")
    p.add_argument("--pneumonia_dir", required=True,
                   help="Directory with test pneumonia X-ray images.")
    p.add_argument("--healthy_masks_dir", required=True,
                   help="Directory with lung masks for healthy images (same filenames).")
    p.add_argument("--pneumonia_masks_dir", required=True,
                   help="Directory with lung masks for pneumonia images (same filenames).")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None,
                   help="JSON output path (default: <checkpoint_stem>_fid_mask.json).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load generators
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = CycleGAN(image_channels=1, use_mask=True)
    model.G_H2P.load_state_dict(ckpt["G_H2P"])
    model.G_P2H.load_state_dict(ckpt["G_P2H"])
    G_H2P = model.G_H2P.to(device).eval()
    G_P2H = model.G_P2H.to(device).eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')} checkpoint.")

    # Datasets and loaders
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    healthy_ds   = MaskedImageDataset(args.healthy_dir,   args.healthy_masks_dir,   args.image_size)
    pneumonia_ds = MaskedImageDataset(args.pneumonia_dir, args.pneumonia_masks_dir, args.image_size)

    healthy_loader   = DataLoader(healthy_ds,   **loader_kwargs)
    pneumonia_loader = DataLoader(pneumonia_ds, **loader_kwargs)

    print(f"\nTest set — healthy: {len(healthy_ds)} | pneumonia: {len(pneumonia_ds)}")

    # Inception extractor
    print("\nLoading InceptionV3 feature extractor...")
    extractor = InceptionFeatureExtractor(device)

    # Collect features
    print("\n--- Collecting features ---")
    feats_real_H  = collect_features(healthy_loader,   extractor, device, desc="real healthy")
    feats_real_P  = collect_features(pneumonia_loader, extractor, device, desc="real pneumonia")
    feats_fake_P  = collect_features(healthy_loader,   extractor, device, generator=G_H2P, desc="fake pneumonia (H→P)")
    feats_fake_H  = collect_features(pneumonia_loader, extractor, device, generator=G_P2H, desc="fake healthy   (P→H)")

    # FID
    print("\n--- Computing FID ---")
    mu_rP, sig_rP = compute_statistics(feats_real_P)
    mu_fP, sig_fP = compute_statistics(feats_fake_P)
    fid_H2P = frechet_distance(mu_rP, sig_rP, mu_fP, sig_fP)

    mu_rH, sig_rH = compute_statistics(feats_real_H)
    mu_fH, sig_fH = compute_statistics(feats_fake_H)
    fid_P2H = frechet_distance(mu_rH, sig_rH, mu_fH, sig_fH)

    mean_fid = (fid_H2P + fid_P2H) / 2

    print("\n" + "=" * 50)
    print("FID Results (mask-guided CycleGAN)")
    print("=" * 50)
    print(f"  H→P  (real pneumonia  vs fake pneumonia) : {fid_H2P:.4f}")
    print(f"  P→H  (real healthy    vs fake healthy)   : {fid_P2H:.4f}")
    print(f"  Mean FID                                 : {mean_fid:.4f}")
    print("=" * 50)

    # Save JSON results
    output_path = Path(args.output) if args.output else (
        Path(args.checkpoint).with_name(Path(args.checkpoint).stem + "_fid_mask.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "fid_H2P": round(fid_H2P, 6),
        "fid_P2H": round(fid_P2H, 6),
        "mean_fid": round(mean_fid, 6),
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt.get("epoch"),
        "healthy_dir": str(args.healthy_dir),
        "pneumonia_dir": str(args.pneumonia_dir),
        "healthy_masks_dir": str(args.healthy_masks_dir),
        "pneumonia_masks_dir": str(args.pneumonia_masks_dir),
        "n_healthy": len(healthy_ds),
        "n_pneumonia": len(pneumonia_ds),
        "image_size": args.image_size,
        "use_mask": True,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return {"fid_H2P": fid_H2P, "fid_P2H": fid_P2H}


if __name__ == "__main__":
    main()
