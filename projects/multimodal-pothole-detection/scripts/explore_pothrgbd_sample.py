#!/usr/bin/env python3
"""Explore one PothRGDB sample using manifest-driven paths and reusable helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from pothole_geometry import (
    CameraSpec,
    calculate_volume_variable_area,
    estimate_road_surface_depth,
    load_yolo_mask,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root folder for project/code.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Manifest row index to analyze when --sample-id is not provided.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Optional sample id (YYYYMMDD_HHMMSS) to analyze.",
    )
    parser.add_argument(
        "--save-figure",
        type=Path,
        default=None,
        help="Optional output path for the visualization image.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open matplotlib window (useful in headless mode).",
    )
    return parser


def choose_sample(manifest: pd.DataFrame, sample_id: str | None, row_index: int) -> pd.Series:
    if sample_id:
        subset = manifest[manifest["sample_id"] == sample_id]
        if subset.empty:
            raise ValueError(f"sample_id not found in manifest: {sample_id}")
        return subset.iloc[0]

    if row_index < 0 or row_index >= len(manifest):
        raise ValueError(f"row-index out of range: {row_index} (size={len(manifest)})")
    return manifest.iloc[row_index]


def main() -> int:
    args = build_parser().parse_args()
    code_root = args.code_root.resolve()

    manifest_path = code_root / "artifacts" / "pothrgbd_manifest.csv"
    dataset_root = code_root / "datasets" / "PothRGDB" / "PUBLIC POTHOLE DATASET"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    row = choose_sample(manifest, args.sample_id, args.row_index)
    if row["status"] != "ok":
        raise ValueError(f"Selected sample is flagged for checks: {row['sample_id']} -> {row['notes']}")

    image_path = dataset_root / row["image_path"]
    depth_path = dataset_root / row["depth_path"]
    label_path = dataset_root / row["label_path"]

    image = np.array(Image.open(image_path).convert("RGB"))
    depth = np.load(depth_path)
    mask = load_yolo_mask(label_path, image_shape=depth.shape)

    road_surface = estimate_road_surface_depth(depth, mask)
    if road_surface is None:
        raise RuntimeError("Could not estimate road surface depth for selected sample")

    metrics = calculate_volume_variable_area(depth, mask, road_surface_depth_mm=road_surface, camera=CameraSpec())

    print("Sample:", row["sample_id"])
    print("Image:", image_path)
    print("Depth:", depth_path)
    print("Label:", label_path)
    print("Road surface depth (mm):", f"{road_surface:.2f}")
    print("Estimated volume (cm^3):", f"{metrics['volume_cm3']:.2f}")
    print("Estimated volume (L):", f"{metrics['volume_liters']:.4f}")
    print("Estimated max depth delta (mm):", f"{metrics['max_depth_mm']:.2f}")
    print("Estimated mean depth delta (mm):", f"{metrics['mean_depth_mm']:.2f}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    depth_vis = depth.astype(np.float32)
    depth_vis[depth_vis == 0] = np.nan
    im1 = axes[1].imshow(depth_vis, cmap="jet")
    axes[1].set_title("Depth (mm)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("YOLO Mask")
    axes[2].axis("off")

    diff_map = metrics["depth_diff_map"].copy()
    diff_map[diff_map == 0] = np.nan
    im2 = axes[3].imshow(diff_map, cmap="hot")
    axes[3].set_title("Depth Delta vs Surface (mm)")
    axes[3].axis("off")
    fig.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(f"PothRGDB Sample {row['sample_id']} | Volume ~ {metrics['volume_cm3']:.2f} cm^3")
    fig.tight_layout()

    if args.save_figure:
        args.save_figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_figure, dpi=150, bbox_inches="tight")
        print("Saved figure:", args.save_figure)

    if not args.no_show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
