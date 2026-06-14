"""Pothole dataset loading and augmentation utilities.

This module loads paired pothole images and point clouds, applies optional
training augmentations, and prepares Point-E batches. Supports val-set
isolation via `sample_ids_filter` (to create reproducible train/val splits)
and augmentation-free evaluation via `disable_augmentation` (always True for
val dataloaders to guarantee clean evaluation metrics).
"""

from __future__ import annotations

import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.transforms.geometric import horizontal_flip
from src.data.transforms.lighting import apply_color_jitter, apply_fake_shadow
from src.data.transforms.noise import apply_cutout, apply_gaussian_blur, apply_motion_blur

# Supported image extensions for dataset pairing.
IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

_DEFAULT_AUGMENTATION_PROBABILITIES = {
    "pure_image": 0.1,
    "horizontal_flip": 0.5,
    "fake_shadow": 0.3,
    "color_jitter": 0.4,
    "gaussian_blur": 0.2,
    "motion_blur": 0.2,
    "cutout": 0.3,
}

_AUGMENTATION_TRANSFORMS = {
    "horizontal_flip": horizontal_flip,
    "fake_shadow": apply_fake_shadow,
    "color_jitter": apply_color_jitter,
    "gaussian_blur": apply_gaussian_blur,
    "motion_blur": apply_motion_blur,
    "cutout": apply_cutout,
}


class PotholeDataset(Dataset):
    """
    Dataset for paired 2D images and 3D point clouds.

    Supports optional val-set isolation via `sample_ids_filter` and
    augmentation-free evaluation via `disable_augmentation`.

    Returns one sample as a dictionary containing:
      - image_for_conditioning: PIL.Image in RGB mode (raw visual input for Point-E CLIP path)
      - point_cloud_6d: torch.FloatTensor with shape [6, K]
      - sample_id: stem name used for traceability/debug
      - applied_transforms: list of dicts with name and parameters
    """
    def __init__(
        self,
        image_dir: str | Path,
        cloud_dir: str | Path,
        augmentation_config: dict | None = None,
        sample_ids_filter: set[str] | None = None,
        disable_augmentation: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.cloud_dir = Path(cloud_dir)
        self.augmentation_config = augmentation_config
        self.sample_ids_filter = sample_ids_filter
        # Val datasets must always be constructed with disable_augmentation=True.
        self.disable_augmentation = disable_augmentation
        self.samples = self._pair_samples()

    def _pair_samples(self):
        """Finds matching image and .npy files between the directories.

        When `sample_ids_filter` is set, only samples whose filename stem
        appears in the filter set are included. Used to create train/val splits.
        """
        samples = []
        if not self.image_dir.exists() or not self.cloud_dir.exists():
            print(f"Warning: Dataset directories not found: {self.image_dir} or {self.cloud_dir}")
            return samples

        image_paths = [
            path
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for img_path in sorted(image_paths):
            cloud_path = self.cloud_dir / f"{img_path.stem}.npy"
            if cloud_path.exists():
                samples.append((img_path, cloud_path))

        if self.sample_ids_filter is not None:
            samples = [
                (img, cld) for img, cld in samples
                if img.stem in self.sample_ids_filter
            ]

        return samples

    def __len__(self):
        return len(self.samples)

    def _normalize_point_cloud(self, pts: np.ndarray) -> torch.Tensor:
        """Convert input point cloud to Point-E expected shape [6, K].

        Accepted source layouts:
        - [K, 3]: XYZ only, RGB channels are appended as zeros.
        - [K, 6]: XYZRGB already present.
        - [3, K] or [6, K]: channel-first variants, transposed as needed.
        """
        if pts.ndim != 2:
            raise ValueError(f"Expected 2D point cloud array, got shape {pts.shape}")

        # Convert channel-first variants to point-first for simpler handling.
        if pts.shape[0] in (3, 6) and pts.shape[1] not in (3, 6):
            pts = pts.T

        if pts.shape[1] == 3:
            xyz_tensor = torch.tensor(pts, dtype=torch.float32)
            dummy_rgb = torch.zeros((xyz_tensor.shape[0], 3), dtype=torch.float32)
            pts_6d = torch.cat([xyz_tensor, dummy_rgb], dim=1)
        elif pts.shape[1] == 6:
            pts_6d = torch.tensor(pts, dtype=torch.float32)
        else:
            raise ValueError(
                "Unsupported point cloud shape "
                f"{pts.shape}. Expected [K,3], [K,6], [3,K], or [6,K]."
            )

        return pts_6d.transpose(0, 1)

    def _apply_augmentations(self, image: Image.Image, pts_raw: np.ndarray) -> tuple[Image.Image, np.ndarray, list[dict]]:
        """Apply configured augmentations to an image and point cloud pair.

        The horizontal flip path must update the image and point cloud together
        before any further transform is applied.
        """
        if not self.augmentation_config or not self.augmentation_config.get("active_transforms"):
            return image, pts_raw, []

        applied_transforms: list[dict] = []
        probabilities = {
            **_DEFAULT_AUGMENTATION_PROBABILITIES,
            **(self.augmentation_config.get("probabilities") or {}),
        }

        pure_image_probability = probabilities.get("pure_image", 0.0)
        if random.random() < pure_image_probability:
            return image, pts_raw, [{"name": "pure_image", "params": {}}]

        current_image = image
        current_pts = pts_raw
        for transform_name in self.augmentation_config.get("active_transforms", []):
            if transform_name == "pure_image":
                continue

            probability = probabilities.get(transform_name, 0.0)
            if random.random() >= probability:
                continue

            transform_fn = _AUGMENTATION_TRANSFORMS.get(transform_name)
            if transform_fn is None:
                continue

            if transform_name == "horizontal_flip":
                current_image, current_pts, params = transform_fn(current_image, current_pts)
                assert isinstance(current_pts, np.ndarray)
            else:
                current_image, params = transform_fn(current_image)

            applied_transforms.append({"name": transform_name, "params": params})

        return current_image, current_pts, applied_transforms

    def __getitem__(self, idx):
        """Return one sample dict. Val datasets bypass augmentation entirely.

        Invariant: Val datasets must always be constructed with
        `disable_augmentation=True` — do not rely on `augmentation_config=None`
        alone to guarantee clean evaluation.
        """
        img_path, cloud_path = self.samples[idx]

        # Load image as raw RGB for Point-E CLIP conditioning path.
        image = Image.open(img_path).convert("RGB")

        pts_raw = np.load(cloud_path)
        # Val datasets bypass all augmentation at this level — do not move this guard.
        if self.disable_augmentation or self.augmentation_config is None:
            applied_transforms = []
        else:
            image, pts_raw, applied_transforms = self._apply_augmentations(image, pts_raw)

        pts_final = self._normalize_point_cloud(pts_raw)

        return {
            "image_for_conditioning": image,
            "point_cloud_6d": pts_final,
            "sample_id": img_path.stem,
            "applied_transforms": applied_transforms,
        }


def point_e_collate_fn(batch: list[dict]) -> dict:
    """Collate samples into a training batch for Point-E.

    Output keys:
      - images: list[PIL.Image]
      - point_cloud_6d: torch.FloatTensor [B, 6, K]
      - sample_id: list[str]
      - applied_transforms: list[list[dict]]
    """
    images = [item["image_for_conditioning"] for item in batch]
    point_clouds = torch.stack([item["point_cloud_6d"] for item in batch], dim=0)
    sample_ids = [item["sample_id"] for item in batch]
    applied_transforms = [item["applied_transforms"] for item in batch]

    return {
        "images": images,
        "point_cloud_6d": point_clouds,
        "sample_id": sample_ids,
        "applied_transforms": applied_transforms,
    }


def create_dataloader(
    image_dir: str | Path,
    cloud_dir: str | Path,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    augmentation_config: dict | None = None,
    sample_ids_filter: set[str] | None = None,
    disable_augmentation: bool = False,
):
    """
    Creates and returns a PyTorch DataLoader for the PotholeDataset.

    Parameters
    ----------
    sample_ids_filter:
        When provided, only samples with a stem matching an entry in this set
        are included. Used to create train/val splits.
    disable_augmentation:
        When True, all augmentations are bypassed regardless of augmentation_config.
        Always set to True for val dataloaders.

    Returns batches with:
        - images: list[PIL.Image]
        - point_cloud_6d: torch.FloatTensor [B, 6, K]
        - sample_id: list[str]
        - applied_transforms: list[list[dict]]
    """
    dataset = PotholeDataset(
        image_dir,
        cloud_dir,
        augmentation_config=augmentation_config,
        sample_ids_filter=sample_ids_filter,
        disable_augmentation=disable_augmentation,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=point_e_collate_fn,
    )
