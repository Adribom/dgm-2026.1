"""
Pothole Dataset DataLoader Module.

This script manages loading paired pothole images and their generated point clouds.
"""

from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Supported image extensions for dataset pairing.
IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

class PotholeDataset(Dataset):
    """
    Dataset for paired 2D images and 3D point clouds.

    Returns one sample as a dictionary containing:
      - image_for_conditioning: PIL.Image in RGB mode (raw visual input for Point-E CLIP path)
      - point_cloud_6d: torch.FloatTensor with shape [6, K]
      - sample_id: stem name used for traceability/debug
    """
    def __init__(self, image_dir: str | Path, cloud_dir: str | Path):
        self.image_dir = Path(image_dir)
        self.cloud_dir = Path(cloud_dir)
        self.samples = self._pair_samples()

    def _pair_samples(self):
        """Finds matching image and .npy files between the directories."""
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

    def __getitem__(self, idx):
        img_path, cloud_path = self.samples[idx]

        # Load image as raw RGB for Point-E CLIP conditioning path.
        image = Image.open(img_path).convert("RGB")
            
        # T007 + T008: Point cloud loading and strict normalization to Point-E [6, K]
        pts = np.load(cloud_path)
        pts_final = self._normalize_point_cloud(pts)

        return {
            "image_for_conditioning": image,
            "point_cloud_6d": pts_final,
            "sample_id": img_path.stem,
        }


def point_e_collate_fn(batch: list[dict]) -> dict:
    """Collate samples into a training batch for Point-E.

    Output keys:
      - images: list[PIL.Image]
      - point_cloud_6d: torch.FloatTensor [B, 6, K]
      - sample_id: list[str]
    """
    images = [item["image_for_conditioning"] for item in batch]
    point_clouds = torch.stack([item["point_cloud_6d"] for item in batch], dim=0)
    sample_ids = [item["sample_id"] for item in batch]

    return {
        "images": images,
        "point_cloud_6d": point_clouds,
        "sample_id": sample_ids,
    }


# T009: Dataloader wrapper setup to serve minibatches globally
def create_dataloader(image_dir: str | Path, cloud_dir: str | Path, batch_size: int = 8, shuffle: bool = True, num_workers: int = 0):
    """
    Creates and returns a PyTorch DataLoader for the PotholeDataset.

        Returns batches with:
            - images: list[PIL.Image]
            - point_cloud_6d: torch.FloatTensor [B, 6, K]
            - sample_id: list[str]
    """
    dataset = PotholeDataset(image_dir, cloud_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=point_e_collate_fn,
    )
