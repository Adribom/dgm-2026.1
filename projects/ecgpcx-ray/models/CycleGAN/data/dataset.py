from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset


class CycleGANDataset(Dataset):

    def __init__(
        self,
        healthy_dir,
        pneumonia_dir,
        transform=None,
        healthy_masks_dir=None,
        pneumonia_masks_dir=None,
    ):

        self.healthy_paths = sorted(
            list(Path(healthy_dir).glob("*"))
        )

        self.pneumonia_paths = sorted(
            list(Path(pneumonia_dir).glob("*"))
        )

        self.transform = transform
        self.healthy_masks_dir = Path(healthy_masks_dir) if healthy_masks_dir else None
        self.pneumonia_masks_dir = Path(pneumonia_masks_dir) if pneumonia_masks_dir else None
        self.use_mask = self.healthy_masks_dir is not None

    def _load_mask(self, masks_dir, image_path):
        return Image.open(masks_dir / image_path.name).convert("L")

    def __len__(self):

        return max(
            len(self.healthy_paths),
            len(self.pneumonia_paths)
        )

    def __getitem__(self, idx):

        healthy_path = self.healthy_paths[
            idx % len(self.healthy_paths)
        ]

        pneumonia_path = random.choice(
            self.pneumonia_paths
        )

        healthy_image = Image.open(healthy_path).convert("L")
        pneumonia_image = Image.open(pneumonia_path).convert("L")

        if self.use_mask:
            healthy_mask = self._load_mask(self.healthy_masks_dir, healthy_path)
            pneumonia_mask = self._load_mask(self.pneumonia_masks_dir, pneumonia_path)

            if self.transform:
                healthy_image = self.transform(healthy_image, healthy_mask)
                pneumonia_image = self.transform(pneumonia_image, pneumonia_mask)
        else:
            if self.transform:
                healthy_image = self.transform(healthy_image)
                pneumonia_image = self.transform(pneumonia_image)

        return {
            "healthy": healthy_image,
            "pneumonia": pneumonia_image,
        }