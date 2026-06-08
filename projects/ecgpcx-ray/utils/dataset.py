"""Dataset creating and loading utilities for NIH Chest X-rays dataset.

This module provides the `PyTorchDataset` class, which is a custom implementation of a PyTorch `Dataset`.
It is designed to work with the NIH Chest X-rays dataset and is compatible with PyTorch's `DataLoader`
for efficient data loading during training and evaluation of models like CVAE and CycleGAN.
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class PyTorchDataset(Dataset):
    """PyTorch Dataset for CVAE and CycleGAN models compatible with DataLoader.

    Provides (image, one-hot label, metadata) tuples for the CVAE and CycleGAN models.
    Images are converted to tensors, labels are one-hot encoded, and metadata
    includes normalized age and encoded gender.

    Attributes:
        images (list): List of PIL Image objects.
        labels (torch.Tensor): One-hot encoded labels (1 for pneumonia, 0 for healthy).
        metadata (torch.Tensor): Combined normalized age and encoded gender.
    """

    def __init__(self, images, labels, metadata, masks=None):
        """Initialize PyTorchDataset.

        Args:
            images (list): List of PIL Image objects.
            labels (torch.Tensor): One-hot encoded binary labels with shape (N, 2).
            metadata (torch.Tensor): Metadata tensor with shape (N, 2)
                                     (normalized age, encoded gender).
        """
        self.images = images
        self.masks = masks
        self.labels = labels
        # Ensure metadata is a float tensor
        self.metadata = metadata.float()
        if self.metadata.dim() == 1:
            self.metadata = self.metadata.unsqueeze(1)
        
        # Validate that all components have the same length
        assert (
            len(self.images) == len(self.labels)
        ), f"Images ({len(self.images)}) and labels ({len(self.labels)}) length mismatch"
        assert (
            len(self.images) == len(self.metadata)
        ), f"Images ({len(self.images)}) and metadata ({len(self.metadata)}) length mismatch"
        if self.masks is not None:
            assert (
                len(self.images) == len(self.masks)
            ), f"Images ({len(self.images)}) and masks ({len(self.masks)}) length mismatch"

    def __len__(self):
        """Return dataset size."""
        return len(self.images)

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image_tensor, label_tensor, metadata_tensor) where:
                   - image_tensor: torch.Tensor with shape (1, H, W)
                   - label_tensor: torch.Tensor with shape (2,) - one-hot encoded
                   - metadata_tensor: torch.Tensor with shape (2,) - [age, gender]
        """
        # Load and convert PIL image to tensor
        img = self.images[idx]
        # Convert to grayscale to ensure consistent single channel
        if img.mode != 'L':
            img = img.convert('L')
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add channel dimension

        if self.masks is not None:
            mask = self.masks[idx]
            if mask.mode != 'L':
                mask = mask.convert('L')
            mask_array = np.array(mask, dtype=np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
            img_tensor = torch.cat([img_tensor, mask_tensor], dim=0)

        # Get label and metadata
        label_tensor = self.labels[idx]
        metadata_tensor = self.metadata[idx]

        return img_tensor, label_tensor, metadata_tensor


class AugmentedDataset(Dataset):
    """Wraps a PyTorchDataset and applies a torchvision transform on the PIL image
    before converting it to tensor. Use for the training split only — val/test should
    keep the plain PyTorchDataset so evaluation stays deterministic.

    Defined here (not inside a notebook) so DataLoader workers using `num_workers > 0`
    can pickle/unpickle it via module import.
    """

    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img = self.base.images[idx].convert("L")  # grayscale PIL
        if self.transform:
            img = self.transform(img)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)
        return img_tensor, self.base.labels[idx], self.base.metadata[idx]
