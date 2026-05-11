from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn, Tensor
from PIL import Image


class BaseFrequencyExtractor(nn.Module, ABC):
    """Abstract base class for frequency-domain feature extractors.

    All extractors are frozen (no trainable parameters). The preprocess()
    method runs on CPU during data collation; forward() runs on GPU under
    torch.no_grad().
    """

    def __init__(self):
        super().__init__()
        self.requires_grad_(False)

    @property
    @abstractmethod
    def output_dim(self) -> int:
        ...

    @abstractmethod
    def preprocess(self, images: List[Image.Image]) -> Tensor:
        """Convert raw PIL images to a batched tensor for forward().

        Returns:
            Tensor of shape [B, C, H, W] on CPU, float32.
        """
        ...

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Extract frequency features.

        Args:
            x: Preprocessed tensor [B, C, H, W] on the model device.

        Returns:
            Feature tensor [B, D] where D = output_dim.
        """
        ...
