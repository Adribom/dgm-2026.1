from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.transforms.functional import resize, to_tensor
from PIL import Image

from .base import BaseFrequencyExtractor


class FFTExtractor(BaseFrequencyExtractor):
    """Extracts log-magnitude FFT spectrum as frequency-domain features.

    Deterministic, no learnable parameters. Applies 2D FFT per channel,
    centers the DC component, takes log-magnitude, and pools to a fixed size.
    """

    def __init__(self, input_size: int = 224, pool_size: int = 32):
        super().__init__()
        self._input_size = input_size
        self._pool_size = pool_size
        self.pool = nn.AdaptiveAvgPool2d(pool_size)

    @property
    def output_dim(self) -> int:
        return 3 * self._pool_size * self._pool_size

    def preprocess(self, images: List[Image.Image]) -> Tensor:
        tensors = []
        for img in images:
            img = img.resize(
                (self._input_size, self._input_size), Image.LANCZOS
            )
            tensors.append(to_tensor(img))
        return torch.stack(tensors)

    def forward(self, x: Tensor) -> Tensor:
        freq = torch.fft.fft2(x, dim=(-2, -1))
        freq = torch.fft.fftshift(freq, dim=(-2, -1))
        mag = torch.log1p(freq.abs())
        pooled = self.pool(mag)
        return pooled.flatten(1)
