from enum import Enum
from typing import List,Callable

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.transforms.functional import resize, to_tensor
from PIL import Image

from .base import BaseFrequencyExtractor


class FFTMode(Enum):
    MAGNITUDE = "magnitude"
    PHASE = "phase"

class FFTExtractor(BaseFrequencyExtractor):
    """Extracts frequency-domain features from images via 2D FFT.

    Deterministic, no learnable parameters. Applies 2D FFT per channel,
    centers the DC component, and extracts either log-magnitude or phase
    of the spectrum, controlled by the mode parameter. The result is
    pooled to a fixed spatial size and flattened into a feature vector.

    Modes:
        "magnitude" : log-magnitude (log1p(|F|)), captures spectral
            energy distribution; robust to phase noise.
        "phase" : raw phase angle (angle(F)), captures structural
            and edge information; sensitive to spatial alignment.
    """

    _TRANSFORMS: dict[FFTMode, Callable[[Tensor], Tensor]] = {
        FFTMode.MAGNITUDE: lambda f: torch.log1p(f.abs()),
        FFTMode.PHASE:     lambda f: f.angle()
    }

    def __init__(self, input_size: int = 224, pool_size: int = 32, mode: str = "magnitude"):
        super().__init__()
        self._input_size = input_size
        self._pool_size = pool_size
        self._transform = self._TRANSFORMS[FFTMode(mode)]
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
        input_dtype = x.dtype
        x = x.float()
        freq = torch.fft.fft2(x, dim=(-2, -1))
        freq = torch.fft.fftshift(freq, dim=(-2, -1))
        feature = self._transform(freq)
        pooled = self.pool(feature)
        return pooled.flatten(1).to(input_dtype)
