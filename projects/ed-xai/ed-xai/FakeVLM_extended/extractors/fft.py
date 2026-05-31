from enum import Enum
from typing import List,Callable

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.transforms.functional import resize, to_tensor
from PIL import Image

from .base import BaseFrequencyExtractor

import os
import sys
import matplotlib.pyplot as plt

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


def plot_fft_spectra(image_path: str, output_path: str = None) -> None:
    """Plots and saves the FFT spectra of an image.

    This function follows the forward process without the pooling method,
    ploting the log-magnitude spectrum and the phase spectrum.

    If output_path is not provided, the plot is saved in the same directory
    as the input image, with the suffix '_fft_spectra.png'.

    Args:
        image_path: Path to the input image file.
        output_path: Path to save the output plot. If None, defaults to
            <image_path_without_extension>_fft_spectra.png.
    """
    if output_path is None:
        base = os.path.splitext(os.path.abspath(image_path))[0]
        output_path = f"{base}_fft_spectra.png"

    img = Image.open(image_path).convert("RGB")
    extractor = FFTExtractor()

    x = extractor.preprocess([img]).float()
    freq = torch.fft.fft2(x, dim=(-2, -1))
    freq = torch.fft.fftshift(freq, dim=(-2, -1))

    magnitude = torch.log1p(freq.abs())[0].mean(0).numpy()
    phase = freq.angle()[0].mean(0).numpy()

    _, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(magnitude, cmap="inferno")
    axes[1].set_title("Log-magnitude spectrum")
    axes[1].axis("off")

    axes[2].imshow(phase, cmap="twilight")        
    axes[2].set_title("Phase spectrum")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Graph saves in {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_fft_spectra(sys.argv[1])