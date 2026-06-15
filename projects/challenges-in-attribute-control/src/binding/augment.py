"""
Light augmentation for the Pipeline B source images.

Applied BEFORE segmentation so the SAM mask aligns naturally with the
augmented frame. Augmentations chosen to be safe across all 12 canonical
objects:

  - random crop (scale 0.85-1.0): keeps the whole object visible
  - horizontal flip (p=0.5): semantically neutral
  - brightness ±10%, contrast ±10%: simulates lighting variation
  - NO hue/saturation jitter: those are controlled by the HSV recoloration
    step; cross-mixing would create interference
  - NO rotation: distorts box-like objects (chalkboard) and breaks
    expected composition

Deterministic given a seed. Two calls with the same seed produce the same
augmentation. The transform always returns RGB uint8.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance


@dataclass(frozen=True)
class AugmentationParams:
    crop_scale_min: float = 0.85
    crop_scale_max: float = 1.0
    hflip_prob: float = 0.5
    brightness_range: float = 0.10   
    contrast_range: float = 0.10     

DEFAULT_AUG = AugmentationParams()

def augment_image(
    image: Image.Image,
    seed: int,
    params: AugmentationParams = DEFAULT_AUG,
) -> Image.Image:
    """
    Apply deterministic light augmentation.

    Parameters
    ----------
    image  : PIL RGB image
    seed   : int — same seed → same augmentation
    params : AugmentationParams — adjust strengths if needed

    Returns
    -------
    PIL RGB image of the same size as input (crop is followed by resize back).
    """
    rng = random.Random(seed)
    out = image.convert("RGB")
    W, H = out.size

    scale = rng.uniform(params.crop_scale_min, params.crop_scale_max)
    new_W, new_H = int(W * scale), int(H * scale)
    if new_W < W or new_H < H:
        x0 = rng.randint(0, W - new_W)
        y0 = rng.randint(0, H - new_H)
        out = out.crop((x0, y0, x0 + new_W, y0 + new_H))
        out = out.resize((W, H), Image.LANCZOS)

    if rng.random() < params.hflip_prob:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)

    if params.brightness_range > 0:
        factor = 1.0 + rng.uniform(-params.brightness_range, params.brightness_range)
        out = ImageEnhance.Brightness(out).enhance(factor)

    if params.contrast_range > 0:
        factor = 1.0 + rng.uniform(-params.contrast_range, params.contrast_range)
        out = ImageEnhance.Contrast(out).enhance(factor)

    return out
