"""
Tests for the HSV recoloration math.

The segmentation pipeline (Grounding DINO + SAM2) is not tested here —
that requires a GPU and is validated empirically via the smoke-test
output. Here we hand-check the color transformations on synthetic inputs.
"""
from __future__ import annotations

import numpy as np
import pytest

from binding.segment_recolor import (
    CANONICAL_HUE,
    RecolorResult,
    _hsv_to_rgb_uint8,
    _rgb_to_hsv_uint8,
    recolor_hsv,
)

def make_solid(color_rgb: tuple[int, int, int], size: int = 64) -> np.ndarray:
    """A uniform solid-color image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[..., 0] = color_rgb[0]
    img[..., 1] = color_rgb[1]
    img[..., 2] = color_rgb[2]
    return img

def make_centered_mask(size: int = 64, fill_frac: float = 0.5) -> np.ndarray:
    """Boolean mask covering a centered square of the given area fraction."""
    side = int(np.sqrt(fill_frac) * size)
    mask = np.zeros((size, size), dtype=bool)
    start = (size - side) // 2
    mask[start:start + side, start:start + side] = True
    return mask

def test_rgb_hsv_roundtrip_keeps_pixels_close():
    """Converting RGB→HSV→RGB shouldn't drift more than a few units (quantization)."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    hsv = _rgb_to_hsv_uint8(img)
    back = _hsv_to_rgb_uint8(hsv)
    diff = np.abs(img.astype(int) - back.astype(int))
    
    assert diff.max() <= 8, f"max roundtrip diff = {diff.max()}"

def test_rejects_mask_too_small():
    img = make_solid((255, 0, 0))
    mask = make_centered_mask(fill_frac=0.02)  
    res = recolor_hsv(img, mask, "blue", min_area=0.05)
    assert not res.accepted
    assert "too small" in res.reason

def test_rejects_mask_too_large():
    img = make_solid((255, 0, 0))
    mask = np.ones((64, 64), dtype=bool)  
    res = recolor_hsv(img, mask, "blue", max_area=0.95)
    assert not res.accepted
    assert "too large" in res.reason

def test_outside_mask_is_preserved():
    """Pixels outside the mask should be untouched by recoloration."""
    img = make_solid((255, 0, 0))  
    mask = make_centered_mask(fill_frac=0.25)
    res = recolor_hsv(img, mask, "blue")
    assert res.accepted
    
    outside = res.image[~mask]
    expected = np.array([255, 0, 0])
    diff = np.abs(outside.astype(int) - expected[None, :]).max()
    assert diff <= 8, f"outside-mask pixel drift: {diff}"

@pytest.mark.parametrize("target,channel_should_dominate", [
    ("red", 0),
    ("green", 1),
    ("blue", 2),
])
def test_chromatic_recolor_makes_target_channel_dominate(target, channel_should_dominate):
    """After recoloring a gray patch to red/green/blue, that channel should be highest."""
    img = make_solid((128, 128, 128))  
    mask = make_centered_mask(fill_frac=0.5)
    res = recolor_hsv(img, mask, target)
    assert res.accepted
    inside = res.image[mask]
    mean_per_channel = inside.mean(axis=0)
    dominant = int(mean_per_channel.argmax())
    assert dominant == channel_should_dominate, (
        f"target={target}: expected channel {channel_should_dominate} to dominate, "
        f"got channel means {mean_per_channel.tolist()}"
    )

def test_white_recolor_produces_near_white():
    """Recoloring to 'white' should produce R≈G≈B and high brightness."""
    img = make_solid((180, 50, 50))  
    mask = make_centered_mask(fill_frac=0.5)
    res = recolor_hsv(img, mask, "white")
    assert res.accepted
    inside = res.image[mask]
    mean_per_channel = inside.mean(axis=0)
    assert (mean_per_channel >= 180).all(), f"white not bright enough: {mean_per_channel}"
    assert mean_per_channel.std() < 12, f"white not gray-balanced: {mean_per_channel}"

def test_black_recolor_produces_near_black():
    """Recoloring to 'black' should produce low values across all channels."""
    img = make_solid((200, 200, 0))  
    mask = make_centered_mask(fill_frac=0.5)
    res = recolor_hsv(img, mask, "black")
    assert res.accepted
    inside = res.image[mask]
    assert inside.max() < 100, f"black still bright: max={inside.max()}"

def test_brown_recolor_lands_in_brown_range():
    """Brown should be desaturated orange — R > G > B, with moderate brightness."""
    img = make_solid((100, 200, 200))  
    mask = make_centered_mask(fill_frac=0.5)
    res = recolor_hsv(img, mask, "brown")
    assert res.accepted
    inside = res.image[mask]
    mean_per_channel = inside.mean(axis=0)
    r, g, b = mean_per_channel
    assert r > g > b, f"brown should have R>G>B, got {mean_per_channel}"
    assert 60 <= r <= 220, f"brown out of brightness range: R={r}"

def test_unknown_color_raises():
    img = make_solid((128, 128, 128))
    mask = make_centered_mask(fill_frac=0.5)
    with pytest.raises(ValueError, match="unknown target_color"):
        recolor_hsv(img, mask, "magenta-burst")  

def test_mismatched_mask_shape_raises():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    bad_mask = np.zeros((48, 48), dtype=bool)
    with pytest.raises(ValueError, match="does not match"):
        recolor_hsv(img, bad_mask, "red")

def test_result_is_frozen():
    """RecolorResult is a frozen dataclass — accidental mutation should error."""
    img = make_solid((128, 128, 128))
    mask = make_centered_mask(fill_frac=0.5)
    res = recolor_hsv(img, mask, "red")
    with pytest.raises((AttributeError, TypeError)):
        res.accepted = False  
