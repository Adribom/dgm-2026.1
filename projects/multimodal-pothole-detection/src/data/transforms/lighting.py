"""Lighting and color augmentation helpers for pothole image prototyping."""

from __future__ import annotations

from PIL import Image
import albumentations as A
import numpy as np


def _pil_to_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a uint8 NumPy array."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _array_to_pil(image_array: np.ndarray) -> Image.Image:
    """Convert a NumPy array back to a PIL RGB image."""
    clipped = np.clip(image_array, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped, mode="RGB")


def _collect_replay_params(serialized: dict) -> dict:
    """Collect applied params from an Albumentations replay tree."""
    params: dict = {}

    if not isinstance(serialized, dict):
        return params

    if serialized.get("applied") and serialized.get("params"):
        params.update({k: v for k, v in serialized["params"].items() if not k.startswith("_")})

    for child in serialized.get("transforms", []):
        params.update(_collect_replay_params(child))

    return params


def _apply_replay_transform(image: Image.Image, transform: A.ReplayCompose) -> tuple[Image.Image, dict]:
    """Apply a replayable albumentations transform and return the result with its metadata."""
    transformed = transform(image=_pil_to_array(image))
    pil_image = _array_to_pil(transformed["image"])

    params = _collect_replay_params(transformed.get("replay", {}))

    return pil_image, params


def apply_fake_shadow(
    image: Image.Image,
    shadow_roi: tuple[float, float, float, float] = (0.0, 0.5, 1.0, 1.0),
    num_shadows_limit: tuple[int, int] = (1, 2),
    shadow_dimension: int = 5,
    shadow_intensity_range: tuple[float, float] = (0.4, 0.7),
) -> tuple[Image.Image, dict]:
    """Apply a road-style shadow overlay to a pothole image.

    Parameters
    ----------
    image:
        Input image.
    shadow_roi:
        Region where shadows may appear, as normalized coordinates.
    num_shadows_limit:
        Lower and upper limits for the number of shadow polygons.
    shadow_dimension:
        Number of polygon vertices used by the shadow generator.
    shadow_intensity_range:
        Shadow darkness range.

    Returns
    -------
    tuple[PIL.Image.Image, dict]
        Shadow-augmented image and transformation parameters.
    """
    transform = A.ReplayCompose(
        [
            A.RandomShadow(
                shadow_roi=shadow_roi,
                num_shadows_limit=num_shadows_limit,
                shadow_dimension=shadow_dimension,
                shadow_intensity_range=shadow_intensity_range,
                p=1.0,
            )
        ],
        p=1.0,
    )
    return _apply_replay_transform(image, transform)


def apply_color_jitter(
    image: Image.Image,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    saturation_limit: float = 0.2,
) -> tuple[Image.Image, dict]:
    """Apply color and brightness variation to a pothole image.

    Parameters
    ----------
    image:
        Input image.
    brightness_limit:
        Brightness variation magnitude.
    contrast_limit:
        Contrast variation magnitude.
    saturation_limit:
        Saturation variation magnitude.

    Returns
    -------
    tuple[PIL.Image.Image, dict]
        Color-jittered image and transformation parameters.
    """
    transform = A.ReplayCompose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=1.0,
            ),
            A.HueSaturationValue(
                sat_shift_limit=saturation_limit * 100,  # Albumentations uses degrees/percent scales
                val_shift_limit=0,
                hue_shift_limit=0,
                p=1.0,
            ),
        ],
        p=1.0,
    )
    return _apply_replay_transform(image, transform)
