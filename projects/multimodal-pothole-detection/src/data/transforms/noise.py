"""Noise and masking augmentation helpers for pothole image prototyping."""

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


def apply_gaussian_blur(
    image: Image.Image,
    blur_limit: tuple[int, int] = (3, 7),
    sigma_limit: tuple[float, float] = (0.5, 1.5),
) -> tuple[Image.Image, dict]:
    """Apply Gaussian blur to a pothole image.

    Parameters
    ----------
    image:
        Input image.
    blur_limit:
        Kernel size range.
    sigma_limit:
        Gaussian sigma range.

    Returns
    -------
    tuple[PIL.Image.Image, dict]
        Blurred image and transformation parameters.
    """
    transform = A.ReplayCompose(
        [
            A.GaussianBlur(
                blur_limit=blur_limit,
                sigma_limit=sigma_limit,
                p=1.0,
            )
        ],
        p=1.0,
    )
    return _apply_replay_transform(image, transform)


def apply_motion_blur(
    image: Image.Image,
    blur_limit: tuple[int, int] = (7, 15),
    angle_range: tuple[float, float] = (0.0, 360.0),
    direction_range: tuple[float, float] = (0.0, 0.0),
    allow_shifted: bool = True,
) -> tuple[Image.Image, dict]:
    """Apply directional motion blur to a pothole image.

    Parameters
    ----------
    image:
        Input image.
    blur_limit:
        Kernel size range.
    angle_range:
        Angle range in degrees used to orient the blur.
    direction_range:
        Direction bias for the blur kernel.
    allow_shifted:
        Whether to allow a shifted kernel.

    Returns
    -------
    tuple[PIL.Image.Image, dict]
        Motion-blurred image and transformation parameters.
    """
    transform = A.ReplayCompose(
        [
            A.MotionBlur(
                blur_limit=blur_limit,
                angle_range=angle_range,
                direction_range=direction_range,
                allow_shifted=allow_shifted,
                p=1.0,
            )
        ],
        p=1.0,
    )
    return _apply_replay_transform(image, transform)


def apply_cutout(
    image: Image.Image,
    num_holes_range: tuple[int, int] = (1, 3),
    hole_height_range: tuple[float, float] = (0.1, 0.2),
    hole_width_range: tuple[float, float] = (0.1, 0.2),
    fill: int = 0,
) -> tuple[Image.Image, dict]:
    """Apply rectangular cutout regions to a pothole image.

    Parameters
    ----------
    image:
        Input image.
    num_holes_range:
        Number of masked regions to apply.
    hole_height_range:
        Relative height range of the masked regions.
    hole_width_range:
        Relative width range of the masked regions.
    fill:
        Fill value used in the masked regions.

    Returns
    -------
    tuple[PIL.Image.Image, dict]
        Image with masked patches and transformation parameters.
    """
    transform = A.ReplayCompose(
        [
            A.CoarseDropout(
                num_holes_range=num_holes_range,
                hole_height_range=hole_height_range,
                hole_width_range=hole_width_range,
                fill=fill,
                p=1.0,
            )
        ],
        p=1.0,
    )
    return _apply_replay_transform(image, transform)
