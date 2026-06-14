"""Geometric augmentation helpers for paired pothole images and point clouds."""

from __future__ import annotations

from PIL import Image, ImageOps
import numpy as np


def _normalize_point_cloud(point_cloud: np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalize a point cloud to point-first layout.

    Parameters
    ----------
    point_cloud:
        Point cloud array in one of the accepted layouts: [K, 3], [K, 6], [3, K], [6, K].

    Returns
    -------
    tuple[np.ndarray, bool]
        A copy of the array normalized to [K, N] and a flag indicating whether the input
        had to be transposed.
    """
    if point_cloud.ndim != 2:
        raise ValueError(f"Expected a 2D point cloud array, got shape {point_cloud.shape}.")

    array = np.array(point_cloud, copy=True)
    was_transposed = array.shape[0] in (3, 6) and array.shape[1] not in (3, 6)
    if was_transposed:
        array = array.T

    if array.shape[1] not in (3, 6):
        raise ValueError(
            "Unsupported point cloud shape "
            f"{point_cloud.shape}. Expected [K, 3], [K, 6], [3, K], or [6, K]."
        )

    return array, was_transposed


def _restore_point_cloud_shape(point_cloud: np.ndarray, was_transposed: bool) -> np.ndarray:
    """Restore a normalized point cloud to its original layout.

    Parameters
    ----------
    point_cloud:
        Normalized point cloud in [K, N] layout.
    was_transposed:
        Whether the original input had a channel-first layout.

    Returns
    -------
    np.ndarray
        Point cloud in the original layout.
    """
    return point_cloud.T if was_transposed else point_cloud


def horizontal_flip(image: Image.Image, point_cloud: np.ndarray) -> tuple[Image.Image, np.ndarray, dict]:
    """Flip an image horizontally and mirror the associated point cloud along the X axis.

    Parameters
    ----------
    image:
        Input pothole image.
    point_cloud:
        Associated point cloud array in any of the accepted Point-E layouts.

    Returns
    -------
    tuple[PIL.Image.Image, np.ndarray, dict]
        The flipped image, the point cloud with its X coordinate negated, and metadata.
    """
    flipped_image = ImageOps.mirror(image)
    normalized_point_cloud, was_transposed = _normalize_point_cloud(point_cloud)
    flipped_point_cloud = np.array(normalized_point_cloud, copy=True)
    flipped_point_cloud[:, 0] *= -1
    return (
        flipped_image,
        _restore_point_cloud_shape(flipped_point_cloud, was_transposed),
        {"flipped": True},
    )