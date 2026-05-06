#!/usr/bin/env python3
"""Reusable geometry helpers for pothole depth and mask analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation


@dataclass(frozen=True)
class CameraSpec:
    """Approximate D415-like camera specification used for exploratory metrics."""

    img_width: int = 640
    img_height: int = 480
    fov_h_deg: float = 69.4
    fov_v_deg: float = 42.5


def parse_yolo_polygon_line(line: str, width: int, height: int) -> np.ndarray:
    """Parse one YOLO segmentation line into integer pixel coordinates."""
    parts = line.strip().split()
    if len(parts) < 7:
        return np.empty((0, 2), dtype=np.int32)

    coords = np.asarray(parts[1:], dtype=np.float32).reshape(-1, 2)
    coords[:, 0] *= width
    coords[:, 1] *= height
    return coords.astype(np.int32)


def load_yolo_mask(label_path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    """Load YOLO segmentation labels and return a binary mask."""
    height, width = image_shape
    mask_img = Image.new("L", (width, height), 0)

    if not label_path.exists():
        return np.zeros((height, width), dtype=np.uint8)

    drawer = ImageDraw.Draw(mask_img)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        pts = parse_yolo_polygon_line(line, width, height)
        if pts.size == 0:
            continue
        drawer.polygon([(int(x), int(y)) for x, y in pts], fill=255)

    return np.array(mask_img, dtype=np.uint8)


def get_pixel_area_mm2(depth_mm: np.ndarray | float, camera: CameraSpec = CameraSpec()) -> np.ndarray:
    """Estimate world-area represented by a pixel at a given depth."""
    depth_mm = np.asarray(depth_mm, dtype=np.float32)
    fov_h_rad = np.deg2rad(camera.fov_h_deg)
    fov_v_rad = np.deg2rad(camera.fov_v_deg)

    width_mm = 2.0 * depth_mm * np.tan(fov_h_rad / 2.0)
    height_mm = 2.0 * depth_mm * np.tan(fov_v_rad / 2.0)
    pixel_width_mm = width_mm / camera.img_width
    pixel_height_mm = height_mm / camera.img_height
    return pixel_width_mm * pixel_height_mm


def estimate_road_surface_depth(depth: np.ndarray, mask: np.ndarray, ring_width: int = 50) -> float | None:
    """Estimate local road surface depth around pothole boundary."""
    pothole = mask > 0
    if not np.any(pothole):
        return None

    ring = binary_dilation(pothole, structure=np.ones((ring_width, ring_width), dtype=bool)) & ~pothole
    ring_depth = depth[ring & (depth > 0)]

    # Fallback to global background if local ring is too sparse.
    if ring_depth.size < 20:
        ring_depth = depth[(~pothole) & (depth > 0)]

    if ring_depth.size == 0:
        return None
    return float(np.median(ring_depth))


def calculate_volume_variable_area(
    depth: np.ndarray,
    mask: np.ndarray,
    road_surface_depth_mm: float,
    camera: CameraSpec = CameraSpec(),
) -> dict[str, float | np.ndarray]:
    """Estimate pothole volume using per-pixel area scaling with depth."""
    pothole = mask > 0
    pothole_depth = depth[pothole]

    if pothole_depth.size == 0:
        return {
            "volume_mm3": 0.0,
            "volume_cm3": 0.0,
            "volume_liters": 0.0,
            "mean_depth_mm": 0.0,
            "max_depth_mm": 0.0,
            "median_depth_mm": 0.0,
            "surface_area_cm2": 0.0,
        }

    depth_diff_map = np.zeros_like(depth, dtype=np.float32)
    depth_diff_map[pothole] = np.maximum(0.0, depth[pothole].astype(np.float32) - road_surface_depth_mm)

    pixel_area = get_pixel_area_mm2(depth[pothole], camera=camera)
    volume_mm3 = float(np.sum(depth_diff_map[pothole] * pixel_area))
    surface_area_mm2 = float(np.sum(pixel_area))

    positive_diffs = depth_diff_map[pothole]
    return {
        "volume_mm3": volume_mm3,
        "volume_cm3": volume_mm3 / 1000.0,
        "volume_liters": volume_mm3 / 1_000_000.0,
        "mean_depth_mm": float(np.mean(positive_diffs)),
        "max_depth_mm": float(np.max(positive_diffs)),
        "median_depth_mm": float(np.median(positive_diffs)),
        "surface_area_cm2": surface_area_mm2 / 100.0,
        "depth_diff_map": depth_diff_map,
    }
