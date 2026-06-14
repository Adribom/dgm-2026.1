"""Prepare the Rui Fan benchmark as an isolated Point-E evaluation dataset.

This script reads the primary Rui Fan repository, merges the ground-truth PLY
sections per physical mold, standardizes each cloud to the fixed Point-E point
count, preserves source provenance, and writes the prepared benchmark into the
separate `data/processed/rui_fan_ready/` output root.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

# Ensure src modules can be resolved when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.point_e_pipeline_utils import format_point_e_tensor

DEFAULT_RUI_FAN_ROOT = PROJECT_ROOT / "data" / "raw" / "rui_fan_dataset"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "rui_fan_ready"
DEFAULT_TARGET_POINT_COUNT = 1024
DEFAULT_SEED = 42
PRIMARY_DATASET_DIRNAME = "rethinking_road_reconstruction_pothole_detection-main"
PRIMARY_DATASET_SUBDIR = "dataset"
SAMPLES_METADATA_KEY = "samples"
SCALES_METADATA_KEY = "scales"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the Rui Fan preparation workflow.

    Returns:
        argparse.ArgumentParser: Parser configured with source root, output root,
        target point count, and deterministic seed options.
    """
    parser = argparse.ArgumentParser(
        description="Prepare the Rui Fan benchmark as an isolated Point-E evaluation dataset.",
    )
    parser.add_argument(
        "--rui-fan-root",
        type=Path,
        default=DEFAULT_RUI_FAN_ROOT,
        help="Root directory containing the cloned Rui Fan repositories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Dedicated output directory for the prepared benchmark.",
    )
    parser.add_argument(
        "--target-point-count",
        type=int,
        default=DEFAULT_TARGET_POINT_COUNT,
        help="Fixed number of points to save per prepared ground-truth cloud.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for reproducible upsampling in point-cloud standardization.",
    )
    return parser


def resolve_primary_dataset_root(rui_fan_root: Path) -> Path:
    """Resolve the primary Rui Fan dataset root from the cloned repository tree.

    Args:
        rui_fan_root: Root directory that contains the Rui Fan repositories.

    Returns:
        Path: Absolute path to `rethinking_road_reconstruction_pothole_detection-main/dataset`.

    Raises:
        FileNotFoundError: If the primary repository or dataset folder cannot be found.
    """
    primary_root = rui_fan_root / PRIMARY_DATASET_DIRNAME / PRIMARY_DATASET_SUBDIR
    if not primary_root.exists():
        raise FileNotFoundError(
            f"Primary Rui Fan dataset not found at {primary_root}. "
            f"Expected {PRIMARY_DATASET_DIRNAME}/{PRIMARY_DATASET_SUBDIR} under {rui_fan_root}."
        )
    return primary_root


def build_output_root(output_root: Path) -> dict[str, Path]:
    """Create the isolated output directory structure for the benchmark.

    Args:
        output_root: Dedicated prepared-data directory for the Rui Fan benchmark.

    Returns:
        dict[str, Path]: Mapping with keys `images`, `clouds`, and `metadata`.
    """
    images_dir = output_root / "images"
    clouds_dir = output_root / "clouds"
    output_root.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    clouds_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.json"
    return {"images": images_dir, "clouds": clouds_dir, "metadata": metadata_path}


def read_ply_points(ply_path: Path) -> np.ndarray:
    """Read a PLY file and return its XYZ coordinates.

    Args:
        ply_path: Path to a `.ply` file.

    Returns:
        np.ndarray: Array of shape [N, 3] with float32 XYZ coordinates.
    """
    point_cloud = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(point_cloud.points, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"Empty point cloud loaded from {ply_path}")
    return points


def merge_gt_sections(ply_paths: list[Path]) -> np.ndarray:
    """Merge multiple PLY sections that describe the same physical mold.

    Args:
        ply_paths: Ordered list of PLY section paths from one model folder.

    Returns:
        np.ndarray: Concatenated XYZ points from all sections.
    """
    point_arrays = [read_ply_points(ply_path) for ply_path in ply_paths]
    return np.concatenate(point_arrays, axis=0)


def standardize_point_cloud(points: np.ndarray, target_point_count: int) -> tuple[np.ndarray, float]:
    """Normalize a point cloud to the Point-E contract using the existing helper.

    The helper from `point_e_pipeline_utils` performs fixed-size sampling and
    computes the spatial scale factor. We reuse it here by passing neutral RGB
    values and then keeping only the normalized XYZ coordinates.

    Args:
        points: Raw merged XYZ points.
        target_point_count: Final number of points to retain or expand to.

    Returns:
        tuple[np.ndarray, float]: A normalized [K, 3] cloud and the scale factor.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected [N, 3] point array, got {points.shape}")

    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    tensor_6d, scale_factor = format_point_e_tensor(points, colors, num_points=target_point_count)
    return tensor_6d[:, :3].astype(np.float32), float(scale_factor)


def discover_model_folders(primary_dataset_root: Path) -> list[Path]:
    """Return the primary model folders in the Rui Fan benchmark tree.

    Args:
        primary_dataset_root: Path to `rethinking_road_reconstruction_pothole_detection-main/dataset`.

    Returns:
        list[Path]: Sorted model folder paths.
    """
    return sorted(
        path for path in primary_dataset_root.iterdir()
        if path.is_dir() and path.name.startswith("model")
    )


def discover_left_images(model_dir: Path) -> list[Path]:
    """Return the left-view images for a given model folder.

    Args:
        model_dir: Path to a source model directory.

    Returns:
        list[Path]: Sorted list of left-view image paths.
    """
    left_dir = model_dir / "left images"
    if not left_dir.exists():
        return []
    return sorted(path for path in left_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")


def discover_gt_sections(model_dir: Path) -> list[Path]:
    """Return the ground-truth PLY sections for a given model folder.

    Args:
        model_dir: Path to a source model directory.

    Returns:
        list[Path]: Sorted list of `.ply` section paths.
    """
    gt_dir = model_dir / "gt"
    if not gt_dir.exists():
        return []
    return sorted(path for path in gt_dir.iterdir() if path.is_file() and path.suffix.lower() == ".ply")


def prepare_model_samples(
    model_dir: Path,
    images_dir: Path,
    clouds_dir: Path,
    target_point_count: int,
) -> tuple[list[dict], int]:
    """Prepare all samples for one physical Rui Fan model folder.

    Args:
        model_dir: Source model directory containing left images and GT sections.
        images_dir: Destination directory for prepared images.
        clouds_dir: Destination directory for prepared clouds.
        target_point_count: Fixed number of points for each prepared cloud.

    Returns:
        tuple[list[dict], int]: Metadata records and number of flagged samples.
    """
    model_name = model_dir.name
    left_images = discover_left_images(model_dir)
    gt_sections = discover_gt_sections(model_dir)

    if not left_images or not gt_sections:
        return [], 1

    merged_points = merge_gt_sections(gt_sections)
    normalized_points, scale_factor = standardize_point_cloud(merged_points, target_point_count)

    sample_records: list[dict] = []
    flagged_items = 0

    for image_path in left_images:
        sample_stem = f"{model_name}_{image_path.stem}"
        output_image_path = images_dir / f"{sample_stem}{image_path.suffix.lower()}"
        output_cloud_path = clouds_dir / f"{sample_stem}.npy"

        if output_image_path.exists() or output_cloud_path.exists():
            flagged_items += 1
            continue

        shutil.copy2(image_path, output_image_path)
        np.save(output_cloud_path, normalized_points)

        sample_records.append(
            {
                "sample_id": sample_stem,
                "source_repository": PRIMARY_DATASET_DIRNAME,
                "source_dataset": PRIMARY_DATASET_SUBDIR,
                "source_model": model_name,
                "source_image": image_path.name,
                "gt_sections": [section.name for section in gt_sections],
                "point_count": int(target_point_count),
                "scale_factor": scale_factor,
            }
        )

    return sample_records, flagged_items


def write_metadata(metadata_path: Path, records: list[dict], target_point_count: int) -> None:
    """Persist the prepared-benchmark metadata contract.

    Args:
        metadata_path: Destination metadata JSON path.
        records: Per-sample metadata records.
        target_point_count: Fixed point count used for standardization.
    """
    metadata = {
        "dataset": "rui_fan_ready",
        "point_count": int(target_point_count),
        SCALES_METADATA_KEY: {record["sample_id"]: float(record["scale_factor"]) for record in records},
        SAMPLES_METADATA_KEY: {
            record["sample_id"]: {
                key: value
                for key, value in record.items()
                if key not in {"sample_id", "scale_factor"}
            }
            for record in records
        },
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    """Parse CLI arguments and prepare the Rui Fan benchmark dataset."""
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    primary_dataset_root = resolve_primary_dataset_root(args.rui_fan_root)
    paths = build_output_root(args.output_dir)

    model_dirs = discover_model_folders(primary_dataset_root)
    if not model_dirs:
        raise FileNotFoundError(f"No model folders found under {primary_dataset_root}")

    all_records: list[dict] = []
    discovered_models = 0
    discovered_images = 0
    flagged_items = 0

    for model_dir in model_dirs:
        discovered_models += 1
        model_records, model_flagged = prepare_model_samples(
            model_dir=model_dir,
            images_dir=paths["images"],
            clouds_dir=paths["clouds"],
            target_point_count=args.target_point_count,
        )
        flagged_items += model_flagged
        discovered_images += len(discover_left_images(model_dir))
        all_records.extend(model_records)

    write_metadata(paths["metadata"], all_records, args.target_point_count)

    summary = {
        "discovered_models": discovered_models,
        "discovered_images": discovered_images,
        "prepared_samples": len(all_records),
        "flagged_items": flagged_items,
        "output_root": str(args.output_dir.resolve()),
        "primary_dataset_root": str(primary_dataset_root.resolve()),
        "point_count": int(args.target_point_count),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Prepared benchmark written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
