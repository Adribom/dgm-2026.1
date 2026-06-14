"""Shared helpers for the Rui Fan Point-E MVP evaluation workflow.

This module centralizes final-test output paths, run metadata, sample matching,
unit conversion, and the per-sample MVP metrics so the scripts can stay thin
and avoid duplicating comparison logic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.chamfer import compute_symmetric_chamfer_distance_points
from src.evaluation.evaluate_3d import calculate_p05_depth, classify_severity_cm


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FINAL_TEST_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "final_test_eval"
FINAL_TEST_PREDICTIONS_DIR = FINAL_TEST_ARTIFACTS_DIR / "predictions"
FINAL_TEST_REPORTS_DIR = PROJECT_ROOT / "reports" / "final_test_eval"
FINAL_TEST_RUN_INFO_PATH = FINAL_TEST_ARTIFACTS_DIR / "run_info.json"
FINAL_TEST_REPORT_CSV = FINAL_TEST_REPORTS_DIR / "evaluation_report.csv"
FINAL_TEST_SUMMARY_JSON = FINAL_TEST_REPORTS_DIR / "summary.json"
EXPECTED_SEVERITY_LABELS = ("low", "medium", "high")


@dataclass(frozen=True)
class FinalTestRunInfo:
    """Metadata captured for a final-test run.

    Attributes:
        checkpoint_path: Path to the checkpoint used for inference.
        prepared_test_set: Path to the prepared test-set root.
        prepared_test_metadata: Path to the prepared metadata.json file.
        predictions_dir: Directory where predictions were written.
        reports_dir: Directory where the reports were written.
        evaluated_samples: Number of samples processed in the run.
        generated_at: Timestamp for the run metadata.
    """

    checkpoint_path: str
    prepared_test_set: str
    prepared_test_metadata: str
    predictions_dir: str
    reports_dir: str
    evaluated_samples: int
    generated_at: str


@dataclass(frozen=True)
class PreparedTestSetPaths:
    """Resolved paths for the prepared Rui Fan benchmark."""

    root: Path
    images_dir: Path
    clouds_dir: Path
    metadata_path: Path


def ensure_final_test_dirs() -> dict[str, Path]:
    """Create the dedicated final-test directories if they do not exist.

    Returns:
        dict[str, Path]: A mapping with the standard final-test locations.
    """

    FINAL_TEST_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_TEST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_TEST_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "artifacts_dir": FINAL_TEST_ARTIFACTS_DIR,
        "predictions_dir": FINAL_TEST_PREDICTIONS_DIR,
        "reports_dir": FINAL_TEST_REPORTS_DIR,
        "run_info_path": FINAL_TEST_RUN_INFO_PATH,
        "report_csv": FINAL_TEST_REPORT_CSV,
        "summary_json": FINAL_TEST_SUMMARY_JSON,
    }


def resolve_prepared_test_paths(prepared_test_set: Path) -> PreparedTestSetPaths:
    """Resolve the prepared Rui Fan benchmark folders from the root path.

    Args:
        prepared_test_set: Root path of the prepared benchmark.

    Returns:
        PreparedTestSetPaths: Resolved benchmark folders and metadata path.
    """

    return PreparedTestSetPaths(
        root=prepared_test_set,
        images_dir=prepared_test_set / "images",
        clouds_dir=prepared_test_set / "clouds",
        metadata_path=prepared_test_set / "metadata.json",
    )


def load_sample_scales(metadata_path: Path) -> dict[str, float]:
    """Load the per-sample scale map from a prepared benchmark metadata file.

    Args:
        metadata_path: Path to the prepared Rui Fan metadata.json file.

    Returns:
        dict[str, float]: Mapping from sample id to scale value.
    """

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    scales = metadata.get("scales", {})
    if not isinstance(scales, dict):
        raise ValueError(f"Invalid scales mapping in metadata at {metadata_path}")

    return {str(key): float(value) for key, value in scales.items()}


def load_prediction_files(predictions_dir: Path) -> list[Path]:
    """Return the saved prediction files for a final-test evaluation run.

    Args:
        predictions_dir: Directory that contains `.npy` prediction files.

    Returns:
        list[Path]: Sorted list of prediction files.
    """

    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found at {predictions_dir}")
    return sorted(predictions_dir.glob("*.npy"))


def _extract_spatial_coords(points: np.ndarray) -> np.ndarray:
    """Return the first three coordinates as a dense spatial cloud.

    Args:
        points: Loaded point-cloud array.

    Returns:
        np.ndarray: Array with shape [N, 3].
    """

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected an [N, >=3] point-cloud array, got shape {points.shape}")
    return np.asarray(points[:, :3], dtype=np.float64)


def _align_ground_truth_for_comparison(ground_truth_spatial: np.ndarray) -> np.ndarray:
    """Align prepared ground truth to the negative-Z comparison convention.

    Args:
        ground_truth_spatial: Prepared ground-truth cloud in normalized space.

    Returns:
        np.ndarray: Ground-truth cloud with Z flipped to match prediction sign.
    """

    aligned = np.array(ground_truth_spatial, copy=True)
    aligned[:, 2] *= -1.0
    return aligned


def _prediction_to_normalized(predicted_spatial_m: np.ndarray, scale: float) -> np.ndarray:
    """Convert a saved prediction cloud in meters back to normalized space.

    Args:
        predicted_spatial_m: Prediction cloud saved in meters.
        scale: Per-sample scale value from the prepared metadata.

    Returns:
        np.ndarray: Prediction cloud in normalized prepared space.
    """

    if scale <= 0:
        raise ValueError("Scale must be positive to convert predictions back to normalized space")
    return predicted_spatial_m * 1000.0 / scale


def _ground_truth_to_meters(ground_truth_spatial: np.ndarray, scale: float) -> np.ndarray:
    """Convert a prepared ground-truth cloud to meters using the sample scale.

    Args:
        ground_truth_spatial: Prepared ground-truth cloud in normalized space.
        scale: Per-sample scale value from the prepared metadata.

    Returns:
        np.ndarray: Ground-truth cloud in meters with depth encoded as negative Z.
    """

    if scale <= 0:
        raise ValueError("Scale must be positive to convert ground truth to meters")

    aligned = _align_ground_truth_for_comparison(ground_truth_spatial)
    return aligned * scale / 1000.0


def _effective_depth_cm(coords_m: np.ndarray) -> float:
    """Compute the outlier-resistant effective depth in centimeters.

    Args:
        coords_m: Point cloud coordinates expressed in meters.

    Returns:
        float: P05-based depth in centimeters.
    """

    return calculate_p05_depth(coords_m) * 100.0


def _severity_confusion_matrix(report: pd.DataFrame) -> dict[str, int]:
    """Build the 3x3 severity confusion matrix from the evaluation report.

    Args:
        report: Per-sample evaluation table.

    Returns:
        dict[str, int]: Flat confusion-matrix counts keyed by expected/predicted labels.
    """

    matrix: dict[str, int] = {}
    if report.empty:
        for expected in EXPECTED_SEVERITY_LABELS:
            for predicted in EXPECTED_SEVERITY_LABELS:
                matrix[f"{expected}_to_{predicted}"] = 0
        return matrix

    table = pd.crosstab(
        report["severity_expected"],
        report["severity_predicted"],
        dropna=False,
    ).reindex(index=EXPECTED_SEVERITY_LABELS, columns=EXPECTED_SEVERITY_LABELS, fill_value=0)

    for expected in EXPECTED_SEVERITY_LABELS:
        for predicted in EXPECTED_SEVERITY_LABELS:
            matrix[f"{expected}_to_{predicted}"] = int(table.loc[expected, predicted])

    return matrix


def evaluate_prediction_file(
    prediction_path: Path,
    ground_truth_dir: Path,
    metadata_scales: dict[str, float],
) -> dict[str, object]:
    """Evaluate a single predicted cloud against its prepared ground truth.

    Args:
        prediction_path: Path to a saved `.npy` prediction.
        ground_truth_dir: Directory containing the prepared test-set `.npy` clouds.
        metadata_scales: Per-sample scale mapping from the prepared metadata.

    Returns:
        dict[str, object]: One row of metrics for the CSV report.
    """

    sample_id = prediction_path.stem
    ground_truth_path = ground_truth_dir / prediction_path.name
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found for {sample_id} at {ground_truth_path}")

    if sample_id not in metadata_scales:
        raise KeyError(f"Scale not found in metadata for sample {sample_id}")

    scale = metadata_scales[sample_id]
    predicted_cloud = np.load(prediction_path)
    ground_truth_cloud = np.load(ground_truth_path)

    predicted_spatial = _extract_spatial_coords(predicted_cloud)
    ground_truth_spatial = _extract_spatial_coords(ground_truth_cloud)

    predicted_normalized = _prediction_to_normalized(predicted_spatial, scale)
    ground_truth_normalized = _align_ground_truth_for_comparison(ground_truth_spatial)
    chamfer_distance = compute_symmetric_chamfer_distance_points(predicted_normalized, ground_truth_normalized)

    predicted_physical = predicted_spatial
    ground_truth_physical = _ground_truth_to_meters(ground_truth_spatial, scale)

    predicted_depth_cm = _effective_depth_cm(predicted_physical)
    ground_truth_depth_cm = _effective_depth_cm(ground_truth_physical)
    predicted_severity = classify_severity_cm(predicted_depth_cm)
    ground_truth_severity = classify_severity_cm(ground_truth_depth_cm)

    return {
        "sample_id": sample_id,
        "prediction_path": str(prediction_path),
        "ground_truth_path": str(ground_truth_path),
        "scale_factor": round(float(scale), 6),
        "geometric_fidelity": round(float(chamfer_distance), 6),
        "predicted_depth_cm": round(float(predicted_depth_cm), 6),
        "ground_truth_depth_cm": round(float(ground_truth_depth_cm), 6),
        "depth_error_cm": round(abs(float(predicted_depth_cm) - float(ground_truth_depth_cm)), 6),
        "severity_expected": ground_truth_severity,
        "severity_predicted": predicted_severity,
        "severity_match": predicted_severity == ground_truth_severity,
    }


def build_report_frame(
    predictions_dir: Path,
    prepared_test_set: Path,
    ground_truth_dir: Path | None = None,
) -> pd.DataFrame:
    """Build the per-sample evaluation table for the final-test run.

    Args:
        predictions_dir: Directory containing saved prediction files.
        prepared_test_set: Root path of the prepared Rui Fan benchmark.
        ground_truth_dir: Optional override for the prepared ground-truth clouds.

    Returns:
        pd.DataFrame: One row per evaluated sample.
    """

    prepared_paths = resolve_prepared_test_paths(prepared_test_set)
    resolved_ground_truth_dir = ground_truth_dir or prepared_paths.clouds_dir
    metadata_scales = load_sample_scales(prepared_paths.metadata_path)

    rows: list[dict[str, object]] = []
    for prediction_path in load_prediction_files(predictions_dir):
        rows.append(evaluate_prediction_file(prediction_path, resolved_ground_truth_dir, metadata_scales))

    report = pd.DataFrame(rows)
    if report.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "prediction_path",
                "ground_truth_path",
                "scale_factor",
                "geometric_fidelity",
                "predicted_depth_cm",
                "ground_truth_depth_cm",
                "depth_error_cm",
                "severity_expected",
                "severity_predicted",
                "severity_match",
            ]
        )

    ordered_columns = [
        "sample_id",
        "prediction_path",
        "ground_truth_path",
        "scale_factor",
        "geometric_fidelity",
        "predicted_depth_cm",
        "ground_truth_depth_cm",
        "depth_error_cm",
        "severity_expected",
        "severity_predicted",
        "severity_match",
    ]
    return report[ordered_columns].sort_values("sample_id").reset_index(drop=True)


def build_summary(report: pd.DataFrame) -> dict[str, object]:
    """Build the aggregate summary for the final-test run.

    Args:
        report: Per-sample metrics DataFrame.

    Returns:
        dict[str, object]: Summary metrics suitable for JSON output.
    """

    if report.empty:
        return {
            "evaluated_samples": 0,
            "mean_geometric_fidelity": None,
            "mean_depth_error_cm": None,
            "severity_match_rate": None,
            "severity_confusion_matrix": _severity_confusion_matrix(report),
        }

    return {
        "evaluated_samples": int(len(report)),
        "mean_geometric_fidelity": float(report["geometric_fidelity"].mean()),
        "mean_depth_error_cm": float(report["depth_error_cm"].mean()),
        "severity_match_rate": float(report["severity_match"].mean()),
        "severity_confusion_matrix": _severity_confusion_matrix(report),
    }


def write_run_info(run_info: FinalTestRunInfo, run_info_path: Path | None = None) -> Path:
    """Persist the final-test run metadata to JSON.

    Args:
        run_info: Structured run metadata to save.
        run_info_path: Optional override for the destination JSON path.

    Returns:
        Path: The path where the metadata was written.
    """

    resolved_path = run_info_path or FINAL_TEST_RUN_INFO_PATH
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(run_info), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return resolved_path


def write_report(report: pd.DataFrame, report_csv_path: Path | None = None) -> Path:
    """Persist the detailed per-sample evaluation report.

    Args:
        report: Per-sample metrics DataFrame.
        report_csv_path: Optional override for the CSV destination.

    Returns:
        Path: The path where the report was written.
    """

    resolved_path = report_csv_path or FINAL_TEST_REPORT_CSV
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(resolved_path, index=False)
    return resolved_path


def write_summary(summary: dict[str, object], summary_json_path: Path | None = None) -> Path:
    """Persist the aggregate evaluation summary.

    Args:
        summary: Aggregate metrics dictionary.
        summary_json_path: Optional override for the JSON destination.

    Returns:
        Path: The path where the summary was written.
    """

    resolved_path = summary_json_path or FINAL_TEST_SUMMARY_JSON
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return resolved_path