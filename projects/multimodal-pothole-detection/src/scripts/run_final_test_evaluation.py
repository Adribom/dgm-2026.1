"""Run final-test evaluation over saved Point-E predictions.

This script keeps the evaluation step separate from inference: it validates the
prepared test set and prediction folders, delegates metric computation to the
shared final-test helper, and writes a detailed CSV plus aggregate summary into
the dedicated reports folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src modules can be resolved when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.final_test_eval import (
    FINAL_TEST_REPORT_CSV,
    FINAL_TEST_SUMMARY_JSON,
    build_report_frame,
    build_summary,
    ensure_final_test_dirs,
    resolve_prepared_test_paths,
    write_report,
    write_summary,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for final-test evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate saved final-test predictions and write the final reports.")
    parser.add_argument("--predictions_dir", type=str, required=True, help="Directory containing saved prediction .npy files")
    parser.add_argument("--prepared_test_set", type=str, required=True, help="Path to the prepared test-set root")
    parser.add_argument("--ground_truth_dir", type=str, default=None, help="Optional override for the prepared ground-truth cloud directory")
    parser.add_argument("--report_csv", type=str, default=str(FINAL_TEST_REPORT_CSV), help="Output CSV path for per-sample metrics")
    parser.add_argument("--summary_json", type=str, default=str(FINAL_TEST_SUMMARY_JSON), help="Output JSON path for the aggregate summary")
    return parser


def main() -> int:
    """Evaluate the saved predictions and write the final outputs."""

    parser = build_parser()
    args = parser.parse_args()

    paths = ensure_final_test_dirs()

    predictions_dir = Path(args.predictions_dir)
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found at {predictions_dir}")

    prepared_test_set = Path(args.prepared_test_set)
    prepared_paths = resolve_prepared_test_paths(prepared_test_set)
    if not prepared_paths.root.exists():
        raise FileNotFoundError(f"Prepared test set not found at {prepared_paths.root}")

    ground_truth_dir = Path(args.ground_truth_dir) if args.ground_truth_dir else prepared_paths.clouds_dir
    if not ground_truth_dir.exists():
        raise FileNotFoundError(f"Prepared ground-truth clouds directory not found at {ground_truth_dir}")
    if not prepared_paths.metadata_path.exists():
        raise FileNotFoundError(f"Prepared metadata file not found at {prepared_paths.metadata_path}")

    report_frame = build_report_frame(predictions_dir, prepared_test_set, ground_truth_dir)
    report_csv_path = Path(args.report_csv)
    if report_csv_path.exists():
        raise FileExistsError(
            f"Final-test report already exists at {report_csv_path}. Remove it before rerunning evaluation."
        )
    write_report(report_frame, report_csv_path)

    summary = build_summary(report_frame)
    summary_path = Path(args.summary_json)
    if summary_path.exists():
        raise FileExistsError(
            f"Final-test summary already exists at {summary_path}. Remove it before rerunning evaluation."
        )
    write_summary(summary, summary_path)

    print(f"Final-test evaluation completed. Report: {report_csv_path.resolve()}")
    print(f"Summary written to: {summary_path.resolve()}")
    print(f"Final-test output area: {paths['reports_dir'].resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())