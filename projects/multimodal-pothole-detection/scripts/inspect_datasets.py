#!/usr/bin/env python3
"""Inspect pothole datasets and build a manifest for downstream processing."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


POTHRGBD_IMAGE_PATTERN = re.compile(r"^(?P<sample_id>\d{8}_\d{6})_color_png\.rf\.[^.]+\.jpg$")
POTHRGBD_DEPTH_PATTERN = re.compile(r"^(?P<sample_id>\d{8}_\d{6})_depth(?: \(\d+\))?\.npy$")
POTHRGBD_LABEL_PATTERN = re.compile(r"^(?P<sample_id>\d{8}_\d{6})_color_png\.rf\.[^.]+\.txt$")


@dataclass(frozen=True)
class SampleRow:
    dataset: str
    sample_id: str
    image_path: str | None
    depth_path: str | None
    label_path: str | None
    status: str
    notes: str


def iter_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".DS_Store"))


def match_pothrgbd_file(name: str) -> tuple[str | None, str | None]:
    for pattern, modality in (
        (POTHRGBD_IMAGE_PATTERN, "image"),
        (POTHRGBD_DEPTH_PATTERN, "depth"),
        (POTHRGBD_LABEL_PATTERN, "label"),
    ):
        match = pattern.match(name)
        if match:
            return match.group("sample_id"), modality
    return None, None


def inspect_pothrgbd(dataset_root: Path) -> tuple[list[SampleRow], dict[str, object]]:
    images_dir = dataset_root / "images"
    depths_dir = dataset_root / "depths"
    labels_dir = dataset_root / "labels"

    grouped: dict[str, dict[str, list[Path]]] = defaultdict(lambda: {"image": [], "depth": [], "label": []})
    extension_counts: Counter[str] = Counter()

    for directory in (images_dir, depths_dir, labels_dir):
        for file_path in iter_files(directory):
            extension_counts[file_path.suffix.lower() or "[no_ext]"] += 1
            sample_id, modality = match_pothrgbd_file(file_path.name)
            if sample_id and modality:
                grouped[sample_id][modality].append(file_path)

    rows: list[SampleRow] = []
    duplicate_samples: list[dict[str, object]] = []
    for sample_id in sorted(grouped):
        modalities = grouped[sample_id]
        image_paths = modalities["image"]
        depth_paths = modalities["depth"]
        label_paths = modalities["label"]

        notes: list[str] = []
        if len(image_paths) != 1:
            notes.append(f"images={len(image_paths)}")
        if len(depth_paths) != 1:
            notes.append(f"depths={len(depth_paths)}")
        if len(label_paths) != 1:
            notes.append(f"labels={len(label_paths)}")

        if len(image_paths) != 1 or len(depth_paths) != 1 or len(label_paths) != 1:
            duplicate_samples.append(
                {
                    "sample_id": sample_id,
                    "image_count": len(image_paths),
                    "depth_count": len(depth_paths),
                    "label_count": len(label_paths),
                }
            )

        status = "ok" if not notes else "check"
        rows.append(
            SampleRow(
                dataset="PothRGDB",
                sample_id=sample_id,
                image_path=str(image_paths[0].relative_to(dataset_root)) if image_paths else None,
                depth_path=str(depth_paths[0].relative_to(dataset_root)) if depth_paths else None,
                label_path=str(label_paths[0].relative_to(dataset_root)) if label_paths else None,
                status=status,
                notes="; ".join(notes),
            )
        )

    summary = {
        "dataset": "PothRGDB",
        "root": str(dataset_root),
        "file_counts": dict(sorted(extension_counts.items())),
        "sample_count": len(rows),
        "samples_with_issues": sum(1 for row in rows if row.status != "ok"),
        "duplicate_samples": duplicate_samples,
    }
    return rows, summary


def inspect_rui_fan(dataset_root: Path) -> dict[str, object]:
    summary: dict[str, object] = {"dataset": "rui_fan_dataset", "root": str(dataset_root), "subsets": []}

    for top_level in sorted(p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        subset_info: dict[str, object] = {"name": top_level.name, "file_counts": {}, "subdirs": []}

        direct_files = [p for p in top_level.iterdir() if p.is_file() and not p.name.startswith(".")]
        if direct_files:
            subset_info["file_counts"] = dict(Counter(p.suffix.lower() or "[no_ext]" for p in direct_files))

        for nested in sorted(p for p in top_level.iterdir() if p.is_dir() and not p.name.startswith(".")):
            counts = Counter()
            for file_path in nested.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith("."):
                    counts[file_path.suffix.lower() or "[no_ext]"] += 1
            subset_info["subdirs"].append(
                {
                    "name": nested.name,
                    "relative_path": str(nested.relative_to(dataset_root)),
                    "file_counts": dict(sorted(counts.items())),
                    "file_total": sum(counts.values()),
                }
            )

        summary["subsets"].append(subset_info)

    return summary


def write_csv(rows: Iterable[SampleRow], path: Path) -> None:
    fieldnames = ["dataset", "sample_id", "image_path", "depth_path", "label_path", "status", "notes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(data: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root folder that contains datasets, docs, notebooks, and scripts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where inspection artifacts will be written. Defaults to <code-root>/artifacts.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    code_root = args.code_root.resolve()
    output_dir = (args.output_dir or (code_root / "artifacts")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pothrgbd_root = code_root / "datasets" / "PothRGDB" / "PUBLIC POTHOLE DATASET"
    rui_fan_root = code_root / "datasets" / "rui_fan_dataset"

    if not pothrgbd_root.exists():
        raise FileNotFoundError(f"Missing PothRGDB dataset root: {pothrgbd_root}")
    if not rui_fan_root.exists():
        raise FileNotFoundError(f"Missing Rui Fan dataset root: {rui_fan_root}")

    poth_rows, poth_summary = inspect_pothrgbd(pothrgbd_root)
    rui_summary = inspect_rui_fan(rui_fan_root)

    write_csv(poth_rows, output_dir / "pothrgbd_manifest.csv")
    write_json(poth_summary, output_dir / "pothrgbd_summary.json")
    write_json(rui_summary, output_dir / "rui_fan_summary.json")

    print(json.dumps({"pothrgbd": poth_summary, "rui_fan": rui_summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())