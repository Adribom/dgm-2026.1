"""
Utility: normalize backslashes to forward slashes in a CSV path column.

Use when a manifest was written on Windows (backslashes in `path` field)
and needs to be consumed on Linux/Colab. Edits the file in place after
making a .bak copy.

Usage:
    python experiments/exp5_fix_manifest_paths.py \\
        --csv data/finetuning/candidates/candidates_manifest.csv \\
        --column path
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--column", default="path",
                   help="Name of the column whose values should be normalized.")
    args = p.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    bak = args.csv.with_suffix(args.csv.suffix + ".bak")
    shutil.copy2(args.csv, bak)

    with args.csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if args.column not in fieldnames:
            raise KeyError(f"column {args.column!r} not in CSV header: {fieldnames}")
        rows = list(reader)

    fixed = 0
    for row in rows:
        original = row[args.column]
        normalized = original.replace("\\", "/")
        if normalized != original:
            row[args.column] = normalized
            fixed += 1

    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[fix] {fixed} of {len(rows)} rows had backslashes in '{args.column}' → normalized")
    print(f"[fix] backup at {bak}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
