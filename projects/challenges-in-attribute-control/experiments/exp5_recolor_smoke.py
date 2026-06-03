"""
Phase 5, Pipeline B — smoke test for segmentation + HSV recoloration.

Processes a small representative sample from the source pool, producing a
"strip" image per case that shows: original | mask | recolored. The strips
are saved to a directory you open and inspect visually to confirm the
pipeline produces usable training data before running on all sources.

For each of the 12 canonical objects we pick up to 3 source images and
recolor each into 2 target colors selected for visual contrast. Total
output: up to 72 strip images.

If the smoke output looks good (object recolored, fundo untouched, no
weird halos), green-light the full run. If it looks bad (masks wrong,
colors leaking into background, objects unrecognizable), this script is
where we iterate.

Usage (Colab, after installing sam2 + transformers):
    python experiments/exp5_recolor_smoke.py \\
        --pool results/exp5_pool/source_pool.csv \\
        --out-dir data/finetuning/recolor_smoke \\
        --per-object 3 \\
        --colors blue purple
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from binding.seeds import set_all_seeds 
from binding.segment_recolor import recolor_hsv  

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", type=Path, required=True,
                   help="source_pool.csv from exp5_build_source_pool.py")
    p.add_argument("--out-dir", type=Path, default=Path("data/finetuning/recolor_smoke"))
    p.add_argument("--per-object", type=int, default=3,
                   help="How many source images per object to process.")
    p.add_argument("--colors", nargs="+",
                   default=["blue", "purple"],
                   help="Target colors to recolor each source into (for the smoke).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def make_strip(original: np.ndarray, mask: np.ndarray, recolored: np.ndarray) -> Image.Image:
    """Build a single-row PIL image: original | mask | recolored."""
    h, w = original.shape[:2]
    mask_vis = np.zeros_like(original)
    mask_vis[mask] = [255, 0, 255]  
    blend = (0.5 * original + 0.5 * mask_vis).astype(np.uint8)
    strip = np.concatenate([original, blend, recolored], axis=1)
    return Image.fromarray(strip)

def main() -> int:
    args = parse_args()
    set_all_seeds(args.seed)
    rng = random.Random(args.seed)

    by_object: dict[str, list[dict]] = defaultdict(list)
    with args.pool.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_object[row["object"]].append(row)
    print(f"[smoke] loaded {sum(len(v) for v in by_object.values())} sources "
          f"across {len(by_object)} objects")

    picks: list[dict] = []
    for obj in sorted(by_object):
        candidates = by_object[obj][:]
        rng.shuffle(candidates)
        for src in candidates[: args.per_object]:
            picks.append(src)
    print(f"[smoke] selected {len(picks)} source images to process")
    print(f"[smoke] loading Grounded-SAM 2…")
    from binding.segment_recolor import SegmentationPipeline
    pipe = SegmentationPipeline()
    print(f"[smoke] pipeline ready on {pipe.device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_seg_fail = 0
    n_recolor_fail = 0

    log_path = args.out_dir / "smoke_log.csv"
    log_f = log_path.open("w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["object", "source_path", "target_color", "outcome",
                  "mask_area_fraction", "reason"])

    for src in picks:
        obj = src["object"]
        src_path = src["path"]
        try:
            image = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"  [skip] cannot open {src_path}: {e}")
            continue

        mask = pipe.segment(image, text_prompt=obj)
        if mask is None:
            n_seg_fail += 1
            for color in args.colors:
                log.writerow([obj, src_path, color, "seg_fail", 0.0, "no_detection"])
            log_f.flush()
            continue

        original = np.array(image)
        for color in args.colors:
            result = recolor_hsv(original, mask, color)
            if not result.success:
                n_recolor_fail += 1
                log.writerow([obj, src_path, color, "recolor_fail",
                              result.mask_area_fraction, result.reason])
                log_f.flush()
                continue
            strip = make_strip(original, mask, result.image)
            safe_obj = obj.replace(" ", "_")
            src_stem = Path(src_path).stem
            out_path = args.out_dir / f"{safe_obj}__{src_stem}__{color}.png"
            strip.save(out_path)
            log.writerow([obj, src_path, color, "ok",
                          result.mask_area_fraction, ""])
            log_f.flush()
            n_ok += 1

    log_f.close()
    total = n_ok + n_seg_fail * len(args.colors) + n_recolor_fail
    print(f"\n[smoke] done.")
    print(f"  recolored OK:        {n_ok}")
    print(f"  segmentation fails:  {n_seg_fail} sources (× {len(args.colors)} colors each)")
    print(f"  recolor fails:       {n_recolor_fail} (bad masks)")
    print(f"  log:                 {log_path}")
    print(f"  inspect strips in:   {args.out_dir}")
    print(f"\n[smoke] each strip is [original | mask overlay | recolored]")
    print(f"[smoke] open them in the file viewer to verify quality.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
