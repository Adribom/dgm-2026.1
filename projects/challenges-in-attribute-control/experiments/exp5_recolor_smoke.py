"""
Phase 5, Pipeline B — smoke test for segmentation + HSV recoloration.

Processes a small representative sample from the source pool, producing
side-by-side "strip" images of [original | mask overlay | recolored] for
visual inspection. Run BEFORE the full pipeline to confirm:
  * Grounded-SAM 2 segments each canonical object reliably
  * HSV recoloration produces plausible images across materials
  * Mask area thresholds aren't rejecting too many cases

If the strips look good, green-light the full run. If they look bad
(masks wrong, colors leaking into background, objects unrecognizable),
this is where we iterate on the pipeline.

Usage (Colab, after installing sam2 + transformers):
    python experiments/exp5_recolor_smoke.py \\
        --pool /content/source_pool_colab.csv \\
        --out-dir /content/recolor_smoke \\
        --per-object 3 \\
        --colors blue purple
"""
from __future__ import annotations

import argparse
import csv
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from binding.seeds import set_all_seeds 
from binding.segment_recolor import SegmentationPipeline, recolor_hsv  

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pool", type=Path, required=True,
                   help="source_pool.csv from exp5_build_source_pool.py")
    p.add_argument("--out-dir", type=Path, default=Path("data/finetuning/recolor_smoke"),
                   help="Where to write strips and the smoke_log.csv.")
    p.add_argument("--per-object", type=int, default=3,
                   help="Sources per canonical object (default 3).")
    p.add_argument("--colors", nargs="+", default=["blue", "purple"],
                   help="Target colors to recolor each source into (default: blue purple).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def load_pool(path: Path) -> dict[str, list[dict]]:
    by_obj: dict[str, list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_obj[row["object"]].append(row)
    return dict(by_obj)

def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a red tint where the mask is True, for visual inspection."""
    out = image_rgb.copy()
    red = np.array([220, 60, 60], dtype=np.uint8)
    blend = (out.astype(np.float32) * 0.55 + red.astype(np.float32) * 0.45).astype(np.uint8)
    out[mask] = blend[mask]
    return out

def make_strip(panels: list[np.ndarray], labels: list[str]) -> Image.Image:
    """Place panels side by side with text labels above each."""
    H = max(p.shape[0] for p in panels)
    W = max(p.shape[1] for p in panels)
    label_h = 28
    canvas = np.full((H + label_h, W * len(panels), 3), 255, dtype=np.uint8)
    for i, p in enumerate(panels):
        h, w = p.shape[:2]
        y0 = label_h + (H - h) // 2
        x0 = i * W + (W - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = p
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for i, label in enumerate(labels):
        draw.text((i * W + 10, 4), label, fill=(0, 0, 0), font=font)
    return img

def main() -> int:
    args = parse_args()
    set_all_seeds(args.seed)

    pool = load_pool(args.pool)
    print(f"[smoke] loaded {sum(len(v) for v in pool.values())} sources across {len(pool)} objects")

    samples: list[tuple[str, str, dict]] = []
    for obj in sorted(pool.keys()):
        for src in pool[obj][: args.per_object]:
            for color in args.colors:
                if color == src["original_color"]:
                    continue
                samples.append((obj, color, src))
    print(f"[smoke] selected {len(samples)} (source, target_color) pairs")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    strips_dir = args.out_dir / "strips"
    strips_dir.mkdir(exist_ok=True)

    print(f"[smoke] loading Grounded-SAM 2...")
    seg = SegmentationPipeline()
    print(f"[smoke] pipeline ready on {seg.device}")

    log_rows = []
    n_ok = 0
    n_seg_fail = 0
    n_mask_fail = 0

    for i, (obj, target, src) in enumerate(samples, 1):
        src_path = Path(src["path"])
        original_color = src["original_color"]
        try:
            img = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"  [{i}/{len(samples)}] {obj:<14} {target:<8} OPEN_FAIL: {e}")
            log_rows.append(dict(
                object=obj, original_color=original_color, target_color=target,
                source_path=str(src_path), strip_path="",
                segmentation="open_fail", mask_area=0.0, recolor_accepted=False,
            ))
            continue

        try:
            mask = seg.segment(img, object_name=obj)
        except Exception as e:
            traceback.print_exc()
            print(f"  [{i}/{len(samples)}] {obj:<14} {target:<8} SEG_ERROR: {e}")
            mask = None

        if mask is None:
            n_seg_fail += 1
            print(f"  [{i}/{len(samples)}] {obj:<14} {target:<8} no_mask")
            log_rows.append(dict(
                object=obj, original_color=original_color, target_color=target,
                source_path=str(src_path), strip_path="",
                segmentation="fail", mask_area=0.0, recolor_accepted=False,
            ))
            continue

        rgb = np.array(img)
        result = recolor_hsv(rgb, mask, target)
        area = result.mask_area_frac

        if result.accepted:
            right_panel = result.image
            right_label = f"recolored ({target})"
        else:
            right_panel = np.full_like(rgb, 200)
            right_label = f"REJECTED: {result.reason}"
        strip = make_strip(
            [rgb, overlay_mask(rgb, mask), right_panel],
            [f"original ({original_color})",
             f"mask: {area:.0%} area",
             right_label],
        )
        safe_obj = obj.replace(" ", "_")
        src_stem = src_path.stem
        strip_path = strips_dir / f"{safe_obj}__{src_stem}__to_{target}.png"
        strip.save(strip_path)

        if not result.accepted:
            n_mask_fail += 1
            print(f"  [{i}/{len(samples)}] {obj:<14} {target:<8} MASK_REJ area={area:.1%}")
        else:
            n_ok += 1
            print(f"  [{i}/{len(samples)}] {obj:<14} {target:<8} OK       area={area:.1%}")

        log_rows.append(dict(
            object=obj, original_color=original_color, target_color=target,
            source_path=str(src_path), strip_path=str(strip_path),
            segmentation="ok", mask_area=area, recolor_accepted=result.accepted,
        ))

    log_path = args.out_dir / "smoke_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "object", "original_color", "target_color", "source_path",
            "strip_path", "segmentation", "mask_area", "recolor_accepted",
        ])
        writer.writeheader()
        for row in log_rows:
            writer.writerow(row)

    total = len(samples)
    print()
    print(f"[smoke] done. processed {total} samples")
    print(f"  recolored (accepted):  {n_ok}")
    print(f"  mask area rejected:    {n_mask_fail}")
    print(f"  segmentation fails:    {n_seg_fail}")
    print(f"  strips in:             {strips_dir}")
    print(f"  log:                   {log_path}")
    print()
    print(f"[smoke] INSPECT THE STRIPS visually before running the full pipeline.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
