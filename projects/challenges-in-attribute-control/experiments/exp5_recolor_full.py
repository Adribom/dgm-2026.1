"""
Phase 5, Pipeline B — full recoloration of treated pairs.

For each treated pair (e.g. banana × blue):
  1. Select source images of the object from the pool (banana sources)
  2. For each target image we want to generate:
       a. Pick a source (cycling through with augmentation seed)
       b. Apply light augmentation (crop, hflip, brightness/contrast)
       c. Segment with Grounding DINO + SAM2
       d. Recolor inside the mask via HSV substitution
       e. Validate with VLM-judge (Qwen2.5-VL)
       f. If approved, save to final dataset; otherwise discard
  3. Continue until --per-pair approvals or --max-attempts hit

Idempotent: rows already in approved.csv are skipped. Reruns add to the
existing dataset instead of overwriting.

Output structure:
    <out-root>/<object>/<target_color>/img_NNN.png    (approved recolorations)
    <out-root>/approved.csv     (final dataset manifest)
    <out-root>/rejected.csv     (with reasons, for debugging)
    <out-root>/summary.json     (per-pair stats)

Usage (Colab):
    python experiments/exp5_recolor_full.py \\
        --pool /content/source_pool_colab.csv \\
        --split results/exp5_split/finetuning_split.csv \\
        --out-root /content/recolor_treated \\
        --config configs/judge_default.yaml \\
        --per-pair 15 \\
        --max-attempts 25
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from binding.augment import augment_image  
from binding.io import load_yaml  
from binding.seeds import set_all_seeds  
from binding.segment_recolor import SegmentationPipeline, recolor_hsv  
from binding.vlm_judge import VLMJudge  

APPROVED_FIELDS = [
    "object", "target_color", "source_path", "source_original_color",
    "augment_seed", "mask_area", "image_path",
    "vlm_object_predicted", "vlm_color_predicted",
]
REJECTED_FIELDS = [
    "object", "target_color", "source_path", "source_original_color",
    "augment_seed", "stage", "reason",
    "vlm_object_predicted", "vlm_color_predicted",
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pool", type=Path, required=True,
                   help="source_pool.csv from exp5_build_source_pool.py")
    p.add_argument("--split", type=Path, required=True,
                   help="finetuning_split.csv from exp5_build_split.py")
    p.add_argument("--out-root", type=Path, default=Path("data/finetuning/recolor_treated"),
                   help="Output directory for approved images, manifests, and rejections.")
    p.add_argument("--config", type=Path, default=Path("configs/judge_default.yaml"),
                   help="VLM-judge config (same model as Phase 2).")
    p.add_argument("--per-pair", type=int, default=15,
                   help="Target number of approved images per pair (default 15).")
    p.add_argument("--max-attempts", type=int, default=25,
                   help="Max recoloration attempts per pair (default 25). Caps work even "
                        "if VLM rejection rate is high.")
    p.add_argument("--groups", nargs="+", default=["treated"],
                   help="Which split groups to process. Default: treated only.")
    p.add_argument("--chromatic-boost-mode", choices=["default", "aggressive_achromatic"],
                   default="default",
                   help="How to handle chromatic recoloration. 'default' (used for the "
                        "188 approved in the first run) applies modest saturation boost. "
                        "'aggressive_achromatic' detects achromatic source regions "
                        "(mean S < 30 in mask) and forces high saturation, which is "
                        "needed for transitions like 'white polar bear → orange'. Use "
                        "this when re-running only the pairs that came up empty.")
    p.add_argument("--pairs", nargs="+", default=None,
                   help="OPTIONAL: explicit list of 'object:color' pairs to process, "
                        "overriding the split filtering. Use when re-running specific "
                        "pairs (e.g. the zero-approved ones from a previous run).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def load_pool(path: Path) -> dict[str, list[dict]]:
    by_obj: dict[str, list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_obj[row["object"]].append(row)
    return dict(by_obj)

def load_treated_pairs(split_path: Path, groups: list[str]) -> list[tuple[str, str]]:
    pairs = []
    with split_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["group"] in groups:
                pairs.append((row["object"], row["color"]))
    return pairs

def already_approved_count(approved_path: Path, obj: str, color: str) -> int:
    """How many approved images this pair already has (for idempotent resume)."""
    if not approved_path.exists():
        return 0
    n = 0
    with approved_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["object"] == obj and row["target_color"] == color:
                n += 1
    return n

def main() -> int:
    args = parse_args()
    set_all_seeds(args.seed)

    pool = load_pool(args.pool)
    if args.pairs:
        treated_pairs = []
        for spec in args.pairs:
            if ":" not in spec:
                print(f"[full] ERROR: --pairs entry {spec!r} must be 'object:color'")
                return 1
            o, c = spec.rsplit(":", 1)
            treated_pairs.append((o.strip(), c.strip()))
        print(f"[full] {len(treated_pairs)} pairs from --pairs override (ignoring --groups)")
    else:
        treated_pairs = load_treated_pairs(args.split, args.groups)
        print(f"[full] {len(treated_pairs)} pairs to recolor")
    print(f"[full] chromatic_boost_mode = {args.chromatic_boost_mode}")
    print(f"[full] pool covers {len(pool)} objects")

    missing = []
    for obj, color in treated_pairs:
        if obj not in pool or not pool[obj]:
            missing.append((obj, color))
    if missing:
        print(f"[full] WARNING: {len(missing)} pairs have no source images:")
        for o, c in missing:
            print(f"    {o:<14} x {c}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    approved_path = args.out_root / "approved.csv"
    rejected_path = args.out_root / "rejected.csv"

    is_new_a = not approved_path.exists()
    is_new_r = not rejected_path.exists()
    af = approved_path.open("a", newline="", encoding="utf-8")
    rf = rejected_path.open("a", newline="", encoding="utf-8")
    aw = csv.DictWriter(af, fieldnames=APPROVED_FIELDS)
    rw = csv.DictWriter(rf, fieldnames=REJECTED_FIELDS)
    if is_new_a:
        aw.writeheader(); af.flush()
    if is_new_r:
        rw.writeheader(); rf.flush()

    print(f"[full] loading Grounded-SAM 2...")
    seg = SegmentationPipeline()

    print(f"[full] loading VLM-judge...")
    cfg = load_yaml(args.config)
    judge = VLMJudge(
        model_id=cfg["judge"]["model_id"],
        dtype=cfg["judge"].get("dtype", "bfloat16"),
    )
    print(f"[full] all models loaded")

    pair_stats: dict[tuple[str, str], dict] = {}

    for pair_idx, (obj, target_color) in enumerate(treated_pairs, 1):
        if obj not in pool or not pool[obj]:
            print(f"\n[full] ({pair_idx}/{len(treated_pairs)}) {obj} x {target_color}: SKIP no sources")
            pair_stats[(obj, target_color)] = {"approved": 0, "attempts": 0, "skip_no_source": True}
            continue

        already = already_approved_count(approved_path, obj, target_color)
        if already >= args.per_pair:
            print(f"\n[full] ({pair_idx}/{len(treated_pairs)}) {obj} x {target_color}: already has {already} approved, skipping")
            pair_stats[(obj, target_color)] = {"approved": already, "attempts": 0, "skip_already_full": True}
            continue

        need = args.per_pair - already
        print(f"\n[full] ({pair_idx}/{len(treated_pairs)}) {obj} x {target_color}: "
              f"need {need} more (have {already})")

        sources = pool[obj]
        attempts = 0
        approved_this_run = 0
        
        safe_obj = obj.replace(" ", "_")
        pair_dir = args.out_root / safe_obj / target_color
        pair_dir.mkdir(parents=True, exist_ok=True)

        while approved_this_run < need and attempts < args.max_attempts:
            src_idx = attempts % len(sources)
            src = sources[src_idx]
            aug_seed = args.seed * 1_000_003 + hash((obj, target_color, attempts)) % 1_000_003
            attempts += 1

            src_path = Path(src["path"])
            try:
                base_img = Image.open(src_path).convert("RGB")
            except Exception as e:
                print(f"    attempt {attempts}: open fail {e}")
                rw.writerow({
                    "object": obj, "target_color": target_color,
                    "source_path": str(src_path), "source_original_color": src["original_color"],
                    "augment_seed": aug_seed,
                    "stage": "open", "reason": str(e),
                    "vlm_object_predicted": "", "vlm_color_predicted": "",
                })
                rf.flush()
                continue

            aug_img = augment_image(base_img, seed=aug_seed)
            rgb_array = np.array(aug_img)

            try:
                mask = seg.segment(aug_img, object_name=obj)
            except Exception as e:
                traceback.print_exc()
                rw.writerow({
                    "object": obj, "target_color": target_color,
                    "source_path": str(src_path), "source_original_color": src["original_color"],
                    "augment_seed": aug_seed,
                    "stage": "segmentation", "reason": f"exception:{type(e).__name__}",
                    "vlm_object_predicted": "", "vlm_color_predicted": "",
                })
                rf.flush()
                continue

            if mask is None:
                rw.writerow({
                    "object": obj, "target_color": target_color,
                    "source_path": str(src_path), "source_original_color": src["original_color"],
                    "augment_seed": aug_seed,
                    "stage": "segmentation", "reason": "no detection",
                    "vlm_object_predicted": "", "vlm_color_predicted": "",
                })
                rf.flush()
                continue

            result = recolor_hsv(
                rgb_array, mask, target_color,
                chromatic_boost_mode=args.chromatic_boost_mode,
            )
            if not result.accepted:
                rw.writerow({
                    "object": obj, "target_color": target_color,
                    "source_path": str(src_path), "source_original_color": src["original_color"],
                    "augment_seed": aug_seed,
                    "stage": "recolor", "reason": result.reason,
                    "vlm_object_predicted": "", "vlm_color_predicted": "",
                })
                rf.flush()
                continue

            tentative_idx = already + approved_this_run
            img_path = pair_dir / f"img_{tentative_idx:03d}.png"
            
            while img_path.exists():
                tentative_idx += 1
                img_path = pair_dir / f"img_{tentative_idx:03d}.png"
            Image.fromarray(result.image).save(img_path)

            
            try:
                judgment = judge.judge_image(
                    image_path=img_path,
                    expected_object=obj,
                    expected_color=target_color,
                )
            except Exception as e:
                traceback.print_exc()
                img_path.unlink(missing_ok=True)
                rw.writerow({
                    "object": obj, "target_color": target_color,
                    "source_path": str(src_path), "source_original_color": src["original_color"],
                    "augment_seed": aug_seed,
                    "stage": "vlm", "reason": f"exception:{type(e).__name__}",
                    "vlm_object_predicted": "", "vlm_color_predicted": "",
                })
                rf.flush()
                continue

            if not judgment.binding_correct:
                img_path.unlink(missing_ok=True)  
                reason_parts = []
                if judgment.object_predicted != obj:
                    reason_parts.append(f"object:{judgment.object_predicted}")
                if judgment.color_predicted != target_color:
                    reason_parts.append(f"color:{judgment.color_predicted}")
                rw.writerow({
                    "object": obj, "target_color": target_color,
                    "source_path": str(src_path), "source_original_color": src["original_color"],
                    "augment_seed": aug_seed,
                    "stage": "vlm", "reason": " ".join(reason_parts) or "binding_no",
                    "vlm_object_predicted": judgment.object_predicted,
                    "vlm_color_predicted": judgment.color_predicted,
                })
                rf.flush()
                continue

            aw.writerow({
                "object": obj, "target_color": target_color,
                "source_path": str(src_path),
                "source_original_color": src["original_color"],
                "augment_seed": aug_seed,
                "mask_area": result.mask_area_frac,
                "image_path": str(img_path.relative_to(args.out_root)).replace("\\", "/"),
                "vlm_object_predicted": judgment.object_predicted,
                "vlm_color_predicted": judgment.color_predicted,
            })
            af.flush()
            approved_this_run += 1
            if approved_this_run % 5 == 0 or approved_this_run == need:
                print(f"    attempt {attempts:>3}: {approved_this_run}/{need} approved")

        if approved_this_run < need:
            print(f"    ⚠️  pair finished with {approved_this_run}/{need} approved "
                  f"(max attempts hit at {attempts})")
        pair_stats[(obj, target_color)] = {
            "approved": already + approved_this_run,
            "attempts": attempts,
        }

    af.close(); rf.close()

    total_approved = sum(s["approved"] for s in pair_stats.values())
    print(f"\n[full] DONE")
    print(f"[full] total approved images: {total_approved}")
    print(f"[full] per-pair breakdown:")
    for (o, c), stats in pair_stats.items():
        if stats.get("skip_no_source"):
            print(f"    {o:<14} {c:<8}  no source — SKIPPED")
        elif stats.get("skip_already_full"):
            print(f"    {o:<14} {c:<8}  {stats['approved']:>2}/{args.per_pair}  (already full)")
        else:
            flag = "  ⚠️ short" if stats["approved"] < args.per_pair else ""
            print(f"    {o:<14} {c:<8}  {stats['approved']:>2}/{args.per_pair}  "
                  f"({stats['attempts']} attempts){flag}")

    summary = {
        "total_approved": total_approved,
        "target_per_pair": args.per_pair,
        "per_pair": {
            f"{o}__{c}": stats for (o, c), stats in pair_stats.items()
        },
    }
    with (args.out_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return 0

if __name__ == "__main__":
    sys.exit(main())
