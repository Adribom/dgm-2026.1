"""
Phase 5, Frente 3: LLM-based caption rewriting.

For each approved training image (treated + control), generate 3 short,
grounded captions using an open-source LLM. Outputs a CSV manifest with
one row per (image, caption) pair, ready to be consumed by Phase 6
(LoRA finetuning).

Inputs:
  --treated-approved : approved.csv from exp5_recolor_full.py
                       (200 recolored images from treated pairs)
  --treated-root     : the directory root for the relative paths in the
                       treated approved.csv
  --control-approved : approved.csv from exp5_verify_candidates.py (control)
                       (71 LAION images of canonically-colored pairs)
  --control-root     : the directory root for the relative paths in the
                       control approved.csv

Output:
  --out-csv : single train.csv with columns
              role, object, color, image_path, caption, caption_idx
              where:
                role        = "treated" | "control"
                image_path  = ABSOLUTE path to the image on disk
                caption     = clean text caption
                caption_idx = 0, 1, 2 (3 captions per image)

Each image yields up to 3 rows (one per caption). If the LLM produces
fewer than 3 valid captions, we record what we got and log it; this is
idempotent — a rerun will retry pairs that came up short.

Captions are also validated post-generation: each caption MUST mention
the object name and color name (case-insensitive). Captions failing this
check are discarded and re-generated up to --max-retries times.

Usage (Colab Pro with A100, after Pipeline B finished):
    python experiments/exp5_recaption_llm.py \\
        --treated-approved /content/recolor_treated/approved.csv \\
        --treated-root /content/recolor_treated \\
        --control-approved /content/drive/MyDrive/binding-research/finetuning/control_verified/approved.csv \\
        --control-root /content/drive/MyDrive/binding-research/finetuning/control_candidates \\
        --out-csv /content/train.csv \\
        --captions-per-image 3 \\
        --max-retries 2
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from binding.llm_recaption import CaptionGenerator, validate_caption  
from binding.seeds import set_all_seeds  

TRAIN_FIELDS = ["role", "object", "color", "image_path", "caption", "caption_idx"]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--treated-approved", type=Path, required=True,
                   help="approved.csv from exp5_recolor_full.py")
    p.add_argument("--treated-root", type=Path, required=True,
                   help="Directory root for relative paths in treated approved.csv")
    p.add_argument("--control-approved", type=Path, required=True,
                   help="approved.csv from exp5_verify_candidates.py (control set)")
    p.add_argument("--control-root", type=Path, required=True,
                   help="Directory root for relative paths in control approved.csv")
    p.add_argument("--out-csv", type=Path, default=Path("data/finetuning/train.csv"),
                   help="Where to write the final train manifest")
    p.add_argument("--captions-per-image", type=int, default=3)
    p.add_argument("--max-retries", type=int, default=2,
                   help="How many times to re-generate if captions are insufficient/invalid")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def load_treated(approved_csv: Path, root: Path) -> list[dict]:
    rows = []
    with approved_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            abs_path = (root / r["image_path"]).resolve()
            rows.append({
                "role": "treated",
                "object": r["object"],
                "color": r["target_color"],
                "image_path": str(abs_path).replace("\\", "/"),
            })
    return rows

def load_control(approved_csv: Path, root: Path) -> list[dict]:
    rows = []
    with approved_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            abs_path = (root / r["path"]).resolve()
            rows.append({
                "role": "control",
                "object": r["object"],
                "color": r["color"],
                "image_path": str(abs_path).replace("\\", "/"),
            })
    return rows


def already_done(out_path: Path) -> set[tuple[str, int]]:
    """Return set of (image_path, caption_idx) already in the output."""
    if not out_path.exists():
        return set()
    seen = set()
    with out_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seen.add((r["image_path"], int(r["caption_idx"])))
    return seen

def main() -> int:
    args = parse_args()
    set_all_seeds(args.seed)

    treated = load_treated(args.treated_approved, args.treated_root)
    control = load_control(args.control_approved, args.control_root)
    all_rows = treated + control
    print(f"[recaption] loaded {len(treated)} treated + {len(control)} control = {len(all_rows)} images")

    seen = already_done(args.out_csv)
    print(f"[recaption] {len(seen)} (image, caption) pairs already in output (resume)")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not args.out_csv.exists()
    fout = args.out_csv.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=TRAIN_FIELDS)
    if is_new:
        writer.writeheader(); fout.flush()

    print(f"[recaption] loading LLM: {args.model_id}")
    gen = CaptionGenerator(model_id=args.model_id)

    n_done = 0
    n_invalid = 0
    n_short = 0
    n_images_complete = 0

    for img_idx, row in enumerate(all_rows, 1):
        img_path = row["image_path"]
        obj = row["object"]
        color = row["color"]

        existing = [c for (p, c) in seen if p == img_path]
        needed_indices = [i for i in range(args.captions_per_image) if i not in existing]
        if not needed_indices:
            n_images_complete += 1
            continue

        captions: list[str] = []
        for attempt in range(args.max_retries + 1):
            attempt_seed = args.seed * 100003 + img_idx * 7 + attempt
            try:
                result = gen.generate(obj, color, seed=attempt_seed)
            except Exception as e:
                print(f"  [{img_idx}/{len(all_rows)}] {obj} x {color}: LLM exception {e}")
                continue

            for c in result.captions:
                if validate_caption(c, obj, color) and c not in captions:
                    captions.append(c)
                if len(captions) >= args.captions_per_image:
                    break

            if len(captions) >= args.captions_per_image:
                break

            n_invalid += len(result.captions) - sum(
                1 for c in result.captions if validate_caption(c, obj, color)
            )

        wrote = 0
        for idx in needed_indices:
            if idx >= len(captions):
                break
            writer.writerow({
                "role": row["role"], "object": obj, "color": color,
                "image_path": img_path,
                "caption": captions[idx],
                "caption_idx": idx,
            })
            wrote += 1
        fout.flush()
        n_done += wrote

        if len(captions) < args.captions_per_image:
            n_short += 1
            short_status = f"  ⚠️ short ({len(captions)}/{args.captions_per_image})"
        else:
            short_status = ""

        if img_idx % 20 == 0 or img_idx == len(all_rows):
            print(f"  [{img_idx}/{len(all_rows)}] {row['role']:<8} {obj} x {color}: "
                  f"wrote {wrote} captions{short_status}")

    fout.close()

    print(f"\n[recaption] DONE")
    print(f"  total (image, caption) rows added: {n_done}")
    print(f"  images already complete (skipped): {n_images_complete}")
    print(f"  images that came up short:         {n_short}")
    print(f"  invalid captions (rejected):       {n_invalid}")
    print(f"  output: {args.out_csv}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
