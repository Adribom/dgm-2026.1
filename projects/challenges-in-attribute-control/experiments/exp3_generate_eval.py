"""
Phase 1 of dataset generation: Generate the evaluation dataset.

For each (object, color) pair in the shared taxonomy, generate N images
with the *unmodified* Stable Diffusion 1.5 model. The resulting image
set is the baseline against which the finetuned model will be compared
later (Phase 6), and the input to the VLM judge (Phase 2-3).

Outputs are written to:
    <output.root>/<object>/<color>/seed_<NNNN>.png    (PNG files)
    <output.root>/manifest.csv                        (one row per image)
    results/exp3_eval_generation/run_<ts>/metadata.json (per-run config)

The manifest is the canonical record of *what was generated*: every row
includes the object, color, seed, prompt, file path, and SHA-256 hash
of the resulting image. Downstream stages (VLM evaluation, analysis)
read this CSV — they never glob the filesystem, so missing or extra
files cannot silently bias results.

Usage (from repo root):
    python experiments/exp3_generate_eval.py --config configs/exp3_default.yaml

Optional flags:
    --output-root PATH      override output.root from the config
    --limit-pairs N         only generate the first N pairs (for smoke testing)
    --dry-run               print what would be generated, don't load the model

This script is idempotent: existing image files are skipped, so a Colab
session that disconnects mid-run resumes cleanly on re-execution.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from binding.io import hash_file, load_yaml, make_run_dir, save_run_metadata 
from binding.sd_generator import SDGenerator, params_from_config 
from binding.seeds import set_all_seeds  

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True,
                   help="Path to the experiment YAML config (e.g. configs/exp3_default.yaml).")
    p.add_argument("--output-root", default=None,
                   help="Override `output.root` from the config (e.g. for a smoke-test run).")
    p.add_argument("--limit-pairs", type=int, default=None,
                   help="Only generate the first N (object, color) pairs. Useful for smoke testing.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the work plan without loading the model or generating anything.")
    return p.parse_args()


def safe_name(s: str) -> str:
    """Make a string safe for a directory name (handles e.g. 'polar bear')."""
    return s.replace(" ", "_").lower()


def build_pair_list(taxonomy: dict, limit: int | None) -> list[tuple[str, str]]:
    """Cross-product of objects × colors, optionally truncated for smoke tests."""
    pairs = [(o, c) for o in taxonomy["objects"] for c in taxonomy["colors"]]
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def main() -> int:
    args = parse_args()

    cfg = load_yaml(args.config)
    taxonomy = load_yaml(cfg["pairs"]["taxonomy_path"])

    pairs = build_pair_list(taxonomy, args.limit_pairs)
    images_per_pair = int(cfg["sampling"]["images_per_pair"])
    seed_start = int(cfg["sampling"]["seed_start"])
    template = cfg["generation"]["template"]
    output_root = Path(args.output_root or cfg["output"]["root"])

    total_images = len(pairs) * images_per_pair
    print(f"[exp3] {len(pairs)} pairs × {images_per_pair} images = {total_images} images")
    print(f"[exp3] output root: {output_root}")
    print(f"[exp3] model: {cfg['model']['model_id']}")

    if args.dry_run:
        print("[exp3] --dry-run: not loading model or generating.")
        for o, c in pairs[:5]:
            print(f"        would generate: {template.format(color=c, object=o)} × {images_per_pair} seeds")
        if len(pairs) > 5:
            print(f"        ... and {len(pairs) - 5} more pairs.")
        return 0

    set_all_seeds(42)

    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir("exp3_eval_generation")
    save_run_metadata(
        run_dir,
        config=cfg,
        extra={"output_root": str(output_root), "n_pairs": len(pairs)},
    )

    manifest_path = output_root / "manifest.csv"
    is_new_manifest = not manifest_path.exists()
    manifest_f = manifest_path.open("a", newline="", encoding="utf-8")
    manifest = csv.writer(manifest_f)
    if is_new_manifest:
        manifest.writerow(["object", "color", "seed", "prompt", "path", "sha256"])

    print("[exp3] loading Stable Diffusion pipeline (this takes ~10-30s)…")
    generator = SDGenerator(
        model_id=cfg["model"]["model_id"],
        scheduler_name=cfg["generation"]["scheduler"],
        dtype=cfg["model"]["dtype"],
        revision=cfg["model"].get("revision"),
    )
    params = params_from_config(cfg["generation"])
    print(f"[exp3] pipeline loaded on device: {generator.device}")

    # ─── Generation loop ────────────────────────────────────────────────────
    # Why per-image hashing: it lets us assert that two runs (e.g. baseline
    # eval before vs after finetuning's own evaluation pipeline) used identical
    # baseline images. If the hashes differ, the comparison is contaminated.
    pbar = tqdm(total=total_images, desc="generating", unit="img")
    n_done = 0
    n_skipped = 0
    try:
        for obj, color in pairs:
            obj_safe = safe_name(obj)
            color_safe = safe_name(color)
            pair_dir = output_root / obj_safe / color_safe
            pair_dir.mkdir(parents=True, exist_ok=True)

            prompt = template.format(color=color, object=obj)

            for offset in range(images_per_pair):
                seed = seed_start + offset
                img_path = pair_dir / f"seed_{seed:04d}.png"

                if img_path.exists():
                    n_skipped += 1
                    pbar.update(1)
                    continue

                image = generator.generate_one(prompt=prompt, seed=seed, params=params)
                image.save(img_path)

                manifest.writerow([
                    obj, color, seed, prompt,
                    str(img_path.relative_to(output_root)),
                    hash_file(img_path),
                ])
                manifest_f.flush() 
                n_done += 1
                pbar.update(1)
    finally:
        pbar.close()
        manifest_f.close()

    print(f"[exp3] done. generated: {n_done}, skipped (resumed): {n_skipped}")
    print(f"[exp3] manifest: {manifest_path}")
    print(f"[exp3] run metadata: {run_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
