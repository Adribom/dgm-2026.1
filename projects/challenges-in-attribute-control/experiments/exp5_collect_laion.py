"""
Phase 5, Pipeline A, Step 1: collect LAION candidates for 'under' pairs.

Reads the finetuning split, selects the treated pairs whose image_source
is 'laion_recaption' (the 'under' category), streams LAION-400M to find
gap candidates (caption mentions object+color but without syntactic
binding), downloads the images, and saves them for VLM verification.

This step does NOT verify with the VLM or rewrite captions yet — it only
harvests raw candidate images. Verification (Step 2) and re-captioning
(Step 3) are separate scripts so each can be re-run independently.

Output:
    <out-root>/<object>/<color>/cand_NNN.png    (downloaded candidates)
    <out-root>/candidates_manifest.csv          (provenance: url, orig caption)

The manifest records the original LAION URL and caption for provenance and
debugging ONLY. Per the design rule, those captions never reach training.

Usage:
    python experiments/exp5_collect_laion.py \\
        --split results/exp5_split/finetuning_split.csv \\
        --out-root data/finetuning/candidates \\
        --per-pair 30 \\
        --max-scan 3000000

Note on --per-pair: collect MORE than the final target (e.g. 30 to end up
with 15), because the VLM verification step will reject a fraction.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from binding.laion_collect import CollectionTargets, download_image, iter_candidates  
from binding.seeds import set_all_seeds 

import re  

def object_color_pair(text: str, obj: str, color: str) -> bool:
    """Mirror of Experiment 1's object_color_pair (syntactic binding detector)."""
    text = text.lower()
    o = re.escape(obj.lower())
    c = re.escape(color.lower())
    patterns = [
        rf"\b{c}\s+{o}\b",                       
        rf"\b{o}\s+(?:is|are|was|were|looks?|appears?|turned|became|got)\s+{c}\b",
        rf"\b{o}'s\s+{c}\b",                      
        rf"\b{c}-colou?red\s+{o}\b",              
    ]
    return any(re.search(p, text) for p in patterns)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", type=Path, required=True,
                   help="finetuning_split.csv from exp5_build_split.py")
    p.add_argument("--out-root", type=Path, default=Path("data/finetuning/candidates"))
    p.add_argument("--per-pair", type=int, default=30,
                   help="Candidates to collect per pair (collect extra; VLM rejects some).")
    p.add_argument("--max-scan", type=int, default=3_000_000,
                   help="Max LAION rows to inspect before giving up.")
    p.add_argument("--groups", nargs="+", default=["treated"],
                   help="Which split groups to collect for. Default: treated only.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_under_pairs(split_path: Path, groups: list[str]) -> list[tuple[str, str]]:
    """Pairs in the requested groups whose image_source is laion_recaption."""
    pairs = []
    with split_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["group"] in groups and row["image_source"] == "laion_recaption":
                pairs.append((row["object"], row["color"]))
    return pairs


def main() -> int:
    args = parse_args()
    set_all_seeds(args.seed)

    pairs = load_under_pairs(args.split, args.groups)
    if not pairs:
        print(f"[collect] no laion_recaption pairs found in groups {args.groups}")
        return 1
    print(f"[collect] {len(pairs)} 'under' pairs to collect for:")
    for o, c in pairs:
        print(f"    {o} × {c}")

    targets = CollectionTargets(needed={pair: args.per_pair for pair in pairs})

    print(f"\n[collect] loading LAION-400M (streaming)…")
    
    import os
    from datasets import load_dataset
    from huggingface_hub import HfFolder

    token = os.environ.get("HF_TOKEN") or HfFolder.get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Either:\n"
            "  (a) set the HF_TOKEN environment variable, or\n"
            "  (b) run `huggingface-cli login` once on this machine.\n"
            "The token also needs access to https://huggingface.co/datasets/laion/laion400m "
            "(accept the dataset terms on the page if you haven't)."
        )

    dataset = load_dataset(
        "laion/laion400m", split="train", streaming=True, token=token,
    )
    dataset = dataset.shuffle(buffer_size=10000, seed=args.seed)

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "candidates_manifest.csv"
    is_new = not manifest_path.exists()
    mf = manifest_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(mf)
    if is_new:
        writer.writerow(["object", "color", "cand_idx", "path", "url", "original_caption"])

    saved_counts: dict[tuple[str, str], int] = {pair: 0 for pair in pairs}
    n_downloaded = 0
    n_download_fail = 0

    print(f"[collect] scanning (up to {args.max_scan:,} rows)…")
    for cand in iter_candidates(dataset, targets, object_color_pair, max_scan=args.max_scan):
        img = download_image(cand.url)
        if img is None:
            n_download_fail += 1
            targets.needed[(cand.object_name, cand.color)] += 1
            continue

        key = (cand.object_name, cand.color)
        idx = saved_counts[key]
        obj_safe = cand.object_name.replace(" ", "_")
        pair_dir = args.out_root / obj_safe / cand.color
        pair_dir.mkdir(parents=True, exist_ok=True)
        img_path = pair_dir / f"cand_{idx:03d}.png"
        img.save(img_path)
    
        rel_path = img_path.relative_to(args.out_root).as_posix()
        writer.writerow([
            cand.object_name, cand.color, idx,
            rel_path,
            cand.url, cand.original_caption[:300],
        ])
        mf.flush()
        saved_counts[key] += 1
        n_downloaded += 1
        if n_downloaded % 10 == 0:
            print(f"    downloaded {n_downloaded} (failed {n_download_fail}) | "
                  f"remaining target {targets.remaining()}")

    mf.close()

    print(f"\n[collect] done. downloaded {n_downloaded}, failed {n_download_fail}")
    print(f"[collect] per-pair results:")
    for pair in pairs:
        got = saved_counts[pair]
        flag = "" if got >= args.per_pair * 0.5 else "  ⚠️ LOW"
        print(f"    {pair[0]:<12} {pair[1]:<8} {got}/{args.per_pair}{flag}")
    print(f"\n[collect] manifest: {manifest_path}")
    print(f"[collect] next: VLM-verify these candidates (exp5_verify_candidates.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
