"""
Phase 5, Pipeline A, Step 2: VLM-verify the collected LAION candidates.

For each candidate image collected by exp5_collect_laion.py, ask the
VLM (Qwen2.5-VL-7B) two decomposed questions about the image alone
(NEVER showing the original LAION caption, which is unreliable):

    "What is the main object?"  → must map to the expected object
    "What color is the {object}?" → must map to the expected color

If both answers map to the expected (object, color), the candidate is
APPROVED for training; otherwise REJECTED.

This is the step that turns a noisy text-filtered candidate pool into
a clean image-verified training set. The acceptance rate measured here
DIMENSIONS the rest of Phase 5: if VLM approves 20% of candidates,
collecting 15 final images per pair requires harvesting ~75 raw candidates.

Idempotent: a row already in approved.csv or rejected.csv is skipped.
Reusable: the same script works on smaller (pilot) or larger candidate
pools without modification.

Outputs (under <out-root>):
    approved.csv   — rows for the training set; VLM confirmed image
                     matches (object, color)
    rejected.csv   — candidates the VLM disagreed with; useful for
                     debugging the collection regex
    summary.json   — acceptance rate per pair and overall
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from binding.io import load_yaml  
from binding.seeds import set_all_seeds  
from binding.vlm_judge import VLMJudge  


VERIFY_FIELDS = [
    "object", "color", "cand_idx", "path",
    "object_raw", "color_raw",
    "object_predicted", "color_predicted",
    "verdict",                    
    "rejection_reason",           
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--candidates-root", type=Path, required=True,
                   help="Output root from exp5_collect_laion.py (has candidates_manifest.csv).")
    p.add_argument("--out-root", type=Path, default=Path("data/finetuning/verified"),
                   help="Where approved.csv and rejected.csv will be written.")
    p.add_argument("--config", type=Path, default=Path("configs/judge_default.yaml"),
                   help="Judge config (reuses the same Qwen2.5-VL-7B model as Phase 2).")
    p.add_argument("--limit", type=int, default=None,
                   help="Verify only the first N candidates. For smoke testing.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_candidates(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Candidates manifest not found: {manifest_path}. "
            "Run experiments/exp5_collect_laion.py first."
        )
    with manifest_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def already_verified(out_path: Path) -> set[tuple[str, str, str]]:
    """Return the set of (object, color, cand_idx) tuples already in the file."""
    if not out_path.exists():
        return set()
    seen = set()
    with out_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seen.add((row["object"], row["color"], row["cand_idx"]))
    return seen


def rejection_reason(
    obj_predicted: str, color_predicted: str,
    expected_obj: str, expected_color: str,
) -> str:
    obj_ok = (obj_predicted == expected_obj)
    color_ok = (color_predicted == expected_color)
    if obj_ok and color_ok:
        return ""
    if not obj_ok and not color_ok:
        return "both"
    return "object" if not obj_ok else "color"


def main() -> int:
    args = parse_args()
    set_all_seeds(args.seed)

    manifest_path = args.candidates_root / "candidates_manifest.csv"
    candidates = load_candidates(manifest_path)
    if args.limit is not None:
        candidates = candidates[: args.limit]
    print(f"[verify] candidates manifest: {manifest_path}")
    print(f"[verify] {len(candidates)} candidates to consider")

    args.out_root.mkdir(parents=True, exist_ok=True)
    approved_path = args.out_root / "approved.csv"
    rejected_path = args.out_root / "rejected.csv"

    seen_approved = already_verified(approved_path)
    seen_rejected = already_verified(rejected_path)
    seen = seen_approved | seen_rejected
    to_verify = [
        c for c in candidates
        if (c["object"], c["color"], c["cand_idx"]) not in seen
    ]
    print(f"[verify] already verified (skipping): {len(seen)}")
    print(f"[verify] to verify now: {len(to_verify)}")
    if not to_verify:
        print("[verify] nothing to do.")
        return 0

    cfg = load_yaml(args.config)
    print(f"[verify] loading {cfg['judge']['model_id']} (this takes ~30-90s)…")
    judge = VLMJudge(
        model_id=cfg["judge"]["model_id"],
        dtype=cfg["judge"].get("dtype", "bfloat16"),
    )
    print(f"[verify] loaded on device: {judge.device}")

    is_new_a = not approved_path.exists()
    is_new_r = not rejected_path.exists()
    af = approved_path.open("a", newline="", encoding="utf-8")
    rf = rejected_path.open("a", newline="", encoding="utf-8")
    aw = csv.DictWriter(af, fieldnames=VERIFY_FIELDS)
    rw = csv.DictWriter(rf, fieldnames=VERIFY_FIELDS)
    if is_new_a:
        aw.writeheader(); af.flush()
    if is_new_r:
        rw.writeheader(); rf.flush()

    n_approved = 0
    n_rejected = 0
    n_failed = 0
    pbar = tqdm(to_verify, desc="verifying", unit="img")
    try:
        for cand in pbar:
            image_path = args.candidates_root / cand["path"]
            try:
                judgment = judge.judge_image(
                    image_path=image_path,
                    expected_object=cand["object"],
                    expected_color=cand["color"],
                )
            except Exception as e:
                n_failed += 1
                rw.writerow({
                    "object": cand["object"], "color": cand["color"],
                    "cand_idx": cand["cand_idx"], "path": cand["path"],
                    "object_raw": "", "color_raw": "",
                    "object_predicted": "", "color_predicted": "",
                    "verdict": "rejected",
                    "rejection_reason": f"image_error:{type(e).__name__}",
                })
                rf.flush()
                pbar.set_postfix(approved=n_approved, rejected=n_rejected, failed=n_failed)
                continue

            row = {
                "object": cand["object"], "color": cand["color"],
                "cand_idx": cand["cand_idx"], "path": cand["path"],
                "object_raw": judgment.object_raw,
                "color_raw":  judgment.color_raw,
                "object_predicted": judgment.object_predicted,
                "color_predicted":  judgment.color_predicted,
            }
            if judgment.binding_correct:
                row["verdict"] = "approved"
                row["rejection_reason"] = ""
                aw.writerow(row); af.flush()
                n_approved += 1
            else:
                row["verdict"] = "rejected"
                row["rejection_reason"] = rejection_reason(
                    judgment.object_predicted, judgment.color_predicted,
                    cand["object"], cand["color"],
                )
                rw.writerow(row); rf.flush()
                n_rejected += 1
            pbar.set_postfix(approved=n_approved, rejected=n_rejected, failed=n_failed)
    finally:
        pbar.close()
        af.close(); rf.close()

    total = n_approved + n_rejected + n_failed
    rate = n_approved / total if total else 0.0
    print(f"\n[verify] done. approved={n_approved}, rejected={n_rejected}, "
          f"image-errors={n_failed}")
    print(f"[verify] global acceptance rate: {rate:.1%}")

    pair_total: Counter = Counter()
    pair_approved: Counter = Counter()
    pair_reasons: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for path in (approved_path, rejected_path):
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["object"], row["color"])
                pair_total[key] += 1
                if row["verdict"] == "approved":
                    pair_approved[key] += 1
                elif row["rejection_reason"]:
                    pair_reasons[key][row["rejection_reason"]] += 1

    print(f"\n[verify] per-pair acceptance:")
    for key in sorted(pair_total):
        appr = pair_approved[key]
        tot = pair_total[key]
        reasons = ", ".join(f"{r}:{n}" for r, n in pair_reasons[key].most_common())
        flag = ""
        if tot >= 5 and appr / tot < 0.20:
            flag = "  ⚠️ very low"
        print(f"    {key[0]:<12} {key[1]:<8} {appr}/{tot}  "
              f"({appr/tot:.0%}){flag}    reasons: {reasons or '—'}")

    summary = {
        "n_approved": n_approved,
        "n_rejected": n_rejected,
        "n_image_errors": n_failed,
        "global_acceptance_rate": rate,
        "per_pair": {
            f"{o}__{c}": {
                "approved": pair_approved[(o, c)],
                "total": pair_total[(o, c)],
                "rate": pair_approved[(o, c)] / pair_total[(o, c)] if pair_total[(o, c)] else 0,
            }
            for (o, c) in pair_total
        },
    }
    with (args.out_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[verify] approved candidates: {approved_path}")
    print(f"[verify] rejected candidates: {rejected_path}")
    print(f"[verify] next: re-caption the approved set (exp5_recaption.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
