"""
Phase 5, Step 1: assign pairs to finetuning groups (treated / held-out / control).

Reads the Phase-3 joined results (per-pair accuracy + NPMI) and produces
a split that supports a clean generalization test:

  Group 1 (treated)   : low-accuracy pairs that WILL be finetuned.
  Group 2 (held-out)  : low-accuracy pairs of matched difficulty that will
                        NOT be finetuned — the generalization probe.
  Group 3 (control)   : high-accuracy pairs, not finetuned, to detect
                        catastrophic forgetting after finetuning.

Stratification (the defensible design):
  Pairs below --hard-threshold are "hard". Within EACH object, hard pairs
  are ranked by accuracy and assigned alternately to treated/held-out
  (hardest→treated, 2nd→held-out, 3rd→treated, ...). This guarantees that
  for an object like 'chalkboard', some non-canonical colors are trained
  and OTHERS of comparable difficulty are held out — so improvement on the
  held-out colors measures genuine within-object generalization, not
  memorization of specific pairs.

  Objects with only ONE hard pair cannot be split; by default that pair
  goes to treated (configurable via --orphan-policy) and is flagged as
  having no held-out counterpart.

Image-source suggestion (for Phase 5 Step 2):
  never  (NPMI == -1)  → external collection / synthetic render
  under  (NPMI  <  0)  → LAION re-captioning (real image, rewritten caption)
  positive             → not applicable (these are control pairs)

This script does NOT fetch or build images — it only produces the plan.
The output finetuning_split.csv drives the data-construction step.
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

from binding.seeds import set_all_seeds  

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--joined", type=Path, required=True,
                   help="results/exp3_analysis/joined_per_pair.csv")
    p.add_argument("--out", type=Path, default=Path("results/exp5_split/finetuning_split.csv"))
    p.add_argument("--hard-threshold", type=float, default=0.5,
                   help="Pairs with accuracy <= this go to treated/held-out. Default 0.5.")
    p.add_argument("--control-threshold", type=float, default=0.9,
                   help="Pairs with accuracy >= this are eligible for control. Default 0.9.")
    p.add_argument("--max-control", type=int, default=20,
                   help="Max control pairs to sample. Default 20.")
    p.add_argument("--orphan-policy", choices=["treated", "exclude"], default="treated",
                   help="What to do with objects that have only one hard pair. "
                        "'treated' (default) puts it in Group 1 with no held-out "
                        "counterpart; 'exclude' drops it.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_joined(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["accuracy"] = float(r["accuracy"])
            r["npmi"] = float(r["npmi"])
            rows.append(r)
    return rows


def suggest_source(category: str) -> str:
    if category == "never":
        return "external_or_synthetic"
    if category == "under":
        return "laion_recaption"
    return "n/a"


def main() -> int:
    args = parse_args()
    set_all_seeds(args.seed)

    rows = load_joined(args.joined)
    print(f"[split] loaded {len(rows)} pairs")

    hard = [r for r in rows if r["accuracy"] <= args.hard_threshold]
    easy = [r for r in rows if r["accuracy"] >= args.control_threshold]
    print(f"[split] hard pairs (acc <= {args.hard_threshold:.0%}): {len(hard)}")
    print(f"[split] easy pairs (acc >= {args.control_threshold:.0%}): {len(easy)}")

    hard_by_obj: dict[str, list[dict]] = defaultdict(list)
    for r in hard:
        hard_by_obj[r["object"]].append(r)

    assignments: dict[tuple[str, str], str] = {}   
    orphan_flags: dict[tuple[str, str], bool] = {}

    for obj, pairs in hard_by_obj.items():
        pairs_sorted = sorted(pairs, key=lambda r: (r["accuracy"], r["color"]))
        if len(pairs_sorted) == 1:
            only = pairs_sorted[0]
            key = (only["object"], only["color"])
            if args.orphan_policy == "treated":
                assignments[key] = "treated"
                orphan_flags[key] = True
        else:
            for i, r in enumerate(pairs_sorted):
                key = (r["object"], r["color"])
                assignments[key] = "treated" if i % 2 == 0 else "held_out"
                orphan_flags[key] = False

    treated_objects = {o for (o, _c), g in assignments.items() if g == "treated"}
    easy_sorted = sorted(easy, key=lambda r: (r["object"] not in treated_objects,
                                              -r["accuracy"], r["object"], r["color"]))
    control = easy_sorted[: args.max_control]
    for r in control:
        assignments[(r["object"], r["color"])] = "control"
        orphan_flags.setdefault((r["object"], r["color"]), False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    by_acc = {(r["object"], r["color"]): r for r in rows}
    n_by_group: dict[str, int] = defaultdict(int)

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "object", "color", "group", "base_accuracy", "npmi", "category",
            "image_source", "is_orphan",
        ])
        for (obj, color), group in sorted(assignments.items()):
            meta = by_acc[(obj, color)]
            writer.writerow([
                obj, color, group,
                f"{meta['accuracy']:.4f}", f"{meta['npmi']:.4f}", meta["category"],
                suggest_source(meta["category"]) if group != "control" else "n/a",
                int(orphan_flags.get((obj, color), False)),
            ])
            n_by_group[group] += 1

    print(f"\n[split] wrote {args.out}")
    print(f"[split] group sizes:")
    for g in ("treated", "held_out", "control"):
        print(f"    {g:<10} {n_by_group[g]}")

    treated_accs = [by_acc[k]["accuracy"] for k, g in assignments.items() if g == "treated"]
    held_accs    = [by_acc[k]["accuracy"] for k, g in assignments.items() if g == "held_out"]
    if treated_accs and held_accs:
        mt = sum(treated_accs) / len(treated_accs)
        mh = sum(held_accs) / len(held_accs)
        print(f"\n[split] difficulty parity check (mean base accuracy):")
        print(f"    treated:  {mt:.1%}")
        print(f"    held_out: {mh:.1%}")
        print(f"    gap:      {abs(mt - mh):.1%}  (smaller is better; alternation keeps this low)")

    orphans = [k for k, v in orphan_flags.items() if v and assignments.get(k) == "treated"]
    if orphans:
        print(f"\n[split] {len(orphans)} orphan pair(s) (treated, no held-out counterpart):")
        for o, c in orphans:
            print(f"    {o} × {c}")
        print("    → note in paper: these have no within-object generalization probe.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
