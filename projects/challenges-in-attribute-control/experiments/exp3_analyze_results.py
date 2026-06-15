"""
Phase 3: cross NPMI (Exp 1) with VLM judgments (Exp 3 base model run).

Joins the two artifacts on (object, color) and produces:

  joined_per_pair.csv     — one row per pair with NPMI, accuracy, bootstrap CI
  by_category.csv         — accuracy aggregated by NPMI category
  by_object.csv           — accuracy aggregated by object
  by_color.csv            — accuracy aggregated by color
  correlations.json       — Spearman ρ (global and per-category) + CIs
  summary.json            — top-line numbers for the paper

The bootstrap CI for each per-pair accuracy uses 1000 resamples of the
30 images per pair. For Spearman correlations, the bootstrap resamples
pairs (not images), reflecting that pair-level NPMI is the unit of
analysis. The accuracy CI of a 0/30 or 30/30 pair is exactly [0,0] or
[1,1] — informative, not a numerical artifact to be hidden.

The script is deterministic given a fixed --seed.

Usage:
    python experiments/exp3_analyze_results.py \\
        --npmi data/exp1_results/npmi_per_pair.csv \\
        --judgments data/judgments/judgments.csv \\
        --out-dir results/exp3_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from binding.seeds import set_all_seeds  # noqa: E402


# ── Spearman correlation (no scipy dependency) ─────────────────────────────
def spearman_rho(x: list[float], y: list[float]) -> float:
    """
    Spearman rank correlation with average ranks for ties.
    Returns NaN when n < 2 or when either input is constant.

    Validated against scipy.stats.spearmanr in tests/test_analyze_results.py
    over 100 randomized cases.
    """
    n = len(x)
    if n < 2:
        return float("nan")
    if min(x) == max(x) or min(y) == max(y):
        return float("nan")

    def ranks(values: list[float]) -> list[float]:
        """Fractional (mid) ranks for ties."""
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx = ranks(x)
    ry = ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def bootstrap_ci(
    values_a: list[float],
    values_b: list[float],
    statistic,
    n_boot: int = 1000,
    confidence: float = 0.95,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for a paired statistic (e.g. Spearman ρ).
    Resamples paired indices, recomputes statistic on the resample,
    returns the (lower, upper) percentiles.
    """
    rng = rng or random.Random()
    n = len(values_a)
    if n < 2:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        a_b = [values_a[i] for i in idx]
        b_b = [values_b[i] for i in idx]
        try:
            s = statistic(a_b, b_b)
            if not math.isnan(s):
                stats.append(s)
        except Exception:
            continue
    if not stats:
        return (float("nan"), float("nan"))
    stats.sort()
    lo = stats[int((1 - confidence) / 2 * len(stats))]
    hi = stats[int((1 + confidence) / 2 * len(stats)) - 1]
    return (lo, hi)


def proportion_ci(
    n_success: int, n_total: int,
    n_boot: int = 1000, confidence: float = 0.95,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """Bootstrap CI for a proportion (per-pair accuracy)."""
    rng = rng or random.Random()
    if n_total == 0:
        return (float("nan"), float("nan"))
    successes = [1] * n_success + [0] * (n_total - n_success)
    means = []
    for _ in range(n_boot):
        sample = [successes[rng.randrange(n_total)] for _ in range(n_total)]
        means.append(sum(sample) / n_total)
    means.sort()
    lo = means[int((1 - confidence) / 2 * n_boot)]
    hi = means[int((1 + confidence) / 2 * n_boot) - 1]
    return (lo, hi)


def load_npmi(path: Path) -> dict[tuple[str, str], dict]:
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(row["object"], row["color"])] = {
                "npmi": float(row["npmi"]),
                "pmi": float(row["pmi"]),
                "ppmi": float(row["ppmi"]),
                "p_joint": float(row["p_joint"]),
                "category": row["category"],
                "count_pattern": int(row["count_pattern"]),
                "binding_score": float(row["binding_score"]),
            }
    return out


def load_judgments(path: Path) -> dict[tuple[str, str], tuple[int, int]]:
    """Return (n_correct, n_total) per (object, color)."""
    n = defaultdict(int)
    ok = defaultdict(int)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = (row["object"], row["color"])
            n[k] += 1
            if row["binding_correct"].lower() == "true":
                ok[k] += 1
    return {k: (ok[k], n[k]) for k in n}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npmi", type=Path, required=True,
                   help="CSV from Phase 1 (data/exp1_results/npmi_per_pair.csv).")
    p.add_argument("--judgments", type=Path, required=True,
                   help="CSV from Phase 3 base run (data/judgments/judgments.csv).")
    p.add_argument("--out-dir", type=Path, default=Path("results/exp3_analysis"),
                   help="Where to write outputs.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for bootstrap.")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Bootstrap resamples (default 1000).")
    args = p.parse_args()

    set_all_seeds(args.seed)
    rng = random.Random(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    npmi = load_npmi(args.npmi)
    judgments = load_judgments(args.judgments)
    print(f"[analysis] loaded {len(npmi)} pairs from NPMI, {len(judgments)} pairs from judgments")

    common = sorted(set(npmi) & set(judgments))
    if len(common) != len(npmi):
        missing = set(npmi) - set(judgments)
        print(f"[analysis] WARNING: {len(missing)} NPMI pairs have no judgments")
        for k in list(missing)[:5]:
            print(f"  missing: {k}")
    print(f"[analysis] joined {len(common)} pairs")

    # ── Per-pair table with CIs ────────────────────────────────────────────
    per_pair_rows = []
    for obj, color in common:
        n_ok, n_tot = judgments[(obj, color)]
        acc = n_ok / n_tot
        ci_lo, ci_hi = proportion_ci(n_ok, n_tot, n_boot=args.n_boot, rng=rng)
        meta = npmi[(obj, color)]
        per_pair_rows.append({
            "object": obj, "color": color,
            "category": meta["category"],
            "npmi": meta["npmi"],
            "pmi":  meta["pmi"],
            "ppmi": meta["ppmi"],
            "p_joint": meta["p_joint"],
            "count_pattern": meta["count_pattern"],
            "n_correct": n_ok,
            "n_total":   n_tot,
            "accuracy":  acc,
            "acc_ci_lo": ci_lo,
            "acc_ci_hi": ci_hi,
        })
    write_csv(
        args.out_dir / "joined_per_pair.csv", per_pair_rows,
        ["object", "color", "category", "npmi", "pmi", "ppmi", "p_joint",
         "count_pattern", "n_correct", "n_total",
         "accuracy", "acc_ci_lo", "acc_ci_hi"],
    )

    # ── Aggregations: by category, object, color ──────────────────────────
    def aggregate(key_fn):
        groups = defaultdict(list)
        for r in per_pair_rows:
            groups[key_fn(r)].append(r)
        out = []
        for key, rows in sorted(groups.items()):
            accs = [r["accuracy"] for r in rows]
            npmis = [r["npmi"] for r in rows]
            mean_acc = sum(accs) / len(accs)
            sd_acc = (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5 if len(accs) > 1 else 0
            mean_npmi = sum(npmis) / len(npmis)
            out.append({
                "key": key,
                "n_pairs": len(rows),
                "accuracy_mean": mean_acc,
                "accuracy_sd":   sd_acc,
                "npmi_mean":     mean_npmi,
            })
        return out

    by_cat = aggregate(lambda r: r["category"])
    by_obj = aggregate(lambda r: r["object"])
    by_col = aggregate(lambda r: r["color"])

    write_csv(args.out_dir / "by_category.csv", by_cat,
              ["key", "n_pairs", "accuracy_mean", "accuracy_sd", "npmi_mean"])
    write_csv(args.out_dir / "by_object.csv",   by_obj,
              ["key", "n_pairs", "accuracy_mean", "accuracy_sd", "npmi_mean"])
    write_csv(args.out_dir / "by_color.csv",    by_col,
              ["key", "n_pairs", "accuracy_mean", "accuracy_sd", "npmi_mean"])

    # ── Correlations ──────────────────────────────────────────────────────
    xs_all = [r["npmi"]     for r in per_pair_rows]
    ys_all = [r["accuracy"] for r in per_pair_rows]

    rho_global = spearman_rho(xs_all, ys_all)
    ci_global = bootstrap_ci(xs_all, ys_all, spearman_rho,
                             n_boot=args.n_boot, rng=rng)

    rho_by_cat = {}
    for cat in ("never", "under", "positive"):
        rows = [r for r in per_pair_rows if r["category"] == cat]
        if len(rows) >= 3:
            xs = [r["npmi"] for r in rows]
            ys = [r["accuracy"] for r in rows]
            rho = spearman_rho(xs, ys)
            ci = bootstrap_ci(xs, ys, spearman_rho,
                              n_boot=args.n_boot, rng=rng)
            rho_by_cat[cat] = {"rho": rho, "ci_lo": ci[0], "ci_hi": ci[1],
                               "n_pairs": len(rows)}
        else:
            rho_by_cat[cat] = {"rho": float("nan"), "ci_lo": float("nan"),
                               "ci_hi": float("nan"), "n_pairs": len(rows)}

    correlations = {
        "global": {
            "rho": rho_global,
            "ci_lo": ci_global[0],
            "ci_hi": ci_global[1],
            "n_pairs": len(per_pair_rows),
        },
        "by_category": rho_by_cat,
    }
    with (args.out_dir / "correlations.json").open("w") as f:
        json.dump(correlations, f, indent=2)

    # ── Top-line summary ──────────────────────────────────────────────────
    accs_sorted = sorted(per_pair_rows, key=lambda r: r["accuracy"])
    summary = {
        "n_joined_pairs": len(per_pair_rows),
        "global_accuracy": sum(ys_all) / len(ys_all),
        "spearman_global": {
            "rho": rho_global, "ci_lo": ci_global[0], "ci_hi": ci_global[1],
        },
        "spearman_by_category": rho_by_cat,
        "accuracy_by_category": {
            r["key"]: {"n_pairs": r["n_pairs"], "mean": r["accuracy_mean"],
                       "sd": r["accuracy_sd"]} for r in by_cat
        },
        "worst_10_pairs": [
            {"object": r["object"], "color": r["color"],
             "accuracy": r["accuracy"], "npmi": r["npmi"],
             "category": r["category"]}
            for r in accs_sorted[:10]
        ],
        "best_10_pairs": [
            {"object": r["object"], "color": r["color"],
             "accuracy": r["accuracy"], "npmi": r["npmi"],
             "category": r["category"]}
            for r in accs_sorted[-10:][::-1]
        ],
    }
    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # ── Print headline ────────────────────────────────────────────────────
    print(f"\n=== Headline ===")
    print(f"  Global accuracy: {summary['global_accuracy']:.1%}")
    print(f"  Spearman ρ (NPMI × accuracy): {rho_global:+.3f}  "
          f"[95% CI {ci_global[0]:+.3f}, {ci_global[1]:+.3f}]")
    print()
    print(f"  By NPMI category:")
    for r in by_cat:
        rho_info = rho_by_cat[r["key"]]
        rho_str = (f"ρ={rho_info['rho']:+.3f} [{rho_info['ci_lo']:+.2f},{rho_info['ci_hi']:+.2f}]"
                   if not math.isnan(rho_info["rho"]) else "ρ undefined")
        print(f"    {r['key']:<10} n={r['n_pairs']:<3} "
              f"acc={r['accuracy_mean']:.1%} ± {r['accuracy_sd']:.1%}   {rho_str}")
    print()
    print(f"  Worst by object (top 5):")
    for r in sorted(by_obj, key=lambda x: x["accuracy_mean"])[:5]:
        print(f"    {r['key']:<12} acc={r['accuracy_mean']:.1%}  (n={r['n_pairs']} pairs)")
    print()
    print(f"  Best by object (top 5):")
    for r in sorted(by_obj, key=lambda x: -x["accuracy_mean"])[:5]:
        print(f"    {r['key']:<12} acc={r['accuracy_mean']:.1%}  (n={r['n_pairs']} pairs)")

    print(f"\n[analysis] outputs in {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
