"""
Phase 7 extra analysis: drill into trade-offs and generalization patterns.

Reads results/exp7_compare/per_pair.csv (produced by exp7_compare.py) and
produces three deeper analyses motivated by the headline numbers:

  1. Control regression pattern — which control pairs lost most accuracy,
     and is there a relationship between the loss and the pair's prior
     strength (NPMI / baseline acc)? Tests whether LoRA disrupts the
     canonically-strongest bindings preferentially (classic catastrophic
     forgetting signature, cf. Ruiz et al. 2023, DreamBooth, on prior
     preservation).

  2. Treated regression pair — identifies the single treated pair that
     worsened despite receiving corrective training, with context.

  3. Per-color and per-object generalization — for each color and each
     object, compares mean Δ-accuracy across treated vs held-out pairs.
     Asks: does the model generalize binding correction uniformly across
     colors and objects, or are some categories easier to "fix"?

Outputs:
  results/exp7_extra/control_regression.csv
  results/exp7_extra/per_color_breakdown.csv
  results/exp7_extra/per_object_breakdown.csv
  results/exp7_extra/control_regression.png
  results/exp7_extra/per_color_delta.png
  results/exp7_extra/per_object_delta.png
  results/exp7_extra/summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--per-pair-csv", type=Path,
                   default=Path("results/exp7_compare/per_pair.csv"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/exp7_extra"))
    return p.parse_args()

def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["npmi"] = float(r["npmi"])
        r["acc_baseline"] = float(r["acc_baseline"])
        r["acc_lora"] = float(r["acc_lora"])
        r["delta"] = float(r["delta"])
    return rows

def spearman_corr(xs: list[float], ys: list[float]) -> tuple[float, int]:
    """Spearman ρ from scratch (no scipy dependency)."""
    n = len(xs)
    if n < 3:
        return float("nan"), n
    def rank(values):
        sorted_idx = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[sorted_idx[j + 1]] == values[sorted_idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg
            i = j + 1
        return ranks
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx)/n, sum(ry)/n
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    dx = (sum((rx[i]-mx)**2 for i in range(n)))**0.5
    dy = (sum((ry[i]-my)**2 for i in range(n)))**0.5
    if dx == 0 or dy == 0:
        return float("nan"), n
    return num/(dx*dy), n

def analysis_1_control_regression(rows: list[dict], out_dir: Path) -> dict:
    """How does control regression relate to prior strength?"""
    print("\n" + "="*70)
    print("ANALYSIS 1: Control regression detail")
    print("="*70)

    ctrl = [r for r in rows if r["group"] == "control"]
    ctrl_sorted = sorted(ctrl, key=lambda r: r["delta"])

    out_csv = out_dir / "control_regression.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["object", "color", "npmi", "acc_baseline", "acc_lora", "delta"])
        for r in ctrl_sorted:
            w.writerow([r["object"], r["color"], r["npmi"],
                        r["acc_baseline"], r["acc_lora"], r["delta"]])
    print(f"  wrote {out_csv}")
    print(f"\n  Worst 10 control regressions:")
    print(f"  {'object':<14} {'color':<8} {'npmi':>7} {'base':>7} {'lora':>7} {'delta':>8}")
    for r in ctrl_sorted[:10]:
        print(f"  {r['object']:<14} {r['color']:<8} {r['npmi']:>+7.3f} "
              f"{r['acc_baseline']:>7.1%} {r['acc_lora']:>7.1%} {r['delta']:>+8.1%}")

    n_pos = sum(1 for r in ctrl if r["delta"] > 0)
    n_neg = sum(1 for r in ctrl if r["delta"] < 0)
    n_zero = sum(1 for r in ctrl if r["delta"] == 0)
    print(f"\n  Distribution: {n_pos} improved, {n_zero} unchanged, {n_neg} regressed")

    deltas = [r["delta"] for r in ctrl]
    npmis = [r["npmi"] for r in ctrl]
    base_accs = [r["acc_baseline"] for r in ctrl]

    rho_npmi, _ = spearman_corr(npmis, deltas)
    rho_base, _ = spearman_corr(base_accs, deltas)
    print(f"\n  Spearman ρ within control:")
    print(f"    Δ vs NPMI:         {rho_npmi:+.3f}  (negative = stronger NPMI lost more)")
    print(f"    Δ vs baseline acc: {rho_base:+.3f}  (negative = canonically perfect lost more)")
    if rho_base < -0.3:
        print(f"    → Evidence that LoRA preferentially disrupts canonically-strongest bindings")
    elif rho_base > 0.3:
        print(f"    → Surprisingly, weaker baselines regressed more (uniform forgetting absent)")
    else:
        print(f"    → No clear monotonic relationship; regression is broad-spectrum")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        xs = base_accs
        ys = deltas
        sc = ax.scatter(xs, ys, c=npmis, cmap="RdYlBu_r", s=70, edgecolor="black", linewidth=0.5)
        for r in ctrl_sorted[:5]:
            ax.annotate(f"{r['object']}×{r['color']}",
                        (r["acc_baseline"], r["delta"]),
                        textcoords="offset points", xytext=(8, -3),
                        fontsize=7, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Baseline accuracy (canonical pair)")
        ax.set_ylabel("Δ accuracy (LoRA − baseline)")
        ax.set_title(f"Control pairs: regression vs baseline strength  (n={len(ctrl)})")
        cbar = plt.colorbar(sc, ax=ax, label="NPMI in LAION-400M")
        ax.grid(alpha=0.3)
        plot_path = out_dir / "control_regression.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"  wrote {plot_path}")
    except ImportError:
        pass

    return {
        "n_pairs": len(ctrl),
        "n_improved": n_pos,
        "n_unchanged": n_zero,
        "n_regressed": n_neg,
        "spearman_delta_vs_npmi": rho_npmi,
        "spearman_delta_vs_baseline": rho_base,
        "worst_5": [
            {"object": r["object"], "color": r["color"], "delta": r["delta"],
             "acc_baseline": r["acc_baseline"], "acc_lora": r["acc_lora"]}
            for r in ctrl_sorted[:5]
        ],
    }

def analysis_2_treated_regression(rows: list[dict]) -> dict:
    """Identify the treated pair(s) that worsened."""
    print("\n" + "="*70)
    print("ANALYSIS 2: Treated pairs that regressed (received corrective training but worsened)")
    print("="*70)

    treated = [r for r in rows if r["group"] == "treated"]
    worsened = sorted([r for r in treated if r["delta"] < 0], key=lambda r: r["delta"])
    unchanged = [r for r in treated if r["delta"] == 0]

    print(f"  Total treated: {len(treated)}")
    print(f"  Worsened: {len(worsened)}, unchanged: {len(unchanged)}")

    if worsened:
        print(f"\n  Worsened pairs:")
        print(f"  {'object':<14} {'color':<8} {'npmi':>7} {'base':>7} {'lora':>7} {'delta':>8}")
        for r in worsened:
            print(f"  {r['object']:<14} {r['color']:<8} {r['npmi']:>+7.3f} "
                  f"{r['acc_baseline']:>7.1%} {r['acc_lora']:>7.1%} {r['delta']:>+8.1%}")
    if unchanged:
        print(f"\n  Unchanged pairs:")
        for r in unchanged:
            print(f"    {r['object']}×{r['color']}: baseline=lora={r['acc_baseline']:.1%}")

    return {
        "n_worsened": len(worsened),
        "n_unchanged": len(unchanged),
        "worsened": [
            {"object": r["object"], "color": r["color"],
             "delta": r["delta"], "npmi": r["npmi"],
             "acc_baseline": r["acc_baseline"], "acc_lora": r["acc_lora"]}
            for r in worsened
        ],
        "unchanged": [
            {"object": r["object"], "color": r["color"],
             "acc_baseline": r["acc_baseline"]}
            for r in unchanged
        ],
    }

def analysis_3_per_color_object(rows: list[dict], out_dir: Path) -> dict:
    """Mean Δ per color and per object, broken down by group (treated vs held_out)."""
    print("\n" + "="*70)
    print("ANALYSIS 3: Per-color and per-object generalization patterns")
    print("="*70)

    color_buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    object_buckets: dict[tuple[str, str], list[float]] = defaultdict(list)

    for r in rows:
        if r["group"] in ("treated", "held_out"):
            color_buckets[(r["color"], r["group"])].append(r["delta"])
            object_buckets[(r["object"], r["group"])].append(r["delta"])

    all_colors = sorted({r["color"] for r in rows
                        if r["group"] in ("treated", "held_out")})
    print(f"\n  Per-color mean Δ-accuracy (treated vs held-out):")
    print(f"  {'color':<10} {'treated':>20}    {'held_out':>20}")
    print(f"  {'':<10} {'n':>4} {'mean Δ':>10} {'min/max':>14}    {'n':>4} {'mean Δ':>10} {'min/max':>14}")
    color_rows = []
    for c in all_colors:
        t = color_buckets.get((c, "treated"), [])
        h = color_buckets.get((c, "held_out"), [])
        t_mean = sum(t)/len(t) if t else float("nan")
        h_mean = sum(h)/len(h) if h else float("nan")
        t_range = f"{min(t):+.0%}/{max(t):+.0%}" if t else "-"
        h_range = f"{min(h):+.0%}/{max(h):+.0%}" if h else "-"
        print(f"  {c:<10} {len(t):>4} {t_mean:>+10.1%} {t_range:>14}    "
              f"{len(h):>4} {h_mean:>+10.1%} {h_range:>14}")
        color_rows.append({
            "color": c,
            "n_treated": len(t),
            "delta_treated_mean": t_mean,
            "n_held_out": len(h),
            "delta_held_out_mean": h_mean,
        })

    out_csv = out_dir / "per_color_breakdown.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["color","n_treated","delta_treated_mean",
                                          "n_held_out","delta_held_out_mean"])
        w.writeheader()
        for r in color_rows:
            w.writerow(r)
    print(f"\n  wrote {out_csv}")

    all_objects = sorted({r["object"] for r in rows
                         if r["group"] in ("treated", "held_out")})
    print(f"\n  Per-object mean Δ-accuracy (treated vs held-out):")
    print(f"  {'object':<14} {'treated':>20}    {'held_out':>20}")
    print(f"  {'':<14} {'n':>4} {'mean Δ':>10} {'min/max':>14}    {'n':>4} {'mean Δ':>10} {'min/max':>14}")
    object_rows = []
    for o in all_objects:
        t = object_buckets.get((o, "treated"), [])
        h = object_buckets.get((o, "held_out"), [])
        t_mean = sum(t)/len(t) if t else float("nan")
        h_mean = sum(h)/len(h) if h else float("nan")
        t_range = f"{min(t):+.0%}/{max(t):+.0%}" if t else "-"
        h_range = f"{min(h):+.0%}/{max(h):+.0%}" if h else "-"
        print(f"  {o:<14} {len(t):>4} {t_mean:>+10.1%} {t_range:>14}    "
              f"{len(h):>4} {h_mean:>+10.1%} {h_range:>14}")
        object_rows.append({
            "object": o,
            "n_treated": len(t),
            "delta_treated_mean": t_mean,
            "n_held_out": len(h),
            "delta_held_out_mean": h_mean,
        })

    out_csv = out_dir / "per_object_breakdown.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["object","n_treated","delta_treated_mean",
                                          "n_held_out","delta_held_out_mean"])
        w.writeheader()
        for r in object_rows:
            w.writerow(r)
    print(f"  wrote {out_csv}")

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(all_colors))
        w = 0.4
        t_means = [next((r["delta_treated_mean"] for r in color_rows if r["color"] == c), 0)
                   for c in all_colors]
        h_means = [next((r["delta_held_out_mean"] for r in color_rows if r["color"] == c), 0)
                   for c in all_colors]
        t_means_plot = [0 if (m != m) else m for m in t_means]
        h_means_plot = [0 if (m != m) else m for m in h_means]
        ax.bar(x - w/2, t_means_plot, w, label="treated", color="#d62728", alpha=0.85)
        ax.bar(x + w/2, h_means_plot, w, label="held_out", color="#ff7f0e", alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(all_colors, rotation=30)
        ax.set_ylabel("Mean Δ accuracy")
        ax.set_title("Per-color generalization: treated vs held-out")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plot_path = out_dir / "per_color_delta.png"
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"  wrote {plot_path}")

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(all_objects))
        t_means = [next((r["delta_treated_mean"] for r in object_rows if r["object"] == o), 0)
                   for o in all_objects]
        h_means = [next((r["delta_held_out_mean"] for r in object_rows if r["object"] == o), 0)
                   for o in all_objects]
        t_means_plot = [0 if (m != m) else m for m in t_means]
        h_means_plot = [0 if (m != m) else m for m in h_means]
        ax.bar(x - w/2, t_means_plot, w, label="treated", color="#d62728", alpha=0.85)
        ax.bar(x + w/2, h_means_plot, w, label="held_out", color="#ff7f0e", alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(all_objects, rotation=30)
        ax.set_ylabel("Mean Δ accuracy")
        ax.set_title("Per-object generalization: treated vs held-out")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plot_path = out_dir / "per_object_delta.png"
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"  wrote {plot_path}")
    except ImportError:
        pass

    return {
        "per_color": color_rows,
        "per_object": object_rows,
    }

def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp7-extra] loading {args.per_pair_csv}")
    rows = load_rows(args.per_pair_csv)
    print(f"[exp7-extra] {len(rows)} pairs loaded")

    summary = {
        "control_regression": analysis_1_control_regression(rows, args.out_dir),
        "treated_regression": analysis_2_treated_regression(rows),
        "per_color_object":   analysis_3_per_color_object(rows, args.out_dir),
    }

    summary_path = args.out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[exp7-extra] summary: {summary_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
