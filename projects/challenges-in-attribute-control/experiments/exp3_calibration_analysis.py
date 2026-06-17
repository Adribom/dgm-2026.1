"""
Phase 2, Step 4: Analyze human + VLM calibration.

Computes:
  - Cohen's kappa between the two human annotators (object, color, binding)
  - Agreement statistics (raw and chance-corrected)
  - VLM accuracy against each human, against the consensus
  - Confusion matrix of color predictions (per axis)
  - List of divergences (rows where VLM disagrees with both humans)

The script writes a compact `summary.json` with all top-line numbers
(for inclusion in the paper) and three CSVs (consensus_table.csv,
color_confusion.csv, divergences.csv) for further inspection.

Why kappa and not just accuracy:
  Raw agreement between annotators is misleading because some agreement
  happens by chance. Cohen's kappa corrects for this: kappa=0 means
  agreement is no better than random; kappa=1 is perfect. Standard
  thresholds (Landis & Koch 1977):
      0.0–0.20 slight | 0.21–0.40 fair | 0.41–0.60 moderate
      0.61–0.80 substantial | 0.81–1.00 almost perfect
  For our binary task (correct/wrong), expect 0.7+ for a well-defined
  task. Lower kappa is a methodological signal, not a failure to hide.

VLM "consensus accuracy" methodology:
  We compute VLM accuracy only on cases where the two humans agree
  (the "consensus" set). This is the standard approach in MJ-Bench and
  similar VLM-as-judge calibration studies: cases where humans disagree
  represent genuine task ambiguity and shouldn't penalize either party.
  We also report per-annotator accuracy for diagnostic transparency.

Usage:
    python experiments/exp3_calibration_analysis.py \\
        --calibration-dir data/calibration \\
        --vlm data/judgments_calibration/judgments.csv \\
        --out-dir results/calibration_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

def row_key(row: dict) -> tuple[str, str, str]:
    return (row["object"], row["color"], str(row["seed"]))

def cohens_kappa(a: list[int], b: list[int]) -> tuple[float, float]:
    """
    Compute Cohen's kappa for two paired binary annotator sequences.

    Returns (kappa, raw_agreement).

    Implementation note:
      For binary 0/1, this is the same as the general-case formula but
      cheaper to compute directly. We confirm via the general case in tests.

    Math:
      P_o = observed agreement = (n_00 + n_11) / N
      P_e = expected agreement by chance
          = (a_zero_rate * b_zero_rate) + (a_one_rate * b_one_rate)
      kappa = (P_o - P_e) / (1 - P_e)
    """
    assert len(a) == len(b), "annotator sequences must have equal length"
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")

    agree = sum(1 for x, y in zip(a, b) if x == y)
    p_o = agree / n

    p_a1 = sum(a) / n
    p_b1 = sum(b) / n
    p_e = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)

    if p_e == 1.0:
        return (1.0 if p_o == 1.0 else 0.0), p_o
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa, p_o

def load_annotations(calibration_dir: Path) -> dict[str, dict]:
    """
    Find all annotations_*.csv in calibration_dir and load them.

    Returns a dict: {annotator_name: {row_key: row_dict}}.

    We index by row_key so the order of rows in each annotator's file
    doesn't matter — they can label in any order and still align.
    """
    files = sorted(calibration_dir.glob("annotations_*.csv"))
    if len(files) < 2:
        raise FileNotFoundError(
            f"Need at least 2 annotation files in {calibration_dir}, found {len(files)}."
        )

    out: dict[str, dict] = {}
    for f in files:
        name = f.stem.replace("annotations_", "")
        with f.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        indexed = {row_key(r): r for r in rows}
        out[name] = indexed
        print(f"[analysis] loaded {len(indexed)} annotations from {name} ({f.name})")
    return out


def load_vlm_judgments(vlm_csv: Path) -> dict[tuple, dict]:
    with vlm_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    indexed = {row_key(r): r for r in rows}
    print(f"[analysis] loaded {len(indexed)} VLM judgments")
    return indexed

def to_int(v: str | int | bool) -> int:
    """'1' / '0' / 'True' / 'False' / 1 / 0 → 0 or 1."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return int(bool(v))
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "t"):
        return 1
    if s in ("0", "false", "no", "n", "f", ""):
        return 0
    raise ValueError(f"Cannot coerce to int: {v!r}")

def humans_kappa_block(humans: dict[str, dict]) -> dict:
    """
    Compute kappa between the two human annotators on each axis:
    object_correct, color_correct, and binding (both correct).
    """
    names = sorted(humans.keys())
    assert len(names) == 2, f"Expected exactly 2 annotators, got {names}"
    a_name, b_name = names
    a, b = humans[a_name], humans[b_name]

    common = sorted(set(a.keys()) & set(b.keys()))
    print(f"[analysis] {len(common)} rows in common between {a_name} and {b_name}")
    if not common:
        raise ValueError("No overlap between annotators. Did they label the same manifest?")

    obj_a = [to_int(a[k]["object_correct"]) for k in common]
    obj_b = [to_int(b[k]["object_correct"]) for k in common]
    col_a = [to_int(a[k]["color_correct"])  for k in common]
    col_b = [to_int(b[k]["color_correct"])  for k in common]
    bind_a = [int(o == 1 and c == 1) for o, c in zip(obj_a, col_a)]
    bind_b = [int(o == 1 and c == 1) for o, c in zip(obj_b, col_b)]

    k_obj,  agr_obj  = cohens_kappa(obj_a, obj_b)
    k_col,  agr_col  = cohens_kappa(col_a, col_b)
    k_bind, agr_bind = cohens_kappa(bind_a, bind_b)

    return {
        "annotator_a": a_name,
        "annotator_b": b_name,
        "n_common": len(common),
        "object":  {"kappa": k_obj,  "raw_agreement": agr_obj},
        "color":   {"kappa": k_col,  "raw_agreement": agr_col},
        "binding": {"kappa": k_bind, "raw_agreement": agr_bind},
    }


def vlm_vs_humans(
    humans: dict[str, dict],
    vlm: dict[tuple, dict],
) -> dict:
    """
    Compute VLM accuracy against each individual human and against
    consensus (cases where both humans agree).

    "Binding correct" from the VLM side is the `binding_correct` column
    in judgments.csv. From the human side it's (object_correct AND
    color_correct).
    """
    names = sorted(humans.keys())
    a_name, b_name = names
    a, b = humans[a_name], humans[b_name]

    keys = sorted(set(a) & set(b) & set(vlm))
    n = len(keys)
    if n == 0:
        raise ValueError("No rows shared by both humans and the VLM.")

    correct_vs_a = 0
    correct_vs_b = 0
    consensus_correct = 0
    consensus_total = 0
    consensus_rows = []  

    for k in keys:
        h_a_bind = int(to_int(a[k]["object_correct"]) and to_int(a[k]["color_correct"]))
        h_b_bind = int(to_int(b[k]["object_correct"]) and to_int(b[k]["color_correct"]))
        v_bind   = to_int(vlm[k]["binding_correct"])

        if v_bind == h_a_bind:
            correct_vs_a += 1
        if v_bind == h_b_bind:
            correct_vs_b += 1
        if h_a_bind == h_b_bind:
            consensus_total += 1
            if v_bind == h_a_bind:
                consensus_correct += 1
            consensus_rows.append({
                "object": a[k]["object"],
                "color":  a[k]["color"],
                "seed":   a[k]["seed"],
                "path":   a[k].get("path", ""),
                "human_consensus_binding": h_a_bind,
                "vlm_binding": v_bind,
                "vlm_agrees_with_consensus": int(v_bind == h_a_bind),
            })

    return {
        "n_joined": n,
        "vs_annotator_a": {
            "annotator": a_name,
            "accuracy": correct_vs_a / n,
            "correct": correct_vs_a,
            "total": n,
        },
        "vs_annotator_b": {
            "annotator": b_name,
            "accuracy": correct_vs_b / n,
            "correct": correct_vs_b,
            "total": n,
        },
        "consensus": {
            "n_consensus": consensus_total,
            "n_disagreement": n - consensus_total,
            "accuracy": consensus_correct / consensus_total if consensus_total else float("nan"),
            "correct": consensus_correct,
        },
        "consensus_rows": consensus_rows,
    }


def color_confusion(
    humans: dict[str, dict],
    vlm: dict[tuple, dict],
    canonical_colors: list[str],
) -> dict:
    """
    Build a confusion matrix of VLM color predictions, conditioned on
    human consensus.

    Rows = human consensus color (only when humans agree the color is
    correct, i.e. the prompted color is what they see).
    Cols = what the VLM predicted as the color.

    The point: when humans agree that 'pink' is correctly rendered,
    what does the VLM call it? If the VLM systematically says 'red'
    for 'pink', we see a hot off-diagonal cell — a calibration concern.
    """
    names = sorted(humans.keys())
    a, b = humans[names[0]], humans[names[1]]

    matrix: dict[str, Counter] = {c: Counter() for c in canonical_colors}
    matrix["unmatched"] = Counter()

    for k, vlm_row in vlm.items():
        if k not in a or k not in b:
            continue
        
        if not (to_int(a[k]["color_correct"]) and to_int(b[k]["color_correct"])):
            continue
        expected = vlm_row["color"]
        predicted = vlm_row["color_predicted"]
        matrix[expected][predicted] += 1

    return matrix


def write_consensus_table(rows: list[dict], path: Path) -> None:
    """Dump per-row consensus / VLM table for downstream inspection."""
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_divergences(
    consensus_rows: list[dict],
    humans: dict[str, dict],
    vlm: dict[tuple, dict],
    path: Path,
) -> int:
    """
    Write a CSV of every row where humans agreed but the VLM disagreed,
    plus rows where the two humans themselves disagreed. These are the
    cases worth eyeballing — they tell you whether the VLM has a
    systematic blind spot or the task itself was ambiguous.
    """
    names = sorted(humans.keys())
    a, b = humans[names[0]], humans[names[1]]

    fieldnames = [
        "category", "object", "color", "seed", "path",
        names[0] + "_object", names[0] + "_color",
        names[1] + "_object", names[1] + "_color",
        "vlm_object_pred", "vlm_color_pred", "vlm_binding",
    ]
    written = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for k in sorted(set(a) & set(b) & set(vlm)):
            a_obj, a_col = to_int(a[k]["object_correct"]), to_int(a[k]["color_correct"])
            b_obj, b_col = to_int(b[k]["object_correct"]), to_int(b[k]["color_correct"])
            a_bind = int(a_obj and a_col)
            b_bind = int(b_obj and b_col)
            v_bind = to_int(vlm[k]["binding_correct"])

            row = {
                "object": a[k]["object"],
                "color":  a[k]["color"],
                "seed":   a[k]["seed"],
                "path":   a[k].get("path", ""),
                names[0] + "_object": a_obj,
                names[0] + "_color":  a_col,
                names[1] + "_object": b_obj,
                names[1] + "_color":  b_col,
                "vlm_object_pred": vlm[k]["object_predicted"],
                "vlm_color_pred":  vlm[k]["color_predicted"],
                "vlm_binding":     v_bind,
            }

            if a_bind == b_bind and v_bind != a_bind:
                row["category"] = "vlm_disagrees_with_human_consensus"
                writer.writerow(row)
                written += 1
            elif a_bind != b_bind:
                row["category"] = "humans_disagree"
                writer.writerow(row)
                written += 1
    return written


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration-dir", type=Path, default=Path("data/calibration"),
                   help="Directory containing annotations_*.csv from each annotator.")
    p.add_argument("--vlm", type=Path, required=True,
                   help="Path to judgments.csv produced by exp3_judge.py.")
    p.add_argument("--taxonomy", type=Path, default=Path("configs/objects_colors.yaml"),
                   help="Taxonomy of canonical colors (for the confusion matrix).")
    p.add_argument("--out-dir", type=Path, default=Path("results/calibration_analysis"),
                   help="Where to write summary.json and the analysis CSVs.")
    args = p.parse_args()

    from binding.io import load_yaml

    args.out_dir.mkdir(parents=True, exist_ok=True)

    humans = load_annotations(args.calibration_dir)
    vlm = load_vlm_judgments(args.vlm)
    taxonomy = load_yaml(args.taxonomy)
    canonical_colors = taxonomy["colors"]

    h_kappa = humans_kappa_block(humans)
    print("\n=== Inter-annotator agreement ===")
    for axis in ("object", "color", "binding"):
        k = h_kappa[axis]["kappa"]
        a = h_kappa[axis]["raw_agreement"]
        print(f"  {axis:8s}  kappa={k:+.3f}  raw_agreement={a:.1%}")

    vlm_stats = vlm_vs_humans(humans, vlm)
    print("\n=== VLM vs humans (binding accuracy) ===")
    print(f"  vs {vlm_stats['vs_annotator_a']['annotator']:>10s}  "
          f"{vlm_stats['vs_annotator_a']['accuracy']:.1%}  "
          f"({vlm_stats['vs_annotator_a']['correct']}/{vlm_stats['vs_annotator_a']['total']})")
    print(f"  vs {vlm_stats['vs_annotator_b']['annotator']:>10s}  "
          f"{vlm_stats['vs_annotator_b']['accuracy']:.1%}  "
          f"({vlm_stats['vs_annotator_b']['correct']}/{vlm_stats['vs_annotator_b']['total']})")
    cons = vlm_stats["consensus"]
    print(f"  vs  consensus  {cons['accuracy']:.1%}  "
          f"({cons['correct']}/{cons['n_consensus']}, "
          f"{vlm_stats['n_joined'] - cons['n_consensus']} disagreement rows excluded)")

    cm = color_confusion(humans, vlm, canonical_colors)
    cm_path = args.out_dir / "color_confusion.csv"
    all_predicted = sorted({col for counter in cm.values() for col in counter})
    with cm_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["expected_color"] + all_predicted)
        for c in canonical_colors:
            writer.writerow([c] + [cm[c].get(p, 0) for p in all_predicted])
    print(f"\n[analysis] wrote color confusion matrix: {cm_path}")

    consensus_path = args.out_dir / "consensus_table.csv"
    write_consensus_table(vlm_stats.pop("consensus_rows"), consensus_path)
    print(f"[analysis] wrote consensus table: {consensus_path}")

    divergences_path = args.out_dir / "divergences.csv"
    n_div = write_divergences(vlm_stats, humans, vlm, divergences_path)
    print(f"[analysis] wrote {n_div} divergence rows: {divergences_path}")

    summary = {
        "n_human_annotations": {name: len(rows) for name, rows in humans.items()},
        "n_vlm_judgments": len(vlm),
        "human_agreement": h_kappa,
        "vlm_vs_humans": vlm_stats,
        "n_divergence_rows": n_div,
    }
    summary_path = args.out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"\n[analysis] full summary: {summary_path}")

    print("\n=== HEADLINE ===")
    bind_kappa = h_kappa["binding"]["kappa"]
    if bind_kappa < 0.6:
        print(f"   Human binding kappa {bind_kappa:.3f} is low — calibration is shaky.")
        print(f"       Consider revising the annotation guidelines and re-labelling.")
    elif bind_kappa < 0.8:
        print(f"  Human binding kappa {bind_kappa:.3f} is substantial.")
    else:
        print(f"  Human binding kappa {bind_kappa:.3f} is almost perfect.")

    vlm_acc = cons["accuracy"]
    if vlm_acc < 0.80:
        print(f"   VLM consensus accuracy {vlm_acc:.1%} is low for color recognition.")
        print(f"       Inspect divergences.csv — the judge may have systematic blind spots.")
    elif vlm_acc < 0.90:
        print(f"   VLM consensus accuracy {vlm_acc:.1%} is acceptable but worth documenting.")
    else:
        print(f"   VLM consensus accuracy {vlm_acc:.1%} is strong — safe to scale to all 3600 images.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
