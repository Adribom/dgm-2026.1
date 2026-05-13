from typing import Dict, List

from .base import MetricFn

# ---------------------------------------------------------------------------
# Standard classification metrics (sklearn-backed)
# ---------------------------------------------------------------------------

def _accuracy(y_true: List[int], y_pred: List[int], **kwargs) -> float:
    from sklearn.metrics import accuracy_score
    return round(float(accuracy_score(y_true, y_pred)), 4)


def _auc(y_true: List[int], y_score: List[float], **kwargs) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return round(float(roc_auc_score(y_true, y_score)), 4)
    except ValueError:
        # Only one class present
        return float("nan")


def _f1(y_true: List[int], y_pred: List[int], **kwargs) -> float:
    from sklearn.metrics import f1_score
    return round(float(f1_score(y_true, y_pred, zero_division=0)), 4)


def _avg_precision(y_true: List[int], y_score: List[float], **kwargs) -> float:
    from sklearn.metrics import average_precision_score
    try:
        return round(float(average_precision_score(y_true, y_score)), 4)
    except ValueError:
        return float("nan")


accuracy = MetricFn(name="Accuracy", fn=_accuracy)
auc = MetricFn(name="AUC", fn=_auc)
f1 = MetricFn(name="F1", fn=_f1)
avg_precision = MetricFn(name="Avg Precision", fn=_avg_precision)


# ---------------------------------------------------------------------------
# CSS: Consistency, Specificity, Selectivity
# (FakeVLM paper metric — evaluates explanation accuracy from text responses)
#
# Default implementation parses responses via keyword matching. Swap the
# `css` MetricFn for a custom one if you have the paper's exact formula.
# ---------------------------------------------------------------------------

_FAKE_KEYWORDS = frozenset(
    {"fake", "synthetic", "manipulated", "generated", "artificial",
     "forged", "deepfake", "fabricated", "inauthentic", "altered"}
)
_REAL_KEYWORDS = frozenset(
    {"real", "authentic", "genuine", "original", "unmodified", "natural"}
)


def _parse_response(response: str) -> int:
    """Returns 0=fake, 1=real, -1=undecided from model text response."""
    lower = response.lower()
    fake_hits = sum(1 for k in _FAKE_KEYWORDS if k in lower)
    real_hits = sum(1 for k in _REAL_KEYWORDS if k in lower)
    if fake_hits > real_hits:
        return 0
    if real_hits > fake_hits:
        return 1
    return -1


def _css(
    y_true: List[int],
    responses: List[str],
    **kwargs,
) -> Dict[str, float]:
    text_preds = [_parse_response(r) for r in responses]

    # Exclude undecided samples from CSS computation
    pairs = [(t, tp) for t, tp in zip(y_true, text_preds) if tp != -1]
    if not pairs:
        return {"CSS-Con": float("nan"), "CSS-Spe": float("nan"), "CSS-Sel": float("nan")}

    y_true_v, tp_v = zip(*pairs)

    # Consistency: overall text-decision accuracy
    consistency = sum(t == tp for t, tp in zip(y_true_v, tp_v)) / len(y_true_v)

    # Specificity: TN / (TN + FP) — real images correctly identified as real via text
    real_pairs = [(t, tp) for t, tp in zip(y_true_v, tp_v) if t == 1]
    specificity = (
        sum(tp == 1 for _, tp in real_pairs) / len(real_pairs)
        if real_pairs else float("nan")
    )

    # Selectivity: TP / (TP + FN) — fake images correctly identified as fake via text
    fake_pairs = [(t, tp) for t, tp in zip(y_true_v, tp_v) if t == 0]
    selectivity = (
        sum(tp == 0 for _, tp in fake_pairs) / len(fake_pairs)
        if fake_pairs else float("nan")
    )

    return {
        "CSS-Con": round(consistency, 4),
        "CSS-Spe": round(specificity, 4),
        "CSS-Sel": round(selectivity, 4),
    }


css = MetricFn(name="CSS", fn=_css)
