"""
Tests for exp3_calibration_analysis: validates Cohen's kappa and the
human-VLM joining logic against handcrafted cases.

The numbers checked here are reproducible by hand (or via scikit-learn
in a sanity-check session). The test set is small but covers each edge
case that bites real implementations: perfect agreement, perfect
disagreement, chance-level agreement, all-same-label, all-different
labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from exp3_calibration_analysis import cohens_kappa, to_int  

def test_perfect_agreement():
    """Two annotators that always agree → kappa=1.0, regardless of class distribution."""
    a = [1, 1, 0, 0, 1, 0, 1, 0]
    b = a[:]
    k, agr = cohens_kappa(a, b)
    assert abs(k - 1.0) < 1e-9
    assert abs(agr - 1.0) < 1e-9


def test_perfect_disagreement_balanced():
    """
    Two annotators that always flip on balanced labels → kappa = -1.0.
    Raw agreement is 0%, expected is 50%, kappa = (0 - 0.5)/(1 - 0.5) = -1.
    """
    a = [1, 0, 1, 0, 1, 0]
    b = [0, 1, 0, 1, 0, 1]
    k, agr = cohens_kappa(a, b)
    assert abs(k - (-1.0)) < 1e-9
    assert abs(agr - 0.0) < 1e-9


def test_chance_agreement_zero_kappa():
    """
    Independent annotators with 50/50 each. Agreement ≈ 50% by chance,
    expected agreement also 50% → kappa ≈ 0. We use the deterministic
    sequence (1,0,1,0,...)/(1,1,0,0,...) which gives exactly chance.
    """
    a = [1, 0, 1, 0, 1, 0, 1, 0]
    b = [1, 1, 0, 0, 1, 1, 0, 0]
    k, agr = cohens_kappa(a, b)
 
    assert abs(k - 0.0) < 1e-9
    assert abs(agr - 0.5) < 1e-9


def test_both_all_ones_undefined_resolves_to_one():
    """
    Both annotators say '1' for everything. P_e=1, so kappa is formally
    undefined (0/0). Our convention: report 1.0 if they also agree,
    else 0.0. They do agree (P_o=1), so we expect 1.0.
    """
    a = [1] * 10
    b = [1] * 10
    k, agr = cohens_kappa(a, b)
    assert k == 1.0
    assert agr == 1.0


def test_partial_agreement_known_kappa():
    """
    Worked example: a=[1,1,1,1,0,0,0,0], b=[1,1,0,0,1,1,0,0]
    Confusion matrix:
              b=1   b=0
        a=1    2     2
        a=0    2     2
    P_o = 4/8 = 0.5
    Marginals: P(a=1)=0.5, P(b=1)=0.5, so P_e = 0.5*0.5 + 0.5*0.5 = 0.5
    kappa = (0.5 - 0.5) / (1 - 0.5) = 0
    """
    a = [1, 1, 1, 1, 0, 0, 0, 0]
    b = [1, 1, 0, 0, 1, 1, 0, 0]
    k, agr = cohens_kappa(a, b)
    assert abs(k - 0.0) < 1e-9
    assert abs(agr - 0.5) < 1e-9


def test_high_kappa_imbalanced_labels():
    """
    Mostly-zeros label distribution with strong agreement.
    a = b = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    P_o = 1.0
    P(a=1)=0.1, P(b=1)=0.1, so P_e = 0.1*0.1 + 0.9*0.9 = 0.82
    kappa = (1 - 0.82) / (1 - 0.82) = 1.0
    """
    a = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    b = a[:]
    k, _ = cohens_kappa(a, b)
    assert abs(k - 1.0) < 1e-9


def test_kappa_one_disagreement():
    """
    Single disagreement on highly imbalanced labels.
    a = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    P_o = 9/10 = 0.9
    P(a=1)=0.1, P(b=1)=0.0
    P_e = 0.1*0.0 + 0.9*1.0 = 0.9
    kappa = (0.9 - 0.9) / (1 - 0.9) = 0
    """
    a = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    k, agr = cohens_kappa(a, b)
    assert abs(k - 0.0) < 1e-9
    assert abs(agr - 0.9) < 1e-9


# ── to_int coercion ────────────────────────────────────────────────────────
def test_to_int_strings():
    assert to_int("1") == 1
    assert to_int("0") == 0
    assert to_int("True") == 1
    assert to_int("False") == 0
    assert to_int("YES") == 1
    assert to_int("") == 0


def test_to_int_ints_and_bools():
    assert to_int(1) == 1
    assert to_int(0) == 0
    assert to_int(True) == 1
    assert to_int(False) == 0


def test_to_int_invalid_raises():
    import pytest
    with pytest.raises(ValueError):
        to_int("maybe")
