"""
Tests for exp3_analyze_results.

The Spearman implementation is custom (no scipy dependency in the
analysis script). It's validated against scipy.stats.spearmanr over
100 randomized cases, plus hand-checked edge cases.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from exp3_analyze_results import bootstrap_ci, proportion_ci, spearman_rho  


def test_spearman_perfect_positive():
    """Strictly increasing → ρ = +1."""
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 30, 40, 50]
    assert abs(spearman_rho(x, y) - 1.0) < 1e-9


def test_spearman_perfect_negative():
    """Strictly decreasing → ρ = −1."""
    x = [1, 2, 3, 4, 5]
    y = [50, 40, 30, 20, 10]
    assert abs(spearman_rho(x, y) - (-1.0)) < 1e-9


def test_spearman_with_ties():
    """Ties resolved via fractional ranks. Hand-checked."""
    x = [1, 2, 2, 3, 4]
    y = [1, 2, 2, 3, 4]
    assert abs(spearman_rho(x, y) - 1.0) < 1e-9


def test_spearman_constant_returns_nan():
    """A constant array has no rank ordering → undefined."""
    x = [1, 1, 1, 1]
    y = [1, 2, 3, 4]
    assert math.isnan(spearman_rho(x, y))


def test_spearman_too_short_returns_nan():
    assert math.isnan(spearman_rho([1.0], [1.0]))
    assert math.isnan(spearman_rho([], []))

def test_spearman_matches_scipy_100_cases():
    """100 random pairs of arrays should match scipy.stats.spearmanr exactly."""
    try:
        from scipy.stats import spearmanr
    except ImportError:
        import pytest
        pytest.skip("scipy not installed in this env")

    rng = random.Random(0)
    mismatches = []
    for trial in range(100):
        n = rng.randint(5, 60)
        x = [rng.uniform(-5, 5) for _ in range(n)]
        y = [rng.uniform(-5, 5) for _ in range(n)]
        mine = spearman_rho(x, y)
        theirs = spearmanr(x, y).correlation
        if not (abs(mine - theirs) < 1e-9):
            mismatches.append((trial, mine, theirs))
    assert not mismatches, f"{len(mismatches)} mismatches: {mismatches[:3]}"

def test_proportion_ci_extreme_zero_pair():
    """A 0/30 pair has CI ≈ [0, 0] — informative, not a numerical issue."""
    lo, hi = proportion_ci(0, 30, n_boot=500, rng=random.Random(0))
    assert lo == 0.0
    assert hi == 0.0


def test_proportion_ci_extreme_full_pair():
    """A 30/30 pair has CI ≈ [1, 1]."""
    lo, hi = proportion_ci(30, 30, n_boot=500, rng=random.Random(0))
    assert lo == 1.0
    assert hi == 1.0


def test_proportion_ci_brackets_estimate():
    """For a 15/30 pair, the CI should bracket 0.5."""
    lo, hi = proportion_ci(15, 30, n_boot=500, rng=random.Random(0))
    assert lo <= 0.5 <= hi
    assert hi - lo > 0  


def test_bootstrap_ci_spearman_perfect_correlation_has_narrow_ci():
    """Perfectly correlated data → bootstrap CI hugs +1."""
    x = list(range(20))
    y = list(range(20))
    lo, hi = bootstrap_ci(x, y, spearman_rho, n_boot=300, rng=random.Random(0))
    assert lo >= 0.95
    assert hi == 1.0 or abs(hi - 1.0) < 1e-9
