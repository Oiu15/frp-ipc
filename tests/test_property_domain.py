"""Property tests for domain/calibration.py — algebraic invariants.

Calibration math uses floating-point arithmetic on real-world sensor data.
These tests verify that the core formulas hold for any valid input, not
just the hand-picked examples in the table-driven tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from domain.calibration import (
    compute_od_b_candidate,
    fit_id_diameter,
    verify_id_calibration,
)


# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------


def _non_empty_floats() -> st.SearchStrategy[list[float]]:
    """List of 1–100 finite floats (avoids NaN/Inf blowing up statistics)."""
    return st.lists(
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=100,
    )


def _full_circle_theta(n: int = 36) -> st.SearchStrategy[np.ndarray]:
    """Generate `n` angles uniformly covering [0, 360) degrees."""
    return st.builds(
        lambda n: np.linspace(0.0, 360.0 * (n - 1) / n, n),
        n=st.just(n),
    )


# ---------------------------------------------------------------------------
# compute_od_b_candidate — algebraic identity
# ---------------------------------------------------------------------------


@given(sums=_non_empty_floats(), d_ref=st.floats(min_value=0.0, max_value=500.0))
def test_od_b_candidate_identity(sums: list[float], d_ref: float) -> None:
    """mean_sum ≈ np.mean(sums), and b_candidate == mean_sum + d_ref."""
    result = compute_od_b_candidate(sums, d_ref)

    assert result.ok
    assert result.n_used == len(sums)
    assert result.mean_sum is not None
    # algebraic identity
    assert result.b_candidate == result.mean_sum + d_ref
    # statistical identity (within float tolerance)
    assert result.mean_sum == pytest.approx(float(np.mean(sums)))


# ---------------------------------------------------------------------------
# fit_id_diameter — constant chord
# ---------------------------------------------------------------------------


@given(
    k=st.floats(
        min_value=10.0, max_value=500.0,
        allow_nan=False, allow_infinity=False, allow_subnormal=False,
    ),
    delta_c=st.floats(
        min_value=-40.0, max_value=40.0,
        allow_nan=False, allow_infinity=False, allow_subnormal=False,
    ),
    n=st.integers(min_value=6, max_value=72),
)
def test_fit_id_diameter_constant_chord(k: float, delta_c: float, n: int) -> None:
    """For a perfect constant effective chord, the fitted diameter equals
    k+delta_c and eccentricity is zero."""
    # effective diameter must be > 0 — zero-diameter circle fit is unstable
    eff = k + delta_c
    assume(eff > 0.0)  # physically irrelevant: no pipe has zero/negative diameter

    theta = np.linspace(0.0, 360.0 * (n - 1) / n, n)
    c_mm = np.full(n, float(k))
    m_mm = np.zeros(n)

    result = fit_id_diameter(theta, c_mm, m_mm, float(delta_c))

    assert result.diam == pytest.approx(float(eff), rel=1e-9)
    assert result.e == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# verify_id_calibration — self-consistent reference
# ---------------------------------------------------------------------------


@given(
    k=st.floats(
        min_value=10.0, max_value=500.0,
        allow_nan=False, allow_infinity=False, allow_subnormal=False,
    ),
    n=st.integers(min_value=30, max_value=72),
)
def test_verify_id_calibration_self_consistent(k: float, n: int) -> None:
    """When we measure a perfect circle and use its diameter as the reference,
    verification must pass with near-zero error."""
    theta = np.linspace(0.0, 359.0, n)
    c_mm = np.full(n, float(k))
    m_mm = np.zeros(n)

    result = verify_id_calibration(theta, c_mm, m_mm, delta_c=0.0, d_ref=float(k))

    assert result.ok
    assert result.err_mm == pytest.approx(0.0, abs=1e-6)
    # linspace(0, 359, n) → ~99.7% coverage (not quite 360°), so relax the check
    assert result.cov_pct >= 99.0
    assert result.sample_count == n
