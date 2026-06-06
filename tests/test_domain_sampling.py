"""Tests for domain/sampling.py — pure data-reduction helpers.

These functions are pure maths and stateless.  They are exercised
indirectly by integration tests, but direct unit tests catch
regressions instantly without needing a full workflow setup.
"""

from __future__ import annotations

import math

import pytest

from domain.sampling import (
    _adaptive_bin_count,
    _max_gap_deg_from_bins,
    _reduce_bin,
    _theta_apply_delay,
    _wrap_deg_180,
)


# ---------------------------------------------------------------------------
# _max_gap_deg_from_bins
# ---------------------------------------------------------------------------


class TestMaxGapDegFromBins:
    """Maximum empty angular window computed from bin-hit counts."""

    @pytest.mark.parametrize(
        "bins",
        [
            [0, 0, 0],
            [0] * 6,
        ],
    )
    def test_no_hits_returns_full_circle(self, bins: list[int]) -> None:
        assert _max_gap_deg_from_bins(bins, len(bins)) == 360.0

    def test_all_hit_returns_zero(self) -> None:
        assert _max_gap_deg_from_bins([1, 1, 1, 1], 4) == 0.0

    def test_single_gap(self) -> None:
        # hit hit _ hit hit hit → one empty bin = 60° for n=6
        cnt = [1, 1, 0, 1, 1, 1]
        assert _max_gap_deg_from_bins(cnt, 6) == pytest.approx(60.0)

    def test_wraparound_gap(self) -> None:
        # gap wraps from end to start
        cnt = [0, 1, 1, 1, 1, 0]
        # two consecutive empty bins across boundary = 2 * (360/6) = 120°
        assert _max_gap_deg_from_bins(cnt, 6) == pytest.approx(120.0)

    def test_n_zero_returns_zero(self) -> None:
        assert _max_gap_deg_from_bins([0, 0, 0], 0) == 0.0


# ---------------------------------------------------------------------------
# _wrap_deg_180
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("d", "expected"),
    [
        (350.0, -10.0),
        (180.0, -180.0),
        (-180.0, -180.0),
        (0.0, 0.0),
        (-10.0, -10.0),
        (-200.0, 160.0),
        (float("inf"), float("nan")),
        (float("nan"), float("nan")),
    ],
)
def test_wrap_deg_180(d: float, expected: float) -> None:
    result = _wrap_deg_180(d)
    if math.isnan(expected):
        assert math.isnan(result)
    else:
        assert result == expected


# ---------------------------------------------------------------------------
# _theta_apply_delay
# ---------------------------------------------------------------------------


class TestThetaApplyDelay:
    def test_zero_delay_is_identity(self) -> None:
        assert _theta_apply_delay(45.0, omega_deg_s=10.0, delay_s=0.0) == 45.0

    def test_positive_delay_shifts_forward(self) -> None:
        # 90° + 10°/s * 2s = 110°
        assert _theta_apply_delay(90.0, omega_deg_s=10.0, delay_s=2.0) == 110.0

    def test_wraps_past_360(self) -> None:
        assert _theta_apply_delay(350.0, omega_deg_s=10.0, delay_s=2.0) == 10.0

    def test_non_finite_theta_returns_nan(self) -> None:
        assert math.isnan(_theta_apply_delay(float("nan"), 1.0, 0.0))
        assert math.isnan(_theta_apply_delay(float("inf"), 1.0, 0.0))

    def test_non_finite_omega_treated_as_zero(self) -> None:
        assert _theta_apply_delay(45.0, float("nan"), delay_s=2.0) == 45.0

    def test_non_finite_delay_treated_as_zero(self) -> None:
        assert _theta_apply_delay(45.0, 10.0, delay_s=float("inf")) == 45.0


# ---------------------------------------------------------------------------
# _reduce_bin
# ---------------------------------------------------------------------------


class TestReduceBin:
    def test_empty_returns_nan(self) -> None:
        assert math.isnan(_reduce_bin([]))

    def test_all_non_finite_returns_nan(self) -> None:
        assert math.isnan(_reduce_bin([float("nan"), float("inf")]))

    def test_median_default(self) -> None:
        assert _reduce_bin([1.0, 2.0, 100.0]) == 2.0

    def test_mean_mode(self) -> None:
        result = _reduce_bin([1.0, 2.0, 3.0], method="mean")
        assert result == pytest.approx(2.0)

    def test_unknown_method_falls_back_to_median(self) -> None:
        assert _reduce_bin([1.0, 5.0, 3.0], method="unknown") == 3.0


# ---------------------------------------------------------------------------
# _adaptive_bin_count
# ---------------------------------------------------------------------------


class TestAdaptiveBinCount:
    def test_above_min_samples_uses_requested(self) -> None:
        assert _adaptive_bin_count(90, n_samples=200) == 90

    def test_few_samples_caps_at_half(self) -> None:
        # n_samples=20 → cap = 10, so requested 90 → 10
        assert _adaptive_bin_count(90, n_samples=20) == 10

    def test_min_bins_does_not_override_tighter_cap(self) -> None:
        # cap (10) is already below min_bins (12), but cap takes priority
        # when n_samples is below min_bins*2 threshold
        assert _adaptive_bin_count(90, n_samples=20, min_bins=8) == 10

    def test_zero_samples_returns_minimum_3(self) -> None:
        assert _adaptive_bin_count(90, n_samples=0) == 3

    def test_at_least_three(self) -> None:
        assert _adaptive_bin_count(1, n_samples=0) == 3
