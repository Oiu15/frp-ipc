from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from core.models import Recipe
from domain.calibration import (
    compute_od_b_candidate,
    fit_id_diameter,
    fit_id_single_from_out2,
    solve_id_delta_candidate,
    verify_id_calibration,
)


class TestCalibrationDomain:
    """Pure calibration computation — parametrised input / expected pairs."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _number(value: float | None) -> float:
        assert value is not None
        return float(value)

    # ------------------------------------------------------------------
    # compute_od_b_candidate
    # ------------------------------------------------------------------

    _OD_B_CASES: list[tuple[list[float], float, dict[str, Any]]] = [
        (
            [10.0, 12.0, 14.0],
            180.0,
            {"ok": True, "mean_sum": 12.0, "b_candidate": 192.0, "n_used": 3},
        ),
        (
            [],
            180.0,
            {"ok": False, "reason": "empty_sums", "b_candidate": None, "n_used": 0},
        ),
    ]

    @pytest.mark.parametrize(("sums", "d_ref", "expected"), _OD_B_CASES)
    def test_compute_od_b_candidate(
        self,
        sums: list[float],
        d_ref: float,
        expected: dict[str, Any],
    ) -> None:
        result = compute_od_b_candidate(sums, d_ref)
        assert result.ok == expected["ok"]
        if expected["b_candidate"] is not None:
            assert self._number(result.b_candidate) == pytest.approx(expected["b_candidate"])
        else:
            assert result.b_candidate is None
        assert result.n_used == expected["n_used"]
        if result.ok:
            assert self._number(result.mean_sum) == pytest.approx(expected["mean_sum"])
        else:
            assert result.reason == expected["reason"]

    # ------------------------------------------------------------------
    # fit_id_single_from_out2
    # ------------------------------------------------------------------

    _ID_SINGLE_CASES: list[tuple[list[float], list[float], dict[str, Any]]] = [
        (
            [0.0, 1.0, 2.0],
            [10.0, 10.0, 10.0],
            {"ok": False, "reason": "too_few_bins"},
        ),
        (
            list(np.linspace(0.0, 330.0, 12)),
            [75.0] * 12,
            {"ok": True, "mean_L2_decenter": 75.0, "id_est_mm": 75.0},
        ),
    ]

    @pytest.mark.parametrize(("theta", "out2", "expected"), _ID_SINGLE_CASES)
    def test_fit_id_single_from_out2(
        self,
        theta: list[float],
        out2: list[float],
        expected: dict[str, Any],
    ) -> None:
        recipe = Recipe(id_single_k=1.0, id_single_b=0.0, bin_count=12, bin_method="median", pp_mode="p99_p1")
        result = fit_id_single_from_out2(theta, out2, recipe)

        assert result.ok == expected["ok"]
        if expected["ok"]:
            assert self._number(result.mean_L2_decenter) == pytest.approx(expected["mean_L2_decenter"])
            assert self._number(result.id_est_mm) == pytest.approx(expected["id_est_mm"])
            assert self._number(result.cov) >= 1.0
        else:
            assert result.reason == expected["reason"]
            assert self._number(result.cov or 0.0) < 0.5

    # ------------------------------------------------------------------
    # fit_id_diameter
    # ------------------------------------------------------------------

    def test_fit_id_diameter_constant_chord(self) -> None:
        theta = np.linspace(0.0, 330.0, 12)
        c_mm = np.full(12, 100.0)
        m_mm = np.zeros(12)
        result = fit_id_diameter(theta, c_mm, m_mm, 0.0)
        assert result.diam == pytest.approx(100.0)
        assert result.e == pytest.approx(0.0)

    # ------------------------------------------------------------------
    # solve_id_delta_candidate
    # ------------------------------------------------------------------

    _DELTA_CASES: list[tuple[list[float], list[float], list[float], float, dict[str, Any]]] = [
        (
            list(np.linspace(0.0, 330.0, 36)),
            list(np.full(36, 100.0)),
            list(np.zeros(36)),
            120.0,
            {"ok": True, "fallback_used": False, "delta_candidate": 20.0, "diam": 120.0, "has_fit": True},
        ),
        (
            list(np.linspace(0.0, 90.0, 10)),
            list(np.full(10, 100.0)),
            list(np.zeros(10)),
            120.0,
            {
                "ok": True,
                "fallback_used": True,
                "reason": "fallback_cmax",
                "delta_candidate": 20.0,
                "has_fit": False,
            },
        ),
    ]

    @pytest.mark.parametrize(("theta", "c_mm", "m_mm", "d_ref", "expected"), _DELTA_CASES)
    def test_solve_id_delta_candidate(
        self,
        theta: list[float],
        c_mm: list[float],
        m_mm: list[float],
        d_ref: float,
        expected: dict[str, Any],
    ) -> None:
        result = solve_id_delta_candidate(np.array(theta), np.array(c_mm), np.array(m_mm), d_ref)

        assert result.ok == expected["ok"]
        assert result.fallback_used == expected["fallback_used"]
        assert self._number(result.delta_candidate) == pytest.approx(expected["delta_candidate"], abs=1e-4)
        if expected["has_fit"]:
            assert result.fit is not None
            assert float(result.fit.diam) == pytest.approx(expected["diam"], abs=1e-4)
        else:
            assert result.fit is None
            assert result.reason == expected["reason"]

    # ------------------------------------------------------------------
    # verify_id_calibration
    # ------------------------------------------------------------------

    _VERIFY_CASES: list[tuple[list[float], list[float], list[float], float, float, dict[str, Any]]] = [
        (
            list(np.linspace(0.0, 359.0, 36)),
            list(np.full(36, 100.0)),
            list(np.zeros(36)),
            0.0,
            100.0,
            {"ok": True, "err_mm": 0.0, "sample_count": 36},
        ),
        (
            list(np.linspace(0.0, 359.0, 36)),
            list(np.full(36, 100.0)),
            list(np.zeros(36)),
            0.0,
            110.0,
            {"ok": False, "err_mm": -10.0, "sample_count": 36},
        ),
    ]

    @pytest.mark.parametrize(("theta", "c_mm", "m_mm", "delta_c", "d_ref", "expected"), _VERIFY_CASES)
    def test_verify_id_calibration(
        self,
        theta: list[float],
        c_mm: list[float],
        m_mm: list[float],
        delta_c: float,
        d_ref: float,
        expected: dict[str, Any],
    ) -> None:
        result = verify_id_calibration(np.array(theta), np.array(c_mm), np.array(m_mm), delta_c=delta_c, d_ref=d_ref)

        assert result.ok == expected["ok"]
        assert self._number(result.err_mm) == pytest.approx(expected["err_mm"])
        assert self._number(result.cov_pct) >= 95.0
        assert result.sample_count == expected["sample_count"]

    def test_verify_id_calibration_raises_for_too_few_samples(self) -> None:
        theta = np.linspace(0.0, 100.0, 10)
        c_mm = np.full(10, 100.0)
        m_mm = np.zeros(10)
        with pytest.raises(ValueError):
            verify_id_calibration(theta, c_mm, m_mm, delta_c=0.0, d_ref=100.0)
