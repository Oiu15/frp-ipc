from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.models import AxisCal
from domain.length_math import (
    LengthPlanStatus,
    LengthRange,
    average_edge_pair,
    length_from_edges,
    normalize_soft_limits_abs,
    plan_length_setup,
    plan_top_edge_approach,
    reject_outliers_sigma,
    z_disp_range,
)


ROOT = Path(__file__).resolve().parents[1]


class TestLengthMath:
    def test_soft_limit_normalization_uses_fallback_for_missing_or_invalid_limits(self) -> None:
        valid = normalize_soft_limits_abs(100.0, -400.0, fallback_min=-350.0, fallback_max=1200.0)
        assert valid.abs_min == -400.0
        assert valid.abs_max == 100.0
        assert not valid.used_fallback

        missing = normalize_soft_limits_abs(0.0, 0.0, fallback_min=-350.0, fallback_max=1200.0)
        assert missing.abs_min == -350.0
        assert missing.abs_max == 1200.0
        assert missing.used_fallback

    def test_z_range_and_length_setup_statuses_are_structured(self) -> None:
        axis_cal = AxisCal(sign=-1)
        z_range = z_disp_range(axis_cal, abs_min=-400.0, abs_max=100.0)
        assert z_range == LengthRange(z_min=-100.0, z_max=400.0, travel=500.0)

        ok = plan_length_setup(
            enabled=True,
            z_range=z_range,
            z_low_approach=100.0,
            low_search_dist=200.0,
            high_search_dist=80.0,
            high_margin=20.0,
            pipe_len=300.0,
        )
        assert ok.status is LengthPlanStatus.OK
        assert ok.lmax == pytest.approx(340.0)

        disabled = plan_length_setup(
            enabled=False,
            z_range=z_range,
            z_low_approach=100.0,
            low_search_dist=200.0,
            high_search_dist=80.0,
            high_margin=20.0,
            pipe_len=300.0,
        )
        assert disabled.status is LengthPlanStatus.DISABLED

        too_long = plan_length_setup(
            enabled=True,
            z_range=z_range,
            z_low_approach=100.0,
            low_search_dist=200.0,
            high_search_dist=80.0,
            high_margin=20.0,
            pipe_len=999.0,
        )
        assert too_long.status is LengthPlanStatus.PIPE_TOO_LONG

    def test_edge_helpers_and_top_approach_are_pure(self) -> None:
        assert length_from_edges(250.0, 100.0) == pytest.approx(150.0)
        assert length_from_edges(100.0, 250.0) is None
        assert average_edge_pair(10.0, 14.0) == pytest.approx(12.0)

        approach = plan_top_edge_approach(
            z_low_edge=250.0,
            pipe_len=500.0,
            high_margin=20.0,
            z_range=LengthRange(z_min=0.0, z_max=300.0, travel=300.0),
        )
        assert approach.z_approach == pytest.approx(0.0)
        assert approach.clamped

    def test_sigma_outlier_filter_keeps_finite_inliers(self) -> None:
        assert reject_outliers_sigma([1.0, 1.0, 1.0, 100.0, float("nan")], sigma=1.0) == (1.0, 1.0, 1.0)


def test_length_math_stays_out_of_ui_and_application_layers() -> None:
    source = (ROOT / "domain" / "length_math.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned_roots = {"application", "ui", "tkinter", "drivers", "repositories", "services"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in banned_roots
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in banned_roots
    assert "未启用" not in source
    assert "行程不足" not in source
