from __future__ import annotations

"""批次3:geometry_v2 单截面分流 + 并列导出(纯追加,legacy 不变)。"""

import csv
import json

import numpy as np
import pytest

from core.models import MeasureRow, Recipe
from domain.geometry_calibration import (
    ToolingCalibration,
    id_predict_L,
    id_tooling_from_simple,
)
from domain.state import CalibrationSnapshot, RunContext, RunIdentity
from frp_workflow.row_math import _compute_section_v2
from repositories.run_repository import RunRepository


def _synth_id_raw_points(n=180, r=76.35, e=(1.2, -0.8)):
    """整圈合成 ID raw_points(theta_deg + id_x1/x2),工装与下方 ToolingCalibration 对齐。"""
    truth = id_tooling_from_simple(D=140.0, s=0.3, axis_deg=12.0, q=(2.0, -1.5))
    ev = np.array(e, dtype=float)
    pts = []
    for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
        l1 = id_predict_L(truth.probe_a, ev, float(th), r)
        l2 = id_predict_L(truth.probe_b, ev, float(th), r)
        pts.append({"theta_deg": float(np.rad2deg(th)), "id_x1_mm": l1, "id_x2_mm": l2})
    return pts


def _matching_tooling():
    return ToolingCalibration(id_D_eff=140.0, id_s_lateral=0.3, id_axis_deg=12.0, id_qx=2.0, id_qy=-1.5)


def test_compute_section_v2_recovers_id_diameter():
    pytest.importorskip("scipy")
    r = 76.35
    pts = _synth_id_raw_points(r=r)
    out = _compute_section_v2(pts, _matching_tooling())
    assert "id_diam_v2" in out
    assert abs(out["id_diam_v2"] - 2 * r) < 0.01
    assert np.allclose([out["id_cx_v2"], out["id_cy_v2"]], [1.2, -0.8], atol=0.05)


def test_compute_section_v2_skips_when_tooling_uncalibrated():
    pts = _synth_id_raw_points()
    # default tooling has D_eff=0 -> not calibrated -> {}
    assert _compute_section_v2(pts, ToolingCalibration()) == {}
    # missing tooling -> {}
    assert _compute_section_v2(pts, None) == {}


def test_compute_section_v2_skips_on_insufficient_points():
    assert _compute_section_v2([{"theta_deg": 0.0, "id_x1_mm": 1.0, "id_x2_mm": 1.0}], _matching_tooling()) == {}


def _make_row(idx, z, *, id_diam_v2=None, id_cx_v2=None, id_cy_v2=None):
    return MeasureRow(
        idx=idx, x_ui=z, x_abs=z,
        od_avg=190.0, od_dev=0.0, od_runout=0.0, od_round=0.0,
        id_avg=152.7, id_dev=0.0, id_runout=0.0, id_round=0.0,
        concentricity=0.0,
        id_diam_v2=id_diam_v2, id_round_v2=0.01, id_cx_v2=id_cx_v2, id_cy_v2=id_cy_v2,
    )


def _make_context(algo_version, rows, raw_points):
    return RunContext(
        identity=RunIdentity(serial="20260101-test-001", run_id="rid-1", started_at_ts=0.0),
        recipe=Recipe(algo_version=algo_version),
        calibration=CalibrationSnapshot(),
        rows=rows,
        raw_points=raw_points,
        finished_at_ts=10.0,
        status="DONE",
        completed=True,
        completed_sections=len(rows),
        expected_sections=len(rows),
    )


def test_export_geometry_v2_writes_parallel_files(tmp_path):
    repo = RunRepository(app_root_dir=tmp_path)
    rows = [
        _make_row(1, 10.0, id_diam_v2=152.70, id_cx_v2=0.10, id_cy_v2=0.05),
        _make_row(2, 250.0, id_diam_v2=152.69, id_cx_v2=0.40, id_cy_v2=0.05),
    ]
    raw_points = [
        {"section_idx": 1, "sample_idx": 0, "theta_deg": 0.0, "id_x1_mm": 1.1,
         "id_x2_mm": 1.2, "od_out1": 5.0, "od_out2": 5.1, "od_delta": 0.1},
    ]
    run_dir = repo.export_run(_make_context("geometry_v2", rows, raw_points))
    from pathlib import Path
    rd = Path(run_dir)
    assert (rd / "section_results.csv").exists()      # legacy still written
    assert (rd / "raw_points.csv").exists()
    assert (rd / "section_results_v2.csv").exists()   # additive v2
    assert (rd / "raw_points_ext.csv").exists()

    with open(rd / "section_results_v2.csv", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert "id_diam_v2_mm" in header and "id_cx_v2_mm" in header

    meta = json.loads((rd / "meta.json").read_text(encoding="utf-8"))
    assert "geometry_v2" in meta
    assert meta["geometry_v2"]["n_sections"] == 2
    assert "id_tau_x_deg" in meta["geometry_v2"]
    assert meta["exports"]["section_results_v2_csv"].endswith("section_results_v2.csv")


def test_export_legacy_does_not_write_v2_files(tmp_path):
    repo = RunRepository(app_root_dir=tmp_path)
    rows = [_make_row(1, 10.0)]
    run_dir = repo.export_run(_make_context("legacy", rows, []))
    from pathlib import Path
    rd = Path(run_dir)
    assert (rd / "section_results.csv").exists()
    assert not (rd / "section_results_v2.csv").exists()
    assert not (rd / "raw_points_ext.csv").exists()
    meta = json.loads((rd / "meta.json").read_text(encoding="utf-8"))
    assert "geometry_v2" not in meta
    assert "section_results_v2_csv" not in meta["exports"]
