from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from frp_workflow.autoflow_orchestrator import AutoFlowOrchestrator
from frp_workflow.steps.measure_section_context import MeasureSectionContext
from frp_workflow.steps.row_build_result import RowBuildResult
from frp_workflow.steps.sampling_result import SamplingResult


@dataclass
class _FakeSample:
    fit_weights_od: object
    fit_weights_id: object


def _minimal_orchestrator() -> Any:
    return cast(Any, AutoFlowOrchestrator.__new__(AutoFlowOrchestrator))


def _context() -> MeasureSectionContext:
    return MeasureSectionContext(
        section_index=4,
        z_pos_mm=12.5,
        x_abs=34.5,
        centers_xyz=[(1.0, 2.0, 3.0)],
        centers_xyz_id=[(4.0, 5.0, 6.0)],
        concentricity_list=[0.125],
    )


def test_build_row_impl_passes_id_sample_weights(monkeypatch) -> None:
    orch = _minimal_orchestrator()
    calls: list[dict[str, Any]] = []
    row = {"row": "sentinel"}
    primary_sample = _FakeSample(fit_weights_od=object(), fit_weights_id=object())
    id_sample = _FakeSample(fit_weights_od=object(), fit_weights_id=object())
    coords_od = np.array([[1.0, 2.0, 3.0]])
    coords_id = np.array([[4.0, 5.0, 6.0]])
    raw_points = [{"theta_deg": 0.0}]
    context = _context()
    sampling_result = SamplingResult(
        scan_mode="SPLIT",
        keep_spinning=False,
        primary_sample=primary_sample,
        id_sample=id_sample,
        coords_od=coords_od,
        coords_id=coords_id,
        raw_od=123,
        raw_id=456,
        raw_points=raw_points,
        split_shift_deg=1.5,
        coax_unreliable=True,
    )

    def fake_build_section_row(**kwargs: Any) -> dict[str, str]:
        calls.append(kwargs)
        return row

    monkeypatch.setattr(orch, "_build_section_row", fake_build_section_row)

    result = AutoFlowOrchestrator._build_row_impl(orch, context, sampling_result)

    assert isinstance(result, RowBuildResult)
    assert result.row is row
    assert len(calls) == 1
    assert calls[0]["section_index"] == context.section_index
    assert calls[0]["z_pos_mm"] == float(context.z_pos_mm)
    assert calls[0]["x_abs"] == float(context.x_abs)
    assert calls[0]["coords_od"] is coords_od
    assert calls[0]["coords_id"] is coords_id
    assert calls[0]["raw_od"] == "123"
    assert calls[0]["raw_id"] == "456"
    assert calls[0]["raw_points"] is raw_points
    assert calls[0]["fit_weights_od"] is primary_sample.fit_weights_od
    assert calls[0]["fit_weights_id"] is id_sample.fit_weights_id
    assert calls[0]["scan_mode"] == "SPLIT"
    assert calls[0]["split_shift_deg"] == 1.5
    assert calls[0]["coax_unreliable"] is True
    assert calls[0]["centers_xyz"] is context.centers_xyz
    assert calls[0]["centers_xyz_id"] is context.centers_xyz_id
    assert calls[0]["concentricity_list"] is context.concentricity_list


def test_build_row_impl_falls_back_to_primary_id_weights(monkeypatch) -> None:
    orch = _minimal_orchestrator()
    calls: list[dict[str, Any]] = []
    row = {"row": "fallback"}
    primary_sample = _FakeSample(fit_weights_od=object(), fit_weights_id=object())
    context = _context()
    sampling_result = SamplingResult(
        scan_mode="SYNC",
        keep_spinning=True,
        primary_sample=primary_sample,
        id_sample=None,
        coords_od=np.array([]),
        coords_id=np.array([]),
        raw_od="od",
        raw_id="id",
        raw_points=[],
        split_shift_deg=None,
        coax_unreliable=None,
    )

    def fake_build_section_row(**kwargs: Any) -> dict[str, str]:
        calls.append(kwargs)
        return row

    monkeypatch.setattr(orch, "_build_section_row", fake_build_section_row)

    result = AutoFlowOrchestrator._build_row_impl(orch, context, sampling_result)

    assert result.row is row
    assert len(calls) == 1
    assert calls[0]["fit_weights_od"] is primary_sample.fit_weights_od
    assert calls[0]["fit_weights_id"] is primary_sample.fit_weights_id
    assert calls[0]["scan_mode"] == "SYNC"
    assert calls[0]["split_shift_deg"] is None
    assert calls[0]["coax_unreliable"] is None
