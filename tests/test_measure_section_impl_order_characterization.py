from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from frp_workflow.autoflow_orchestrator import AutoFlowOrchestrator
from frp_workflow.steps.measure_section_context import MeasureSectionContext
from frp_workflow.steps.row_build_result import RowBuildResult
from frp_workflow.steps.sampling_result import SamplingResult


@dataclass
class _FakeSample:
    sample_cov: tuple[int, int, int]
    sample_reason: tuple[str, float, float]
    n_od: int
    n_id: int
    max_gap_deg: float


class _FakeEventSink:
    def __init__(self, calls: list[str], rows: list[Any]) -> None:
        self._calls = calls
        self._rows = rows

    def publish_row(self, row: Any) -> None:
        self._calls.append("row_publish")
        self._rows.append(row)


def test_measure_section_impl_keeps_publish_build_record_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    rows: list[Any] = []
    raw_points = [{"theta_deg": 45.0}]
    row = {"section_idx": 5}
    context = MeasureSectionContext(
        section_index=5,
        z_pos_mm=12.5,
        x_abs=34.5,
        centers_xyz=[],
        centers_xyz_id=[],
        concentricity_list=[],
    )
    primary_sample = _FakeSample(
        sample_cov=(4, 3, 1),
        sample_reason=("ok", 1.25, 0.5),
        n_od=3,
        n_id=0,
        max_gap_deg=12.0,
    )
    sampling_result = SamplingResult(
        scan_mode="SYNC",
        keep_spinning=True,
        primary_sample=primary_sample,
        id_sample=None,
        coords_od=np.array([]),
        coords_id=np.array([]),
        raw_od="od",
        raw_id="id",
        raw_points=raw_points,
        split_shift_deg=None,
        coax_unreliable=None,
    )
    orchestrator = cast(Any, AutoFlowOrchestrator.__new__(AutoFlowOrchestrator))
    orchestrator.event_sink = _FakeEventSink(calls, rows)
    orchestrator.production_workflow = object()

    def fake_sample_section_impl(actual_context: MeasureSectionContext) -> SamplingResult:
        assert actual_context is context
        return sampling_result

    def fake_publish_section_raw_points(
        *,
        raw_points: list[dict],
        section_index: int,
        z_pos_mm: float,
    ) -> None:
        calls.append("raw_publish")
        assert raw_points is sampling_result.raw_points
        assert section_index == context.section_index
        assert z_pos_mm == context.z_pos_mm

    def fake_publish_section_coverage(*, payload: dict[str, Any]) -> None:
        calls.append("coverage_publish")
        assert payload["idx"] == context.section_index
        assert payload["cov"] == 0.75

    def fake_build_row_impl(
        actual_context: MeasureSectionContext,
        actual_sampling_result: SamplingResult,
    ) -> RowBuildResult:
        calls.append("row_build")
        assert actual_context is context
        assert actual_sampling_result is sampling_result
        return RowBuildResult(row=row)

    def fake_record_row_impl(actual_row: Any) -> None:
        calls.append("record_row")
        assert actual_row is row

    monkeypatch.setattr(orchestrator, "_sample_section_impl", fake_sample_section_impl)
    monkeypatch.setattr(orchestrator, "_publish_section_raw_points", fake_publish_section_raw_points)
    monkeypatch.setattr(orchestrator, "_publish_section_coverage", fake_publish_section_coverage)
    monkeypatch.setattr(orchestrator, "_build_row_impl", fake_build_row_impl)
    monkeypatch.setattr(orchestrator, "_record_row_impl", fake_record_row_impl)

    AutoFlowOrchestrator._measure_section_impl(orchestrator, context)

    assert calls == [
        "raw_publish",
        "coverage_publish",
        "row_build",
        "record_row",
        "row_publish",
    ]
    assert rows == [row]
    assert rows[0] is row
