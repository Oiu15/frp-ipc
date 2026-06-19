from __future__ import annotations

import numpy as np

from frp_workflow.steps.measure_section_context import MeasureSectionContext
from frp_workflow.steps.row_build import RowBuildStep
from frp_workflow.steps.row_build_result import RowBuildResult
from frp_workflow.steps.sampling_result import SamplingResult


class FakePort:
    def __init__(self) -> None:
        self.context: MeasureSectionContext | None = None
        self.sampling_result: SamplingResult | None = None
        self.result = RowBuildResult(row={"section_idx": 2})

    def _build_row_impl(
        self,
        context: MeasureSectionContext,
        sampling_result: SamplingResult,
    ) -> RowBuildResult:
        self.context = context
        self.sampling_result = sampling_result
        return self.result


def test_row_build_step_delegates_to_impl() -> None:
    context = MeasureSectionContext(
        section_index=2,
        z_pos_mm=12.5,
        x_abs=34.5,
        centers_xyz=[],
        centers_xyz_id=[],
        concentricity_list=[],
    )
    sampling_result = SamplingResult(
        scan_mode="SYNC",
        keep_spinning=True,
        primary_sample=object(),
        id_sample=None,
        coords_od=np.array([]),
        coords_id=np.array([]),
        raw_od="",
        raw_id="",
        raw_points=[],
        split_shift_deg=None,
        coax_unreliable=None,
    )
    port = FakePort()
    step = RowBuildStep(port)

    result = step.execute(context, sampling_result)

    assert result is port.result
    assert port.context is context
    assert port.sampling_result is sampling_result
    assert step.name == "row_build"
