from __future__ import annotations

import numpy as np

from frp_workflow.steps.measure_section_context import MeasureSectionContext
from frp_workflow.steps.sampling import SamplingStep
from frp_workflow.steps.sampling_result import SamplingResult


class FakePort:
    def __init__(self) -> None:
        self.context: MeasureSectionContext | None = None
        self.result = SamplingResult(
            scan_mode="SPLIT",
            keep_spinning=True,
            primary_sample=object(),
            id_sample=object(),
            coords_od=np.array([[1.0, 2.0, 3.0]]),
            coords_id=np.array([[4.0, 5.0, 6.0]]),
            raw_od="od",
            raw_id="id",
            raw_points=[{"theta_deg": 0.0}],
            split_shift_deg=1.5,
            coax_unreliable=False,
        )

    def _sample_section_impl(self, context: MeasureSectionContext) -> SamplingResult:
        self.context = context
        return self.result


def test_sampling_step_delegates_to_impl() -> None:
    context = MeasureSectionContext(
        section_index=2,
        z_pos_mm=12.5,
        x_abs=34.5,
        centers_xyz=[],
        centers_xyz_id=[],
        concentricity_list=[],
    )
    port = FakePort()
    step = SamplingStep(port)

    result = step.execute(context)

    assert result is port.result
    assert port.context is context
    assert step.name == "sampling"
