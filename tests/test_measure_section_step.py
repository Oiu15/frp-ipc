from __future__ import annotations

from frp_workflow.steps.measure_section import MeasureSectionStep
from frp_workflow.steps.measure_section_context import MeasureSectionContext


class FakePort:
    def __init__(self) -> None:
        self.context: MeasureSectionContext | None = None

    def _measure_section_impl(self, context: MeasureSectionContext) -> str:
        self.context = context
        return "measure-ok"


def test_measure_section_step_delegates_context_and_returns_result() -> None:
    port = FakePort()
    context = MeasureSectionContext(
        section_index=1,
        z_pos_mm=12.5,
        x_abs=101.0,
        centers_xyz=[],
        centers_xyz_id=[],
        concentricity_list=[],
    )
    step = MeasureSectionStep(port)

    result = step.execute(context)

    assert result == "measure-ok"
    assert step.name == "measure_section"
    assert port.context is context
