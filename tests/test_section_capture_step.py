from __future__ import annotations

from frp_workflow.steps.section_capture import SectionCaptureStep
from frp_workflow.steps.section_context import SectionExecutionContext


class FakePort:
    def __init__(self) -> None:
        self.context: SectionExecutionContext | None = None

    def _capture_section_impl(self, context: SectionExecutionContext) -> str:
        self.context = context
        return "capture-ok"


def test_section_capture_step_delegates_context_and_returns_result() -> None:
    port = FakePort()
    context = SectionExecutionContext(
        section=object(),
        section_index=1,
        total_sections=3,
        centers_xyz=[],
        centers_xyz_id=[],
        concentricity_list=[],
    )
    step = SectionCaptureStep(port)

    result = step.execute(context)

    assert result == "capture-ok"
    assert step.name == "section_capture"
    assert port.context is context
