from __future__ import annotations

from frp_workflow.steps.section_context import SectionExecutionContext
from frp_workflow.steps.section_execution import SectionExecutionStep


class FakePort:
    def __init__(self) -> None:
        self.context: SectionExecutionContext | None = None
        self.result = object()

    def _execute_section_impl(self, context: SectionExecutionContext) -> object:
        self.context = context
        return self.result


def test_section_execution_step_delegates_context_and_returns_result() -> None:
    port = FakePort()
    section = object()
    centers_xyz = [(1.0, 2.0, 3.0)]
    centers_xyz_id = [(4.0, 5.0, 6.0)]
    concentricity_list = [0.12]
    context = SectionExecutionContext(
        section=section,
        section_index=3,
        total_sections=7,
        centers_xyz=centers_xyz,
        centers_xyz_id=centers_xyz_id,
        concentricity_list=concentricity_list,
    )
    step = SectionExecutionStep(port)

    result = step.execute(context)

    assert result is port.result
    assert step.name == "section_execution"
    assert port.context is context
    captured = port.context
    assert captured is not None
    assert captured.section is section
    assert captured.section_index == 3
    assert captured.total_sections == 7
    assert captured.centers_xyz is centers_xyz
    assert captured.centers_xyz_id is centers_xyz_id
    assert captured.concentricity_list is concentricity_list
