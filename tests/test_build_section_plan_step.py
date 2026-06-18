from __future__ import annotations

from frp_workflow.steps.build_section_plan import BuildSectionPlanStep


class FakePort:
    def __init__(self) -> None:
        self.called = False
        self.result = object()

    def _build_section_plan_impl(self) -> object:
        self.called = True
        return self.result


def test_build_section_plan_step_delegates_to_impl_and_returns_result() -> None:
    port = FakePort()
    step = BuildSectionPlanStep(port)

    result = step.execute()

    assert result is port.result
    assert port.called is True
    assert step.name == "build_section_plan"
