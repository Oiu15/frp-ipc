from __future__ import annotations

from frp_workflow.steps.prepare_run_context import PrepareRunContextStep


class FakePort:
    def __init__(self) -> None:
        self.called = False

    def _prepare_run_context_impl(self) -> str:
        self.called = True
        return "ok"


def test_prepare_run_context_step_delegates_to_impl() -> None:
    port = FakePort()
    step = PrepareRunContextStep(port)

    result = step.execute()

    assert result == "ok"
    assert port.called is True
    assert step.name == "prepare_run_context"
