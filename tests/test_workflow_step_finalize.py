from __future__ import annotations

from frp_workflow.steps.finalize_run import FinalizeRunStep


class FakePort:
    def __init__(self) -> None:
        self.called = False
        self.calls: list[tuple[str, str]] = []

    def _finalize_run_impl(self, status: str, message: str) -> str:
        self.called = True
        self.calls.append((status, message))
        return "ok"


def test_finalize_run_step_delegates_to_impl() -> None:
    port = FakePort()
    step = FinalizeRunStep(port, "DONE", "complete")

    result = step.execute()

    assert result == "ok"
    assert port.called is True
    assert port.calls == [("DONE", "complete")]
    assert step.name == "finalize_run"
