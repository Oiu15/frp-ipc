from __future__ import annotations

from frp_workflow.steps.rotation_control import RotationControlStep


class FakePort:
    def __init__(self) -> None:
        self.called = 0

    def _restart_rotation_for_split_impl(self) -> None:
        self.called += 1


def test_rotation_control_step_delegates_restart_for_split() -> None:
    port = FakePort()
    step = RotationControlStep(port)

    step.restart_for_split()

    assert port.called == 1
    assert step.name == "rotation_control"
