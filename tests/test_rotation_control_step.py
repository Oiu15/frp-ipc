from __future__ import annotations

from typing import Any, cast

import pytest

from frp_workflow.autoflow_orchestrator import AutoFlowOrchestrator
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


def _minimal_orchestrator() -> Any:
    return cast(Any, AutoFlowOrchestrator.__new__(AutoFlowOrchestrator))


def test_restart_rotation_for_split_stops_before_starting(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = _minimal_orchestrator()
    calls: list[object] = []

    def stop() -> None:
        calls.append("stop")

    def start(*, emit_state: bool) -> None:
        calls.append(("start", emit_state))

    monkeypatch.setattr(orch, "_stop_ax3_rotation", stop)
    monkeypatch.setattr(orch, "_start_ax3_rotation", start)

    orch._restart_rotation_for_split_impl()

    assert calls == ["stop", ("start", False)]


def test_restart_rotation_for_split_still_starts_when_stop_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = _minimal_orchestrator()
    calls: list[object] = []

    def stop() -> None:
        calls.append("stop")
        raise RuntimeError("stop failed")

    def start(*, emit_state: bool) -> None:
        calls.append(("start", emit_state))

    monkeypatch.setattr(orch, "_stop_ax3_rotation", stop)
    monkeypatch.setattr(orch, "_start_ax3_rotation", start)

    orch._restart_rotation_for_split_impl()

    assert calls == ["stop", ("start", False)]


def test_restart_rotation_for_split_swallows_start_error(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = _minimal_orchestrator()
    calls: list[object] = []

    def stop() -> None:
        calls.append("stop")

    def start(*, emit_state: bool) -> None:
        calls.append(("start", emit_state))
        raise RuntimeError("start failed")

    monkeypatch.setattr(orch, "_stop_ax3_rotation", stop)
    monkeypatch.setattr(orch, "_start_ax3_rotation", start)

    orch._restart_rotation_for_split_impl()

    assert calls == ["stop", ("start", False)]
