from __future__ import annotations

import sys
import types

import pytest

_pymodbus = types.ModuleType("pymodbus")
_pymodbus_client = types.ModuleType("pymodbus.client")


class _FakeModbusTcpClient:
    pass


setattr(_pymodbus_client, "ModbusTcpClient", _FakeModbusTcpClient)
setattr(_pymodbus, "client", _pymodbus_client)
sys.modules.setdefault("pymodbus", _pymodbus)
sys.modules.setdefault("pymodbus.client", _pymodbus_client)

from core.models import Recipe
from frp_workflow.autoflow_orchestrator import AutoFlowOrchestrator


class _Axis:
    def __init__(self, act_pos: float = 0.0) -> None:
        self.act_pos = act_pos


class _RuntimeApp:
    def __init__(self, *, y10: int = 0, y11: int = 0, confirm: str = "confirm") -> None:
        self.y = {10: int(y10), 11: int(y11)}
        self.writes: list[tuple[int, int]] = []
        self.confirm = confirm
        self.confirm_calls = 0

    def get_y_point(self, point: int) -> int:
        return int(self.y.get(int(point), 0))

    def plc_write_y_point(self, point: int, value: int) -> None:
        self.writes.append((int(point), int(value)))
        self.y[int(point)] = int(value)

    def operator_confirm(self, *args, **kwargs) -> str:
        self.confirm_calls += 1
        return self.confirm


class _Gateway:
    def __init__(self, ax2_pos: float = 0.0) -> None:
        self.axis = _Axis(ax2_pos)

    def get_axis_copy(self, axis: int) -> _Axis:
        return self.axis


class _Host:
    _prepare_ax2_and_clamps = AutoFlowOrchestrator._prepare_ax2_and_clamps
    _move_ax2_to_rotate_position = AutoFlowOrchestrator._move_ax2_to_rotate_position
    _verify_ax2_rotate_position_when_length_disabled = AutoFlowOrchestrator._verify_ax2_rotate_position_when_length_disabled
    _operator_confirm_or_stop = AutoFlowOrchestrator._operator_confirm_or_stop
    _write_y_point = AutoFlowOrchestrator._write_y_point
    _read_y_point = AutoFlowOrchestrator._read_y_point
    _clamps_are_closed = AutoFlowOrchestrator._clamps_are_closed

    def __init__(self, recipe: Recipe, app: _RuntimeApp, gateway: _Gateway | None = None) -> None:
        self.recipe = recipe
        self._runtime_app = app
        self.gateway = gateway or _Gateway()
        self.states: list[tuple[str, str]] = []
        self.waits: list[float] = []
        self.moves: list[tuple] = []

    def _ensure_axis_ready(self, axis: int) -> None:
        pass

    def _emit_state(self, state: str, message: str) -> None:
        self.states.append((str(state), str(message)))

    def _wait_cancelable(self, seconds: float) -> None:
        self.waits.append(float(seconds))

    def _raise_if_stop_requested(self) -> None:
        pass

    def _move_axis_abs(self, *args, **kwargs) -> None:
        self.moves.append((args, kwargs))


def test_orchestrator_skips_clamp_output_when_already_closed() -> None:
    app = _RuntimeApp(y10=1, y11=1)
    host = _Host(Recipe(clamp_confirm_wait_s=3.0), app)

    host._prepare_ax2_and_clamps()

    assert app.writes == []
    assert app.confirm_calls == 0
    assert host.waits == []


def test_orchestrator_closes_dual_clamps_and_auto_waits() -> None:
    app = _RuntimeApp(y10=0, y11=0)
    host = _Host(Recipe(clamp_confirm_wait_s=3.0), app)

    host._prepare_ax2_and_clamps()

    assert app.writes == [(10, 1), (11, 1)]
    assert host.waits == [3.0]
    assert app.confirm_calls == 0


def test_orchestrator_minus_one_waits_for_operator_confirm() -> None:
    app = _RuntimeApp(y10=0, y11=0, confirm="confirm")
    host = _Host(Recipe(clamp_confirm_wait_s=-1.0), app)

    host._prepare_ax2_and_clamps()

    assert app.confirm_calls == 1


def test_orchestrator_minus_one_cancel_stops() -> None:
    app = _RuntimeApp(y10=0, y11=0, confirm="stop")
    host = _Host(Recipe(clamp_confirm_wait_s=-1.0), app)

    with pytest.raises(RuntimeError, match="Operator canceled"):
        host._prepare_ax2_and_clamps()


def test_len_disabled_verifies_ax2_without_move() -> None:
    app = _RuntimeApp(confirm="stop")
    recipe = Recipe(len_enable=False, ax2_rot_valid=True, ax2_rot_abs=100.0)
    host = _Host(recipe, app, _Gateway(ax2_pos=105.0))

    host._move_ax2_to_rotate_position()

    assert host.moves == []
    assert app.confirm_calls == 0


def test_len_disabled_ax2_deviation_uses_operator_choice() -> None:
    app = _RuntimeApp(confirm="stop")
    recipe = Recipe(len_enable=False, ax2_rot_valid=True, ax2_rot_abs=100.0)
    host = _Host(recipe, app, _Gateway(ax2_pos=120.0))

    with pytest.raises(RuntimeError, match="Operator canceled"):
        host._move_ax2_to_rotate_position()

    assert host.moves == []
    assert app.confirm_calls == 1
