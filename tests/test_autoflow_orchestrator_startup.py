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
        self.sts = 1
        self.err = 0


class _RuntimeApp:
    def __init__(self, *, y10: int = 0, y11: int = 0, confirm: str = "confirm") -> None:
        self.y = {10: int(y10), 11: int(y11)}
        self.writes: list[tuple[int, int]] = []
        self.confirm = confirm
        self.confirm_calls = 0

    def get_y_point(self, point: int) -> int:
        return int(self.y.get(int(point), 0))

    def get_x_point(self, point: int) -> int:
        return 1

    def plc_write_y_point(self, point: int, value: int) -> None:
        self.writes.append((int(point), int(value)))
        self.y[int(point)] = int(value)

    def operator_confirm(self, *args, **kwargs) -> str:
        self.confirm_calls += 1
        return self.confirm


class _Gateway:
    def __init__(self, ax2_pos: float = 0.0) -> None:
        self.axes = {axis: _Axis(0.0) for axis in (0, 1, 2, 4)}
        self.axes[2].act_pos = float(ax2_pos)
        self.moves: list[tuple[int, float, str]] = []
        self.applied: list[tuple[int, float, bool, str]] = []

    def get_axis_copy(self, axis: int) -> _Axis:
        return self.axes.setdefault(int(axis), _Axis())

    def apply_soft_limits_abs(self, axis: int, target_abs: float, *, strict: bool = False, context: str = "") -> float:
        self.applied.append((int(axis), float(target_abs), bool(strict), str(context)))
        return float(target_abs)

    def movea_abs(self, axis: int, pos_abs: float, *, context: str = "MoveA") -> None:
        self.moves.append((int(axis), float(pos_abs), str(context)))
        self.axes.setdefault(int(axis), _Axis()).act_pos = float(pos_abs)


class _Host:
    _prepare_ax2_and_clamps = AutoFlowOrchestrator._prepare_ax2_and_clamps
    _move_ax2_to_rotate_position = AutoFlowOrchestrator._move_ax2_to_rotate_position
    _verify_ax2_rotate_position_when_length_disabled = AutoFlowOrchestrator._verify_ax2_rotate_position_when_length_disabled
    _operator_confirm_or_stop = AutoFlowOrchestrator._operator_confirm_or_stop
    _write_y_point = AutoFlowOrchestrator._write_y_point
    _read_y_point = AutoFlowOrchestrator._read_y_point
    _clamps_are_closed = AutoFlowOrchestrator._clamps_are_closed
    _return_to_standby_after_user_stop = AutoFlowOrchestrator._return_to_standby_after_user_stop
    _wait_in_position_ignoring_user_stop = AutoFlowOrchestrator._wait_in_position_ignoring_user_stop

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

    def _is_fault(self, sts: int, err: int) -> bool:
        return int(err) != 0

    def _is_moving(self, sts: int) -> bool:
        return False


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


def test_user_stop_returns_linear_axes_to_standby() -> None:
    app = _RuntimeApp()
    gateway = _Gateway()
    recipe = Recipe(
        standby_valid=True,
        standby_ax0_abs=10.0,
        standby_ax1_abs=20.0,
        standby_ax4_abs=40.0,
    )
    host = _Host(recipe, app, gateway)

    host._return_to_standby_after_user_stop()

    assert gateway.moves == [
        (1, 20.0, "AUTO_STOP_STANDBY"),
        (4, 40.0, "AUTO_STOP_STANDBY"),
        (0, 10.0, "AUTO_STOP_STANDBY"),
    ]
    assert ("STOPPING", "Return AX0/AX1/AX4 to standby after stop") in host.states
