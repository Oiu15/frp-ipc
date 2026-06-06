from __future__ import annotations

from typing import Any

import pytest

from core.models import Recipe
from frp_workflow.autoflow_orchestrator import AutoFlowOrchestrator


# ---------------------------------------------------------------------------
# test doubles
# ---------------------------------------------------------------------------


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

    def operator_confirm(self, *args: Any, **kwargs: Any) -> str:
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

    def apply_soft_limits_abs(
        self, axis: int, target_abs: float, *, strict: bool = False, context: str = ""
    ) -> float:
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
        self._runtime_host = app
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

    def _move_axis_abs(self, *args: Any, **kwargs: Any) -> None:
        self.moves.append((args, kwargs))

    def _is_fault(self, sts: int, err: int) -> bool:
        return int(err) != 0

    def _is_moving(self, sts: int) -> bool:
        return False


# ---------------------------------------------------------------------------
# _prepare_ax2_and_clamps — 4 scenarios
# ---------------------------------------------------------------------------

_PREPARE_CASES = [
    # (y10, y11, wait_s, confirm, expected_writes, expected_waits, expected_confirm_calls, raises_msg)
    (1,    1,    3.0,  "confirm",  [],                       [],       0,  None),
    (0,    0,    3.0,  "confirm",  [(10, 1), (11, 1)],       [3.0],    0,  None),
    (0,    0,   -1.0,  "confirm",  [(10, 1), (11, 1)],       [],       1,  None),
    (0,    0,   -1.0,  "stop",     [(10, 1), (11, 1)],       [],       1,  "Operator canceled"),
]


@pytest.mark.parametrize(
    ("y10", "y11", "wait_s", "confirm", "expected_writes", "expected_waits", "expected_confirm_calls", "raises_msg"),
    _PREPARE_CASES,
)
def test_prepare_ax2_and_clamps(
    y10: int,
    y11: int,
    wait_s: float,
    confirm: str,
    expected_writes: list,
    expected_waits: list,
    expected_confirm_calls: int,
    raises_msg: str | None,
) -> None:
    app = _RuntimeApp(y10=y10, y11=y11, confirm=confirm)
    host = _Host(Recipe(clamp_confirm_wait_s=wait_s), app)

    if raises_msg is not None:
        with pytest.raises(RuntimeError, match=raises_msg):
            host._prepare_ax2_and_clamps()
    else:
        host._prepare_ax2_and_clamps()

    assert app.writes == expected_writes
    assert host.waits == expected_waits
    assert app.confirm_calls == expected_confirm_calls


# ---------------------------------------------------------------------------
# _move_ax2_to_rotate_position (len_enable=False) — 2 scenarios
# ---------------------------------------------------------------------------

_VERIFY_AX2_CASES = [
    # (ax2_pos, confirm, expected_moves, expected_confirm_calls, raises_msg)
    (105.0, "stop",   [],  0,  None),
    (120.0, "stop",   [],  1,  "Operator canceled"),
]


@pytest.mark.parametrize(
    ("ax2_pos", "confirm", "expected_moves", "expected_confirm_calls", "raises_msg"),
    _VERIFY_AX2_CASES,
)
def test_len_disabled_ax2_verify(
    ax2_pos: float,
    confirm: str,
    expected_moves: list,
    expected_confirm_calls: int,
    raises_msg: str | None,
) -> None:
    app = _RuntimeApp(confirm=confirm)
    recipe = Recipe(len_enable=False, ax2_rot_valid=True, ax2_rot_abs=100.0)
    host = _Host(recipe, app, _Gateway(ax2_pos=ax2_pos))

    if raises_msg is not None:
        with pytest.raises(RuntimeError, match=raises_msg):
            host._move_ax2_to_rotate_position()
    else:
        host._move_ax2_to_rotate_position()

    assert host.moves == expected_moves
    assert app.confirm_calls == expected_confirm_calls


# ---------------------------------------------------------------------------
# _return_to_standby_after_user_stop
# ---------------------------------------------------------------------------


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
