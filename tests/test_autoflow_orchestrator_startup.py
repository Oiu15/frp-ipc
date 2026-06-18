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
        gw = gateway or _Gateway()
        self.gateway = gw
        self.motion = gw
        self.sensors = app  # type: ignore[assignment]
        self.operator = app  # type: ignore[assignment]
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


# ===================================================================
# _LegacyAppAdapter — PlcCommandPort methods raise on missing impl
# ===================================================================


def test_legacy_adapter_base_raises_if_motion_port_lacks_it() -> None:
    from frp_workflow.autoflow_orchestrator import _LegacyAppAdapter

    class _MinimalMotion:
        pass

    adapter = _LegacyAppAdapter(_MinimalMotion(), _MinimalMotion(), _MinimalMotion())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="MotionPort does not provide PlcCommandPort"):
        adapter.base_for_axis(0)


def test_legacy_adapter_write_regs_raises_if_motion_port_lacks_it() -> None:
    from frp_workflow.autoflow_orchestrator import _LegacyAppAdapter

    class _MinimalMotion:
        pass

    adapter = _LegacyAppAdapter(_MinimalMotion(), _MinimalMotion(), _MinimalMotion())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="MotionPort does not provide PlcCommandPort"):
        adapter.write_regs(0, [])


def test_legacy_adapter_set_cmd_bits_raises_if_motion_port_lacks_it() -> None:
    from frp_workflow.autoflow_orchestrator import _LegacyAppAdapter

    class _MinimalMotion:
        pass

    adapter = _LegacyAppAdapter(_MinimalMotion(), _MinimalMotion(), _MinimalMotion())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="MotionPort does not provide PlcCommandPort"):
        adapter.set_cmd_bits(0, set_mask=1)


def test_legacy_adapter_pulse_cmd_bits_raises_if_motion_port_lacks_it() -> None:
    from frp_workflow.autoflow_orchestrator import _LegacyAppAdapter

    class _MinimalMotion:
        pass

    adapter = _LegacyAppAdapter(_MinimalMotion(), _MinimalMotion(), _MinimalMotion())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="MotionPort does not provide PlcCommandPort"):
        adapter.pulse_cmd_bits(0, 1)


def test_legacy_adapter_plc_command_port_proxies_to_motion_port() -> None:
    from frp_workflow.autoflow_orchestrator import _LegacyAppAdapter

    class _MotionWithPlc:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def base_for_axis(self, axis: int) -> int:
            self.calls.append(("base_for_axis", axis))
            return 100

        def write_regs(self, d_addr: int, values: list[int]) -> None:
            self.calls.append(("write_regs", d_addr, values))

        def set_cmd_bits(self, axis: int, set_mask: int = 0, clr_mask: int = 0) -> None:
            self.calls.append(("set_cmd_bits", axis, set_mask, clr_mask))

        def pulse_cmd_bits(self, axis: int, pulse_mask: int, pulse_ms: int = 120) -> None:
            self.calls.append(("pulse_cmd_bits", axis, pulse_mask, pulse_ms))

        def start_velocity_move(
            self,
            axis: int,
            velocity: float,
            *,
            acc: float = 80.0,
            dec: float = 80.0,
            jerk: float = 300.0,
        ) -> None:
            self.calls.append(("start_velocity_move", axis, velocity, acc, dec, jerk))

        def get_ax0_z_disp_limits(self) -> tuple[float, float, float]:
            self.calls.append(("get_ax0_z_disp_limits",))
            return (-50.0, 500.0, 550.0)

    motion = _MotionWithPlc()
    adapter = _LegacyAppAdapter(motion, motion, motion)  # type: ignore[arg-type]

    assert adapter.base_for_axis(0) == 100
    adapter.write_regs(10, [1, 2])
    adapter.set_cmd_bits(1, set_mask=2, clr_mask=4)
    adapter.pulse_cmd_bits(2, 8, pulse_ms=60)
    adapter.start_velocity_move(0, 50.0, acc=100.0, dec=100.0, jerk=200.0)
    assert adapter.get_ax0_z_disp_limits() == (-50.0, 500.0, 550.0)

    assert motion.calls == [
        ("base_for_axis", 0),
        ("write_regs", 10, [1, 2]),
        ("set_cmd_bits", 1, 2, 4),
        ("pulse_cmd_bits", 2, 8, 60),
        ("start_velocity_move", 0, 50.0, 100.0, 100.0, 200.0),
        ("get_ax0_z_disp_limits",),
    ]


def test_legacy_adapter_velmove_start_axis_raises_if_motion_port_lacks_it() -> None:
    from frp_workflow.autoflow_orchestrator import _LegacyAppAdapter

    class _MinimalMotion:
        pass

    adapter = _LegacyAppAdapter(_MinimalMotion(), _MinimalMotion(), _MinimalMotion())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="MotionPort does not provide PlcCommandPort"):
        adapter.start_velocity_move(0, 50.0)


def test_legacy_adapter_get_ax0_z_disp_limits_raises_if_motion_port_lacks_it() -> None:
    from frp_workflow.autoflow_orchestrator import _LegacyAppAdapter

    class _MinimalMotion:
        pass

    adapter = _LegacyAppAdapter(_MinimalMotion(), _MinimalMotion(), _MinimalMotion())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="MotionPort does not provide PlcCommandPort"):
        adapter.get_ax0_z_disp_limits()
