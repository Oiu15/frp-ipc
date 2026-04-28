import queue
import sys
import types

_pymodbus = types.ModuleType("pymodbus")
_pymodbus_client = types.ModuleType("pymodbus.client")


class _FakeModbusTcpClient:
    pass


setattr(_pymodbus_client, "ModbusTcpClient", _FakeModbusTcpClient)
setattr(_pymodbus, "client", _pymodbus_client)
sys.modules.setdefault("pymodbus", _pymodbus)
sys.modules.setdefault("pymodbus.client", _pymodbus_client)

from core.models import Recipe
from services.autoflow_service import AutoFlow


class _Axis:
    def __init__(self, act_pos: float) -> None:
        self.act_pos = act_pos


class _Device:
    def __init__(self, act_pos: float) -> None:
        self.axis = _Axis(act_pos)
        self.movea_calls = 0

    def get_axis_copy(self, axis: int) -> _Axis:
        return self.axis

    def apply_soft_limits_abs(self, *args, **kwargs) -> float:
        self.movea_calls += 1
        return float(args[1])


class _App:
    def __init__(self, confirm: str = "confirm") -> None:
        self.ui_q = queue.Queue()
        self.confirm = confirm
        self.confirm_calls = 0

    def get_x_point(self, point: int) -> int:
        return 1

    def operator_confirm(self, *args, **kwargs) -> str:
        self.confirm_calls += 1
        return self.confirm


def _flow(app: _App, act_pos: float) -> AutoFlow:
    flow = AutoFlow(app)
    flow.device = _Device(act_pos)  # type: ignore[assignment]
    return flow


def test_len_disabled_ax2_within_tolerance_continues_without_confirm() -> None:
    app = _App()
    recipe = Recipe(len_enable=False, ax2_rot_valid=True, ax2_rot_abs=100.0)

    assert _flow(app, act_pos=108.0)._verify_ax2_when_length_disabled(recipe)

    assert app.confirm_calls == 0


def test_len_disabled_ax2_over_tolerance_can_continue_by_x3() -> None:
    app = _App(confirm="confirm")
    recipe = Recipe(len_enable=False, ax2_rot_valid=True, ax2_rot_abs=100.0)

    assert _flow(app, act_pos=111.0)._verify_ax2_when_length_disabled(recipe)

    assert app.confirm_calls == 1


def test_len_disabled_ax2_over_tolerance_can_cancel_by_x4() -> None:
    app = _App(confirm="stop")
    recipe = Recipe(len_enable=False, ax2_rot_valid=True, ax2_rot_abs=100.0)

    assert not _flow(app, act_pos=111.0)._verify_ax2_when_length_disabled(recipe)

    assert app.confirm_calls == 1


def test_len_disabled_missing_ax2_rot_position_requires_confirm() -> None:
    app = _App(confirm="confirm")
    recipe = Recipe(len_enable=False, ax2_rot_valid=False)

    assert _flow(app, act_pos=0.0)._verify_ax2_when_length_disabled(recipe)

    assert app.confirm_calls == 1


def test_len_enabled_keeps_existing_move_path_outside_safety_check() -> None:
    app = _App(confirm="stop")
    recipe = Recipe(len_enable=True, ax2_rot_valid=False)

    assert _flow(app, act_pos=999.0)._verify_ax2_when_length_disabled(recipe)

    assert app.confirm_calls == 0
