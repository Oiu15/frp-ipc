# pyright: reportArgumentType=false
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


class _App:
    def __init__(self, *, y10: int = 0, y11: int = 0, confirm: str = "confirm") -> None:
        self.ui_q = queue.Queue()
        self.y = {10: y10, 11: y11}
        self.writes: list[tuple[int, int]] = []
        self.confirm = confirm
        self.confirm_calls = 0

    def get_y_point(self, point: int) -> int:
        return int(self.y.get(point, 0))

    def get_x_point(self, point: int) -> int:
        return 1

    def plc_write_y_point(self, point: int, value: int) -> None:
        self.writes.append((int(point), int(value)))
        self.y[int(point)] = int(value)

    def operator_confirm(self, *args, **kwargs) -> str:
        self.confirm_calls += 1
        return self.confirm


def _flow(app: _App) -> AutoFlow:
    return AutoFlow(app)


def test_prepare_clamps_skips_output_when_already_clamped() -> None:
    app = _App(y10=1, y11=1)
    recipe = Recipe(clamp_confirm_wait_s=3.0)

    assert _flow(app)._prepare_clamps_for_auto(recipe)

    assert app.writes == []
    assert app.confirm_calls == 0


def test_prepare_clamps_closes_both_outputs_and_waits_recipe_seconds() -> None:
    app = _App(y10=0, y11=0)
    recipe = Recipe(clamp_confirm_wait_s=3.0)
    flow = _flow(app)
    waited = []
    flow._sleep_cancelable = lambda seconds: waited.append(seconds) or True  # type: ignore[method-assign]

    assert flow._prepare_clamps_for_auto(recipe)

    assert app.writes == [(10, 1), (11, 1)]
    assert waited == [3.0]


def test_prepare_clamps_zero_wait_confirms_immediately() -> None:
    app = _App(y10=0, y11=0)
    recipe = Recipe(clamp_confirm_wait_s=0.0)

    assert _flow(app)._prepare_clamps_for_auto(recipe)

    assert app.writes == [(10, 1), (11, 1)]
    assert app.confirm_calls == 0


def test_prepare_clamps_minus_one_uses_hardware_confirm() -> None:
    app = _App(y10=0, y11=0, confirm="confirm")
    recipe = Recipe(clamp_confirm_wait_s=-1.0)

    assert _flow(app)._prepare_clamps_for_auto(recipe)

    assert app.confirm_calls == 1


def test_prepare_clamps_minus_one_cancel_stops_flow() -> None:
    app = _App(y10=0, y11=0, confirm="stop")
    recipe = Recipe(clamp_confirm_wait_s=-1.0)

    assert not _flow(app)._prepare_clamps_for_auto(recipe)

    assert app.confirm_calls == 1
