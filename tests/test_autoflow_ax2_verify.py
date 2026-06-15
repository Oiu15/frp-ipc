# pyright: reportArgumentType=false
import queue

import pytest

from core.models import Recipe
from frp_workflow.autoflow_executor import AutoFlow
from tests.fakes import NoOpEventSink


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
    flow = AutoFlow(device=object(), event_sink=NoOpEventSink(), motion=app, sensors=app, operator=app, plc=app)
    flow.device = _Device(act_pos)  # type: ignore[assignment]
    return flow


# ---------------------------------------------------------------------------
# table-driven — 5 verification scenarios
# ---------------------------------------------------------------------------

_VERIFY_CASES = [
    # (len_enable, ax2_rot_valid, ax2_rot_abs, act_pos, confirm, expected_return, expected_confirm)
    (False, True,  100.0, 108.0, "confirm", True,  0),
    (False, True,  100.0, 111.0, "confirm", True,  1),
    (False, True,  100.0, 111.0, "stop",    False, 1),
    (False, False,   0.0,   0.0, "confirm", True,  1),
    (True,  False,   0.0, 999.0, "stop",    True,  0),
]


@pytest.mark.parametrize(
    ("len_enable", "ax2_rot_valid", "ax2_rot_abs", "act_pos", "confirm",
     "expected_return", "expected_confirm"),
    _VERIFY_CASES,
)
def test_verify_ax2_when_length_disabled(
    len_enable: bool,
    ax2_rot_valid: bool,
    ax2_rot_abs: float,
    act_pos: float,
    confirm: str,
    expected_return: bool,
    expected_confirm: int,
) -> None:
    app = _App(confirm=confirm)
    recipe = Recipe(
        len_enable=len_enable,
        ax2_rot_valid=ax2_rot_valid,
        ax2_rot_abs=ax2_rot_abs,
    )
    result = _flow(app, act_pos=act_pos)._verify_ax2_when_length_disabled(recipe)

    assert result is expected_return
    assert app.confirm_calls == expected_confirm
