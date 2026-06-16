# pyright: reportArgumentType=false
import queue

import pytest

from core.models import Recipe
from frp_workflow.autoflow_executor import AutoFlow
from tests.fakes import NoOpEventSink


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
    return AutoFlow(device=object(), event_sink=NoOpEventSink(), motion=app, sensors=app, operator=app, plc=app)


# ---------------------------------------------------------------------------
# table-driven — 5 clamp scenarios
# ---------------------------------------------------------------------------


_PREPARE_CLAMP_CASES = [
    # (y10, y11, wait_s, confirm, expected_writes, expected_waits, expected_confirm, expected_return)
    (1,    1,    3.0,  "confirm", [],                 [],     0,  True),
    (0,    0,    3.0,  "confirm", [(10, 1), (11, 1)], [3.0],  0,  True),
    (0,    0,    0.0,  "confirm", [(10, 1), (11, 1)], [],     0,  True),
    (0,    0,   -1.0,  "confirm", [(10, 1), (11, 1)], [],     1,  True),
    (0,    0,   -1.0,  "stop",    [(10, 1), (11, 1)], [],     1,  False),
]


@pytest.mark.parametrize(
    ("y10", "y11", "wait_s", "confirm", "expected_writes", "expected_waits",
     "expected_confirm", "expected_return"),
    _PREPARE_CLAMP_CASES,
)
def test_prepare_clamps_for_auto(
    y10: int,
    y11: int,
    wait_s: float,
    confirm: str,
    expected_writes: list,
    expected_waits: list,
    expected_confirm: int,
    expected_return: bool,
) -> None:
    app = _App(y10=y10, y11=y11, confirm=confirm)
    recipe = Recipe(clamp_confirm_wait_s=wait_s)
    flow = _flow(app)

    # Only the "auto-wait" scenario needs the sleep mock
    waited: list[float] = []
    if wait_s > 0:
        flow._sleep_cancelable = lambda seconds: waited.append(seconds) or True  # type: ignore[method-assign]

    assert flow._prepare_clamps_for_auto(recipe) is expected_return

    assert app.writes == expected_writes
    assert waited == expected_waits
    assert app.confirm_calls == expected_confirm
