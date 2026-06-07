from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from application.adapters.device_gateway import AppDeviceGateway
from machine.validation_gateway import ValidationActionCancelled, ValidationActionGateway
from core.models import AxisCal

if TYPE_CHECKING:  # pragma: no cover
    from app import App


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _FakeApp:
    def __init__(self) -> None:
        self.stopped_axes: list[int] = []
        self.y_writes: list[tuple[int, int]] = []
        self.movea_calls: list[tuple[int, float, str]] = []
        self.limit_calls: list[tuple[int, float, bool, str]] = []
        self.axis_positions: dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0, 4: 0.0}
        self.axis_position_reads: dict[int, list[float]] = {}
        self.validation_cancel_requested = False
        self.axis_cal = AxisCal()
        self._plc_poll_profile_req = ""

    def get_axis_copy(self, axis: int):
        ax = int(axis)
        reads = self.axis_position_reads.get(ax)
        if reads:
            if len(reads) > 1:
                value = reads.pop(0)
            else:
                value = reads[0]
        else:
            value = self.axis_positions.get(ax, 0.0)
        return SimpleNamespace(act_pos=value, softlim_pos=0.0, softlim_neg=0.0)

    def apply_soft_limits_abs(self, axis: int, target_abs: float, *, strict: bool = False, context: str = "") -> float:
        self.limit_calls.append((int(axis), float(target_abs), bool(strict), str(context)))
        return float(target_abs)

    def movea_abs(self, axis: int, pos_abs: float, *, context: str = "MoveA") -> None:
        self.movea_calls.append((int(axis), float(pos_abs), str(context)))

    def stop(self, axis: int) -> None:
        self.stopped_axes.append(int(axis))

    def plc_write_y_point(self, y_point: int, value: int) -> None:
        self.y_writes.append((int(y_point), int(value)))

    def is_validation_cancel_requested(self) -> bool:
        return bool(self.validation_cancel_requested)


def _gw(app: _FakeApp | None = None) -> AppDeviceGateway:
    return AppDeviceGateway(cast("App", app or _FakeApp()))


# ---------------------------------------------------------------------------
# protocol / single-action tests (not parametrized — unique assertion shapes)
# ---------------------------------------------------------------------------


class TestAppDeviceGateway:
    """AppDeviceGateway — validation action delegation and protocol checks."""

    def test_implements_validation_action_gateway_protocol(self) -> None:
        gateway = _gw()
        assert isinstance(gateway, ValidationActionGateway)

    def test_stop_rotation_targets_ax3(self) -> None:
        app = _FakeApp()
        _gw(app).stop_rotation()
        assert app.stopped_axes == [3]

    def test_read_axis_position_mm_reads_latest_axis_snapshot(self) -> None:
        app = _FakeApp()
        app.axis_positions[2] = 123.456
        assert _gw(app).read_axis_position_mm(2) == 123.456

    # ------------------------------------------------------------------
    # clamp methods — dual-clamp output is identical for both APIs
    # ------------------------------------------------------------------

    _CLAMP_CASES = [
        ("validate", ["clamp_release", "clamp_close"]),
        ("legacy",   ["open_dual_clamps", "close_dual_clamps"]),
    ]

    @pytest.mark.parametrize(("label", "methods"), _CLAMP_CASES)
    def test_clamp_methods_write_dual_clamp_outputs(self, label: str, methods: list[str]) -> None:
        app = _FakeApp()
        gateway = _gw(app)
        for method in methods:
            getattr(gateway, method)()
        assert app.y_writes == [(10, 0), (11, 0), (10, 1), (11, 1)]

    # ------------------------------------------------------------------
    # wait_cancelable — normal / callback-cancel / app-cancel
    # ------------------------------------------------------------------

    _WAIT_CANCELABLE_CASES = [
        # (cancel_callback, app_cancel, should_raise)
        (None,        False, False),  # normal
        (lambda: True, False, True),   # callback cancel
        (None,        True,  True),   # app-level cancel
    ]

    @pytest.mark.parametrize(
        ("cancel_callback", "app_cancel", "should_raise"), _WAIT_CANCELABLE_CASES
    )
    def test_wait_cancelable(
        self,
        cancel_callback: Callable[[], bool] | None,
        app_cancel: bool,
        should_raise: bool,
    ) -> None:
        app = _FakeApp()
        if app_cancel:
            app.validation_cancel_requested = True

        gateway = _gw(app)

        if should_raise:
            with pytest.raises(ValidationActionCancelled):
                gateway.wait_cancelable(
                    0.5, poll_interval_s=0.001, cancel_check=cancel_callback
                )
        else:
            gateway.wait_cancelable(0.001, poll_interval_s=0.001)

    # ------------------------------------------------------------------
    # move delegation — single / relative / multi-axis all share pattern
    # ------------------------------------------------------------------

    _MOVE_CASES = [
        # (method, kwargs, expected_limit_calls, expected_movea_calls, expected_return)
        (
            "move_axis_absolute",
            {"axis": 0, "target_pos_mm": 125.5, "context": "VALIDATION_MOVE_AWAY"},
            [(0, 125.5, False, "VALIDATION_MOVE_AWAY")],
            [(0, 125.5, "VALIDATION_MOVE_AWAY")],
            125.5,
        ),
        (
            "move_axis_relative",
            {"axis": 0, "delta_mm": -12.5, "context": "VALIDATION_MOVE_AWAY"},
            [(0, 87.5, False, "VALIDATION_MOVE_AWAY")],
            [(0, 87.5, "VALIDATION_MOVE_AWAY")],
            87.5,
        ),
        (
            "move_axes_absolute",
            {"targets_abs": {1: 25.0, 4: 75.0}, "context": "VALIDATION_MOVE_AWAY"},
            [(1, 25.0, False, "VALIDATION_MOVE_AWAY"), (4, 75.0, False, "VALIDATION_MOVE_AWAY")],
            [(1, 25.0, "VALIDATION_MOVE_AWAY"), (4, 75.0, "VALIDATION_MOVE_AWAY")],
            {1: 25.0, 4: 75.0},
        ),
    ]

    @pytest.mark.parametrize(
        ("method", "kwargs", "expected_limit_calls", "expected_movea_calls", "expected_return"),
        _MOVE_CASES,
    )
    def test_move_delegation(
        self,
        method: str,
        kwargs: dict,
        expected_limit_calls: list,
        expected_movea_calls: list,
        expected_return: object,
    ) -> None:
        app = _FakeApp()
        app.axis_positions[0] = 100.0  # needed for relative move

        gateway = _gw(app)
        result = getattr(gateway, method)(**kwargs)

        assert result == expected_return
        assert app.limit_calls == expected_limit_calls
        assert app.movea_calls == expected_movea_calls

    # ------------------------------------------------------------------
    # wait_axis_in_position — success / cancel / timeout-logging
    # ------------------------------------------------------------------

    def test_wait_axis_in_position_returns_when_position_reaches_tolerance(self) -> None:
        app = _FakeApp()
        app.axis_position_reads[0] = [95.0, 99.95]
        actual = _gw(app).wait_axis_in_position(0, 100.0, tolerance_mm=0.1, timeout_s=0.1, poll_interval_s=0.001)
        assert actual == 99.95

    def test_wait_axis_in_position_is_cancel_aware(self) -> None:
        app = _FakeApp()
        app.axis_positions[0] = 95.0
        with pytest.raises(ValidationActionCancelled):
            _gw(app).wait_axis_in_position(0, 100.0, timeout_s=1.0, poll_interval_s=0.001, cancel_check=lambda: True)

    def test_wait_axis_in_position_logs_timeout_source_and_poll_profile(self) -> None:
        app = _FakeApp()
        app.axis_positions[0] = 95.0
        app._plc_poll_profile_req = "sampling"
        gateway = _gw(app)

        with patch("application.adapters.device_gateway.log") as mock_log:
            with pytest.raises(TimeoutError):
                gateway.wait_axis_in_position(0, 100.0, timeout_s=0.0, poll_interval_s=0.001)

        mock_log.assert_called_once_with(
            "VALIDATION_WAIT_INPOS_TIMEOUT",
            axis=0,
            target=100.0,
            actual=95.0,
            timeout_s=0.0,
            tolerance=0.1,
            actual_source="axis_snapshot",
            current_poll_profile="sampling",
        )
