# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false
"""Contract tests for AppDeviceGateway — net-new coverage beyond test_app_device_gateway.py.

tests/test_app_device_gateway.py already covers:
  - ValidationActionGateway protocol, stop_rotation, clamp methods, wait_cancelable,
    move_axis_absolute/relative/multi-axis, wait_axis_in_position.

This file adds contract-level coverage for the remaining DeviceGateway surface:
  - motion commands: movea_abs, velmove, stop, halt, reset, enable, pulse_cmd_mask, abort_motion
  - sync reads:   read_regs_sync, read_axis_angle_deg_sync, read_cl_sync
  - boundaries:   apply_soft_limits_abs, operator_confirm, is_x3_confirm_pressed,
                  set_plc_poll_profile, write_coil, get_axis_cal, get_soft_limits_abs
"""

from __future__ import annotations

from typing import Any

import pytest

from application.adapters.device_gateway import AppDeviceGateway
from core.models import AxisCal


# ---------------------------------------------------------------------------
# shared contract fake
# ---------------------------------------------------------------------------

class _ContractApp:
    """Fake host that records every delegated call for contract verification."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._return_values: dict[str, Any] = {}
        self.axis_cal = AxisCal()

    def _record(self, method: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append({"method": method, "args": args, "kwargs": dict(kwargs)})

    # -- motion commands --------------------------------------------------

    def movea_abs(self, axis: int, pos_abs: float, *, context: str = "MoveA") -> None:
        self._record("movea_abs", axis, pos_abs, context=context)

    def velmove(
        self, axis: int, velocity: float, *,
        acc: float = 80.0, dec: float = 80.0, jerk: float = 300.0,
    ) -> None:
        self._record("velmove", axis, velocity, acc=acc, dec=dec, jerk=jerk)

    def stop(self, axis: int) -> None:
        self._record("stop", axis)

    def halt(self, axis: int) -> None:
        self._record("halt", axis)

    def reset(self, axis: int) -> None:
        self._record("reset", axis)

    def enable(self, axis: int) -> None:
        self._record("enable", axis)

    def abort_motion(self, axes: Any = None) -> None:
        self._record("abort_motion", axes=axes)

    def pulse_cmd_mask(self, axis: int, pulse_mask: int, pulse_ms: int = 120) -> None:
        self._record("pulse_cmd_mask", axis, pulse_mask, pulse_ms=pulse_ms)

    # -- synchronous reads ------------------------------------------------

    def read_regs_sync(self, d_addr: int, count: int, timeout_s: float = 0.35) -> Any:
        self._record("read_regs_sync", d_addr, count, timeout_s=timeout_s)
        return self._return_values.get("read_regs_sync")

    def read_axis_act_pos_deg_sync(self, axis: int = 3, timeout_s: float = 0.35) -> Any:
        self._record("read_axis_act_pos_deg_sync", axis, timeout_s=timeout_s)
        return self._return_values.get("read_axis_act_pos_deg_sync")

    def read_cl_sync(self, channel: str, *, timeout_s: float = 0.5) -> Any:
        self._record("read_cl_sync", channel, timeout_s=timeout_s)
        return self._return_values.get("read_cl_sync")

    # -- soft limits ------------------------------------------------------

    def apply_soft_limits_abs(
        self, axis: int, target_abs: float, *, strict: bool = False, context: str = "",
    ) -> float:
        self._record("apply_soft_limits_abs", axis, target_abs, strict=strict, context=context)
        return float(target_abs)

    # -- PLC / poll profile / coils --------------------------------------

    def set_plc_poll_profile(self, profile: str = "normal") -> None:
        self._record("set_plc_poll_profile", profile)

    def write_coil(self, coil_addr: int, value: Any) -> None:
        self._record("write_coil", coil_addr, value)

    # -- operator interaction ---------------------------------------------

    def operator_confirm(
        self, title: str, message: str, *,
        allow_stop: bool = True, timeout_s: float | None = None,
    ) -> str:
        self._record("operator_confirm", title, message, allow_stop=allow_stop, timeout_s=timeout_s)
        return self._return_values.get("operator_confirm", "confirmed")

    def get_x_point(self, x_point: int) -> int:
        self._record("get_x_point", x_point)
        return self._return_values.get("get_x_point", 0)

    # -- AxisComm snapshot ------------------------------------------------

    def get_axis_copy(self, axis: int) -> Any:
        self._record("get_axis_copy", axis)
        return self._return_values.get("get_axis_copy")


def _gw(app: _ContractApp | None = None) -> AppDeviceGateway:
    return AppDeviceGateway(app or _ContractApp())  # type: ignore[arg-type]


# ===================================================================
# Motion commands (not covered by existing tests)
# ===================================================================

class TestMotionCommands:
    """movea_abs / velmove / stop / halt / reset / enable / pulse_cmd_mask / abort_motion."""

    _MOTION_METHODS = [
        ("movea_abs",       (0, 125.5),              {"context": "MoveA"},       "movea_abs"),
        ("velmove",         (3, 120.0),              {"acc": 80.0, "dec": 80.0, "jerk": 300.0}, "velmove"),
        ("stop",            (2,),                    {},                         "stop"),
        ("halt",            (4,),                    {},                         "halt"),
        ("reset",           (1,),                    {},                         "reset"),
        ("enable",          (0,),                    {},                         "enable"),
        ("pulse_cmd_mask",  (0, 0x0002),             {"pulse_ms": 120},         "pulse_cmd_mask"),
    ]

    @pytest.mark.parametrize(
        ("gw_method", "gw_args", "gw_kwargs", "expected_host_method"),
        _MOTION_METHODS,
    )
    def test_motion_delegates(
        self, gw_method: str, gw_args: tuple, gw_kwargs: dict, expected_host_method: str,
    ) -> None:
        app = _ContractApp()
        getattr(_gw(app), gw_method)(*gw_args, **gw_kwargs)
        assert len(app.calls) == 1
        assert app.calls[0]["method"] == expected_host_method

    def test_abort_motion_delegates_with_explicit_axes(self) -> None:
        app = _ContractApp()
        _gw(app).abort_motion(axes=[0, 1, 2])
        assert app.calls == [{"method": "abort_motion", "args": (), "kwargs": {"axes": [0, 1, 2]}}]

    def test_abort_motion_defaults_to_none_axes(self) -> None:
        app = _ContractApp()
        _gw(app).abort_motion()
        assert app.calls[0]["method"] == "abort_motion"
        assert app.calls[0]["kwargs"]["axes"] is None


# ===================================================================
# Synchronous reads (not covered by existing tests)
# ===================================================================

class TestSynchronousReads:
    """read_regs_sync / read_axis_angle_deg_sync / read_cl_sync."""

    def test_read_regs_sync_delegates_and_returns_value(self) -> None:
        app = _ContractApp()
        app._return_values["read_regs_sync"] = [100, 200]
        result = _gw(app).read_regs_sync(d_addr=500, count=4, timeout_s=0.5)
        assert result == [100, 200]
        assert app.calls[0] == {
            "method": "read_regs_sync", "args": (500, 4), "kwargs": {"timeout_s": 0.5},
        }

    def test_read_axis_angle_deg_sync_delegates(self) -> None:
        app = _ContractApp()
        app._return_values["read_axis_act_pos_deg_sync"] = 45.0
        result = _gw(app).read_axis_angle_deg_sync(axis=3, timeout_s=0.2)
        assert result == 45.0
        assert app.calls[0]["method"] == "read_axis_act_pos_deg_sync"

    def test_read_axis_angle_deg_sync_default_arguments(self) -> None:
        app = _ContractApp()
        _gw(app).read_axis_angle_deg_sync()
        assert app.calls[0] == {
            "method": "read_axis_act_pos_deg_sync", "args": (3,), "kwargs": {"timeout_s": 0.35},
        }

    @pytest.mark.parametrize("channel", ["out145", "out3"])
    def test_read_cl_sync_delegates_per_channel(self, channel: str) -> None:
        app = _ContractApp()
        app._return_values["read_cl_sync"] = (1.0, 2.0, 3.0, 4.0, {}, {})
        result = _gw(app).read_cl_sync(channel, timeout_s=1.0)  # type: ignore[arg-type]
        assert result is not None
        assert app.calls[0] == {
            "method": "read_cl_sync", "args": (channel,), "kwargs": {"timeout_s": 1.0},
        }


# ===================================================================
# Soft limits (not covered by existing tests)
# ===================================================================

class TestSoftLimits:
    """apply_soft_limits_abs delegates with correct args."""

    @pytest.mark.parametrize("axis,target,strict,context", [
        (0, 150.0, False, "MoveA"),
        (1, -10.0, True,  "ValidationMoveA"),
        (4, 0.0,   False, ""),
    ])
    def test_delegates(
        self, axis: int, target: float, strict: bool, context: str,
    ) -> None:
        app = _ContractApp()
        result = _gw(app).apply_soft_limits_abs(axis, target, strict=strict, context=context)
        assert result == target
        assert app.calls[0] == {
            "method": "apply_soft_limits_abs",
            "args": (axis, target),
            "kwargs": {"strict": strict, "context": context},
        }


# ===================================================================
# Operator confirm (not covered by existing tests)
# ===================================================================

class TestOperatorConfirm:
    """operator_confirm / is_x3_confirm_pressed."""

    def test_operator_confirm_delegates_and_returns_value(self) -> None:
        app = _ContractApp()
        app._return_values["operator_confirm"] = "stop"
        result = _gw(app).operator_confirm("title", "msg", allow_stop=False, timeout_s=30.0)
        assert result == "stop"
        assert app.calls[0] == {
            "method": "operator_confirm",
            "args": ("title", "msg"),
            "kwargs": {"allow_stop": False, "timeout_s": 30.0},
        }

    def test_is_x3_confirm_pressed_reads_x_point_3(self) -> None:
        app = _ContractApp()
        app._return_values["get_x_point"] = 1
        assert _gw(app).is_x3_confirm_pressed() is True
        assert app.calls[0] == {"method": "get_x_point", "args": (3,), "kwargs": {}}

    def test_is_x3_confirm_pressed_returns_false_on_host_error(self) -> None:
        app = _ContractApp()
        def _raise(*a: Any, **kw: Any) -> int:
            raise RuntimeError
        app.get_x_point = _raise  # type: ignore[method-assign]
        assert _gw(app).is_x3_confirm_pressed() is False


# ===================================================================
# Poll profile / coils / halt (not covered by existing tests)
# ===================================================================

class TestPollProfileAndCoils:
    """set_plc_poll_profile / write_coil / halt."""

    @pytest.mark.parametrize("profile", ["normal", "sampling"])
    def test_set_plc_poll_profile_delegates(self, profile: str) -> None:
        app = _ContractApp()
        _gw(app).set_plc_poll_profile(profile)  # type: ignore[arg-type]
        assert app.calls == [{"method": "set_plc_poll_profile", "args": (profile,), "kwargs": {}}]

    def test_write_coil_delegates(self) -> None:
        app = _ContractApp()
        _gw(app).write_coil(coil_addr=100, value=True)
        assert app.calls == [{"method": "write_coil", "args": (100, True), "kwargs": {}}]


# ===================================================================
# Axis accessors (not covered by existing tests)
# ===================================================================

class TestAxisAccessors:
    """get_axis_cal / get_soft_limits_abs."""

    def test_get_axis_cal_returns_host_cal(self) -> None:
        app = _ContractApp()
        assert _gw(app).get_axis_cal() is app.axis_cal

    def test_get_soft_limits_abs_reads_snapshots(self) -> None:
        app = _ContractApp()
        from types import SimpleNamespace
        app._return_values["get_axis_copy"] = SimpleNamespace(
            act_pos=0.0, softlim_pos=200.0, softlim_neg=-10.0,
        )
        limits = _gw(app).get_soft_limits_abs(axes=[0, 1])
        assert limits == {0: (200.0, -10.0), 1: (200.0, -10.0)}
