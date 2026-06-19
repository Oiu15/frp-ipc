from __future__ import annotations

"""Adapters that let application-layer boundaries reuse the App host directly."""

import math
import time
from collections.abc import Callable, Iterable
from typing import Any, Mapping, Protocol, Sequence, cast

from machine.validation_gateway import ValidationActionCancelled
from machine.ports import MotionPort, OperatorPort, PlcCommandPort, RotationPort, SensorPort
from services.calibration_ports import (
    CalibrationSensorPort,
    CalibrationStateSink,
    PollProfilePort,
    SchedulerPort,
)
from services.calibration_context import CalibrationProgress, ClSample, GaugeSample
from domain.state import (
    FIXED_SECTION_PRIMARY_METRICS,
    VALIDATION_MOVE_CHANNELS,
    VALIDATION_MOVE_SCENARIOS,
)
from core.models import AxisComm, Recipe
from machine.device_gateway import ClChannel, ClReadResult, PollProfile, RegsRead
from utils.logger import log


class _AppDeviceGatewayHost(Protocol):
    """Host methods used by AppDeviceGateway during the boundary migration."""

    def get_axis_copy(self, axis: int) -> AxisComm: ...

    def movea_abs(self, axis: int, pos_abs: float, *, context: str = "MoveA") -> None: ...

    def velmove(
        self,
        axis: int,
        velocity: float,
        *,
        acc: float = 80.0,
        dec: float = 80.0,
        jerk: float = 300.0,
    ) -> None: ...

    def stop(self, axis: int) -> None: ...

    def halt(self, axis: int) -> None: ...

    def reset(self, axis: int) -> None: ...

    def enable(self, axis: int) -> None: ...

    def abort_motion(self, axes: Sequence[int] | None = None) -> None: ...

    def apply_soft_limits_abs(
        self,
        axis: int,
        target_abs: float,
        *,
        strict: bool = False,
        context: str = "",
    ) -> float: ...

    def read_regs_sync(self, d_addr: int, count: int, timeout_s: float = 0.35) -> RegsRead | None: ...

    def read_axis_act_pos_deg_sync(self, axis: int = 3, timeout_s: float = 0.35) -> float | None: ...

    def read_cl_sync(self, channel: ClChannel, *, timeout_s: float = 0.5) -> ClReadResult | None: ...

    def set_plc_poll_profile(self, profile: PollProfile = "normal") -> None: ...

    def pulse_cmd_mask(self, axis: int, pulse_mask: int, pulse_ms: int = 120) -> None: ...

    def write_coil(self, coil_addr: int, value: int | bool) -> None: ...

    def plc_write_y_point(self, y_point: int, value: int) -> None: ...

    # -- PlcCommandPort host backing methods ------------------------------

    def _base(self, axis: int) -> int: ...

    def _write_regs(self, d_addr: int, values: list[int]) -> None: ...

    def set_cmd_bits(self, axis: int, set_mask: int = 0, clr_mask: int = 0) -> None: ...

    def _pulse_cmd_bits(self, axis: int, pulse_mask: int, pulse_ms: int = 120) -> None: ...

    def _velmove_start_axis(
        self, axis: int, vel_velmove: float, *, acc: float = 80.0, dec: float = 80.0, jerk: float = 300.0,
    ) -> None: ...

    def _get_ax0_z_disp_limits(self) -> tuple[float, float, float]: ...

    def get_x_point(self, x_point: int) -> int: ...

    def get_y_point(self, y_point: int) -> int: ...

    # -- SensorPort backing methods ---------------------------------------

    def _get_latest_ax3_angle_deg(self) -> float | None: ...

    def _get_latest_cl145(self) -> Any: ...

    # -- SchedulerPort / CalibrationStateSink backing methods ---------------

    def after(self, ms: Any, func: Callable[..., Any] | None = None, *args: Any) -> Any: ...

    after_cancel: Any
    _odcal_points: list[dict[str, Any]]
    _odcal_drop_cnt: int
    idcal_chk_err_var: Any
    idcal_chk_cov_var: Any
    idcal_chk_n_var: Any
    idcal_chk_dtheta_var: Any
    idcal_msg_var: Any

    @property
    def calibration_mode(self) -> Any: ...

    @property
    def id_single_cal_state_var(self) -> Any: ...

    @property
    def idcal_state_var(self) -> Any: ...

    @property
    def sim_gauge_enabled(self) -> bool: ...

    @property
    def sim_disp_enabled(self) -> bool: ...

    def simulate_gauge_once(self, recipe: "Recipe") -> tuple[float, str]: ...

    def simulate_disp_once(self, recipe: "Recipe") -> tuple[float, str]: ...

    def calc_id_single_from_out2(
        self, theta_deg: "Iterable[float]", out2_mm: "Iterable[float]", recipe: "Recipe",
    ) -> dict[str, Any]: ...

    def get_recipe_copy(self) -> "Recipe": ...

    def get_calibration_snapshot(self) -> Any | None: ...

    @property
    def axis_cal(self) -> Any: ...

    @property
    def gauge_worker(self) -> Any: ...

    def operator_confirm(
        self,
        title: str,
        message: str,
        *,
        allow_stop: bool = True,
        timeout_s: float | None = None,
    ) -> str: ...


def _coerce_bool(value: bool | str | int) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"", "0", "false", "no", "n", "off"}:
        return False
    return bool(value)


def _coerce_non_negative_float(value: str | int | float, field_name: str) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        numeric = float(text)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if numeric < 0.0:
        raise ValueError(f"{field_name} must be >= 0")
    return numeric


def _coerce_positive_float(value: str | int | float, field_name: str) -> float:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be > 0")
    try:
        numeric = float(text)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if numeric <= 0.0:
        raise ValueError(f"{field_name} must be > 0")
    return numeric


def _coerce_choice(value: str, field_name: str, choices: Sequence[str]) -> str:
    text = str(value or "").strip()
    if text not in choices:
        raise ValueError(f"{field_name} must be one of: " + ", ".join(choices))
    return text


def _coerce_positive_int(value: str | int | float, field_name: str) -> int:
    text = str(value or "").strip()
    if ":" in text:
        text = text.split(":", 1)[0].strip()
    if not text:
        raise ValueError(f"{field_name} must be >= 1")
    try:
        numeric = int(float(text))
    except Exception as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if numeric < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return numeric


class AppDeviceGateway(MotionPort, SensorPort, OperatorPort, PlcCommandPort, RotationPort, CalibrationSensorPort, SchedulerPort, CalibrationStateSink, PollProfilePort):
    """Thin device-gateway adapter backed by the existing App methods.

    This class intentionally delegates to the current app host instead of
    reimplementing control logic. It exists so the new gateway boundary can
    be introduced without rewriting the current measurement chain first.
    """

    def __init__(self, app: _AppDeviceGatewayHost) -> None:
        self.app = app

    def get_axis_copy(self, axis: int) -> AxisComm:
        return self.app.get_axis_copy(axis)

    def movea_abs(self, axis: int, pos_abs: float, *, context: str = "MoveA") -> None:
        self.app.movea_abs(axis, pos_abs, context=context)

    def velmove(
        self,
        axis: int,
        velocity: float,
        *,
        acc: float = 80.0,
        dec: float = 80.0,
        jerk: float = 300.0,
    ) -> None:
        self.app.velmove(axis, velocity, acc=acc, dec=dec, jerk=jerk)

    def stop(self, axis: int) -> None:
        self.app.stop(axis)

    def halt(self, axis: int) -> None:
        self.app.halt(axis)

    def reset(self, axis: int) -> None:
        self.app.reset(axis)

    def enable(self, axis: int) -> None:
        self.app.enable(axis)

    def abort_motion(self, axes: Sequence[int] | None = None) -> None:
        self.app.abort_motion(axes)

    def apply_soft_limits_abs(
        self,
        axis: int,
        target_abs: float,
        *,
        strict: bool = False,
        context: str = "",
    ) -> float:
        return self.app.apply_soft_limits_abs(axis, target_abs, strict=strict, context=context)

    def read_regs_sync(self, d_addr: int, count: int, timeout_s: float = 0.35) -> RegsRead | None:
        return self.app.read_regs_sync(d_addr, count, timeout_s=timeout_s)

    def read_axis_angle_deg_sync(self, axis: int = 3, timeout_s: float = 0.35) -> float | None:
        return self.app.read_axis_act_pos_deg_sync(axis=axis, timeout_s=timeout_s)

    def read_cl_sync(
        self,
        channel: ClChannel,
        *,
        timeout_s: float = 0.5,
    ) -> ClReadResult | None:
        return self.app.read_cl_sync(channel, timeout_s=timeout_s)

    def set_plc_poll_profile(self, profile: PollProfile = "normal") -> None:
        self.app.set_plc_poll_profile(profile)

    # -- OperatorPort methods ------------------------------------------------

    def get_x_point(self, x_point: int) -> int:
        return int(self.app.get_x_point(x_point))

    def get_y_point(self, y_point: int) -> int:
        return int(self.app.get_y_point(y_point) if hasattr(self.app, "get_y_point") else 0)

    def plc_write_y_point(self, y_point: int, value: int) -> None:
        self.app.plc_write_y_point(y_point, value)

    def pulse_cmd_mask(self, axis: int, pulse_mask: int, pulse_ms: int = 120) -> None:
        self.app.pulse_cmd_mask(axis, pulse_mask, pulse_ms=pulse_ms)

    def write_coil(self, coil_addr: int, value: int | bool) -> None:
        self.app.write_coil(coil_addr, value)

    # -- PlcCommandPort methods -------------------------------------------

    def base_for_axis(self, axis: int) -> int:
        return self.app._base(int(axis))

    def write_regs(self, addr: int, values: list[int]) -> None:
        self.app._write_regs(addr, values)

    def set_cmd_bits(self, axis: int, *, set_mask: int = 0, clr_mask: int = 0) -> None:
        self.app.set_cmd_bits(axis, set_mask=set_mask, clr_mask=clr_mask)

    def pulse_cmd_bits(self, axis: int, mask: int, pulse_ms: int = 120) -> None:
        self.app._pulse_cmd_bits(axis, mask, pulse_ms=pulse_ms)

    def start_velocity_move(
        self, axis: int, velocity: float, *, acc: float = 80.0, dec: float = 80.0, jerk: float = 300.0,
    ) -> None:
        self.app._velmove_start_axis(axis, velocity, acc=acc, dec=dec, jerk=jerk)

    def get_ax0_z_disp_limits(self) -> tuple[float, float, float]:
        return self.app._get_ax0_z_disp_limits()

    # Deprecated compatibility shims. Workflow/executor code must use the
    # public PlcCommandPort names above; these remain for older tests/callers.
    def _base(self, axis: int) -> int:
        return self.base_for_axis(axis)

    def _write_regs(self, d_addr: int, values: list[int]) -> None:
        self.write_regs(d_addr, values)

    def _pulse_cmd_bits(self, axis: int, pulse_mask: int, pulse_ms: int = 120) -> None:
        self.pulse_cmd_bits(axis, pulse_mask, pulse_ms=pulse_ms)

    def _velmove_start_axis(
        self, axis: int, vel_velmove: float, *, acc: float = 80.0, dec: float = 80.0, jerk: float = 300.0,
    ) -> None:
        self.start_velocity_move(axis, vel_velmove, acc=acc, dec=dec, jerk=jerk)

    def _get_ax0_z_disp_limits(self) -> tuple[float, float, float]:
        return self.get_ax0_z_disp_limits()

    def stop_rotation(self) -> None:
        self.stop(3)

    def clamp_release(self) -> None:
        self.app.plc_write_y_point(10, 0)
        self.app.plc_write_y_point(11, 0)

    def clamp_close(self) -> None:
        self.app.plc_write_y_point(10, 1)
        self.app.plc_write_y_point(11, 1)

    def get_axis_cal(self) -> Any:
        axis_cal = getattr(self.app, "axis_cal", None)
        if axis_cal is None:
            raise RuntimeError("AxisCal is not available")
        return axis_cal

    @property
    def axis_cal(self) -> Any:
        return self.app.axis_cal

    def get_soft_limits_abs(self, axes: Sequence[int]) -> Mapping[int, tuple[float, float]]:
        limits: dict[int, tuple[float, float]] = {}
        for axis in axes:
            ax = int(axis)
            try:
                snapshot = self.app.get_axis_copy(ax)
                pos_limit = float(getattr(snapshot, "softlim_pos", float("nan")))
                neg_limit = float(getattr(snapshot, "softlim_neg", float("nan")))
            except Exception:
                continue
            if not (math.isfinite(pos_limit) and math.isfinite(neg_limit)):
                continue
            if abs(pos_limit) + abs(neg_limit) < 1e-6:
                continue
            limits[ax] = (pos_limit, neg_limit)
        return limits

    def read_axis_position_mm(self, axis: int) -> float:
        ax = int(axis)
        try:
            snapshot = self.app.get_axis_copy(ax)
            position = float(getattr(snapshot, "act_pos"))
        except Exception as exc:
            raise RuntimeError(f"AX{ax} position feedback is not available") from exc
        if not math.isfinite(position):
            raise RuntimeError(f"AX{ax} position feedback is invalid")
        return position

    def move_axis_absolute(
        self,
        axis: int,
        target_pos_mm: float,
        *,
        context: str = "ValidationMoveA",
    ) -> float:
        ax = int(axis)
        try:
            target = float(target_pos_mm)
        except Exception as exc:
            raise ValueError("target_pos_mm must be a number") from exc
        if not math.isfinite(target):
            raise ValueError("target_pos_mm must be finite")

        apply_limits = getattr(self.app, "apply_soft_limits_abs", None)
        if callable(apply_limits):
            apply_limits_fn = cast(Callable[..., float], apply_limits)
            target = float(apply_limits_fn(ax, target, strict=False, context=str(context)))
        self.app.movea_abs(ax, target, context=str(context))
        return target

    def move_axis_relative(
        self,
        axis: int,
        delta_mm: float,
        *,
        context: str = "ValidationMoveR",
    ) -> float:
        ax = int(axis)
        try:
            delta = float(delta_mm)
        except Exception as exc:
            raise ValueError("delta_mm must be a number") from exc
        if not math.isfinite(delta):
            raise ValueError("delta_mm must be finite")
        current = self.read_axis_position_mm(ax)
        return self.move_axis_absolute(ax, current + delta, context=context)

    def move_axes_absolute(
        self,
        targets_abs: Mapping[int, float],
        *,
        context: str = "ValidationMoveA",
    ) -> Mapping[int, float]:
        resolved: dict[int, float] = {}
        apply_limits = getattr(self.app, "apply_soft_limits_abs", None)
        apply_limits_fn = cast(Callable[..., float] | None, apply_limits) if callable(apply_limits) else None
        for axis, target_pos in targets_abs.items():
            ax = int(axis)
            try:
                target = float(target_pos)
            except Exception as exc:
                raise ValueError(f"AX{ax} target must be a number") from exc
            if not math.isfinite(target):
                raise ValueError(f"AX{ax} target must be finite")
            if apply_limits_fn is not None:
                target = float(apply_limits_fn(ax, target, strict=False, context=str(context)))
            resolved[ax] = target

        for ax, target in resolved.items():
            self.app.movea_abs(ax, target, context=str(context))
        return resolved

    def wait_axis_in_position(
        self,
        axis: int,
        target_pos_mm: float,
        *,
        tolerance_mm: float = 0.1,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.05,
        cancel_check: Callable[[], bool] | None = None,
    ) -> float:
        ax = int(axis)
        try:
            target = float(target_pos_mm)
        except Exception as exc:
            raise ValueError("target_pos_mm must be a number") from exc
        if not math.isfinite(target):
            raise ValueError("target_pos_mm must be finite")

        tolerance = max(0.0, float(tolerance_mm or 0.0))
        timeout = max(0.0, float(timeout_s or 0.0))
        poll_s = max(0.001, float(poll_interval_s or 0.05))
        deadline = time.monotonic() + timeout
        current = self.read_axis_position_mm(ax)

        while True:
            self._raise_if_validation_cancelled(cancel_check)
            if abs(current - target) <= tolerance:
                return current
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                log(
                    "VALIDATION_WAIT_INPOS_TIMEOUT",
                    axis=ax,
                    target=target,
                    actual=current,
                    timeout_s=timeout,
                    tolerance=tolerance,
                    actual_source="axis_snapshot",
                    current_poll_profile=str(getattr(self.app, "_plc_poll_profile_req", "") or ""),
                )
                raise TimeoutError(
                    f"AX{ax} in-position timeout: "
                    f"target={target:.3f}, actual={current:.3f}, tol={tolerance:.3f}"
                )
            self.wait_cancelable(
                min(poll_s, remaining_s),
                poll_interval_s=min(poll_s, remaining_s),
                cancel_check=cancel_check,
            )
            current = self.read_axis_position_mm(ax)

    def wait_axes_in_position(
        self,
        targets_abs: Mapping[int, float],
        *,
        tolerance_mm: float = 0.1,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.05,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Mapping[int, float]:
        actuals: dict[int, float] = {}
        for axis, target_pos in targets_abs.items():
            ax = int(axis)
            actuals[ax] = self.wait_axis_in_position(
                ax,
                float(target_pos),
                tolerance_mm=tolerance_mm,
                timeout_s=timeout_s,
                poll_interval_s=poll_interval_s,
                cancel_check=cancel_check,
            )
        return actuals

    def wait_cancelable(
        self,
        duration_s: float,
        *,
        poll_interval_s: float = 0.05,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        deadline = time.monotonic() + max(0.0, float(duration_s or 0.0))
        poll_s = max(0.001, float(poll_interval_s or 0.05))
        self._raise_if_validation_cancelled(cancel_check)
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                return
            time.sleep(min(poll_s, remaining_s))
            self._raise_if_validation_cancelled(cancel_check)

    def open_dual_clamps(self) -> None:
        self.clamp_release()

    def close_dual_clamps(self) -> None:
        self.clamp_close()

    def is_x3_confirm_pressed(self) -> bool:
        try:
            return bool(self.app.get_x_point(3))
        except Exception:
            return False

    def operator_confirm(
        self,
        title: str,
        message: str,
        *,
        allow_stop: bool = True,
        timeout_s: float | None = None,
    ) -> str:
        return str(self.app.operator_confirm(title, message, allow_stop=allow_stop, timeout_s=timeout_s))

    def _raise_if_validation_cancelled(self, cancel_check: Callable[[], bool] | None = None) -> None:
        if self._validation_cancel_requested(cancel_check):
            raise ValidationActionCancelled("validation action cancelled")

    def _validation_cancel_requested(self, cancel_check: Callable[[], bool] | None = None) -> bool:
        if callable(cancel_check):
            try:
                return bool(cancel_check())
            except ValidationActionCancelled:
                raise
            except Exception:
                return False

        is_cancel_requested = getattr(self.app, "is_validation_cancel_requested", None)
        if callable(is_cancel_requested):
            try:
                return bool(is_cancel_requested())
            except Exception:
                return False

        cancel_event = getattr(self.app, "_validation_cancel_event", None)
        if cancel_event is None:
            cancel_event = getattr(self.app, "_validation_debug_cancel_event", None)
        is_set = getattr(cancel_event, "is_set", None)
        if callable(is_set):
            try:
                return bool(is_set())
            except Exception:
                return False

        try:
            return bool(
                getattr(
                    self.app,
                    "_validation_cancel_requested",
                    getattr(self.app, "_validation_debug_cancel_requested", False),
                )
            )
        except Exception:
            return False

    # -- SensorPort methods ------------------------------------------------

    @property
    def latest_ax3_angle_deg(self) -> float | None:
        return self.app._get_latest_ax3_angle_deg()  # type: ignore[no-any-return]

    @property
    def latest_cl145(self) -> Any:
        return self.app._get_latest_cl145()  # type: ignore[no-any-return]

    @property
    def latest_cl3(self) -> Any:
        return self.app._get_latest_cl3()  # type: ignore[no-any-return]

    @property
    def sim_gauge_enabled(self) -> bool:
        return bool(getattr(self.app, "sim_gauge_enabled", False))

    @property
    def sim_disp_enabled(self) -> bool:
        return bool(getattr(self.app, "sim_disp_enabled", False))

    def simulate_gauge_once(self, recipe: Recipe) -> tuple[float, str]:
        return self.app.simulate_gauge_once(recipe)  # type: ignore[no-any-return]

    def simulate_disp_once(self, recipe: Recipe) -> tuple[float, str]:
        return self.app.simulate_disp_once(recipe)  # type: ignore[no-any-return]

    def calc_id_single_from_out2(
        self, theta_deg: Iterable[float], out2_mm: Iterable[float], recipe: Recipe,
    ) -> dict[str, Any]:
        return self.app.calc_id_single_from_out2(theta_deg, out2_mm, recipe)  # type: ignore[no-any-return]

    def fit_id_diameter(
        self, theta_deg: Any, c_mm: Any, m_mm: Any, delta_c: float,
    ) -> dict[str, Any] | None:
        fn = getattr(self.app, "_idcal_fit_diameter", None)
        if callable(fn):
            return fn(theta_deg, c_mm, m_mm, float(delta_c))  # type: ignore[no-any-return]
        return None

    def get_recipe_copy(self) -> Recipe:
        return self.app.get_recipe_copy()  # type: ignore[no-any-return]

    def get_calibration_snapshot(self) -> Any | None:
        fn = getattr(self.app, "get_calibration_snapshot", None)
        if callable(fn):
            return fn()
        repo = getattr(self.app, "calibration_repository", None)
        if repo is not None:
            return repo.load_snapshot()  # type: ignore[no-any-return]
        return None

    @property
    def gauge_worker(self) -> Any:
        return getattr(self.app, "gauge_worker", None)

    # -- RotationPort ---------------------------------------------------------

    def start_rotation(self, rpm: float) -> None:
        self.app._velmove_start_axis(3, rpm)

    # -- CalibrationSensorPort ------------------------------------------------

    def read_axis_angle_deg(self) -> float:
        return float(self.app._get_latest_ax3_angle_deg() or 0.0)

    def set_gauge_command(self, cmd: str) -> None:
        if getattr(self.app, "gauge_worker", None) is not None:
            self.app.gauge_worker.request_cmd = cmd  # type: ignore[attr-defined]

    def read_cl_out145_cached(self) -> ClSample:
        out = self.app._get_latest_cl145()
        try:
            out1, out2, out4, out5, _, _ = out  # x1_mm, x2_mm, c_mm, m_mm, raw, cnt
            return ClSample(
                out1=float(out1) if out1 is not None else None,
                out2=float(out2) if out2 is not None else None,
                out4=float(out4) if out4 is not None else None,
                out5=float(out5) if out5 is not None else None,
                timestamp=0.0,
                ok=True,
            )
        except Exception:
            return ClSample(ok=False)

    def request_gauge_sample(self) -> GaugeSample:
        gw = self.gauge_worker
        if gw is None:
            return GaugeSample(value_mm=0.0, ok=False, error="no gauge worker")
        try:
            gw.send_request()
            import time
            time.sleep(0.02)
            s = gw.get_last()
            if s is None:
                return GaugeSample(value_mm=0.0, ok=False, error="no sample")
            return GaugeSample(
                value_mm=float(getattr(s, "od", 0.0) or 0.0),
                raw=s,
                ok=True,
            )
        except Exception as exc:
            return GaugeSample(value_mm=0.0, ok=False, error=str(exc))

    # -- SchedulerPort --------------------------------------------------------

    def schedule_once(self, delay_ms: int, callback: Callable[[], None]) -> object:
        return self.app.after(int(delay_ms), callback)

    def cancel(self, handle: object) -> None:
        self.app.after_cancel(handle)

    # -- CalibrationStateSink -------------------------------------------------

    def begin_capture(self) -> None:
        self.app.calibration_mode.begin_acquiring()

    def end_capture(self) -> None:
        self.app.calibration_mode.complete()

    def capture_failed(self, msg: str) -> None:
        self.app.calibration_mode.fail(msg)

    def publish_progress(self, progress: CalibrationProgress) -> None:
        """Generic progress — delegates to per-type methods."""
        pass

    def publish_od_progress(self, progress: CalibrationProgress) -> None:
        try:
            state_var = getattr(self.app, "odcal_state_var", None)
            if state_var is not None:
                state_var.set(
                    f"{progress.angle_deg:.1f}° / {progress.elapsed_s:.1f}s / {progress.sample_count}"
                )
        except Exception:
            pass

    def publish_od_sample(self, point: Mapping[str, Any], total_count: int, drop_count: int) -> None:
        try:
            points = getattr(self.app, "_odcal_points", None)
            if not isinstance(points, list):
                points = []
                self.app._odcal_points = points
            points.append(dict(point))
            self.app._odcal_drop_cnt = int(drop_count)
            n_var = getattr(self.app, "odcal_n_var", None)
            if n_var is not None:
                n_var.set(str(int(total_count)))
            update_stats = getattr(self.app, "_odcal_update_stats", None)
            if callable(update_stats):
                update_stats()
        except Exception:
            pass

    def publish_id_progress(self, progress: CalibrationProgress) -> None:
        try:
            if hasattr(self.app, "idcal_state_var"):
                self.app.idcal_state_var.set(
                    f"{progress.angle_deg:.1f}° / {progress.sample_count}"
                )
        except Exception:
            pass

    def publish_id_single_progress(self, progress: CalibrationProgress) -> None:
        try:
            if hasattr(self.app, "id_single_cal_state_var"):
                self.app.id_single_cal_state_var.set(
                    f"{progress.angle_deg:.1f}° / {progress.sample_count}"
                )
        except Exception:
            pass

    def publish_id_verify_result(self, result: Mapping[str, Any]) -> None:
        try:
            err = result.get("err_mm")
            cov = result.get("cov_pct")
            n = result.get("n", result.get("sample_count", 0))
            dtheta = result.get("dtheta_max_deg")

            err_value = None if err is None else float(err)
            cov_value = None if cov is None else float(cov)
            n_value = None if n is None else int(n or 0)
            dtheta_value = None if dtheta is None else float(dtheta)

            if err_value is not None:
                self.app.idcal_chk_err_var.set(f"{err_value:+.4f}")
            else:
                self.app.idcal_chk_err_var.set("--")
            if cov_value is not None:
                self.app.idcal_chk_cov_var.set(f"{cov_value:.2f}%")
            else:
                self.app.idcal_chk_cov_var.set("--")
            self.app.idcal_chk_n_var.set(str(n_value) if n_value is not None else "--")
            if dtheta_value is not None:
                self.app.idcal_chk_dtheta_var.set(f"{dtheta_value:.3f}")
            else:
                self.app.idcal_chk_dtheta_var.set("--")

            err_msg = "--" if err_value is None else f"{err_value:+.4f}mm"
            cov_msg = "--" if cov_value is None else f"{cov_value:.2f}%"
            n_msg = "--" if n_value is None else str(n_value)

            if result.get("ok"):
                self.app.idcal_state_var.set("CHK_OK")
                self.app.idcal_msg_var.set(
                    f"复核OK: ΔD={err_msg}  N={n_msg}  cover={cov_msg}"
                )
                return
            reason = str(result.get("reason", "复核失败"))
            if err_value is None:
                self.app.idcal_state_var.set("ERR")
                self.app.idcal_msg_var.set(reason)
            else:
                self.app.idcal_state_var.set("CHK_NG")
                self.app.idcal_msg_var.set(
                    f"复核NG: ΔD={err_msg}  N={n_msg}  cover={cov_msg}"
                )
        except Exception:
            pass

    # -- PollProfilePort ------------------------------------------------------

    def use_poll_profile(self, profile: PollProfile) -> None:
        self.app.set_plc_poll_profile(profile)


class ScreenPresenter:
    """Read-mostly presenter proxy for screens during migration.

    In addition to host-backed view state, it can own screen-local widget
    references and other display-only metadata so screens no longer write
    them back onto the host object.
    """

    def __init__(self, app: Any) -> None:
        object.__setattr__(self, "_app", app)
        object.__setattr__(self, "_widgets", {})
        object.__setattr__(self, "_view_state", {})

    @property
    def host_app(self) -> Any:
        return object.__getattribute__(self, "_app")

    def remember_widget(self, name: str, widget: Any) -> Any:
        object.__getattribute__(self, "_widgets")[name] = widget
        return widget

    def widget(self, name: str) -> Any:
        return object.__getattribute__(self, "_widgets").get(name)

    def remember_view_state(self, name: str, value: Any) -> Any:
        object.__getattribute__(self, "_view_state")[name] = value
        return value

    def view_state(self, name: str, default: Any = None) -> Any:
        return object.__getattribute__(self, "_view_state").get(name, default)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(name)


class ScreenController:
    """Callable-only controller proxy for legacy screens during migration."""

    def __init__(self, app: Any) -> None:
        object.__setattr__(self, "_app", app)

    @property
    def host_app(self) -> Any:
        return object.__getattribute__(self, "_app")

    def list_validation_section_choices(self) -> list[str]:
        provider = getattr(self.host_app, "list_validation_section_choices", None)
        if callable(provider):
            provider_fn = cast(Callable[[], Sequence[str]], provider)
            return list(provider_fn())
        return ["1"]

    def start_validation_run(
        self,
        section_name: str,
        metric_name: str,
        repeat_count: str | int,
        reclamp_between_repeats: bool | str | int = False,
        *,
        reclamp_enabled: bool | str | int = False,
        rotation_stop_before_measure: bool | str | int = False,
        release_settle_s: str | int | float = 0.0,
        clamp_settle_s: str | int | float = 0.0,
        position_settle_s: str | int | float = 0.0,
        sample_delay_s: str | int | float = 0.0,
        validation_ax3_speed_dps: str | int | float = 60.0,
        move_enabled: bool | str | int = False,
        move_channel: str = "od_channel",
        move_away_delta_mm: str | int | float = 0.0,
        move_scenario: str = "distance_round_trip",
        move_from_section_index: str | int | float = 1,
        move_target_section_index: str | int | float = 1,
        move_return_section_index: str | int | float = 1,
    ) -> Any:
        try:
            section = str(section_name or "").strip()
            metric = str(metric_name or "").strip()
            repeat_raw = str(repeat_count).strip()
            if not repeat_raw:
                raise ValueError("repeat_count cannot be empty")
            try:
                repeat = int(repeat_raw)
            except Exception as exc:
                raise ValueError("repeat_count must be a positive integer") from exc
            if repeat < 1:
                raise ValueError("repeat_count must be >= 1")
            if metric not in FIXED_SECTION_PRIMARY_METRICS:
                raise ValueError(
                    "metric_name must be one of: " + ", ".join(FIXED_SECTION_PRIMARY_METRICS)
                )
            return self.host_app.start_validation_run(
                section_name=section,
                metric_name=metric,
                repeat_count=repeat,
                reclamp_between_repeats=_coerce_bool(reclamp_between_repeats),
                reclamp_enabled=_coerce_bool(reclamp_enabled),
                rotation_stop_before_measure=_coerce_bool(rotation_stop_before_measure),
                release_settle_s=_coerce_non_negative_float(release_settle_s, "release_settle_s"),
                clamp_settle_s=_coerce_non_negative_float(clamp_settle_s, "clamp_settle_s"),
                position_settle_s=_coerce_non_negative_float(position_settle_s, "position_settle_s"),
                sample_delay_s=_coerce_non_negative_float(sample_delay_s, "sample_delay_s"),
                validation_ax3_speed_dps=_coerce_positive_float(
                    validation_ax3_speed_dps,
                    "validation_ax3_speed_dps",
                ),
                move_enabled=_coerce_bool(move_enabled),
                move_channel=_coerce_choice(
                    move_channel,
                    "move_channel",
                    VALIDATION_MOVE_CHANNELS,
                ),
                move_away_delta_mm=_coerce_non_negative_float(
                    move_away_delta_mm,
                    "move_away_delta_mm",
                ),
                move_scenario=_coerce_choice(
                    move_scenario,
                    "move_scenario",
                    VALIDATION_MOVE_SCENARIOS,
                ),
                move_from_section_index=_coerce_positive_int(
                    move_from_section_index,
                    "move_from_section_index",
                ),
                move_target_section_index=_coerce_positive_int(
                    move_target_section_index,
                    "move_target_section_index",
                ),
                move_return_section_index=_coerce_positive_int(
                    move_return_section_index,
                    "move_return_section_index",
                ),
            )
        except Exception as exc:
            setter = getattr(self.host_app, '_set_validation_feedback', None)
            if not callable(setter):
                setter = getattr(self.host_app, '_set_validation_debug_feedback', None)
            if callable(setter):
                setter(status='ERR', result='', error=str(exc), export_path='')
            return None

    def stop_validation_run(self) -> Any:
        stopper = getattr(self.host_app, "stop_validation_run", None)
        if not callable(stopper):
            stopper = getattr(self.host_app, "stop_fixed_section_repeatability_debug", None)
        if callable(stopper):
            return stopper()
        return None

    def start_fixed_section_repeatability_debug(
        self,
        section_name: str,
        metric_name: str,
        repeat_count: str | int,
        reclamp_between_repeats: bool | str | int = False,
        *,
        reclamp_enabled: bool | str | int = False,
        rotation_stop_before_measure: bool | str | int = False,
        release_settle_s: str | int | float = 0.0,
        clamp_settle_s: str | int | float = 0.0,
        position_settle_s: str | int | float = 0.0,
        sample_delay_s: str | int | float = 0.0,
        validation_ax3_speed_dps: str | int | float = 60.0,
        move_enabled: bool | str | int = False,
        move_channel: str = "od_channel",
        move_away_delta_mm: str | int | float = 0.0,
        move_scenario: str = "distance_round_trip",
        move_from_section_index: str | int | float = 1,
        move_target_section_index: str | int | float = 1,
        move_return_section_index: str | int | float = 1,
    ) -> Any:
        return self.start_validation_run(
            section_name=section_name,
            metric_name=metric_name,
            repeat_count=repeat_count,
            reclamp_between_repeats=reclamp_between_repeats,
            reclamp_enabled=reclamp_enabled,
            rotation_stop_before_measure=rotation_stop_before_measure,
            release_settle_s=release_settle_s,
            clamp_settle_s=clamp_settle_s,
            position_settle_s=position_settle_s,
            sample_delay_s=sample_delay_s,
            validation_ax3_speed_dps=validation_ax3_speed_dps,
            move_enabled=move_enabled,
            move_channel=move_channel,
            move_away_delta_mm=move_away_delta_mm,
            move_scenario=move_scenario,
            move_from_section_index=move_from_section_index,
            move_target_section_index=move_target_section_index,
            move_return_section_index=move_return_section_index,
        )

    def stop_fixed_section_repeatability_debug(self) -> Any:
        return self.stop_validation_run()

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(name)


class ScreenUiContext:
    """Read-only UI-state proxy for legacy screens during migration."""

    def __init__(self, app: Any) -> None:
        object.__setattr__(self, "_app", app)

    @property
    def host_app(self) -> Any:
        return object.__getattribute__(self, "_app")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(name)

    def __delattr__(self, name: str) -> None:
        raise AttributeError(name)


__all__ = ["AppDeviceGateway", "ScreenController", "ScreenPresenter", "ScreenUiContext"]
