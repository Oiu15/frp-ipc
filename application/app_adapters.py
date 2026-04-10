from __future__ import annotations

"""Adapters that let application-layer boundaries reuse the App host directly."""

import time
from typing import TYPE_CHECKING, Any, Callable, Sequence

from application.contracts import ValidationActionCancelled
from application.state import FIXED_SECTION_PRIMARY_METRICS
from machine.device_gateway import ClChannel, ClReadResult, PollProfile, RegsRead

if TYPE_CHECKING:  # pragma: no cover
    from app import App
    from core.models import AxisComm


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


class AppDeviceGateway:
    """Thin device-gateway adapter backed by the existing App methods.

    This class intentionally delegates to the current app host instead of
    reimplementing control logic. It exists so the new gateway boundary can
    be introduced without rewriting the current measurement chain first.
    """

    def __init__(self, app: "App") -> None:
        self.app = app

    def get_axis_copy(self, axis: int) -> "AxisComm":
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

    def pulse_cmd_mask(self, axis: int, pulse_mask: int, pulse_ms: int = 120) -> None:
        self.app.pulse_cmd_mask(axis, pulse_mask, pulse_ms=pulse_ms)

    def write_coil(self, coil_addr: int, value: int | bool) -> None:
        self.app.write_coil(coil_addr, value)

    def stop_rotation(self) -> None:
        self.stop(3)

    def clamp_release(self) -> None:
        self.app.plc_write_y_point(10, 0)
        self.app.plc_write_y_point(11, 0)

    def clamp_close(self) -> None:
        self.app.plc_write_y_point(10, 1)
        self.app.plc_write_y_point(11, 1)

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

        cancel_event = getattr(self.app, "_validation_debug_cancel_event", None)
        is_set = getattr(cancel_event, "is_set", None)
        if callable(is_set):
            try:
                return bool(is_set())
            except Exception:
                return False

        try:
            return bool(getattr(self.app, "_validation_debug_cancel_requested", False))
        except Exception:
            return False


class ScreenPresenter:
    """Read-mostly presenter proxy for screens during migration.

    In addition to host-backed view state, it can own screen-local widget
    references and other display-only metadata so screens no longer write
    them back onto the host object.
    """

    def __init__(self, app: "App") -> None:
        object.__setattr__(self, "_app", app)
        object.__setattr__(self, "_widgets", {})
        object.__setattr__(self, "_view_state", {})

    @property
    def host_app(self) -> "App":
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

    def __getattr__(self, name: str) -> Any:
        widgets = object.__getattribute__(self, "_widgets")
        if name in widgets:
            return widgets[name]
        view_state = object.__getattribute__(self, "_view_state")
        if name in view_state:
            return view_state[name]
        attr = getattr(self.host_app, name)
        if callable(attr) and not (name.startswith("_refresh") or name.startswith("_list")):
            raise AttributeError(name)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(name)


class ScreenController:
    """Callable-only controller proxy for legacy screens during migration."""

    def __init__(self, app: "App") -> None:
        object.__setattr__(self, "_app", app)

    @property
    def host_app(self) -> "App":
        return object.__getattribute__(self, "_app")

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
            return self.host_app.start_fixed_section_repeatability_debug(
                section_name=section,
                metric_name=metric,
                repeat_count=repeat,
                reclamp_between_repeats=_coerce_bool(reclamp_between_repeats),
                reclamp_enabled=_coerce_bool(reclamp_enabled),
                rotation_stop_before_measure=_coerce_bool(rotation_stop_before_measure),
                release_settle_s=_coerce_non_negative_float(release_settle_s, "release_settle_s"),
                clamp_settle_s=_coerce_non_negative_float(clamp_settle_s, "clamp_settle_s"),
            )
        except Exception as exc:
            setter = getattr(self.host_app, '_set_validation_debug_feedback', None)
            if callable(setter):
                setter(status='ERR', result='', error=str(exc), export_path='')
            return None

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.host_app, name)
        if not callable(attr):
            raise AttributeError(name)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(name)


class ScreenUiContext:
    """Read-only UI-state proxy for legacy screens during migration."""

    def __init__(self, app: "App") -> None:
        object.__setattr__(self, "_app", app)

    @property
    def host_app(self) -> "App":
        return object.__getattribute__(self, "_app")

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.host_app, name)
        if callable(attr):
            raise AttributeError(name)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(name)

    def __delattr__(self, name: str) -> None:
        raise AttributeError(name)


__all__ = ["AppDeviceGateway", "ScreenController", "ScreenPresenter", "ScreenUiContext"]
