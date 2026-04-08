from __future__ import annotations

"""Legacy adapters that let new application-layer boundaries reuse App directly."""

import logging
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from core.models import MeasureRow
from machine.device_gateway import ClChannel, ClReadResult, PollProfile, RegsRead
from application.ui_queue_adapters import WorkflowUiEventAdapter

if TYPE_CHECKING:  # pragma: no cover
    from app import App
    from core.models import AxisComm


logger = logging.getLogger("frp.app.compat")


class LegacyAppDeviceGateway:
    """Thin device-gateway adapter backed by the existing App methods.

    This class intentionally delegates to the legacy runtime host instead of
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


class LegacyScreenAppAdapter:
    """Screen-facing adapter that temporarily preserves the old App surface.

    Existing ``ui/screens/*`` modules still expect an ``app`` object that mixes
    Tk variables, widget references, and command callbacks. This adapter keeps
    those screens working while giving the application shell a stable handoff
    point for future controller/presenter injection.
    """

    _BLOCKED_COMPAT_NAMES = frozenset(
        {
            "_auto_start",
            "_auto_stop",
            "_prepare_new_run",
            "_ensure_run_identity",
            "_export_current_run",
        }
    )

    def __init__(self, app: "App") -> None:
        object.__setattr__(self, "_app", app)
        object.__setattr__(self, "_warned_legacy_method_names", set())

    @property
    def host_app(self) -> "App":
        return object.__getattribute__(self, "_app")

    @classmethod
    def _is_blocked_name(cls, name: str) -> bool:
        return name in cls._BLOCKED_COMPAT_NAMES

    def _log_blocked_access(self, name: str, *, operation: str) -> None:
        logger.error(
            "LEGACY_SCREEN_ADAPTER_BLOCKED name=%s operation=%s host=%s",
            name,
            operation,
            type(self.host_app).__name__,
        )

    def _warn_legacy_method_once(self, name: str) -> None:
        warned_names: set[str] = object.__getattribute__(self, "_warned_legacy_method_names")
        if name in warned_names:
            return
        warned_names.add(name)
        logger.warning(
            "LEGACY_SCREEN_ADAPTER_METHOD name=%s host=%s",
            name,
            type(self.host_app).__name__,
        )

    def __getattr__(self, name: str) -> Any:
        if self._is_blocked_name(name):
            self._log_blocked_access(name, operation="get")
            raise AttributeError(name)
        attr = getattr(self.host_app, name)
        if callable(attr) and name.startswith("_"):
            self._warn_legacy_method_once(name)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        if self._is_blocked_name(name):
            self._log_blocked_access(name, operation="set")
            raise AttributeError(name)
        setattr(self.host_app, name, value)

    def __delattr__(self, name: str) -> None:
        if self._is_blocked_name(name):
            self._log_blocked_access(name, operation="delete")
            raise AttributeError(name)
        delattr(self.host_app, name)

    def __dir__(self) -> list[str]:
        names = set(super().__dir__())
        try:
            names.update(dir(self.host_app))
        except Exception:
            pass
        names.difference_update(self._BLOCKED_COMPAT_NAMES)
        return sorted(names)


class LegacyScreenPresenter:
    """Read-only presenter proxy for legacy screens during migration."""

    def __init__(self, app: "App") -> None:
        object.__setattr__(self, "_app", app)

    @property
    def host_app(self) -> "App":
        return object.__getattribute__(self, "_app")

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.host_app, name)
        if callable(attr) and not (name.startswith("_refresh") or name.startswith("_list")):
            raise AttributeError(name)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(name)


class LegacyScreenController:
    """Callable-only controller proxy for legacy screens during migration."""

    def __init__(self, app: "App") -> None:
        object.__setattr__(self, "_app", app)

    @property
    def host_app(self) -> "App":
        return object.__getattribute__(self, "_app")

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.host_app, name)
        if not callable(attr):
            raise AttributeError(name)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(name)


class LegacyScreenUiContext:
    """UI-state proxy for legacy screens during migration."""

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
        setattr(self.host_app, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self.host_app, name)


class LegacyAppEventSink:
    """Compatibility wrapper around the queue-based workflow event adapter."""

    def __init__(self, app: "App") -> None:
        self.app = app
        self._adapter = WorkflowUiEventAdapter(app.ui_q)

    def publish_state(self, state: str, message: str) -> None:
        self._adapter.publish_state(state, message)

    def publish_progress(
        self,
        *,
        section_index: int,
        section_total: int,
        z_pos_mm: float,
        ax0_abs: float,
    ) -> None:
        self._adapter.publish_progress(
            section_index=section_index,
            section_total=section_total,
            z_pos_mm=z_pos_mm,
            ax0_abs=ax0_abs,
        )

    def publish_length(self, payload: Mapping[str, Any]) -> None:
        self._adapter.publish_length(payload)

    def publish_coverage(self, payload: Mapping[str, Any]) -> None:
        self._adapter.publish_coverage(payload)

    def publish_raw_points(self, points: Sequence[Mapping[str, Any]]) -> None:
        self._adapter.publish_raw_points(points)

    def publish_row(self, row: MeasureRow) -> None:
        self._adapter.publish_row(row)

    def publish_straightness(self, payload: Mapping[str, Any]) -> None:
        self._adapter.publish_straightness(payload)

    def publish_postcalc(self, payload: Mapping[str, Any]) -> None:
        self._adapter.publish_postcalc(payload)

__all__ = ["LegacyAppDeviceGateway", "LegacyAppEventSink", "LegacyScreenAppAdapter", "LegacyScreenController", "LegacyScreenPresenter", "LegacyScreenUiContext"]
