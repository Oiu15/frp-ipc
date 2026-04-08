from __future__ import annotations

"""Legacy adapters that let new application-layer boundaries reuse App directly."""

from typing import TYPE_CHECKING, Any, Sequence

from machine.device_gateway import ClChannel, ClReadResult, PollProfile, RegsRead

if TYPE_CHECKING:  # pragma: no cover
    from app import App
    from core.models import AxisComm


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


class LegacyScreenPresenter:
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


__all__ = ["LegacyAppDeviceGateway", "LegacyScreenController", "LegacyScreenPresenter", "LegacyScreenUiContext"]
