from __future__ import annotations

"""Legacy adapters that let new application-layer boundaries reuse App directly."""

from typing import TYPE_CHECKING, Sequence

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


__all__ = ["LegacyAppDeviceGateway"]
