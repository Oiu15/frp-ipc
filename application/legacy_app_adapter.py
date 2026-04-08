from __future__ import annotations

"""Legacy adapters that let new application-layer boundaries reuse App directly."""

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from core.models import MeasureRow
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


class LegacyAppEventSink:
    """Thin event-sink adapter backed by the existing App ui_q protocol."""

    def __init__(self, app: "App") -> None:
        self.app = app

    def publish_state(self, state: str, message: str) -> None:
        self.app.ui_q.put(("auto_state", {"state": state, "msg": message}))

    def publish_progress(
        self,
        *,
        section_index: int,
        section_total: int,
        z_pos_mm: float,
        ax0_abs: float,
    ) -> None:
        self.app.ui_q.put(
            (
                "auto_progress",
                {
                    "idx": max(0, int(section_index) - 1),
                    "total": int(section_total),
                    "x_ui": float(z_pos_mm),
                    "x_abs": float(ax0_abs),
                },
            )
        )

    def publish_length(self, payload: Mapping[str, Any]) -> None:
        self.app.ui_q.put(("auto_len", dict(payload)))

    def publish_coverage(self, payload: Mapping[str, Any]) -> None:
        self.app.ui_q.put(("auto_cov", dict(payload)))

    def publish_raw_points(self, points: Sequence[Mapping[str, Any]]) -> None:
        self.app.ui_q.put(("auto_raw_points", {"points": list(points)}))

    def publish_row(self, row: MeasureRow) -> None:
        self.app.ui_q.put(("auto_row", {"row": row}))

    def publish_straightness(self, payload: Mapping[str, Any]) -> None:
        self.app.ui_q.put(("auto_straightness", dict(payload)))

    def publish_postcalc(self, payload: Mapping[str, Any]) -> None:
        self.app.ui_q.put(("auto_postcalc", dict(payload)))


__all__ = ["LegacyAppDeviceGateway", "LegacyAppEventSink"]
