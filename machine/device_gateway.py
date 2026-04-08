from __future__ import annotations

"""Minimal machine gateway contract for the formal measurement flow.

This protocol is intentionally narrow. It only includes the machine-side
operations that the current formal measurement chain should depend on.
It does not try to expose UI helpers, recipe access, event dispatch, or
every raw PLC capability.
"""

from typing import Literal, Mapping, Protocol, Sequence, runtime_checkable

from core.models import AxisComm

PollProfile = Literal["normal", "sampling"]
ClChannel = Literal["out145", "out3"]
RegsRead = list[int]
ClOut145Read = tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    Mapping[str, int | None],
    Mapping[str, int | None],
]
ClOut3Read = tuple[float | None, int | None, int | None]
ClReadResult = ClOut145Read | ClOut3Read


@runtime_checkable
class DeviceGateway(Protocol):
    """Minimal device-facing boundary for the formal measurement chain."""

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

    def read_regs_sync(
        self,
        d_addr: int,
        count: int,
        timeout_s: float = 0.35,
    ) -> RegsRead | None: ...

    def read_cl_sync(
        self,
        channel: ClChannel,
        *,
        timeout_s: float = 0.5,
    ) -> ClReadResult | None: ...

    def set_plc_poll_profile(self, profile: PollProfile = "normal") -> None: ...

    def pulse_cmd_mask(self, axis: int, pulse_mask: int, pulse_ms: int = 120) -> None: ...

    def write_coil(self, coil_addr: int, value: int | bool) -> None: ...


__all__ = [
    "ClChannel",
    "ClOut145Read",
    "ClOut3Read",
    "ClReadResult",
    "DeviceGateway",
    "PollProfile",
    "RegsRead",
]
