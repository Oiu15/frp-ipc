from __future__ import annotations

"""Validation-specific machine gateway contracts.

These protocols define the machine-side operations that validation mode
measurement needs — motion control, clamp operations, rotation control —
layered beside the production ``DeviceGateway`` in ``machine.device_gateway``.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Protocol, runtime_checkable


class ValidationActionCancelled(RuntimeError):
    """Raised when a validation motion/action wait is cancelled."""


@runtime_checkable
class ValidationActionGateway(Protocol):
    """Validation-only motion/action hooks layered beside the production gateway."""

    def stop_rotation(self) -> None: ...

    def clamp_release(self) -> None: ...

    def clamp_close(self) -> None: ...

    def get_axis_cal(self) -> Any: ...

    def get_soft_limits_abs(self, axes: Sequence[int]) -> Mapping[int, tuple[float, float]]: ...

    def read_axis_position_mm(self, axis: int) -> float: ...

    def move_axis_absolute(
        self,
        axis: int,
        target_pos_mm: float,
        *,
        context: str = "ValidationMoveA",
    ) -> float: ...

    def move_axis_relative(
        self,
        axis: int,
        delta_mm: float,
        *,
        context: str = "ValidationMoveR",
    ) -> float: ...

    def move_axes_absolute(
        self,
        targets_abs: Mapping[int, float],
        *,
        context: str = "ValidationMoveA",
    ) -> Mapping[int, float]: ...

    def wait_axis_in_position(
        self,
        axis: int,
        target_pos_mm: float,
        *,
        tolerance_mm: float = 0.1,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.05,
        cancel_check: Callable[[], bool] | None = None,
    ) -> float: ...

    def wait_axes_in_position(
        self,
        targets_abs: Mapping[int, float],
        *,
        tolerance_mm: float = 0.1,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.05,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Mapping[int, float]: ...

    def wait_cancelable(
        self,
        duration_s: float,
        *,
        poll_interval_s: float = 0.05,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None: ...


__all__ = ["ValidationActionCancelled", "ValidationActionGateway"]
