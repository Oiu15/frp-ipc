from __future__ import annotations

"""Calibration-specific ports for the calibration service layer.

These ports live in services/ because they are application-layer boundaries,
not generic machine capabilities.  machine/ports.py holds the reusable
machine-level ports (MotionPort, SensorPort, RotationPort, etc.).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from machine.device_gateway import PollProfile
from services.calibration_context import CalibrationProgress, ClSample, GaugeSample


# ---------------------------------------------------------------------------
# CalibrationSensorPort — gauge / CL / angle reads for calibration capture
# ---------------------------------------------------------------------------


@runtime_checkable
class CalibrationSensorPort(Protocol):
    """Sensor read surface used by calibration capture loops."""

    def read_axis_angle_deg(self) -> float: ...
    def read_cl_out145_cached(self) -> ClSample: ...
    def request_gauge_sample(self) -> GaugeSample: ...
    def set_gauge_command(self, cmd: str) -> None: ...


# ---------------------------------------------------------------------------
# SchedulerPort — abstracts tkinter after/after_cancel for capture loops
# ---------------------------------------------------------------------------


@runtime_checkable
class SchedulerPort(Protocol):
    """Scheduling abstraction for capture-loop tick management.

    Uses one-shot scheduling (not repeating) to match tkinter's ``after``
    behaviour.  The service explicitly reschedules each tick, so lifecycle
    control (stop → cancel) is explicit and testable.
    """

    def schedule_once(self, delay_ms: int, callback: Callable[[], None]) -> object: ...

    def cancel(self, handle: object) -> None: ...


# ---------------------------------------------------------------------------
# CalibrationStateSink — mode state machine feedback
# ---------------------------------------------------------------------------


@runtime_checkable
class CalibrationStateSink(Protocol):
    """Calibration-mode state feedback port.

    Reports capture lifecycle to the calibration mode state machine
    and UI layer.  These replace the old ``host.calibration_mode.*``
    and Tk variable writes in the legacy service.
    """

    def begin_capture(self) -> None: ...
    def end_capture(self) -> None: ...
    def capture_failed(self, msg: str) -> None: ...
    def publish_progress(self, progress: CalibrationProgress) -> None: ...
    # per-calibration-type progress (avoids coupling to specific Tk vars)
    def publish_od_progress(self, progress: CalibrationProgress) -> None: ...
    def publish_id_progress(self, progress: CalibrationProgress) -> None: ...
    def publish_id_single_progress(self, progress: CalibrationProgress) -> None: ...
    def publish_id_verify_result(self, result: Mapping[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# PollProfilePort — PLC poll rate control for capture sessions
# ---------------------------------------------------------------------------


@runtime_checkable
class PollProfilePort(Protocol):
    """PLC poll profile control for calibration capture."""

    def use_poll_profile(self, profile: PollProfile) -> None: ...


# ---------------------------------------------------------------------------
# CalibrationViewPort — UI variable access for transitional controllers
# ---------------------------------------------------------------------------


@runtime_checkable
class CalibrationViewPort(Protocol):
    """Narrow UI-state surface used by CalibrationController.

    This keeps Tk variable reads/writes out of controller logic while the
    screen layer still exposes legacy host-backed variables.
    """

    def get_value(self, name: str, default: Any = None) -> Any: ...
    def set_value(self, name: str, value: Any) -> None: ...
    def get_float(self, name: str, default: float) -> float: ...


# ---------------------------------------------------------------------------
# CalibrationRepositoryProtocol — persistence contract
# ---------------------------------------------------------------------------


class CalibrationRepositoryProtocol(Protocol):
    """Persistence contract for calibration services.

    Defines only the methods that port-based calibration services actually
    call.  New code depends on this protocol, not the concrete repository.
    """

    def save_od_active(self, data: dict) -> None: ...
    def save_id_active(self, data: dict) -> None: ...
    def save_id_single_active(self, data: dict) -> None: ...
    def load_id_active(self) -> dict[str, Any]: ...
    def export_od_raw(self, points: list[Mapping[str, Any]]) -> Path: ...
    def export_id_raw(self, points: list[Mapping[str, Any]]) -> Path: ...


__all__ = [
    "CalibrationRepositoryProtocol",
    "CalibrationSensorPort",
    "CalibrationStateSink",
    "CalibrationViewPort",
    "PollProfilePort",
    "SchedulerPort",
]
