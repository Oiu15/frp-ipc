from __future__ import annotations

"""Typed result objects for the calibration service layer.

These replace the ``Any`` return types in the legacy calibration service
with concrete dataclasses, making the contract explicit and testable.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GaugeSample:
    """A single gauge reading from the OD gauge (Keyence CL-3000 OUT1)."""

    value_mm: float
    raw: object | None = None
    ok: bool = True
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ClSample:
    """A cached CL-3000 multi-channel snapshot (OUT1/OUT4/OUT5)."""

    out1: float | None = None
    out4: float | None = None
    out5: float | None = None
    timestamp: float = 0.0
    ok: bool = True


@dataclass(frozen=True, slots=True)
class CalibrationProgress:
    """Progress update emitted during a calibration capture session."""

    angle_deg: float
    elapsed_s: float
    sample_count: int


# -- capture configuration dataclasses ---------------------------------------


@dataclass(frozen=True, slots=True)
class OdCalibrationSettings:
    rotation_speed_dps: float = 10.0
    sampling_hz: float = 20.0
    capture_duration_s: float = 10.0
    reference_diameter_mm: float = 180.0
    mode: str = "timed"
    angle_enabled: bool = True
    filter_mode: str = ""
    outlier_sigma: float = 3.0
    gauge_cmd: str = "M0,1"


@dataclass(frozen=True, slots=True)
class IdCalibrationSettings:
    rotation_speed_dps: float = 10.0
    sampling_hz: float = 20.0
    capture_duration_s: float = 10.0
    reference_diameter_mm: float = 150.0
    mode: str = "timed"
    force_one_rev: bool = False


@dataclass(frozen=True, slots=True)
class IdSingleCalibrationSettings:
    rotation_speed_dps: float = 10.0
    sampling_hz: float = 20.0
    capture_duration_s: float = 10.0
    reference_diameter_mm: float = 150.0


__all__ = [
    "CalibrationProgress",
    "ClSample",
    "GaugeSample",
    "IdCalibrationSettings",
    "IdSingleCalibrationSettings",
    "OdCalibrationSettings",
]
