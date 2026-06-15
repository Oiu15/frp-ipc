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


__all__ = ["CalibrationProgress", "ClSample", "GaugeSample"]
