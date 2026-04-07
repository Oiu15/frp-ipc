from __future__ import annotations

"""Lightweight application state objects for the measurement main flow."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from core.models import MeasureRow, Recipe


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Stable run identity allocated before a measurement starts."""

    serial: str
    run_id: str
    started_at_ts: float


@dataclass(frozen=True, slots=True)
class CalibrationSnapshot:
    """Calibration values consumed by the measurement flow."""

    od_b_active_mm: float = 0.0
    od_out1_map: str = "L"
    od_d_ref_mm: float | None = None
    od_request_cmd: str = ""
    id_delta_c_mm: float = 0.0
    id_d_ref_mm: float | None = None


@dataclass(slots=True)
class RunContext:
    """Mutable run context owned by the application/orchestrator layer."""

    identity: RunIdentity
    recipe: Recipe
    calibration: CalibrationSnapshot
    rows: list[MeasureRow] = field(default_factory=list)
    raw_points: list[Mapping[str, Any]] = field(default_factory=list)
    section_coverage: dict[int, Mapping[str, Any]] = field(default_factory=dict)
    length_result: Mapping[str, Any] | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    finished_at_ts: float | None = None
    status: str = "RUNNING"


__all__ = [
    "CalibrationSnapshot",
    "RunContext",
    "RunIdentity",
]
