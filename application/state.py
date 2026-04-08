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
    id_single_enabled: bool = False
    id_single_k: float = 1.0
    id_single_b_mm: float = 0.0
    id_single_d_ref_mm: float | None = None


@dataclass(slots=True)
class RunSession:
    """Mutable run-session state for the current measurement run."""

    serial: str | None = None
    run_id: str | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    rows: list[MeasureRow] = field(default_factory=list)
    raw_points: list[dict[str, Any]] = field(default_factory=list)
    summary_cache: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationSession:
    """Mutable session state for validation mode.

    This is intentionally narrower than production RunSession and focuses on
    validation-specific identity and result bookkeeping needed by future
    standard-part, R&R, Cg/Cgk, and truth-piece workflows.
    """

    serial: str | None = None
    run_id: str | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    standard_piece_id: str | None = None
    validation_batch_id: str | None = None
    repeat_measurement_count: int = 0
    summary_cache: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeState:
    """Workflow-owned runtime state, independent from UI/App objects."""

    serial: str | None = None
    run_id: str | None = None
    started_at_ts: float | None = None
    finished_at_ts: float | None = None
    status: str = "idle"
    message: str = ""
    rows: list[MeasureRow] = field(default_factory=list)
    raw_points: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    length_result: dict[str, Any] | None = None
    last_error: str | None = None

    @classmethod
    def from_run_session(cls, session: RunSession) -> "RuntimeState":
        """Create a runtime state snapshot from the current run session."""

        return cls(
            serial=session.serial,
            run_id=session.run_id,
            started_at_ts=session.start_ts,
            finished_at_ts=session.end_ts,
            rows=list(session.rows),
            raw_points=list(session.raw_points),
            summary=dict(session.summary_cache),
        )

    @classmethod
    def from_validation_session(cls, session: ValidationSession) -> "RuntimeState":
        """Create a runtime state snapshot from a validation session."""

        return cls(
            serial=session.serial,
            run_id=session.run_id,
            started_at_ts=session.start_ts,
            finished_at_ts=session.end_ts,
            summary=dict(session.summary_cache),
        )


@dataclass(slots=True)
class ValidationExportContext:
    """Mutable export context owned by validation mode."""

    identity: RunIdentity
    recipe: Recipe
    calibration: CalibrationSnapshot
    standard_piece_id: str | None = None
    validation_batch_id: str | None = None
    repeat_measurement_count: int = 0
    summary: dict[str, Any] = field(default_factory=dict)
    events: list[Mapping[str, Any]] = field(default_factory=list)
    started_at_ts: float | None = None
    finished_at_ts: float | None = None
    status: str = "RUNNING"
    message: str = ""


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
    "RunSession",
    "RuntimeState",
    "ValidationExportContext",
    "ValidationSession",
]
