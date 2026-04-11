from __future__ import annotations

"""Lightweight application state objects for the measurement main flow."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from core.models import MeasureRow, Recipe

FIXED_SECTION_PRIMARY_METRICS = (
    "od_avg",
    "od_dev",
    "od_runout",
    "od_round",
    "od_round_fit_mm",
    "od_round_fit_rob_mm",
    "od_pp_mm",
    "od_pp_rob_mm",
    "od_e",
    "od_phi_deg",
    "id_avg",
    "id_dev",
    "id_runout",
    "id_round",
    "id_round_fit_mm",
    "id_round_fit_rob_mm",
    "id_pp_mm",
    "id_pp_rob_mm",
    "id_e",
    "id_phi_deg",
    "concentricity",
    "split_shift_deg",
)

VALIDATION_MOVE_CHANNELS = (
    "od_channel",
    "id_channel",
    "od_id_sync",
    "ax0_only",
    "ax1_only",
    "ax4_only",
)

VALIDATION_MOVE_SCENARIOS = (
    "distance_round_trip",
    "switch_and_return",
    "switch_and_measure_target",
)


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
class FixedSectionRepeatabilitySession:
    """Mutable session state for fixed-section repeatability validation."""

    task_name: str = "fixed_section_repeatability"
    section_name: str = ""
    metric_name: str = ""
    requested_repeat_count: int = 3
    reclamp_between_repeats: bool = False
    reclamp_enabled: bool = False
    rotation_stop_before_measure: bool = False
    release_settle_s: float = 0.0
    clamp_settle_s: float = 0.0
    validation_ax3_speed_dps: float = 60.0
    move_enabled: bool = False
    move_channel: str = "od_channel"
    move_away_delta_mm: float = 0.0
    move_scenario: str = "distance_round_trip"
    move_from_section_index: int = 1
    move_target_section_index: int = 1
    move_return_section_index: int = 1
    completed_repeat_count: int = 0
    rows_cache: list[dict[str, Any]] = field(default_factory=list)
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
    mode_kind: str = "none"
    mode_state: str = "idle"
    mode_error: str | None = None

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

    def sync_from_run_session(self, session: RunSession) -> None:
        """Refresh shared runtime state from the active production run session."""

        self.serial = session.serial
        self.run_id = session.run_id
        self.started_at_ts = session.start_ts
        self.finished_at_ts = session.end_ts
        self.rows = list(session.rows)
        self.raw_points = list(session.raw_points)
        self.summary = dict(session.summary_cache)
        self.length_result = None
        self.status = "idle"
        self.message = ""
        self.last_error = None

    def sync_from_validation_session(self, session: ValidationSession) -> None:
        """Refresh shared runtime state from the active validation session."""

        self.serial = session.serial
        self.run_id = session.run_id
        self.started_at_ts = session.start_ts
        self.finished_at_ts = session.end_ts
        self.rows = []
        self.raw_points = []
        self.summary = dict(session.summary_cache)
        self.length_result = None
        self.status = "idle"
        self.message = ""
        self.last_error = None


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
    "FIXED_SECTION_PRIMARY_METRICS",
    "FixedSectionRepeatabilitySession",
    "RunContext",
    "RunIdentity",
    "RunSession",
    "RuntimeState",
    "VALIDATION_MOVE_CHANNELS",
    "VALIDATION_MOVE_SCENARIOS",
    "ValidationExportContext",
    "ValidationSession",
]
