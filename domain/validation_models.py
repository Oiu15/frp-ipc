from __future__ import annotations

"""Shared validation request, result, and capture models."""

from dataclasses import dataclass
from typing import Any, Mapping

from core.models import MeasureRow
from domain.state import ValidationFitResult


@dataclass(frozen=True, slots=True)
class FixedSectionRepeatabilityRequest:
    task_name: str = "fixed_section_repeatability"
    section_name: str = ""
    metric_name: str = ""
    repeat_count: int = 3
    reclamp_between_repeats: bool = False
    reclamp_enabled: bool = False
    rotation_stop_before_measure: bool = False
    release_settle_s: float = 0.0
    clamp_settle_s: float = 0.0
    position_settle_s: float = 0.0
    sample_delay_s: float = 0.0
    validation_ax3_speed_dps: float = 60.0
    move_enabled: bool = False
    move_channel: str = "od_channel"
    move_away_delta_mm: float = 0.0
    move_scenario: str = "distance_round_trip"
    move_from_section_index: int = 1
    move_target_section_index: int = 1
    move_return_section_index: int = 1


@dataclass(frozen=True, slots=True)
class FixedSectionRepeatRow:
    repeat_index: int
    section_name: str
    metric_name: str
    measured_value_mm: float
    settle_s_used: float
    sample_delay_s_used: float
    capture_start_ts: float | None
    capture_end_ts: float | None
    measured_at_ts: float
    measure_section_index: int | None = None
    measure_section_name: str = ""
    measured_z_pos_mm: float = 0.0


@dataclass(frozen=True, slots=True)
class FixedSectionWindow:
    repeat_index: int
    window_index: int
    window_role: str
    point_start_index: int | None
    point_end_index: int | None
    point_count: int
    ts_start: float | None
    ts_end: float | None
    theta_start_deg: float | None
    theta_end_deg: float | None
    theta_span_deg: float
    filled_bins: int | None
    total_bins: int | None
    miss_bins: int | None
    n_od: int | None
    n_id: int | None
    reason: str
    revs: float | None
    elapsed_s: float | None
    max_gap_deg: float | None


@dataclass(frozen=True, slots=True)
class FixedSectionRepeatCapture:
    repeat_index: int
    section_name: str
    metric_name: str
    measured_at_ts: float
    measured_value_mm: float
    settle_s_used: float
    sample_delay_s_used: float
    capture_start_ts: float | None
    capture_end_ts: float | None
    section_result: MeasureRow
    windows: tuple[FixedSectionWindow, ...]
    raw_points: tuple[Mapping[str, Any], ...]
    coverage: Mapping[str, Any]
    measure_section_index: int | None = None
    measure_section_name: str = ""
    measured_z_pos_mm: float = 0.0
    fit_result: ValidationFitResult | None = None


__all__ = [
    "FixedSectionRepeatabilityRequest",
    "FixedSectionRepeatCapture",
    "FixedSectionRepeatRow",
    "FixedSectionWindow",
]
