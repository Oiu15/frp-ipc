from __future__ import annotations

"""Pure workflow boundary objects for validation mode.

The first version intentionally stays minimal: it models workflow-owned state,
typed events, and a result object without assuming full validation sampling or
hardware choreography.
"""

import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Callable, Literal, Mapping, TypeAlias

from application.contracts import MachineGateway, RunRepositoryProtocol
from application.state import (
    CalibrationSnapshot,
    FIXED_SECTION_PRIMARY_METRICS,
    FixedSectionRepeatabilitySession,
    RunIdentity,
    RuntimeState,
    ValidationExportContext,
    ValidationFitResult,
    ValidationSession,
)
from core.models import AxisCal, MeasureRow, Recipe
from domain.planning import plan_section_positions, resolve_measured_section, resolve_recipe_section, resolve_section_targets
from frp_workflow.autoflow_orchestrator import measure_current_position_section_capture

SummaryPayload: TypeAlias = dict[str, Any]
ValidationResultStatus = Literal["DONE", "STOP", "ERR"]
WaitCallback: TypeAlias = Callable[[str, int, int, float], None]


class ValidationWorkflowEventType(StrEnum):
    STATE = "state"
    PROGRESS = "progress"
    PHASE = "phase"
    SUMMARY = "summary"


class ValidationPhase(StrEnum):
    PREPARE = "prepare"
    BEFORE_CAPTURE = "before_capture"
    STOP_ROTATION = "stop_rotation"
    UNCLAMP = "unclamp"
    WAIT_UNCLAMP_SETTLE = "wait_unclamp_settle"
    CLAMP = "clamp"
    WAIT_CLAMP_SETTLE = "wait_clamp_settle"
    MOVE_AWAY = "move_away"
    MOVE_BACK_TO_TARGET = "move_back_to_target"
    MOVE_TO_FROM_SECTION = "move_to_from_section"
    MOVE_TO_TARGET_SECTION = "move_to_target_section"
    MOVE_TO_RETURN_SECTION = "move_to_return_section"
    RESTORE_ROTATION_READY = "restore_rotation_ready"
    WAIT_POSITION_SETTLE = "wait_position_settle"
    WAIT_SAMPLE_DELAY = "wait_sample_delay"
    CAPTURE = "capture"
    FIT_CALC = "fit_calc"
    SAVE_RESULT = "save_result"


@dataclass(frozen=True, slots=True)
class StateEvent:
    state: str
    message: str
    type: ValidationWorkflowEventType = ValidationWorkflowEventType.STATE


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    step: str
    index: int
    total: int
    message: str = ""
    type: ValidationWorkflowEventType = ValidationWorkflowEventType.PROGRESS


@dataclass(frozen=True, slots=True)
class PhaseEvent:
    phase: str
    repeat_index: int
    total: int
    task_name: str = ""
    message: str = ""
    ts: float = field(default_factory=time.time)
    payload: Mapping[str, Any] = field(default_factory=dict)
    type: ValidationWorkflowEventType = ValidationWorkflowEventType.PHASE


@dataclass(frozen=True, slots=True)
class SummaryEvent:
    source: str
    payload: Mapping[str, Any]
    type: ValidationWorkflowEventType = ValidationWorkflowEventType.SUMMARY


TypedEvent: TypeAlias = StateEvent | ProgressEvent | PhaseEvent | SummaryEvent


@dataclass(frozen=True, slots=True)
class ValidationResult:
    identity: RunIdentity | None
    status: ValidationResultStatus
    message: str
    started_at_ts: float | None
    finished_at_ts: float | None
    standard_piece_id: str | None
    validation_batch_id: str | None
    repeat_measurement_count: int
    summary: Mapping[str, Any]


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


@dataclass(frozen=True, slots=True)
class _FixedSectionCapturePayload:
    section_result: MeasureRow
    raw_points: list[dict[str, Any]]
    windows_payload: list[dict[str, Any]]
    coverage_payload: dict[str, Any]
    capture_start_ts: float | None
    capture_end_ts: float | None
    measured_at_ts: float
    fit_payload: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _ValidationMovePlan:
    channel: str
    axes: tuple[int, ...]
    return_targets: Mapping[int, float]
    away_targets: Mapping[int, float]
    initial_positions: Mapping[int, float]
    return_z_disp_mm: float | None = None
    away_z_disp_mm: float | None = None


@dataclass(frozen=True, slots=True)
class _ValidationSectionMoveStep:
    role: str
    phase: ValidationPhase
    section_index: int
    z_pos_mm: float
    planned_targets: Mapping[int, float]
    move_targets: Mapping[int, float]


_MIN_VALID_OD_SAMPLE_COUNT = 6
_MIN_VALID_OD_BIN_COUNT = 6
_MIN_VALID_ROTATION_SPAN_DEG = 30.0
_ROTATION_READY_TIMEOUT_S = 2.0
_ROTATION_READY_POLL_S = 0.05
_ROTATION_READY_MIN_DELTA_DEG = 1.0
_VALIDATION_MOVE_IN_POSITION_TIMEOUT_S = 10.0
_VALIDATION_MOVE_IN_POSITION_TOLERANCE_MM = 0.1
_VALIDATION_MOVE_IN_POSITION_POLL_S = 0.05
_VALIDATION_DEBUG_AXIS_CHANNELS = {
    "ax0_only": 0,
    "ax1_only": 1,
    "ax4_only": 4,
}
_VALIDATION_MOVE_CHANNEL_AXES = {
    "od_channel": (0,),
    "id_channel": (1, 4),
    "od_id_sync": (0, 1, 4),
    "ax0_only": (0,),
    "ax1_only": (1,),
    "ax4_only": (4,),
}
_VALIDATION_MOVE_SCENARIOS = frozenset({
    "distance_round_trip",
    "switch_and_return",
    "switch_and_measure_target",
})
_VALIDATION_WAIT_PROGRESS_SLICE_S = 0.1


def _unwrap_theta_span_deg(theta_values_deg: list[float]) -> float:
    if len(theta_values_deg) < 2:
        return 0.0

    prev = float(theta_values_deg[0])
    cursor = prev
    min_cursor = cursor
    max_cursor = cursor
    for raw_theta in theta_values_deg[1:]:
        theta = float(raw_theta)
        delta = theta - prev
        while delta <= -180.0:
            delta += 360.0
        while delta > 180.0:
            delta -= 360.0
        cursor += delta
        min_cursor = min(min_cursor, cursor)
        max_cursor = max(max_cursor, cursor)
        prev = theta
    return float(max_cursor - min_cursor)


def _validate_fixed_section_od_sampling(raw_points: list[Mapping[str, Any]]) -> None:
    valid_points: list[Mapping[str, Any]] = []
    theta_values_deg: list[float] = []
    unique_bins: set[int] = set()
    for point in raw_points or []:
        if not isinstance(point, Mapping):
            continue
        od_mm = point.get("od_mm")
        theta_deg = point.get("theta_deg")
        if od_mm is None or theta_deg is None:
            continue
        valid_points.append(point)
        theta_values_deg.append(float(theta_deg))
        bin_index = point.get("bin")
        if bin_index is not None:
            try:
                unique_bins.add(int(bin_index))
            except Exception:
                pass

    if len(valid_points) < _MIN_VALID_OD_SAMPLE_COUNT:
        raise RuntimeError("有效采样点不足，验证结果无效")

    theta_span_deg = _unwrap_theta_span_deg(theta_values_deg)
    if len(unique_bins) < _MIN_VALID_OD_BIN_COUNT or theta_span_deg < _MIN_VALID_ROTATION_SPAN_DEG:
        raise RuntimeError("未检测到有效旋转，无法完成固定截面重复性验证")


def _reclamp_between_repeats(
    gateway: MachineGateway,
    *,
    repeat_index: int,
    total: int,
    record_state: Callable[[str, str], StateEvent],
    status_callback: Callable[[str], None] | None = None,
) -> None:
    open_clamps = getattr(gateway, "open_dual_clamps", None)
    close_clamps = getattr(gateway, "close_dual_clamps", None)
    is_pressed = getattr(gateway, "is_x3_confirm_pressed", None)
    operator_confirm = getattr(gateway, "operator_confirm", None)
    if not callable(open_clamps) or not callable(close_clamps):
        raise RuntimeError("dual clamp control is not available")
    if not callable(operator_confirm):
        raise RuntimeError("operator confirm is not available")

    record_state("RECLAMP", f"reclamp before repeat {repeat_index + 1}/{total}")
    if callable(status_callback):
        status_callback(f"RECLAMP {repeat_index}/{total}")

    open_clamps()
    time.sleep(0.25)
    close_clamps()
    time.sleep(0.25)

    record_state("WAIT_X3_CONFIRM", f"wait X3 confirm before repeat {repeat_index + 1}/{total}")
    if callable(status_callback):
        status_callback(f"WAIT_X3_CONFIRM {repeat_index}/{total}")

    if callable(is_pressed):
        t0 = time.monotonic()
        while bool(is_pressed()):
            if (time.monotonic() - t0) > 2.0:
                break
            time.sleep(0.05)

    result = operator_confirm(
        "Clamp Confirm",
        "Confirm clamps are closed.\n\n- Press X3 or click confirm to continue\n- Stop to abort",
        allow_stop=True,
        timeout_s=60.0,
    )
    if str(result) != "confirm":
        raise RuntimeError(f"operator canceled: {result}")


def _extract_primary_metric_value(section_result: MeasureRow, metric_name: str) -> float:
    metric = str(metric_name or "").strip()
    if metric not in FIXED_SECTION_PRIMARY_METRICS:
        raise ValueError(f"fixed_section_repeatability does not support metric_name='{metric}'")
    value = getattr(section_result, metric, None)
    if value is None:
        raise RuntimeError(f"section result missing metric '{metric}'")
    return float(value)


def _summarize_numeric_values(values: list[float]) -> dict[str, float | int]:
    def _rounded(value: float) -> float:
        return round(float(value), 6)

    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "range": 0.0,
        }
    return {
        "count": len(values),
        "mean": _rounded(statistics.fmean(values)),
        "std": _rounded(statistics.pstdev(values)),
        "min": _rounded(min(values)),
        "max": _rounded(max(values)),
        "range": _rounded(max(values) - min(values)),
    }


def _summarize_section_result_metrics(
    captures: list[FixedSectionRepeatCapture],
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for field_name in FIXED_SECTION_PRIMARY_METRICS:
        values: list[float] = []
        for capture in captures:
            value = getattr(capture.section_result, field_name, None)
            if value is None or isinstance(value, bool):
                continue
            try:
                numeric = float(value)
            except Exception:
                continue
            if not math.isfinite(numeric):
                continue
            values.append(numeric)
        if values:
            metrics[field_name] = _summarize_numeric_values(values)
    return metrics


def _summarize_fixed_section_repeatability_rows(
    rows: list[FixedSectionRepeatRow],
    *,
    captures: list[FixedSectionRepeatCapture] | None = None,
) -> dict[str, Any]:
    values = [float(row.measured_value_mm) for row in rows]
    if not rows:
        return {
            "task_name": "fixed_section_repeatability",
            "section_name": "",
            "measure_section_index": None,
            "measure_section_name": "",
            "measured_z_pos_mm": None,
            "metric_name": "",
            "primary_metric": {},
            "section_metrics": {},
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "range": 0.0,
        }

    primary_metric_summary = _summarize_numeric_values(values)
    return {
        "task_name": "fixed_section_repeatability",
        "section_name": str(rows[0].section_name),
        "measure_section_index": rows[0].measure_section_index,
        "measure_section_name": str(rows[0].measure_section_name),
        "measured_z_pos_mm": float(rows[0].measured_z_pos_mm),
        "metric_name": str(rows[0].metric_name),
        "primary_metric": {
            str(rows[0].metric_name): dict(primary_metric_summary),
        },
        "section_metrics": _summarize_section_result_metrics(list(captures or [])),
        "count": int(primary_metric_summary["count"]),
        "mean": float(primary_metric_summary["mean"]),
        "std": float(primary_metric_summary["std"]),
        "min": float(primary_metric_summary["min"]),
        "max": float(primary_metric_summary["max"]),
        "range": float(primary_metric_summary["range"]),
    }


@dataclass(slots=True)
class ValidationWorkflow:
    """Minimal validation-workflow boundary.

    Input dependencies are intentionally aligned with production workflow so the
    future validation orchestrator can evolve without depending on UI/App state.
    """

    recipe: Recipe
    calibration: CalibrationSnapshot
    runtime_state: RuntimeState
    gateway: MachineGateway
    run_repository: RunRepositoryProtocol
    validation_session: ValidationSession | None = None
    _events: list[TypedEvent] = field(default_factory=list, init=False)
    _result: ValidationResult | None = field(default=None, init=False)
    _fixed_section_repeat_captures: list[FixedSectionRepeatCapture] = field(default_factory=list, init=False)
    _fixed_section_repeat_rows: list[FixedSectionRepeatRow] = field(default_factory=list, init=False)
    _current_phase: ValidationPhase | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._sync_runtime_from_session()

    @property
    def events(self) -> tuple[TypedEvent, ...]:
        return tuple(self._events)

    @property
    def summary(self) -> Mapping[str, Any]:
        return dict(self.runtime_state.summary)

    @property
    def result(self) -> ValidationResult | None:
        return self._result

    @property
    def current_phase(self) -> ValidationPhase | None:
        return self._current_phase

    @property
    def fixed_section_repeat_captures(self) -> tuple[FixedSectionRepeatCapture, ...]:
        return tuple(self._fixed_section_repeat_captures)

    def build_fixed_section_repeatability_summary(self) -> dict[str, Any]:
        return _summarize_fixed_section_repeatability_rows(
            list(self._fixed_section_repeat_rows),
            captures=list(self._fixed_section_repeat_captures),
        )

    def ensure_identity(self) -> RunIdentity:
        session = self.validation_session
        if session is not None and session.serial and session.run_id and session.start_ts is not None:
            self.runtime_state.serial = session.serial
            self.runtime_state.run_id = session.run_id
            self.runtime_state.started_at_ts = session.start_ts
            return RunIdentity(
                serial=str(session.serial),
                run_id=str(session.run_id),
                started_at_ts=float(session.start_ts),
            )

        if self.runtime_state.serial and self.runtime_state.run_id and self.runtime_state.started_at_ts is not None:
            identity = RunIdentity(
                serial=str(self.runtime_state.serial),
                run_id=str(self.runtime_state.run_id),
                started_at_ts=float(self.runtime_state.started_at_ts),
            )
            self._sync_session_from_runtime()
            return identity

        identity = self.run_repository.prepare_run(getattr(self.recipe, 'name', ''))
        self.runtime_state.serial = identity.serial
        self.runtime_state.run_id = identity.run_id
        self.runtime_state.started_at_ts = identity.started_at_ts
        self._sync_session_from_runtime()
        return identity

    def record_state(self, state: str, message: str = "") -> StateEvent:
        self.runtime_state.status = self._normalize_status(state)
        self.runtime_state.message = str(message or "")
        if str(state or "").upper() == "ERR":
            self.runtime_state.last_error = self.runtime_state.message or "Validation workflow error"
        event = StateEvent(state=str(state), message=str(message or ""))
        self._events.append(event)
        return event

    def record_progress(self, *, step: str, index: int, total: int, message: str = "") -> ProgressEvent:
        event = ProgressEvent(
            step=str(step),
            index=int(index),
            total=int(total),
            message=str(message or ""),
        )
        self._events.append(event)
        return event

    def record_phase(
        self,
        phase: ValidationPhase | str,
        *,
        repeat_index: int,
        total: int,
        task_name: str = "",
        message: str = "",
        payload: Mapping[str, Any] | None = None,
        phase_callback: Callable[[PhaseEvent], None] | None = None,
    ) -> PhaseEvent:
        phase_value = self._coerce_phase(phase)
        self._current_phase = phase_value
        event = PhaseEvent(
            phase=phase_value.value,
            repeat_index=int(repeat_index),
            total=int(total),
            task_name=str(task_name or ""),
            message=str(message or ""),
            payload=dict(payload or {}),
        )
        self._events.append(event)
        if callable(phase_callback):
            phase_callback(event)
        return event

    def record_summary(self, payload: Mapping[str, Any], *, source: str) -> SummaryEvent:
        copied = dict(payload)
        self.runtime_state.summary.update(copied)
        self._sync_session_from_runtime()
        event = SummaryEvent(source=str(source), payload=copied)
        self._events.append(event)
        return event

    def build_result(
        self,
        *,
        status: ValidationResultStatus,
        message: str = "",
        finished_at_ts: float | None = None,
    ) -> ValidationResult:
        identity = self.ensure_identity()
        self.runtime_state.status = self._normalize_status(status)
        self.runtime_state.message = str(message or "")
        self.runtime_state.finished_at_ts = (
            float(finished_at_ts)
            if finished_at_ts is not None
            else self.runtime_state.finished_at_ts
        )
        if status == "ERR":
            self.runtime_state.last_error = self.runtime_state.message or "Validation workflow error"
        self._sync_session_from_runtime()
        session = self.validation_session
        result = ValidationResult(
            identity=identity,
            status=status,
            message=self.runtime_state.message,
            started_at_ts=self.runtime_state.started_at_ts,
            finished_at_ts=self.runtime_state.finished_at_ts,
            standard_piece_id=(None if session is None else session.standard_piece_id),
            validation_batch_id=(None if session is None else session.validation_batch_id),
            repeat_measurement_count=(0 if session is None else int(session.repeat_measurement_count or 0)),
            summary=dict(self.runtime_state.summary),
        )
        self._result = result
        return result

    def build_export_context(
        self,
        *,
        status: ValidationResultStatus | None = None,
        message: str = "",
        finished_at_ts: float | None = None,
    ) -> ValidationExportContext:
        result = self._result
        if result is None:
            if status is None:
                raise ValueError('build_result() must be called first or status must be provided')
            result = self.build_result(status=status, message=message, finished_at_ts=finished_at_ts)
        identity = result.identity or self.ensure_identity()
        session = self.validation_session
        return ValidationExportContext(
            identity=identity,
            recipe=self.recipe,
            calibration=self.calibration,
            standard_piece_id=(None if session is None else session.standard_piece_id),
            validation_batch_id=(None if session is None else session.validation_batch_id),
            repeat_measurement_count=(0 if session is None else int(session.repeat_measurement_count or 0)),
            summary=dict(result.summary),
            events=[asdict(event) for event in self._events],
            started_at_ts=result.started_at_ts,
            finished_at_ts=result.finished_at_ts,
            status=result.status,
            message=result.message,
        )

    def run_fixed_section_repeatability(
        self,
        request: FixedSectionRepeatabilityRequest,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        status_callback: Callable[[str], None] | None = None,
        phase_callback: Callable[[PhaseEvent], None] | None = None,
        wait_callback: WaitCallback | None = None,
    ) -> tuple[list[FixedSectionRepeatRow], dict[str, Any]]:
        identity = self.ensure_identity()
        self.runtime_state.rows.clear()
        self.runtime_state.raw_points.clear()
        self.runtime_state.summary.clear()
        self._fixed_section_repeat_captures.clear()
        self._fixed_section_repeat_rows.clear()
        total = int(request.repeat_count or 3)
        section_name = str(request.section_name or "")
        metric_name = str(request.metric_name or "")
        validation_ax3_speed_dps = self._get_validation_rotation_velocity(request)
        local_session = FixedSectionRepeatabilitySession(
            section_name=section_name,
            metric_name=metric_name,
            requested_repeat_count=int(request.repeat_count or total),
            reclamp_between_repeats=bool(getattr(request, "reclamp_between_repeats", False)),
            reclamp_enabled=bool(getattr(request, "reclamp_enabled", False)),
            rotation_stop_before_measure=bool(getattr(request, "rotation_stop_before_measure", False)),
            release_settle_s=float(getattr(request, "release_settle_s", 0.0) or 0.0),
            clamp_settle_s=float(getattr(request, "clamp_settle_s", 0.0) or 0.0),
            position_settle_s=float(getattr(request, "position_settle_s", 0.0) or 0.0),
            sample_delay_s=float(getattr(request, "sample_delay_s", 0.0) or 0.0),
            validation_ax3_speed_dps=validation_ax3_speed_dps,
            move_enabled=bool(getattr(request, "move_enabled", False)),
            move_channel=str(getattr(request, "move_channel", "od_channel") or "od_channel"),
            move_away_delta_mm=float(getattr(request, "move_away_delta_mm", 0.0) or 0.0),
            move_scenario=str(getattr(request, "move_scenario", "distance_round_trip") or "distance_round_trip"),
            move_from_section_index=int(getattr(request, "move_from_section_index", 1) or 1),
            move_target_section_index=int(getattr(request, "move_target_section_index", 1) or 1),
            move_return_section_index=int(getattr(request, "move_return_section_index", 1) or 1),
        )

        self.record_state("RUN", f"{request.task_name} running")
        base_ts = float(time.time())
        rows: list[FixedSectionRepeatRow] = []
        try:
            for repeat_index in range(1, total + 1):
                self._run_fixed_section_prepare(
                    request=request,
                    repeat_index=repeat_index,
                    total=total,
                    phase_callback=phase_callback,
                )
                self._run_fixed_section_before_capture(
                    request=request,
                    repeat_index=repeat_index,
                    total=total,
                    phase_callback=phase_callback,
                    wait_callback=wait_callback,
                )
                capture_payload = self._run_fixed_section_capture(
                    request=request,
                    repeat_index=repeat_index,
                    total=total,
                    phase_callback=phase_callback,
                    wait_callback=wait_callback,
                )
                row, capture = self._run_fixed_section_fit_calc(
                    request=request,
                    repeat_index=repeat_index,
                    total=total,
                    section_name=section_name,
                    metric_name=metric_name,
                    capture_payload=capture_payload,
                    phase_callback=phase_callback,
                )
                self._run_fixed_section_save_result(
                    request=request,
                    repeat_index=repeat_index,
                    total=total,
                    row=row,
                    capture=capture,
                    rows=rows,
                    local_session=local_session,
                    progress_callback=progress_callback,
                    phase_callback=phase_callback,
                )
                if bool(getattr(request, "reclamp_between_repeats", False)) and repeat_index < total:
                    _reclamp_between_repeats(
                        self.gateway,
                        repeat_index=repeat_index,
                        total=total,
                        record_state=self.record_state,
                        status_callback=status_callback,
                    )

            summary = _summarize_fixed_section_repeatability_rows(
                rows,
                captures=self._fixed_section_repeat_captures,
            )
            local_session.summary_cache = dict(summary)

            self.runtime_state.started_at_ts = identity.started_at_ts
            self.runtime_state.finished_at_ts = rows[-1].measured_at_ts if rows else base_ts
            if self.validation_session is not None:
                self.validation_session.repeat_measurement_count = local_session.completed_repeat_count

            self.record_summary(summary, source=str(request.task_name or "fixed_section_repeatability"))
            done_message = f"{request.task_name} completed"
            self.record_state("DONE", done_message)
            self.build_result(
                status="DONE",
                message=done_message,
                finished_at_ts=self.runtime_state.finished_at_ts,
            )
            return rows, dict(summary)
        except Exception as exc:
            self.runtime_state.started_at_ts = identity.started_at_ts
            self.runtime_state.finished_at_ts = float(time.time())
            if self.validation_session is not None:
                self.validation_session.repeat_measurement_count = local_session.completed_repeat_count
            error_message = f"{request.task_name} failed: {exc}"
            self.record_state("ERR", error_message)
            self.build_result(
                status="ERR",
                message=error_message,
                finished_at_ts=self.runtime_state.finished_at_ts,
            )
            raise

    def _run_fixed_section_prepare(
        self,
        *,
        request: FixedSectionRepeatabilityRequest,
        repeat_index: int,
        total: int,
        phase_callback: Callable[[PhaseEvent], None] | None,
    ) -> None:
        self.record_phase(
            ValidationPhase.PREPARE,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"prepare repeat {repeat_index}/{total}",
            phase_callback=phase_callback,
        )

    def _run_fixed_section_before_capture(
        self,
        *,
        request: FixedSectionRepeatabilityRequest,
        repeat_index: int,
        total: int,
        phase_callback: Callable[[PhaseEvent], None] | None,
        wait_callback: WaitCallback | None,
    ) -> None:
        self.record_phase(
            ValidationPhase.BEFORE_CAPTURE,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"before_capture repeat {repeat_index}/{total}",
            phase_callback=phase_callback,
        )
        rotation_stop_before_measure = bool(getattr(request, "rotation_stop_before_measure", False))
        reclamp_enabled = bool(getattr(request, "reclamp_enabled", False))
        move_enabled = bool(getattr(request, "move_enabled", False))
        if rotation_stop_before_measure:
            self._run_validation_stop_rotation(
                request=request,
                repeat_index=repeat_index,
                total=total,
                phase_callback=phase_callback,
            )
        if reclamp_enabled:
            self._run_validation_reclamp_sequence(
                request=request,
                repeat_index=repeat_index,
                total=total,
                release_settle_s=float(getattr(request, "release_settle_s", 0.0) or 0.0),
                clamp_settle_s=float(getattr(request, "clamp_settle_s", 0.0) or 0.0),
                phase_callback=phase_callback,
                wait_callback=wait_callback,
            )
        if move_enabled:
            self._run_validation_section_relocation(
                request=request,
                repeat_index=repeat_index,
                total=total,
                phase_callback=phase_callback,
            )
        if rotation_stop_before_measure or reclamp_enabled:
            self._run_validation_restore_rotation_ready(
                request=request,
                repeat_index=repeat_index,
                total=total,
                phase_callback=phase_callback,
            )
        self._run_validation_position_settle(
            request=request,
            repeat_index=repeat_index,
            total=total,
            phase_callback=phase_callback,
            wait_callback=wait_callback,
        )

    def _run_validation_stop_rotation(
        self,
        *,
        request: FixedSectionRepeatabilityRequest,
        repeat_index: int,
        total: int,
        phase_callback: Callable[[PhaseEvent], None] | None,
    ) -> None:
        self.record_phase(
            ValidationPhase.STOP_ROTATION,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"stop_rotation repeat {repeat_index}/{total}",
            phase_callback=phase_callback,
        )
        stop_rotation = getattr(self.gateway, "stop_rotation", None)
        if not callable(stop_rotation):
            raise RuntimeError("validation stop_rotation action is not available")
        stop_rotation()

    def _run_validation_reclamp_sequence(
        self,
        *,
        request: FixedSectionRepeatabilityRequest,
        repeat_index: int,
        total: int,
        release_settle_s: float,
        clamp_settle_s: float,
        phase_callback: Callable[[PhaseEvent], None] | None,
        wait_callback: WaitCallback | None,
    ) -> None:
        clamp_release = getattr(self.gateway, "clamp_release", None)
        clamp_close = getattr(self.gateway, "clamp_close", None)
        if not callable(clamp_release) or not callable(clamp_close):
            raise RuntimeError("validation clamp actions are not available")
        self.record_phase(
            ValidationPhase.UNCLAMP,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"unclamp repeat {repeat_index}/{total}",
            phase_callback=phase_callback,
        )
        clamp_release()
        self.record_phase(
            ValidationPhase.WAIT_UNCLAMP_SETTLE,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"wait_unclamp_settle {release_settle_s:.3f}s",
            phase_callback=phase_callback,
        )
        self._wait_validation_action(
            release_settle_s,
            wait_phase=ValidationPhase.WAIT_UNCLAMP_SETTLE,
            repeat_index=repeat_index,
            total=total,
            wait_callback=wait_callback,
        )
        self.record_phase(
            ValidationPhase.CLAMP,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"clamp repeat {repeat_index}/{total}",
            phase_callback=phase_callback,
        )
        clamp_close()
        self.record_phase(
            ValidationPhase.WAIT_CLAMP_SETTLE,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"wait_clamp_settle {clamp_settle_s:.3f}s",
            phase_callback=phase_callback,
        )
        self._wait_validation_action(
            clamp_settle_s,
            wait_phase=ValidationPhase.WAIT_CLAMP_SETTLE,
            repeat_index=repeat_index,
            total=total,
            wait_callback=wait_callback,
        )

    def _run_validation_section_relocation(
        self,
        *,
        request: FixedSectionRepeatabilityRequest,
        repeat_index: int,
        total: int,
        phase_callback: Callable[[PhaseEvent], None] | None,
    ) -> None:
        channel = self._validation_move_channel(request)
        scenario = self._validation_move_scenario(request)
        if scenario != "distance_round_trip":
            self._run_validation_section_switch(
                request=request,
                channel=channel,
                scenario=scenario,
                repeat_index=repeat_index,
                total=total,
                phase_callback=phase_callback,
            )
            return

        move_away_delta_mm = self._validation_move_away_delta(request)
        plan = self._build_validation_move_plan(channel, move_away_delta_mm)
        self.record_phase(
            ValidationPhase.MOVE_AWAY,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=(
                f"move_away {channel} "
                f"delta={move_away_delta_mm:.3f}"
            ),
            payload=self._validation_move_payload(plan, plan.away_targets, plan.initial_positions),
            phase_callback=phase_callback,
        )
        actual_away_targets = self._move_validation_axes_absolute(
            plan.away_targets,
            context="VALIDATION_MOVE_AWAY",
        )
        actual_away_positions = self._wait_validation_axes_in_position(actual_away_targets)
        self._notify_validation_phase_update(
            ValidationPhase.MOVE_AWAY,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"move_away reached {channel}",
            payload=self._validation_move_payload(plan, actual_away_targets, actual_away_positions),
            phase_callback=phase_callback,
        )

        self.record_phase(
            ValidationPhase.MOVE_BACK_TO_TARGET,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"move_back_to_target {channel}",
            payload=self._validation_move_payload(plan, plan.return_targets, actual_away_positions),
            phase_callback=phase_callback,
        )
        actual_return_targets = self._move_validation_axes_absolute(
            plan.return_targets,
            context="VALIDATION_MOVE_BACK_TO_TARGET",
        )
        actual_return_positions = self._wait_validation_axes_in_position(actual_return_targets)
        self._notify_validation_phase_update(
            ValidationPhase.MOVE_BACK_TO_TARGET,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"move_back_to_target reached {channel}",
            payload=self._validation_move_payload(plan, actual_return_targets, actual_return_positions),
            phase_callback=phase_callback,
        )

    def _run_validation_section_switch(
        self,
        *,
        request: FixedSectionRepeatabilityRequest,
        channel: str,
        scenario: str,
        repeat_index: int,
        total: int,
        phase_callback: Callable[[PhaseEvent], None] | None,
    ) -> None:
        steps = self._build_validation_section_switch_steps(request, channel=channel, scenario=scenario)
        actual_positions = self._read_validation_axes_positions((0, 1, 4))
        for step in steps:
            self.record_phase(
                step.phase,
                repeat_index=repeat_index,
                total=total,
                task_name=request.task_name,
                message=(
                    f"{step.role} section={step.section_index} "
                    f"z={step.z_pos_mm:.3f} channel={channel}"
                ),
                payload=self._validation_section_move_payload(
                    request=request,
                    scenario=scenario,
                    channel=channel,
                    step=step,
                    actual_positions=actual_positions,
                    actual_after_wait=None,
                ),
                phase_callback=phase_callback,
            )
            actual_targets = self._move_validation_axes_absolute(
                step.move_targets,
                context=f"VALIDATION_{step.role.upper()}",
            )
            waited_positions = self._wait_validation_axes_in_position(actual_targets)
            actual_positions = {
                int(axis): float(actual_positions.get(int(axis), 0.0))
                for axis in (0, 1, 4)
            }
            for axis, position in waited_positions.items():
                actual_positions[int(axis)] = float(position)
            self._notify_validation_phase_update(
                step.phase,
                repeat_index=repeat_index,
                total=total,
                task_name=request.task_name,
                message=f"{step.role} reached section={step.section_index}",
                payload=self._validation_section_move_payload(
                    request=request,
                    scenario=scenario,
                    channel=channel,
                    step=step,
                    actual_positions=actual_positions,
                    actual_after_wait=actual_positions,
                ),
                phase_callback=phase_callback,
            )

    def _notify_validation_phase_update(
        self,
        phase: ValidationPhase,
        *,
        repeat_index: int,
        total: int,
        task_name: str,
        message: str,
        payload: Mapping[str, Any],
        phase_callback: Callable[[PhaseEvent], None] | None,
    ) -> None:
        if not callable(phase_callback):
            return
        phase_callback(
            PhaseEvent(
                phase=phase.value,
                repeat_index=int(repeat_index),
                total=int(total),
                task_name=str(task_name or ""),
                message=str(message or ""),
                payload=dict(payload),
            )
        )

    def _validation_move_channel(self, request: FixedSectionRepeatabilityRequest) -> str:
        channel = str(getattr(request, "move_channel", "od_channel") or "od_channel").strip()
        if channel not in _VALIDATION_MOVE_CHANNEL_AXES:
            raise ValueError(f"unsupported move_channel: {channel}")
        return channel

    def _validation_move_scenario(self, request: FixedSectionRepeatabilityRequest) -> str:
        scenario = str(getattr(request, "move_scenario", "distance_round_trip") or "distance_round_trip").strip()
        if scenario not in _VALIDATION_MOVE_SCENARIOS:
            raise ValueError(f"unsupported move_scenario: {scenario}")
        return scenario

    def _validation_move_away_delta(self, request: FixedSectionRepeatabilityRequest) -> float:
        try:
            delta = float(getattr(request, "move_away_delta_mm", 0.0) or 0.0)
        except Exception as exc:
            raise ValueError("move_away_delta_mm must be a number") from exc
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError("move_away_delta_mm must be > 0 when move_enabled")
        return delta

    def _build_validation_section_switch_steps(
        self,
        request: FixedSectionRepeatabilityRequest,
        *,
        channel: str,
        scenario: str,
    ) -> list[_ValidationSectionMoveStep]:
        positions = tuple(float(value) for value in plan_section_positions(self.recipe).positions_z)
        if not positions:
            raise ValueError("recipe section list is empty")
        from_index = self._validation_section_index(request, "move_from_section_index", len(positions))
        target_index = self._validation_section_index(request, "move_target_section_index", len(positions))
        return_index = self._validation_section_index(request, "move_return_section_index", len(positions))
        final_index = target_index if scenario == "switch_and_measure_target" else return_index

        axis_cal = self._get_validation_axis_cal()
        steps: list[_ValidationSectionMoveStep] = [
            self._build_validation_section_move_step(
                role="from_section",
                phase=ValidationPhase.MOVE_TO_FROM_SECTION,
                section_index=from_index,
                z_pos_mm=positions[from_index - 1],
                axis_cal=axis_cal,
                channel=channel,
            ),
            self._build_validation_section_move_step(
                role="target_section",
                phase=ValidationPhase.MOVE_TO_TARGET_SECTION,
                section_index=target_index,
                z_pos_mm=positions[target_index - 1],
                axis_cal=axis_cal,
                channel=channel,
            ),
        ]
        if final_index != target_index or scenario == "switch_and_return":
            steps.append(
                self._build_validation_section_move_step(
                    role=("return_section" if scenario == "switch_and_return" else "measure_section"),
                    phase=ValidationPhase.MOVE_TO_RETURN_SECTION,
                    section_index=final_index,
                    z_pos_mm=positions[final_index - 1],
                    axis_cal=axis_cal,
                    channel=channel,
                )
            )
        return steps

    def _validation_section_index(
        self,
        request: FixedSectionRepeatabilityRequest,
        field_name: str,
        section_count: int,
    ) -> int:
        try:
            index = int(getattr(request, field_name, 1) or 1)
        except Exception as exc:
            raise ValueError(f"{field_name} must be an integer") from exc
        if index < 1 or index > int(section_count):
            raise ValueError(f"{field_name} must be between 1 and {int(section_count)}")
        return index

    def _planned_measure_section_index(self, request: FixedSectionRepeatabilityRequest) -> int | None:
        if not bool(getattr(request, "move_enabled", False)):
            return None
        scenario = self._validation_move_scenario(request)
        if scenario not in {"switch_and_measure_target", "switch_and_return"}:
            return None
        positions = tuple(float(value) for value in plan_section_positions(self.recipe).positions_z)
        if not positions:
            raise ValueError("recipe section list is empty")
        field_name = (
            "move_target_section_index"
            if scenario == "switch_and_measure_target"
            else "move_return_section_index"
        )
        return self._validation_section_index(request, field_name, len(positions))

    @staticmethod
    def _resolve_fixed_section_measured_z_pos_mm(
        section_result: MeasureRow,
        raw_points: list[Mapping[str, Any]],
    ) -> float:
        for point in raw_points:
            if not isinstance(point, Mapping):
                continue
            try:
                z_pos_mm = float(point.get("z_pos_mm"))
            except Exception:
                continue
            if math.isfinite(z_pos_mm):
                return float(z_pos_mm)
        try:
            z_pos_mm = float(getattr(section_result, "x_ui", 0.0) or 0.0)
        except Exception:
            z_pos_mm = 0.0
        if not math.isfinite(z_pos_mm):
            raise ValueError("captured measured z position is invalid")
        return float(z_pos_mm)

    def _resolve_fixed_section_measure_metadata(
        self,
        *,
        request: FixedSectionRepeatabilityRequest,
        section_result: MeasureRow,
        raw_points: list[Mapping[str, Any]],
    ) -> tuple[int | None, str, float]:
        measured_z_pos_mm = self._resolve_fixed_section_measured_z_pos_mm(section_result, raw_points)
        resolved = resolve_measured_section(
            self.recipe,
            measured_z_pos_mm=measured_z_pos_mm,
            measure_section_index=self._planned_measure_section_index(request),
        )
        return (
            resolved.measure_section_index,
            resolved.measure_section_name,
            resolved.measured_z_pos_mm,
        )

    @staticmethod
    def _normalize_fixed_section_raw_points(
        raw_points: list[Mapping[str, Any]],
        *,
        measure_section_index: int | None,
        measure_section_name: str,
        measured_z_pos_mm: float,
    ) -> tuple[dict[str, Any], ...]:
        normalized: list[dict[str, Any]] = []
        for point in raw_points:
            point_dict = dict(point)
            point_dict["section_idx"] = (
                None if measure_section_index is None else int(measure_section_index)
            )
            point_dict["section_name"] = str(measure_section_name)
            point_dict["measure_section_index"] = (
                None if measure_section_index is None else int(measure_section_index)
            )
            point_dict["measure_section_name"] = str(measure_section_name)
            point_dict["measured_z_pos_mm"] = float(measured_z_pos_mm)
            point_dict["z_pos_mm"] = float(measured_z_pos_mm)
            normalized.append(point_dict)
        return tuple(normalized)

    @staticmethod
    def _optional_finite_float(value: Any) -> float | None:
        try:
            numeric = float(value)
        except Exception:
            return None
        if not math.isfinite(numeric):
            return None
        return float(numeric)

    def _build_validation_fit_result(
        self,
        *,
        measure_section_index: int | None,
        measure_section_name: str,
        measured_z_pos_mm: float,
        fit_payload: Mapping[str, Any] | None,
    ) -> ValidationFitResult | None:
        if fit_payload is None:
            return None
        return ValidationFitResult(
            measure_section_index=measure_section_index,
            measure_section_name=str(measure_section_name),
            measured_z_pos_mm=float(measured_z_pos_mm),
            od_center_x_mm=self._optional_finite_float(fit_payload.get("od_center_x_mm")),
            od_center_y_mm=self._optional_finite_float(fit_payload.get("od_center_y_mm")),
            od_radius_mm=self._optional_finite_float(fit_payload.get("od_radius_mm")),
            od_diameter_fit_mm=self._optional_finite_float(fit_payload.get("od_diameter_fit_mm")),
            id_center_x_mm=self._optional_finite_float(fit_payload.get("id_center_x_mm")),
            id_center_y_mm=self._optional_finite_float(fit_payload.get("id_center_y_mm")),
            id_radius_mm=self._optional_finite_float(fit_payload.get("id_radius_mm")),
            id_diameter_fit_mm=self._optional_finite_float(fit_payload.get("id_diameter_fit_mm")),
            od_ecc_mm=self._optional_finite_float(fit_payload.get("od_ecc_mm")),
            id_ecc_mm=self._optional_finite_float(fit_payload.get("id_ecc_mm")),
            concentricity_mm=self._optional_finite_float(fit_payload.get("concentricity_mm")),
        )

    def _build_validation_section_move_step(
        self,
        *,
        role: str,
        phase: ValidationPhase,
        section_index: int,
        z_pos_mm: float,
        axis_cal: AxisCal,
        channel: str,
    ) -> _ValidationSectionMoveStep:
        planned_targets = self._resolve_validation_all_section_targets(axis_cal, float(z_pos_mm))
        axes = _VALIDATION_MOVE_CHANNEL_AXES[channel]
        return _ValidationSectionMoveStep(
            role=str(role),
            phase=phase,
            section_index=int(section_index),
            z_pos_mm=float(z_pos_mm),
            planned_targets=planned_targets,
            move_targets={int(axis): float(planned_targets[int(axis)]) for axis in axes},
        )

    def _build_validation_move_plan(self, channel: str, move_away_delta_mm: float) -> _ValidationMovePlan:
        axes = tuple(_VALIDATION_MOVE_CHANNEL_AXES[channel])
        initial_positions = self._read_validation_axes_positions(axes)
        if channel in _VALIDATION_DEBUG_AXIS_CHANNELS:
            axis = _VALIDATION_DEBUG_AXIS_CHANNELS[channel]
            return_target = float(initial_positions[axis])
            return _ValidationMovePlan(
                channel=channel,
                axes=axes,
                return_targets={axis: return_target},
                away_targets={axis: return_target + float(move_away_delta_mm)},
                initial_positions=initial_positions,
            )

        axis_cal = self._get_validation_axis_cal()
        return_z = self._validation_current_od_z_disp(axis_cal, channel, initial_positions)
        away_z = return_z + float(move_away_delta_mm)
        return_targets = self._resolve_validation_channel_targets(axis_cal, channel, return_z)
        away_targets = self._resolve_validation_channel_targets(axis_cal, channel, away_z)
        return _ValidationMovePlan(
            channel=channel,
            axes=axes,
            return_targets=return_targets,
            away_targets=away_targets,
            initial_positions=initial_positions,
            return_z_disp_mm=return_z,
            away_z_disp_mm=away_z,
        )

    def _validation_current_od_z_disp(
        self,
        axis_cal: AxisCal,
        channel: str,
        initial_positions: Mapping[int, float],
    ) -> float:
        if channel in {"od_channel", "od_id_sync"}:
            return float(axis_cal.abs_to_z_disp(0, float(initial_positions[0])))
        if channel == "id_channel":
            z1_raw = float(axis_cal.abs_to_z_raw(1, float(initial_positions[1])))
            z4_raw = float(axis_cal.abs_to_z_raw(4, float(initial_positions[4])))
            z_id_disp = float(axis_cal.z_raw_to_z_disp(z1_raw + z4_raw))
            return float(z_id_disp - float(getattr(axis_cal, "b14", 0.0)))
        raise ValueError(f"unsupported move_channel: {channel}")

    def _resolve_validation_channel_targets(
        self,
        axis_cal: AxisCal,
        channel: str,
        z_od_disp_mm: float,
    ) -> dict[int, float]:
        section_targets = self._resolve_validation_all_section_targets(axis_cal, float(z_od_disp_mm))
        axes = _VALIDATION_MOVE_CHANNEL_AXES[channel]
        return {int(axis): float(section_targets[int(axis)]) for axis in axes}

    def _resolve_validation_all_section_targets(
        self,
        axis_cal: AxisCal,
        z_od_disp_mm: float,
    ) -> dict[int, float]:
        ax2_abs = self._get_validation_ax2_keepout_reference_abs()
        soft_limits = self._get_validation_soft_limits_abs((0, 1, 4))
        return resolve_section_targets(
            axis_cal,
            float(z_od_disp_mm),
            ax2_abs=float(ax2_abs),
            soft_limits_abs=soft_limits,
        ).linear_targets()

    def _get_validation_axis_cal(self) -> AxisCal:
        get_axis_cal = getattr(self.gateway, "get_axis_cal", None)
        if not callable(get_axis_cal):
            raise RuntimeError("validation AxisCal is not available")
        axis_cal = get_axis_cal()
        if not isinstance(axis_cal, AxisCal):
            required = getattr(axis_cal, "od_z_disp_to_targets", None)
            if not callable(required):
                raise RuntimeError("validation AxisCal is invalid")
        return axis_cal

    def _get_validation_ax2_keepout_reference_abs(self) -> float:
        get_ref = getattr(self.gateway, "get_ax2_keepout_reference_abs", None)
        if callable(get_ref):
            value = float(get_ref())
        else:
            value = self._read_validation_axis_position(2)
        if not math.isfinite(value):
            raise RuntimeError("AX2 keepout reference is invalid")
        return value

    def _get_validation_soft_limits_abs(self, axes: tuple[int, ...]) -> Mapping[int, tuple[float, float]]:
        get_limits = getattr(self.gateway, "get_soft_limits_abs", None)
        if not callable(get_limits):
            return {}
        limits = get_limits(tuple(int(axis) for axis in axes))
        return dict(limits or {})

    def _validation_move_payload(
        self,
        plan: _ValidationMovePlan,
        target_positions: Mapping[int, float],
        actual_positions: Mapping[int, float],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "move_channel": plan.channel,
            "axis_names": tuple(f"AX{int(axis)}" for axis in plan.axes),
            "target_positions_mm": self._axis_position_payload(target_positions),
            "actual_positions_mm": self._axis_position_payload(actual_positions),
            "return_target_positions_mm": self._axis_position_payload(plan.return_targets),
        }
        if len(plan.axes) == 1:
            axis = int(plan.axes[0])
            payload["axis_name"] = f"AX{axis}"
            payload["target_position_mm"] = float(target_positions[axis])
            payload["actual_position_mm"] = float(actual_positions[axis])
            payload["return_target_position_mm"] = float(plan.return_targets[axis])
        if plan.return_z_disp_mm is not None:
            payload["return_z_disp_mm"] = float(plan.return_z_disp_mm)
        if plan.away_z_disp_mm is not None:
            payload["away_z_disp_mm"] = float(plan.away_z_disp_mm)
        return payload

    def _validation_section_move_payload(
        self,
        *,
        request: FixedSectionRepeatabilityRequest,
        scenario: str,
        channel: str,
        step: _ValidationSectionMoveStep,
        actual_positions: Mapping[int, float],
        actual_after_wait: Mapping[int, float] | None,
    ) -> dict[str, Any]:
        from_index = int(getattr(request, "move_from_section_index", 1) or 1)
        target_index = int(getattr(request, "move_target_section_index", 1) or 1)
        return_index = int(getattr(request, "move_return_section_index", 1) or 1)
        payload: dict[str, Any] = {
            "move_scenario": str(scenario),
            "move_channel": str(channel),
            "step_role": str(step.role),
            "section_index": int(step.section_index),
            "from_section_index": from_index,
            "target_section_index": target_index,
            "return_section_index": return_index,
            "z_pos_mm": float(step.z_pos_mm),
            "planned_targets_mm": self._axis_position_payload(step.planned_targets),
            "target_positions_mm": self._axis_position_payload(step.move_targets),
            "actual_positions_mm": self._axis_position_payload(actual_positions),
        }
        if str(scenario) == "switch_and_measure_target":
            measured_section = resolve_recipe_section(self.recipe, section_index=target_index)
            payload["measure_section_index"] = measured_section.measure_section_index
            payload["measure_section_name"] = measured_section.measure_section_name
            payload["measured_z_pos_mm"] = measured_section.measured_z_pos_mm
        elif str(scenario) == "switch_and_return":
            measured_section = resolve_recipe_section(self.recipe, section_index=return_index)
            payload["measure_section_index"] = measured_section.measure_section_index
            payload["measure_section_name"] = measured_section.measure_section_name
            payload["measured_z_pos_mm"] = measured_section.measured_z_pos_mm
        if actual_after_wait is not None:
            payload["actual_positions_after_wait_mm"] = self._axis_position_payload(actual_after_wait)
        return payload

    @staticmethod
    def _axis_position_payload(positions: Mapping[int, float]) -> dict[str, float]:
        return {f"AX{int(axis)}": float(value) for axis, value in sorted(positions.items())}

    def _read_validation_axis_position(self, axis: int) -> float:
        read_position = getattr(self.gateway, "read_axis_position_mm", None)
        if not callable(read_position):
            raise RuntimeError("validation axis position feedback is not available")
        position = float(read_position(int(axis)))
        if not math.isfinite(position):
            raise RuntimeError(f"AX{int(axis)} position feedback is invalid")
        return position

    def _read_validation_axes_positions(self, axes: tuple[int, ...]) -> dict[int, float]:
        return {int(axis): self._read_validation_axis_position(int(axis)) for axis in axes}

    def _move_validation_axes_absolute(
        self,
        targets_abs: Mapping[int, float],
        *,
        context: str,
    ) -> dict[int, float]:
        move_absolute = getattr(self.gateway, "move_axes_absolute", None)
        if not callable(move_absolute):
            raise RuntimeError("validation synchronized absolute move action is not available")
        actual_targets = dict(move_absolute(dict(targets_abs), context=context))
        for axis, target in actual_targets.items():
            if not math.isfinite(float(target)):
                raise RuntimeError(f"AX{int(axis)} move target is invalid")
        return {int(axis): float(target) for axis, target in actual_targets.items()}

    def _wait_validation_axes_in_position(self, targets_abs: Mapping[int, float]) -> dict[int, float]:
        wait_in_position = getattr(self.gateway, "wait_axes_in_position", None)
        if not callable(wait_in_position):
            raise RuntimeError("validation synchronized in-position wait is not available")
        actual_positions = dict(
            wait_in_position(
                dict(targets_abs),
                tolerance_mm=_VALIDATION_MOVE_IN_POSITION_TOLERANCE_MM,
                timeout_s=_VALIDATION_MOVE_IN_POSITION_TIMEOUT_S,
                poll_interval_s=_VALIDATION_MOVE_IN_POSITION_POLL_S,
            )
        )
        for axis, position in actual_positions.items():
            if not math.isfinite(float(position)):
                raise RuntimeError(f"AX{int(axis)} in-position feedback is invalid")
        return {int(axis): float(position) for axis, position in actual_positions.items()}

    def _run_validation_restore_rotation_ready(
        self,
        *,
        request: FixedSectionRepeatabilityRequest,
        repeat_index: int,
        total: int,
        phase_callback: Callable[[PhaseEvent], None] | None,
    ) -> None:
        self.record_phase(
            ValidationPhase.RESTORE_ROTATION_READY,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"restore_rotation_ready repeat {repeat_index}/{total}",
            phase_callback=phase_callback,
        )
        self._start_validation_rotation(request)
        if not self._wait_validation_rotation_ready():
            raise RuntimeError("AX3 验证旋转未建立，无法开始采样")

    def _run_validation_position_settle(
        self,
        *,
        request: FixedSectionRepeatabilityRequest,
        repeat_index: int,
        total: int,
        phase_callback: Callable[[PhaseEvent], None] | None,
        wait_callback: WaitCallback | None,
    ) -> None:
        position_settle_s = float(getattr(request, "position_settle_s", 0.0) or 0.0)
        self.record_phase(
            ValidationPhase.WAIT_POSITION_SETTLE,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"wait_position_settle {position_settle_s:.3f}s",
            phase_callback=phase_callback,
        )
        self._wait_validation_action(
            position_settle_s,
            wait_phase=ValidationPhase.WAIT_POSITION_SETTLE,
            repeat_index=repeat_index,
            total=total,
            wait_callback=wait_callback,
        )

    def _start_validation_rotation(self, request: FixedSectionRepeatabilityRequest) -> None:
        velocity = self._get_validation_rotation_velocity(request)
        velmove = getattr(self.gateway, "velmove", None)
        if not callable(velmove):
            raise RuntimeError("validation rotation restart action is not available")
        velmove(3, float(velocity))

    def _get_validation_rotation_velocity(self, request: FixedSectionRepeatabilityRequest) -> float:
        if not hasattr(request, "validation_ax3_speed_dps"):
            velocity = 60.0
        else:
            raw_velocity = request.validation_ax3_speed_dps
            try:
                velocity = float(raw_velocity)
            except Exception as exc:
                raise ValueError("validation_ax3_speed_dps must be a number") from exc
        if not math.isfinite(velocity) or abs(velocity) <= 1e-9:
            raise ValueError("validation_ax3_speed_dps must be > 0")
        return float(velocity)

    def _wait_validation_rotation_ready(self) -> bool:
        read_angle = getattr(self.gateway, "read_axis_angle_deg_sync", None)
        if not callable(read_angle):
            raise RuntimeError("validation rotation angle feedback is not available")

        deadline = time.monotonic() + _ROTATION_READY_TIMEOUT_S
        previous_angle: float | None = None
        while time.monotonic() < deadline:
            try:
                angle = read_angle(axis=3, timeout_s=0.2)
            except TypeError:
                angle = read_angle(3)
            if angle is not None:
                current_angle = float(angle)
                if previous_angle is not None:
                    delta = self._angle_delta_abs_deg(previous_angle, current_angle)
                    if delta >= _ROTATION_READY_MIN_DELTA_DEG:
                        return True
                previous_angle = current_angle
            self._wait_validation_action(min(_ROTATION_READY_POLL_S, max(0.0, deadline - time.monotonic())))
        return False

    def _wait_validation_action(
        self,
        duration_s: float,
        *,
        wait_phase: ValidationPhase | str | None = None,
        repeat_index: int | None = None,
        total: int | None = None,
        wait_callback: WaitCallback | None = None,
    ) -> None:
        duration = max(0.0, float(duration_s or 0.0))
        phase_name = ""
        if wait_phase is not None:
            phase_name = self._coerce_phase(wait_phase).value
        if callable(wait_callback) and phase_name and repeat_index is not None and total is not None:
            wait_callback(phase_name, int(repeat_index), int(total), float(duration))
        if duration <= 0.0:
            return
        wait_cancelable = getattr(self.gateway, "wait_cancelable", None)
        if not callable(wait_cancelable):
            raise RuntimeError("validation cancel-aware wait is not available")
        if not (callable(wait_callback) and phase_name and repeat_index is not None and total is not None):
            wait_cancelable(duration)
            return
        remaining_s = float(duration)
        while remaining_s > 1e-9:
            step_s = min(_VALIDATION_WAIT_PROGRESS_SLICE_S, remaining_s)
            wait_cancelable(step_s)
            remaining_s = max(0.0, remaining_s - step_s)
            if callable(wait_callback) and phase_name and repeat_index is not None and total is not None:
                wait_callback(phase_name, int(repeat_index), int(total), float(remaining_s))

    @staticmethod
    def _angle_delta_abs_deg(previous_angle: float, current_angle: float) -> float:
        delta = (float(current_angle) - float(previous_angle) + 180.0) % 360.0 - 180.0
        return abs(delta)

    @staticmethod
    def _resolve_capture_time_range(
        *,
        raw_points: list[Mapping[str, Any]],
        windows_payload: list[Mapping[str, Any]],
    ) -> tuple[float | None, float | None]:
        start_candidates: list[float] = []
        end_candidates: list[float] = []
        for window in windows_payload:
            if not isinstance(window, Mapping):
                continue
            ts_start = window.get("ts_start")
            ts_end = window.get("ts_end")
            if ts_start is not None:
                try:
                    start_value = float(ts_start)
                except Exception:
                    start_value = math.nan
                if math.isfinite(start_value):
                    start_candidates.append(start_value)
            if ts_end is not None:
                try:
                    end_value = float(ts_end)
                except Exception:
                    end_value = math.nan
                if math.isfinite(end_value):
                    end_candidates.append(end_value)
        for point in raw_points:
            if not isinstance(point, Mapping):
                continue
            ts_value = point.get("ts")
            if ts_value is None:
                continue
            try:
                sample_ts = float(ts_value)
            except Exception:
                sample_ts = math.nan
            if math.isfinite(sample_ts):
                start_candidates.append(sample_ts)
                end_candidates.append(sample_ts)
        capture_start_ts = min(start_candidates) if start_candidates else None
        capture_end_ts = max(end_candidates) if end_candidates else None
        return capture_start_ts, capture_end_ts

    def _run_fixed_section_capture(
        self,
        *,
        request: FixedSectionRepeatabilityRequest,
        repeat_index: int,
        total: int,
        phase_callback: Callable[[PhaseEvent], None] | None,
        wait_callback: WaitCallback | None,
    ) -> _FixedSectionCapturePayload:
        sample_delay_s = float(getattr(request, "sample_delay_s", 0.0) or 0.0)
        self.record_phase(
            ValidationPhase.WAIT_SAMPLE_DELAY,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"wait_sample_delay {sample_delay_s:.3f}s",
            phase_callback=phase_callback,
        )
        self._wait_validation_action(
            sample_delay_s,
            wait_phase=ValidationPhase.WAIT_SAMPLE_DELAY,
            repeat_index=repeat_index,
            total=total,
            wait_callback=wait_callback,
        )
        self.record_phase(
            ValidationPhase.CAPTURE,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"capture repeat {repeat_index}/{total}",
            phase_callback=phase_callback,
        )
        capture_result = measure_current_position_section_capture(
            gateway=self.gateway,
            recipe=self.recipe,
            calibration=self.calibration,
        )
        if len(capture_result) == 4:
            section_result, raw_points, windows_payload, coverage_payload = capture_result
            fit_payload = None
        elif len(capture_result) == 5:
            section_result, raw_points, windows_payload, coverage_payload, fit_payload = capture_result
        else:
            raise RuntimeError("validation capture returned unexpected payload shape")
        capture_start_ts, capture_end_ts = self._resolve_capture_time_range(
            raw_points=list(raw_points or []),
            windows_payload=list(windows_payload or []),
        )
        return _FixedSectionCapturePayload(
            section_result=section_result,
            raw_points=list(raw_points or []),
            windows_payload=list(windows_payload or []),
            coverage_payload=dict(coverage_payload or {}),
            capture_start_ts=capture_start_ts,
            capture_end_ts=capture_end_ts,
            measured_at_ts=float(time.time()),
            fit_payload=(None if fit_payload is None else dict(fit_payload or {})),
        )

    def _run_fixed_section_fit_calc(
        self,
        *,
        request: FixedSectionRepeatabilityRequest,
        repeat_index: int,
        total: int,
        section_name: str,
        metric_name: str,
        capture_payload: _FixedSectionCapturePayload,
        phase_callback: Callable[[PhaseEvent], None] | None,
    ) -> tuple[FixedSectionRepeatRow, FixedSectionRepeatCapture]:
        self.record_phase(
            ValidationPhase.FIT_CALC,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"fit_calc repeat {repeat_index}/{total}",
            phase_callback=phase_callback,
        )
        _validate_fixed_section_od_sampling(list(capture_payload.raw_points or []))
        measured_value_mm = _extract_primary_metric_value(capture_payload.section_result, metric_name)
        settle_s_used = float(getattr(request, "position_settle_s", 0.0) or 0.0)
        sample_delay_s_used = float(getattr(request, "sample_delay_s", 0.0) or 0.0)
        measure_section_index, measure_section_name, measured_z_pos_mm = self._resolve_fixed_section_measure_metadata(
            request=request,
            section_result=capture_payload.section_result,
            raw_points=list(capture_payload.raw_points or []),
        )
        if not measure_section_name:
            measure_section_name = str(section_name or "")
        normalized_raw_points = self._normalize_fixed_section_raw_points(
            list(capture_payload.raw_points or []),
            measure_section_index=measure_section_index,
            measure_section_name=measure_section_name,
            measured_z_pos_mm=measured_z_pos_mm,
        )
        fit_result = self._build_validation_fit_result(
            measure_section_index=measure_section_index,
            measure_section_name=measure_section_name,
            measured_z_pos_mm=measured_z_pos_mm,
            fit_payload=capture_payload.fit_payload,
        )
        row = FixedSectionRepeatRow(
            repeat_index=repeat_index,
            section_name=measure_section_name,
            metric_name=metric_name,
            measured_value_mm=measured_value_mm,
            settle_s_used=settle_s_used,
            sample_delay_s_used=sample_delay_s_used,
            capture_start_ts=capture_payload.capture_start_ts,
            capture_end_ts=capture_payload.capture_end_ts,
            measured_at_ts=capture_payload.measured_at_ts,
            measure_section_index=measure_section_index,
            measure_section_name=measure_section_name,
            measured_z_pos_mm=measured_z_pos_mm,
        )
        windows = tuple(
            FixedSectionWindow(
                repeat_index=repeat_index,
                window_index=int(window.get("window_index", 0) or 0),
                window_role=str(window.get("window_role", "") or ""),
                point_start_index=(
                    None if window.get("point_start_index") is None else int(window.get("point_start_index"))
                ),
                point_end_index=(
                    None if window.get("point_end_index") is None else int(window.get("point_end_index"))
                ),
                point_count=int(window.get("point_count", 0) or 0),
                ts_start=(None if window.get("ts_start") is None else float(window.get("ts_start"))),
                ts_end=(None if window.get("ts_end") is None else float(window.get("ts_end"))),
                theta_start_deg=(
                    None if window.get("theta_start_deg") is None else float(window.get("theta_start_deg"))
                ),
                theta_end_deg=(
                    None if window.get("theta_end_deg") is None else float(window.get("theta_end_deg"))
                ),
                theta_span_deg=float(window.get("theta_span_deg", 0.0) or 0.0),
                filled_bins=(None if window.get("filled_bins") is None else int(window.get("filled_bins"))),
                total_bins=(None if window.get("total_bins") is None else int(window.get("total_bins"))),
                miss_bins=(None if window.get("miss_bins") is None else int(window.get("miss_bins"))),
                n_od=(None if window.get("n_od") is None else int(window.get("n_od"))),
                n_id=(None if window.get("n_id") is None else int(window.get("n_id"))),
                reason=str(window.get("reason", "") or ""),
                revs=(None if window.get("revs") is None else float(window.get("revs"))),
                elapsed_s=(None if window.get("elapsed_s") is None else float(window.get("elapsed_s"))),
                max_gap_deg=(None if window.get("max_gap_deg") is None else float(window.get("max_gap_deg"))),
            )
            for window in (capture_payload.windows_payload or [])
        )
        capture = FixedSectionRepeatCapture(
            repeat_index=repeat_index,
            section_name=measure_section_name,
            metric_name=metric_name,
            measured_at_ts=capture_payload.measured_at_ts,
            measured_value_mm=float(measured_value_mm),
            settle_s_used=settle_s_used,
            sample_delay_s_used=sample_delay_s_used,
            capture_start_ts=capture_payload.capture_start_ts,
            capture_end_ts=capture_payload.capture_end_ts,
            section_result=capture_payload.section_result,
            windows=windows,
            raw_points=normalized_raw_points,
            coverage=dict(capture_payload.coverage_payload or {}),
            measure_section_index=measure_section_index,
            measure_section_name=measure_section_name,
            measured_z_pos_mm=measured_z_pos_mm,
            fit_result=fit_result,
        )
        return row, capture

    def _run_fixed_section_save_result(
        self,
        *,
        request: FixedSectionRepeatabilityRequest,
        repeat_index: int,
        total: int,
        row: FixedSectionRepeatRow,
        capture: FixedSectionRepeatCapture,
        rows: list[FixedSectionRepeatRow],
        local_session: FixedSectionRepeatabilitySession,
        progress_callback: Callable[[int, int], None] | None,
        phase_callback: Callable[[PhaseEvent], None] | None,
    ) -> None:
        self.record_phase(
            ValidationPhase.SAVE_RESULT,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"save_result repeat {repeat_index}/{total}",
            phase_callback=phase_callback,
        )
        rows.append(row)
        self._fixed_section_repeat_rows.append(row)
        self._fixed_section_repeat_captures.append(capture)
        self.runtime_state.rows.append(capture.section_result)
        self.runtime_state.raw_points.extend(dict(point) for point in capture.raw_points)
        local_session.completed_repeat_count = repeat_index
        local_session.rows_cache.append(
            {
                "row": asdict(row),
                "section_result": asdict(capture.section_result),
                "window_count": len(capture.windows),
                "raw_point_count": len(capture.raw_points),
            }
        )
        self.record_progress(
            step=str(request.task_name or "fixed_section_repeatability"),
            index=repeat_index,
            total=total,
            message=f"{row.metric_name} repeat {repeat_index}/{total}",
        )
        if callable(progress_callback):
            progress_callback(repeat_index, total)

    def _sync_runtime_from_session(self) -> None:
        session = self.validation_session
        if session is None:
            return
        if self.runtime_state.serial is None:
            self.runtime_state.serial = session.serial
        if self.runtime_state.run_id is None:
            self.runtime_state.run_id = session.run_id
        if self.runtime_state.started_at_ts is None:
            self.runtime_state.started_at_ts = session.start_ts
        if self.runtime_state.finished_at_ts is None:
            self.runtime_state.finished_at_ts = session.end_ts
        if not self.runtime_state.summary and session.summary_cache:
            self.runtime_state.summary.update(dict(session.summary_cache))

    def _sync_session_from_runtime(self) -> None:
        session = self.validation_session
        if session is None:
            return
        session.serial = self.runtime_state.serial
        session.run_id = self.runtime_state.run_id
        session.start_ts = self.runtime_state.started_at_ts
        session.end_ts = self.runtime_state.finished_at_ts
        session.summary_cache = dict(self.runtime_state.summary)

    @staticmethod
    def _normalize_status(state: str) -> str:
        normalized = str(state or 'IDLE').upper()
        if normalized in {'PREP', 'LEN'}:
            return 'preparing'
        if normalized == 'RUN':
            return 'running'
        if normalized == 'STOPPING':
            return 'stopping'
        if normalized == 'DONE':
            return 'completed'
        if normalized == 'ERR':
            return 'error'
        if normalized == 'STOP':
            return 'idle'
        return normalized.lower()

    @staticmethod
    def _coerce_phase(phase: ValidationPhase | str) -> ValidationPhase:
        if isinstance(phase, ValidationPhase):
            return phase
        raw = str(phase or "").strip()
        for item in ValidationPhase:
            if raw == item.value or raw.upper() == item.name:
                return item
        raise ValueError(f"unknown validation phase: {phase!r}")


__all__ = [
    'FIXED_SECTION_PRIMARY_METRICS',
    'FixedSectionRepeatabilityRequest',
    'FixedSectionRepeatCapture',
    'FixedSectionRepeatRow',
    'FixedSectionWindow',
    'PhaseEvent',
    'ProgressEvent',
    'StateEvent',
    'SummaryEvent',
    'SummaryPayload',
    'TypedEvent',
    'ValidationPhase',
    'ValidationResult',
    'ValidationResultStatus',
    'ValidationWorkflow',
    'ValidationWorkflowEventType',
    '_summarize_fixed_section_repeatability_rows',
]
