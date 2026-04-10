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
    ValidationSession,
)
from core.models import MeasureRow, Recipe
from frp_workflow.autoflow_orchestrator import measure_current_position_section_capture

SummaryPayload: TypeAlias = dict[str, Any]
ValidationResultStatus = Literal["DONE", "STOP", "ERR"]


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
    RESTORE_ROTATION_READY = "restore_rotation_ready"
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
    validation_ax3_speed_dps: float = 60.0
    move_enabled: bool = False
    move_axis_name: str = "AX0"
    move_away_delta_mm: float = 0.0
    move_return_mode: str = "target_section"
    target_section_pos_mm: float = 0.0


@dataclass(frozen=True, slots=True)
class FixedSectionRepeatRow:
    repeat_index: int
    section_name: str
    metric_name: str
    measured_value_mm: float
    measured_at_ts: float


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
    section_result: MeasureRow
    windows: tuple[FixedSectionWindow, ...]
    raw_points: tuple[Mapping[str, Any], ...]
    coverage: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _FixedSectionCapturePayload:
    section_result: MeasureRow
    raw_points: list[dict[str, Any]]
    windows_payload: list[dict[str, Any]]
    coverage_payload: dict[str, Any]
    measured_at_ts: float


_MIN_VALID_OD_SAMPLE_COUNT = 6
_MIN_VALID_OD_BIN_COUNT = 6
_MIN_VALID_ROTATION_SPAN_DEG = 30.0
_ROTATION_READY_TIMEOUT_S = 2.0
_ROTATION_READY_POLL_S = 0.05
_ROTATION_READY_MIN_DELTA_DEG = 1.0
_VALIDATION_MOVE_IN_POSITION_TIMEOUT_S = 10.0
_VALIDATION_MOVE_IN_POSITION_TOLERANCE_MM = 0.1
_VALIDATION_MOVE_IN_POSITION_POLL_S = 0.05


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
    ) -> tuple[list[FixedSectionRepeatRow], dict[str, Any]]:
        identity = self.ensure_identity()
        self.runtime_state.rows.clear()
        self.runtime_state.raw_points.clear()
        self.runtime_state.summary.clear()
        self._fixed_section_repeat_captures.clear()
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
            validation_ax3_speed_dps=validation_ax3_speed_dps,
            move_enabled=bool(getattr(request, "move_enabled", False)),
            move_axis_name=str(getattr(request, "move_axis_name", "AX0") or "AX0"),
            move_away_delta_mm=float(getattr(request, "move_away_delta_mm", 0.0) or 0.0),
            move_return_mode=str(getattr(request, "move_return_mode", "target_section") or "target_section"),
            target_section_pos_mm=float(getattr(request, "target_section_pos_mm", 0.0) or 0.0),
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
                )
                capture_payload = self._run_fixed_section_capture(
                    request=request,
                    repeat_index=repeat_index,
                    total=total,
                    phase_callback=phase_callback,
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
        self._wait_validation_action(release_settle_s)
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
        self._wait_validation_action(clamp_settle_s)

    def _run_validation_section_relocation(
        self,
        *,
        request: FixedSectionRepeatabilityRequest,
        repeat_index: int,
        total: int,
        phase_callback: Callable[[PhaseEvent], None] | None,
    ) -> None:
        axis = self._validation_move_axis_index(request)
        move_away_delta_mm = self._validation_move_away_delta(request)
        return_mode = str(getattr(request, "move_return_mode", "target_section") or "target_section")
        initial_position = self._read_validation_axis_position(axis)
        if return_mode == "initial_position":
            return_target = initial_position
        elif return_mode == "target_section":
            return_target = self._validation_target_section_position(request)
        else:
            raise ValueError(f"unsupported move_return_mode: {return_mode}")

        axis_name = f"AX{axis}"
        away_target = return_target + move_away_delta_mm
        self.record_phase(
            ValidationPhase.MOVE_AWAY,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=(
                f"move_away AX{axis} target={away_target:.3f} "
                f"delta={move_away_delta_mm:.3f}"
            ),
            payload={
                "axis_name": axis_name,
                "target_position_mm": away_target,
                "actual_position_mm": initial_position,
                "return_target_position_mm": return_target,
                "move_return_mode": return_mode,
            },
            phase_callback=phase_callback,
        )
        actual_away_target = self._move_validation_axis_absolute(
            axis,
            away_target,
            context="VALIDATION_MOVE_AWAY",
        )
        actual_away_position = self._wait_validation_axis_in_position(axis, actual_away_target)
        self._notify_validation_phase_update(
            ValidationPhase.MOVE_AWAY,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"move_away reached AX{axis} actual={actual_away_position:.3f}",
            payload={
                "axis_name": axis_name,
                "target_position_mm": actual_away_target,
                "actual_position_mm": actual_away_position,
                "return_target_position_mm": return_target,
                "move_return_mode": return_mode,
            },
            phase_callback=phase_callback,
        )

        self.record_phase(
            ValidationPhase.MOVE_BACK_TO_TARGET,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"move_back_to_target AX{axis} target={return_target:.3f}",
            payload={
                "axis_name": axis_name,
                "target_position_mm": return_target,
                "actual_position_mm": actual_away_position,
                "return_target_position_mm": return_target,
                "move_return_mode": return_mode,
            },
            phase_callback=phase_callback,
        )
        actual_return_target = self._move_validation_axis_absolute(
            axis,
            return_target,
            context="VALIDATION_MOVE_BACK_TO_TARGET",
        )
        actual_return_position = self._wait_validation_axis_in_position(axis, actual_return_target)
        self._notify_validation_phase_update(
            ValidationPhase.MOVE_BACK_TO_TARGET,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"move_back_to_target reached AX{axis} actual={actual_return_position:.3f}",
            payload={
                "axis_name": axis_name,
                "target_position_mm": actual_return_target,
                "actual_position_mm": actual_return_position,
                "return_target_position_mm": return_target,
                "move_return_mode": return_mode,
            },
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

    def _validation_move_axis_index(self, request: FixedSectionRepeatabilityRequest) -> int:
        axis_name = str(getattr(request, "move_axis_name", "AX0") or "AX0").strip().upper()
        if not axis_name.startswith("AX"):
            raise ValueError(f"unsupported move_axis_name: {axis_name}")
        try:
            axis = int(axis_name[2:])
        except Exception as exc:
            raise ValueError(f"unsupported move_axis_name: {axis_name}") from exc
        if axis not in (0, 1, 2, 4):
            raise ValueError(f"unsupported move_axis_name: {axis_name}")
        return axis

    def _validation_move_away_delta(self, request: FixedSectionRepeatabilityRequest) -> float:
        try:
            delta = float(getattr(request, "move_away_delta_mm", 0.0) or 0.0)
        except Exception as exc:
            raise ValueError("move_away_delta_mm must be a number") from exc
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError("move_away_delta_mm must be > 0 when move_enabled")
        return delta

    def _validation_target_section_position(self, request: FixedSectionRepeatabilityRequest) -> float:
        try:
            target = float(getattr(request, "target_section_pos_mm", 0.0) or 0.0)
        except Exception as exc:
            raise ValueError("target_section_pos_mm must be a number") from exc
        if not math.isfinite(target):
            raise ValueError("target_section_pos_mm must be finite")
        return target

    def _read_validation_axis_position(self, axis: int) -> float:
        read_position = getattr(self.gateway, "read_axis_position_mm", None)
        if not callable(read_position):
            raise RuntimeError("validation axis position feedback is not available")
        position = float(read_position(int(axis)))
        if not math.isfinite(position):
            raise RuntimeError(f"AX{int(axis)} position feedback is invalid")
        return position

    def _move_validation_axis_absolute(self, axis: int, target_pos_mm: float, *, context: str) -> float:
        move_absolute = getattr(self.gateway, "move_axis_absolute", None)
        if not callable(move_absolute):
            raise RuntimeError("validation absolute move action is not available")
        actual_target = float(move_absolute(int(axis), float(target_pos_mm), context=context))
        if not math.isfinite(actual_target):
            raise RuntimeError(f"AX{int(axis)} move target is invalid")
        return actual_target

    def _wait_validation_axis_in_position(self, axis: int, target_pos_mm: float) -> float:
        wait_in_position = getattr(self.gateway, "wait_axis_in_position", None)
        if not callable(wait_in_position):
            raise RuntimeError("validation in-position wait is not available")
        actual_position = float(
            wait_in_position(
                int(axis),
                float(target_pos_mm),
                tolerance_mm=_VALIDATION_MOVE_IN_POSITION_TOLERANCE_MM,
                timeout_s=_VALIDATION_MOVE_IN_POSITION_TIMEOUT_S,
                poll_interval_s=_VALIDATION_MOVE_IN_POSITION_POLL_S,
            )
        )
        if not math.isfinite(actual_position):
            raise RuntimeError(f"AX{int(axis)} in-position feedback is invalid")
        return actual_position

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

    def _wait_validation_action(self, duration_s: float) -> None:
        duration = max(0.0, float(duration_s or 0.0))
        if duration <= 0.0:
            return
        wait_cancelable = getattr(self.gateway, "wait_cancelable", None)
        if not callable(wait_cancelable):
            raise RuntimeError("validation cancel-aware wait is not available")
        wait_cancelable(duration)

    @staticmethod
    def _angle_delta_abs_deg(previous_angle: float, current_angle: float) -> float:
        delta = (float(current_angle) - float(previous_angle) + 180.0) % 360.0 - 180.0
        return abs(delta)

    def _run_fixed_section_capture(
        self,
        *,
        request: FixedSectionRepeatabilityRequest,
        repeat_index: int,
        total: int,
        phase_callback: Callable[[PhaseEvent], None] | None,
    ) -> _FixedSectionCapturePayload:
        self.record_phase(
            ValidationPhase.CAPTURE,
            repeat_index=repeat_index,
            total=total,
            task_name=request.task_name,
            message=f"capture repeat {repeat_index}/{total}",
            phase_callback=phase_callback,
        )
        section_result, raw_points, windows_payload, coverage_payload = measure_current_position_section_capture(
            gateway=self.gateway,
            recipe=self.recipe,
            calibration=self.calibration,
        )
        return _FixedSectionCapturePayload(
            section_result=section_result,
            raw_points=list(raw_points or []),
            windows_payload=list(windows_payload or []),
            coverage_payload=dict(coverage_payload or {}),
            measured_at_ts=float(time.time()),
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
        row = FixedSectionRepeatRow(
            repeat_index=repeat_index,
            section_name=section_name,
            metric_name=metric_name,
            measured_value_mm=measured_value_mm,
            measured_at_ts=capture_payload.measured_at_ts,
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
            section_name=section_name,
            metric_name=metric_name,
            measured_at_ts=capture_payload.measured_at_ts,
            measured_value_mm=float(measured_value_mm),
            section_result=capture_payload.section_result,
            windows=windows,
            raw_points=tuple(dict(point) for point in (capture_payload.raw_points or [])),
            coverage=dict(capture_payload.coverage_payload or {}),
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
