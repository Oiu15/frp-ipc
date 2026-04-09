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
    SUMMARY = "summary"


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
class SummaryEvent:
    source: str
    payload: Mapping[str, Any]
    type: ValidationWorkflowEventType = ValidationWorkflowEventType.SUMMARY


TypedEvent: TypeAlias = StateEvent | ProgressEvent | SummaryEvent


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


_MIN_VALID_OD_SAMPLE_COUNT = 6
_MIN_VALID_OD_BIN_COUNT = 6
_MIN_VALID_ROTATION_SPAN_DEG = 30.0


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
    ) -> tuple[list[FixedSectionRepeatRow], dict[str, Any]]:
        identity = self.ensure_identity()
        self.runtime_state.rows.clear()
        self.runtime_state.raw_points.clear()
        self.runtime_state.summary.clear()
        self._fixed_section_repeat_captures.clear()
        total = int(request.repeat_count or 3)
        section_name = str(request.section_name or "")
        metric_name = str(request.metric_name or "")
        local_session = FixedSectionRepeatabilitySession(
            section_name=section_name,
            metric_name=metric_name,
            requested_repeat_count=int(request.repeat_count or total),
        )

        self.record_state("RUN", f"{request.task_name} running")
        base_ts = float(time.time())
        rows: list[FixedSectionRepeatRow] = []
        try:
            for repeat_index in range(1, total + 1):
                section_result, raw_points, windows_payload, coverage_payload = measure_current_position_section_capture(
                    gateway=self.gateway,
                    recipe=self.recipe,
                    calibration=self.calibration,
                )
                _validate_fixed_section_od_sampling(list(raw_points or []))
                measured_at_ts = float(time.time())
                measured_value_mm = _extract_primary_metric_value(section_result, metric_name)
                row = FixedSectionRepeatRow(
                    repeat_index=repeat_index,
                    section_name=section_name,
                    metric_name=metric_name,
                    measured_value_mm=measured_value_mm,
                    measured_at_ts=measured_at_ts,
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
                    for window in (windows_payload or [])
                )
                capture = FixedSectionRepeatCapture(
                    repeat_index=repeat_index,
                    section_name=section_name,
                    metric_name=metric_name,
                    measured_at_ts=measured_at_ts,
                    measured_value_mm=float(measured_value_mm),
                    section_result=section_result,
                    windows=windows,
                    raw_points=tuple(dict(point) for point in (raw_points or [])),
                    coverage=dict(coverage_payload or {}),
                )
                rows.append(row)
                self._fixed_section_repeat_captures.append(capture)
                self.runtime_state.rows.append(section_result)
                self.runtime_state.raw_points.extend(dict(point) for point in capture.raw_points)
                local_session.completed_repeat_count = repeat_index
                local_session.rows_cache.append(
                    {
                        "row": asdict(row),
                        "section_result": asdict(section_result),
                        "window_count": len(windows),
                        "raw_point_count": len(capture.raw_points),
                    }
                )
                self.record_progress(
                    step=str(request.task_name or "fixed_section_repeatability"),
                    index=repeat_index,
                    total=total,
                    message=f"{metric_name} repeat {repeat_index}/{total}",
                )
                if callable(progress_callback):
                    progress_callback(repeat_index, total)

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


__all__ = [
    'FIXED_SECTION_PRIMARY_METRICS',
    'FixedSectionRepeatabilityRequest',
    'FixedSectionRepeatCapture',
    'FixedSectionRepeatRow',
    'FixedSectionWindow',
    'ProgressEvent',
    'StateEvent',
    'SummaryEvent',
    'SummaryPayload',
    'TypedEvent',
    'ValidationResult',
    'ValidationResultStatus',
    'ValidationWorkflow',
    'ValidationWorkflowEventType',
    '_summarize_fixed_section_repeatability_rows',
]
