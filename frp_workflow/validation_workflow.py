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
from typing import Any, Literal, Mapping, TypeAlias

import numpy as np

from application.contracts import MachineGateway, RunRepositoryProtocol
from application.state import (
    CalibrationSnapshot,
    FixedSectionRepeatabilitySession,
    RunIdentity,
    RuntimeState,
    ValidationExportContext,
    ValidationSession,
)
from core.models import Recipe
from services.autoflow_service import AutoFlow

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


def _validate_fixed_section_od_sampling(raw_points: list[dict[str, Any]]) -> None:
    valid_points: list[dict[str, Any]] = []
    theta_values_deg: list[float] = []
    unique_bins: set[int] = set()
    for point in raw_points or []:
        if not isinstance(point, dict):
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


def _normalize_sampling_failure(exc: Exception, legacy_flow: AutoFlow) -> RuntimeError:
    n_od = int(getattr(legacy_flow, "_last_sample_n_od", 0) or 0)
    sample_cov = getattr(legacy_flow, "_last_sample_cov", None)
    filled_bins = 0
    if isinstance(sample_cov, tuple) and len(sample_cov) >= 2:
        try:
            filled_bins = int(sample_cov[1] or 0)
        except Exception:
            filled_bins = 0

    if n_od < _MIN_VALID_OD_SAMPLE_COUNT:
        return RuntimeError("有效采样点不足，验证结果无效")
    if filled_bins < _MIN_VALID_OD_BIN_COUNT:
        return RuntimeError("未检测到有效旋转，无法完成固定截面重复性验证")
    return RuntimeError(str(exc))


def _measure_current_position_od_avg_validated(
    *,
    gateway: MachineGateway,
    recipe: Recipe,
    calibration: CalibrationSnapshot,
) -> float:
    runtime_app = getattr(gateway, "app", None)
    if runtime_app is None:
        raise RuntimeError("固定截面重复性验证需要 gateway.app")

    legacy_flow = AutoFlow(runtime_app)
    legacy_flow._current_recipe = recipe
    legacy_flow._calibration_snapshot = calibration

    try:
        coords_od, _coords_id, _raw_od, _raw_id, raw_points = legacy_flow._sample_circle_points_dual(
            recipe,
            section_idx=0,
            sample_od=True,
            sample_id=False,
            phase="VALIDATION_OD",
        )
    except Exception as exc:
        raise _normalize_sampling_failure(exc, legacy_flow) from exc

    _validate_fixed_section_od_sampling(raw_points)

    if bool(getattr(recipe, "od_use_edges", False)):
        od_vals = np.asarray(
            [
                float(point.get("od_mm"))
                for point in (raw_points or [])
                if isinstance(point, dict) and point.get("od_mm") is not None
            ],
            dtype=float,
        )
        if od_vals.size == 0:
            raise RuntimeError("有效采样点不足，验证结果无效")
        return float(np.mean(od_vals))

    if coords_od is None or len(coords_od) < 3:
        raise RuntimeError("有效采样点不足，验证结果无效")

    xc, yc, _r_fit, _sigma = legacy_flow._fit_circle(
        coords_od,
        weights=getattr(legacy_flow, "_last_fit_weights_od", None),
    )
    dx = coords_od[:, 0] - float(xc)
    dy = coords_od[:, 1] - float(yc)
    od_list = 2.0 * np.sqrt(dx * dx + dy * dy)
    if od_list.size == 0:
        raise RuntimeError("有效采样点不足，验证结果无效")
    if math.isclose(float(np.ptp(od_list)), 0.0, abs_tol=1e-12) and len(coords_od) < _MIN_VALID_OD_SAMPLE_COUNT:
        raise RuntimeError("有效采样点不足，验证结果无效")
    return float(np.mean(od_list))


def _summarize_fixed_section_repeatability_rows(
    rows: list[FixedSectionRepeatRow],
) -> dict[str, float | int | str]:
    def _rounded(value: float) -> float:
        return round(float(value), 6)

    values = [float(row.measured_value_mm) for row in rows]
    if not rows:
        return {
            "task_name": "fixed_section_repeatability",
            "section_name": "",
            "metric_name": "",
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "range": 0.0,
        }

    return {
        "task_name": "fixed_section_repeatability",
        "section_name": str(rows[0].section_name),
        "metric_name": str(rows[0].metric_name),
        "count": len(values),
        "mean": _rounded(statistics.fmean(values)),
        "std": _rounded(statistics.pstdev(values)),
        "min": _rounded(min(values)),
        "max": _rounded(max(values)),
        "range": _rounded(max(values) - min(values)),
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
    ) -> tuple[list[FixedSectionRepeatRow], dict[str, Any]]:
        identity = self.ensure_identity()
        self.runtime_state.summary.clear()
        total = 3
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
            if metric_name != "od_avg":
                raise ValueError("fixed_section_repeatability currently supports only metric_name='od_avg'")

            for repeat_index in range(1, total + 1):
                measured_value_mm = float(
                    _measure_current_position_od_avg_validated(
                        gateway=self.gateway,
                        recipe=self.recipe,
                        calibration=self.calibration,
                    )
                )
                measured_at_ts = float(time.time())
                row = FixedSectionRepeatRow(
                    repeat_index=repeat_index,
                    section_name=section_name,
                    metric_name=metric_name,
                    measured_value_mm=measured_value_mm,
                    measured_at_ts=measured_at_ts,
                )
                rows.append(row)
                local_session.completed_repeat_count = repeat_index
                local_session.rows_cache.append(asdict(row))
                self.record_progress(
                    step=str(request.task_name or "fixed_section_repeatability"),
                    index=repeat_index,
                    total=total,
                    message=f"{metric_name} repeat {repeat_index}/{total}",
                )

            summary = _summarize_fixed_section_repeatability_rows(rows)
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
    'FixedSectionRepeatabilityRequest',
    'FixedSectionRepeatRow',
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
