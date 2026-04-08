from __future__ import annotations

"""Pure production-workflow boundary objects.

This module intentionally narrows workflow-facing IO so production measurement
logic can evolve without depending on UI state, screen widgets, or legacy App
members.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Mapping, Sequence, TypeAlias

from application.contracts import MachineGateway, RunRepositoryProtocol
from application.state import CalibrationSnapshot, RunIdentity, RuntimeState
from core.models import MeasureRow, Recipe

SummaryPayload: TypeAlias = dict[str, Any]
RawPointPayload: TypeAlias = dict[str, Any]
RunResultStatus = Literal["DONE", "STOP", "ERR"]


class ProductionWorkflowEventType(StrEnum):
    STATE = "state"
    PROGRESS = "progress"
    LENGTH = "length"
    COVERAGE = "coverage"
    ROW = "row"
    RAW_POINTS = "raw_points"
    SUMMARY = "summary"


@dataclass(frozen=True, slots=True)
class StateEvent:
    state: str
    message: str
    type: ProductionWorkflowEventType = ProductionWorkflowEventType.STATE


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    section_index: int
    section_total: int
    z_pos_mm: float
    ax0_abs: float
    type: ProductionWorkflowEventType = ProductionWorkflowEventType.PROGRESS


@dataclass(frozen=True, slots=True)
class LengthEvent:
    payload: Mapping[str, Any]
    type: ProductionWorkflowEventType = ProductionWorkflowEventType.LENGTH


@dataclass(frozen=True, slots=True)
class CoverageEvent:
    payload: Mapping[str, Any]
    type: ProductionWorkflowEventType = ProductionWorkflowEventType.COVERAGE


@dataclass(frozen=True, slots=True)
class RowEvent:
    row: MeasureRow
    type: ProductionWorkflowEventType = ProductionWorkflowEventType.ROW


@dataclass(frozen=True, slots=True)
class RawPointsEvent:
    raw_points: tuple[Mapping[str, Any], ...]
    type: ProductionWorkflowEventType = ProductionWorkflowEventType.RAW_POINTS


@dataclass(frozen=True, slots=True)
class SummaryEvent:
    source: str
    payload: Mapping[str, Any]
    type: ProductionWorkflowEventType = ProductionWorkflowEventType.SUMMARY


TypedEvent: TypeAlias = (
    StateEvent
    | ProgressEvent
    | LengthEvent
    | CoverageEvent
    | RowEvent
    | RawPointsEvent
    | SummaryEvent
)


@dataclass(frozen=True, slots=True)
class RawPoints:
    points: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class RunResult:
    identity: RunIdentity | None
    status: RunResultStatus
    message: str
    started_at_ts: float | None
    finished_at_ts: float | None
    rows: tuple[MeasureRow, ...]
    summary: Mapping[str, Any]
    length_result: Mapping[str, Any] | None


@dataclass(slots=True)
class ProductionWorkflow:
    """Pure workflow boundary for the production measurement chain."""

    recipe: Recipe
    calibration: CalibrationSnapshot
    runtime_state: RuntimeState
    gateway: MachineGateway
    run_repository: RunRepositoryProtocol
    _events: list[TypedEvent] = field(default_factory=list, init=False)
    _run_result: RunResult | None = field(default=None, init=False)

    @property
    def events(self) -> tuple[TypedEvent, ...]:
        return tuple(self._events)

    @property
    def raw_points(self) -> RawPoints:
        return RawPoints(points=tuple(dict(point) for point in self.runtime_state.raw_points))

    @property
    def summary(self) -> Mapping[str, Any]:
        return dict(self.runtime_state.summary)

    @property
    def run_result(self) -> RunResult | None:
        return self._run_result

    def ensure_identity(self) -> RunIdentity:
        if self.runtime_state.serial and self.runtime_state.run_id and self.runtime_state.started_at_ts is not None:
            return RunIdentity(
                serial=str(self.runtime_state.serial),
                run_id=str(self.runtime_state.run_id),
                started_at_ts=float(self.runtime_state.started_at_ts),
            )
        identity = self.run_repository.prepare_run(getattr(self.recipe, 'name', ''))
        self.runtime_state.serial = identity.serial
        self.runtime_state.run_id = identity.run_id
        self.runtime_state.started_at_ts = identity.started_at_ts
        return identity

    def record_state(self, state: str, message: str) -> StateEvent:
        self.runtime_state.status = self._normalize_status(state)
        self.runtime_state.message = str(message or '')
        if str(state or '').upper() == 'ERR':
            self.runtime_state.last_error = self.runtime_state.message or 'Workflow error'
        event = StateEvent(state=str(state), message=str(message))
        self._events.append(event)
        return event

    def record_progress(
        self,
        *,
        section_index: int,
        section_total: int,
        z_pos_mm: float,
        ax0_abs: float,
    ) -> ProgressEvent:
        event = ProgressEvent(
            section_index=int(section_index),
            section_total=int(section_total),
            z_pos_mm=float(z_pos_mm),
            ax0_abs=float(ax0_abs),
        )
        self._events.append(event)
        return event

    def record_length(self, payload: Mapping[str, Any]) -> LengthEvent:
        copied = dict(payload)
        self.runtime_state.length_result = copied
        event = LengthEvent(payload=copied)
        self._events.append(event)
        return event

    def record_coverage(self, payload: Mapping[str, Any]) -> CoverageEvent:
        event = CoverageEvent(payload=dict(payload))
        self._events.append(event)
        return event

    def record_row(self, row: MeasureRow) -> RowEvent:
        self.runtime_state.rows.append(row)
        event = RowEvent(row=row)
        self._events.append(event)
        return event

    def record_raw_points(self, points: Sequence[Mapping[str, Any]]) -> RawPointsEvent:
        copied_points = [dict(point) for point in points]
        self.runtime_state.raw_points.extend(copied_points)
        event = RawPointsEvent(raw_points=tuple(copied_points))
        self._events.append(event)
        return event

    def record_summary(self, payload: Mapping[str, Any], *, source: str) -> SummaryEvent:
        copied = dict(payload)
        self.runtime_state.summary.update(copied)
        event = SummaryEvent(source=str(source), payload=copied)
        self._events.append(event)
        return event

    def build_run_result(
        self,
        *,
        status: RunResultStatus,
        message: str = '',
        finished_at_ts: float | None = None,
    ) -> RunResult:
        identity = self.ensure_identity()
        self.runtime_state.status = self._normalize_status(status)
        self.runtime_state.message = str(message or '')
        self.runtime_state.finished_at_ts = (
            float(finished_at_ts)
            if finished_at_ts is not None
            else self.runtime_state.finished_at_ts
        )
        if status == 'ERR':
            self.runtime_state.last_error = self.runtime_state.message or 'Workflow error'
        result = RunResult(
            identity=identity,
            status=status,
            message=self.runtime_state.message,
            started_at_ts=self.runtime_state.started_at_ts,
            finished_at_ts=self.runtime_state.finished_at_ts,
            rows=tuple(self.runtime_state.rows),
            summary=dict(self.runtime_state.summary),
            length_result=(
                dict(self.runtime_state.length_result)
                if self.runtime_state.length_result is not None
                else None
            ),
        )
        self._run_result = result
        return result

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
    'CoverageEvent',
    'LengthEvent',
    'ProductionWorkflow',
    'ProductionWorkflowEventType',
    'ProgressEvent',
    'RawPointPayload',
    'RawPoints',
    'RawPointsEvent',
    'RowEvent',
    'RunResult',
    'RunResultStatus',
    'StateEvent',
    'SummaryEvent',
    'SummaryPayload',
    'TypedEvent',
]
