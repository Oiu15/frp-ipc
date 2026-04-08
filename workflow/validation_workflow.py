from __future__ import annotations

"""Pure workflow boundary objects for validation mode.

The first version intentionally stays minimal: it models workflow-owned state,
typed events, and a result object without assuming full validation sampling or
hardware choreography.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Mapping, TypeAlias

from application.contracts import MachineGateway, RunRepositoryProtocol
from application.state import CalibrationSnapshot, RunIdentity, RuntimeState, ValidationSession
from core.models import Recipe

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
    'ProgressEvent',
    'StateEvent',
    'SummaryEvent',
    'SummaryPayload',
    'TypedEvent',
    'ValidationResult',
    'ValidationResultStatus',
    'ValidationWorkflow',
    'ValidationWorkflowEventType',
]
