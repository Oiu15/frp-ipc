from __future__ import annotations

"""Application-layer contracts for the measurement main flow.

This module intentionally defines only the boundaries that the current
"主界面开始测量 -> AutoFlow/Orchestrator -> 结果回传 -> 导出" main chain
really needs.

Scope notes:
- The workflow should receive a `Recipe` snapshot directly, so a recipe
  repository protocol is intentionally not included here yet.
- These protocols are transition-oriented: they keep the boundary small,
  but still map cleanly to the current App/worker capabilities.
"""

from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from application.state import CalibrationSnapshot, RunContext, RunIdentity
from core.models import MeasureRow
from machine.device_gateway import DeviceGateway, PollProfile

EventPayload = Mapping[str, Any]
RawPoint = Mapping[str, Any]
RunStatus = Literal["DONE", "STOP", "ERR"]
OperatorConfirmResult = Literal["confirm", "stop", "timeout", "cancel"]


@runtime_checkable
class EventSink(Protocol):
    """Workflow -> outside world event publishing contract."""

    def publish_state(self, state: str, message: str) -> None: ...

    def publish_progress(
        self,
        *,
        section_index: int,
        section_total: int,
        z_pos_mm: float,
        ax0_abs: float,
    ) -> None: ...

    def publish_length(self, payload: EventPayload) -> None: ...

    def publish_coverage(self, payload: EventPayload) -> None: ...

    def publish_raw_points(self, points: Sequence[RawPoint]) -> None: ...

    def publish_row(self, row: MeasureRow) -> None: ...

    def publish_straightness(self, payload: EventPayload) -> None: ...

    def publish_postcalc(self, payload: EventPayload) -> None: ...


MachineGateway = DeviceGateway


@runtime_checkable
class RunRepositoryProtocol(Protocol):
    """Run identity allocation + export boundary for the measurement main flow."""

    def prepare_run(self, recipe_name: str) -> RunIdentity: ...

    def export_run(self, context: RunContext) -> str: ...

    def export_daily_summary(self, context: RunContext) -> None: ...


@runtime_checkable
class CalibrationRepositoryProtocol(Protocol):
    """Read-only calibration access required by the measurement main flow."""

    def load_snapshot(self) -> CalibrationSnapshot: ...


__all__ = [
    "CalibrationRepositoryProtocol",
    "CalibrationSnapshot",
    "DeviceGateway",
    "EventPayload",
    "EventSink",
    "MachineGateway",
    "OperatorConfirmResult",
    "PollProfile",
    "RawPoint",
    "RunContext",
    "RunIdentity",
    "RunRepositoryProtocol",
    "RunStatus",
]
