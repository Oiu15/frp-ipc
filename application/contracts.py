from __future__ import annotations

"""Repository contracts for the measurement main flow.

These protocols define the persistence boundaries that the workflow
orchestrator needs — run identity allocation, export, and calibration
snapshot access.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import domain.state as app_state

if TYPE_CHECKING:  # pragma: no cover
    from frp_workflow.validation_workflow import (
        FixedSectionRepeatCapture,
        FixedSectionRepeatabilityRequest,
        FixedSectionRepeatRow,
    )


@runtime_checkable
class RunRepositoryProtocol(Protocol):
    """Run identity allocation + export boundary for the measurement main flow."""

    def prepare_run(self, recipe_name: str) -> app_state.RunIdentity: ...

    def export_run(self, context: app_state.RunContext) -> str: ...

    def export_daily_summary(self, context: app_state.RunContext) -> None: ...


@runtime_checkable
class ValidationRepositoryProtocol(Protocol):
    """Validation export boundary kept separate from production exports."""

    def export_run(self, context: app_state.ValidationExportContext) -> str: ...

    def export_fixed_section_repeatability(
        self,
        *,
        context: app_state.ValidationExportContext,
        request: 'FixedSectionRepeatabilityRequest',
        rows: list['FixedSectionRepeatRow'],
        summary: Mapping[str, Any],
        captures: Sequence['FixedSectionRepeatCapture'] | None = None,
    ) -> str: ...

    def export_daily_summary(self, context: app_state.ValidationExportContext) -> None: ...


@runtime_checkable
class CalibrationRepositoryProtocol(Protocol):
    """Read-only calibration access required by the measurement main flow."""

    def load_snapshot(self) -> app_state.CalibrationSnapshot: ...


# Backward-compat re-exports for types that moved out of this module.
# Consumers should import from the canonical locations instead.
from events.protocols import EventPayload, EventSink, RawPoint  # noqa: F401, E402
from machine.validation_gateway import ValidationActionCancelled, ValidationActionGateway  # noqa: F401, E402

__all__ = [
    "CalibrationRepositoryProtocol",
    "EventPayload",
    "EventSink",
    "RawPoint",
    "RunRepositoryProtocol",
    "ValidationActionCancelled",
    "ValidationActionGateway",
    "ValidationRepositoryProtocol",
]
