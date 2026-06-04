from __future__ import annotations

"""Domain-level persistence protocols.

These protocols define the repository boundaries that the workflow
orchestrator needs.  Placing them in ``domain/`` (rather than
``application/``) keeps the dependency direction inward and allows
inner layers (``frp_workflow/``, ``repositories/``) to import them
without depending on the outer ``application/`` layer.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from domain.state import CalibrationSnapshot, RunContext, RunIdentity, ValidationExportContext

if TYPE_CHECKING:  # pragma: no cover
    from frp_workflow.validation_workflow import (
        FixedSectionRepeatCapture,
        FixedSectionRepeatabilityRequest,
        FixedSectionRepeatRow,
    )


@runtime_checkable
class RunRepositoryProtocol(Protocol):
    """Run identity allocation + export boundary for the measurement main flow."""

    def prepare_run(self, recipe_name: str) -> RunIdentity: ...

    def export_run(self, context: RunContext) -> str: ...

    def export_daily_summary(self, context: RunContext) -> None: ...


@runtime_checkable
class ValidationRepositoryProtocol(Protocol):
    """Validation export boundary kept separate from production exports."""

    def export_run(self, context: ValidationExportContext) -> str: ...

    def export_fixed_section_repeatability(
        self,
        *,
        context: ValidationExportContext,
        request: 'FixedSectionRepeatabilityRequest',
        rows: list['FixedSectionRepeatRow'],
        summary: Mapping[str, Any],
        captures: Sequence['FixedSectionRepeatCapture'] | None = None,
    ) -> str: ...

    def export_daily_summary(self, context: ValidationExportContext) -> None: ...


@runtime_checkable
class CalibrationRepositoryProtocol(Protocol):
    """Read-only calibration access required by the measurement main flow."""

    def load_snapshot(self) -> CalibrationSnapshot: ...


__all__ = [
    "CalibrationRepositoryProtocol",
    "RunRepositoryProtocol",
    "ValidationRepositoryProtocol",
]
