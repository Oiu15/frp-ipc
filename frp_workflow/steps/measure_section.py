from __future__ import annotations

"""Measure-section step boundary.

This step delegates the existing section measurement implementation without
splitting sampling, rotation, fit, event, or persistence behavior.

No UI host, no PLC connection or gauge serial I/O.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from frp_workflow.steps.measure_section_context import MeasureSectionContext


class MeasureSectionPort(Protocol):
    """Narrow surface that the measure-section step needs."""

    def _measure_section_impl(self, context: MeasureSectionContext) -> Any: ...


@dataclass(slots=True)
class MeasureSectionStep:
    """Run the existing section measurement phase."""

    port: MeasureSectionPort
    name: str = "measure_section"

    def execute(self, context: MeasureSectionContext) -> Any:
        """Run measurement via the legacy implementation boundary."""
        return self.port._measure_section_impl(context)


__all__ = ["MeasureSectionPort", "MeasureSectionStep"]
