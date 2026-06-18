from __future__ import annotations

"""Data context for publishing section workflow events."""

from dataclasses import dataclass
from typing import Any

from frp_workflow.steps.measure_section_context import MeasureSectionContext


@dataclass(frozen=True, slots=True)
class PublishEventsContext:
    """Event payload data for one publish boundary call."""

    measure_context: MeasureSectionContext
    raw_points: list[dict] | None = None
    coverage_payload: dict[str, Any] | None = None
    row: Any | None = None


__all__ = ["PublishEventsContext"]
