from __future__ import annotations

"""Build-section-plan step - Phase 5 third extraction.

This step delegates section-plan construction to the orchestrator's legacy
implementation method. It preserves planning behavior while adding a third
workflow step boundary.

No UI host, no PLC connection or gauge serial I/O.
"""

from dataclasses import dataclass
from typing import Any, Protocol


class BuildSectionPlanPort(Protocol):
    """Narrow surface that the build-section-plan step needs."""

    def _build_section_plan_impl(self) -> Any: ...


@dataclass(slots=True)
class BuildSectionPlanStep:
    """Build the formal measurement section plan."""

    port: BuildSectionPlanPort
    name: str = "build_section_plan"

    def execute(self) -> Any:
        """Run the section planning phase via the legacy implementation boundary."""
        return self.port._build_section_plan_impl()


__all__ = ["BuildSectionPlanPort", "BuildSectionPlanStep"]
