from __future__ import annotations

"""Section-capture step boundary for one workflow section.

This step delegates the existing section measurement/capture implementation
without splitting sampling, rotation, fit, event, or persistence behavior.

No UI host, no PLC connection or gauge serial I/O.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from frp_workflow.steps.section_context import SectionExecutionContext


class SectionCapturePort(Protocol):
    """Narrow surface that the section-capture step needs."""

    def _capture_section_impl(self, context: SectionExecutionContext) -> Any: ...


@dataclass(slots=True)
class SectionCaptureStep:
    """Run the existing section capture/measurement phase."""

    port: SectionCapturePort
    name: str = "section_capture"

    def execute(self, context: SectionExecutionContext) -> Any:
        """Run capture via the legacy implementation boundary."""
        return self.port._capture_section_impl(context)


__all__ = ["SectionCapturePort", "SectionCaptureStep"]
