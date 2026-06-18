from __future__ import annotations

"""Section-execution step - Phase 6 first core-flow boundary.

This step delegates one formal measurement section to the orchestrator's
legacy implementation method. It establishes the boundary without splitting
capture, sampling, rotation, clamp, cancellation, or result behavior.

No UI host, no PLC connection or gauge serial I/O.
"""

from dataclasses import dataclass
from typing import Any, Protocol


class SectionExecutionPort(Protocol):
    """Narrow surface that the section-execution step needs."""

    def _execute_section_impl(
        self,
        section: Any,
        *,
        section_total: int,
        centers_xyz: list[tuple[float, float, float]],
        centers_xyz_id: list[tuple[float, float, float]],
        concentricity_list: list[float],
    ) -> Any: ...


@dataclass(slots=True)
class SectionExecutionStep:
    """Execute one formal measurement section."""

    port: SectionExecutionPort
    name: str = "section_execution"

    def execute(
        self,
        section: Any,
        *,
        section_total: int,
        centers_xyz: list[tuple[float, float, float]],
        centers_xyz_id: list[tuple[float, float, float]],
        concentricity_list: list[float],
    ) -> Any:
        """Run one section via the legacy implementation boundary."""
        return self.port._execute_section_impl(
            section,
            section_total=section_total,
            centers_xyz=centers_xyz,
            centers_xyz_id=centers_xyz_id,
            concentricity_list=concentricity_list,
        )


__all__ = ["SectionExecutionPort", "SectionExecutionStep"]
