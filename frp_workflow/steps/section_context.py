from __future__ import annotations

"""Data context for executing one workflow section."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SectionExecutionContext:
    """Input data for one formal measurement section execution."""

    section: Any
    section_index: int
    total_sections: int
    centers_xyz: list[tuple[float, float, float]]
    centers_xyz_id: list[tuple[float, float, float]]
    concentricity_list: list[float]


__all__ = ["SectionExecutionContext"]
