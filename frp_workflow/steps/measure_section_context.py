from __future__ import annotations

"""Data context for measuring one workflow section."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MeasureSectionContext:
    """Input data for the existing section measurement implementation."""

    section_index: int
    z_pos_mm: float
    x_abs: float
    centers_xyz: list[tuple[float, float, float]]
    centers_xyz_id: list[tuple[float, float, float]]
    concentricity_list: list[float]


__all__ = ["MeasureSectionContext"]
