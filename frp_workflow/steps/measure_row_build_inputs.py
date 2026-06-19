from __future__ import annotations

"""Input DTO for building a MeasureRow from sampling outputs."""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MeasureRowBuildInputs:
    """Data required by the existing measure-row build helper."""

    legacy: Any
    recipe: Any
    sensors: Any
    section_index: int
    z_pos_mm: float
    x_abs: float
    coords_od: Any
    coords_id: Any
    raw_od: Any
    raw_id: Any
    raw_points: Any
    fit_weights_od: Any
    fit_weights_id: Any
    scan_mode: str
    split_shift_deg: Any
    coax_unreliable: Any
    centers_xyz: list[tuple[float, float, float]]
    centers_xyz_id: list[tuple[float, float, float]]
    concentricity_list: list[float]
    validation_fit_payload: Any | None = None


__all__ = ["MeasureRowBuildInputs"]
