from __future__ import annotations

"""Computed values used to build a MeasureRow."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MeasureRowComputationResult:
    od_center: tuple[float, float, float]
    id_center: tuple[float, float, float] | None
    center_od_x: float
    center_od_y: float
    center_id_x: float | None
    center_id_y: float | None
    od_radius_fit_mm: float | None
    od_diameter_fit_mm: float | None
    id_radius_fit_mm: float | None
    id_diameter_fit_mm: float | None
    od_avg: float
    od_dev: float
    od_runout: float
    od_round: float
    od_round_fit_mm: Any
    od_round_fit_rob_mm: Any
    od_pp_mm: Any
    od_pp_rob_mm: Any
    id_avg: Any
    id_dev: Any
    id_runout: Any
    id_round: Any
    id_round_fit_mm: Any
    id_round_fit_rob_mm: Any
    id_pp_mm: Any
    id_pp_rob_mm: Any
    od_e: Any
    od_phi_deg: Any
    id_e: Any
    id_phi_deg: Any
    id_mode: str
    concentricity: Any


__all__ = ["MeasureRowComputationResult"]
