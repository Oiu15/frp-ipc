from __future__ import annotations

"""Sampling output DTO for measure-section internals."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SamplingResult:
    """Outputs produced by the existing sampling block."""

    primary_sample: Any
    id_sample: Any | None
    coords_od: np.ndarray
    coords_id: np.ndarray
    raw_od: Any
    raw_id: Any
    raw_points: list[dict]
    split_shift_deg: float | None
    coax_unreliable: bool | None


__all__ = ["SamplingResult"]
