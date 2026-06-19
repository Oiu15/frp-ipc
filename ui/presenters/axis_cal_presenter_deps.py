from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AxisCalUiState:
    axis_cal_vars: dict[str, Any]
    axis_cal_field_status_vars: dict[str, Any]
    axis_cal_status_vars: dict[str, Any]


__all__ = ["AxisCalUiState"]
