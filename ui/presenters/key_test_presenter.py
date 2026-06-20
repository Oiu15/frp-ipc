from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class KeyTestUiState:
    keytest_x_vars: Any
    keytest_y_vars: Any
    keytest_y_lastcmd_vars: Any


__all__ = ["KeyTestUiState"]
