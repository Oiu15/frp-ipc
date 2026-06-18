from __future__ import annotations

"""Row-build output DTO for measure-section internals."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RowBuildResult:
    """Outputs produced by the existing row build block."""

    row: Any


__all__ = ["RowBuildResult"]
