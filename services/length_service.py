from __future__ import annotations

"""Length measurement use-case service (Phase 4.2 pilot).

Thin wrapper around ``domain.length_math`` pure functions.  No tkinter,
no messagebox, no AppHost, no StringVar / Treeview / widget access.
"""

from dataclasses import dataclass

from domain.length_math import length_from_edges


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LengthCalcRequest:
    """Input for a length calculation from two edge positions."""

    edge_low: float
    edge_high: float


@dataclass(frozen=True, slots=True)
class LengthCalcResult:
    """Result of a length calculation.

    ``length`` is ``None`` when the inputs are invalid (non-positive or
    non-finite length).
    """

    length: float | None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class LengthService:
    """Pure-Python length-measurement service.

    Delegates all computation to ``domain.length_math``.  No dependencies
    on tkinter, AppHost, or UI widgets.
    """

    def calculate_length(self, request: LengthCalcRequest) -> LengthCalcResult:
        """Compute pipe length from two edge positions in Z_disp (mm).

        ``edge_low`` is the bottom edge, ``edge_high`` is the top edge.
        Length = edge_low - edge_high.  Returns ``None`` for invalid
        or non-positive results.
        """
        return LengthCalcResult(
            length=length_from_edges(request.edge_low, request.edge_high),
        )


__all__ = ["LengthCalcRequest", "LengthCalcResult", "LengthService"]
