from __future__ import annotations

"""Teach use-case service (Phase 4 pilot).

This module extracts a single teach action into a plain-Python service that
depends only on Protocols — no tkinter, no messagebox, no AppHost, no
StringVar / Treeview / widget access.
"""

from dataclasses import dataclass
from typing import Protocol


# ---------------------------------------------------------------------------
# Ports (Protocols)
# ---------------------------------------------------------------------------


class TeachMotionPort(Protocol):
    """Motion surface needed by TeachService."""

    def movea_abs(
        self, axis: int, pos_abs: float, *, context: str = "MoveA"
    ) -> None: ...


class TeachOperatorPort(Protocol):
    """User-interaction surface needed by TeachService."""

    def show_warning(self, title: str, message: str) -> None: ...

    def show_error(self, title: str, message: str) -> None: ...


class TeachRecipePort(Protocol):
    """Recipe surface needed by TeachService — lightweight placeholder.

    Currently TeachService works with DTOs constructed by the caller;
    this port exists for future teach actions that need recipe access.
    """

    pass


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StandbyTarget:
    """Positions to move to the standby point."""

    ax0_abs: float
    ax1_abs: float
    ax4_abs: float


@dataclass(slots=True)
class TeachTargetRequest:
    """Request to move axes to a teach target.

    For a simple standby move, the caller fills ``targets`` with the three
    axis positions from the recipe.
    """

    targets: StandbyTarget
    context: str = "TeachStandby"


@dataclass(slots=True)
class TeachTargetResult:
    """Result of a teach-target move."""

    ok: bool
    moved_axes: list[int]
    reason: str = ""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TeachService:
    """Pure-Python teach use-case service.

    All dependencies are injected as Protocol-obeying objects — no tkinter,
    no AppHost, no global state.
    """

    motion: TeachMotionPort
    operator: TeachOperatorPort
    recipes: TeachRecipePort

    # -- public API -------------------------------------------------------

    def move_to_standby(self, request: TeachTargetRequest) -> TeachTargetResult:
        """Move AX0/AX1/AX4 to the standby positions in *request*.

        Does NOT read UI state — the caller is responsible for constructing
        the DTO from recipe / UI variables.
        """
        t = request.targets

        try:
            self.motion.movea_abs(0, t.ax0_abs, context=request.context)
            self.motion.movea_abs(1, t.ax1_abs, context=request.context)
            self.motion.movea_abs(4, t.ax4_abs, context=request.context)
            return TeachTargetResult(
                ok=True,
                moved_axes=[0, 1, 4],
            )
        except Exception as exc:
            return TeachTargetResult(
                ok=False,
                moved_axes=[],
                reason=str(exc),
            )

    def move_to_targets(
        self, request: TeachTargetRequest
    ) -> TeachTargetResult:
        """Move axes to the positions given in *request.targets*."""
        t = request.targets
        moved: list[int] = []

        try:
            for axis, pos in [(0, t.ax0_abs), (1, t.ax1_abs), (4, t.ax4_abs)]:
                self.motion.movea_abs(axis, pos, context=request.context)
                moved.append(axis)
            return TeachTargetResult(ok=True, moved_axes=moved)
        except Exception as exc:
            return TeachTargetResult(ok=False, moved_axes=moved, reason=str(exc))


__all__ = [
    "StandbyTarget",
    "TeachMotionPort",
    "TeachOperatorPort",
    "TeachRecipePort",
    "TeachService",
    "TeachTargetRequest",
    "TeachTargetResult",
]
