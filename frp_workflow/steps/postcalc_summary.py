from __future__ import annotations

"""Postcalc summary step boundary for a completed measurement run."""

from dataclasses import dataclass
from typing import Protocol


class PostcalcSummaryPort(Protocol):
    """Narrow surface that the postcalc summary step needs."""

    def _run_postcalc_impl(
        self,
        *,
        centers_xyz: list[tuple[float, float, float]],
        centers_xyz_id: list[tuple[float, float, float]],
        concentricity_list: list[float],
    ) -> None: ...


@dataclass(slots=True)
class PostcalcSummaryStep:
    """Run existing postcalc summary logic via the orchestrator boundary."""

    port: PostcalcSummaryPort
    name: str = "postcalc_summary"

    def execute(
        self,
        *,
        centers_xyz: list[tuple[float, float, float]],
        centers_xyz_id: list[tuple[float, float, float]],
        concentricity_list: list[float],
    ) -> None:
        self.port._run_postcalc_impl(
            centers_xyz=centers_xyz,
            centers_xyz_id=centers_xyz_id,
            concentricity_list=concentricity_list,
        )


__all__ = ["PostcalcSummaryPort", "PostcalcSummaryStep"]
