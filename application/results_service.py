from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.models import MeasureRow, Recipe
from domain.summaries import (
    EccentricityUpdate,
    SummarySnapshot,
    apply_eccentricity_updates as _apply_eccentricity_updates,
    build_eccentricity_updates as _build_eccentricity_updates,
    compute_run_summary as _compute_run_summary,
    merge_summary_snapshot as _merge_summary_snapshot,
    summary_snapshot_from_payload as _summary_snapshot_from_payload,
)


class ResultsService:
    """Thin application wrapper around pure result-domain functions."""

    def summary_snapshot_from_payload(self, payload: Any) -> SummarySnapshot:
        return _summary_snapshot_from_payload(payload)

    def merge_summary_snapshot(
        self,
        summary_cache: Mapping[str, Any] | None,
        snapshot: SummarySnapshot,
    ) -> dict[str, Any]:
        return _merge_summary_snapshot(summary_cache, snapshot)

    def build_eccentricity_updates(self, payload: Any) -> list[EccentricityUpdate]:
        return _build_eccentricity_updates(payload)

    def apply_eccentricity_updates(
        self,
        rows: Sequence[MeasureRow],
        updates: Sequence[EccentricityUpdate],
    ) -> list[MeasureRow]:
        return _apply_eccentricity_updates(rows, updates)

    def compute_run_summary(
        self,
        *,
        recipe: Recipe,
        rows: Sequence[MeasureRow],
        raw_points: Sequence[Mapping[str, Any]],
        summary_cache: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _compute_run_summary(
            recipe=recipe,
            rows=rows,
            raw_points=raw_points,
            summary_cache=summary_cache,
        )


__all__ = ['EccentricityUpdate', 'ResultsService', 'SummarySnapshot']
