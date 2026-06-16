from __future__ import annotations

"""Event publishing protocols for the measurement main flow.

These protocols define the producer-side contracts for event publishing,
pairing with the typed event dispatch infrastructure in ``events.dispatcher``.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from core.models import MeasureRow

EventPayload = Mapping[str, Any]
RawPoint = Mapping[str, Any]


@runtime_checkable
class EventSink(Protocol):
    """Workflow -> outside world event publishing contract."""

    # -- canonical names (used by the new orchestrator) --------------------

    def publish_state(self, state: str, message: str) -> None: ...

    def publish_progress(
        self,
        *,
        section_index: int,
        section_total: int,
        z_pos_mm: float,
        ax0_abs: float,
    ) -> None: ...

    def publish_length(self, payload: EventPayload) -> None: ...

    def publish_coverage(self, payload: EventPayload) -> None: ...

    def publish_raw_points(self, points: Sequence[RawPoint]) -> None: ...

    def publish_row(self, row: MeasureRow) -> None: ...

    def publish_straightness(self, payload: EventPayload) -> None: ...

    def publish_postcalc(self, payload: EventPayload) -> None: ...

    # -- auto_* aliases (used by the legacy executor) ---------------------

    def publish_auto_state(self, state: str, message: str) -> None: ...

    def publish_auto_done(self, message: str) -> None: ...

    def publish_auto_error(self, message: str) -> None: ...

    def publish_auto_row(self, row: MeasureRow) -> None: ...

    def publish_auto_len(self, payload: EventPayload) -> None: ...

    def publish_auto_progress(
        self,
        *,
        section_index: int,
        section_total: int,
        z_pos_mm: float,
        ax0_abs: float,
    ) -> None: ...

    def publish_auto_cov(self, payload: EventPayload) -> None: ...

    def publish_auto_raw_points(self, points: Sequence[RawPoint]) -> None: ...

    def publish_auto_clear(self) -> None: ...

    def publish_auto_postcalc(self, payload: EventPayload) -> None: ...


__all__ = ["EventPayload", "EventSink", "RawPoint"]
