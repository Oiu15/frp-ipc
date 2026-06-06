from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from events.protocols import EventSink
from core.models import MeasureRow
from events.adapters import UiQueueCompatAdapter, WorkerUiEventAdapter


class WorkflowUiEventAdapter(UiQueueCompatAdapter, EventSink):
    """Workflow-side adapter that preserves the legacy UI queue payloads."""

    def publish_state(self, state: str, message: str) -> None:
        self.publish_legacy('auto_state', {'state': state, 'msg': message})

    def publish_progress(
        self,
        *,
        section_index: int,
        section_total: int,
        z_pos_mm: float,
        ax0_abs: float,
    ) -> None:
        self.publish_legacy(
            'auto_progress',
            {
                'idx': max(0, int(section_index) - 1),
                'total': int(section_total),
                'x_ui': float(z_pos_mm),
                'x_abs': float(ax0_abs),
            },
        )

    def publish_length(self, payload: Mapping[str, Any]) -> None:
        self.publish_legacy('auto_len', dict(payload))

    def publish_coverage(self, payload: Mapping[str, Any]) -> None:
        self.publish_legacy('auto_cov', dict(payload))

    def publish_raw_points(self, points: Sequence[Mapping[str, Any]]) -> None:
        self.publish_legacy('auto_raw_points', {'points': [dict(point) for point in points]})

    def publish_row(self, row: MeasureRow) -> None:
        self.publish_legacy('auto_row', {'row': row})

    def publish_straightness(self, payload: Mapping[str, Any]) -> None:
        self.publish_legacy('auto_straightness', dict(payload))

    def publish_postcalc(self, payload: Mapping[str, Any]) -> None:
        self.publish_legacy('auto_postcalc', dict(payload))


__all__ = ['UiQueueCompatAdapter', 'WorkerUiEventAdapter', 'WorkflowUiEventAdapter']
