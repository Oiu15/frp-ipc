from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from events.protocols import EventSink
from core.models import MeasureRow
from events.adapters import UiQueueCompatAdapter, WorkerUiEventAdapter


class WorkflowUiEventAdapter(UiQueueCompatAdapter, EventSink):
    """Workflow-side adapter that preserves the legacy UI queue payloads.

    Implements the EventSink protocol and adds convenience methods
    matching the auto_* event naming convention used by the production
    workflow orchestrator.
    """

    # -- EventSink protocol ------------------------------------------------

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

    # -- convenience wrappers matching auto_* naming -----------------------

    def publish_auto_state(self, state: str, message: str) -> None:
        """Publish auto_state event (convenience alias for publish_state)."""
        self.publish_state(state, message)

    def publish_auto_done(self, message: str = "Measurement completed") -> None:
        """Publish auto_state DONE event."""
        self.publish_state("DONE", message)

    def publish_auto_error(self, message: str) -> None:
        """Publish auto_state ERR event."""
        self.publish_state("ERR", message)

    def publish_auto_row(self, row: MeasureRow) -> None:
        """Publish auto_row event (convenience alias for publish_row)."""
        self.publish_row(row)

    def publish_auto_len(self, payload: Mapping[str, Any]) -> None:
        """Publish auto_len event (convenience alias for publish_length)."""
        self.publish_length(payload)

    def publish_auto_progress(
        self,
        *,
        section_index: int,
        section_total: int,
        z_pos_mm: float,
        ax0_abs: float,
    ) -> None:
        """Publish auto_progress event (convenience alias for publish_progress)."""
        self.publish_progress(
            section_index=section_index,
            section_total=section_total,
            z_pos_mm=z_pos_mm,
            ax0_abs=ax0_abs,
        )

    def publish_auto_cov(self, payload: Mapping[str, Any]) -> None:
        """Publish auto_cov event (convenience alias for publish_coverage)."""
        self.publish_coverage(payload)

    def publish_auto_postcalc(self, payload: Mapping[str, Any]) -> None:
        """Publish auto_postcalc event (convenience alias for publish_postcalc)."""
        self.publish_postcalc(payload)

    def publish_auto_raw_points(self, points: Sequence[Mapping[str, Any]]) -> None:
        """Publish auto_raw_points event (convenience alias for publish_raw_points)."""
        self.publish_raw_points(points)

    def publish_auto_clear(self) -> None:
        """Publish auto_clear event — clears result tables while preserving run identity."""
        self.publish_legacy('auto_clear', {'ts': None})


__all__ = ['UiQueueCompatAdapter', 'WorkerUiEventAdapter', 'WorkflowUiEventAdapter']
