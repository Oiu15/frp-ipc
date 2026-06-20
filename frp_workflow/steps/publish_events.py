from __future__ import annotations

"""Publish event step boundary for one measured section."""

from dataclasses import dataclass
from typing import Protocol

from frp_workflow.steps.publish_events_context import PublishEventsContext


class PublishEventsPort(Protocol):
    """Narrow surface that the publish events step needs."""

    def _publish_section_events_impl(self, context: PublishEventsContext) -> None: ...


@dataclass(slots=True)
class PublishEventsStep:
    """Publish existing section events via the orchestrator boundary."""

    port: PublishEventsPort
    name: str = "publish_events"

    def execute(self, context: PublishEventsContext) -> None:
        self.port._publish_section_events_impl(context)


__all__ = ["PublishEventsPort", "PublishEventsStep"]
