from __future__ import annotations

"""UI-queue event dispatcher.

Current scope is intentionally small:
- keep the event payload unchanged
- only manage "event name -> handler" routing
- let callers decide when/how to invoke the dispatcher
"""

from collections.abc import Callable, Mapping
from typing import Any

UiEventPayload = Any
UiEventHandler = Callable[[UiEventPayload], None]
UiEvent = tuple[str, UiEventPayload]


class UiEventDispatcher:
    """Dispatch UI queue events by event name."""

    def __init__(self, handlers: Mapping[str, UiEventHandler] | None = None) -> None:
        self._handlers: dict[str, UiEventHandler] = dict(handlers or {})

    def register(self, event_name: str, handler: UiEventHandler) -> None:
        self._handlers[str(event_name)] = handler

    def register_many(self, handlers: Mapping[str, UiEventHandler]) -> None:
        for event_name, handler in handlers.items():
            self.register(event_name, handler)

    def unregister(self, event_name: str) -> None:
        self._handlers.pop(str(event_name), None)

    def get_handler(self, event_name: str) -> UiEventHandler | None:
        return self._handlers.get(str(event_name))

    def handles(self, event_name: str) -> bool:
        return str(event_name) in self._handlers

    def dispatch(self, event_name: str, payload: UiEventPayload) -> bool:
        handler = self.get_handler(event_name)
        if handler is None:
            return False
        handler(payload)
        return True

    def dispatch_event(self, event: UiEvent) -> bool:
        event_name, payload = event
        return self.dispatch(event_name, payload)

    def event_names(self) -> tuple[str, ...]:
        return tuple(self._handlers.keys())

    def snapshot(self) -> dict[str, UiEventHandler]:
        return dict(self._handlers)


__all__ = [
    "UiEvent",
    "UiEventDispatcher",
    "UiEventHandler",
    "UiEventPayload",
]
