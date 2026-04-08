from __future__ import annotations

"""UI-queue event dispatcher.

Dispatcher now prefers typed event routing based on ``application.ui_events``.
String-key registration remains available as a compatibility fallback for
callers that still work with raw ``(event_name, payload)`` tuples.
"""

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias, cast

from application.ui_events import UI_EVENT_TYPES, UiEventBase, parse_ui_event

UiEventPayload = Any
UiEvent = tuple[str, UiEventPayload]
UiEventHandler = Callable[[UiEventPayload], None]
TypedUiEventHandler = Callable[[UiEventBase], None]
UiEventHandlerLike = Callable[[Any], None]
UiEventKey: TypeAlias = str | type[UiEventBase]


class UiEventDispatcher:
    """Dispatch UI queue events by typed event class when available."""

    def __init__(self, handlers: Mapping[UiEventKey, UiEventHandlerLike] | None = None) -> None:
        self._typed_handlers: dict[type[UiEventBase], TypedUiEventHandler] = {}
        self._name_handlers: dict[str, UiEventHandler] = {}
        if handlers:
            self.register_many(handlers)

    def register(self, event_key: UiEventKey, handler: UiEventHandlerLike) -> None:
        event_cls = self._resolve_typed_event_class(event_key)
        if event_cls is not None:
            self._typed_handlers[event_cls] = cast(TypedUiEventHandler, handler)
            return
        self._name_handlers[str(event_key)] = cast(UiEventHandler, handler)

    def register_many(self, handlers: Mapping[UiEventKey, UiEventHandlerLike]) -> None:
        for event_key, handler in handlers.items():
            self.register(event_key, handler)

    def unregister(self, event_key: UiEventKey) -> None:
        event_cls = self._resolve_typed_event_class(event_key)
        if event_cls is not None:
            self._typed_handlers.pop(event_cls, None)
            return
        event_name = str(event_key)
        self._name_handlers.pop(event_name, None)
        known_event_cls = UI_EVENT_TYPES.get(event_name)
        if known_event_cls is not None:
            self._typed_handlers.pop(known_event_cls, None)

    def get_handler(self, event_key: UiEventKey) -> UiEventHandlerLike | None:
        event_cls = self._resolve_typed_event_class(event_key)
        if event_cls is not None:
            return self._typed_handlers.get(event_cls)
        event_name = str(event_key)
        handler = self._name_handlers.get(event_name)
        if handler is not None:
            return handler
        known_event_cls = UI_EVENT_TYPES.get(event_name)
        if known_event_cls is not None:
            return self._typed_handlers.get(known_event_cls)
        return None

    def handles(self, event_key: UiEventKey | UiEventBase) -> bool:
        if isinstance(event_key, UiEventBase):
            return type(event_key) in self._typed_handlers or event_key.event_name in self._name_handlers
        event_cls = self._resolve_typed_event_class(event_key)
        if event_cls is not None:
            return event_cls in self._typed_handlers
        event_name = str(event_key)
        return event_name in self._name_handlers or UI_EVENT_TYPES.get(event_name) in self._typed_handlers

    def dispatch(self, event_name: str, payload: UiEventPayload) -> bool:
        typed_event = parse_ui_event(str(event_name), payload)
        if typed_event is not None and self.dispatch_typed(typed_event):
            return True
        handler = self._name_handlers.get(str(event_name))
        if handler is None:
            return False
        handler(payload)
        return True

    def dispatch_typed(self, event: UiEventBase) -> bool:
        handler = self._typed_handlers.get(type(event))
        if handler is not None:
            handler(event)
            return True
        fallback = self._name_handlers.get(event.event_name)
        if fallback is not None:
            fallback(event.to_payload())
            return True
        return False

    def dispatch_event(self, event: UiEvent) -> bool:
        event_name, payload = event
        return self.dispatch(event_name, payload)

    def event_names(self) -> tuple[str, ...]:
        typed_names = [event_cls.event_name for event_cls in self._typed_handlers]
        return tuple(dict.fromkeys([*typed_names, *self._name_handlers.keys()]))

    def snapshot(self) -> dict[UiEventKey, UiEventHandlerLike]:
        data: dict[UiEventKey, UiEventHandlerLike] = {}
        for event_cls, handler in self._typed_handlers.items():
            data[event_cls] = handler
        for event_name, handler in self._name_handlers.items():
            data[event_name] = handler
        return data

    @staticmethod
    def _resolve_typed_event_class(event_key: UiEventKey | UiEventBase) -> type[UiEventBase] | None:
        if isinstance(event_key, UiEventBase):
            return type(event_key)
        if isinstance(event_key, type) and issubclass(event_key, UiEventBase):
            return event_key
        return None


__all__ = [
    "UiEvent",
    "UiEventDispatcher",
    "UiEventHandler",
    "UiEventHandlerLike",
    "UiEventKey",
    "UiEventPayload",
    "TypedUiEventHandler",
]
