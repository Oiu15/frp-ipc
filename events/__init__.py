"""Event subsystem for FRP-IPC.

This package provides typed UI events, dispatch infrastructure, queue adapters,
and the queue pump — decoupled from the application layer so that drivers and
other low-level components can publish events without depending on ``application/``.

Modules:
    types       — UiEventBase, typed event dataclasses, parse_ui_event helpers
    dispatcher  — UiEventDispatcher (typed + string-keyed routing)
    adapters    — UiQueueCompatAdapter, WorkerUiEventAdapter
    pump        — UiQueuePump (legacy queue drain loop)
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from events.adapters import UiQueueCompatAdapter, WorkerUiEventAdapter
    from events.dispatcher import (
        TypedUiEventHandler,
        UiEvent,
        UiEventDispatcher,
        UiEventHandler,
        UiEventHandlerLike,
        UiEventKey,
        UiEventPayload,
    )
    from events.pump import UiQueuePump, UiQueuePumpResult
    from events.types import (
        AutoClearEvent,
        AutoCoverageEvent,
        AutoLenEvent,
        AutoPostcalcEvent,
        AutoProgressEvent,
        AutoRawPointsEvent,
        AutoRowEvent,
        AutoStateEvent,
        AutoStraightnessEvent,
        GaugeConnEvent,
        GaugeErrEvent,
        GaugeOkEvent,
        GaugeRawEvent,
        GaugeTxEvent,
        KnownUiEvent,
        OpConfirmCloseEvent,
        OpConfirmShowEvent,
        PlcErrEvent,
        PlcGiveupEvent,
        PlcManualEvent,
        PlcOkEvent,
        PlcReadEvent,
        UI_EVENT_TYPES,
        UiEventBase,
        UiEventTuple,
        parse_ui_event,
        parse_ui_event_tuple,
    )

_TYPE_EXPORTS = (
    "AutoClearEvent",
    "AutoCoverageEvent",
    "AutoLenEvent",
    "AutoPostcalcEvent",
    "AutoProgressEvent",
    "AutoRawPointsEvent",
    "AutoRowEvent",
    "AutoStateEvent",
    "AutoStraightnessEvent",
    "GaugeConnEvent",
    "GaugeErrEvent",
    "GaugeOkEvent",
    "GaugeRawEvent",
    "GaugeTxEvent",
    "KnownUiEvent",
    "OpConfirmCloseEvent",
    "OpConfirmShowEvent",
    "PlcErrEvent",
    "PlcGiveupEvent",
    "PlcManualEvent",
    "PlcOkEvent",
    "PlcReadEvent",
    "UI_EVENT_TYPES",
    "UiEventBase",
    "UiEventTuple",
    "parse_ui_event",
    "parse_ui_event_tuple",
)
_DISPATCHER_EXPORTS = (
    "TypedUiEventHandler",
    "UiEvent",
    "UiEventDispatcher",
    "UiEventHandler",
    "UiEventHandlerLike",
    "UiEventKey",
    "UiEventPayload",
)
_ADAPTER_EXPORTS = ("UiQueueCompatAdapter", "WorkerUiEventAdapter")
_PUMP_EXPORTS = ("UiQueuePump", "UiQueuePumpResult")

_EXPORT_MODULE = {
    **dict.fromkeys(_TYPE_EXPORTS, "events.types"),
    **dict.fromkeys(_DISPATCHER_EXPORTS, "events.dispatcher"),
    **dict.fromkeys(_ADAPTER_EXPORTS, "events.adapters"),
    **dict.fromkeys(_PUMP_EXPORTS, "events.pump"),
}
__all__ = [
    "AutoClearEvent",
    "AutoCoverageEvent",
    "AutoLenEvent",
    "AutoPostcalcEvent",
    "AutoProgressEvent",
    "AutoRawPointsEvent",
    "AutoRowEvent",
    "AutoStateEvent",
    "AutoStraightnessEvent",
    "GaugeConnEvent",
    "GaugeErrEvent",
    "GaugeOkEvent",
    "GaugeRawEvent",
    "GaugeTxEvent",
    "KnownUiEvent",
    "OpConfirmCloseEvent",
    "OpConfirmShowEvent",
    "PlcErrEvent",
    "PlcGiveupEvent",
    "PlcManualEvent",
    "PlcOkEvent",
    "PlcReadEvent",
    "UI_EVENT_TYPES",
    "UiEventBase",
    "UiEventTuple",
    "parse_ui_event",
    "parse_ui_event_tuple",
    "TypedUiEventHandler",
    "UiEvent",
    "UiEventDispatcher",
    "UiEventHandler",
    "UiEventHandlerLike",
    "UiEventKey",
    "UiEventPayload",
    "UiQueueCompatAdapter",
    "WorkerUiEventAdapter",
    "UiQueuePump",
    "UiQueuePumpResult",
]


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
