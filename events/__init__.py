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

from events.types import (  # noqa: F401
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

from events.dispatcher import (  # noqa: F401
    TypedUiEventHandler,
    UiEvent,
    UiEventDispatcher,
    UiEventHandler,
    UiEventHandlerLike,
    UiEventKey,
    UiEventPayload,
)

from events.adapters import (  # noqa: F401
    UiQueueCompatAdapter,
    WorkerUiEventAdapter,
)

from events.pump import (  # noqa: F401
    UiQueuePump,
    UiQueuePumpResult,
)
