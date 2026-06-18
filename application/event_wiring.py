from __future__ import annotations

"""UI event dispatcher and queue-pump wiring (Phase 2).

This module owns the assembly of ``UiEventDispatcher`` instances and the
``UiQueuePump`` that bridges worker threads to the Tk main thread.  The
handler *methods* stay on AppHost; only the registration/wiring is moved
here.
"""

from typing import Any

from events.pump import UiQueuePump


def wire_ui_event_handlers(host: Any) -> None:
    """Create event dispatchers and queue pump, attaching them to *host*.

    The handler callables (``host._handle_plc_read_event``, etc.) are
    resolved via ``host._build_device_ui_event_dispatcher()`` and
    ``host._build_measurement_ui_event_dispatcher()`` — those factory
    methods remain on AppHost and are the single source of truth for the
    handler mapping.
    """
    # Import here to avoid circular dependency with the LOG_UI_EVENT_FILTER
    # constant defined in app_host.py at module level.
    from application.app_host import LOG_UI_EVENT_FILTER  # noqa: F811

    device_dispatcher = host._build_device_ui_event_dispatcher()
    host._device_ui_event_dispatcher = device_dispatcher

    measurement_dispatcher = host._build_measurement_ui_event_dispatcher()
    host._measurement_ui_event_dispatcher = measurement_dispatcher

    host._ui_queue_pump = UiQueuePump(
        ui_q=host.ui_q,
        device_dispatcher=device_dispatcher,
        measurement_dispatcher=measurement_dispatcher,
        perf_ui_queue=host._perf_ui_queue,
        log_filter=LOG_UI_EVENT_FILTER,
    )


__all__ = ["wire_ui_event_handlers"]
