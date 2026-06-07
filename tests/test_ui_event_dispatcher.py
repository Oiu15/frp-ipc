from __future__ import annotations

from typing import Any

import pytest

from events.dispatcher import UiEventDispatcher
from events.types import GaugeOkEvent, PlcErrEvent


class TestUiEventDispatcher:
    """UiEventDispatcher — typed vs fallback routing."""

    # ------------------------------------------------------------------
    # string dispatch — typed handler vs fallback
    # ------------------------------------------------------------------

    _DISPATCH_CASES = [
        # (handler_map, event_name, payload, expected_handler_type, expected_attrs)
        (
            {PlcErrEvent: lambda: None},
            "plc_err",
            {"err": "boom", "retry": 1, "max": 3},
            PlcErrEvent,
            {"err": "boom", "retry": 1},
        ),
        (
            {"custom_evt": lambda: None},
            "custom_evt",
            {"value": 42},
            dict,
            None,  # fallback: check raw payload equals the input dict
        ),
    ]

    @pytest.mark.parametrize(
        ("handler_map", "event_name", "payload", "expected_type", "expected_attrs"),
        _DISPATCH_CASES,
    )
    def test_dispatch_routes_to_correct_handler(
        self,
        handler_map: dict[Any, Any],
        event_name: str,
        payload: dict[str, Any],
        expected_type: type,
        expected_attrs: dict[str, Any] | None,
    ) -> None:
        seen: list[Any] = []
        # Bind the list's append to the same keys used in the map
        handlers = {key: seen.append for key in handler_map}
        dispatcher = UiEventDispatcher(handlers)

        handled = dispatcher.dispatch(event_name, payload)

        assert handled is True
        assert len(seen) == 1
        item = seen[0]
        assert isinstance(item, expected_type)
        if expected_attrs is not None:
            for key, value in expected_attrs.items():
                assert getattr(item, key) == value
        else:
            # fallback path: raw dict identity
            assert item == payload

    # ------------------------------------------------------------------
    # typed dispatch — event object preserved by identity
    # ------------------------------------------------------------------

    def test_dispatch_typed_hits_registered_class_handler(self) -> None:
        seen: list[GaugeOkEvent] = []
        dispatcher = UiEventDispatcher({GaugeOkEvent: seen.append})
        event = GaugeOkEvent(
            ts=1.0, od=12.3, judge="GO", od2=12.1, judge2="GO", raw="M0,..."
        )

        handled = dispatcher.dispatch_typed(event)

        assert handled is True
        assert seen == [event]
