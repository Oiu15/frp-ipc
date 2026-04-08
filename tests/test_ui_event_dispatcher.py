import unittest

from application.ui_event_dispatcher import UiEventDispatcher
from application.ui_events import GaugeOkEvent, PlcErrEvent


class UiEventDispatcherTest(unittest.TestCase):
    def test_dispatch_prefers_typed_handler_for_known_event(self) -> None:
        seen: list[PlcErrEvent] = []
        dispatcher = UiEventDispatcher({PlcErrEvent: seen.append})

        handled = dispatcher.dispatch('plc_err', {'err': 'boom', 'retry': 1, 'max': 3})

        self.assertTrue(handled)
        self.assertEqual(len(seen), 1)
        self.assertIsInstance(seen[0], PlcErrEvent)
        self.assertEqual(seen[0].err, 'boom')
        self.assertEqual(seen[0].retry, 1)

    def test_dispatch_uses_string_fallback_for_unknown_event(self) -> None:
        seen: list[dict] = []
        dispatcher = UiEventDispatcher({'custom_evt': seen.append})

        handled = dispatcher.dispatch('custom_evt', {'value': 42})

        self.assertTrue(handled)
        self.assertEqual(seen, [{'value': 42}])

    def test_dispatch_typed_event_hits_registered_class_handler(self) -> None:
        seen: list[GaugeOkEvent] = []
        dispatcher = UiEventDispatcher({GaugeOkEvent: seen.append})
        event = GaugeOkEvent(ts=1.0, od=12.3, judge='GO', od2=12.1, judge2='GO', raw='M0,...')

        handled = dispatcher.dispatch_typed(event)

        self.assertTrue(handled)
        self.assertEqual(seen, [event])


if __name__ == '__main__':
    unittest.main()
