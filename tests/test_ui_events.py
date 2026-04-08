import unittest

from application.ui_events import (
    AutoRowEvent,
    AutoStateEvent,
    GaugeOkEvent,
    PlcErrEvent,
    PlcOkEvent,
    parse_ui_event,
    parse_ui_event_tuple,
)
from core.models import AxisComm, MeasureRow


class UiEventsTest(unittest.TestCase):
    def test_plc_err_roundtrip(self) -> None:
        payload = {'err': 'connect failed', 'retry': 2, 'max': 5, 'backoff_s': 15.0}
        event = PlcErrEvent.from_payload(payload)
        self.assertEqual(event.err, 'connect failed')
        self.assertEqual(event.retry, 2)
        self.assertEqual(event.max, 5)
        self.assertEqual(event.backoff_s, 15.0)
        self.assertEqual(event.to_ui_event(), ('plc_err', payload))

    def test_plc_ok_parse_preserves_axes_and_bits(self) -> None:
        payload = {
            'axes': [AxisComm(act_pos=12.5)],
            'cl_out4_mm': 152.7,
            'cl_out4_cnt': 3,
            'keytest_x_bits': [1, 0, 1],
            'keytest_y_bits': [0, 1],
        }
        event = parse_ui_event('plc_ok', payload)
        self.assertIsInstance(event, PlcOkEvent)
        assert isinstance(event, PlcOkEvent)
        self.assertEqual(len(event.axes), 1)
        self.assertAlmostEqual(event.axes[0].act_pos, 12.5)
        self.assertEqual(event.cl_out4_mm, 152.7)
        self.assertEqual(event.keytest_x_bits, [1, 0, 1])
        self.assertEqual(event.keytest_y_bits, [0, 1])

    def test_gauge_ok_roundtrip(self) -> None:
        event = GaugeOkEvent(ts=123.4, od=187.3, judge='GO', od2=187.1, judge2='GO', raw='M0,1,+187.3,GO,+187.1,GO')
        parsed = parse_ui_event_tuple(event.to_ui_event())
        self.assertIsInstance(parsed, GaugeOkEvent)
        assert isinstance(parsed, GaugeOkEvent)
        self.assertAlmostEqual(parsed.od, 187.3)
        self.assertEqual(parsed.judge, 'GO')
        self.assertAlmostEqual(parsed.od2 or 0.0, 187.1)

    def test_auto_state_and_row_roundtrip(self) -> None:
        state = AutoStateEvent(state='DONE', msg='completed')
        parsed_state = parse_ui_event_tuple(state.to_ui_event())
        self.assertIsInstance(parsed_state, AutoStateEvent)
        assert isinstance(parsed_state, AutoStateEvent)
        self.assertEqual(parsed_state.state, 'DONE')
        self.assertEqual(parsed_state.msg, 'completed')

        row = MeasureRow(
            idx=1,
            x_ui=100.0,
            x_abs=200.0,
            od_avg=187.31,
            od_dev=0.01,
            od_runout=0.02,
            od_round=0.03,
            id_avg=152.70,
            id_dev=0.01,
            id_runout=0.02,
            id_round=0.03,
            concentricity=0.04,
        )
        row_event = AutoRowEvent(row=row)
        parsed_row = parse_ui_event_tuple(row_event.to_ui_event())
        self.assertIsInstance(parsed_row, AutoRowEvent)
        assert isinstance(parsed_row, AutoRowEvent)
        self.assertIs(parsed_row.row, row)


if __name__ == '__main__':
    unittest.main()
