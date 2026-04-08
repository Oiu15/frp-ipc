import types
import unittest

from application.gauge_presenter import GaugeScreenPresenter
from application.app_adapters import ScreenPresenter
from application.app_host import AppHost


class _FakeVar:
    def __init__(self, value=None) -> None:
        self._value = value

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value


class _FakeModeMachine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def sync_production_workflow_state(self, state: str, message: str) -> None:
        self.calls.append((state, message))


class _FakeHost:
    def __init__(self) -> None:
        self.gauge_conn_var = _FakeVar('')
        self.plc_status_var = _FakeVar('')
        self.auto_progress_var = _FakeVar('')
        self.auto_done_var = _FakeVar('')
        self.auto_state_var = _FakeVar('IDLE')
        self.auto_msg_var = _FakeVar('-')
        self._auto_cur_sec_idx = None
        self.mode_machine = _FakeModeMachine()
        self._trigger_run_export_calls = 0
        self._freeze_run_end_ts_if_missing_calls = 0

    def _trigger_run_export(self) -> None:
        self._trigger_run_export_calls += 1

    def _freeze_run_end_ts_if_missing(self) -> None:
        self._freeze_run_end_ts_if_missing_calls += 1


def _bind_routing_methods(host: _FakeHost) -> None:
    names = [
        '_handle_plc_ok_event',
        '_handle_plc_err_event',
        '_handle_plc_giveup_event',
        '_handle_plc_manual_event',
        '_handle_plc_read_event',
        '_handle_gauge_conn_event',
        '_handle_gauge_tx_event',
        '_handle_gauge_ok_event',
        '_handle_gauge_raw_event',
        '_handle_gauge_err_event',
        '_handle_op_confirm_show_event',
        '_handle_op_confirm_close_event',
        '_handle_auto_clear_event',
        '_handle_auto_len_event',
        '_handle_auto_progress_event',
        '_handle_auto_coverage_event',
        '_handle_auto_straightness_event',
        '_handle_auto_postcalc_event',
        '_handle_auto_raw_points_event',
        '_handle_auto_row_event',
        '_handle_auto_state_event',
    ]
    for name in names:
        setattr(host, name, types.MethodType(getattr(AppHost, name), host))


class TypedUiEventRoutingTest(unittest.TestCase):
    def test_gauge_conn_event_routes_to_device_handler_and_presenter_state(self) -> None:
        host = _FakeHost()
        _bind_routing_methods(host)
        dispatcher = AppHost._build_device_ui_event_dispatcher(host)
        presenter = GaugeScreenPresenter(host, controller=object())

        handled = dispatcher.dispatch('gauge_conn', {'ts': 1.0, 'connected': True, 'port': 'COM3', 'baud': 115200})

        self.assertTrue(handled)
        self.assertIn('COM3@115200', presenter.gauge_conn_var.get())

    def test_auto_progress_event_routes_to_measurement_handler_and_presenter_state(self) -> None:
        host = _FakeHost()
        _bind_routing_methods(host)
        dispatcher = AppHost._build_measurement_ui_event_dispatcher(host)
        presenter = ScreenPresenter(host)

        handled = dispatcher.dispatch('auto_progress', {'idx': 1, 'total': 5, 'x_ui': 100.0, 'x_abs': 200.0})

        self.assertTrue(handled)
        self.assertEqual(host._auto_cur_sec_idx, 2)
        self.assertIn('2', presenter.auto_progress_var.get())
        self.assertIsNotNone(presenter.auto_done_var.get())

    def test_auto_state_done_routes_to_state_handler_and_done_side_effect(self) -> None:
        host = _FakeHost()
        _bind_routing_methods(host)
        dispatcher = AppHost._build_measurement_ui_event_dispatcher(host)
        presenter = ScreenPresenter(host)

        handled = dispatcher.dispatch('auto_state', {'state': 'DONE', 'msg': 'completed'})

        self.assertTrue(handled)
        self.assertEqual(host.mode_machine.calls, [('DONE', 'completed')])
        self.assertEqual(presenter.auto_state_var.get(), 'DONE')
        self.assertEqual(presenter.auto_msg_var.get(), 'completed')
        self.assertEqual(host._trigger_run_export_calls, 1)
        self.assertEqual(host._freeze_run_end_ts_if_missing_calls, 0)

    def test_plc_err_event_routes_to_plc_status_presenter(self) -> None:
        host = _FakeHost()
        _bind_routing_methods(host)
        dispatcher = AppHost._build_device_ui_event_dispatcher(host)
        presenter = ScreenPresenter(host)

        handled = dispatcher.dispatch('plc_err', {'err': 'connect failed', 'retry': 2, 'max': 5, 'backoff_s': 10.0})

        self.assertTrue(handled)
        self.assertIn('connect failed', presenter.plc_status_var.get())
        self.assertIn('2/5', presenter.plc_status_var.get())


if __name__ == '__main__':
    unittest.main()
