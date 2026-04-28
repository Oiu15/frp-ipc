from __future__ import annotations

import tkinter as tk
import unittest
from typing import Any, cast

from application.app_adapters import ScreenController
from application.axis_presenter import AxisScreenPresenter
from application.gauge_presenter import GaugeScreenPresenter


class _FakeVar:
    def __init__(self, value=None) -> None:
        self._value = value

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value


class _FakeHost:
    def __init__(self) -> None:
        self.axis_idx = _FakeVar(0)
        self._axis_snapshot = [object() for _ in range(5)]
        self.some_state = 'ok'


class _FakeAxisController:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def _refresh_axis_panel(self) -> None:
        self.calls.append(('_refresh_axis_panel',))

    def _do_movea(self) -> None:
        self.calls.append(('_do_movea',))

    def _jog_hold(self, direction: str, on: bool) -> None:
        self.calls.append(('_jog_hold', direction, on))


class _FakeGaugeController:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def set_gauge_request_command(self, cmd: str) -> str:
        self.commands.append(cmd)
        return cmd


class _FakeValidationHost:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.feedback: list[dict] = []
        self.stop_calls = 0
        self.navigation_calls = 0

    def start_validation_run(self, **kwargs):
        self.calls.append(dict(kwargs))
        return 'started'

    def stop_validation_run(self):
        self.stop_calls += 1
        return 'stopped'

    def _set_validation_feedback(self, **kwargs) -> None:
        self.feedback.append(dict(kwargs))

    def open_validation_screen(self):
        self.navigation_calls += 1
        return None

    start_fixed_section_repeatability_debug = start_validation_run
    stop_fixed_section_repeatability_debug = stop_validation_run
    _set_validation_debug_feedback = _set_validation_feedback


class ScreenPresenterTest(unittest.TestCase):
    def test_axis_presenter_tracks_current_axis_and_forwards_intent(self) -> None:
        host = _FakeHost()
        controller = _FakeAxisController()
        presenter = AxisScreenPresenter(host, controller)
        presenter.register_axis_widgets(2, {'ent_pos': object()}, _FakeVar(0))

        presenter.handle_axis_selected(2)
        presenter.handle_action(2, '_do_movea')
        presenter.handle_jog(2, 'fwd', True)

        self.assertEqual(host.axis_idx.get(), 2)
        self.assertEqual(presenter.current_axis, 2)
        self.assertIsNotNone(presenter.current_widget('ent_pos'))
        self.assertEqual(
            controller.calls,
            [('_refresh_axis_panel',), ('_do_movea',), ('_jog_hold', 'fwd', True)],
        )

    def test_gauge_presenter_translates_request_change_to_controller_intent(self) -> None:
        host = _FakeHost()
        controller = _FakeGaugeController()
        presenter = GaugeScreenPresenter(host, controller)

        presenter.handle_request_command_changed('M0,1')
        presenter.handle_request_command_changed('')

        self.assertEqual(controller.commands, ['M0,1', 'M1,1'])

    def test_gauge_presenter_initializes_validation_progress_vars(self) -> None:
        host = _FakeHost()
        controller = _FakeGaugeController()
        presenter = GaugeScreenPresenter(host, controller)
        root = tk.Tcl()

        presenter.ensure_vars(master=root)

        self.assertEqual(presenter.validation_phase_var.get(), 'IDLE')
        self.assertEqual(presenter.validation_wait_phase_var.get(), '')
        self.assertEqual(presenter.validation_wait_remaining_s_var.get(), '')
        self.assertEqual(presenter.validation_current_repeat_var.get(), '0/0')
        self.assertEqual(presenter.validation_current_metric_value_var.get(), '')
        self.assertEqual(presenter.validation_current_section_var.get(), '')
        self.assertEqual(presenter.validation_summary_count_var.get(), '0')
        self.assertEqual(presenter.validation_summary_mean_var.get(), '')
        self.assertIs(presenter.validation_phase_var, presenter.validation_debug_phase_var)
        self.assertIs(presenter.validation_section_name_var, presenter.validation_debug_section_name_var)
        self.assertIs(presenter.validation_status_var, presenter.validation_debug_status_var)

    def test_screen_controller_forwards_validation_motion_options(self) -> None:
        host = _FakeValidationHost()
        controller = ScreenController(cast(Any, host))

        result = controller.start_validation_run(
            section_name=' S1 ',
            metric_name='od_avg',
            repeat_count='2',
            reclamp_enabled='true',
            rotation_stop_before_measure=True,
            release_settle_s='0.25',
            clamp_settle_s='0.5',
            position_settle_s='0.75',
            sample_delay_s='0.125',
            validation_ax3_speed_dps='45',
            move_enabled='true',
            move_channel='id_channel',
            move_away_delta_mm='12.5',
            move_scenario='switch_and_return',
            move_from_section_index='1: 100.000',
            move_target_section_index='2: 200.000',
            move_return_section_index='1: 100.000',
        )

        self.assertEqual(result, 'started')
        self.assertEqual(
            host.calls,
            [
                {
                    'section_name': 'S1',
                    'metric_name': 'od_avg',
                    'repeat_count': 2,
                    'reclamp_between_repeats': False,
                    'reclamp_enabled': True,
                    'rotation_stop_before_measure': True,
                    'release_settle_s': 0.25,
                    'clamp_settle_s': 0.5,
                    'position_settle_s': 0.75,
                    'sample_delay_s': 0.125,
                    'validation_ax3_speed_dps': 45.0,
                    'move_enabled': True,
                    'move_channel': 'id_channel',
                    'move_away_delta_mm': 12.5,
                    'move_scenario': 'switch_and_return',
                    'move_from_section_index': 1,
                    'move_target_section_index': 2,
                    'move_return_section_index': 1,
                }
            ],
        )

    def test_screen_controller_forwards_validation_stop(self) -> None:
        host = _FakeValidationHost()
        controller = ScreenController(cast(Any, host))

        result = controller.stop_validation_run()

        self.assertEqual(result, 'stopped')
        self.assertEqual(host.stop_calls, 1)

    def test_screen_controller_exposes_validation_screen_navigation(self) -> None:
        host = _FakeValidationHost()
        controller = ScreenController(cast(Any, host))

        result = controller.open_validation_screen()

        self.assertIsNone(result)
        self.assertEqual(host.navigation_calls, 1)

    def test_screen_controller_validation_debug_aliases_forward_to_existing_chain(self) -> None:
        host = _FakeValidationHost()
        controller = ScreenController(cast(Any, host))

        result = controller.start_fixed_section_repeatability_debug(
            section_name='S1',
            metric_name='od_avg',
            repeat_count='1',
            move_enabled=False,
            move_channel='od_channel',
            move_away_delta_mm='0.0',
            move_scenario='distance_round_trip',
            move_from_section_index='1',
            move_target_section_index='1',
            move_return_section_index='1',
        )
        stop_result = controller.stop_fixed_section_repeatability_debug()

        self.assertEqual(result, 'started')
        self.assertEqual(stop_result, 'stopped')
        self.assertEqual(len(host.calls), 1)
        self.assertEqual(host.calls[0]['section_name'], 'S1')
        self.assertEqual(host.calls[0]['metric_name'], 'od_avg')
        self.assertEqual(host.stop_calls, 1)


if __name__ == '__main__':
    unittest.main()
