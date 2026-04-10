from __future__ import annotations

import unittest

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

    def start_fixed_section_repeatability_debug(self, **kwargs):
        self.calls.append(dict(kwargs))
        return 'started'

    def _set_validation_debug_feedback(self, **kwargs) -> None:
        self.feedback.append(dict(kwargs))


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

    def test_screen_controller_forwards_validation_motion_options(self) -> None:
        host = _FakeValidationHost()
        controller = ScreenController(host)

        result = controller.start_fixed_section_repeatability_debug(
            section_name=' S1 ',
            metric_name='od_avg',
            repeat_count='2',
            reclamp_enabled='true',
            rotation_stop_before_measure=True,
            release_settle_s='0.25',
            clamp_settle_s='0.5',
            validation_ax3_speed_dps='45',
            move_enabled='true',
            move_axis_name='AX2',
            move_away_delta_mm='12.5',
            move_return_mode='initial_position',
            target_section_pos_mm='300',
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
                    'validation_ax3_speed_dps': 45.0,
                    'move_enabled': True,
                    'move_axis_name': 'AX2',
                    'move_away_delta_mm': 12.5,
                    'move_return_mode': 'initial_position',
                    'target_section_pos_mm': 300.0,
                }
            ],
        )


if __name__ == '__main__':
    unittest.main()
