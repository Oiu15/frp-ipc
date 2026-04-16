import unittest
from types import SimpleNamespace
from unittest.mock import patch

from application.app_adapters import AppDeviceGateway
from application.contracts import ValidationActionCancelled, ValidationActionGateway
from core.models import AxisCal


class FakeGatewayApp:
    def __init__(self) -> None:
        self.stopped_axes: list[int] = []
        self.y_writes: list[tuple[int, int]] = []
        self.movea_calls: list[tuple[int, float, str]] = []
        self.limit_calls: list[tuple[int, float, bool, str]] = []
        self.axis_positions: dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0, 4: 0.0}
        self.axis_position_reads: dict[int, list[float]] = {}
        self.validation_cancel_requested = False
        self.axis_cal = AxisCal()

    def get_axis_copy(self, axis: int):
        ax = int(axis)
        reads = self.axis_position_reads.get(ax)
        if reads:
            if len(reads) > 1:
                value = reads.pop(0)
            else:
                value = reads[0]
        else:
            value = self.axis_positions.get(ax, 0.0)
        return SimpleNamespace(act_pos=value, softlim_pos=0.0, softlim_neg=0.0)

    def apply_soft_limits_abs(self, axis: int, target_abs: float, *, strict: bool = False, context: str = "") -> float:
        self.limit_calls.append((int(axis), float(target_abs), bool(strict), str(context)))
        return float(target_abs)

    def movea_abs(self, axis: int, pos_abs: float, *, context: str = "MoveA") -> None:
        self.movea_calls.append((int(axis), float(pos_abs), str(context)))

    def stop(self, axis: int) -> None:
        self.stopped_axes.append(int(axis))

    def plc_write_y_point(self, y_point: int, value: int) -> None:
        self.y_writes.append((int(y_point), int(value)))

    def is_validation_cancel_requested(self) -> bool:
        return bool(self.validation_cancel_requested)


class AppDeviceGatewayValidationActionTest(unittest.TestCase):
    def test_validation_action_gateway_protocol_is_implemented(self) -> None:
        gateway = AppDeviceGateway(FakeGatewayApp())

        self.assertIsInstance(gateway, ValidationActionGateway)

    def test_stop_rotation_targets_ax3(self) -> None:
        app = FakeGatewayApp()
        gateway = AppDeviceGateway(app)

        gateway.stop_rotation()

        self.assertEqual(app.stopped_axes, [3])

    def test_clamp_release_and_close_write_dual_clamp_outputs(self) -> None:
        app = FakeGatewayApp()
        gateway = AppDeviceGateway(app)

        gateway.clamp_release()
        gateway.clamp_close()

        self.assertEqual(
            app.y_writes,
            [
                (10, 0),
                (11, 0),
                (10, 1),
                (11, 1),
            ],
        )

    def test_legacy_dual_clamp_methods_delegate_to_validation_actions(self) -> None:
        app = FakeGatewayApp()
        gateway = AppDeviceGateway(app)

        gateway.open_dual_clamps()
        gateway.close_dual_clamps()

        self.assertEqual(
            app.y_writes,
            [
                (10, 0),
                (11, 0),
                (10, 1),
                (11, 1),
            ],
        )

    def test_wait_cancelable_returns_after_duration(self) -> None:
        gateway = AppDeviceGateway(FakeGatewayApp())

        gateway.wait_cancelable(0.001, poll_interval_s=0.001)

    def test_wait_cancelable_raises_when_callback_requests_cancel(self) -> None:
        gateway = AppDeviceGateway(FakeGatewayApp())

        with self.assertRaises(ValidationActionCancelled):
            gateway.wait_cancelable(1.0, poll_interval_s=0.001, cancel_check=lambda: True)

    def test_wait_cancelable_raises_when_app_requests_cancel(self) -> None:
        app = FakeGatewayApp()
        app.validation_cancel_requested = True
        gateway = AppDeviceGateway(app)

        with self.assertRaises(ValidationActionCancelled):
            gateway.wait_cancelable(1.0, poll_interval_s=0.001)

    def test_read_axis_position_mm_reads_latest_axis_snapshot(self) -> None:
        app = FakeGatewayApp()
        app.axis_positions[2] = 123.456
        gateway = AppDeviceGateway(app)

        self.assertEqual(gateway.read_axis_position_mm(2), 123.456)

    def test_move_axis_absolute_delegates_movea_and_returns_target(self) -> None:
        app = FakeGatewayApp()
        gateway = AppDeviceGateway(app)

        target = gateway.move_axis_absolute(0, 125.5, context="VALIDATION_MOVE_AWAY")

        self.assertEqual(target, 125.5)
        self.assertEqual(app.limit_calls, [(0, 125.5, False, "VALIDATION_MOVE_AWAY")])
        self.assertEqual(app.movea_calls, [(0, 125.5, "VALIDATION_MOVE_AWAY")])

    def test_move_axis_relative_reads_current_position_and_moves_to_delta_target(self) -> None:
        app = FakeGatewayApp()
        app.axis_positions[0] = 100.0
        gateway = AppDeviceGateway(app)

        target = gateway.move_axis_relative(0, -12.5, context="VALIDATION_MOVE_AWAY")

        self.assertEqual(target, 87.5)
        self.assertEqual(app.movea_calls, [(0, 87.5, "VALIDATION_MOVE_AWAY")])

    def test_move_axes_absolute_issues_all_move_commands_after_resolving_targets(self) -> None:
        app = FakeGatewayApp()
        gateway = AppDeviceGateway(app)

        targets = gateway.move_axes_absolute(
            {1: 25.0, 4: 75.0},
            context="VALIDATION_MOVE_AWAY",
        )

        self.assertEqual(dict(targets), {1: 25.0, 4: 75.0})
        self.assertEqual(
            app.limit_calls,
            [
                (1, 25.0, False, "VALIDATION_MOVE_AWAY"),
                (4, 75.0, False, "VALIDATION_MOVE_AWAY"),
            ],
        )
        self.assertEqual(
            app.movea_calls,
            [
                (1, 25.0, "VALIDATION_MOVE_AWAY"),
                (4, 75.0, "VALIDATION_MOVE_AWAY"),
            ],
        )

    def test_wait_axis_in_position_returns_when_position_reaches_tolerance(self) -> None:
        app = FakeGatewayApp()
        app.axis_position_reads[0] = [95.0, 99.95]
        gateway = AppDeviceGateway(app)

        actual = gateway.wait_axis_in_position(
            0,
            100.0,
            tolerance_mm=0.1,
            timeout_s=0.1,
            poll_interval_s=0.001,
        )

        self.assertEqual(actual, 99.95)

    def test_wait_axis_in_position_is_cancel_aware(self) -> None:
        app = FakeGatewayApp()
        app.axis_positions[0] = 95.0
        gateway = AppDeviceGateway(app)

        with self.assertRaises(ValidationActionCancelled):
            gateway.wait_axis_in_position(
                0,
                100.0,
                timeout_s=1.0,
                poll_interval_s=0.001,
                cancel_check=lambda: True,
            )

    def test_wait_axis_in_position_logs_timeout_source_and_poll_profile(self) -> None:
        app = FakeGatewayApp()
        app.axis_positions[0] = 95.0
        app._plc_poll_profile_req = "sampling"
        gateway = AppDeviceGateway(app)

        with patch("application.app_adapters.log") as mock_log:
            with self.assertRaises(TimeoutError):
                gateway.wait_axis_in_position(
                    0,
                    100.0,
                    timeout_s=0.0,
                    poll_interval_s=0.001,
                )

        mock_log.assert_called_once_with(
            "VALIDATION_WAIT_INPOS_TIMEOUT",
            axis=0,
            target=100.0,
            actual=95.0,
            timeout_s=0.0,
            tolerance=0.1,
            actual_source="axis_snapshot",
            current_poll_profile="sampling",
        )


if __name__ == "__main__":
    unittest.main()
