import unittest

from application.app_adapters import AppDeviceGateway
from application.contracts import ValidationActionCancelled, ValidationActionGateway


class FakeGatewayApp:
    def __init__(self) -> None:
        self.stopped_axes: list[int] = []
        self.y_writes: list[tuple[int, int]] = []
        self.validation_cancel_requested = False

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


if __name__ == "__main__":
    unittest.main()
