import unittest

from application.calibration_controller import CalibrationController
from application.measurement_controller import MeasurementController


class _FakeMode:
    def __init__(self) -> None:
        self.start_calls = 0

    def start(self):
        self.start_calls += 1
        return "started"


class _FakeModeMachine:
    def __init__(self) -> None:
        self.entered: list[str] = []
        self.current_mode = None
        self.stop_calls = 0

    def enter_production(self):
        self.entered.append("production")
        self.current_mode = _FakeMode()
        return self.current_mode

    def enter_calibration(self):
        self.entered.append("calibration")
        return object()

    def stop_current(self):
        self.stop_calls += 1
        return "stopped"


class _FakeCalibrationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        def _recorder(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return _recorder


class ControllerModeMachineTest(unittest.TestCase):
    def test_measurement_controller_uses_mode_machine(self) -> None:
        machine = _FakeModeMachine()
        controller = MeasurementController(mode_machine=machine)

        self.assertEqual(controller.start_measurement(), "started")
        self.assertEqual(machine.entered, ["production"])
        self.assertIsNotNone(machine.current_mode)
        self.assertEqual(machine.current_mode.start_calls, 1)
        self.assertEqual(controller.stop_measurement(), "stopped")
        self.assertEqual(machine.stop_calls, 1)

    def test_calibration_controller_enters_calibration_before_service_call(self) -> None:
        machine = _FakeModeMachine()
        service = _FakeCalibrationService()
        host = object()
        controller = CalibrationController(host=host, service=service, mode_machine=machine)

        controller.compute_id_calibration()

        self.assertEqual(machine.entered, ["calibration"])
        self.assertEqual(len(service.calls), 1)
        name, args, kwargs = service.calls[0]
        self.assertEqual(name, "compute_id_candidate")
        self.assertEqual(args, (host,))
        self.assertEqual(kwargs, {})


if __name__ == "__main__":
    unittest.main()
