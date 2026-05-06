import unittest
from typing import cast

from application.calibration_controller import CalibrationController
from application.calibration_service import CalibrationService
from application.measurement_controller import MeasurementController
from modes.mode_machine import ModeMachine


class _FakeMode:
    def __init__(self) -> None:
        self.start_calls = 0
        self.state_name = "idle"
        self.last_error = None

    def start(self):
        self.start_calls += 1
        self.state_name = "preparing"
        return "started"


class _FakeRuntimeState:
    def __init__(self) -> None:
        self.mode_kind = "none"
        self.mode_state = "idle"
        self.mode_error = None


class _FakeModeMachine:
    def __init__(self) -> None:
        self.entered: list[str] = []
        self.current_mode = None
        self.stop_calls = 0
        self.sync_calls = 0
        self.runtime_state = _FakeRuntimeState()

    def enter_production(self):
        self.entered.append("production")
        self.runtime_state.mode_kind = "production"
        self.current_mode = _FakeMode()
        return self.current_mode

    def enter_calibration(self):
        self.entered.append("calibration")
        self.runtime_state.mode_kind = "calibration"
        return object()

    def stop_current(self):
        self.stop_calls += 1
        self.runtime_state.mode_state = "idle"
        return "stopped"

    def sync_current_mode_state(self):
        self.sync_calls += 1
        if self.current_mode is not None:
            self.runtime_state.mode_state = self.current_mode.state_name
            self.runtime_state.mode_error = self.current_mode.last_error


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
        controller = MeasurementController(mode_machine=cast(ModeMachine, machine))

        self.assertEqual(controller.start_measurement(), "started")
        self.assertEqual(machine.entered, ["production"])
        self.assertIsNotNone(machine.current_mode)
        current_mode = cast(_FakeMode, machine.current_mode)
        self.assertEqual(current_mode.start_calls, 1)
        self.assertEqual(machine.sync_calls, 1)
        self.assertEqual(machine.runtime_state.mode_kind, "production")
        self.assertEqual(machine.runtime_state.mode_state, "preparing")
        self.assertEqual(controller.stop_measurement(), "stopped")
        self.assertEqual(machine.stop_calls, 1)
        self.assertEqual(machine.sync_calls, 2)

    def test_calibration_controller_enters_calibration_before_service_call(self) -> None:
        machine = _FakeModeMachine()
        service = _FakeCalibrationService()
        host = object()
        controller = CalibrationController(
            host=host,
            service=cast(CalibrationService, service),
            mode_machine=cast(ModeMachine, machine),
        )

        controller.compute_id_calibration()

        self.assertEqual(machine.entered, ["calibration"])
        self.assertEqual(machine.sync_calls, 1)
        self.assertEqual(machine.runtime_state.mode_kind, "calibration")
        self.assertEqual(len(service.calls), 1)
        name, args, kwargs = service.calls[0]
        self.assertEqual(name, "compute_id_candidate")
        self.assertEqual(args, (host,))
        self.assertEqual(kwargs, {})


if __name__ == "__main__":
    unittest.main()
