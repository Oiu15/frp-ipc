from __future__ import annotations

from typing import cast

from services.calibration_controller import CalibrationController
from services.calibration_service import CalibrationService
from services.measurement_service import MeasurementController
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


class TestControllerModeMachine:
    def test_measurement_controller_uses_mode_machine(self) -> None:
        machine = _FakeModeMachine()
        controller = MeasurementController(mode_machine=cast(ModeMachine, machine))

        assert controller.start_measurement() == "started"
        assert machine.entered == ["production"]
        assert machine.current_mode is not None
        current_mode = cast(_FakeMode, machine.current_mode)
        assert current_mode.start_calls == 1
        assert machine.sync_calls == 1
        assert machine.runtime_state.mode_kind == "production"
        assert machine.runtime_state.mode_state == "preparing"
        assert controller.stop_measurement() == "stopped"
        assert machine.stop_calls == 1
        assert machine.sync_calls == 2

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

        assert machine.entered == ["calibration"]
        assert machine.sync_calls == 1
        assert machine.runtime_state.mode_kind == "calibration"
        assert len(service.calls) == 1
        name, args, kwargs = service.calls[0]
        assert name == "compute_id_candidate"
        assert args == (host,)
        assert kwargs == {}
