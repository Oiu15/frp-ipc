
from domain.state import RuntimeState
from modes import CalibrationMode, ModeKind, ModeMachine, ProductionMode, ValidationMode


class _Runner:
    def __init__(self, alive_ref: dict[str, bool], key: str) -> None:
        self._alive_ref = alive_ref
        self._key = key

    def is_alive(self) -> bool:
        return bool(self._alive_ref.get(self._key, False))


class TestModeMachine:
    def test_transition_matrix_minimum_paths(self) -> None:
        alive = {"production": False, "validation": False}
        stop_calls = {"production": 0, "validation": 0}
        runtime_state = RuntimeState()

        def start_production() -> str:
            alive["production"] = True
            return "started"

        def stop_production() -> str:
            stop_calls["production"] += 1
            alive["production"] = False
            return "stopped"

        production_mode = ProductionMode(
            start_impl=start_production,
            stop_impl=stop_production,
            runner_getter=lambda: _Runner(alive, "production"),
        )
        validation_mode = ValidationMode(
            start_impl=lambda: alive.__setitem__("validation", True),
            stop_impl=lambda: stop_calls.__setitem__("validation", stop_calls["validation"] + 1),
            runner_getter=lambda: _Runner(alive, "validation"),
        )
        calibration_mode = CalibrationMode()
        machine = ModeMachine(
            production_mode=production_mode,
            calibration_mode=calibration_mode,
            validation_mode=validation_mode,
            runtime_state=runtime_state,
        )

        assert machine.current_mode_name == "none"
        assert machine.current_state_name == "idle"

        machine.enter_production()
        assert machine.current_mode_kind == ModeKind.PRODUCTION
        assert runtime_state.mode_kind == "production"
        assert runtime_state.mode_state == "idle"

        production_mode.start()
        production_mode.sync_from_workflow_state("RUN")
        machine.sync_current_mode_state()
        assert machine.current_state_name == "running"
        assert runtime_state.mode_state == "running"

        machine.stop_current()
        assert stop_calls["production"] == 1
        assert machine.current_mode_kind == ModeKind.PRODUCTION
        assert runtime_state.mode_kind == "production"
        assert machine.current_state_name == "stopping"
        assert runtime_state.mode_state == "stopping"

        machine.enter_calibration()
        assert machine.current_mode_kind == ModeKind.CALIBRATION
        assert runtime_state.mode_kind == "calibration"
        assert machine.current_state_name == "idle"
        assert runtime_state.mode_state == "idle"

        calibration_mode.begin_acquiring()
        machine.sync_current_mode_state()
        assert machine.current_state_name == "acquiring"
        assert runtime_state.mode_state == "acquiring"

        machine.enter_validation()
        assert machine.current_mode_kind == ModeKind.VALIDATION
        assert runtime_state.mode_kind == "validation"
        assert machine.current_state_name == "idle"
        assert runtime_state.mode_state == "idle"

        machine.sync_validation_workflow_state("ERR", "validation failed")
        assert machine.current_state_name == "error"
        assert runtime_state.mode_error == "validation failed"

        machine.recover_error()
        assert machine.current_state_name == "idle"
        assert runtime_state.mode_state == "idle"
        assert runtime_state.mode_error is None

    def test_enter_transitions_and_stop_current(self) -> None:
        alive = {"production": False, "validation": False}
        stop_calls = {"production": 0, "validation": 0}
        runtime_state = RuntimeState()

        def start_production() -> str:
            alive["production"] = True
            return "started"

        def stop_production() -> str:
            stop_calls["production"] += 1
            alive["production"] = False
            return "stopped"

        production_mode = ProductionMode(
            start_impl=start_production,
            stop_impl=stop_production,
            runner_getter=lambda: _Runner(alive, "production"),
        )
        validation_mode = ValidationMode(
            start_impl=lambda: alive.__setitem__("validation", True),
            stop_impl=lambda: stop_calls.__setitem__("validation", stop_calls["validation"] + 1),
            runner_getter=lambda: _Runner(alive, "validation"),
        )
        machine = ModeMachine(
            production_mode=production_mode,
            calibration_mode=CalibrationMode(),
            validation_mode=validation_mode,
            runtime_state=runtime_state,
        )

        assert machine.current_mode_name == "none"
        assert machine.current_state_name == "idle"
        assert runtime_state.mode_kind == "none"
        assert runtime_state.mode_state == "idle"

        machine.enter_production()
        assert machine.current_mode_kind == ModeKind.PRODUCTION
        assert machine.current_state_name == "idle"
        assert runtime_state.mode_kind == "production"

        production_mode.start()
        machine.sync_current_mode_state()
        production_mode.sync_from_workflow_state("RUN")
        machine.sync_current_mode_state()
        assert machine.current_state_name == "running"
        assert runtime_state.mode_state == "running"

        machine.enter_calibration()
        assert machine.current_mode_kind == ModeKind.CALIBRATION
        assert stop_calls["production"] == 1
        assert machine.current_state_name == "idle"
        assert runtime_state.mode_kind == "calibration"

        machine.enter_validation()
        assert machine.current_mode_kind == ModeKind.VALIDATION
        assert machine.current_state_name == "idle"
        assert runtime_state.mode_kind == "validation"

        machine.stop_current()
        assert machine.current_mode_kind == ModeKind.VALIDATION
        assert machine.current_state_name == "idle"
        assert runtime_state.mode_state == "idle"

    def test_recover_error_resets_current_mode(self) -> None:
        runtime_state = RuntimeState()
        machine = ModeMachine(
            production_mode=ProductionMode(
                start_impl=lambda: None,
                stop_impl=lambda: None,
                runner_getter=lambda: None,
            ),
            calibration_mode=CalibrationMode(),
            validation_mode=ValidationMode(),
            runtime_state=runtime_state,
        )

        machine.enter_calibration()
        machine.calibration_mode.fail("fit failed")
        machine.sync_current_mode_state()

        assert machine.current_mode_kind == ModeKind.CALIBRATION
        assert machine.current_state_name == "error"
        assert machine.last_error == "fit failed"
        assert runtime_state.mode_error == "fit failed"

        machine.recover_error()
        assert machine.current_state_name == "idle"
        assert machine.last_error is None
        assert runtime_state.mode_error is None
