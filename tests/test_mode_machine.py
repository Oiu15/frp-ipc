import unittest

from modes import CalibrationMode, ModeKind, ModeMachine, ProductionMode, ValidationMode


class _Runner:
    def __init__(self, alive_ref: dict[str, bool], key: str) -> None:
        self._alive_ref = alive_ref
        self._key = key

    def is_alive(self) -> bool:
        return bool(self._alive_ref.get(self._key, False))


class ModeMachineTest(unittest.TestCase):
    def test_enter_transitions_and_stop_current(self) -> None:
        alive = {"production": False, "validation": False}
        stop_calls = {"production": 0, "validation": 0}

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
        )

        self.assertEqual(machine.current_mode_name, "none")
        self.assertEqual(machine.current_state_name, "idle")

        machine.enter_production()
        self.assertEqual(machine.current_mode_kind, ModeKind.PRODUCTION)
        self.assertEqual(machine.current_state_name, "idle")

        production_mode.start()
        production_mode.sync_from_workflow_state("RUN")
        self.assertEqual(machine.current_state_name, "running")

        machine.enter_calibration()
        self.assertEqual(machine.current_mode_kind, ModeKind.CALIBRATION)
        self.assertEqual(stop_calls["production"], 1)
        self.assertEqual(machine.current_state_name, "idle")

        machine.enter_validation()
        self.assertEqual(machine.current_mode_kind, ModeKind.VALIDATION)
        self.assertEqual(machine.current_state_name, "idle")

        self.assertEqual(machine.stop_current(), validation_mode.reset())
        self.assertEqual(machine.current_mode_kind, ModeKind.VALIDATION)
        self.assertEqual(machine.current_state_name, "idle")

    def test_recover_error_resets_current_mode(self) -> None:
        machine = ModeMachine(
            production_mode=ProductionMode(
                start_impl=lambda: None,
                stop_impl=lambda: None,
                runner_getter=lambda: None,
            ),
            calibration_mode=CalibrationMode(),
            validation_mode=ValidationMode(),
        )

        machine.enter_calibration()
        machine.calibration_mode.fail("fit failed")
        machine.last_error = machine.calibration_mode.last_error

        self.assertEqual(machine.current_mode_kind, ModeKind.CALIBRATION)
        self.assertEqual(machine.current_state_name, "error")
        self.assertEqual(machine.last_error, "fit failed")

        self.assertEqual(machine.recover_error(), machine.calibration_mode.reset())
        self.assertEqual(machine.current_state_name, "idle")
        self.assertIsNone(machine.last_error)


if __name__ == "__main__":
    unittest.main()
