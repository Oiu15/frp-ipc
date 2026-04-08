import unittest

from modes.calibration_mode import CalibrationMode, CalibrationState


class CalibrationModeTest(unittest.TestCase):
    def test_state_transitions(self) -> None:
        mode = CalibrationMode()

        self.assertEqual(mode.state, CalibrationState.IDLE)
        self.assertEqual(mode.state_name, "idle")
        self.assertIsNone(mode.last_error)

        self.assertEqual(mode.begin_acquiring(), CalibrationState.ACQUIRING)
        self.assertEqual(mode.begin_fitting(), CalibrationState.FITTING)
        self.assertEqual(mode.begin_saving(), CalibrationState.SAVING)

        self.assertEqual(mode.fail("save failed"), CalibrationState.ERROR)
        self.assertEqual(mode.last_error, "save failed")

        self.assertEqual(mode.complete(), CalibrationState.IDLE)
        self.assertIsNone(mode.last_error)

        mode.begin_acquiring()
        self.assertEqual(mode.reset(), CalibrationState.IDLE)
        self.assertIsNone(mode.last_error)


if __name__ == "__main__":
    unittest.main()
