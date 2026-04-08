import unittest

from modes.validation_mode import ValidationMode, ValidationModeState


class ValidationModeTest(unittest.TestCase):
    def test_unwired_start_fails_fast(self) -> None:
        mode = ValidationMode()

        self.assertEqual(mode.state, ValidationModeState.IDLE)
        self.assertEqual(mode.start(), None)
        self.assertEqual(mode.state, ValidationModeState.ERROR)
        self.assertEqual(mode.last_error, 'Validation start is not wired')

    def test_state_sync_and_reset(self) -> None:
        mode = ValidationMode()

        self.assertEqual(mode.sync_from_workflow_state('PREP'), ValidationModeState.PREPARING)
        self.assertEqual(mode.sync_from_workflow_state('RUN'), ValidationModeState.RUNNING)
        self.assertEqual(mode.sync_from_workflow_state('DONE'), ValidationModeState.COMPLETED)
        self.assertEqual(mode.sync_from_workflow_state('ERR', 'bad data'), ValidationModeState.ERROR)
        self.assertEqual(mode.last_error, 'bad data')
        self.assertEqual(mode.reset(), ValidationModeState.IDLE)
        self.assertIsNone(mode.last_error)


if __name__ == '__main__':
    unittest.main()
