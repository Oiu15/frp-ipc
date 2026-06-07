from __future__ import annotations

import pytest

from modes.calibration_mode import CalibrationMode, CalibrationState


class TestCalibrationMode:
    """CalibrationMode state machine — transition table."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _new() -> CalibrationMode:
        return CalibrationMode()

    # ------------------------------------------------------------------
    # initial state
    # ------------------------------------------------------------------

    def test_initial_state_is_idle(self) -> None:
        mode = self._new()
        assert mode.state == CalibrationState.IDLE
        assert mode.state_name == "idle"
        assert mode.last_error is None

    # ------------------------------------------------------------------
    # forward transitions
    # ------------------------------------------------------------------

    _FORWARD_TRANSITIONS = [
        ("begin_acquiring", CalibrationState.ACQUIRING),
        ("begin_fitting",    CalibrationState.FITTING),
        ("begin_saving",     CalibrationState.SAVING),
    ]

    @pytest.mark.parametrize(("method", "expected_state"), _FORWARD_TRANSITIONS)
    def test_forward_transition(self, method: str, expected_state: CalibrationState) -> None:
        mode = self._new()
        # walk up to the state just before this transition
        for m, _ in self._FORWARD_TRANSITIONS:
            if m == method:
                break
            getattr(mode, m)()
        assert getattr(mode, method)() == expected_state

    # ------------------------------------------------------------------
    # error / recovery
    # ------------------------------------------------------------------

    def test_fail_sets_error_state_and_message(self) -> None:
        mode = self._new()
        mode.begin_acquiring()
        mode.begin_fitting()
        mode.begin_saving()

        assert mode.fail("save failed") == CalibrationState.ERROR
        assert mode.last_error == "save failed"

    def test_complete_clears_error_and_returns_to_idle(self) -> None:
        mode = self._new()
        mode.fail("boom")
        assert mode.state == CalibrationState.ERROR

        assert mode.complete() == CalibrationState.IDLE
        assert mode.last_error is None

    def test_reset_returns_to_idle_from_any_state(self) -> None:
        mode = self._new()
        mode.begin_acquiring()
        assert mode.state == CalibrationState.ACQUIRING

        assert mode.reset() == CalibrationState.IDLE
        assert mode.last_error is None
