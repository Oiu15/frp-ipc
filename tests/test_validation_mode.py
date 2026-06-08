from __future__ import annotations

import pytest

from modes.validation_mode import ValidationMode, ValidationModeState


class TestValidationMode:
    """ValidationMode — unwired start + state-sync transitions."""

    # ------------------------------------------------------------------
    # unwired start
    # ------------------------------------------------------------------

    def test_unwired_start_fails_fast(self) -> None:
        mode = ValidationMode()
        assert mode.state == ValidationModeState.IDLE
        assert mode.start() is None
        assert mode.state == ValidationModeState.ERROR
        assert mode.last_error == "Validation start is not wired"

    # ------------------------------------------------------------------
    # sync_from_workflow_state — table-driven
    # ------------------------------------------------------------------

    _SYNC_CASES: list[tuple[str, str, ValidationModeState, str | None]] = [
        ("PREP", "",        ValidationModeState.PREPARING,  None),
        ("RUN",  "",        ValidationModeState.RUNNING,    None),
        ("DONE", "",        ValidationModeState.COMPLETED,  None),
        ("ERR",  "bad data", ValidationModeState.ERROR,     "bad data"),
    ]

    @pytest.mark.parametrize(
        ("workflow_state", "message", "expected_state", "expected_error"),
        _SYNC_CASES,
    )
    def test_sync_from_workflow_state(
        self,
        workflow_state: str,
        message: str,
        expected_state: ValidationModeState,
        expected_error: str | None,
    ) -> None:
        mode = ValidationMode()
        assert mode.sync_from_workflow_state(workflow_state, message) == expected_state
        assert mode.last_error == expected_error

    def test_reset_returns_to_idle_and_clears_error(self) -> None:
        mode = ValidationMode()
        mode.sync_from_workflow_state("ERR", "bad data")
        assert mode.state == ValidationModeState.ERROR

        assert mode.reset() == ValidationModeState.IDLE
        assert mode.last_error is None
