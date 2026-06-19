from __future__ import annotations

from ui.presenters.key_test_presenter import KeyTestUiState


def test_key_test_ui_state_keeps_field_identity() -> None:
    x_vars = [object()]
    y_vars = [object()]
    y_lastcmd_vars = [object()]

    state = KeyTestUiState(
        keytest_x_vars=x_vars,
        keytest_y_vars=y_vars,
        keytest_y_lastcmd_vars=y_lastcmd_vars,
    )

    assert state.keytest_x_vars is x_vars
    assert state.keytest_y_vars is y_vars
    assert state.keytest_y_lastcmd_vars is y_lastcmd_vars
