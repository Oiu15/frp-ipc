from __future__ import annotations

from dataclasses import fields

from ui.presenters.validation_presenter_deps import ValidationUiState


def test_validation_ui_state_keeps_field_identity_and_widgets() -> None:
    values = {
        field.name: object()
        for field in fields(ValidationUiState)
        if field.init
    }

    state = ValidationUiState(**values)

    for name, value in values.items():
        assert getattr(state, name) is value

    widget = object()
    assert state.remember_widget("validation_screen_start_btn", widget) is widget
    assert state.widget("validation_screen_start_btn") is widget
    assert state.ensure_vars(object()) is None
