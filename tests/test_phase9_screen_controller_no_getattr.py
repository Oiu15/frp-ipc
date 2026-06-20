from __future__ import annotations

from application.adapters.device_gateway import (
    ScreenController,
    ScreenPresenter,
    ScreenUiContext,
)


def test_screen_controller_has_no_getattr() -> None:
    assert "__getattr__" not in ScreenController.__dict__


def test_presenter_and_ui_context_getattr_are_removed() -> None:
    assert "__getattr__" not in ScreenPresenter.__dict__
    assert "__getattr__" not in ScreenUiContext.__dict__
