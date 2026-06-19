from __future__ import annotations

from application.adapters.device_gateway import ScreenPresenter, ScreenUiContext


def test_screen_ui_context_has_no_getattr() -> None:
    assert "__getattr__" not in ScreenUiContext.__dict__


def test_screen_presenter_getattr_is_removed() -> None:
    assert "__getattr__" not in ScreenPresenter.__dict__
