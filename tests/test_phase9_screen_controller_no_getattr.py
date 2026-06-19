from __future__ import annotations

from application.adapters.device_gateway import (
    ScreenController,
    ScreenPresenter,
    ScreenUiContext,
)


def test_screen_controller_has_no_getattr() -> None:
    assert "__getattr__" not in ScreenController.__dict__


def test_only_presenter_getattr_remains_for_now() -> None:
    assert "__getattr__" in ScreenPresenter.__dict__
    assert "__getattr__" not in ScreenUiContext.__dict__
