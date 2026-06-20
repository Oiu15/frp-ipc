from __future__ import annotations

from application.adapters.device_gateway import ScreenPresenter


def test_screen_presenter_has_no_getattr() -> None:
    assert "__getattr__" not in ScreenPresenter.__dict__


def test_screen_presenter_keeps_explicit_registry_methods() -> None:
    for method_name in (
        "remember_widget",
        "widget",
        "remember_view_state",
        "view_state",
    ):
        assert method_name in ScreenPresenter.__dict__
