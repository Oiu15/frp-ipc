from __future__ import annotations

from pathlib import Path

from application.adapters.device_gateway import (
    ScreenController,
    ScreenPresenter,
    ScreenUiContext,
)


ROOT = Path(__file__).resolve().parents[1]
def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8-sig")


def test_all_generic_getattr_methods_are_removed() -> None:
    assert "__getattr__" not in ScreenController.__dict__
    assert "__getattr__" not in ScreenUiContext.__dict__
    assert "__getattr__" not in ScreenPresenter.__dict__


def test_screen_presenter_host_fallback_allowlists_are_removed() -> None:
    source = _read("application/adapters/device_gateway.py")
    for name in (
        "_SCREEN_PRESENTER_HOST_ATTR_ALLOWLIST",
        "_SCREEN_PRESENTER_HOST_ATTR_PREFIX_ALLOWLIST",
        "_SCREEN_PRESENTER_HOST_CALL_ALLOWLIST",
        "_SCREEN_PRESENTER_HOST_CALL_PREFIX_ALLOWLIST",
    ):
        assert name not in source


def test_screen_presenter_keeps_explicit_registry_api() -> None:
    for method_name in (
        "remember_widget",
        "widget",
        "remember_view_state",
        "view_state",
    ):
        assert method_name in ScreenPresenter.__dict__


def test_migrated_screens_do_not_use_presenter_dynamic_fallback() -> None:
    for path in (ROOT / "ui" / "screens").glob("*_screen.py"):
        source = path.read_text(encoding="utf-8-sig")
        assert "getattr(presenter" not in source, path.name
        assert "presenter._host" not in source, path.name


def test_generic_screen_presenter_is_not_passed_to_screen_builders() -> None:
    source = _read("application/host/ui.py")

    assert "presenter=self._screen_presenter" not in source


def test_app_host_uses_only_explicit_screen_presenter_registry_methods() -> None:
    source = _read("application/app_host.py")

    assert "presenter = getattr(self, '_screen_presenter', None)" in source
    assert "widget_getter = getattr(presenter, 'widget', None)" in source
    assert "getter = getattr(presenter, 'view_state', None)" in source
    assert "getattr(presenter, name" not in source
