from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8-sig")


def test_generic_screen_fallback_methods_remain_legacy_inventory() -> None:
    source = _read("application/adapters/device_gateway.py")

    assert "class ScreenPresenter" in source
    assert "class ScreenController" in source
    assert "class ScreenUiContext" in source
    assert "_SCREEN_PRESENTER_HOST_ATTR_ALLOWLIST" in source
    assert "_SCREEN_CONTROLLER_HOST_CALL_ALLOWLIST" in source
    assert "_SCREEN_UI_CONTEXT_ATTR_ALLOWLIST" in source
    assert 'def __getattr__(self, name: str) -> Any:' in source
    assert '"axis_cal_"' in source
    assert '"validation_"' in source


def test_recipe_and_keytest_are_wired_to_explicit_objects() -> None:
    source = _read("application/host/ui.py")

    assert "build_recipe_screen(tab_recipe, presenter=self._recipe_screen_presenter, controller=self.recipe_controller" in source
    assert "build_key_test_screen(tab_keytest, presenter=self.key_test_ui, controller=self.key_test_controller" in source
    assert "build_key_test_screen(tab_keytest, presenter=self._screen_presenter" not in source
    assert "build_key_test_screen(tab_keytest, presenter=self.key_test_ui, controller=self._screen_controller" not in source


def test_recipe_presenter_does_not_use_host_fallback_tokens() -> None:
    source = _read("ui/presenters/recipe_presenter.py")

    for forbidden in (
        "self._host",
        "host_app",
        "getattr(",
        "__getattr__",
    ):
        assert forbidden not in source


def test_recipe_and_keytest_screens_do_not_use_dynamic_fallback_tokens() -> None:
    for path in (
        "ui/screens/recipe_screen.py",
        "ui/screens/key_test_screen.py",
    ):
        source = _read(path)
        for forbidden in (
            "getattr(controller",
            "getattr(presenter",
            "getattr(ui",
            "controller._host",
            "presenter._host",
            "ui._host",
        ):
            assert forbidden not in source


def test_keytest_generic_allowlist_entries_are_legacy_only() -> None:
    source = _read("application/adapters/device_gateway.py")
    ui_source = _read("application/host/ui.py")

    assert '"keytest_"' in source
    assert '"write_keytest_"' in source
    assert "presenter=self.key_test_ui" in ui_source
    assert "controller=self.key_test_controller" in ui_source
