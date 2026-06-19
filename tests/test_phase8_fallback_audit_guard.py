from __future__ import annotations

from pathlib import Path


def test_recipe_screen_does_not_add_dynamic_fallback_access() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "ui"
        / "screens"
        / "recipe_screen.py"
    ).read_text(encoding="utf-8-sig")

    for forbidden in (
        "getattr(controller",
        "getattr(presenter",
        "getattr(ui",
        "controller._host",
        "presenter._host",
        "ui._host",
    ):
        assert forbidden not in source


def test_key_test_screen_does_not_add_dynamic_fallback_access() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "ui"
        / "screens"
        / "key_test_screen.py"
    ).read_text(encoding="utf-8-sig")

    for forbidden in (
        "getattr(controller",
        "getattr(ui",
        "controller._host",
        "ui._host",
    ):
        assert forbidden not in source


def test_key_test_screen_is_wired_with_explicit_objects() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "application"
        / "host"
        / "ui.py"
    ).read_text(encoding="utf-8-sig")

    assert "presenter=self.key_test_ui" in source
    assert "controller=self.key_test_controller" in source
    assert "build_key_test_screen(tab_keytest, presenter=self._screen_presenter" not in source
    assert "build_key_test_screen(tab_keytest, presenter=self.key_test_ui, controller=self._screen_controller" not in source
