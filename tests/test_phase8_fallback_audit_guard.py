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


def test_axis_cal_screen_does_not_add_dynamic_fallback_access() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "ui"
        / "screens"
        / "axis_cal_screen.py"
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


def test_axis_cal_screen_is_wired_with_explicit_objects() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "application"
        / "host"
        / "ui.py"
    ).read_text(encoding="utf-8-sig")

    assert "presenter=self.axis_cal_ui" in source
    assert "controller=self.axis_cal_controller" in source
    assert "build_axis_cal_screen(tab_axis_cal, presenter=self._screen_presenter" not in source
    assert "build_axis_cal_screen(tab_axis_cal, presenter=self.axis_cal_ui, controller=self._screen_controller" not in source


def test_validation_screen_does_not_add_dynamic_fallback_access() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "ui"
        / "screens"
        / "validation_screen.py"
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


def test_validation_screen_is_wired_with_explicit_objects() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "application"
        / "host"
        / "ui.py"
    ).read_text(encoding="utf-8-sig")

    assert "presenter=self.validation_ui" in source
    assert "controller=self.validation_controller" in source
    assert "build_validation_screen(tab_validation, presenter=self._gauge_screen_presenter" not in source
    assert "build_validation_screen(tab_validation, presenter=self.validation_ui, controller=self._screen_controller" not in source


def test_gauge_screen_does_not_add_dynamic_fallback_access() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "ui"
        / "screens"
        / "gauge_screen.py"
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


def test_gauge_screen_is_wired_with_explicit_objects() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "application"
        / "host"
        / "ui.py"
    ).read_text(encoding="utf-8-sig")

    assert "controller=self.gauge_controller" in source
    assert "ui=self.gauge_ui" in source
    assert "build_gauge_screen(tab_gauge, presenter=self._gauge_screen_presenter, controller=self._screen_controller" not in source
    assert "build_gauge_screen(tab_gauge, presenter=self._gauge_screen_presenter, controller=self.gauge_controller, ui=self._screen_ui_context" not in source


def test_main_screen_does_not_add_dynamic_fallback_access() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "ui"
        / "screens"
        / "main_screen.py"
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


def test_main_screen_is_wired_with_explicit_objects() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "application"
        / "host"
        / "ui.py"
    ).read_text(encoding="utf-8-sig")

    assert "presenter=self.main_ui" in source
    assert "controller=self.main_controller" in source
    assert "build_main_screen(tab_main, presenter=self._screen_presenter" not in source
    assert "build_main_screen(tab_main, presenter=self.main_ui, controller=self._screen_controller" not in source
