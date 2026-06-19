from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8-sig")


def test_migrated_screens_do_not_use_dynamic_fallback_tokens() -> None:
    for path in (
        "ui/screens/recipe_screen.py",
        "ui/screens/key_test_screen.py",
        "ui/screens/axis_cal_screen.py",
        "ui/screens/validation_screen.py",
        "ui/screens/gauge_screen.py",
        "ui/screens/main_screen.py",
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
            assert forbidden not in source, f"{forbidden} found in {path}"


def test_generic_fallback_classes_still_exist_as_legacy_inventory() -> None:
    source = _read("application/adapters/device_gateway.py")

    assert "class ScreenController" in source
    assert "class ScreenPresenter" in source
    assert "class ScreenUiContext" in source
    assert "def __getattr__(self, name: str) -> Any:" in source
    assert "legacy dynamic proxy" in source
    assert "legacy screens during migration" in source


def test_generic_fallback_is_not_used_by_explicit_screen_wiring() -> None:
    source = _read("application/host/ui.py")

    assert "build_main_screen(tab_main, presenter=self.main_ui, controller=self.main_controller" in source
    assert "build_key_test_screen(tab_keytest, presenter=self.key_test_ui, controller=self.key_test_controller" in source
    assert "build_axis_cal_screen(tab_axis_cal, presenter=self.axis_cal_ui, controller=self.axis_cal_controller" in source
    assert "build_validation_screen(tab_validation, presenter=self.validation_ui, controller=self.validation_controller" in source
    assert "build_gauge_screen(tab_gauge, presenter=self._gauge_screen_presenter, controller=self.gauge_controller" in source

    forbidden_wiring = (
        "build_main_screen(tab_main, presenter=self._screen_presenter",
        "build_key_test_screen(tab_keytest, presenter=self._screen_presenter",
        "build_axis_cal_screen(tab_axis_cal, presenter=self._screen_presenter",
        "build_validation_screen(tab_validation, presenter=self._screen_presenter",
        "build_gauge_screen(tab_gauge, presenter=self._screen_presenter, controller=self._screen_controller",
    )
    for forbidden in forbidden_wiring:
        assert forbidden not in source


def test_axis_screen_no_longer_uses_generic_controller_for_action_dispatch() -> None:
    ui_source = _read("application/host/ui.py")
    presenter_source = _read("ui/presenters/axis_presenter.py")

    assert "build_axis_screen(tab_axis, presenter=self._axis_screen_presenter, controller=self.axis_controller" in ui_source
    assert "getattr(self.controller" not in presenter_source
    assert "dispatch_axis_action(action_name)" in presenter_source
