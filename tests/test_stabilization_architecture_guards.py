from __future__ import annotations

from pathlib import Path

from application.adapters.device_gateway import (
    ScreenController,
    ScreenPresenter,
    ScreenUiContext,
)
from ui.presenters.gauge_presenter import GaugeScreenPresenter


ROOT = Path(__file__).resolve().parents[1]


def test_generic_and_gauge_presenters_have_no_dynamic_getattr() -> None:
    for adapter_type in (
        ScreenController,
        ScreenPresenter,
        ScreenUiContext,
        GaugeScreenPresenter,
    ):
        assert "__getattr__" not in adapter_type.__dict__, adapter_type.__name__


def test_primary_screens_do_not_use_dynamic_ui_collaborator_access() -> None:
    screen_paths = (
        "ui/screens/recipe_screen.py",
        "ui/screens/key_test_screen.py",
        "ui/screens/axis_screen.py",
        "ui/screens/axis_cal_screen.py",
        "ui/screens/validation_screen.py",
        "ui/screens/gauge_screen.py",
        "ui/screens/main_screen.py",
    )
    forbidden_tokens = (
        "getattr(controller",
        "getattr(presenter",
        "getattr(ui",
        "controller._host",
        "presenter._host",
        "ui._host",
    )

    for relative_path in screen_paths:
        source = (ROOT / relative_path).read_text(encoding="utf-8-sig")
        for token in forbidden_tokens:
            assert token not in source, f"{token} found in {relative_path}"
