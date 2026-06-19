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
