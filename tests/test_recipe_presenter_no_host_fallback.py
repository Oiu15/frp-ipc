from __future__ import annotations

from pathlib import Path


def test_recipe_presenter_does_not_use_host_fallback() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "ui"
        / "presenters"
        / "recipe_presenter.py"
    ).read_text(encoding="utf-8-sig")

    for forbidden in (
        "self._host",
        "host_app",
        "getattr(",
        "__getattr__",
    ):
        assert forbidden not in source
