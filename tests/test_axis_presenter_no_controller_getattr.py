from __future__ import annotations

from pathlib import Path


def test_axis_presenter_does_not_use_controller_getattr() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "ui"
        / "presenters"
        / "axis_presenter.py"
    ).read_text(encoding="utf-8-sig")

    for forbidden in (
        "getattr(self.controller",
        "getattr(controller",
        ".__getattr__",
    ):
        assert forbidden not in source
