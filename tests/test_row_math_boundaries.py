from __future__ import annotations

from pathlib import Path


def test_row_math_has_no_ui_or_driver_imports() -> None:
    text = Path("frp_workflow/row_math.py").read_text(encoding="utf-8")
    forbidden = [
        "tkinter",
        "messagebox",
        "application",
        "application.app_host",
        "application.host",
        "ui.",
        "drivers.plc_client",
        "drivers.gauge_driver",
        "AppHost",
    ]

    for token in forbidden:
        assert token not in text, token
