from __future__ import annotations

"""Boundary tests: frp_workflow/steps/ files must not import tkinter/messagebox,
drivers, or application layer modules."""

from pathlib import Path


_STEPS_DIR = Path(__file__).resolve().parents[1] / "frp_workflow" / "steps"

FORBIDDEN = [
    "tkinter",
    "messagebox",
    "application.app_host",
    "application.host",
    "ui.",
    "drivers.plc_client",
    "drivers.gauge_driver",
]


def _step_files():
    return sorted(_STEPS_DIR.rglob("*.py"))


def test_workflow_steps_do_not_import_ui_or_drivers() -> None:
    offenders: list[str] = []
    for path in _step_files():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            for token in FORBIDDEN:
                if token in stripped:
                    offenders.append(f"{path.name}:{line[:120]}")
    assert offenders == [], (
        "Step files must not import UI / driver / application modules:\n"
        + "\n".join(offenders)
    )


def test_step_files_are_parseable() -> None:
    import ast

    for path in _step_files():
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            raise AssertionError(f"{path.name}: syntax error: {exc}") from exc
