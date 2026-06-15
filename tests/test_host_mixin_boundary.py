"""Architecture boundary tests: application/host/ mixins.

These tests document the CURRENT state of host mixin dependencies.
Tests are marked xfail where violations exist — the goal is to
gradually eliminate each category.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


@pytest.mark.xfail(reason="validation.py still imports frp_workflow.validation_workflow")
def test_host_mixins_do_not_import_frp_workflow() -> None:
    root = Path(__file__).resolve().parents[1] / "application" / "host"
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("frp_workflow"):
                    offenders.append(f"{path.name}:{node.lineno}: {node.module}")
    assert offenders == [], (
        f"application/host/ must not import frp_workflow:\n"
        + "\n".join(f"  {o}" for o in offenders)
    )


@pytest.mark.xfail(reason="gauge_connection.py has lazy import of list_serial_ports (pragmatic)")
def test_host_mixins_do_not_import_drivers() -> None:
    root = Path(__file__).resolve().parents[1] / "application" / "host"
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("drivers"):
                    offenders.append(f"{path.name}:{node.lineno}: {node.module}")
    assert offenders == [], (
        f"application/host/ must not import drivers:\n"
        + "\n".join(f"  {o}" for o in offenders)
    )


@pytest.mark.xfail(reason="confirm.py, export.py, keytest.py import services")
def test_host_mixins_do_not_import_services() -> None:
    root = Path(__file__).resolve().parents[1] / "application" / "host"
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("services"):
                    offenders.append(f"{path.name}:{node.lineno}: {node.module}")
    assert offenders == [], (
        f"application/host/ must not import services:\n"
        + "\n".join(f"  {o}" for o in offenders)
    )
