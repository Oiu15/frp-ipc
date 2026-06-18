from __future__ import annotations

"""Boundary tests: frp_workflow/steps/ files must not import tkinter/messagebox,
drivers, or application layer modules."""

from pathlib import Path


_STEPS_DIR = Path(__file__).resolve().parents[1] / "frp_workflow" / "steps"

FORBIDDEN = [
    "tkinter",
    "messagebox",
    "application",
    "application.app_host",
    "application.host",
    "ui.",
    "frp_workflow.autoflow_orchestrator",
    "drivers",
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


def test_section_execution_step_is_covered_by_boundary_scan() -> None:
    assert _STEPS_DIR.joinpath("section_execution.py") in _step_files()


def test_section_execution_context_is_covered_by_boundary_scan() -> None:
    assert _STEPS_DIR.joinpath("section_context.py") in _step_files()


def test_section_capture_step_is_covered_by_boundary_scan() -> None:
    assert _STEPS_DIR.joinpath("section_capture.py") in _step_files()


def test_measure_section_step_is_covered_by_boundary_scan() -> None:
    assert _STEPS_DIR.joinpath("measure_section.py") in _step_files()


def test_measure_section_context_is_covered_by_boundary_scan() -> None:
    assert _STEPS_DIR.joinpath("measure_section_context.py") in _step_files()


def test_sampling_result_is_covered_by_boundary_scan() -> None:
    assert _STEPS_DIR.joinpath("sampling_result.py") in _step_files()


def test_sampling_step_is_covered_by_boundary_scan() -> None:
    assert _STEPS_DIR.joinpath("sampling.py") in _step_files()


def test_rotation_control_step_is_covered_by_boundary_scan() -> None:
    assert _STEPS_DIR.joinpath("rotation_control.py") in _step_files()


def test_row_build_step_is_covered_by_boundary_scan() -> None:
    assert _STEPS_DIR.joinpath("row_build.py") in _step_files()


def test_row_build_result_is_covered_by_boundary_scan() -> None:
    assert _STEPS_DIR.joinpath("row_build_result.py") in _step_files()


def test_publish_events_step_is_covered_by_boundary_scan() -> None:
    assert _STEPS_DIR.joinpath("publish_events.py") in _step_files()


def test_publish_events_context_is_covered_by_boundary_scan() -> None:
    assert _STEPS_DIR.joinpath("publish_events_context.py") in _step_files()


def test_record_row_step_is_covered_by_boundary_scan() -> None:
    assert _STEPS_DIR.joinpath("record_row.py") in _step_files()


def test_step_files_are_parseable() -> None:
    import ast

    for path in _step_files():
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            raise AssertionError(f"{path.name}: syntax error: {exc}") from exc
