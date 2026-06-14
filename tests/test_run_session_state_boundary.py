"""Boundary tests: RunSession / RuntimeState are the source of truth.

These tests lock that:
- RuntimeState.from_run_session() copies status/message/length_result
- RuntimeState.sync_from_run_session() preserves length_result
- DONE export reads from _run_session.status, not auto_state_var
- auto_len event writes to RunSession.length_result
- frp_workflow/ and services/ must not call .get() on Tk variables
"""
from __future__ import annotations

import ast
from pathlib import Path

from domain.state import RunSession, RuntimeState, normalize_workflow_status


# ---------------------------------------------------------------------------
# 1. RuntimeState captures all RunSession fields
# ---------------------------------------------------------------------------

def test_normalize_workflow_status_maps_all_raw_states() -> None:
    assert normalize_workflow_status(None) == "idle"
    assert normalize_workflow_status("") == "idle"
    assert normalize_workflow_status("IDLE") == "idle"
    assert normalize_workflow_status("RUN") == "running"
    assert normalize_workflow_status("PREP") == "preparing"
    assert normalize_workflow_status("LEN") == "preparing"
    assert normalize_workflow_status("DONE") == "completed"
    assert normalize_workflow_status("ERR") == "error"
    assert normalize_workflow_status("STOP") == "idle"
    assert normalize_workflow_status("STOPPING") == "stopping"
    assert normalize_workflow_status("WARN") == "warn"


def test_from_run_session_copies_status_and_message() -> None:
    session = RunSession(
        serial="s-001",
        run_id="r-001",
        start_ts=1.0,
        end_ts=2.0,
        status="RUN",
        message="measurement started",
        length_result={"ok": True, "length_mm": 500.0},
    )
    rt = RuntimeState.from_run_session(session)

    assert rt.status == "running"   # normalized from "RUN"
    assert rt.message == "measurement started"
    assert rt.length_result == {"ok": True, "length_mm": 500.0}


def test_sync_from_run_session_does_not_clobber_length_result() -> None:
    rt = RuntimeState()
    rt.length_result = {"ok": True, "length_mm": 99.0}
    rt.status = "running"

    session = RunSession(
        serial="s-002", run_id="r-002", start_ts=3.0, end_ts=4.0,
        status="DONE", message="done", length_result={"ok": True, "length_mm": 500.0},
    )
    rt.sync_from_run_session(session)

    assert rt.status == "completed"  # normalized from "DONE"
    assert rt.message == "done"
    assert rt.length_result == {"ok": True, "length_mm": 500.0}  # synced, not clobbered


# ---------------------------------------------------------------------------
# 2. DONE export reads from RunSession.status, not Tk
# ---------------------------------------------------------------------------

def test_export_gate_reads_run_session_status_not_tk_var() -> None:
    """The export gate in _refresh_done_run_summary_and_export reads
    _run_session.status, not auto_state_var.get()."""
    app_host_path = Path(__file__).resolve().parents[1] / "application" / "app_host.py"
    tree = ast.parse(app_host_path.read_text(encoding="utf-8-sig"))

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.violations: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node.name == "_refresh_done_run_summary_and_export":
                source = ast.unparse(node)
                # Must NOT read auto_state_var.get()
                if "auto_state_var.get()" in source:
                    self.violations.append(
                        "_refresh_done_run_summary_and_export still reads auto_state_var.get()"
                    )
                # Must read _run_session.status
                if "_run_session.status" not in source:
                    self.violations.append(
                        "_refresh_done_run_summary_and_export does not read _run_session.status"
                    )
            self.generic_visit(node)

    v = _Visitor()
    v.visit(tree)
    assert not v.violations, "\n".join(v.violations)


# ---------------------------------------------------------------------------
# 3. frp_workflow/ and services/ must not call .get() on Tk variables
# ---------------------------------------------------------------------------

# Production run Tk variables that business logic must NOT read.
# Calibration config variables (odcal_*, idcal_*, etc.) are excluded
# for now — those are legitimate config inputs from the UI layer.
_PRODUCTION_RUN_TK_VARS = frozenset({
    "pipe_sn_var", "meas_seq_var", "auto_state_var", "auto_msg_var",
    "auto_done_var", "auto_progress_var",
})


def _is_tk_var_get(node: ast.AST) -> str | None:
    """Return the variable name if node is a .get() on a Tk variable, else None."""
    if not isinstance(node, ast.Call):
        return None
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == "get"):
        return None
    target = node.func.value
    # Case 1: self.some_var.get()
    if isinstance(target, ast.Attribute) and target.attr in _PRODUCTION_RUN_TK_VARS:
        return target.attr
    # Case 2: host.some_var.get() or bare some_var.get()
    if isinstance(target, ast.Name) and target.id in _PRODUCTION_RUN_TK_VARS:
        return target.id
    return None


def _scan_dir(root: Path, label: str) -> list[str]:
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            name = _is_tk_var_get(node)  # type: ignore[arg-type]
            if name is not None:
                offenders.append(f"{path.name}:{node.lineno}: {ast.unparse(node)}")  # type: ignore[arg-type]
    return offenders


def test_frp_workflow_has_no_production_tk_var_get() -> None:
    """frp_workflow/ must not read production-run Tk variables."""
    offenders = _scan_dir(
        Path(__file__).resolve().parents[1] / "frp_workflow", "frp_workflow"
    )
    assert not offenders, (
        f"frp_workflow/ has {len(offenders)} production tk-var read(s):\n"
        + "\n".join(f"  {o}" for o in offenders)
    )


def test_services_has_no_production_tk_var_get() -> None:
    """services/ must not read production-run Tk variables.

    NOTE: this does NOT yet cover calibration config variables
    (odcal_*, idcal_*, etc.).  Those are accepted as UI config inputs
    during the transitional calibration service.  The scope is limited
    to production-run state (auto_state, meas_seq, pipe_sn, auto_msg,
    auto_done, auto_progress).
    """
    root = Path(__file__).resolve().parents[1] / "services"
    if not root.exists():
        return
    offenders = _scan_dir(root, "services")
    assert not offenders, (
        f"services/ has {len(offenders)} production tk-var read(s):\n"
        + "\n".join(f"  {o}" for o in offenders)
    )
