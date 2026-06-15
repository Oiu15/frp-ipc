"""Architecture enforcement: frp_workflow/ must NOT write to ui_q directly.

All workflow-to-UI communication must go through the EventSink protocol,
implemented by WorkflowUiEventAdapter.  The legacy tuple payload shape
(("auto_*", {payload})) is preserved so the existing UiEventDispatcher
continues to consume events without changes.
"""
from __future__ import annotations

import queue
from pathlib import Path
from typing import Any

from application.adapters.ui_queue import WorkflowUiEventAdapter
from events.dispatcher import UiEventDispatcher
from events.types import AutoStateEvent, AutoRowEvent, AutoLenEvent
from core.models import MeasureRow


# ---------------------------------------------------------------------------
# 1. frp_workflow/ must not call ui_q.put directly
# ---------------------------------------------------------------------------

def test_frp_workflow_has_no_direct_ui_q_put() -> None:
    """frp_workflow/ must NOT contain any self.app.ui_q.put(...) call.

    All UI events must go through EventSink.  Fallback branches have been
    removed — the executor requires event_sink at construction time.
    """
    root = Path(__file__).resolve().parents[1] / "frp_workflow"
    offenders: list[str] = []

    for path in sorted(root.rglob("*.py")):
        for i, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            stripped = raw.strip()
            if "ui_q.put" in stripped and not stripped.startswith("#"):
                offenders.append(f"{path.name}:{i}: {stripped[:100]}")

    assert offenders == [], (
        f"frp_workflow/ has {len(offenders)} direct ui_q.put call(s):\n"
        + "\n".join(f"  {o}" for o in offenders)
    )


# ---------------------------------------------------------------------------
# 1b. Orchestrator must consume public executor results, not private side effects
# ---------------------------------------------------------------------------

def test_autoflow_orchestrator_does_not_read_legacy_private_state() -> None:
    """AutoFlowOrchestrator must not couple to legacy executor internals."""
    path = Path(__file__).resolve().parents[1] / "frp_workflow" / "autoflow_orchestrator.py"
    source = path.read_text(encoding="utf-8-sig")

    forbidden = [
        "legacy._",
        "getattr(legacy,",
        "setattr(legacy,",
        "self._legacy_flow._",
        "_last_sample",
        "_last_fit_weights",
        "_sample_circle_points_dual",
    ]
    offenders = [token for token in forbidden if token in source]

    assert offenders == [], (
        "AutoFlowOrchestrator must use public executor result/method APIs; "
        f"found forbidden token(s): {offenders}"
    )


# ---------------------------------------------------------------------------
# 2. WorkflowUiEventAdapter produces tuples consumable by UiEventDispatcher
# ---------------------------------------------------------------------------

def _make_adapter() -> WorkflowUiEventAdapter:
    return WorkflowUiEventAdapter(queue.Queue())


def test_publish_auto_state_produces_consumable_tuple() -> None:
    """auto_state published via EventSink can be dispatched."""
    adapter = _make_adapter()
    dispatcher = UiEventDispatcher()
    dispatched: list[tuple[str, Any]] = []
    dispatcher.register(AutoStateEvent, lambda e: dispatched.append(("typed", e)))

    adapter.publish_auto_state("DONE", "msg")
    name, payload = adapter.ui_q.get_nowait()
    dispatcher.dispatch(name, payload)

    assert name == "auto_state"
    assert payload == {"state": "DONE", "msg": "msg"}
    assert len(dispatched) == 1


def test_publish_auto_row_produces_consumable_tuple() -> None:
    """auto_row published via EventSink can be dispatched."""
    adapter = _make_adapter()
    dispatcher = UiEventDispatcher()
    row = MeasureRow(
        idx=0, x_ui=100.0, x_abs=0.0,
        od_avg=100.0, od_dev=0.0, od_runout=0.0, od_round=0.005,
        id_avg=60.0, id_dev=0.0, id_runout=0.0, id_round=0.005,
        concentricity=0.01,
    )
    dispatched: list[tuple[str, Any]] = []
    dispatcher.register(AutoRowEvent, lambda e: dispatched.append(("typed", e)))

    adapter.publish_auto_row(row)
    name, payload = adapter.ui_q.get_nowait()
    dispatcher.dispatch(name, payload)

    assert name == "auto_row"
    assert payload["row"] is row
    assert len(dispatched) == 1


def test_publish_auto_len_produces_consumable_tuple() -> None:
    """auto_len published via EventSink can be dispatched."""
    adapter = _make_adapter()
    dispatcher = UiEventDispatcher()
    dispatched: list[tuple[str, Any]] = []
    dispatcher.register(AutoLenEvent, lambda e: dispatched.append(("typed", e)))

    payload = {"ok": True, "length_mm": 500.0}
    adapter.publish_auto_len(payload)
    name, p = adapter.ui_q.get_nowait()
    dispatcher.dispatch(name, p)

    assert name == "auto_len"
    assert p == payload
    assert len(dispatched) == 1


# ---------------------------------------------------------------------------
# 3. Tuple payload shape is preserved (not changed from legacy format)
# ---------------------------------------------------------------------------

def test_auto_progress_preserves_legacy_payload_shape() -> None:
    """auto_progress payload keys match legacy format."""
    adapter = _make_adapter()
    adapter.publish_auto_progress(section_index=3, section_total=5, z_pos_mm=150.0, ax0_abs=200.0)
    name, payload = adapter.ui_q.get_nowait()
    assert name == "auto_progress"
    assert payload == {"idx": 2, "total": 5, "x_ui": 150.0, "x_abs": 200.0}


def test_auto_clear_produces_correct_payload_shape() -> None:
    """auto_clear payload is {ts: None} matching legacy."""
    adapter = _make_adapter()
    adapter.publish_auto_clear()
    name, payload = adapter.ui_q.get_nowait()
    assert name == "auto_clear"
    assert payload == {"ts": None}


def test_auto_raw_points_preserves_legacy_payload_shape() -> None:
    """auto_raw_points wraps points in {points: [...]} matching legacy."""
    adapter = _make_adapter()
    points = [{"x": 1.0}, {"x": 2.0}]
    adapter.publish_auto_raw_points(points)
    name, payload = adapter.ui_q.get_nowait()
    assert name == "auto_raw_points"
    assert payload == {"points": points}


def test_auto_done_produces_correct_payload() -> None:
    """publish_auto_done emits auto_state DONE."""
    adapter = _make_adapter()
    adapter.publish_auto_done("done")
    name, payload = adapter.ui_q.get_nowait()
    assert name == "auto_state"
    assert payload == {"state": "DONE", "msg": "done"}


def test_auto_error_produces_correct_payload() -> None:
    """publish_auto_error emits auto_state ERR."""
    adapter = _make_adapter()
    adapter.publish_auto_error("something broke")
    name, payload = adapter.ui_q.get_nowait()
    assert name == "auto_state"
    assert payload == {"state": "ERR", "msg": "something broke"}
