# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false
"""Characterization tests for AutoFlow event handling and export flow.

These tests document the current behaviour of the AppHost handlers for
AutoFlow-emitted events (auto_row, auto_state, auto_len) and the export
trigger / run-identity lifecycle.  They use minimal host objects and
inline fakes – no real Tk, PLC, or gauge workers.
"""

import datetime
import queue
import types
from pathlib import Path
from typing import Any

from application.adapters.device_gateway import AppDeviceGateway
from application.adapters.ui_queue import WorkflowUiEventAdapter
from application.app_host import AppHost
from core.models import MeasureRow, Recipe
from domain.state import CalibrationSnapshot, RunContext, RunIdentity, RunSession, RuntimeState
from events.types import AutoLenEvent, AutoRowEvent, AutoStateEvent
from frp_workflow.autoflow_orchestrator import AutoFlowOrchestrator
from modes.calibration_mode import CalibrationMode
from modes.mode_machine import ModeMachine
from modes.production_mode import ProductionMode
from modes.validation_mode import ValidationMode
from services.measurement_service import MeasurementController
from tests.fakes import FakeVar


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _today() -> str:
    return datetime.date.today().strftime("%Y%m%d")


def _make_measure_row(idx: int = 0, x_ui: float = 100.0) -> MeasureRow:
    return MeasureRow(
        idx=idx,
        x_ui=x_ui,
        x_abs=0.0,
        od_avg=100.0,
        od_dev=0.0,
        od_runout=0.0,
        od_round=0.005,
        id_avg=60.0,
        id_dev=0.0,
        id_runout=0.0,
        id_round=0.005,
        concentricity=0.01,
        od_e=0.002,
        od_phi_deg=45.0,
        od_ecc=0.0,
        id_e=0.001,
        id_phi_deg=30.0,
        id_ecc=0.0,
    )


# ===================================================================
# Test 1 – full startup chain creates run identity
# ===================================================================

def test_measurement_start_creates_run_identity_and_starts_runner() -> None:
    """Exercise the real production startup chain end-to-end.

    Chain: MeasurementController.start_measurement()
      → ModeMachine.enter_production()
      → ProductionMode.start()
      → _start_measurement_impl()
        → _prepare_new_run()        # creates run identity
        → _make_auto_runner()       # creates AutoFlowOrchestrator wrapper
        → runner.start()            # starts the background thread

    If any link in the chain breaks, this test will catch it.
    """
    repo_calls: list[str] = []
    started: list[bool] = []

    class _FakeRepo:
        def prepare_run(self, recipe_name: str) -> RunIdentity:
            repo_calls.append(recipe_name)
            return RunIdentity(
                serial=f"{_today()}-{recipe_name}-001",
                run_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                started_at_ts=1735000000.0,
            )

    class _FakeRunner:
        def __init__(self) -> None:
            self._started = False

        def start(self) -> None:
            started.append(True)
            self._started = True

        def is_alive(self) -> bool:
            return self._started

        def stop(self) -> None:
            self._started = False

    host = object.__new__(AppHost)
    # --- wire the minimal host surface that _start_measurement_impl needs ---
    host._make_run_repository = lambda: _FakeRepo()  # type: ignore[method-assign]
    host._run_session = RunSession()
    host.pipe_sn_var = FakeVar(value="--")
    host.meas_seq_var = FakeVar(value="--")
    host.meas_start_var = FakeVar(value="--")
    host.meas_elapsed_var = FakeVar(value="--")
    host._auto_export_done = True
    host._last_run_export_path = None
    host._last_straight_od = None
    host._last_straight_id = None
    host._last_axis_dist = None
    host._run_len_result = None
    host._reset_summary_extrema = lambda: None  # type: ignore[method-assign]
    host._display_seq_text = lambda s: (  # type: ignore[method-assign]
        str(s or "").split("-")[-1] if "-" in str(s or "") else "--"
    )
    host._auto_clear_ui = lambda: None  # type: ignore[method-assign]
    host._recipe_apply_from_ui = lambda: None  # type: ignore[method-assign]
    host._refresh_auto_std_panel = lambda: None  # type: ignore[method-assign]
    host._log_ax3_speed_trace = lambda tag: None  # type: ignore[method-assign]
    host.recipe = types.SimpleNamespace(name="test-pipe")
    host._auto_thread = None
    host._make_auto_runner = lambda: _FakeRunner()  # type: ignore[method-assign]
    # wire the real _start_measurement_impl
    host._start_measurement_impl = AppHost._start_measurement_impl.__get__(host)  # type: ignore[assignment]

    # --- build the real ModeMachine and MeasurementController ---
    host.runtime_state = RuntimeState()
    host.production_mode = ProductionMode(
        start_impl=host._start_measurement_impl,
        stop_impl=lambda: None,
        runner_getter=lambda: host._auto_thread,
    )
    host.calibration_mode = CalibrationMode()
    host.validation_mode = ValidationMode(
        stop_impl=lambda: None,
        runner_getter=lambda: None,
    )
    host.mode_machine = ModeMachine(
        production_mode=host.production_mode,
        calibration_mode=host.calibration_mode,
        validation_mode=host.validation_mode,
        runtime_state=host.runtime_state,
    )
    controller = MeasurementController(mode_machine=host.mode_machine)

    # --- act ---
    result = controller.start_measurement()

    # --- assert ---
    # mode machine entered production
    assert host.runtime_state.mode_kind == "production"
    # ProductionMode.start() transitions to PREPARING; the orchestrator emits
    # "RUN" later (when measurements actually begin), which syncs to RUNNING.
    assert host.production_mode.state.value == "preparing"
    # _start_measurement_impl has no return value — result is None, and
    # MeasurementController passes it through.  Callers treat this as fire-and-forget.
    assert result is None

    # run identity created (via wired _prepare_new_run → _make_run_repository().prepare_run)
    assert len(repo_calls) == 1
    assert repo_calls[0] == "test-pipe"
    assert host._run_session.serial == f"{_today()}-test-pipe-001"
    assert host._run_session.run_id == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    # runner was created and started
    assert host._auto_thread is not None
    assert len(started) == 1


# ===================================================================
# Test 1b – _make_auto_runner wires correct dependency types
# ===================================================================

def test_make_auto_runner_wires_correct_dependency_types() -> None:
    """_make_auto_runner() creates an AutoFlowOrchestrator with the
    expected gateway, event_sink, and state objects.

    If someone changes the wiring (e.g. swaps the gateway implementation
    or drops the run_repository), this test catches the regression.
    """
    from domain.protocols import RunRepositoryProtocol

    class _FakeRepo:
        def prepare_run(self, recipe_name: str) -> RunIdentity:
            return RunIdentity(serial="s", run_id="r", started_at_ts=1.0)

        def export_run(self, ctx: RunContext) -> str:
            return ""

        def export_daily_summary(self, ctx: RunContext) -> None:
            pass

    fake_repo = _FakeRepo()
    host = object.__new__(AppHost)
    host.runtime_state = RuntimeState()
    host._run_session = RunSession()
    host._run_session.serial = "s-001"
    host._run_session.run_id = "r-001"
    host._run_session.start_ts = 1735000000.0
    host.recipe = Recipe(name="test-pipe")
    host.ui_q = queue.Queue()
    host._make_run_repository = lambda: fake_repo  # type: ignore[method-assign]
    host.get_recipe_copy = lambda: host.recipe  # type: ignore[method-assign]
    host.get_calibration_snapshot = lambda: CalibrationSnapshot()  # type: ignore[method-assign]

    runner = host._make_auto_runner()

    assert isinstance(runner, AutoFlowOrchestrator)
    assert isinstance(runner.gateway, AppDeviceGateway)
    assert isinstance(runner.event_sink, WorkflowUiEventAdapter)
    assert isinstance(runner.run_repository, RunRepositoryProtocol)
    # object identity: same instances injected, not copies
    assert runner.run_repository is fake_repo
    assert runner.run_session is host._run_session
    assert runner.runtime_state is host.runtime_state
    # runtime_state was synced from the session
    assert host.runtime_state.serial == "s-001"
    assert host.runtime_state.run_id == "r-001"
    # event_sink writes to host.ui_q
    assert host.ui_q.empty()
    runner.event_sink.publish_state("TEST", "hello")
    name, payload = host.ui_q.get_nowait()
    assert name == "auto_state"
    assert payload == {"state": "TEST", "msg": "hello"}


# ===================================================================
# Test 2 – auto_row event appends through the real _append_result_row
# ===================================================================

def test_auto_row_event_appends_to_auto_rows() -> None:
    """auto_row → _append_result_row → row lands in _auto_rows.

    This exercises the actual handler path, not a stubbed shortcut.
    The real _append_result_row needs a result_tree (via _main_ui_widget),
    and if the tree is missing it returns early without appending.
    We provide a minimal fake tree so the append path executes.
    """
    inserted: list[tuple] = []

    class _FakeTree:
        def insert(self, parent: str, index: Any, values: tuple) -> str:
            inserted.append((parent, index, values))
            return "iid-001"

    host = object.__new__(AppHost)
    # _auto_rows property delegates to _run_session.rows — supply the session
    host._run_session = RunSession()
    host._main_ui_widget = lambda name: _FakeTree() if name == "result_tree" else None  # type: ignore[method-assign]
    host._section_cov_info = {}
    host._update_summary_extrema_from_row = lambda r: None  # type: ignore[method-assign]
    host._sec_iid_map = {}
    host._result_iids = []

    row = _make_measure_row(idx=2, x_ui=150.0)
    event = AutoRowEvent(row=row)
    host._handle_auto_row_event(event)

    # row landed in _auto_rows — this is what export reads from
    assert len(host._auto_rows) == 1
    assert host._auto_rows[0].idx == 2
    assert host._auto_rows[0].x_ui == 150.0
    # also verify the tree got the insertion
    assert len(inserted) == 1


# ===================================================================
# Test 3 – auto_len event stores length result on the host
# ===================================================================

def test_auto_len_event_stores_length_on_host() -> None:
    """auto_len → _cache_auto_len_result → _run_len_result is set.

    Exercises the real handler path — no lambda stubs — and verifies
    the host attribute that downstream code depends on.
    """
    host = object.__new__(AppHost)
    host._run_session = RunSession()
    host._run_len_result = None
    # handler also touches len_meas_var and recipe — supply minimal stubs
    host.len_meas_var = FakeVar(value="--")
    host.recipe = types.SimpleNamespace(pipe_len_mm=500.0, len_tol_mm=2.0)

    payload = {
        "ok": True, "skipped": False, "reason": "",
        "z_low": 10.0, "z_high": 510.0, "length_mm": 500.0,
        "enabled": True,
    }
    event = AutoLenEvent(data=payload)
    host._handle_auto_len_event(event)

    assert isinstance(host._run_len_result, dict)
    assert host._run_len_result["length_mm"] == 500.0
    assert host._run_len_result["ok"] is True


# ===================================================================
# Test 4 – auto_state DONE triggers export
# ===================================================================

def test_auto_state_done_triggers_export() -> None:
    """auto_state DONE → _trigger_run_export called with completed=True."""
    mode_calls: list[tuple] = []
    export_calls: list[dict] = []

    class _FakeModeMachine:
        def sync_production_workflow_state(self, state: str, msg: str) -> None:
            mode_calls.append((state, msg))

    class _Repo:
        def export_run(self, ctx: RunContext) -> str:
            export_calls.append({"status": ctx.status, "completed": ctx.completed})
            return str(Path("/fake/exports/run"))

    host = object.__new__(AppHost)
    host._run_session = RunSession()
    host.mode_machine = _FakeModeMachine()
    host.auto_state_var = FakeVar(value="--")
    host.auto_msg_var = FakeVar(value="--")
    host.auto_done_var = FakeVar(value="--")
    host._refresh_stack_light_for_state = lambda s: None  # type: ignore[method-assign]
    host._auto_export_done = False

    # supply a fully-controlled _trigger_run_export that records calls
    triggered: list[dict] = []
    host._trigger_run_export = lambda **kw: (  # type: ignore[method-assign]
        triggered.append(kw) or setattr(host, "_auto_export_done", True)
    )

    event = AutoStateEvent(state="DONE", msg="测量完成")
    host._handle_auto_state_event(event)

    assert host.auto_state_var.value == "DONE"
    assert "是" in str(host.auto_done_var.value)
    assert mode_calls == [("DONE", "测量完成")]
    assert len(triggered) == 1
    assert triggered[0].get("status") == "DONE"
    assert triggered[0].get("completed") is True
    assert host._auto_export_done is True


# ===================================================================
# Test 5 – stop restores normal poll profile
# ===================================================================

def test_stop_restores_normal_poll_profile() -> None:
    poll_calls: list[str] = []
    abort_calls: list[bool] = []

    host = object.__new__(AppHost)
    host.set_plc_poll_profile = lambda profile, **kw: poll_calls.append(profile)  # type: ignore[method-assign]
    host.abort_motion = lambda axes=None: abort_calls.append(True)  # type: ignore[method-assign]

    class _FakeThread:
        def is_alive(self) -> bool:
            return True

        def stop(self) -> None:
            pass

    host._auto_thread = _FakeThread()

    host._stop_measurement_impl()

    assert "normal" in poll_calls
    assert len(abort_calls) == 1


# ===================================================================
# Test 6 – stop aborts motion
# ===================================================================

def test_stop_aborts_motion() -> None:
    abort_calls: list[list] = []

    host = object.__new__(AppHost)
    host.abort_motion = lambda axes=None: abort_calls.append(list(axes) if axes else [])  # type: ignore[method-assign]
    host.set_plc_poll_profile = lambda profile, **kw: None  # type: ignore[method-assign]

    class _FakeThread:
        def is_alive(self) -> bool:
            return True

        def stop(self) -> None:
            pass

    host._auto_thread = _FakeThread()

    host._stop_measurement_impl()

    assert len(abort_calls) == 1


# ===================================================================
# Test 7 – incomplete run context blocks export
# ===================================================================

# ===================================================================
# Test 7 – ensure-run-identity fails → export is suppressed
# ===================================================================

def test_ensure_run_identity_fails_blocks_export() -> None:
    """When _ensure_run_identity is a no-op and the session has no
    serial/run_id/start_ts, _build_run_context_for_export raises
    ValueError.  _trigger_run_export catches it, sets ok=False, and
    does NOT call export_run.  _auto_export_done stays False.

    NOTE: this tests the failure branch — the real _ensure_run_identity
    WOULD try to allocate a new identity.  To exercise that success
    path, see test_ensure_run_identity_succeeds_triggers_export below.
    """
    export_calls: list[RunContext] = []

    class _Repo:
        def export_run(self, ctx: RunContext) -> str:
            export_calls.append(ctx)
            raise AssertionError("export_run must not be called when identity is missing")

    host = object.__new__(AppHost)
    host._auto_export_done = False
    host._make_run_repository = lambda: _Repo()  # type: ignore[method-assign]
    host.auto_msg_var = FakeVar(value="--")
    host._abort_reason_for_status = lambda s: None  # type: ignore[method-assign]
    host._compact_status_path = lambda p: "exports/run"  # type: ignore[method-assign]
    host._compute_and_apply_run_summary = lambda: None  # type: ignore[method-assign]
    host._apply_run_summary_to_ui = lambda s: None  # type: ignore[method-assign]
    host._completed_section_count_for_export = lambda: 0  # type: ignore[method-assign]
    # _build_run_context_for_export calls _ensure_run_identity — stub it
    # so it does NOT try to allocate a new identity (which would need
    # pipe_sn_var / recipe / etc. and would succeed, defeating the test)
    host._ensure_run_identity = lambda: None  # type: ignore[method-assign]
    host._run_session = RunSession()
    host._run_session.serial = None
    host._run_session.run_id = None
    host._run_session.start_ts = None

    host._trigger_run_export(status="DONE", completed=True)

    assert export_calls == []
    assert host._auto_export_done is False


# ===================================================================
# Test 7b – ensure-run-identity succeeds → export triggered
# ===================================================================

def test_ensure_run_identity_succeeds_triggers_export() -> None:
    """When the session is missing identity BUT _ensure_run_identity
    (the real implementation) successfully fills it in, export proceeds
    normally — export_run is called and _auto_export_done becomes True.

    This is the success branch that complements the failure branch
    tested above.  Uses a minimal host that provides the attributes
    the real _ensure_run_identity needs: pipe_sn_var, meas_seq_var,
    recipe, _make_run_repository.
    """
    export_calls: list[RunContext] = []

    class _Repo:
        def prepare_run(self, recipe_name: str) -> RunIdentity:
            return RunIdentity(
                serial=f"{_today()}-{recipe_name}-001",
                run_id="run-ensure-ok",
                started_at_ts=1735000000.0,
            )

        def export_run(self, ctx: RunContext) -> str:
            export_calls.append(ctx)
            return str(Path("/fake/exports/run"))

    host = object.__new__(AppHost)
    # _run_session must exist before _auto_rows (property) is accessed
    host._run_session = RunSession()
    host._run_session.serial = None
    host._run_session.run_id = None
    host._run_session.start_ts = None
    host._auto_export_done = False
    host._make_run_repository = lambda: _Repo()  # type: ignore[method-assign]
    host.auto_msg_var = FakeVar(value="--")
    host.pipe_sn_var = FakeVar(value="--")
    host.meas_seq_var = FakeVar(value="--")
    host.meas_start_var = FakeVar(value="--")
    host.recipe = types.SimpleNamespace(name="ensure-test")
    host.get_recipe_copy = lambda: host.recipe  # type: ignore[method-assign]
    host.get_calibration_snapshot = lambda: CalibrationSnapshot()  # type: ignore[method-assign]
    host._auto_rows = []        # property setter → _run_session.rows
    host._auto_raw_points = []  # type: ignore[assignment]
    host._section_cov_info = {}  # type: ignore[assignment]
    host._abort_reason_for_status = lambda s: None  # type: ignore[method-assign]
    host._compact_status_path = lambda p: "exports/run"  # type: ignore[method-assign]
    host._calc_run_summary = lambda: {"ok": True}  # type: ignore[method-assign]
    host._compute_and_apply_run_summary = lambda: None  # type: ignore[method-assign]
    host._apply_run_summary_to_ui = lambda s: None  # type: ignore[method-assign]
    host._completed_section_count_for_export = lambda: 3  # type: ignore[method-assign]
    host._expected_section_count_for_export = lambda: 3  # type: ignore[method-assign]

    host._trigger_run_export(status="DONE", completed=True)

    assert len(export_calls) == 1
    assert export_calls[0].identity.serial == f"{_today()}-ensure-test-001"
    assert export_calls[0].identity.run_id == "run-ensure-ok"
    assert host._auto_export_done is True


# ===================================================================
# Test 8 – export guard prevents double export
# ===================================================================

def test_export_guard_prevents_double_export() -> None:
    export_calls: list[RunContext] = []

    class _Repo:
        def export_run(self, ctx: RunContext) -> str:
            export_calls.append(ctx)
            return str(Path("/fake/exports/run"))

    host = object.__new__(AppHost)
    host._auto_export_done = True
    host._make_run_repository = lambda: _Repo()  # type: ignore[method-assign]
    host.auto_msg_var = FakeVar(value="--")

    host._trigger_run_export(status="DONE", completed=True)

    assert export_calls == []
