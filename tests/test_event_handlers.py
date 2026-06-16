from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from application.app_host import AppHost
from application.handlers.actions import (
    AxisViewActions,
    DeviceStateActions,
    ExportActions,
    RunStateActions,
    RunViewActions,
    WorkflowStatusActions,
)
from application.handlers.device import GaugeErrEventHandler, PlcErrEventHandler, PlcOkEventHandler
from application.handlers.measurement import (
    AutoCoverageEventHandler,
    AutoLenEventHandler,
    AutoPostcalcEventHandler,
    AutoProgressEventHandler,
    AutoRowEventHandler,
    AutoStateEventHandler,
)
from core.models import AxisComm, MeasureRow
from domain.state import RunSession
from events.types import (
    AutoCoverageEvent,
    AutoLenEvent,
    AutoPostcalcEvent,
    AutoProgressEvent,
    AutoRowEvent,
    AutoStateEvent,
    GaugeErrEvent,
    PlcErrEvent,
    PlcOkEvent,
)
from services.run_export_coordinator import ExportKind, ExportResult, ExportStatus


class _RunView:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def set_auto_progress(self, idx: int, total: int) -> None:
        self.calls.append(("progress", idx, total))

    def set_auto_done(self, completed: bool) -> None:
        self.calls.append(("done", completed))

    def project_auto_len_result(self, payload: dict[str, Any]) -> None:
        self.calls.append(("len", dict(payload)))

    def show_section_coverage(self, info: dict[str, Any]) -> None:
        self.calls.append(("coverage", dict(info)))

    def set_auto_state(self, state: str, message: str) -> None:
        self.calls.append(("state", state, message))

    def refresh_done_run_summary(self) -> None:
        self.calls.append(("summary",))


class _RunState:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.show_coverage = True

    def set_current_section_index(self, section_index: int) -> None:
        self.calls.append(("section", section_index))

    def cache_auto_len_result(self, payload: Any) -> dict[str, Any]:
        self.calls.append(("cache_len", dict(payload)))
        return dict(payload)

    def cache_section_coverage(self, payload: Any) -> tuple[int | None, dict[str, Any]]:
        info = dict(payload)
        self.calls.append(("cache_cov", info))
        return info.get("idx"), info

    def should_show_section_coverage(self, section_index: int | None) -> bool:
        self.calls.append(("should_cov", section_index))
        return self.show_coverage

    def update_run_status(self, state: str, message: str) -> None:
        self.calls.append(("run_status", state, message))

    def freeze_run_end_ts_if_missing(self) -> None:
        self.calls.append(("freeze",))

    def append_result_row(self, row: MeasureRow) -> None:
        self.calls.append(("row", row))

    def apply_postcalc_result(self, payload: Any) -> None:
        self.calls.append(("postcalc", dict(payload)))


class _WorkflowStatus:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def sync_production_workflow_state(self, state: str, message: str) -> None:
        self.calls.append(("sync", state, message))

    def refresh_stack_light_for_state(self, state: str | None = None) -> None:
        self.calls.append(("stack", state))


class _Export:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def trigger_terminal_export(self, status: str, completed: bool) -> None:
        self.calls.append(("export", status, completed))

    def maybe_retry_terminal_export(self) -> None:
        self.calls.append(("retry",))


class _DeviceState:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def set_plc_ok_status(self) -> None:
        self.calls.append(("plc_ok",))

    def set_plc_error_status(
        self, err: str, retry: int | None, max_attempts: int | None, backoff_s: float | None
    ) -> None:
        self.calls.append(("plc_err", err, retry, max_attempts, backoff_s))

    def update_axis_snapshot_from_plc(self, event: PlcOkEvent) -> None:
        self.calls.append(("axis_snapshot", event))

    def update_cl_cache_and_ui(self, event: PlcOkEvent) -> None:
        self.calls.append(("cl", event))

    def set_gauge_error(self, message: str) -> None:
        self.calls.append(("gauge_err", message))


class _AxisView:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def update_keytest_from_plc(self, event: PlcOkEvent) -> None:
        self.calls.append(("keytest", event))

    def refresh_axis_panel_from_snapshot(self) -> None:
        self.calls.append(("axis_panel",))

    def handle_axis_cal_one_shot_read(self) -> None:
        self.calls.append(("axis_cal_read",))

    def refresh_axis_cal_status(self) -> None:
        self.calls.append(("axis_cal_status",))


def _row() -> MeasureRow:
    return MeasureRow(
        idx=1,
        x_ui=10.0,
        x_abs=20.0,
        od_avg=100.0,
        od_dev=0.0,
        od_runout=0.0,
        od_round=0.0,
        id_avg=50.0,
        id_dev=0.0,
        id_runout=0.0,
        id_round=0.0,
        concentricity=0.0,
    )


def test_simple_measurement_handlers_delegate_to_small_actions() -> None:
    run_state = _RunState()
    run_view = _RunView()

    AutoProgressEventHandler(run_state, run_view).handle(AutoProgressEvent(idx=2, total=5, x_ui=1.0, x_abs=2.0))
    AutoCoverageEventHandler(run_state, run_view).handle(AutoCoverageEvent(idx=3, cov=0.8, miss=1))
    AutoLenEventHandler(run_state, run_view).handle(AutoLenEvent({"ok": True, "length_mm": 12.0}))

    assert ("section", 3) in run_state.calls
    assert ("progress", 2, 5) in run_view.calls
    assert ("done", False) in run_view.calls
    assert any(call[0] == "cache_cov" for call in run_state.calls)
    assert any(call[0] == "coverage" for call in run_view.calls)
    assert any(call[0] == "cache_len" for call in run_state.calls)
    assert any(call[0] == "len" for call in run_view.calls)


def test_auto_state_handler_updates_run_before_export() -> None:
    calls: list[tuple[Any, ...]] = []

    class _OrderedRunState(_RunState):
        def update_run_status(self, state: str, message: str) -> None:
            calls.append(("run_status", state, message))

    class _OrderedRunView(_RunView):
        def set_auto_state(self, state: str, message: str) -> None:
            calls.append(("ui_state", state, message))

        def set_auto_done(self, completed: bool) -> None:
            calls.append(("ui_done", completed))

    class _OrderedWorkflow(_WorkflowStatus):
        def sync_production_workflow_state(self, state: str, message: str) -> None:
            calls.append(("mode", state, message))

        def refresh_stack_light_for_state(self, state: str | None = None) -> None:
            calls.append(("stack", state))

    class _OrderedExport(_Export):
        def trigger_terminal_export(self, status: str, completed: bool) -> None:
            calls.append(("export", status, completed))

    AutoStateEventHandler(
        _OrderedRunState(),
        _OrderedRunView(),
        _OrderedWorkflow(),
        _OrderedExport(),
    ).handle(AutoStateEvent(state="DONE", msg="ok"))

    assert calls == [
        ("run_status", "DONE", "ok"),
        ("mode", "DONE", "ok"),
        ("ui_state", "DONE", "ok"),
        ("stack", "DONE"),
        ("ui_done", True),
        ("export", "DONE", True),
    ]


def test_auto_row_and_postcalc_handlers_delegate_to_actions() -> None:
    run_state = _RunState()
    run_view = _RunView()
    export = _Export()
    row = _row()

    AutoRowEventHandler(run_state).handle(AutoRowEvent(row))
    AutoPostcalcEventHandler(run_state, run_view, export).handle(AutoPostcalcEvent(straight_od=1.0))

    assert ("row", row) in run_state.calls
    assert any(call[0] == "postcalc" for call in run_state.calls)
    assert ("retry",) in export.calls
    assert ("summary",) in run_view.calls


def test_device_handlers_delegate_to_small_actions_in_order() -> None:
    device = _DeviceState()
    axis = _AxisView()
    workflow = _WorkflowStatus()
    event = PlcOkEvent(axes=[AxisComm()])

    GaugeErrEventHandler(device).handle(GaugeErrEvent(ts=1.0, err="bad"))
    PlcErrEventHandler(device).handle(PlcErrEvent(err="down", retry=2, max=5, backoff_s=1.5))
    PlcOkEventHandler(device, axis, workflow).handle(event)

    assert ("gauge_err", "Gauge ERROR: bad") in device.calls
    assert ("plc_err", "down", 2, 5, 1.5) in device.calls
    assert [call[0] for call in device.calls[-3:]] == ["plc_ok", "axis_snapshot", "cl"]
    assert [call[0] for call in axis.calls] == ["keytest", "axis_cal_read", "axis_panel", "axis_cal_status"]
    assert workflow.calls == [("stack", None)]


def test_app_host_export_retry_only_for_terminal_pending_or_failed() -> None:
    host = object.__new__(AppHost)
    host._run_session = RunSession(status="DONE")
    host._last_run_export_result = ExportResult(
        status=ExportStatus.PENDING,
        kind=ExportKind.COMPLETED,
        completed=False,
        message="pending",
    )
    calls: list[str] = []
    host._maybe_trigger_completed_export = lambda: calls.append("retry")  # type: ignore[method-assign]

    AppHost._maybe_retry_terminal_export(host)
    host._last_run_export_result = ExportResult(
        status=ExportStatus.EXPORTED,
        kind=ExportKind.COMPLETED,
        completed=True,
        message="exported",
    )
    AppHost._maybe_retry_terminal_export(host)
    host._run_session.status = "RUNNING"
    host._last_run_export_result = ExportResult(
        status=ExportStatus.FAILED,
        kind=ExportKind.COMPLETED,
        completed=False,
        message="failed",
    )
    AppHost._maybe_retry_terminal_export(host)

    assert calls == ["retry"]


def test_handler_constructors_and_action_protocols_stay_narrow() -> None:
    handler_classes = [
        AutoCoverageEventHandler,
        AutoLenEventHandler,
        AutoPostcalcEventHandler,
        AutoProgressEventHandler,
        AutoRowEventHandler,
        AutoStateEventHandler,
        GaugeErrEventHandler,
        PlcErrEventHandler,
        PlcOkEventHandler,
    ]
    for handler_cls in handler_classes:
        params = list(inspect.signature(handler_cls).parameters)
        assert "app" not in params
        assert "host" not in params
        assert len(params) <= 4
        handler = handler_cls.__new__(handler_cls)
        assert not hasattr(handler, "__dict__")

    forbidden_names = {"handle", "process", "update_all", "sync_all", "apply_event", "refresh_all"}
    protocols = [
        AxisViewActions,
        DeviceStateActions,
        ExportActions,
        RunStateActions,
        RunViewActions,
        WorkflowStatusActions,
    ]
    for protocol in protocols:
        methods = [
            name
            for name, value in protocol.__dict__.items()
            if callable(value) and not name.startswith("_")
        ]
        assert len(methods) <= 8
        assert not (set(methods) & forbidden_names)


def test_handlers_do_not_import_app_host_or_tk() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in (root / "application" / "handlers").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "application.app_host" not in text
        assert "tkinter" not in text
