# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false
import threading
import sys
import types
import queue
import inspect
from pathlib import Path

_pymodbus = types.ModuleType("pymodbus")
_pymodbus_client = types.ModuleType("pymodbus.client")


class _FakeModbusTcpClient:
    pass


setattr(_pymodbus_client, "ModbusTcpClient", _FakeModbusTcpClient)
setattr(_pymodbus, "client", _pymodbus_client)
sys.modules.setdefault("pymodbus", _pymodbus)
sys.modules.setdefault("pymodbus.client", _pymodbus_client)

from application.app_host import AppHost
from application.state import RunSession


class _Controller:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    def start_measurement(self) -> None:
        self.started += 1

    def stop_measurement(self) -> None:
        self.stopped += 1


class _Var:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


def _host() -> AppHost:
    host = object.__new__(AppHost)
    host.measurement_controller = _Controller()
    host._run_session = RunSession()
    host._keytest_bits_lock = threading.Lock()
    host._keytest_x_points_state = [1] * 16
    host._keytest_y_points_state = [0] * 16
    host._keytest_y_points_has_read = True
    host._keytest_y_last_command_state = [0] * 16
    host._flow_confirm_lock = threading.Lock()
    host._flow_confirm_token = None
    host._flow_confirm_evt = None
    host._flow_confirm_result = None
    host._flow_confirm_popup = None
    host._flow_confirm_confirm_cb = None
    host._flow_confirm_cancel_cb = None
    host._op_confirm_lock = threading.Lock()
    host._op_confirm_token = None
    host._op_confirm_evt = None
    host._op_confirm_result = None
    host._op_confirm_popup = None
    host._stack_light_state = None
    host._stack_light_buzzer_after_id = None
    host.after = lambda ms, cb: "after-id"  # type: ignore[method-assign]
    host.after_cancel = lambda after_id: None  # type: ignore[method-assign]
    host.plc_status_var = _Var()
    host.pipe_sn_var = _Var()
    host.meas_seq_var = _Var()
    host.auto_msg_var = _Var()
    host.ui_q = queue.Queue()
    host.cmd_q = []
    host.plc_write_y_point = lambda y, v: host.cmd_q.append((int(y), int(v)))  # type: ignore[method-assign]
    host._is_auto_thread_alive = lambda: False  # type: ignore[method-assign]
    return host


def test_x2_starts_directly_on_main_page() -> None:
    host = _host()
    host._is_main_tab_selected = lambda: True  # type: ignore[method-assign]

    host._handle_x2_edge()

    assert host.measurement_controller.started == 1


def test_x2_non_main_requires_confirm_before_start() -> None:
    host = _host()
    host._is_main_tab_selected = lambda: False  # type: ignore[method-assign]
    captured = {}

    def show(**kwargs):
        captured.update(kwargs)

    host._show_flow_confirm_popup = show  # type: ignore[method-assign]

    host._handle_x2_edge()
    assert host.measurement_controller.started == 0

    captured["on_confirm"]()
    assert host.measurement_controller.started == 1


def test_x4_stops_running_flow_before_unclamp() -> None:
    host = _host()
    host._is_auto_thread_alive = lambda: True  # type: ignore[method-assign]

    host._handle_x4_edge()

    assert host.measurement_controller.stopped == 1
    assert host.cmd_q == []


def test_x4_idle_unclamp_skips_duplicate_release() -> None:
    host = _host()
    host._keytest_y_points_state[8] = 0
    host._keytest_y_points_state[9] = 0

    host._handle_x4_edge()

    assert host.cmd_q == []
    assert host.plc_status_var.value == "夹爪已松开"


def test_flow_confirm_worker_roundtrip_uses_ui_queue_and_x3() -> None:
    host = _host()
    result = []

    def run_confirm() -> None:
        result.append(host.flow_confirm("t", "m"))

    t = threading.Thread(target=run_confirm)
    t.start()
    event_name, payload = host.ui_q.get(timeout=1)

    assert event_name == "flow_confirm_show"
    assert payload["title"] == "t"

    assert host._flow_confirm_set("confirm", token=payload["token"])
    t.join(timeout=1)
    assert result == ["confirm"]


def test_operator_confirm_maps_flow_cancel_to_stop() -> None:
    host = _host()
    host.flow_confirm = lambda *args, **kwargs: "cancel"  # type: ignore[method-assign]

    assert host.operator_confirm("t", "m") == "stop"


def test_stack_light_running_writes_green_once() -> None:
    host = _host()

    host.set_stack_light("RUNNING")
    host.set_stack_light("RUNNING")

    assert host.cmd_q == [(4, 0), (5, 0), (6, 0), (6, 1)]


def test_stack_light_red_triggers_buzzer_once_until_state_changes() -> None:
    host = _host()

    host.set_stack_light("ERROR_OR_ESTOP")
    host.set_stack_light("ERROR_OR_ESTOP")

    assert host.cmd_q == [(4, 0), (5, 0), (6, 0), (4, 1), (7, 1)]


def test_stack_light_estop_overrides_running_state() -> None:
    host = _host()
    host._keytest_x_points_state[0] = 0

    host._refresh_stack_light_for_state("RUN")

    assert host.cmd_q[-2:] == [(4, 1), (7, 1)]


def test_refresh_measurement_display_preserves_current_run_identity() -> None:
    host = _host()
    calls = []
    host._run_serial = "20260428-demo-007"
    host._auto_clear_ui = lambda preserve_run=False: calls.append(preserve_run)  # type: ignore[method-assign]

    host.clear_measurement_results()

    assert calls == [True]
    assert host.pipe_sn_var.value == "20260428-demo-007"
    assert host.meas_seq_var.value == "007"


def test_partial_export_keeps_error_message_visible() -> None:
    host = _host()
    host._auto_export_done = False
    host.auto_msg_var.set("AX3 fault")
    host._build_run_context_for_export = lambda **kwargs: object()  # type: ignore[method-assign]
    host._compact_status_path = lambda path: "exports/run"  # type: ignore[method-assign]

    class _Repo:
        def export_run(self, ctx):
            return Path("exports/run")

    host._make_run_repository = lambda: _Repo()  # type: ignore[method-assign]
    host._compute_and_apply_run_summary = lambda: None  # type: ignore[method-assign]

    host._trigger_run_export(status="ERR", completed=False)

    assert host.auto_msg_var.value == "AX3 fault | 导出完成: exports/run"


def test_export_history_empty_does_not_allocate_run_identity(monkeypatch) -> None:
    host = _host()
    calls = []

    class _Service:
        def list_exportable_entries(self):
            return []

    host._make_history_export_service = lambda: _Service()  # type: ignore[method-assign]
    host._ensure_run_identity = lambda: calls.append("ensure")  # type: ignore[method-assign]
    monkeypatch.setattr("application.app_host.messagebox.showinfo", lambda *args, **kwargs: calls.append("showinfo"))
    monkeypatch.setattr(
        "application.app_host.filedialog.asksaveasfilename",
        lambda *args, **kwargs: calls.append("save_dialog"),
    )

    host.export_history_results()

    assert calls == ["showinfo"]


def test_history_export_dialog_uses_checkbox_state_by_date() -> None:
    source = inspect.getsource(AppHost._show_history_export_dialog)

    assert "selected_iids" in source
    assert "date_children" in source
    assert "_set_parent_checked" in source
    assert "selectmode=\"none\"" in source
    assert "\"[x]\"" in source
