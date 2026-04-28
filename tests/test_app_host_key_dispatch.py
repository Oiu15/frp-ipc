import threading
import sys
import types

_pymodbus = types.ModuleType("pymodbus")
_pymodbus_client = types.ModuleType("pymodbus.client")


class _FakeModbusTcpClient:
    pass


setattr(_pymodbus_client, "ModbusTcpClient", _FakeModbusTcpClient)
setattr(_pymodbus, "client", _pymodbus_client)
sys.modules.setdefault("pymodbus", _pymodbus)
sys.modules.setdefault("pymodbus.client", _pymodbus_client)

from application.app_host import AppHost


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


def _host() -> AppHost:
    host = object.__new__(AppHost)
    host.measurement_controller = _Controller()
    host._keytest_bits_lock = threading.Lock()
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
    host.plc_status_var = _Var()
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
