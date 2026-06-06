from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

import pytest

from tests.fakes import FakeCombo, FakeVar

from application.host.calibration.gauge_connection import HostGaugeConnectionMixin
from config.addresses import DEFAULT_GAUGE_PORT


class _FakeGaugeWorker:
    def __init__(self, *, fail_configure: bool = False) -> None:
        self.fail_configure = fail_configure
        self.configures: list[dict] = []
        self.request_cmd = ""
        self.send_count = 0

    def configure(self, **kwargs) -> None:
        self.configures.append(dict(kwargs))
        if self.fail_configure and bool(kwargs.get("enabled")):
            raise RuntimeError("serial failed")
        self.request_cmd = str(kwargs.get("request_cmd", self.request_cmd))

    def send_request(self) -> None:
        self.send_count += 1


class _FakeGaugeHost(HostGaugeConnectionMixin):
    gauge_worker: Any
    baud_var: Any
    req_cmd_var: Any
    gauge_conn_var: Any
    gauge_err_var: Any
    sim_gauge_var: Any
    sim_disp_var: Any

    def __init__(self, *, worker=None, port_combo: FakeCombo | None = None) -> None:
        self.gauge_worker = worker if worker is not None else _FakeGaugeWorker()
        self.baud_var = FakeVar("115200")
        self.req_cmd_var = FakeVar("M1,1")
        self.gauge_conn_var = FakeVar("")
        self.gauge_err_var = FakeVar("")
        self.sim_gauge_var = FakeVar(1)
        self.sim_disp_var = FakeVar(0)
        self.sim_gauge_enabled = True
        self.sim_disp_enabled = False
        self.port_combo = port_combo if port_combo is not None else FakeCombo("")

    def _gauge_ui_widget(self, name: str):
        if name == "port_combo":
            return self.port_combo
        return None


_PORT_SELECTION_CASES = [
    # (initial_combo, serial_ports, expected)
    ("COM9", ["COM2", "COM9"],             "COM9"),
    ("",     [DEFAULT_GAUGE_PORT, "COM9"], DEFAULT_GAUGE_PORT),
    ("",     ["COM8"],                     "COM8"),
]


@pytest.mark.parametrize(("initial", "ports", "expected"), _PORT_SELECTION_CASES)
def test_refresh_ports_selection(initial: str, ports: list[str], expected: str) -> None:
    host = _FakeGaugeHost(port_combo=FakeCombo(initial))
    with patch("application.host.calibration.gauge_connection.list_serial_ports", return_value=ports):
        host._refresh_ports()
    assert host.port_combo.value == expected


class AppHostGaugeConnectionTest(unittest.TestCase):
    def test_connect_configures_worker_and_disables_simulated_gauge(self) -> None:
        worker = _FakeGaugeWorker()
        host = _FakeGaugeHost(worker=worker, port_combo=FakeCombo("COM9"))
        host.baud_var.set("57600")
        host.req_cmd_var.set("M0,1")

        host.connect_gauge()

        self.assertFalse(host.sim_gauge_enabled)
        self.assertEqual(host.sim_gauge_var.get(), 0)
        self.assertEqual(worker.configures[-1]["port"], "COM9")
        self.assertEqual(worker.configures[-1]["baud"], 57600)
        self.assertEqual(worker.configures[-1]["request_cmd"], "M0,1")
        self.assertEqual(host.gauge_err_var.get(), "")

    def test_request_once_syncs_latest_command_before_sending(self) -> None:
        worker = _FakeGaugeWorker()
        host = _FakeGaugeHost(worker=worker)
        host.req_cmd_var.set("M0,1")

        host.request_gauge_once()

        self.assertEqual(worker.request_cmd, "M0,1")
        self.assertEqual(worker.send_count, 1)

    def test_auto_connect_failure_disables_worker_and_reports_error(self) -> None:
        worker = _FakeGaugeWorker(fail_configure=True)
        host = _FakeGaugeHost(worker=worker)

        host._auto_connect_gauge()

        self.assertEqual(worker.configures[-1]["enabled"], False)
        self.assertIn("失败", str(host.gauge_err_var.get()))
        self.assertIn("未连接", str(host.gauge_conn_var.get()))

    def test_disconnect_disables_worker(self) -> None:
        worker = _FakeGaugeWorker()
        host = _FakeGaugeHost(worker=worker)

        host.disconnect_gauge()

        self.assertEqual(worker.configures[-1]["enabled"], False)
        self.assertEqual(worker.configures[-1]["port"], "")
        self.assertIn("断开", str(host.gauge_err_var.get()))


if __name__ == "__main__":
    unittest.main()
