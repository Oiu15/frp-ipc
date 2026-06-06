from __future__ import annotations

import threading
import unittest
from typing import Any
from unittest.mock import patch

from application._host_length_measurement import (
    AX0_SOFTLIM_NEG_ABS,
    AX0_SOFTLIM_POS_ABS,
    HostLengthMeasurementMixin,
)
from core.models import AxisCal, AxisComm, Recipe


class _FakeVar:
    def __init__(self, value: object = "") -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class _FakeButton:
    def __init__(self) -> None:
        self.configs: list[dict[str, Any]] = []

    def configure(self, **kwargs: Any) -> None:
        self.configs.append(dict(kwargs))


class _AliveThread:
    def __init__(self) -> None:
        self.started = False

    def is_alive(self) -> bool:
        return True


class _FakeThread:
    created: list["_FakeThread"] = []

    def __init__(self, *, target: Any, args: tuple[Any, ...], daemon: bool) -> None:
        self.target = target
        self.args = args
        self.daemon = daemon
        self.started = False
        _FakeThread.created.append(self)

    def is_alive(self) -> bool:
        return self.started

    def start(self) -> None:
        self.started = True


class _FakeLengthHost(HostLengthMeasurementMixin):
    sim_gauge_var: Any
    len_enable_var: Any
    len_z_low_approach_var: Any
    len_low_search_dist_var: Any
    len_high_search_dist_var: Any
    len_search_vel_var: Any
    len_search_timeout_var: Any
    len_tol_var: Any
    len_high_margin_var: Any
    len_debounce_k_var: Any
    len_backoff_var: Any
    pipe_len_var: Any
    len_info_var: Any
    len_status_var: Any
    len_edge_state_var: Any
    len_edge_low_var: Any
    len_edge_high_var: Any
    len_edge_len_var: Any
    _len_edge_search_thread: Any
    _len_edge_search_high_thread: Any

    def __init__(self) -> None:
        self.axis_cal = AxisCal(sign=-1)
        self.recipe = Recipe(len_enable=True)
        self.axes = {
            0: AxisComm(act_pos=25.0, softlim_pos=100.0, softlim_neg=-400.0),
        }
        self.gauge_worker = None
        self.sim_gauge_var = _FakeVar(0)
        self.sim_gauge_enabled = False
        self.len_enable_var = _FakeVar(True)
        self.len_z_low_approach_var = _FakeVar("-100")
        self.len_low_search_dist_var = _FakeVar("200")
        self.len_high_search_dist_var = _FakeVar("80")
        self.len_search_vel_var = _FakeVar("10")
        self.len_search_timeout_var = _FakeVar("8")
        self.len_tol_var = _FakeVar("5")
        self.len_high_margin_var = _FakeVar("20")
        self.len_debounce_k_var = _FakeVar("2")
        self.len_backoff_var = _FakeVar("0")
        self.pipe_len_var = _FakeVar("300")
        self.len_info_var = _FakeVar("--")
        self.len_status_var = _FakeVar("--")
        self.len_edge_state_var = _FakeVar("--")
        self.len_edge_low_var = _FakeVar("--")
        self.len_edge_high_var = _FakeVar("--")
        self.len_edge_len_var = _FakeVar("--")
        self.btn_low = _FakeButton()
        self.btn_high = _FakeButton()

    def get_axis_copy(self, axis: int) -> AxisComm:
        return self.axes[int(axis)]

    def _recipe_ui_widget(self, name: str) -> Any:
        if name == "btn_len_search_low":
            return self.btn_low
        if name == "btn_len_search_high":
            return self.btn_high
        return None

    def _teach_len_search_low_worker(self, stop_evt: threading.Event) -> None:
        raise AssertionError("test should not execute worker")

    def _teach_len_search_high_worker(self, stop_evt: threading.Event) -> None:
        raise AssertionError("test should not execute worker")


class AppHostLengthMeasurementTest(unittest.TestCase):
    def test_z_disp_limits_use_plc_soft_limits_and_fallback_when_invalid(self) -> None:
        host = _FakeLengthHost()

        self.assertEqual(host._get_ax0_softlims_abs(), (-400.0, 100.0))
        self.assertEqual(host._get_ax0_z_disp_limits(), (-100.0, 400.0, 500.0))

        host.axes[0] = AxisComm(act_pos=25.0, softlim_pos=0.0, softlim_neg=0.0)

        self.assertEqual(host._get_ax0_softlims_abs(), (AX0_SOFTLIM_NEG_ABS, AX0_SOFTLIM_POS_ABS))

    def test_refresh_length_info_reports_status_for_enabled_disabled_and_too_long(self) -> None:
        host = _FakeLengthHost()

        with patch("application._host_length_measurement.tk.StringVar", _FakeVar):
            host._refresh_length_info()

        self.assertEqual(host.len_info_var.get(), "340")
        self.assertEqual(host.len_status_var.get(), "OK")

        host.len_enable_var.set(False)
        with patch("application._host_length_measurement.tk.StringVar", _FakeVar):
            host._refresh_length_info()
        self.assertEqual(host.len_status_var.get(), "未启用")

        host.len_enable_var.set(True)
        host.pipe_len_var.set("999")
        with patch("application._host_length_measurement.tk.StringVar", _FakeVar):
            host._refresh_length_info()
        self.assertEqual(host.len_status_var.get(), "将跳过(管长>340)")

    def test_pick_low_approach_uses_current_axis_abs_and_refreshes_info(self) -> None:
        host = _FakeLengthHost()

        with patch("application._host_length_measurement.tk.StringVar", _FakeVar):
            host._len_pick_low_approach()

        self.assertEqual(host.len_z_low_approach_var.get(), "25.000")
        self.assertNotEqual(host.len_info_var.get(), "--")

    def test_len_try_update_measured_length_sets_len_when_both_edges_are_valid(self) -> None:
        host = _FakeLengthHost()
        host.len_edge_low_var.set("250.0")
        host.len_edge_high_var.set("100.0")

        host._len_try_update_measured_length()

        self.assertEqual(host.len_edge_len_var.get(), "150.000")
        self.assertIn("L=150.0", str(host.len_edge_state_var.get()))

    def test_search_toggles_start_threads_and_stop_existing_searches(self) -> None:
        host = _FakeLengthHost()
        _FakeThread.created.clear()

        with patch("application._host_length_measurement.threading.Thread", _FakeThread):
            host._teach_len_search_low_toggle()

        self.assertTrue(_FakeThread.created[-1].started)
        self.assertTrue(host.btn_low.configs[-1]["text"])
        self.assertIn("准备", str(host.len_edge_state_var.get()))

        existing_evt = threading.Event()
        host._len_edge_search_thread = _AliveThread()
        host._len_edge_search_stop_evt = existing_evt

        host._teach_len_search_low_toggle()

        self.assertTrue(existing_evt.is_set())
        self.assertIn("停止中", str(host.len_edge_state_var.get()))

        low_evt = threading.Event()
        host._len_edge_search_thread = _AliveThread()
        host._len_edge_search_stop_evt = low_evt
        with patch("application._host_length_measurement.threading.Thread", _FakeThread):
            host._teach_len_search_high_toggle()

        self.assertTrue(low_evt.is_set())
        self.assertTrue(_FakeThread.created[-1].started)
        self.assertIn("准备", str(host.len_edge_state_var.get()))


if __name__ == "__main__":
    unittest.main()
