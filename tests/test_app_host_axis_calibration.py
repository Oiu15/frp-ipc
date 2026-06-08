from __future__ import annotations

import queue
from typing import Any

from tests.fakes import FakeVar

from application.host.calibration.axis import HostAxisCalibrationMixin
from application.host.calibration.state import AxisCalibrationState
from config.addresses import AXISCAL_MB_BASE, AXISCAL_WORDS
from core.models import AxisCal, AxisComm
from drivers.plc_client import CmdReadRegs, CmdWriteRegs


class _FakeAxisCalHost(HostAxisCalibrationMixin):
    def __init__(self) -> None:
        self.cmd_q: queue.Queue[Any] = queue.Queue()
        self.axis_cal = AxisCal()
        self._axis_cal_state = AxisCalibrationState(self.axis_cal)
        self._axis_cal_write_expect_regs: list[int] | None = None
        self.axis_cal_vars: dict[str, Any] = {
            "sign": FakeVar("-1"),
            "off_ax0": FakeVar("0"),
            "off_ax1": FakeVar("0"),
            "off_ax2": FakeVar("0"),
            "off_ax4": FakeVar("0"),
            "b14": FakeVar("0"),
            "b2": FakeVar("0"),
            "keepout_w": FakeVar("0"),
            "z_pos": FakeVar("0"),
        }
        self.axis_cal_field_status_vars: dict[str, Any] = {
            key: FakeVar("initial")
            for key in (
                "sign",
                "off_ax0",
                "off_ax1",
                "off_ax2",
                "off_ax4",
                "b14",
                "b2",
                "keepout_w",
                "z_pos",
            )
        }
        self.axis_cal_status_vars: dict[str, Any] = {
            key: FakeVar("-")
            for key in (
                "off_abs",
                "act_abs",
                "softlim_pos",
                "softlim_neg",
                "z_raw",
                "z_disp",
                "keepout_raw",
                "keepout_disp",
            )
        }
        self.axes = {
            0: AxisComm(act_pos=10.0, softlim_pos=100.0, softlim_neg=-100.0),
            1: AxisComm(act_pos=4.0, softlim_pos=80.0, softlim_neg=-80.0),
            2: AxisComm(act_pos=7.0, softlim_pos=50.0, softlim_neg=-50.0),
            4: AxisComm(act_pos=2.0, softlim_pos=60.0, softlim_neg=-60.0),
        }

    def get_axis_copy(self, axis: int) -> AxisComm:
        return self.axes[int(axis)]


class TestAppHostAxisCalibration:
    def test_axis_cal_read_enqueues_plc_read_and_marks_fields(self) -> None:
        host = _FakeAxisCalHost()

        host.axis_cal_read()

        cmd = host.cmd_q.get_nowait()
        assert isinstance(cmd, CmdReadRegs)
        assert cmd.d_addr == AXISCAL_MB_BASE
        assert cmd.count == AXISCAL_WORDS
        assert cmd.tag == "axis_cal"
        assert host.axis_cal_field_status_vars["sign"].get() != "initial"

    def test_axis_cal_write_enqueues_write_then_verify_read(self) -> None:
        host = _FakeAxisCalHost()
        host.axis_cal_vars["off_ax0"].set("1.25")
        host.axis_cal_vars["off_ax1"].set("2.5")
        host.axis_cal_vars["off_ax2"].set("3.75")
        host.axis_cal_vars["off_ax4"].set("4.0")
        host.axis_cal_vars["b14"].set("5.5")
        host.axis_cal_vars["b2"].set("6.25")
        host.axis_cal_vars["keepout_w"].set("7.75")
        host.axis_cal_vars["z_pos"].set("8.5")

        host.axis_cal_write()

        write_cmd = host.cmd_q.get_nowait()
        verify_cmd = host.cmd_q.get_nowait()
        assert isinstance(write_cmd, CmdWriteRegs)
        assert isinstance(verify_cmd, CmdReadRegs)
        assert write_cmd.d_addr == AXISCAL_MB_BASE
        assert verify_cmd.tag == "axis_cal_verify"
        assert host._axis_cal_write_expect_regs == write_cmd.values
        assert host._get_axis_calibration_state().matches_expected_regs(write_cmd.values)

    def test_axis_cal_capture_and_calibration_helpers_update_ui_values(self) -> None:
        host = _FakeAxisCalHost()

        host.axis_cal_capture_offsets()

        assert host.axis_cal_vars["off_ax0"].get() == "10.000000"
        assert host.axis_cal_vars["off_ax1"].get() == "4.000000"
        assert host.axis_cal_vars["off_ax2"].get() == "7.000000"
        assert host.axis_cal_vars["off_ax4"].get() == "2.000000"
        assert host.axis_cal_field_status_vars["off_ax0"].get() != "initial"

        for key in ("off_ax0", "off_ax1", "off_ax2", "off_ax4"):
            host.axis_cal_vars[key].set("0")
        host.axis_cal_calibrate_b14()
        host.axis_cal_calibrate_keepout()
        host.axis_cal_set_zpos_zero()

        assert host.axis_cal_vars["b14"].get() == "4.000000"
        assert host.axis_cal_vars["b2"].get() == "0.000000"
        assert host.axis_cal_vars["keepout_w"].get() == "3.000000"
        assert host.axis_cal_vars["z_pos"].get() == "-10.000000"

    def test_axis_cal_refresh_status_computes_current_display_values(self) -> None:
        host = _FakeAxisCalHost()

        host.axis_cal_refresh_status()

        assert "AX0=10.000" in str(host.axis_cal_status_vars["act_abs"].get())
        assert "AX0=100.000" in str(host.axis_cal_status_vars["softlim_pos"].get())
        assert "Z0=-10.000" in str(host.axis_cal_status_vars["z_raw"].get())
        assert "Zod_disp=-10.000" in str(host.axis_cal_status_vars["z_disp"].get())
