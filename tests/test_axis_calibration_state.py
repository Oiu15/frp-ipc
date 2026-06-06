from __future__ import annotations

from tests.fakes import FakeVar

from application.host.calibration.state import AxisCalibrationState
from core.models import AxisCal


class TestAxisCalibrationState:
    def test_reads_and_writes_axis_cal_ui_vars(self) -> None:
        state = AxisCalibrationState()
        vars_by_key = {
            "sign": FakeVar("-1"),
            "off_ax0": FakeVar("1.25"),
            "off_ax1": FakeVar("2.5"),
            "off_ax2": FakeVar("3.75"),
            "off_ax4": FakeVar("4.0"),
            "b14": FakeVar("5.5"),
            "b2": FakeVar("6.25"),
            "keepout_w": FakeVar("7.75"),
            "z_pos": FakeVar("8.5"),
        }

        cal = state.read_from_vars(vars_by_key)

        assert cal.sign == -1
        assert cal.off_ax0 == 1.25
        assert cal.off_ax1 == 2.5
        assert cal.off_ax2 == 3.75
        assert cal.off_ax4 == 4.0
        assert cal.b14 == 5.5
        assert cal.b2 == 6.25
        assert cal.keepout_w == 7.75
        assert cal.z_pos == 8.5

        state.write_to_vars(vars_by_key, AxisCal(sign=1, off_ax0=9.0, b14=10.0, z_pos=11.0))

        assert vars_by_key["sign"].get() == "1"
        assert vars_by_key["off_ax0"].get() == "9.000000"
        assert vars_by_key["b14"].get() == "10.000000"
        assert vars_by_key["z_pos"].get() == "11.000000"

    def test_tracks_expected_regs_for_verify_readback(self) -> None:
        state = AxisCalibrationState()

        state.set_expected_regs([1, 2, 3])

        assert state.matches_expected_regs([1, 2, 3]) is True
        assert state.matches_expected_regs([1, 2, 4]) is False

        state.clear_expected_regs()

        assert state.matches_expected_regs([1, 2, 3]) is False

    def test_sets_only_known_field_status_vars(self) -> None:
        state = AxisCalibrationState()
        status_vars = {"sign": FakeVar("old"), "b14": FakeVar("old")}

        state.set_field_status(status_vars, ["sign", "missing"], "updated")

        assert status_vars["sign"].get() == "updated"
        assert status_vars["b14"].get() == "old"
