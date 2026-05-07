import unittest

from application.axis_calibration_state import AxisCalibrationState
from core.models import AxisCal


class _FakeVar:
    def __init__(self, value: object = "") -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class AxisCalibrationStateTest(unittest.TestCase):
    def test_reads_and_writes_axis_cal_ui_vars(self) -> None:
        state = AxisCalibrationState()
        vars_by_key = {
            "sign": _FakeVar("-1"),
            "off_ax0": _FakeVar("1.25"),
            "off_ax1": _FakeVar("2.5"),
            "off_ax2": _FakeVar("3.75"),
            "off_ax4": _FakeVar("4.0"),
            "b14": _FakeVar("5.5"),
            "b2": _FakeVar("6.25"),
            "keepout_w": _FakeVar("7.75"),
            "z_pos": _FakeVar("8.5"),
        }

        cal = state.read_from_vars(vars_by_key)

        self.assertEqual(cal.sign, -1)
        self.assertAlmostEqual(cal.off_ax0, 1.25)
        self.assertAlmostEqual(cal.off_ax1, 2.5)
        self.assertAlmostEqual(cal.off_ax2, 3.75)
        self.assertAlmostEqual(cal.off_ax4, 4.0)
        self.assertAlmostEqual(cal.b14, 5.5)
        self.assertAlmostEqual(cal.b2, 6.25)
        self.assertAlmostEqual(cal.keepout_w, 7.75)
        self.assertAlmostEqual(cal.z_pos, 8.5)

        state.write_to_vars(vars_by_key, AxisCal(sign=1, off_ax0=9.0, b14=10.0, z_pos=11.0))

        self.assertEqual(vars_by_key["sign"].get(), "1")
        self.assertEqual(vars_by_key["off_ax0"].get(), "9.000000")
        self.assertEqual(vars_by_key["b14"].get(), "10.000000")
        self.assertEqual(vars_by_key["z_pos"].get(), "11.000000")

    def test_tracks_expected_regs_for_verify_readback(self) -> None:
        state = AxisCalibrationState()

        state.set_expected_regs([1, 2, 3])

        self.assertTrue(state.matches_expected_regs([1, 2, 3]))
        self.assertFalse(state.matches_expected_regs([1, 2, 4]))

        state.clear_expected_regs()

        self.assertFalse(state.matches_expected_regs([1, 2, 3]))

    def test_sets_only_known_field_status_vars(self) -> None:
        state = AxisCalibrationState()
        status_vars = {"sign": _FakeVar("old"), "b14": _FakeVar("old")}

        state.set_field_status(status_vars, ["sign", "missing"], "updated")

        self.assertEqual(status_vars["sign"].get(), "updated")
        self.assertEqual(status_vars["b14"].get(), "old")


if __name__ == "__main__":
    unittest.main()
