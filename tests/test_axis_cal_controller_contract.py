from __future__ import annotations

from application.controllers.axis_cal_controller import AxisCalController


class _FakeAxisCalHost:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def axis_cal_read(self) -> None:
        self.calls.append("axis_cal_read")

    def axis_cal_write(self) -> None:
        self.calls.append("axis_cal_write")

    def axis_cal_capture_offsets(self) -> None:
        self.calls.append("axis_cal_capture_offsets")

    def axis_cal_calibrate_b14(self) -> None:
        self.calls.append("axis_cal_calibrate_b14")

    def axis_cal_calibrate_keepout(self) -> None:
        self.calls.append("axis_cal_calibrate_keepout")

    def axis_cal_set_zpos_zero(self) -> None:
        self.calls.append("axis_cal_set_zpos_zero")


def test_axis_cal_controller_delegates_commands_to_host() -> None:
    host = _FakeAxisCalHost()
    controller = AxisCalController(host)

    controller.axis_cal_read()
    controller.axis_cal_write()
    controller.axis_cal_capture_offsets()
    controller.axis_cal_calibrate_b14()
    controller.axis_cal_calibrate_keepout()
    controller.axis_cal_set_zpos_zero()

    assert host.calls == [
        "axis_cal_read",
        "axis_cal_write",
        "axis_cal_capture_offsets",
        "axis_cal_calibrate_b14",
        "axis_cal_calibrate_keepout",
        "axis_cal_set_zpos_zero",
    ]
