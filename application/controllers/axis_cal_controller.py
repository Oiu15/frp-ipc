from __future__ import annotations

from typing import Protocol


class AxisCalHostPort(Protocol):
    def axis_cal_read(self) -> None: ...

    def axis_cal_write(self) -> None: ...

    def axis_cal_capture_offsets(self) -> None: ...

    def axis_cal_calibrate_b14(self) -> None: ...

    def axis_cal_calibrate_keepout(self) -> None: ...

    def axis_cal_set_zpos_zero(self) -> None: ...


class AxisCalController:
    def __init__(self, host: AxisCalHostPort) -> None:
        self._host = host

    def axis_cal_read(self) -> None:
        self._host.axis_cal_read()

    def axis_cal_write(self) -> None:
        self._host.axis_cal_write()

    def axis_cal_capture_offsets(self) -> None:
        self._host.axis_cal_capture_offsets()

    def axis_cal_calibrate_b14(self) -> None:
        self._host.axis_cal_calibrate_b14()

    def axis_cal_calibrate_keepout(self) -> None:
        self._host.axis_cal_calibrate_keepout()

    def axis_cal_set_zpos_zero(self) -> None:
        self._host.axis_cal_set_zpos_zero()


__all__ = ["AxisCalController", "AxisCalHostPort"]
