from __future__ import annotations

from typing import Any, Iterable, Mapping

from core.models import AxisCal


AXIS_CAL_PLC_FIELD_KEYS = ("sign", "off_ax0", "off_ax1", "off_ax2", "off_ax4", "b14", "b2", "keepout_w")


class AxisCalibrationState:
    """State holder for AxisCal UI values and write-readback expectations."""

    def __init__(self, current: AxisCal | None = None) -> None:
        self.current = current if current is not None else AxisCal()
        self.expected_regs: list[int] | None = None

    def set_current(self, cal: AxisCal) -> AxisCal:
        self.current = cal
        return cal

    def read_from_vars(self, vars_by_key: Mapping[str, Any]) -> AxisCal:
        def _f(key: str, default: float = 0.0) -> float:
            try:
                return float(str(vars_by_key[key].get()).strip())
            except Exception:
                return float(default)

        def _i(key: str, default: int = -1) -> int:
            try:
                return int(float(str(vars_by_key[key].get()).strip()))
            except Exception:
                return int(default)

        return AxisCal(
            sign=-1 if _i("sign", -1) < 0 else +1,
            off_ax0=_f("off_ax0"),
            off_ax1=_f("off_ax1"),
            off_ax2=_f("off_ax2"),
            off_ax4=_f("off_ax4"),
            b14=_f("b14"),
            b2=_f("b2"),
            keepout_w=_f("keepout_w"),
            z_pos=_f("z_pos"),
        )

    def write_to_vars(self, vars_by_key: Mapping[str, Any], cal: AxisCal) -> None:
        vars_by_key["sign"].set(str(int(cal.sign)))
        vars_by_key["off_ax0"].set(f"{cal.off_ax0:.6f}")
        vars_by_key["off_ax1"].set(f"{cal.off_ax1:.6f}")
        vars_by_key["off_ax2"].set(f"{cal.off_ax2:.6f}")
        vars_by_key["off_ax4"].set(f"{cal.off_ax4:.6f}")
        vars_by_key["b14"].set(f"{cal.b14:.6f}")
        vars_by_key["b2"].set(f"{cal.b2:.6f}")
        vars_by_key["keepout_w"].set(f"{cal.keepout_w:.6f}")
        vars_by_key["z_pos"].set(f"{cal.z_pos:.6f}")

    def set_field_status(self, status_vars: Mapping[str, Any], keys: Iterable[str], text: str) -> None:
        for key in keys:
            var = status_vars.get(key)
            if var is None:
                continue
            try:
                var.set(text)
            except Exception:
                pass

    def set_expected_regs(self, regs: Iterable[int]) -> list[int]:
        self.expected_regs = [int(reg) for reg in regs]
        return list(self.expected_regs)

    def clear_expected_regs(self) -> None:
        self.expected_regs = None

    def matches_expected_regs(self, regs: Iterable[int]) -> bool:
        return self.expected_regs is not None and list(self.expected_regs) == list(regs)


__all__ = ["AXIS_CAL_PLC_FIELD_KEYS", "AxisCalibrationState"]
