from __future__ import annotations

"""Application adapter from AppHost Tk variables to CalibrationViewPort."""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AppCalibrationViewAdapter:
    """Read/write calibration UI values through the legacy AppHost var surface."""

    host: Any

    def get_value(self, name: str, default: Any = None) -> Any:
        var = getattr(self.host, name, None)
        if var is None:
            return default
        try:
            return var.get()
        except Exception:
            return default

    def set_value(self, name: str, value: Any) -> None:
        var = getattr(self.host, name, None)
        if var is None:
            return
        try:
            var.set(value)
        except Exception:
            pass

    def get_float(self, name: str, default: float) -> float:
        parser = getattr(self.host, "_parse_float", None)
        raw = self.get_value(name, default)
        try:
            if callable(parser):
                parsed: Any = parser(raw, default)
                return float(parsed)
            return float(raw)
        except Exception:
            return float(default)


__all__ = ["AppCalibrationViewAdapter"]
