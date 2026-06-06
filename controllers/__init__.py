"""Compatibility wrappers for controller imports now hosted in services."""

from importlib import import_module
from typing import Any

__all__ = [
    "calibration_controller",
    "measurement_controller",
]

_EXPORT_MODULES = {
    "calibration_controller": "controllers.calibration_controller",
    "measurement_controller": "controllers.measurement_controller",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(module_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
