from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


MeasurementAction = Callable[[], Any]


@dataclass(slots=True)
class MeasurementController:
    """Thin application-layer entrypoint for measurement start/stop."""

    start_impl: MeasurementAction
    stop_impl: MeasurementAction

    def start_measurement(self) -> Any:
        return self.start_impl()

    def stop_measurement(self) -> Any:
        return self.stop_impl()


__all__ = ["MeasurementController"]
