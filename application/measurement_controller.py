from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modes.production_mode import ProductionMode


@dataclass(slots=True)
class MeasurementController:
    """Thin application-layer entrypoint for measurement start/stop."""

    production_mode: ProductionMode

    def start_measurement(self) -> Any:
        return self.production_mode.start()

    def stop_measurement(self) -> Any:
        return self.production_mode.stop()


__all__ = ["MeasurementController"]
