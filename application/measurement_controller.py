from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modes.mode_machine import ModeMachine


@dataclass(slots=True)
class MeasurementController:
    """Thin application-layer entrypoint for measurement start/stop."""

    mode_machine: ModeMachine

    def start_measurement(self) -> Any:
        self.mode_machine.enter_production()
        mode = self.mode_machine.current_mode
        if mode is None:
            return None
        return mode.start()

    def stop_measurement(self) -> Any:
        return self.mode_machine.stop_current()


__all__ = ["MeasurementController"]
