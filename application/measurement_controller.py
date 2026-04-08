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
        result = mode.start()
        self.mode_machine.sync_current_mode_state()
        return result

    def stop_measurement(self) -> Any:
        result = self.mode_machine.stop_current()
        self.mode_machine.sync_current_mode_state()
        return result


__all__ = ["MeasurementController"]
