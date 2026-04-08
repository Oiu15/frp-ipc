from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum


logger = logging.getLogger("frp.app.mode")


class CalibrationState(StrEnum):
    IDLE = "idle"
    ACQUIRING = "acquiring"
    FITTING = "fitting"
    SAVING = "saving"
    ERROR = "error"


@dataclass(slots=True)
class CalibrationMode:
    """Application-level calibration mode state.

    This keeps calibration lifecycle state out of controllers and UI code,
    while remaining lightweight enough to sit beside the legacy calibration
    implementation during migration.
    """

    state: CalibrationState = field(default=CalibrationState.IDLE, init=False)
    last_error: str | None = field(default=None, init=False)

    @property
    def state_name(self) -> str:
        return str(self.state.value)

    def begin_acquiring(self) -> CalibrationState:
        self.last_error = None
        self._set_state(CalibrationState.ACQUIRING)
        return self.state

    def begin_fitting(self) -> CalibrationState:
        self._set_state(CalibrationState.FITTING)
        return self.state

    def begin_saving(self) -> CalibrationState:
        self._set_state(CalibrationState.SAVING)
        return self.state

    def complete(self) -> CalibrationState:
        self.last_error = None
        self._set_state(CalibrationState.IDLE)
        return self.state

    def fail(self, message: str = "Calibration error") -> CalibrationState:
        self.last_error = str(message or "Calibration error")
        self._set_state(CalibrationState.ERROR)
        return self.state

    def reset(self) -> CalibrationState:
        self.last_error = None
        self._set_state(CalibrationState.IDLE)
        return self.state

    def _set_state(self, state: CalibrationState) -> None:
        if self.state == state:
            return
        previous = self.state
        self.state = state
        logger.info(
            "CALIBRATION_MODE_STATE from=%s to=%s",
            previous.value,
            state.value,
        )


__all__ = ["CalibrationMode", "CalibrationState"]
