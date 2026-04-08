from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from application.state import RuntimeState

from .calibration_mode import CalibrationMode, CalibrationState
from .production_mode import ProductionMode, ProductionModeState
from .validation_mode import ValidationMode, ValidationModeState


logger = logging.getLogger("frp.app.mode")


class ModeKind(StrEnum):
    PRODUCTION = "production"
    CALIBRATION = "calibration"
    VALIDATION = "validation"


@dataclass(slots=True)
class ModeMachine:
    production_mode: ProductionMode
    calibration_mode: CalibrationMode
    validation_mode: ValidationMode
    runtime_state: RuntimeState

    def __post_init__(self) -> None:
        self.runtime_state.mode_kind = "none"
        self.runtime_state.mode_state = "idle"
        self.runtime_state.mode_error = None

    @property
    def current_mode_kind(self) -> ModeKind | None:
        raw = str(self.runtime_state.mode_kind or "none")
        if raw == "none":
            return None
        try:
            return ModeKind(raw)
        except ValueError:
            return None

    @property
    def last_error(self) -> str | None:
        return self.runtime_state.mode_error

    @property
    def current_mode_name(self) -> str:
        return str(self.runtime_state.mode_kind or "none")

    @property
    def current_state_name(self) -> str:
        return str(self.runtime_state.mode_state or "idle")

    @property
    def current_mode(self) -> ProductionMode | CalibrationMode | ValidationMode | None:
        if self.current_mode_kind == ModeKind.PRODUCTION:
            return self.production_mode
        if self.current_mode_kind == ModeKind.CALIBRATION:
            return self.calibration_mode
        if self.current_mode_kind == ModeKind.VALIDATION:
            return self.validation_mode
        return None

    def enter_production(self) -> ProductionMode:
        self._deactivate_non_target_modes(ModeKind.PRODUCTION)
        self._set_current_mode(ModeKind.PRODUCTION)
        return self.production_mode

    def enter_calibration(self) -> CalibrationMode:
        self._deactivate_non_target_modes(ModeKind.CALIBRATION)
        self._set_current_mode(ModeKind.CALIBRATION)
        return self.calibration_mode

    def enter_validation(self) -> ValidationMode:
        self._deactivate_non_target_modes(ModeKind.VALIDATION)
        self._set_current_mode(ModeKind.VALIDATION)
        return self.validation_mode

    def stop_current(self) -> Any:
        if self.current_mode_kind is None:
            return None

        if self.current_mode_kind == ModeKind.PRODUCTION:
            result = self._stop_production_mode()
        elif self.current_mode_kind == ModeKind.CALIBRATION:
            result = self.calibration_mode.reset()
        else:
            result = self._stop_validation_mode()

        self.sync_current_mode_state()
        return result

    def recover_error(self) -> Any:
        mode = self.current_mode
        if mode is None or self.current_state_name != "error":
            return None

        if self.current_mode_kind == ModeKind.PRODUCTION:
            result = self.production_mode.reset()
        elif self.current_mode_kind == ModeKind.CALIBRATION:
            result = self.calibration_mode.reset()
        else:
            result = self.validation_mode.reset()
        self.sync_current_mode_state()
        return result

    def sync_current_mode_state(self) -> None:
        mode = self.current_mode
        if mode is None:
            self.runtime_state.mode_state = "idle"
            self.runtime_state.mode_error = None
            return
        self.runtime_state.mode_state = str(getattr(mode, "state_name", "idle") or "idle")
        self.runtime_state.mode_error = getattr(mode, "last_error", None)

    def sync_production_workflow_state(self, workflow_state: str, message: str = "") -> ProductionModeState:
        self._set_current_mode(ModeKind.PRODUCTION)
        state = self.production_mode.sync_from_workflow_state(workflow_state, message)
        self.sync_current_mode_state()
        return state

    def sync_validation_workflow_state(self, workflow_state: str, message: str = "") -> ValidationModeState:
        self._set_current_mode(ModeKind.VALIDATION)
        state = self.validation_mode.sync_from_workflow_state(workflow_state, message)
        self.sync_current_mode_state()
        return state

    def _stop_production_mode(self) -> Any:
        if self.production_mode.state in {ProductionModeState.RUNNING, ProductionModeState.PREPARING, ProductionModeState.STOPPING}:
            return self.production_mode.stop()
        return self.production_mode.reset()

    def _stop_validation_mode(self) -> Any:
        if self.validation_mode.state in {ValidationModeState.RUNNING, ValidationModeState.PREPARING, ValidationModeState.STOPPING}:
            return self.validation_mode.stop()
        return self.validation_mode.reset()

    def _deactivate_non_target_modes(self, target: ModeKind) -> None:
        if target != ModeKind.PRODUCTION:
            self._deactivate_production_mode()
        if target != ModeKind.CALIBRATION:
            self._deactivate_calibration_mode()
        if target != ModeKind.VALIDATION:
            self._deactivate_validation_mode()

    def _deactivate_production_mode(self) -> None:
        if self.production_mode.state in {ProductionModeState.RUNNING, ProductionModeState.PREPARING, ProductionModeState.STOPPING}:
            self.production_mode.stop()
        else:
            self.production_mode.reset()
        if self.current_mode_kind == ModeKind.PRODUCTION:
            self.sync_current_mode_state()

    def _deactivate_calibration_mode(self) -> None:
        if self.calibration_mode.state != CalibrationState.IDLE:
            self.calibration_mode.reset()
        if self.current_mode_kind == ModeKind.CALIBRATION:
            self.sync_current_mode_state()

    def _deactivate_validation_mode(self) -> None:
        if self.validation_mode.state in {ValidationModeState.RUNNING, ValidationModeState.PREPARING, ValidationModeState.STOPPING}:
            self.validation_mode.stop()
        else:
            self.validation_mode.reset()
        if self.current_mode_kind == ModeKind.VALIDATION:
            self.sync_current_mode_state()

    def _set_current_mode(self, target: ModeKind) -> None:
        previous = self.current_mode_kind
        self.runtime_state.mode_kind = target.value
        if previous == target:
            self.sync_current_mode_state()
            return
        logger.info(
            "MODE_MACHINE_TRANSITION from=%s to=%s",
            "none" if previous is None else previous.value,
            target.value,
        )
        self.sync_current_mode_state()


__all__ = ["ModeKind", "ModeMachine"]
