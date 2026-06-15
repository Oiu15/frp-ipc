from __future__ import annotations

"""Calibration control service — thin entrypoint for calibration actions.

Migrated from ``controllers/calibration_controller.py``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from modes.mode_machine import ModeMachine
from services.calibration_service import CalibrationService
from services.id_single_calibration import IdSingleCalibrationService
from services.od_calibration import OdCalibrationService
from services.id_calibration import IdCalibrationService

CalibrationAction = Callable[[], Any]


@dataclass(slots=True)
class CalibrationController:
    """Application-layer entrypoint for calibration actions.

    Supports both legacy (host-based) and new (port-based) calibration
    services.  New code should use the port-based services; the legacy
    service is retained for backward compatibility.
    """

    host: Any
    service: CalibrationService
    mode_machine: ModeMachine
    # New port-based services — set by the host when available
    od_service: OdCalibrationService | None = None
    id_service: IdCalibrationService | None = None
    id_single_service: IdSingleCalibrationService | None = None

    # -- legacy host-based methods (kept for backward compat) --------------

    def start_od_b_capture(self) -> None:
        self._run_in_calibration_mode(lambda: self.service.start_od_capture(self.host))

    def stop_od_b_capture(self, reason: str = 'manual') -> None:
        self._run_in_calibration_mode(lambda: self.service.stop_od_capture(self.host, reason))

    def clear_od_b_capture(self) -> None:
        self._run_in_calibration_mode(lambda: self.service.clear_od_capture(self.host))

    def compute_od_b(self) -> None:
        self._run_in_calibration_mode(lambda: self.service.compute_od_candidate(self.host))

    def apply_od_b(self) -> None:
        self._run_in_calibration_mode(lambda: self.service.apply_od_candidate(self.host))

    def export_od_b_raw(self) -> None:
        self._run_in_calibration_mode(lambda: self.service.export_od_raw(self.host))

    def start_id_capture(self) -> None:
        self._run_in_calibration_mode(lambda: self.service.start_id_capture(self.host))

    def stop_id_capture(self) -> None:
        self._run_in_calibration_mode(lambda: self.service.stop_id_capture(self.host))

    def clear_id_capture(self) -> None:
        self._run_in_calibration_mode(lambda: self.service.clear_id_capture(self.host))

    def compute_id_calibration(self) -> None:
        self._run_in_calibration_mode(lambda: self.service.compute_id_candidate(self.host))

    def apply_id_calibration(self) -> None:
        self._run_in_calibration_mode(lambda: self.service.apply_id_candidate(self.host))

    def export_id_raw(self) -> None:
        self._run_in_calibration_mode(lambda: self.service.export_id_raw(self.host))

    def verify_id_calibration(self) -> None:
        self._run_in_calibration_mode(lambda: self.service.verify_id(self.host))

    def start_id_single_capture(self) -> None:
        self._run_in_calibration_mode(lambda: self.service.start_id_single_capture(self.host))

    def stop_id_single_capture(self, reason: str = 'manual') -> None:
        self._run_in_calibration_mode(lambda: self.service.stop_id_single_capture(self.host, reason))

    def clear_id_single_capture(self) -> None:
        self._run_in_calibration_mode(lambda: self.service.clear_id_single_capture(self.host))

    def compute_and_write_id_single_calibration(self) -> None:
        self._run_in_calibration_mode(lambda: self.service.compute_apply_id_single(self.host))

    def _run_in_calibration_mode(self, action: CalibrationAction) -> Any:
        self.mode_machine.enter_calibration()
        result = action()
        self.mode_machine.sync_current_mode_state()
        return result


__all__ = ['CalibrationAction', 'CalibrationController']
