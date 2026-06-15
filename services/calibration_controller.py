from __future__ import annotations
# pyright: reportOptionalMemberAccess=false

"""Calibration control service — thin entrypoint for calibration actions.

Migrated from ``controllers/calibration_controller.py``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from modes.mode_machine import ModeMachine
from services.calibration_context import (
    IdCalibrationSettings,
    IdSingleCalibrationSettings,
    OdCalibrationSettings,
)
from services.calibration_service import CalibrationService
from services.id_calibration import IdCalibrationService
from services.id_single_calibration import IdSingleCalibrationService
from services.od_calibration import OdCalibrationService

CalibrationAction = Callable[[], Any]


@dataclass(slots=True)
class CalibrationController:
    """Application-layer entrypoint for calibration actions.

    Supports both new (port-based) and legacy (host-based) calibration
    services.  New callers should use the ``start_*_capture(settings)``
    methods which accept typed settings objects.  The legacy methods
    (without parameters) are retained for backward compatibility.
    """

    host: Any
    service: CalibrationService
    mode_machine: ModeMachine
    od_service: OdCalibrationService | None = None
    id_service: IdCalibrationService | None = None
    id_single_service: IdSingleCalibrationService | None = None

    # -- new port-based entrypoints ----------------------------------------

    def start_od_capture(self, settings: OdCalibrationSettings) -> None:
        if self.od_service is None:
            raise RuntimeError("OdCalibrationService not injected")
        self._run_in_calibration_mode(
            lambda: self.od_service.start_capture(
                rotation_speed_dps=settings.rotation_speed_dps,
                sampling_hz=settings.sampling_hz,
                capture_duration_s=settings.capture_duration_s,
                mode=settings.mode,
                angle_enabled=settings.angle_enabled,
                filter_mode=settings.filter_mode,
                outlier_sigma=settings.outlier_sigma,
                gauge_cmd=settings.gauge_cmd,
            )
        )

    def stop_od_capture(self, reason: str = "manual") -> None:
        if self.od_service is None:
            raise RuntimeError("OdCalibrationService not injected")
        self._run_in_calibration_mode(lambda: self.od_service.stop_capture(reason))

    def start_id_capture_new(self, settings: IdCalibrationSettings) -> None:
        if self.id_service is None:
            raise RuntimeError("IdCalibrationService not injected")
        self._run_in_calibration_mode(
            lambda: self.id_service.start_capture(
                rotation_speed_dps=settings.rotation_speed_dps,
                sampling_hz=settings.sampling_hz,
                capture_duration_s=settings.capture_duration_s,
                mode=settings.mode,
                force_one_rev=settings.force_one_rev,
            )
        )

    def stop_id_capture_new(self, reason: str = "manual") -> None:
        if self.id_service is None:
            raise RuntimeError("IdCalibrationService not injected")
        self._run_in_calibration_mode(lambda: self.id_service.stop_capture(reason))

    def compute_id_new(self, reference_diameter_mm: float = 150.0) -> Any:
        if self.id_service is None:
            raise RuntimeError("IdCalibrationService not injected")
        result: Any = None
        self._run_in_calibration_mode(lambda: self.id_service.compute_candidate(reference_diameter_mm))
        return result

    def apply_id_new(self, reference_diameter_mm: float = 150.0) -> Any:
        if self.id_service is None:
            raise RuntimeError("IdCalibrationService not injected")
        result: Any = None
        self._run_in_calibration_mode(lambda: self.id_service.apply_result(reference_diameter_mm))
        return result

    def start_id_single_capture_new(self, settings: IdSingleCalibrationSettings) -> None:
        if self.id_single_service is None:
            raise RuntimeError("IdSingleCalibrationService not injected")
        self._run_in_calibration_mode(
            lambda: self.id_single_service.start_capture(
                rotation_speed_dps=settings.rotation_speed_dps,
                sampling_hz=settings.sampling_hz,
                capture_duration_s=settings.capture_duration_s,
                reference_diameter_mm=settings.reference_diameter_mm,
            )
        )

    def stop_id_single_capture_new(self, reason: str = "manual") -> None:
        if self.id_single_service is None:
            raise RuntimeError("IdSingleCalibrationService not injected")
        self._run_in_calibration_mode(lambda: self.id_single_service.stop_capture(reason))

    def compute_id_single_new(self, reference_diameter_mm: float = 150.0) -> Any:
        if self.id_single_service is None:
            raise RuntimeError("IdSingleCalibrationService not injected")
        result: Any = None
        self._run_in_calibration_mode(
            lambda: self.id_single_service.compute_and_apply(reference_diameter_mm)
        )
        return result

    # -- legacy methods (kept for backward compat) -------------------------

    def start_od_b_capture(self) -> None:
        self._run_in_calibration_mode(lambda: self.service.start_od_capture(self.host))

    def stop_od_b_capture(self, reason: str = "manual") -> None:
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

    def stop_id_single_capture(self, reason: str = "manual") -> None:
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


__all__ = ["CalibrationAction", "CalibrationController"]
