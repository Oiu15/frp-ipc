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
from services.calibration_ports import CalibrationViewPort
from services.calibration_service import CalibrationService
from services.id_calibration import IdCalibrationService
from services.id_single_calibration import IdSingleCalibrationService
from services.od_calibration import OdCalibrationService

CalibrationAction = Callable[[], Any]


@dataclass(slots=True)
class HostCalibrationViewAdapter:
    """Transitional adapter from legacy host Tk variables to CalibrationViewPort."""

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
    view: CalibrationViewPort | None = None

    def __post_init__(self) -> None:
        if self.view is None:
            self.view = HostCalibrationViewAdapter(self.host)

    # -- host compatibility helpers ---------------------------------------

    def _var(self, name: str, default: Any = None) -> Any:
        return self._view().get_value(name, default)

    def _set_var(self, name: str, value: Any) -> None:
        self._view().set_value(name, value)

    def _float_var(self, name: str, default: float) -> float:
        return self._view().get_float(name, default)

    def _view(self) -> CalibrationViewPort:
        if self.view is None:
            self.view = HostCalibrationViewAdapter(self.host)
        return self.view

    def _run_legacy_host_service(self, action: CalibrationAction) -> Any:
        """Deprecated fallback for legacy CalibrationService(host: Any) paths."""
        return self._run_in_calibration_mode(action)

    def _od_settings_from_host(self) -> OdCalibrationSettings:
        angle_src = str(self._var("odcal_angle_src_var", "AX3") or "AX3").strip()
        return OdCalibrationSettings(
            rotation_speed_dps=self._float_var("odcal_rot_degps_var", 10.0),
            sampling_hz=self._float_var("odcal_hz_var", 20.0),
            capture_duration_s=self._float_var("odcal_duration_var", 10.0),
            reference_diameter_mm=self._float_var("odcal_dref_var", 180.0),
            mode=str(self._var("odcal_mode_var", "timed") or "timed").strip(),
            angle_enabled=(("无" not in angle_src) and angle_src.upper() != "NONE"),
            filter_mode=str(self._var("odcal_filter_var", "") or "").strip(),
            outlier_sigma=self._float_var("odcal_outlier_sigma_var", 3.0),
            gauge_cmd=str(self._var("odcal_cmd_var", "M0,1") or "M0,1").strip(),
        )

    def _id_settings_from_host(self) -> IdCalibrationSettings:
        return IdCalibrationSettings(
            rotation_speed_dps=self._float_var("idcal_rot_degps_var", 10.0),
            sampling_hz=self._float_var("idcal_hz_var", 20.0),
            capture_duration_s=self._float_var("idcal_duration_var", 10.0),
            reference_diameter_mm=self._float_var("idcal_dref_var", 150.0),
            mode=str(self._var("idcal_mode_var", "one_rev") or "one_rev").strip(),
        )

    def _id_single_settings_from_host(self) -> IdSingleCalibrationSettings:
        return IdSingleCalibrationSettings(
            rotation_speed_dps=self._float_var("idcal_rot_degps_var", 10.0),
            sampling_hz=self._float_var("idcal_hz_var", 20.0),
            capture_duration_s=self._float_var("idcal_duration_var", 10.0),
            reference_diameter_mm=self._float_var("id_single_cal_dref_var", 150.0),
        )

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
        return self._run_in_calibration_mode(lambda: self.id_service.compute_candidate(reference_diameter_mm))

    def apply_id_new(self, reference_diameter_mm: float = 150.0) -> Any:
        if self.id_service is None:
            raise RuntimeError("IdCalibrationService not injected")
        return self._run_in_calibration_mode(lambda: self.id_service.apply_result(reference_diameter_mm))

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
        return self._run_in_calibration_mode(
            lambda: self.id_single_service.compute_and_apply(reference_diameter_mm)
        )

    # -- legacy methods (kept for backward compat) -------------------------

    def start_od_b_capture(self) -> None:
        if self.od_service is None:
            self._run_legacy_host_service(lambda: self.service.start_od_capture(self.host))
            return
        settings = self._od_settings_from_host()
        self.start_od_capture(settings)
        self._set_var("odcal_state_var", "CAPTURING")
        self._set_var("odcal_msg_var", "采集中...")

    def stop_od_b_capture(self, reason: str = "manual") -> None:
        if self.od_service is None:
            self._run_legacy_host_service(lambda: self.service.stop_od_capture(self.host, reason))
            return
        self.stop_od_capture(reason)
        self._set_var("odcal_state_var", "DONE")
        self._set_var("odcal_msg_var", reason or "已停止")

    def clear_od_b_capture(self) -> None:
        if self.od_service is None:
            self._run_legacy_host_service(lambda: self.service.clear_od_capture(self.host))
            return
        self._run_in_calibration_mode(lambda: self.od_service.clear_capture())
        self._set_var("odcal_state_var", "IDLE")
        self._set_var("odcal_msg_var", "-")
        self._set_var("odcal_B_candidate_var", "--")
        self._set_var("odcal_n_var", "0")

    def compute_od_b(self) -> None:
        if self.od_service is None:
            self._run_legacy_host_service(lambda: self.service.compute_od_candidate(self.host))
            return
        settings = self._od_settings_from_host()
        result = self._run_in_calibration_mode(
            lambda: self.od_service.compute_candidate(settings.reference_diameter_mm, settings.outlier_sigma)
        )
        if result.get("ok"):
            self._set_var("odcal_B_candidate_var", f"{float(result['b_mm']):.5f}")
            self._set_var("odcal_state_var", "DONE")
            self._set_var("odcal_msg_var", "已计算 B_candidate，可应用")
        else:
            self._set_var("odcal_state_var", "ERROR")
            self._set_var("odcal_msg_var", str(result.get("reason", "计算失败")))

    def apply_od_b(self) -> None:
        if self.od_service is None:
            self._run_legacy_host_service(lambda: self.service.apply_od_candidate(self.host))
            return
        settings = self._od_settings_from_host()
        result = self._run_in_calibration_mode(
            lambda: self.od_service.apply_result(
                settings.reference_diameter_mm,
                gauge_cmd=settings.gauge_cmd,
                out1_map=str(self._var("odcal_map_out1_var", "L") or "L"),
            )
        )
        if result.get("ok"):
            self._set_var("odcal_B_active_var", f"{float(result['b_mm']):.5f}")
            self._set_var("odcal_state_var", "APPLIED")
            self._set_var("odcal_msg_var", "已应用并保存")
        else:
            self._set_var("odcal_state_var", "ERROR")
            self._set_var("odcal_msg_var", str(result.get("reason", "应用失败")))

    def export_od_b_raw(self) -> None:
        self._run_legacy_host_service(lambda: self.service.export_od_raw(self.host))

    def start_id_capture(self) -> None:
        if self.id_service is None:
            self._run_legacy_host_service(lambda: self.service.start_id_capture(self.host))
            return
        self.start_id_capture_new(self._id_settings_from_host())
        self._set_var("idcal_state_var", "CAPTURING")
        self._set_var("idcal_msg_var", "采集中...")

    def stop_id_capture(self) -> None:
        if self.id_service is None:
            self._run_legacy_host_service(lambda: self.service.stop_id_capture(self.host))
            return
        self.stop_id_capture_new()
        self._set_var("idcal_state_var", "STOP")
        self._set_var("idcal_msg_var", "已停止")

    def clear_id_capture(self) -> None:
        if self.id_service is None:
            self._run_legacy_host_service(lambda: self.service.clear_id_capture(self.host))
            return
        self._run_in_calibration_mode(lambda: self.id_service.clear_capture())
        self._set_var("idcal_state_var", "IDLE")
        self._set_var("idcal_msg_var", "已清空")
        self._set_var("idcal_delta_candidate_var", "--")

    def compute_id_calibration(self) -> None:
        if self.id_service is None:
            self._run_legacy_host_service(lambda: self.service.compute_id_candidate(self.host))
            return
        settings = self._id_settings_from_host()
        result = self.compute_id_new(settings.reference_diameter_mm)
        if result.get("ok"):
            self._set_var("idcal_delta_candidate_var", f"{float(result['delta_c_mm']):.4f}")
            self._set_var("idcal_state_var", "READY")
            self._set_var("idcal_msg_var", "计算完成")
        else:
            self._set_var("idcal_state_var", "ERR")
            self._set_var("idcal_msg_var", str(result.get("reason", "计算失败")))

    def apply_id_calibration(self) -> None:
        if self.id_service is None:
            self._run_legacy_host_service(lambda: self.service.apply_id_candidate(self.host))
            return
        settings = self._id_settings_from_host()
        result = self.apply_id_new(settings.reference_diameter_mm)
        if result.get("ok"):
            self._set_var("idcal_delta_active_var", f"{float(result['delta_c_mm']):.4f}")
            self._set_var("idcal_state_var", "APPLIED")
            self._set_var("idcal_msg_var", "已应用并保存")
        else:
            self._set_var("idcal_state_var", "ERR")
            self._set_var("idcal_msg_var", str(result.get("reason", "应用失败")))

    def export_id_raw(self) -> None:
        self._run_legacy_host_service(lambda: self.service.export_id_raw(self.host))

    def verify_id_calibration(self) -> None:
        self._run_legacy_host_service(lambda: self.service.verify_id(self.host))

    def start_id_single_capture(self) -> None:
        if self.id_single_service is None:
            self._run_legacy_host_service(lambda: self.service.start_id_single_capture(self.host))
            return
        self.start_id_single_capture_new(self._id_single_settings_from_host())
        self._set_var("id_single_cal_state_var", "CAPTURING")
        self._set_var("id_single_cal_msg_var", "采集中...")

    def stop_id_single_capture(self, reason: str = "manual") -> None:
        if self.id_single_service is None:
            self._run_legacy_host_service(lambda: self.service.stop_id_single_capture(self.host, reason))
            return
        self.stop_id_single_capture_new(reason)
        self._set_var("id_single_cal_state_var", "STOP")
        self._set_var("id_single_cal_msg_var", reason or "已停止")

    def clear_id_single_capture(self) -> None:
        if self.id_single_service is None:
            self._run_legacy_host_service(lambda: self.service.clear_id_single_capture(self.host))
            return
        self._run_in_calibration_mode(lambda: self.id_single_service.clear_capture())
        self._set_var("id_single_cal_state_var", "IDLE")
        self._set_var("id_single_cal_msg_var", "已清空")
        self._set_var("id_single_cal_mean_var", "--")
        self._set_var("id_single_cal_B_var", "--")
        self._set_var("id_single_cal_cov_var", "--")

    def compute_and_write_id_single_calibration(self) -> None:
        if self.id_single_service is None:
            self._run_legacy_host_service(lambda: self.service.compute_apply_id_single(self.host))
            return
        settings = self._id_single_settings_from_host()
        result = self.compute_id_single_new(settings.reference_diameter_mm)
        if result.get("ok"):
            self._set_var("id_single_cal_mean_var", f"{float(result['mean_l2_mm']):.5f}")
            self._set_var("id_single_cal_B_var", f"{float(result['b_mm']):.5f}")
            self._set_var("id_single_cal_cov_var", f"{float(result['cov_pct']):.1f}%")
            self._set_var("id_single_cal_state_var", "APPLIED")
            self._set_var("id_single_cal_msg_var", "已写入仓储")
        else:
            self._set_var("id_single_cal_state_var", "ERR")
            self._set_var("id_single_cal_msg_var", str(result.get("reason", "计算失败")))

    def _run_in_calibration_mode(self, action: CalibrationAction) -> Any:
        self.mode_machine.enter_calibration()
        result = action()
        self.mode_machine.sync_current_mode_state()
        return result


__all__ = ["CalibrationAction", "CalibrationController", "HostCalibrationViewAdapter"]
