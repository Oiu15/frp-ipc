from __future__ import annotations
# pyright: reportOptionalMemberAccess=false

"""Calibration control service — thin entrypoint for calibration actions.

Migrated from ``controllers/calibration_controller.py``.
"""

import math
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
from services.id_calibration import IdCalibrationService
from services.id_single_calibration import IdSingleCalibrationService
from services.od_calibration import OdCalibrationService
from services.tooling_calibration import ToolingCalibrationService

CalibrationAction = Callable[[], Any]


@dataclass(slots=True)
class CalibrationController:
    """Application-layer entrypoint for calibration actions.

    UI entrypoints use port-based calibration services.  Methods without
    parameters are retained as screen command names and read settings through
    ``CalibrationViewPort``.
    """

    mode_machine: ModeMachine
    view: CalibrationViewPort
    od_service: OdCalibrationService | None = None
    id_service: IdCalibrationService | None = None
    id_single_service: IdSingleCalibrationService | None = None
    tooling_service: ToolingCalibrationService | None = None

    # -- view helpers ------------------------------------------------------

    def _var(self, name: str, default: Any = None) -> Any:
        return self.view.get_value(name, default)

    def _set_var(self, name: str, value: Any) -> None:
        self.view.set_value(name, value)

    def _float_var(self, name: str, default: float) -> float:
        return self.view.get_float(name, default)

    def _publish_raw_export_result(self, *, state_var: str, msg_var: str, result: Any) -> None:
        if isinstance(result, dict) and result.get("ok"):
            path = result.get("path")
            name = getattr(path, "name", str(path or ""))
            self._set_var(msg_var, f"已导出: {name}")
            return
        reason = str(result.get("reason", "无数据") if isinstance(result, dict) else "无数据")
        self._set_var(state_var, "ERR")
        self._set_var(msg_var, reason)

    def _optional_float(self, value: Any) -> float | None:
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None
        except Exception:
            return None

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

    # -- UI command methods ------------------------------------------------

    def start_od_b_capture(self) -> None:
        if self.od_service is None:
            raise RuntimeError("OdCalibrationService not injected")
        settings = self._od_settings_from_host()
        self.start_od_capture(settings)
        self._set_var("odcal_state_var", "CAPTURING")
        self._set_var("odcal_msg_var", "采集中...")

    def stop_od_b_capture(self, reason: str = "manual") -> None:
        if self.od_service is None:
            raise RuntimeError("OdCalibrationService not injected")
        self.stop_od_capture(reason)
        self._set_var("odcal_state_var", "DONE")
        self._set_var("odcal_msg_var", reason or "已停止")

    def clear_od_b_capture(self) -> None:
        if self.od_service is None:
            raise RuntimeError("OdCalibrationService not injected")
        self._run_in_calibration_mode(lambda: self.od_service.clear_capture())
        self._set_var("odcal_state_var", "IDLE")
        self._set_var("odcal_msg_var", "-")
        self._set_var("odcal_B_candidate_var", "--")
        self._set_var("odcal_n_var", "0")

    def compute_od_b(self) -> None:
        if self.od_service is None:
            raise RuntimeError("OdCalibrationService not injected")
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
            raise RuntimeError("OdCalibrationService not injected")
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
        if self.od_service is None:
            raise RuntimeError("OdCalibrationService not injected")
        result = self._run_in_calibration_mode(lambda: self.od_service.export_raw())
        self._publish_raw_export_result(
            state_var="odcal_state_var",
            msg_var="odcal_msg_var",
            result=result,
        )

    def start_id_capture(self) -> None:
        if self.id_service is None:
            raise RuntimeError("IdCalibrationService not injected")
        self.start_id_capture_new(self._id_settings_from_host())
        self._set_var("idcal_state_var", "CAPTURING")
        self._set_var("idcal_msg_var", "采集中...")

    def stop_id_capture(self) -> None:
        if self.id_service is None:
            raise RuntimeError("IdCalibrationService not injected")
        self.stop_id_capture_new()
        self._set_var("idcal_state_var", "STOP")
        self._set_var("idcal_msg_var", "已停止")

    def clear_id_capture(self) -> None:
        if self.id_service is None:
            raise RuntimeError("IdCalibrationService not injected")
        self._run_in_calibration_mode(lambda: self.id_service.clear_capture())
        self._set_var("idcal_state_var", "IDLE")
        self._set_var("idcal_msg_var", "已清空")
        self._set_var("idcal_delta_candidate_var", "--")

    def compute_id_calibration(self) -> None:
        if self.id_service is None:
            raise RuntimeError("IdCalibrationService not injected")
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
            raise RuntimeError("IdCalibrationService not injected")
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
        if self.id_service is None:
            raise RuntimeError("IdCalibrationService not injected")
        result = self._run_in_calibration_mode(lambda: self.id_service.export_raw())
        self._publish_raw_export_result(
            state_var="idcal_state_var",
            msg_var="idcal_msg_var",
            result=result,
        )

    # -- geometry_v2 tooling calibration (几何标定 V2 页) -------------------

    def _require_tooling(self) -> ToolingCalibrationService:
        if self.tooling_service is None:
            raise RuntimeError("ToolingCalibrationService not injected")
        return self.tooling_service

    def start_tcal_id_capture(self) -> None:
        svc = self._require_tooling()
        self._run_in_calibration_mode(
            lambda: svc.start_capture(
                mode="id",
                rotation_speed_dps=self._float_var("tcal_rot_degps_var", 10.0),
                sampling_hz=self._float_var("tcal_hz_var", 20.0),
            )
        )
        self._set_var("tcal_msg_var", "ID 位姿采集中...")

    def stop_tcal_id_capture(self, reason: str = "manual") -> None:
        svc = self._require_tooling()
        self._run_in_calibration_mode(lambda: svc.stop_capture(reason))
        self._set_var("tcal_msg_var", reason or "已停止")

    def add_tcal_dataset(self) -> None:
        svc = self._require_tooling()
        result = svc.add_dataset()
        if result.get("ok"):
            self._set_var("tcal_id_nsets_var", str(int(result.get("n_sets", 0))))
            self._set_var("tcal_msg_var", f"已加入第 {result.get('n_sets')} 组装夹 ({result.get('n_points')} 点)")
        else:
            self._set_var("tcal_msg_var", str(result.get("reason", "加入失败")))

    def clear_tcal_datasets(self) -> None:
        svc = self._require_tooling()
        svc.clear_datasets()
        self._set_var("tcal_id_nsets_var", "0")
        self._set_var("tcal_msg_var", "已清空数据集")

    def fit_tcal_id_pose(self) -> None:
        svc = self._require_tooling()
        result = svc.compute_id_pose(
            r_known=self._float_var("tcal_r_known_var", 76.35),
            d_init=self._float_var("tcal_d_init_var", 140.0),
        )
        if result.get("ok"):
            self._set_var("tcal_id_s_var", f"{float(result['s_lateral']):+.4f}")
            self._set_var("tcal_id_axis_var", f"{float(result['axis_deg']):+.3f}")
            self._set_var("tcal_id_q_var", f"({float(result['qx']):+.3f}, {float(result['qy']):+.3f})")
            self._set_var("tcal_id_cost_var", f"{float(result['cost']):.2e}")
            self._set_var("tcal_msg_var", "ID 位姿拟合完成,可应用")
        else:
            self._set_var("tcal_msg_var", str(result.get("reason", "拟合失败")))

    def apply_tcal_id_pose(self) -> None:
        svc = self._require_tooling()
        result = svc.apply_id_pose()
        if result.get("ok"):
            self._set_var("tcal_msg_var", "ID 位姿已应用并保存")
            self.reload_tooling()
        else:
            self._set_var("tcal_msg_var", str(result.get("reason", "应用失败")))

    def start_tcal_od_capture(self) -> None:
        svc = self._require_tooling()
        self._run_in_calibration_mode(
            lambda: svc.start_capture(
                mode="od",
                rotation_speed_dps=self._float_var("tcal_rot_degps_var", 10.0),
                sampling_hz=self._float_var("tcal_hz_var", 20.0),
            )
        )
        self._set_var("tcal_msg_var", "OD 支撑采集中...")

    def stop_tcal_od_capture(self, reason: str = "manual") -> None:
        svc = self._require_tooling()
        self._run_in_calibration_mode(lambda: svc.stop_capture(reason))
        self._set_var("tcal_msg_var", reason or "已停止")

    def compute_tcal_od_psi(self) -> None:
        svc = self._require_tooling()
        result = svc.compute_od_psi()
        if result.get("ok"):
            self._set_var("tcal_od_psi_var", f"{float(result['psi_deg']):+.3f}")
            note = "" if result.get("has_reference") else "(无角向基准→0)"
            self._set_var("tcal_msg_var", f"OD ψ 计算完成 {note}")
        else:
            self._set_var("tcal_msg_var", str(result.get("reason", "计算失败")))

    def apply_tcal_od_psi(self) -> None:
        svc = self._require_tooling()
        result = svc.apply_od_psi()
        if result.get("ok"):
            self._set_var("tcal_msg_var", "OD ψ 已应用并保存")
            self.reload_tooling()
        else:
            self._set_var("tcal_msg_var", str(result.get("reason", "应用失败")))

    def run_tcal_selftest(self) -> None:
        svc = self._require_tooling()
        report = svc.run_selftest()
        passed = sum(1 for c in report.get("checks", []) if c.get("passed"))
        total = len(report.get("checks", []))
        ok = bool(report.get("ok"))
        details = "; ".join(
            f"{c.get('name')}:{'OK' if c.get('passed') else 'NG'}" for c in report.get("checks", [])
        )
        self._set_var("tcal_selftest_var", f"{'全部通过' if ok else '存在失败'} ({passed}/{total})  {details}")

    # -- OD zero (Phase 1) --------------------------------------------------

    def compute_tcal_od_zero(self) -> None:
        svc = self._require_tooling()
        result = svc.compute_od_zero(known_od=self._float_var("tcal_known_od_var", 190.0))
        if result.get("ok"):
            self._set_var("tcal_od_b_var", f"{float(result['od_b']):+.4f}")
            self._set_var("tcal_msg_var", "OD 零位计算完成,可应用")
        else:
            self._set_var("tcal_msg_var", str(result.get("reason", "计算失败")))

    def apply_tcal_od_zero(self) -> None:
        svc = self._require_tooling()
        result = svc.apply_od_zero()
        self._set_var("tcal_msg_var", "OD 零位已应用并保存" if result.get("ok") else str(result.get("reason", "应用失败")))
        if result.get("ok"):
            self.reload_tooling()

    def capture_tcal_od_reference(self) -> None:
        svc = self._require_tooling()
        result = svc.capture_od_reference()
        self._set_var("tcal_msg_var", f"已存基准廓线({result.get('n')}点)" if result.get("ok") else str(result.get("reason", "失败")))

    # -- spindle axis + chuck (Phase 0) -------------------------------------

    def start_tcal_delta_capture(self) -> None:
        svc = self._require_tooling()
        self._run_in_calibration_mode(
            lambda: svc.start_capture(
                mode="delta",
                rotation_speed_dps=self._float_var("tcal_rot_degps_var", 10.0),
                sampling_hz=self._float_var("tcal_hz_var", 20.0),
            )
        )
        self._set_var("tcal_msg_var", "同测内外采集中...")

    def stop_tcal_delta_capture(self, reason: str = "manual") -> None:
        svc = self._require_tooling()
        self._run_in_calibration_mode(lambda: svc.stop_capture(reason))
        self._set_var("tcal_msg_var", reason or "已停止")

    def record_tcal_axis_low(self) -> None:
        svc = self._require_tooling()
        result = svc.record_axis_station(z=self._float_var("tcal_axis_z_low_var", 0.0))
        self._set_var("tcal_msg_var", f"已记低位站点(n={result.get('n_stations')})" if result.get("ok") else str(result.get("reason", "失败")))

    def record_tcal_axis_high(self) -> None:
        svc = self._require_tooling()
        result = svc.record_axis_station(z=self._float_var("tcal_axis_z_high_var", 1700.0))
        self._set_var("tcal_msg_var", f"已记高位站点(n={result.get('n_stations')})" if result.get("ok") else str(result.get("reason", "失败")))

    def compute_tcal_axis(self) -> None:
        svc = self._require_tooling()
        result = svc.compute_axis()
        if result.get("ok"):
            self._set_var("tcal_axis_slope_var", f"x={float(result['axis_slope_x']):+.2e} y={float(result['axis_slope_y']):+.2e}")
            self._set_var("tcal_msg_var", "回转轴直线计算完成")
        else:
            self._set_var("tcal_msg_var", str(result.get("reason", "计算失败")))

    def apply_tcal_axis(self) -> None:
        svc = self._require_tooling()
        result = svc.apply_axis()
        self._set_var("tcal_msg_var", "回转轴已应用并保存" if result.get("ok") else str(result.get("reason", "应用失败")))
        if result.get("ok"):
            self.reload_tooling()

    def compute_tcal_chuck(self) -> None:
        svc = self._require_tooling()
        result = svc.compute_chuck_bound(cert_roundness=self._float_var("tcal_cert_round_var", 0.0))
        if result.get("ok"):
            over = bool(result.get("over_budget"))
            self._set_var("tcal_chuck_var", f"{float(result['chuck_error_bound']) * 1000:.1f} µm{' ⚠超35µm' if over else ''}")
            self._set_var("tcal_msg_var", "卡盘误差定界完成" + ("(超预算!)" if over else ""))
        else:
            self._set_var("tcal_msg_var", str(result.get("reason", "计算失败")))

    def apply_tcal_chuck(self) -> None:
        svc = self._require_tooling()
        result = svc.apply_chuck_bound()
        self._set_var("tcal_msg_var", "卡盘误差已应用并保存" if result.get("ok") else str(result.get("reason", "应用失败")))
        if result.get("ok"):
            self.reload_tooling()

    # -- cross-registration Δ_reg (Phase 4) ---------------------------------

    def compute_tcal_delta_reg(self) -> None:
        svc = self._require_tooling()
        result = svc.compute_delta_reg()
        if result.get("ok"):
            d = result["delta_reg"]
            self._set_var("tcal_delta_reg_var", f"({float(d[0]):+.4f}, {float(d[1]):+.4f})")
            self._set_var("tcal_msg_var", "Δ_reg 计算完成,可应用")
        else:
            self._set_var("tcal_msg_var", str(result.get("reason", "计算失败")))

    def apply_tcal_delta_reg(self) -> None:
        svc = self._require_tooling()
        result = svc.apply_delta_reg()
        self._set_var("tcal_msg_var", "Δ_reg 已应用并保存" if result.get("ok") else str(result.get("reason", "应用失败")))
        if result.get("ok"):
            self.reload_tooling()

    def reload_tooling(self) -> None:
        svc = self._require_tooling()
        data = svc.load_active()
        from domain.geometry_calibration import ToolingCalibration

        tc = ToolingCalibration.from_dict(data)
        any_cal = tc.id_calibrated() or tc.od_calibrated() or tc.axis_calibrated() or tc.cross_calibrated()
        self._set_var("tcal_status_var", "已标定" if any_cal else "未标定")
        self._set_var("tcal_id_Deff_var", f"{tc.id_D_eff:.3f}" if tc.id_calibrated() else "--")
        self._set_var("tcal_id_s_active_var", f"{tc.id_s_lateral:+.4f}" if tc.id_calibrated() else "--")
        self._set_var("tcal_od_psi_active_var", f"{tc.od_psi_deg:+.3f}" if tc.od_calibrated() else "--")
        self._set_var("tcal_od_b_active_var", f"{tc.od_b:+.4f}" if tc.od_calibrated() else "--")
        self._set_var("tcal_axis_active_var", "已标" if tc.axis_calibrated() else "--")
        self._set_var(
            "tcal_chuck_active_var",
            (f"{float(tc.chuck_error_bound) * 1000:.1f} µm" if tc.chuck_error_bound is not None else "--"),
        )
        self._set_var("tcal_delta_active_var2", "已标" if tc.cross_calibrated() else "--")

    def clear_tooling(self) -> None:
        svc = self._require_tooling()
        svc.clear_all()
        for name in ("tcal_id_Deff_var", "tcal_id_s_active_var", "tcal_od_psi_active_var",
                     "tcal_od_b_active_var", "tcal_axis_active_var", "tcal_chuck_active_var",
                     "tcal_delta_active_var2"):
            self._set_var(name, "--")
        self._set_var("tcal_status_var", "未标定")
        self._set_var("tcal_msg_var", "已清除工装标定")

    def verify_id_calibration(self) -> None:
        if self.id_service is None:
            raise RuntimeError("IdCalibrationService not injected")
        active: dict[str, Any] = {}
        try:
            active = self.id_service.load_active()
        except Exception:
            active = {}
        delta = self._optional_float(active.get("delta_c_mm"))
        if delta is None:
            delta = self._optional_float(self._var("idcal_delta_active_var", None))
        dref = self._optional_float(active.get("D_ref"))
        if dref is None:
            dref = self._float_var("idcal_dref_var", 150.0)
        if delta is None:
            self._run_in_calibration_mode(
                lambda: (
                    self._set_var("idcal_state_var", "ERR"),
                    self._set_var("idcal_msg_var", "复核失败：未找到 δc_active（请先“应用”）"),
                )
            )
            return
        settings = self._id_settings_from_host()
        self._set_var("idcal_chk_err_var", "--")
        self._set_var("idcal_chk_cov_var", "--")
        self._set_var("idcal_chk_n_var", "--")
        self._set_var("idcal_chk_dtheta_var", "--")
        self._set_var("idcal_state_var", "CHK")
        self._set_var("idcal_msg_var", "复核采集中...")
        self._run_in_calibration_mode(
            lambda: self.id_service.start_verify_capture(
                rotation_speed_dps=settings.rotation_speed_dps,
                sampling_hz=settings.sampling_hz,
                capture_duration_s=settings.capture_duration_s,
                delta_c_mm=float(delta),
                reference_diameter_mm=float(dref),
            )
        )

    def start_id_single_capture(self) -> None:
        if self.id_single_service is None:
            raise RuntimeError("IdSingleCalibrationService not injected")
        self.start_id_single_capture_new(self._id_single_settings_from_host())
        self._set_var("id_single_cal_state_var", "CAPTURING")
        self._set_var("id_single_cal_msg_var", "采集中...")

    def stop_id_single_capture(self, reason: str = "manual") -> None:
        if self.id_single_service is None:
            raise RuntimeError("IdSingleCalibrationService not injected")
        self.stop_id_single_capture_new(reason)
        self._set_var("id_single_cal_state_var", "STOP")
        self._set_var("id_single_cal_msg_var", reason or "已停止")

    def clear_id_single_capture(self) -> None:
        if self.id_single_service is None:
            raise RuntimeError("IdSingleCalibrationService not injected")
        self._run_in_calibration_mode(lambda: self.id_single_service.clear_capture())
        self._set_var("id_single_cal_state_var", "IDLE")
        self._set_var("id_single_cal_msg_var", "已清空")
        self._set_var("id_single_cal_mean_var", "--")
        self._set_var("id_single_cal_B_var", "--")
        self._set_var("id_single_cal_cov_var", "--")

    def compute_and_write_id_single_calibration(self) -> None:
        if self.id_single_service is None:
            raise RuntimeError("IdSingleCalibrationService not injected")
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


__all__ = ["CalibrationAction", "CalibrationController"]
