from __future__ import annotations

"""自动测量流程核心模块。

ExecutorCoreMixin with lifecycle methods and AutoFlow class composition.
"""

import math
import threading
import time
from typing import Any, List, Tuple, TYPE_CHECKING

import numpy as np


from config.addresses import (
    CMD_EN_REQ,
    CMD_MOVEA_REQ,
    CMD_VELMOVE_REQ,
    OFF_POS_MOVEA,
    OFF_VEL_VELMOVE,
)
from domain.state import CalibrationSnapshot
from core.models import MeasureRow
from domain.sampling import (
    _robust_span,
    _split_slip_diag,
)
from frp_workflow.executor._executor_helpers import (
    log,
    log_exc,
    perf_logger,
    logger,
)

if TYPE_CHECKING:  # pragma: no cover
    from app import App


# =========================
# Module-level helpers (extracted from run() nested functions)
# =========================


def _pp_strict(a: "np.ndarray") -> float:
    """Strict peak-to-peak span."""
    return float(_robust_span(a, "strict"))


def _pp_robust(a: "np.ndarray", pp_mode: str = "robust") -> float:
    """Robust peak-to-peak/span.

    Compatibility: some older call-sites pass `trim_ratio=`.
    Robustness is controlled by recipe.pp_mode, so we ignore
    extra keywords.
    """
    return float(_robust_span(a, pp_mode))


def _fit_line_and_dist(points_xyz: List[Tuple[float, float, float]]):
    """Fit a 3D line to points via PCA.

    Returns:
        straight: float, (max(dist)-min(dist)) to the fitted line
        dist_list: per-point distance to the fitted line
        p0: a point on the fitted line (mean)
        d: direction vector (unit)
    """
    if len(points_xyz) < 2:
        # Degenerate: return a default Z-axis line so downstream distance
        # computations won't crash.
        return 0.0, [0.0 for _ in points_xyz], np.zeros(3, dtype=float), np.array([0.0, 0.0, 1.0], dtype=float)
    P = np.array(points_xyz, dtype=float)
    p0 = P.mean(axis=0)
    Q = P - p0
    C = (Q.T @ Q) / max(1, Q.shape[0])
    w, v = np.linalg.eigh(C)
    d = v[:, int(np.argmax(w))]
    d = d / (np.linalg.norm(d) + 1e-12)
    t = (Q @ d)
    proj = np.outer(t, d)
    R = Q - proj
    dist = np.linalg.norm(R, axis=1)
    straight = float(dist.max() - dist.min()) if dist.size else 0.0
    return straight, [float(x) for x in dist.tolist()], p0, d


def _line_distance(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> float:
    """Minimum distance between two 3D lines.

    Line1: p1 + t*d1; Line2: p2 + s*d2
    """
    d1n = d1 / (np.linalg.norm(d1) + 1e-12)
    d2n = d2 / (np.linalg.norm(d2) + 1e-12)
    n = np.cross(d1n, d2n)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        # Parallel (or nearly): distance from (p2-p1) to line1
        v = (p2 - p1)
        return float(np.linalg.norm(np.cross(v, d1n)))
    return float(abs(np.dot((p2 - p1), n)) / nn)


def _tilt_and_end_offset(p0: np.ndarray, d: np.ndarray, pts_xyz: List[Tuple[float, float, float]]):
    """Compute axis-line tilt (deg) and end-point offset (mm) along Z span.

    Coordinate frame: (x,y,z) where z is UI Z position.
    Tilt is relative to +Z (rotation axis ideal direction).
    End offset is the XY distance between fitted line points at z_min and z_max.
    """
    try:
        if not pts_xyz or len(pts_xyz) < 2:
            return None, None, None
        z_list = [float(p[2]) for p in pts_xyz]
        z_min = float(min(z_list))
        z_max = float(max(z_list))
        dz = float(d[2])
        if abs(dz) < 1e-12:
            return None, None, None
        sx = float(d[0] / dz)
        sy = float(d[1] / dz)
        slope = float(math.hypot(sx, sy))  # mm/mm
        tilt_deg = float(math.degrees(math.atan(slope)))

        t_min = (z_min - float(p0[2])) / dz
        t_max = (z_max - float(p0[2])) / dz
        p_min = p0 + t_min * d
        p_max = p0 + t_max * d
        end_off = float(math.hypot(float(p_max[0] - p_min[0]), float(p_max[1] - p_min[1])))
        return tilt_deg, end_off, slope
    except Exception:
        return None, None, None


class ExecutorCoreMixin:
    """Mixin providing core lifecycle and the master run() orchestrator.

    Expects the following attributes/methods on ``self``:
        app: Any
        device: Any
        stop_event: threading.Event
    """

    app: Any
    device: Any
    stop_event: threading.Event
    _current_recipe: Any
    _calibration_snapshot: Any
    _last_sample_cov: Any
    _last_sample_reason: Any
    _last_sample_max_gap_deg: Any
    _last_fit_weights_od: Any
    _last_fit_weights_id: Any
    _last_sample_n_od: Any
    _last_sample_n_id: Any
    _last_sample_n_od_pass: Any
    _last_sample_n_id_pass: Any
    _last_sample_cov_id: Any
    _last_sample_reason_id: Any
    _last_sample_max_gap_deg_id: Any
    _last_sample_debug: Any

    def __init__(self, app: "App", *, device=None):
        super().__init__(daemon=True)
        self.app = app
        if device is not None:
            self.device = device
        else:
            raise TypeError(
                "AutoFlow requires a gateway object via device=... parameter. "
                "Pass an AppDeviceGateway or compatible implementation."
            )
        self.stop_event = threading.Event()
        self._current_recipe = None
        self._calibration_snapshot: CalibrationSnapshot | None = None
        self._last_sample_cov = (0, 0, 0)
        self._last_sample_reason = ("-", 0.0, 0.0)
        self._last_sample_max_gap_deg = None
        self._last_fit_weights_od = None
        self._last_fit_weights_id = None

    def start(self):
        try:
            logger.debug("AUTOFLOW_THREAD_START_REQUEST")
            self.app._log_ax3_speed_trace("autoflow_start_entry")
        except Exception:
            pass
        super().start()

    def stop(self):
        self.stop_event.set()

    def _should_stop(self) -> bool:
        """Return True if AutoFlow should stop.

        Stop conditions:
        - UI stop button (stop_event)
        - X0 (NC E-STOP) opened => read as 0

        When X0 triggers, stop_event will be latched to prevent the flow from resuming.
        """
        if self.stop_event.is_set():
            return True
        try:
            # X0 is NC: 1 = healthy, 0 = E-STOP pressed / opened
            if int(self.app.get_x_point(0)) == 0:
                try:
                    self.stop_event.set()
                except Exception:
                    pass
                return True
        except Exception:
                pass
        return False

    def _sleep_cancelable(self, seconds: float, poll_s: float = 0.05) -> bool:
        end_t = time.time() + max(0.0, float(seconds))
        while time.time() < end_t:
            if self._should_stop():
                return False
            time.sleep(min(float(poll_s), max(0.0, end_t - time.time())))
        return not self._should_stop()

    def _emit_auto_state(self, state: str, msg: str) -> None:
        try:
            self.app.ui_q.put(("auto_state", {"state": str(state), "msg": str(msg)}))
        except Exception:
            pass

    def run(self):  # pyright: ignore[reportGeneralTypeIssues]
        try:
            self.app.ui_q.put(("auto_state", {"state": "RUN", "msg": "自动测量开始"}))
            try:
                r0 = self.app.get_recipe_copy()
                log("AUTO_FLOW_START", section_count=getattr(r0,"section_count",None), points_per_rev=getattr(r0,"points_per_rev",None), min_bin_coverage=getattr(r0,"min_bin_coverage",None), timeout_s=getattr(r0,"sample_timeout_s",None), max_revolutions=getattr(r0,"max_revolutions",None))
            except Exception as e:
                log("AUTO_FLOW_START", err=str(e))

            recipe = self.app.get_recipe_copy()
            self._current_recipe = recipe
            self._calibration_snapshot = self._get_calibration_snapshot(refresh=True)
            if recipe.section_count <= 0:
                raise ValueError("截面数量必须>0")

            # Ensure section_pos_z exists
            if len(getattr(recipe, "section_pos_z", []) or []) != recipe.section_count:
                recipe.section_pos_z = recipe.compute_default_positions_z()

            # AutoFlow f8: always use OD/ID group in Z_Pos coordinate
            cal = getattr(self.app, "axis_cal", None)
            if cal is None:
                raise RuntimeError("AxisCal 未加载：请先在“轴位标定”页读取标定参数")

            ax_od = 0
            ax_id1 = 1
            ax_id4 = 4

            # Pre-check + enable OD/ID axes
            for ax in (ax_od, ax_id1, ax_id4):
                ac = self.device.get_axis_copy(ax)
                if self._is_fault(int(ac.sts), int(ac.err)):
                    raise RuntimeError(f"轴 AX{ax} 故障，Err={int(ac.err)}")
                if not self._is_enabled(int(ac.sts)):
                    self.app.set_cmd_bits(ax, set_mask=CMD_EN_REQ, clr_mask=0)
                    time.sleep(0.15)



            # ---------------------
            # S20/S50: Clamp outputs + AX2 positioning + operator confirm (no clamp feedback)
            # ---------------------
            ax_clamp = 2  # AX2 center clamp
            try:
                ac2 = self.device.get_axis_copy(ax_clamp)
                if self._is_fault(int(ac2.sts), int(ac2.err)):
                    raise RuntimeError(f"中心架 AX2 故障，Err={int(ac2.err)}")
                if not self._is_enabled(int(ac2.sts)):
                    self.app.set_cmd_bits(ax_clamp, set_mask=CMD_EN_REQ, clr_mask=0)
                    time.sleep(0.15)
            except Exception as e:
                raise RuntimeError(f"中心架 AX2 使能失败：{e}")

            if not self._prepare_clamps_for_auto(recipe):
                return

            # Optional: move AX2 to length measurement position
            if bool(getattr(recipe, 'len_enable', False)):
                if bool(getattr(recipe, 'ax2_len_valid', False)):
                    try:
                        tgt2 = float(getattr(recipe, 'ax2_len_abs', 0.0))
                        tgt2 = self.device.apply_soft_limits_abs(ax_clamp, tgt2, strict=True, context='AUTO_AX2_LEN')
                        self.app.ui_q.put(("auto_state", {"state": "PREP", "msg": f"中心架到长度测量位：{tgt2:.3f}"}))
                        self._write_fp64(ax_clamp, OFF_POS_MOVEA, float(tgt2))
                        self._ensure_movea_setpoints(ax_clamp)
                        self.app._pulse_cmd_bits(ax_clamp, CMD_MOVEA_REQ)
                        ok2 = self._wait_in_position(ax_clamp, float(tgt2), pos_tol=0.05, timeout_s=25.0)
                        if not ok2:
                            raise TimeoutError(f"AX2 到位超时（目标 {tgt2:.3f}）")
                    except Exception as e:
                        # Length step is optional; do not stop AutoFlow here.
                        try:
                            self.app.ui_q.put(("auto_state", {"state": "WARN", "msg": f"AX2 长度位定位失败：{e}"}))
                        except Exception:
                            pass
                else:
                    try:
                        self.app.ui_q.put(("auto_state", {"state": "WARN", "msg": "长度检测已启用，但未保存 AX2 长度测量位（ax2_len_valid=0）"}))
                    except Exception:
                        pass

            # S30: auto length measurement (optional; failures must not stop the flow)
            if bool(getattr(recipe, 'len_enable', False)):
                if not bool(getattr(recipe, 'ax2_len_valid', False)):
                    # safety: do not run length search if AX2 length position isn't defined
                    len_payload = {
                        "enabled": True,
                        "skipped": True,
                        "ok": False,
                        "reason": "NO_AX2_LEN_POS",
                        "z_low": None,
                        "z_high": None,
                        "length_mm": None,
                        "t_s": 0.0,
                    }
                else:
                    try:
                        self.app.ui_q.put(("auto_state", {"state": "LEN", "msg": "自动测量长度"}))
                    except Exception:
                        pass
                    try:
                        len_payload = self._auto_measure_length(recipe)
                    except Exception as e:
                        len_payload = {
                            "enabled": True,
                            "skipped": False,
                            "ok": False,
                            "reason": f"EXC({e})",
                            "z_low": None,
                            "z_high": None,
                            "length_mm": None,
                            "t_s": 0.0,
                        }

                # publish to UI and store to app run-context
                try:
                    self.app.ui_q.put(("auto_len", len_payload))
                except Exception:
                    pass
                try:
                    setattr(self.app, "_run_len_result", len_payload)
                except Exception:
                    pass

                # after length step, return AX0 to standby if standby positions saved
                if bool(getattr(recipe, 'standby_valid', False)):
                    try:
                        tgt0 = float(getattr(recipe, 'standby_ax0_abs', 0.0))
                        tgt0 = self.device.apply_soft_limits_abs(0, tgt0, strict=True, context='AUTO_AX0_STANDBY_AFTER_LEN')
                        self.app.ui_q.put(("auto_state", {"state": "PREP", "msg": f"AX0 回待机位：{tgt0:.3f}"}))
                        self._write_fp64(0, OFF_POS_MOVEA, float(tgt0))
                        self._ensure_movea_setpoints(0)
                        self.app._pulse_cmd_bits(0, CMD_MOVEA_REQ)
                        self._wait_in_position(0, float(tgt0), pos_tol=0.05, timeout_s=25.0)
                    except Exception as e:
                        try:
                            self.app.ui_q.put(("auto_state", {"state": "WARN", "msg": f"AX0 待机位定位失败：{e}"}))
                        except Exception:
                            pass

                if self._should_stop():
                    self.app.ui_q.put(("auto_state", {"state": "STOP", "msg": "用户停止"}))
                    return

            # Move AX2 only when length measurement is enabled. When disabled, AX2 is a safety check only.
            if not bool(getattr(recipe, 'len_enable', False)):
                if not self._verify_ax2_when_length_disabled(recipe, ax_clamp=ax_clamp):
                    return
            elif bool(getattr(recipe, 'ax2_rot_valid', False)):
                try:
                    tgt2r = float(getattr(recipe, 'ax2_rot_abs', 0.0))
                    tgt2r = self.device.apply_soft_limits_abs(ax_clamp, tgt2r, strict=True, context='AUTO_AX2_ROT')
                    self.app.ui_q.put(("auto_state", {"state": "PREP", "msg": f"中心架到旋转测量位：{tgt2r:.3f}"}))
                    self._write_fp64(ax_clamp, OFF_POS_MOVEA, float(tgt2r))
                    self._ensure_movea_setpoints(ax_clamp)
                    self.app._pulse_cmd_bits(ax_clamp, CMD_MOVEA_REQ)
                    ok2r = self._wait_in_position(ax_clamp, float(tgt2r), pos_tol=0.05, timeout_s=25.0)
                    if not ok2r:
                        raise TimeoutError(f"AX2 到位超时（目标 {tgt2r:.3f}）")
                except Exception as e:
                    raise RuntimeError(f"AX2 旋转位定位失败：{e}")
            else:
                raise RuntimeError("未保存 AX2 旋转测量位（ax2_rot_valid=0），无法开始旋转测量")

            # Prepare rotate axis (AX3): enable + ensure velmove params
            a3 = self.device.get_axis_copy(3)
            if self._is_fault(int(a3.sts), int(a3.err)):
                raise RuntimeError(f"旋转轴 AX3 故障，Err={int(a3.err)}")

            if not self._is_enabled(int(a3.sts)):
                self.app.set_cmd_bits(3, set_mask=CMD_EN_REQ, clr_mask=0)
                time.sleep(0.25)

            # Apply rotation speed from recipe every time (AX3 VelMove speed),
            # to make behavior deterministic and not rely on previous manual settings.
            try:
                target_vel = float(recipe.rot_vel_velmove)
            except Exception:
                target_vel = 0.0
            try:
                self._write_fp64(3, OFF_VEL_VELMOVE, float(target_vel))
            except Exception:
                pass

            self._ensure_velmove_setpoints(3)
            time.sleep(0.05)

            # start rotate (AX3) - level command
            try:
                self.app._log_ax3_speed_trace("autoflow_ax3_velmove_start_pre", recipe_obj=recipe)
            except Exception:
                pass
            self.app.set_cmd_bits(3, set_mask=CMD_VELMOVE_REQ, clr_mask=0)
            time.sleep(0.20)

            # Clear results first
            # NOTE: UI clear/run identity setup is handled before workflow start.
            # 这里再发一次 auto_clear 会把 _run_start_ts 置空，导致自动导出失败。
            # self.app.ui_q.put(("auto_clear", {"ts": time.time()}))

            # Move + sample per section
            # Use absolute fitted centers (same coordinate frame for OD/ID) so we can
            # compute both straightness/eccentricity and the distance between OD/ID axes.
            centers_xyz: List[Tuple[float, float, float]] = []      # (xc, yc, z)
            centers_xyz_id: List[Tuple[float, float, float]] = []   # (xci, yci, z)
            concentricity_list: List[float] = []          # per-section OD/ID concentricity

            for i in range(recipe.section_count):
                if self._should_stop():
                    self.app.ui_q.put(
                        ("auto_state", {"state": "STOP", "msg": "用户停止"})
                    )
                    return

                z_od_disp = float(recipe.section_pos_z[i])
                # Soft limits (abs) for target solving (OD clamp + ID split)
                softlims = {
                    0: (float(self.device.get_axis_copy(0).softlim_pos), float(self.device.get_axis_copy(0).softlim_neg)),
                    1: (float(self.device.get_axis_copy(1).softlim_pos), float(self.device.get_axis_copy(1).softlim_neg)),
                    4: (float(self.device.get_axis_copy(4).softlim_pos), float(self.device.get_axis_copy(4).softlim_neg)),
                }

                tg = cal.od_z_disp_to_targets(z_od_disp, softlims_abs=softlims)
                x_ui = float(z_od_disp)  # for UI payload compatibility
                x_abs = float(tg["ax0_abs"])  # AX0 target abs

                self.app.ui_q.put(
                    (
                        "auto_progress",
                        {
                            "idx": i,
                            "total": recipe.section_count,
                            "x_ui": x_ui,
                            "x_abs": x_abs,
                        },
                    )
                )

                # Motion: Fire all MoveA commands first (AX0/AX1/AX4 move simultaneously), then wait.
                targets = {
                    ax_id1: float(tg["ax1_abs"]),
                    ax_id4: float(tg["ax4_abs"]),
                    ax_od: float(tg["ax0_abs"]),
                }

                # Soft limits (absolute): prevent AutoFlow from driving linear axes beyond PLC soft limits.
                # strict=True will raise and stop AutoFlow if a target is out of range.
                for ax, tgt in list(targets.items()):
                    targets[ax] = self.device.apply_soft_limits_abs(
                        int(ax), float(tgt), strict=True, context=f"AUTO_SEC_{i+1}"
                    )

                try:
                    log("SECTION_START", section=i+1, z_disp=x_ui, ax0_abs=targets.get(ax_od), ax1_abs=targets.get(ax_id1), ax4_abs=targets.get(ax_id4))
                except Exception:
                    pass

                for ax, tgt in targets.items():
                    self._write_fp64(ax, OFF_POS_MOVEA, float(tgt))
                    self._ensure_movea_setpoints(ax)
                    self.app._pulse_cmd_bits(ax, CMD_MOVEA_REQ)

                for ax, tgt in targets.items():
                    ok = self._wait_in_position(ax, tgt, pos_tol=0.05, timeout_s=25.0)
                    if not ok:
                        if self._should_stop():
                            self.app.ui_q.put(("auto_state", {"state": "STOP", "msg": "用户停止"}))
                            return
                        raise TimeoutError(f"AX{ax} 到位超时（目标 {tgt:.3f}）")

                # Sampling (angle + OD/ID), circle fit
                scan_mode = str(getattr(recipe, "scan_mode", "SYNC") or "SYNC").strip().upper()

                split_shift_deg = None


                coax_unreliable = None


                # split-scan options


                keep_spinning = True


                slip_check = bool(getattr(recipe, 'split_slip_check', True))


                slip_max_deg = float(getattr(recipe, 'split_slip_max_deg', 5.0) or 5.0)


                omega_cv_max = float(getattr(recipe, 'split_omega_cv_max', 0.25) or 0.25)



                if scan_mode == "SPLIT":
                    # Pass-1: OD only
                    coords_od, _coords_id0, raw_od, _raw_id0, raw_points_od = self._sample_circle_points_dual(
                        recipe,
                        section_idx=i,
                        sample_od=True,
                        sample_id=False,
                        phase="OD",
                    )
                    cov_od = getattr(self, "_last_sample_cov", (0, 0, 0))
                    reason_od = getattr(self, "_last_sample_reason", ("-", 0.0, 0.0))
                    max_gap_od = getattr(self, "_last_sample_max_gap_deg", None)
                    w_od = getattr(self, "_last_fit_weights_od", None)

                    n_od_pass = getattr(self, "_last_sample_n_od", None)
                    # If configured, stop rotate axis between OD/ID passes.
                    # NOTE: keep_spinning=False is less reliable for coax metrics; slip_check will likely flag it.
                    if not keep_spinning:
                        try:
                            # Clear level velmove and request stop pulse.
                            try:
                                self.app._log_ax3_speed_trace("autoflow_ax3_velmove_stop_pre", recipe_obj=recipe)
                            except Exception:
                                pass
                            self.device.stop(3)
                            t_stop0 = time.time()
                            while (time.time() - t_stop0) < 10.0:
                                if self._should_stop():
                                    break
                                ac3s = self.device.get_axis_copy(3)
                                if not self._is_moving(int(getattr(ac3s, 'sts', 0))):
                                    break
                                time.sleep(0.06)
                        except Exception:
                            pass
                        try:
                            # Restart rotation with recipe speed.
                            try:
                                rot_v2 = float(getattr(recipe, 'rot_vel_velmove', getattr(recipe, 'rot_speed', 200.0)) or 0.0)
                            except Exception:
                                rot_v2 = 200.0
                            if abs(rot_v2) <= 1e-9:
                                rot_v2 = 200.0
                            try:
                                self._write_fp64(3, OFF_VEL_VELMOVE, float(rot_v2))
                            except Exception:
                                pass
                            self._ensure_velmove_setpoints(3)
                            time.sleep(0.05)
                            try:
                                self.app._log_ax3_speed_trace("autoflow_ax3_velmove_restart_pre", recipe_obj=recipe)
                            except Exception:
                                pass
                            self.app.set_cmd_bits(3, set_mask=CMD_VELMOVE_REQ, clr_mask=0)
                            time.sleep(0.20)
                        except Exception:
                            pass

                    # Pass-2: ID only
                    _coords_od0, coords_id, _raw_od0, raw_id, raw_points_id = self._sample_circle_points_dual(
                        recipe,
                        section_idx=i,
                        sample_od=False,
                        sample_id=True,
                        phase="ID",
                    )
                    cov_id = getattr(self, "_last_sample_cov", (0, 0, 0))
                    reason_id = getattr(self, "_last_sample_reason", ("-", 0.0, 0.0))
                    max_gap_id = getattr(self, "_last_sample_max_gap_deg", None)
                    w_id = getattr(self, "_last_fit_weights_id", None)

                    n_id_pass = getattr(self, "_last_sample_n_id", None)
                    # Split-scan diagnostics: lightweight slip / speed stability check.
                    if slip_check:
                        try:
                            split_shift_deg, coax_unreliable = _split_slip_diag(
                                raw_points_od=raw_points_od,
                                raw_points_id=raw_points_id,
                                slip_max_deg=float(slip_max_deg),
                                omega_cv_max=float(omega_cv_max),
                            )
                        except Exception:
                            split_shift_deg, coax_unreliable = None, None

                    # Merge raw points (keep phase) and restore per-channel weights.
                    raw_points = list(raw_points_od or []) + list(raw_points_id or [])
                    self._last_fit_weights_od = w_od
                    self._last_fit_weights_id = w_id

                    # For backward compatible UI/export coverage columns, report OD pass as the main cov.
                    self._last_sample_cov = cov_od
                    self._last_sample_reason = reason_od
                    self._last_sample_max_gap_deg = max_gap_od

                    # Keep a copy of ID pass stats for diagnostics (UI may ignore extra keys).
                    self._last_sample_cov_id = cov_id
                    try:
                        self._last_sample_n_od_pass = n_od_pass
                    except Exception:
                        self._last_sample_n_od_pass = None
                    try:
                        self._last_sample_n_id_pass = n_id_pass
                    except Exception:
                        self._last_sample_n_id_pass = None
                    self._last_sample_reason_id = reason_id
                    self._last_sample_max_gap_deg_id = max_gap_id

                else:
                    coords_od, coords_id, raw_od, raw_id, raw_points = self._sample_circle_points_dual(
                        recipe,
                        section_idx=i,
                        sample_od=True,
                        sample_id=True,
                        phase="SYNC",
                    )
                # Attach section metadata for export
                try:
                    for j, p in enumerate(raw_points):
                        if isinstance(p, dict):
                            p["section_idx"] = int(i + 1)
                            p["z_pos_mm"] = float(z_od_disp)
                            p["sample_idx"] = int(j)
                except Exception:
                    pass
                try:
                    self.app.ui_q.put(("auto_raw_points", {"points": raw_points}))
                except Exception:
                    pass


                try:
                    n_total, n_hit, n_miss = getattr(self, "_last_sample_cov", (0, 0, 0))
                    cov = (float(n_hit) / float(n_total)) if n_total else None
                    reason, revs, elapsed = getattr(self, "_last_sample_reason", ("-", 0.0, 0.0))

                    payload = {
                        "idx": i + 1,
                        "cov": cov,
                        "cov_od": cov,
                        "n_od": getattr(self, "_last_sample_n_od", None),
                        "n_id": getattr(self, "_last_sample_n_id", None),
                        "miss": n_miss,
                        "max_gap_deg": getattr(self, "_last_sample_max_gap_deg", None),
                        "reason": reason,
                        "revs": revs,
                        "elapsed": elapsed,
                    }

                    # Optional: in SPLIT mode, also attach ID-pass coverage stats for diagnostics.
                    if scan_mode == "SPLIT":
                        n_total_i, n_hit_i, n_miss_i = getattr(self, "_last_sample_cov_id", (0, 0, 0))
                        cov_i = (float(n_hit_i) / float(n_total_i)) if n_total_i else None
                        reason_i, revs_i, elapsed_i = getattr(self, "_last_sample_reason_id", ("-", 0.0, 0.0))
                        payload.update({
                            "cov_id": cov_i,
                            "n_od": n_od_pass,
                            "n_id": n_id_pass,
                            "miss_id": n_miss_i,
                            "max_gap_deg_id": getattr(self, "_last_sample_max_gap_deg_id", None),
                            "reason_id": reason_i,
                            "revs_id": revs_i,
                            "elapsed_id": elapsed_i,
                        })
                        # Attach split diagnostics (may be None).
                        payload.update({
                            "split_shift_deg": split_shift_deg,
                            "coax_unreliable": coax_unreliable,
                            "keep_spinning": keep_spinning,
                        })

                    # Attach 1-based section index so UI can cache per-section coverage.
                    self.app.ui_q.put(("auto_cov", payload))
                except Exception:
                    pass
                try:
                    id_single_enable = bool(getattr(recipe, "id_single_enable", False))
                except Exception:
                    id_single_enable = False

                try:
                    raw_total = int(len(raw_points or []))
                    od_raw_in = int(
                        sum(1 for _p in (raw_points or []) if isinstance(_p, dict) and _p.get("od_mm", None) is not None)
                    )
                    if id_single_enable:
                        id_raw_in = int(
                            sum(
                                1
                                for _p in (raw_points or [])
                                if isinstance(_p, dict) and _p.get("id_out2_mm", None) is not None
                            )
                        )
                    else:
                        id_raw_in = int(
                            sum(1 for _p in (raw_points or []) if isinstance(_p, dict) and _p.get("id_mm", None) is not None)
                        )
                    od_fit_in = int(len(coords_od))
                    id_fit_in = 0 if id_single_enable else int(len(coords_id))
                    perf_logger.info(
                        "[FIT_INPUT] section=%d scan_mode=%s raw_total=%d od_raw_in=%d id_raw_in=%d od_fit_in=%d id_fit_in=%d calc_input_mode=%s fit_strategy=%s",
                        int(i + 1),
                        str(scan_mode),
                        int(raw_total),
                        int(od_raw_in),
                        int(id_raw_in),
                        int(od_fit_in),
                        int(id_fit_in),
                        str(getattr(recipe, "calc_input_mode", "bin")),
                        str(getattr(recipe, "fit_strategy", "")),
                    )
                except Exception:
                    pass

                xc, yc, _r_fit, _sigma = self._fit_circle(coords_od, weights=getattr(self, "_last_fit_weights_od", None))
                xci = yci = _r_fit_i = _sigma_i = 0.0
                if not id_single_enable:
                    xci, yci, _r_fit_i, _sigma_i = self._fit_circle(
                        coords_id, weights=getattr(self, "_last_fit_weights_id", None)
                    )

                # For axis-line fitting (straightness/tilt/end-offset), we want the *center offset vector* (xc,yc)
                # relative to the rotation axis as a function of axial position.
                # - Old OD algorithm: we approximate center offset by circle-fit center (xc,yc) from coords_od.
                # - New OD algorithm (edge distances): coords_od is synthesized as (r*cosθ,r*sinθ) about origin,
                #   so circle-fit center is ~0 and would erase eccentricity. In that mode we must use the fitted
                #   delta(θ)=a*cosθ+b*sinθ+c coefficients: (a,b) is the center offset (ex,ey).
                center_od_x = float(xc)
                center_od_y = float(yc)
                od_ex = None
                od_ey = None

                # Radial runout w.r.t rotation axis (origin): peak-to-peak of radius (mm)
                # NOTE: Use a trimmed peak-to-peak (drop a small fraction of extremes) to avoid
                # inflating runout from occasional serial glitches/outliers.
                # Robust span strategy for runout / peak-to-peak
                # (pp_mode is read per-call from recipe)

                # OD/ID runout (diameter peak-to-peak, mm): computed from raw samples (od_mm/id_mm),
                # so that section_results matches raw_points verification (max-min of od_mm for the section).
                # Use a trimmed peak-to-peak to reduce the influence of rare outliers.
                try:
                    od_vals = np.asarray([float(p.get("od_mm")) for p in raw_points if p.get("od_mm") is not None], dtype=float)
                except Exception:
                    od_vals = np.asarray([], dtype=float)
                od_pp_mm = _pp_strict(od_vals)
                od_pp_rob_mm = _pp_robust(od_vals)

                # Backward-compat: od_runout is the (robust) diameter peak-to-peak of raw od_mm series
                od_runout = float(od_pp_rob_mm)

                if not id_single_enable:
                    try:
                        id_vals = np.asarray([float(p.get("id_mm")) for p in raw_points if p.get("id_mm") is not None], dtype=float)
                    except Exception:
                        id_vals = np.asarray([], dtype=float)
                    id_pp_mm = _pp_strict(id_vals)
                    id_pp_rob_mm = _pp_robust(id_vals)
                else:
                    id_vals = np.asarray([], dtype=float)
                    id_pp_mm = 0.0
                    id_pp_rob_mm = 0.0

                # ID new algorithm: fit from chord OUT4 (id_c_mm) + m OUT5 (id_m_mm), then reconstruct diameter series.
                id_fit = None
                id_fit_diam = None
                id_fit_vals = None
                if (not id_single_enable) and bool(getattr(recipe, "id_use_fit", False)) and (not getattr(self.app, "sim_disp_enabled", False)):
                    delta_c = float(self._idcal_get_delta_c_active())
                    id_fit, id_fit_vals = self._id_fit_from_raw_points(
                        raw_points,
                        delta_c,
                        theta_delay_s=float(getattr(recipe, 'theta_delay_s', 0.0) or 0.0),
                    )

                    if id_fit is not None:
                        try:
                            id_fit_diam = float(id_fit.get("diam", None))
                        except Exception:
                            id_fit_diam = None

                    if id_fit_vals is None:
                        # fallback: use corrected chord c as proxy series
                        try:
                            c_list = [float(p.get("id_c_mm")) for p in raw_points if p.get("id_c_mm") is not None]
                            if c_list:
                                id_fit_vals = np.asarray(c_list, dtype=float) + float(delta_c)
                        except Exception:
                            id_fit_vals = None

                    if id_fit_vals is not None and getattr(id_fit_vals, "size", 0) >= 2:
                        id_pp_mm = _pp_strict(np.asarray(id_fit_vals, dtype=float))
                        id_pp_rob_mm = _pp_robust(np.asarray(id_fit_vals, dtype=float))
                        id_runout = float(id_pp_rob_mm)
                    else:
                        id_pp_mm = _pp_strict(id_vals)
                        id_pp_rob_mm = _pp_robust(id_vals)
                        id_runout = float(id_pp_rob_mm)
                elif not id_single_enable:
                    id_runout = _pp_robust(id_vals)
                else:
                    id_runout = 0.0
                # OD diameter stats
                od_use_edges = bool(getattr(recipe, "od_use_edges", False))

                # Compatibility path: derive OD stats from fitted circle (coords_od)
                dx = coords_od[:, 0] - float(xc)
                dy = coords_od[:, 1] - float(yc)
                r_list = np.sqrt(dx * dx + dy * dy)
                od_list = 2.0 * r_list

                if od_use_edges and od_vals.size:
                    # New OD algorithm (edge distances): od_mm already computed as B-(L+R) in raw_points.
                    od_avg = float(np.mean(od_vals))
                    # OD diameter peak-to-peak within section (trimmed): used as OD_d_pp
                    od_round = _pp_robust(od_vals)

                    # OD eccentricity amplitude (mm) and phase angle (deg):
                    # Fit delta(theta)=a*cosθ+b*sinθ+c where delta=(L-R)/2.
                    # Then e = hypot(a,b), phi = atan2(b,a).
                    od_e = 0.0
                    od_phi_deg: float | None = None
                    try:
                        deltas_list = []
                        th_list = []
                        for p in raw_points:
                            d = p.get("od_delta") if isinstance(p, dict) else None
                            t = p.get("theta_deg") if isinstance(p, dict) else None
                            if d is None or t is None:
                                continue
                            deltas_list.append(float(d))
                            th_list.append(float(t))
                        deltas = np.asarray(deltas_list, dtype=float)
                        th_deg = np.asarray(th_list, dtype=float)
                        if deltas.size >= 3:
                            th = np.deg2rad(th_deg)
                            A = np.stack([np.cos(th), np.sin(th), np.ones_like(th)], axis=1)
                            coef, *_ = np.linalg.lstsq(A, deltas, rcond=None)
                            a, b, _c = [float(x) for x in coef]
                            od_ex, od_ey = float(a), float(b)
                            od_e = float(math.hypot(a, b))
                            try:
                                od_phi_deg = float(np.rad2deg(math.atan2(b, a)))
                                # normalize to (-180, 180]
                                if od_phi_deg <= -180.0:
                                    od_phi_deg += 360.0
                                elif od_phi_deg > 180.0:
                                    od_phi_deg -= 360.0
                            except Exception:
                                od_phi_deg = None
                    except Exception:
                        od_e = 0.0
                        od_phi_deg = None

                    # For new algorithm, interpret od_runout as radial runout (diameter peak-to-peak) = 2*e
                    od_runout = float(2.0 * od_e)

                    # Use fitted (ex,ey) as OD center offset for axis-line fit.
                    if (od_ex is not None) and (od_ey is not None):
                        center_od_x = float(od_ex)
                        center_od_y = float(od_ey)
                else:
                    # Old algorithm: od_runout is diameter peak-to-peak from od_vals (already computed above)
                    od_avg = float(np.mean(od_list)) if od_list.size else 0.0
                    od_round = float(np.max(od_list) - np.min(od_list)) if od_list.size >= 2 else 0.0
                    od_e = 0.0
                    od_phi_deg = None

                od_dev = float(od_avg) - float(recipe.od_std_mm)

                # OD roundness by fit residual (diameter mm). Export-only in f9_7_1.
                od_round_fit_mm = None
                od_round_fit_rob_mm = None
                try:
                    od_round_fit_mm, od_round_fit_rob_mm = self._od_round_fit_from_raw_points(
                        raw_points,
                        calc_input_mode=str(getattr(recipe, 'calc_input_mode', 'bin')),
                        bin_count=int(getattr(recipe, 'bin_count', 90)),
                        bin_method=str(getattr(recipe, 'bin_method', 'median')),
                        pp_mode=str(getattr(recipe, 'pp_mode', 'p99_p1')),
                        theta_delay_s=float(getattr(recipe, 'theta_delay_s', 0.0) or 0.0),
                    )
                except Exception:
                    od_round_fit_mm, od_round_fit_rob_mm = None, None

                # ID roundness by fit residual (diameter mm).
                id_round_fit_mm = None
                id_round_fit_rob_mm = None
                try:
                    delta_c = float(self._idcal_get_delta_c_active())
                except Exception:
                    delta_c = 0.0
                if not id_single_enable:
                    try:
                        id_round_fit_mm, id_round_fit_rob_mm = self._id_round_fit_from_raw_points(
                            raw_points,
                            use_fit=bool(getattr(recipe, 'id_use_fit', False)),
                            delta_c=float(delta_c),
                            calc_input_mode=str(getattr(recipe, 'calc_input_mode', 'bin')),
                            bin_count=int(getattr(recipe, 'bin_count', 90)),
                            bin_method=str(getattr(recipe, 'bin_method', 'median')),
                            pp_mode=str(getattr(recipe, 'pp_mode', 'p99_p1')),
                            theta_delay_s=float(getattr(recipe, 'theta_delay_s', 0.0) or 0.0),
                        )
                    except Exception:
                        id_round_fit_mm, id_round_fit_rob_mm = None, None
                else:
                    id_round_fit_mm, id_round_fit_rob_mm = None, None

                # Use Z_Pos (x_ui) as the axial coordinate for straightness.
                centers_xyz.append((float(center_od_x), float(center_od_y), float(x_ui)))

                # ID diameter stats
                id_e = None
                id_phi_deg = None
                if not id_single_enable:
                    dxi = coords_id[:, 0] - float(xci)
                    dyi = coords_id[:, 1] - float(yci)
                    ri_list = np.sqrt(dxi * dxi + dyi * dyi)
                    id_list = 2.0 * ri_list
                    id_avg = float(np.mean(id_list)) if id_list.size else 0.0
                    id_round = float(np.max(id_list) - np.min(id_list)) if id_list.size >= 2 else 0.0
                    id_dev = float(id_avg) - float(recipe.id_std_mm)

                    # Override ID stats with new ID fit algorithm (diameter from chord+m) when enabled.
                    if bool(getattr(recipe, "id_use_fit", False)) and (id_fit_diam is not None) and math.isfinite(float(id_fit_diam)) and float(id_fit_diam) > 0.0:
                        try:
                            id_avg = float(id_fit_diam)
                            id_dev = float(id_avg) - float(recipe.id_std_mm)
                        except Exception:
                            pass
                        try:
                            if id_fit_vals is not None and getattr(id_fit_vals, "size", 0) >= 2:
                                id_round = _pp_robust(np.asarray(id_fit_vals, dtype=float))
                        except Exception:
                            pass

                    # Concentricity (distance between OD/ID fitted centers)
                    # ID center uses fitted (ex,ey) from m(theta) when id_use_fit is enabled;
                    # otherwise fall back to circle-fit center (xci,yci).
                    center_id_x = float(xci)
                    center_id_y = float(yci)
                    try:
                        if bool(getattr(recipe, "id_use_fit", False)) and (id_fit is not None):
                            _ex = id_fit.get("ex", None) if isinstance(id_fit, dict) else None
                            _ey = id_fit.get("ey", None) if isinstance(id_fit, dict) else None
                            if _ex is not None and _ey is not None and math.isfinite(float(_ex)) and math.isfinite(float(_ey)):
                                center_id_x = float(_ex)
                                center_id_y = float(_ey)
                    except Exception:
                        pass

                    concentricity = float(math.hypot(float(center_id_x) - float(center_od_x), float(center_id_y) - float(center_od_y)))
                    concentricity_list.append(float(concentricity))
                    centers_xyz_id.append((float(center_id_x), float(center_id_y), float(x_ui)))

                    # ID eccentricity (per-section), available when using new ID fit algorithm (chord+m).
                    try:
                        if bool(getattr(recipe, "id_use_fit", False)) and (id_fit is not None):
                            _e = id_fit.get("e", None)
                            _phi = id_fit.get("phi_rad", None)
                            if _e is not None and math.isfinite(float(_e)):
                                id_e = float(_e)
                            if _phi is not None and math.isfinite(float(_phi)):
                                id_phi_deg = float(np.rad2deg(float(_phi)))
                                # normalize to (-180, 180]
                                if id_phi_deg <= -180.0:
                                    id_phi_deg += 360.0
                                elif id_phi_deg > 180.0:
                                    id_phi_deg -= 360.0
                    except Exception:
                        id_e = None
                        id_phi_deg = None
                else:
                    id_single_res = None
                    try:
                        th_list = []
                        out2_list = []
                        for p in raw_points:
                            if not isinstance(p, dict):
                                continue
                            th = p.get("theta_deg", None)
                            v = p.get("id_out2_mm", None)
                            if th is None or v is None:
                                continue
                            th_list.append(float(th))
                            out2_list.append(float(v))
                        if len(out2_list) >= 3:
                            id_single_res = self.app.calc_id_single_from_out2(th_list, out2_list, recipe)
                    except Exception:
                        id_single_res = None

                    if id_single_res and bool(id_single_res.get("ok", False)):
                        id_avg = id_single_res.get("id_est_mm", None)
                        try:
                            id_dev = None if id_avg is None else float(id_avg) - float(recipe.id_std_mm)
                        except Exception:
                            id_dev = None
                        id_pp_mm = id_single_res.get("id_pp_mm", None)
                        id_pp_rob_mm = id_single_res.get("id_pp_rob_mm", None)
                        id_round = id_pp_rob_mm
                        id_e = id_single_res.get("id_ecc_amp_mm", None)
                        id_phi_deg = id_single_res.get("id_ecc_ang_deg", None)
                        try:
                            if id_e is not None:
                                id_runout = float(2.0 * float(id_e))
                            elif id_pp_rob_mm is not None:
                                id_runout = float(id_pp_rob_mm)
                            else:
                                id_runout = None
                        except Exception:
                            id_runout = None
                    else:
                        id_avg = None
                        id_dev = None
                        id_round = None
                        id_runout = None
                        id_pp_mm = None
                        id_pp_rob_mm = None

                    concentricity = None

                # ID runout definition:
                # - legacy (OUT3): use diameter peak-to-peak of id_mm series (computed above).
                # - new algorithm (OUT4 chord + OUT5 m): interpret runout as *eccentricity-driven*
                #   radial runout (diameter) ~= 2 * e, where e is the fitted eccentricity amplitude.
                #   This intentionally differs from "roundness" (diameter variation pp).
                try:
                    if bool(getattr(recipe, "id_use_fit", False)) and (id_e is not None) and math.isfinite(float(id_e)):
                        id_runout = float(2.0 * float(id_e))
                except Exception:
                    pass

                try:
                    od_tol_v = float(recipe.od_tol_mm)
                except Exception:
                    od_tol_v = 0.0
                if id_dev is None:
                    ok_flag = (abs(od_dev) <= float(od_tol_v))
                else:
                    ok_flag = (abs(od_dev) <= float(od_tol_v)) and (abs(id_dev) <= float(od_tol_v))

                row = MeasureRow(
                    idx=i + 1,
                    x_ui=x_ui,
                    x_abs=x_abs,
                    od_avg=od_avg,
                    od_dev=od_dev,
                    od_runout=od_runout,
                    od_round=od_round,
                    od_round_fit_mm=od_round_fit_mm,
                    od_round_fit_rob_mm=od_round_fit_rob_mm,
                    od_pp_mm=(None if od_pp_mm is None else float(od_pp_mm)),
                    od_pp_rob_mm=(None if od_pp_rob_mm is None else float(od_pp_rob_mm)),
                    id_round_fit_mm=id_round_fit_mm,
                    id_round_fit_rob_mm=id_round_fit_rob_mm,
                    id_pp_mm=(None if id_pp_mm is None else float(id_pp_mm)),
                    id_pp_rob_mm=(None if id_pp_rob_mm is None else float(id_pp_rob_mm)),
                    od_e=(float(od_e) if od_use_edges else None),
                    od_phi_deg=(float(od_phi_deg) if (od_use_edges and od_phi_deg is not None) else None),
                    id_e=id_e,
                    id_phi_deg=id_phi_deg,
                    id_mode=("single" if id_single_enable else "dual"),
                    id_avg=id_avg,
                    id_dev=id_dev,
                    id_runout=id_runout,
                    id_round=id_round,
                    concentricity=concentricity,
                    split_shift_deg=split_shift_deg,
                    coax_unreliable=coax_unreliable,
                    ok=ok_flag,
                    raw=f"OD:{raw_od}  ID:{raw_id}",
                )
                self.app.ui_q.put(("auto_row", {"row": row}))
            # Post-calc: straightness + eccentricity (OD and ID)

            try:
                id_single_enable = bool(getattr(recipe, "id_single_enable", False))
            except Exception:
                id_single_enable = False

            try:
                straight_od, ecc_od, p_od, d_od = _fit_line_and_dist(centers_xyz)
                if id_single_enable:
                    straight_id = None
                    ecc_id = []
                    axis_dist = None
                    conc_max = None
                    axis_span_max = None
                    p_id = None
                    d_id = None
                else:
                    straight_id, ecc_id, p_id, d_id = _fit_line_and_dist(centers_xyz_id)
                    axis_dist = _line_distance(p_od, d_od, p_id, d_id)

                    # overall concentricity metrics
                    conc_max = float(max(concentricity_list)) if concentricity_list else None
                    axis_span_max = None
                    try:
                        z_list = [float(p[2]) for p in centers_xyz] if centers_xyz else []
                        dz_od = float(d_od[2])
                        dz_id = float(d_id[2])
                        if (not z_list) or (abs(dz_od) < 1e-12) or (abs(dz_id) < 1e-12):
                            axis_span_max = None
                        else:
                            axis_span_max = 0.0
                            for z in z_list:
                                t_od = (z - float(p_od[2])) / dz_od
                                t_id = (z - float(p_id[2])) / dz_id
                                pz_od = p_od + t_od * d_od
                                pz_id = p_id + t_id * d_id
                                dxy = float(math.hypot(float(pz_od[0] - pz_id[0]), float(pz_od[1] - pz_id[1])))
                                if dxy > float(axis_span_max):
                                    axis_span_max = dxy
                    except Exception:
                        axis_span_max = None

                od_tilt_deg, od_end_off_mm, od_slope = _tilt_and_end_offset(p_od, d_od, centers_xyz)
                if id_single_enable:
                    id_tilt_deg, id_end_off_mm, id_slope = None, None, None
                else:
                    id_tilt_deg, id_end_off_mm, id_slope = _tilt_and_end_offset(p_id, d_id, centers_xyz_id)
                # Update overall label (outer/inner + overall concentricity)
                self.app.ui_q.put(
                    (
                        "auto_straightness",
                        {
                            "straight_od": straight_od,
                            "straight_id": straight_id,
                            "axis_dist": axis_dist,
                            "conc_max": conc_max,
                            "axis_span_max": axis_span_max,
                            "od_tilt_deg": od_tilt_deg,
                            "od_end_off_mm": od_end_off_mm,
                            "od_slope": od_slope,
                            "id_tilt_deg": id_tilt_deg,
                            "id_end_off_mm": id_end_off_mm,
                            "id_slope": id_slope,
                        },
                    )
                )
                # Update table eccentricities + straightness
                self.app.ui_q.put(
                    (
                        "auto_postcalc",
                        {
                            "ecc_od": ecc_od,
                            "ecc_id": ecc_id,
                            "straight_od": straight_od,
                            "straight_id": straight_id,
                            "axis_dist": axis_dist,
                            "conc_max": conc_max,
                            "axis_span_max": axis_span_max,
                            "od_tilt_deg": od_tilt_deg,
                            "od_end_off_mm": od_end_off_mm,
                            "od_slope": od_slope,
                            "id_tilt_deg": id_tilt_deg,
                            "id_end_off_mm": id_end_off_mm,
                            "id_slope": id_slope,
                        },
                    )
                )
            except Exception:
                # do not break completion on post-calc
                self.app.ui_q.put(("auto_straightness", {"straight_od": None, "straight_id": None, "axis_dist": None, "conc_max": None, "axis_span_max": None}))
            # End of auto-measure: stop AX3 first, then return AX0/AX1/AX4 to standby point (if configured).
            try:
                # Stop rotate first
                self.device.stop(3)
                t0 = time.time()
                while (time.time() - t0) < 10.0:
                    if self._should_stop():
                        break
                    ac3 = self.device.get_axis_copy(3)
                    if not self._is_moving(int(getattr(ac3, "sts", 0))):
                        break
                    time.sleep(0.08)
            except Exception:
                pass

            try:
                if bool(getattr(recipe, "standby_valid", False)):
                    targets2 = {
                        ax_id1: float(getattr(recipe, "standby_ax1_abs", 0.0)),
                        ax_id4: float(getattr(recipe, "standby_ax4_abs", 0.0)),
                        ax_od: float(getattr(recipe, "standby_ax0_abs", 0.0)),
                    }

                    # Soft limits for standby return: clamp if needed (do not block completion).
                    for ax, tgt in list(targets2.items()):
                        targets2[ax] = self.device.apply_soft_limits_abs(
                            int(ax), float(tgt), strict=False, context="AUTO_STANDBY"
                        )

                    for ax, tgt in targets2.items():
                        self._write_fp64(ax, OFF_POS_MOVEA, float(tgt))
                        self._ensure_movea_setpoints(ax)
                        self.app._pulse_cmd_bits(ax, CMD_MOVEA_REQ)

                    for ax, tgt in targets2.items():
                        ok = self._wait_in_position(ax, tgt, pos_tol=0.05, timeout_s=30.0)
                        if not ok:
                            break
            except Exception:
                # never block completion
                pass

            # Mark completion (UI will trigger export once per run).
            self.app.ui_q.put(("auto_state", {"state": "DONE", "msg": "测量完成"}))

        except Exception as e:
            try:
                log_exc("AUTO_FLOW_EXCEPTION", e)
            except Exception:
                pass
            # If user pressed STOP, show STOP instead of ERR.
            if self._should_stop():
                self.app.ui_q.put(("auto_state", {"state": "STOP", "msg": "用户停止"}))
            else:
                self.app.ui_q.put(("auto_state", {"state": "ERR", "msg": str(e)}))
        finally:
            # 无论如何都停旋转（清电平位）
            self.device.stop(3)

            # 若用户停止：对所有轴发一次 STOP/HALT，避免继续运动
            if self._should_stop():
                try:
                    self.app.abort_motion()
                except Exception:
                    pass


from frp_workflow.executor._executor_clamps import ExecutorClampsMixin
from frp_workflow.executor._executor_motion import ExecutorMotionMixin
from frp_workflow.executor._executor_length import ExecutorLengthMixin
from frp_workflow.executor._executor_sampling import ExecutorSamplingMixin
from frp_workflow.executor._executor_fitting import ExecutorFittingMixin


class AutoFlow(
    ExecutorCoreMixin,
    ExecutorClampsMixin,
    ExecutorMotionMixin,
    ExecutorLengthMixin,
    ExecutorSamplingMixin,
    ExecutorFittingMixin,
    threading.Thread,
):
    """Threaded measurement runner composed from executor mixins."""


__all__ = ["AutoFlow", "ExecutorCoreMixin"]
