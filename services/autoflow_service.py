# ./services/autoflow_service.py
from __future__ import annotations

"""自动测量流程（线程）。

当前版本：外径（OD）自动测量 + 截面圆拟合 + 直线度（基于截面圆心的 3D 拟合）。

说明：
- 该模块不直接依赖 tkinter，但会通过 app.ui_q 回传 UI 事件。
- app 作为“控制器”由 app.App 提供，AutoFlow 只使用其公开方法/属性（duck typing）。
"""

import math
import threading
import time
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

try:
    import circle_fit as cf  # type: ignore
except Exception:  # pragma: no cover
    cf = None  # type: ignore

from config.addresses import (
    AXIS_COUNT,
    CMD_EN_REQ,
    CMD_VELMOVE_REQ,
    CMD_STOP_REQ,
    CMD_HALT_REQ,
    CMD_MOVEA_REQ,
    # Axis_Ctrl setpoint offsets
    OFF_POS_MOVEA,
    OFF_VEL_MOVEA,
    OFF_VEL_VELMOVE,
    OFF_ACC,
    OFF_DEC,
    OFF_JERK,
    # raw state enum (Axis_Ctrl.Sts)
    STS_RAW_NOT_ENABLED,
    STS_RAW_ENABLED_IDLE,
    STS_RAW_MOVING,
    STS_RAW_VELRUN,
    STS_RAW_SYNC,
    STS_RAW_HOMING,
    STS_RAW_STOPPING,
    STS_RAW_FAULT,
    STS_RAW_GROUP,
    FLOAT64_WORD_ORDER,
)

from drivers.plc_client import encode_float64_to_4regs
from core.models import MeasureRow, Recipe

if TYPE_CHECKING:  # pragma: no cover
    from app import App

class AutoFlow(threading.Thread):
    def __init__(self, app: "App"):
        super().__init__(daemon=True)
        self.app = app
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    # =========================
    # Helpers (Axis_Ctrl raw state)
    # =========================
    def _is_fault(self, sts: int, err: int) -> bool:
        return (int(err) != 0) or (int(sts) == STS_RAW_FAULT)

    def _is_enabled(self, sts: int) -> bool:
        return int(sts) != STS_RAW_NOT_ENABLED

    def _is_moving(self, sts: int) -> bool:
        s = int(sts)
        return s in {
            STS_RAW_MOVING,
            STS_RAW_VELRUN,
            STS_RAW_SYNC,
            STS_RAW_HOMING,
            STS_RAW_STOPPING,
            STS_RAW_GROUP,
        }

    def _write_fp64(self, axis: int, off: int, value: float) -> None:
        base = self.app._base(int(axis))
        self.app._write_regs(base + int(off), encode_float64_to_4regs(float(value), FLOAT64_WORD_ORDER))

    def _ensure_movea_setpoints(
        self,
        axis: int,
        default_vel: float = 100.0,
        default_acc: float = 200.0,
        default_dec: float = 200.0,
        default_jerk: float = 500.0,
    ) -> None:
        """Ensure MoveA-related setpoints exist (non-zero) in PLC."""
        ac = self.app.get_axis_copy(int(axis))
        # vel is legacy mirror of Vel_MoveA
        if float(getattr(ac, "vel", 0.0) or 0.0) <= 0.0:
            self._write_fp64(axis, OFF_VEL_MOVEA, default_vel)
        if float(getattr(ac, "acc", 0.0) or 0.0) <= 0.0:
            self._write_fp64(axis, OFF_ACC, default_acc)
        if float(getattr(ac, "dec", 0.0) or 0.0) <= 0.0:
            self._write_fp64(axis, OFF_DEC, default_dec)
        if float(getattr(ac, "jerk", 0.0) or 0.0) <= 0.0:
            self._write_fp64(axis, OFF_JERK, default_jerk)

    def _ensure_velmove_setpoints(
        self,
        axis: int,
        default_vel: float = 200.0,
        default_acc: float = 200.0,
        default_dec: float = 200.0,
        default_jerk: float = 500.0,
    ) -> None:
        """Ensure VelMove-related setpoints exist (non-zero) in PLC."""
        ac = self.app.get_axis_copy(int(axis))
        v = float(getattr(ac, "vel_velmove", 0.0) or 0.0)
        if v <= 0.0:
            self._write_fp64(axis, OFF_VEL_VELMOVE, default_vel)

        # For simplicity, reuse common acc/dec/jerk
        if float(getattr(ac, "acc", 0.0) or 0.0) <= 0.0:
            self._write_fp64(axis, OFF_ACC, default_acc)
        if float(getattr(ac, "dec", 0.0) or 0.0) <= 0.0:
            self._write_fp64(axis, OFF_DEC, default_dec)
        if float(getattr(ac, "jerk", 0.0) or 0.0) <= 0.0:
            self._write_fp64(axis, OFF_JERK, default_jerk)

    def run(self):
        try:
            self.app.ui_q.put(("auto_state", {"state": "RUN", "msg": "自动测量开始"}))

            recipe = self.app.get_recipe_copy()
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
            scan_ax = ax_od

            # Pre-check + enable OD/ID axes
            for ax in (ax_od, ax_id1, ax_id4):
                ac = self.app.get_axis_copy(ax)
                if self._is_fault(int(ac.sts), int(ac.err)):
                    raise RuntimeError(f"轴 AX{ax} 故障，Err={int(ac.err)}")
                if not self._is_enabled(int(ac.sts)):
                    self.app.set_cmd_bits(ax, set_mask=CMD_EN_REQ, clr_mask=0)
                    time.sleep(0.15)

            # Prepare rotate axis (AX3): enable + ensure velmove params
            a3 = self.app.get_axis_copy(3)
            if self._is_fault(int(a3.sts), int(a3.err)):
                raise RuntimeError(f"旋转轴 AX3 故障，Err={int(a3.err)}")

            if not self._is_enabled(int(a3.sts)):
                self.app.set_cmd_bits(3, set_mask=CMD_EN_REQ, clr_mask=0)
                time.sleep(0.25)

            self._ensure_velmove_setpoints(3)
            time.sleep(0.05)

            # start rotate (AX3) - level command
            self.app.set_cmd_bits(3, set_mask=CMD_VELMOVE_REQ, clr_mask=0)
            time.sleep(0.20)

            # Clear results first
            self.app.ui_q.put(("auto_clear", {"ts": time.time()}))

            # Move + sample per section
            # Use absolute fitted centers (same coordinate frame for OD/ID) so we can
            # compute both straightness/eccentricity and the distance between OD/ID axes.
            centers_xyz: List[Tuple[float, float, float]] = []      # (xc, yc, z)
            centers_xyz_id: List[Tuple[float, float, float]] = []   # (xci, yci, z)

            for i in range(recipe.section_count):
                if self.stop_event.is_set():
                    self.app.ui_q.put(
                        ("auto_state", {"state": "STOP", "msg": "用户停止"})
                    )
                    return

                z_od_disp = float(recipe.section_pos_z[i])
                tg = cal.od_z_disp_to_targets(z_od_disp)
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

                for ax, tgt in targets.items():
                    self._write_fp64(ax, OFF_POS_MOVEA, float(tgt))
                    self._ensure_movea_setpoints(ax)
                    self.app._pulse_cmd_bits(ax, CMD_MOVEA_REQ)

                for ax, tgt in targets.items():
                    ok = self._wait_in_position(ax, tgt, pos_tol=0.05, timeout_s=25.0)
                    if not ok:
                        if self.stop_event.is_set():
                            self.app.ui_q.put(("auto_state", {"state": "STOP", "msg": "用户停止"}))
                            return
                        raise TimeoutError(f"AX{ax} 到位超时（目标 {tgt:.3f}）")

                # Sampling (angle + OD/ID), circle fit
                coords_od, coords_id, raw_od, raw_id = self._sample_circle_points_dual(recipe, section_idx=i)

                try:
                    n_total, n_hit, n_miss = getattr(self, "_last_sample_cov", (0, 0, 0))
                    cov = (float(n_hit) / float(n_total)) if n_total else None
                    reason, revs, elapsed = getattr(self, "_last_sample_reason", ("-", 0.0, 0.0))
                    # Attach 1-based section index so UI can cache per-section coverage.
                    self.app.ui_q.put((
                        "auto_cov",
                        {"idx": i + 1, "cov": cov, "miss": n_miss, "reason": reason, "revs": revs, "elapsed": elapsed},
                    ))
                except Exception:
                    pass
                xc, yc, _r_fit, _sigma = self._fit_circle(coords_od)
                xci, yci, _r_fit_i, _sigma_i = self._fit_circle(coords_id)

                # Use Z_Pos (z_disp) as the axial coordinate for straightness.
                centers_xyz.append((float(xc), float(yc), float(x_ui)))

                # Compute OD for each point w.r.t reference origin
                # OD diameter stats
                dx = coords_od[:, 0] - float(xc)
                dy = coords_od[:, 1] - float(yc)
                r_list = np.sqrt(dx * dx + dy * dy)
                od_list = 2.0 * r_list
                od_avg = float(np.mean(od_list)) if od_list.size else 0.0
                od_round = float(np.max(od_list) - np.min(od_list)) if od_list.size >= 2 else 0.0
                od_dev = float(od_avg) - float(recipe.od_std_mm)

                # ID diameter stats
                dxi = coords_id[:, 0] - float(xci)
                dyi = coords_id[:, 1] - float(yci)
                ri_list = np.sqrt(dxi * dxi + dyi * dyi)
                id_list = 2.0 * ri_list
                id_avg = float(np.mean(id_list)) if id_list.size else 0.0
                id_round = float(np.max(id_list) - np.min(id_list)) if id_list.size >= 2 else 0.0
                id_dev = float(id_avg) - float(recipe.id_std_mm)

                # Concentricity (distance between fitted centers)
                concentricity = float(math.hypot(float(xci) - float(xc), float(yci) - float(yc)))

                centers_xyz_id.append((float(xci), float(yci), float(x_ui)))

                ok_flag = (abs(od_dev) <= float(recipe.od_tol_mm)) and (abs(id_dev) <= float(recipe.od_tol_mm))

                row = MeasureRow(
                    idx=i + 1,
                    x_ui=x_ui,
                    x_abs=x_abs,
                    od_avg=od_avg,
                    od_dev=od_dev,
                    od_round=od_round,
                    id_avg=id_avg,
                    id_dev=id_dev,
                    id_round=id_round,
                    concentricity=concentricity,
                    ok=ok_flag,
                    raw=f"OD:{raw_od}  ID:{raw_id}",
                )
                self.app.ui_q.put(("auto_row", {"row": row}))
            # Post-calc: straightness + eccentricity (OD and ID)
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

            try:
                straight_od, ecc_od, p_od, d_od = _fit_line_and_dist(centers_xyz)
                straight_id, ecc_id, p_id, d_id = _fit_line_and_dist(centers_xyz_id)
                axis_dist = _line_distance(p_od, d_od, p_id, d_id)
                # Update overall label (outer/inner + overall concentricity)
                self.app.ui_q.put(
                    (
                        "auto_straightness",
                        {"straight_od": straight_od, "straight_id": straight_id, "axis_dist": axis_dist},
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
                        },
                    )
                )
            except Exception:
                # do not break completion on post-calc
                self.app.ui_q.put(("auto_straightness", {"straight_od": None, "straight_id": None, "axis_dist": None}))
            # End of auto-measure: stop AX3 first, then return AX0/AX1/AX4 to standby point (if configured).
            try:
                # Stop rotate first
                self.app.set_cmd_bits(3, set_mask=0, clr_mask=CMD_VELMOVE_REQ)
                self.app._pulse_cmd_bits(3, CMD_STOP_REQ)
                t0 = time.time()
                while (time.time() - t0) < 10.0:
                    if self.stop_event.is_set():
                        break
                    ac3 = self.app.get_axis_copy(3)
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

            self.app.ui_q.put(("auto_state", {"state": "DONE", "msg": "测量完成"}))

        except Exception as e:
            # If user pressed STOP, show STOP instead of ERR.
            if self.stop_event.is_set():
                self.app.ui_q.put(("auto_state", {"state": "STOP", "msg": "用户停止"}))
            else:
                self.app.ui_q.put(("auto_state", {"state": "ERR", "msg": str(e)}))
        finally:
            # 无论如何都停旋转（清电平位）
            self.app.set_cmd_bits(3, set_mask=0, clr_mask=CMD_VELMOVE_REQ)
            # 保险起见给一个 STOP 脉冲（可选，但建议）
            self.app._pulse_cmd_bits(3, CMD_STOP_REQ)

            # 若用户停止：对所有轴发一次 STOP/HALT，避免继续运动
            if self.stop_event.is_set():
                try:
                    self.app.abort_motion()
                except Exception:
                    pass

    def _wait_in_position(
        self, axis: int, tgt_abs: float, pos_tol: float, timeout_s: float
    ) -> bool:
        t0 = time.time()
        while (time.time() - t0) < float(timeout_s):
            if self.stop_event.is_set():
                return False
            ac = self.app.get_axis_copy(axis)

            sts = int(getattr(ac, "sts", 0))
            err_code = int(getattr(ac, "err", 0))
            if self._is_fault(sts, err_code):
                raise RuntimeError(f"AX{axis} 故障中，Err={err_code}")

            pos_err = abs(float(getattr(ac, "act_pos", 0.0)) - float(tgt_abs))

            # acceptance: position error small AND axis not in a moving state
            if (pos_err <= float(pos_tol)) and (not self._is_moving(sts)):
                return True

            time.sleep(0.08)

        return False

    def _sample_circle_points_dual(self, recipe: Recipe, section_idx: int = 0) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """Equal-angle sampling for both OD and ID.

        - Angle source: AX3 act_pos (deg)
        - OD source: gauge (real or simulated)
        - ID source: CL-3000 OUT3 (mapped via PLC) or simulated displacement meter

        Returns:
            (coords_od, coords_id, raw_last_od, raw_last_id)
        """
        n = max(3, int(getattr(recipe, "points_per_rev", 120)))
        min_cov = float(getattr(recipe, "min_bin_coverage", 0.95))
        min_cov = max(0.0, min(1.0, min_cov))
        timeout_s = float(getattr(recipe, "sample_timeout_s", 5.0))
        timeout_s = max(0.5, timeout_s)
        max_revs = float(getattr(recipe, "max_revolutions", 2.0))
        max_revs = max(0.25, max_revs)

        # 等角bin：将 0~360° 划分为 n 个bin，每个bin做均值（更抗噪）
        sum_x_od = [0.0] * n
        sum_y_od = [0.0] * n
        sum_x_id = [0.0] * n
        sum_y_id = [0.0] * n
        cnt = [0] * n
        filled = 0
        raw_last_od = ""
        raw_last_id = ""

        t_start = time.time()
        need = max(3, int(math.ceil(min_cov * n)))
        reason = "COV"  # COV / TIMEOUT / REV
        prev_theta = None
        unwrapped_deg = 0.0
        revs = 0.0

        while True:
            if self.stop_event.is_set():
                raise RuntimeError("测量被用户停止")

            if filled >= need:
                reason = "COV"
                break
            if revs >= max_revs:
                reason = "REV"
                break
            if (time.time() - t_start) >= timeout_s:
                reason = "TIMEOUT"
                break

            # OD from gauge (real or simulated)
            if self.app.sim_gauge_enabled:
                od, raw = self.app.simulate_gauge_once(recipe)
                raw_last_od = raw
            else:
                gw = self.app.gauge_worker
                if gw is None:
                    raise RuntimeError("测径仪未启用：请勾选“模拟测径仪”或连接真实串口。")

                t_req = time.time()
                gw.send_request()
                t0 = time.time()
                while (time.time() - t0) < 1.2:
                    if self.stop_event.is_set():
                        raise RuntimeError("测量被用户停止")
                    s = gw.get_last()
                    if s and s.ts >= t_req:
                        od = float(s.od)
                        raw_last_od = s.raw
                        break
                    time.sleep(0.02)
                else:
                    # 未收到新值：继续尝试，避免立刻失败导致覆盖率不足
                    continue

            # Angle snapshot for this OD sample (deg)
            theta_deg = None
            try:
                theta_deg = self.app.read_axis_act_pos_deg_sync(axis=3, timeout_s=0.25)
            except Exception:
                theta_deg = None
            if theta_deg is None:
                # fallback to background snapshot
                a3 = self.app.get_axis_copy(3)
                theta_deg = float(a3.act_pos) % 360.0

            # unwrap to estimate revolutions (robust to wrap-around)
            if prev_theta is None:
                prev_theta = float(theta_deg)
            else:
                d = float(theta_deg) - float(prev_theta)
                if d < -180.0:
                    d += 360.0
                elif d > 180.0:
                    d -= 360.0
                unwrapped_deg += abs(d)
                prev_theta = float(theta_deg)
                revs = unwrapped_deg / 360.0
            theta = math.radians(float(theta_deg))

            # ID from CL OUT3 mapped into PLC (or simulation fallback)
            if getattr(self.app, "sim_disp_enabled", False):
                id_mm, raw_id = self.app.simulate_disp_once(recipe)
                raw_last_id = raw_id
                # tiny deterministic eccentricity for simulation only
                ecc_x = 0.05 * math.sin(0.9 * float(section_idx))
                ecc_y = 0.05 * math.cos(0.7 * float(section_idx))
            else:
                id_mm, raw_i, cnt_i = self.app.read_cl_out3_sync(timeout_s=0.25)
                raw_last_id = f"OUT3={raw_i} cnt={cnt_i}"
                if id_mm is None:
                    # invalid/standby/overrange or read failed: skip this angle bin
                    continue
                ecc_x = 0.0
                ecc_y = 0.0

            r_od = 0.5 * float(od)
            x_od = r_od * math.cos(theta)
            y_od = r_od * math.sin(theta)

            r_id = 0.5 * float(id_mm)
            x_id = r_id * math.cos(theta) + float(ecc_x)
            y_id = r_id * math.sin(theta) + float(ecc_y)

            # map to bin
            b = int((theta_deg / 360.0) * n)
            if b >= n:
                b = 0

            if cnt[b] == 0:
                filled += 1
            cnt[b] += 1
            sum_x_od[b] += x_od
            sum_y_od[b] += y_od
            sum_x_id[b] += x_id
            sum_y_id[b] += y_id

            # modest pace to avoid saturating serial/PLC
            time.sleep(0.005)

        # build averaged coords
        coords_od: List[Tuple[float, float]] = []
        coords_id: List[Tuple[float, float]] = []
        miss = 0
        for i in range(n):
            if cnt[i] > 0:
                coords_od.append((sum_x_od[i] / cnt[i], sum_y_od[i] / cnt[i]))
                coords_id.append((sum_x_id[i] / cnt[i], sum_y_id[i] / cnt[i]))
            else:
                miss += 1

        # store last coverage/reason for UI/debug
        elapsed = float(time.time() - t_start)
        self._last_sample_cov = (n, n - miss, miss)
        self._last_sample_reason = (reason, revs, elapsed)

        if len(coords_od) < 3 or len(coords_id) < 3:
            raise RuntimeError("等角采样覆盖不足：有效点数 < 3，无法拟合圆。")

        return (
            np.asarray(coords_od, dtype=float),
            np.asarray(coords_id, dtype=float),
            raw_last_od,
            raw_last_id,
        )

    def _sample_circle_points(self, recipe: Recipe) -> Tuple[np.ndarray, str]:
        """Backward-compatible OD-only sampling wrapper."""
        coords_od, _coords_id, raw_od, _raw_id = self._sample_circle_points_dual(recipe, section_idx=0)
        return coords_od, raw_od

    def _fit_circle(self, coords: np.ndarray) -> Tuple[float, float, float, float]:
        """圆拟合：优先使用 circle-fit；不可用则用最小二乘兜底。

        Returns: (xc, yc, r, sigma)
        """
        if coords is None or len(coords) < 3:
            raise ValueError("圆拟合需要至少3个点")

        # circle-fit library (AlliedToasters/circle-fit)
        if cf is not None:
            try:
                xc, yc, r, sigma = cf.hyper_fit(coords)
                return float(xc), float(yc), float(r), float(sigma)
            except Exception:
                try:
                    xc, yc, r, sigma = cf.least_squares_circle(coords)
                    return float(xc), float(yc), float(r), float(sigma)
                except Exception:
                    pass

        # fallback: algebraic least squares (Kåsa)
        x = coords[:, 0]
        y = coords[:, 1]
        A = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
        b = x * x + y * y
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        xc, yc, c = sol
        r = math.sqrt(max(0.0, float(c) + float(xc) * float(xc) + float(yc) * float(yc)))
        # RMS residual on radius
        rr = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
        sigma = float(np.sqrt(np.mean((rr - r) ** 2))) if rr.size else 0.0
        return float(xc), float(yc), float(r), float(sigma)

