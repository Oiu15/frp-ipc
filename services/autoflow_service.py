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

            # Ensure section_pos_ui exists
            if len(recipe.section_pos_ui) != recipe.section_count:
                recipe.section_pos_ui = recipe.compute_default_positions_ui()

            scan_ax = int(recipe.scan_axis)
            if not (0 <= scan_ax < AXIS_COUNT):
                raise ValueError("scan_axis 超界")

            # Pre-check scan axis fault (Axis_Ctrl: raw_axis_state/raw_axis_err)
            ac = self.app.get_axis_copy(scan_ax)
            if self._is_fault(int(ac.sts), int(ac.err)):
                raise RuntimeError(f"扫描轴 AX{scan_ax} 故障，Err={int(ac.err)}")

            # Enable scan axis if not enabled
            if not self._is_enabled(int(ac.sts)):
                self.app.set_cmd_bits(scan_ax, set_mask=CMD_EN_REQ, clr_mask=0)
                time.sleep(0.25)

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
            centers_xyz: List[Tuple[float, float, float]] = []  # (cx_rel, cy_rel, z_act)

            for i in range(recipe.section_count):
                if self.stop_event.is_set():
                    self.app.ui_q.put(
                        ("auto_state", {"state": "STOP", "msg": "用户停止"})
                    )
                    return

                x_ui = float(recipe.section_pos_ui[i])
                x_abs = self.app.ui_coord.ui_to_abs(x_ui)

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

                # Motion: MoveA (write Pos_MoveA + pulse CMD_MOVEA_REQ)
                self._write_fp64(scan_ax, OFF_POS_MOVEA, float(x_abs))
                self._ensure_movea_setpoints(scan_ax)
                self.app._pulse_cmd_bits(scan_ax, CMD_MOVEA_REQ)

                # Wait in-position
                ok = self._wait_in_position(
                    scan_ax, x_abs, pos_tol=0.05, timeout_s=20.0
                )
                if not ok:
                    # If user pressed STOP, _wait_in_position returns False; treat as STOP not ERR.
                    if self.stop_event.is_set():
                        self.app.ui_q.put(
                            ("auto_state", {"state": "STOP", "msg": "用户停止"})
                        )
                        return
                    raise TimeoutError(f"AX{scan_ax} 到位超时（目标 {x_abs:.3f}）")

                # Sampling (angle + OD), circle fit
                coords, raw_last = self._sample_circle_points(recipe)

                try:
                    n_total, n_hit, n_miss = getattr(self, "_last_sample_cov", (0, 0, 0))
                    cov = (float(n_hit) / float(n_total)) if n_total else None
                    reason, revs, elapsed = getattr(self, "_last_sample_reason", ("-", 0.0, 0.0))
                    self.app.ui_q.put(("auto_cov", {"cov": cov, "miss": n_miss, "reason": reason, "revs": revs, "elapsed": elapsed}))
                except Exception:
                    pass
                xc, yc, r_fit, _sigma = self._fit_circle(coords)

                # Build coordinate system using first section fitted center as origin
                if i == 0:
                    ref_cx, ref_cy = float(xc), float(yc)
                    self._ref_center = (ref_cx, ref_cy)
                else:
                    ref_cx, ref_cy = getattr(self, "_ref_center", (float(xc), float(yc)))

                # Store section center in global coordinate (origin = first section center)
                cx_rel = float(xc) - float(ref_cx)
                cy_rel = float(yc) - float(ref_cy)
                ax0 = self.app.get_axis_copy(0)
                z_act = float(ax0.act_pos)
                centers_xyz.append((cx_rel, cy_rel, z_act))

                # Compute OD for each point w.r.t reference origin
                dx = coords[:, 0] - float(ref_cx)
                dy = coords[:, 1] - float(ref_cy)
                r_list = np.sqrt(dx * dx + dy * dy)
                od_list = 2.0 * r_list

                od_avg = float(np.mean(od_list)) if od_list.size else 0.0
                od_max = float(np.max(od_list)) if od_list.size else 0.0
                od_min = float(np.min(od_list)) if od_list.size else 0.0
                od_round = float(od_max - od_min) if od_list.size >= 2 else 0.0

                dev = float(od_avg) - float(recipe.od_std_mm)
                ok_flag = abs(dev) <= float(recipe.od_tol_mm)

                row = MeasureRow(
                    idx=i + 1,
                    x_ui=x_ui,
                    x_abs=x_abs,
                    od_avg=od_avg,
                    od_max=od_max,
                    od_min=od_min,
                    dev=dev,
                    od_round=od_round,
                    ok=ok_flag,
                    raw=raw_last,
                )
                self.app.ui_q.put(("auto_row", {"row": row}))
            # Straightness: fit 3D axis line using section centers (cx_rel,cy_rel,z)
            try:
                if len(centers_xyz) >= 2:
                    P = np.array(centers_xyz, dtype=float)
                    # PCA line fit
                    p0 = P.mean(axis=0)
                    Q = P - p0
                    C = (Q.T @ Q) / max(1, Q.shape[0])
                    w, v = np.linalg.eigh(C)
                    d = v[:, int(np.argmax(w))]
                    d = d / (np.linalg.norm(d) + 1e-12)
                    # distances to line
                    t = (Q @ d)
                    proj = np.outer(t, d)
                    R = Q - proj
                    dist = np.linalg.norm(R, axis=1)
                    straightness = float(dist.max() - dist.min()) if dist.size else 0.0
                else:
                    straightness = 0.0
                self.app.ui_q.put(("auto_straightness", {"straightness": straightness}))
            except Exception:
                # do not break completion on straightness calc
                self.app.ui_q.put(("auto_straightness", {"straightness": None}))

            # Return AX0 to UI zero position after auto-measure
            try:
                ax0_abs0 = self.app.ui_coord.ui_to_abs(0.0)
                self._write_fp64(0, OFF_POS_MOVEA, float(ax0_abs0))
                self._ensure_movea_setpoints(0)
                self.app._pulse_cmd_bits(0, CMD_MOVEA_REQ)
                self._wait_in_position(0, ax0_abs0, pos_tol=0.05, timeout_s=30.0)
            except Exception:
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

    def _sample_circle_points(self, recipe: Recipe) -> Tuple[np.ndarray, str]:
        """采样一圈的数据点：每点由 (角度deg, 外径ODmm) -> (x,y)。

        约定：将 OD/2 作为半径 r，使用 (x, y) = (r*cos(theta), r*sin(theta)) 生成点。
        """
        n = max(3, int(getattr(recipe, "points_per_rev", 120)))
        min_cov = float(getattr(recipe, "min_bin_coverage", 0.95))
        min_cov = max(0.0, min(1.0, min_cov))
        timeout_s = float(getattr(recipe, "sample_timeout_s", 5.0))
        timeout_s = max(0.5, timeout_s)
        max_revs = float(getattr(recipe, "max_revolutions", 2.0))
        max_revs = max(0.25, max_revs)

        # 等角bin：将 0~360° 划分为 n 个bin，每个bin做均值（更抗噪）
        sum_x = [0.0] * n
        sum_y = [0.0] * n
        cnt = [0] * n
        filled = 0
        raw_last = ""

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

            # angle from AX3 (deg)
            a3 = self.app.get_axis_copy(3)
            theta_deg = float(a3.act_pos) % 360.0
            # unwrap to estimate revolutions (robust to wrap-around)
            if prev_theta is None:
                prev_theta = theta_deg
            else:
                d = theta_deg - prev_theta
                if d < -180.0:
                    d += 360.0
                elif d > 180.0:
                    d -= 360.0
                unwrapped_deg += abs(d)
                prev_theta = theta_deg
                revs = unwrapped_deg / 360.0
            theta = math.radians(theta_deg)

            # OD from gauge (real or simulated)
            if self.app.sim_gauge_enabled:
                od, raw = self.app.simulate_gauge_once(recipe)
                raw_last = raw
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
                        raw_last = s.raw
                        break
                    time.sleep(0.02)
                else:
                    # 未收到新值：继续尝试，避免立刻失败导致覆盖率不足
                    continue

            r = 0.5 * float(od)
            x = r * math.cos(theta)
            y = r * math.sin(theta)

            # map to bin
            b = int((theta_deg / 360.0) * n)
            if b >= n:
                b = 0

            if cnt[b] == 0:
                filled += 1
            cnt[b] += 1
            sum_x[b] += x
            sum_y[b] += y

            # modest pace to avoid saturating serial/PLC
            time.sleep(0.005)

        # build averaged coords
        coords: List[Tuple[float, float]] = []
        miss = 0
        for i in range(n):
            if cnt[i] > 0:
                coords.append((sum_x[i] / cnt[i], sum_y[i] / cnt[i]))
            else:
                miss += 1

        # store last coverage/reason for UI/debug
        elapsed = float(time.time() - t_start)
        self._last_sample_cov = (n, n - miss, miss)
        self._last_sample_reason = (reason, revs, elapsed)

        if len(coords) < 3:
            raise RuntimeError("等角采样覆盖不足：有效点数 < 3，无法拟合圆。")

        return np.asarray(coords, dtype=float), raw_last

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


