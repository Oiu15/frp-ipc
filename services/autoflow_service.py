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

from utils.logger import log, log_exc

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



# -------------------------
# Speedtest knobs
# -------------------------
# When True, AutoFlow will skip reading ID (CL Modbus) during sampling.
# This is useful to benchmark OD sampling rate and isolate comm bottlenecks.
SPEEDTEST_DISABLE_ID_MODBUS: bool = False


def _max_gap_deg_from_bins(cnt: List[int], n: int) -> float:
    """Compute maximum empty angular window (deg) based on bin hit counts."""
    try:
        n = int(n)
        if n <= 0:
            return 0.0
        hits = [i for i in range(n) if int(cnt[i]) > 0]
        if len(hits) == 0:
            return 360.0
        if len(hits) == n:
            return 0.0
        hits.sort()
        max_zero = 0
        for a, b in zip(hits, hits[1:]):
            gap0 = int(b - a - 1)
            if gap0 > max_zero:
                max_zero = gap0
        gapw = int(hits[0] + n - hits[-1] - 1)
        if gapw > max_zero:
            max_zero = gapw
        return float(max_zero) * (360.0 / float(n))
    except Exception:
        return 0.0


import re


def _reduce_bin(vals: list[float], method: str = "median") -> float:
    """Reduce a list of float values to a scalar (median/mean)."""
    if not vals:
        return float("nan")
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    m = (method or "").strip().lower()
    if m in ("mean", "avg", "average"):
        return float(np.mean(a))
    # default: median
    return float(np.median(a))


def _robust_span(a: np.ndarray, mode: str = "p99_p1") -> float:
    """Robust span (like peak-to-peak) for 1D array.

    mode:
      - strict: max-min
      - trim_0p01 / trim_0.01: trimmed max-min
      - p99_p1 / p99.5_p0.5: percentile(high)-percentile(low)
    """
    if a is None:
        return 0.0
    b = np.asarray(a, dtype=float).reshape(-1)
    b = b[np.isfinite(b)]
    if b.size < 2:
        return 0.0

    m = (mode or "").strip().lower()
    if m in ("strict", "maxmin", "pp"):
        return float(np.max(b) - np.min(b))

    # trim
    if m.startswith("trim"):
        ratio = 0.01
        mm = re.search(r"trim[_-]?([0-9]+(?:\.[0-9]+)?|[0-9]+p[0-9]+)", m)
        if mm:
            s = mm.group(1).replace("p", ".")
            try:
                ratio = float(s)
            except Exception:
                ratio = 0.01
        ratio = max(0.0, min(0.49, float(ratio)))
        bb = np.sort(b)
        n = int(bb.size)
        k = int(max(0, math.floor(ratio * n)))
        if (2 * k) >= (n - 1):
            k = 0
        return float(bb[n - 1 - k] - bb[k])

    # percentiles
    mm = re.match(r"p(\d+(?:\.\d+)?)_p(\d+(?:\.\d+)?)$", m)
    if mm:
        try:
            hi = float(mm.group(1))
            lo = float(mm.group(2))
            if hi < lo:
                hi, lo = lo, hi
            hi = max(0.0, min(100.0, hi))
            lo = max(0.0, min(100.0, lo))
            return float(np.percentile(b, hi) - np.percentile(b, lo))
        except Exception:
            return float(np.max(b) - np.min(b))

    # default: p99_p1
    try:
        return float(np.percentile(b, 99.0) - np.percentile(b, 1.0))
    except Exception:
        return float(np.max(b) - np.min(b))


def _estimate_omega_deg_s(theta_deg: list[float], ts: list[float]) -> float:
    """Estimate average angular speed (deg/s) from (theta_deg, ts) pairs.

    theta_deg is assumed wrapped to [0,360). This function unwraps by the shortest jump (±180 rule).
    """
    try:
        if (theta_deg is None) or (ts is None):
            return 0.0
        if len(theta_deg) < 2 or len(ts) < 2:
            return 0.0
        n = min(len(theta_deg), len(ts))
        th = np.asarray(theta_deg[:n], dtype=float)
        tt = np.asarray(ts[:n], dtype=float)
        m = np.isfinite(th) & np.isfinite(tt)
        th = th[m]
        tt = tt[m]
        if th.size < 2:
            return 0.0
        # unwrap
        th_u = np.empty_like(th)
        th_u[0] = th[0]
        for i in range(1, th.size):
            d = float(th[i] - th[i - 1])
            if d < -180.0:
                d += 360.0
            elif d > 180.0:
                d -= 360.0
            th_u[i] = th_u[i - 1] + d
        dt = tt - float(tt[0])
        if float(np.max(dt) - np.min(dt)) < 1e-6:
            return 0.0
        k, _b = np.polyfit(dt, th_u, 1)
        if not math.isfinite(float(k)):
            return 0.0
        return float(k)
    except Exception:
        return 0.0




def _omega_cv_deg_s(theta_deg: list[float], ts: list[float]) -> float:
    """Coefficient of variation (std/abs(mean)) of instantaneous angular speed (deg/s).

    Returns +inf if speed cannot be estimated.
    """
    try:
        if theta_deg is None or ts is None:
            return float('inf')
        n = min(len(theta_deg), len(ts))
        if n < 3:
            return float('inf')
        th = np.asarray(theta_deg[:n], dtype=float)
        tt = np.asarray(ts[:n], dtype=float)
        m = np.isfinite(th) & np.isfinite(tt)
        th = th[m]
        tt = tt[m]
        if th.size < 3:
            return float('inf')
        # unwrap
        th_u = np.empty_like(th)
        th_u[0] = th[0]
        for i in range(1, th.size):
            d = float(th[i] - th[i - 1])
            if d < -180.0:
                d += 360.0
            elif d > 180.0:
                d -= 360.0
            th_u[i] = th_u[i - 1] + d
        dt = np.diff(tt)
        dth = np.diff(th_u)
        m2 = dt > 1e-6
        if not np.any(m2):
            return float('inf')
        w = dth[m2] / dt[m2]
        if w.size < 2:
            return float('inf')
        mu = float(np.mean(w))
        if not math.isfinite(mu) or abs(mu) < 1e-6:
            return float('inf')
        sd = float(np.std(w))
        if not math.isfinite(sd):
            return float('inf')
        return abs(sd / mu)
    except Exception:
        return float('inf')


def _wrap_deg_180(d: float) -> float:
    try:
        x = float(d)
        if not math.isfinite(x):
            return float('nan')
        return ((x + 180.0) % 360.0) - 180.0
    except Exception:
        return float('nan')


def _split_slip_diag(
    raw_points_od: list[dict],
    raw_points_id: list[dict],
    slip_max_deg: float = 5.0,
    omega_cv_max: float = 0.25,
) -> tuple[float | None, bool | None]:
    """Lightweight split-scan diagnostics.

    Computes:
      - split_shift_deg: phase discontinuity between OD pass end and ID pass start (deg, wrapped to [-180,180)).
      - coax_unreliable: True if shift or speed stability exceeds thresholds; None if cannot evaluate.

    Notes:
      This does NOT prove mechanical slip; it's a sanity check to flag potentially unreliable coax metrics.
    """
    try:
        th_od = [float(p.get('theta_deg')) for p in (raw_points_od or []) if isinstance(p, dict) and p.get('theta_deg') is not None]
        ts_od = [float(p.get('ts')) for p in (raw_points_od or []) if isinstance(p, dict) and p.get('ts') is not None]
        th_id = [float(p.get('theta_deg')) for p in (raw_points_id or []) if isinstance(p, dict) and p.get('theta_deg') is not None]
        ts_id = [float(p.get('ts')) for p in (raw_points_id or []) if isinstance(p, dict) and p.get('ts') is not None]
        n_od = min(len(th_od), len(ts_od))
        n_id = min(len(th_id), len(ts_id))
        if n_od < 2 or n_id < 2:
            return None, None
        th_od = th_od[:n_od]
        ts_od = ts_od[:n_od]
        th_id = th_id[:n_id]
        ts_id = ts_id[:n_id]

        omega_od = _estimate_omega_deg_s(th_od, ts_od)
        # boundary shift: compare predicted theta at t0 of ID pass based on OD omega
        last_th = float(th_od[-1]) % 360.0
        last_ts = float(ts_od[-1])
        first_th = float(th_id[0]) % 360.0
        first_ts = float(ts_id[0])
        dt = float(first_ts - last_ts)
        pred = (last_th + float(omega_od) * dt) % 360.0
        shift = _wrap_deg_180(first_th - pred)

        cv_od = _omega_cv_deg_s(th_od, ts_od)
        cv_id = _omega_cv_deg_s(th_id, ts_id)

        # Decide unreliable
        bad = False
        if not math.isfinite(float(shift)):
            return None, None
        if abs(float(shift)) > float(slip_max_deg):
            bad = True
        if math.isfinite(float(cv_od)) and float(cv_od) > float(omega_cv_max):
            bad = True
        if math.isfinite(float(cv_id)) and float(cv_id) > float(omega_cv_max):
            bad = True
        # If CV is inf (not estimable), flag unreliable.
        if not math.isfinite(float(cv_od)) or not math.isfinite(float(cv_id)):
            bad = True

        return float(shift), bool(bad)
    except Exception:
        return None, None

def _theta_apply_delay(theta_deg: float, omega_deg_s: float, delay_s: float) -> float:
    """Shift theta forward by omega*delay and wrap to [0,360)."""
    try:
        th = float(theta_deg)
        if not math.isfinite(th):
            return float("nan")
        dd = float(delay_s)
        if not math.isfinite(dd) or abs(dd) < 1e-9:
            return th % 360.0
        w = float(omega_deg_s)
        if not math.isfinite(w):
            w = 0.0
        return (th + w * dd) % 360.0
    except Exception:
        return float("nan")

def _adaptive_bin_count(requested: int, n_samples: int, *, min_bins: int = 12) -> int:
    """Adaptive bin_count to avoid sparse bins when samples are limited.

    Heuristic: each bin should have ~>=2 samples on average.
    """
    try:
        req = int(requested)
    except Exception:
        req = 90
    req = max(3, req)
    try:
        ns = int(n_samples)
    except Exception:
        ns = 0
    # cap so that average samples per bin >= 2
    cap = max(3, ns // 2) if ns > 0 else 3
    eff = min(req, cap)
    eff = max(min_bins, eff) if ns >= min_bins * 2 else max(3, min(eff, cap))
    return int(max(3, eff))


class AutoFlow(threading.Thread):
    def __init__(self, app: "App"):
        super().__init__(daemon=True)
        self.app = app
        self.stop_event = threading.Event()
        self._last_sample_cov = (0, 0, 0)
        self._last_sample_reason = ("-", 0.0, 0.0)
        self._last_sample_max_gap_deg = None
        self._last_fit_weights_od = None
        self._last_fit_weights_id = None

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

    # =========================
    # Length measurement (AX0 + gauge)
    # =========================
    def _auto_measure_length(self, recipe: Recipe) -> dict:
        """Run length measurement inside auto flow.

        Returns a dict payload for UI/telemetry:
            {
              enabled: bool,
              skipped: bool,
              ok: bool,
              reason: str,
              z_low: float|None,
              z_high: float|None,
              length_mm: float|None,
              t_s: float,
            }

        Notes:
        - Failure must NOT break the overall auto flow.
        - Uses gauge "judge" (GO -> HI) for bottom edge.
        - For top edge we also use judge GO->HI, and lock the last GO position as the edge.
        """

        t0 = time.time()
        payload: dict = {
            "enabled": bool(getattr(recipe, "len_enable", False)),
            "skipped": False,
            "ok": False,
            "reason": "",
            "z_low": None,
            "z_high": None,
            "length_mm": None,
            "t_s": 0.0,
        }

        if not bool(getattr(recipe, "len_enable", False)):
            payload["skipped"] = True
            payload["reason"] = "DISABLED"
            payload["t_s"] = time.time() - t0
            return payload

        # prerequisites
        cal = getattr(self.app, "axis_cal", None)
        gw = getattr(self.app, "gauge_worker", None)
        if cal is None:
            payload["reason"] = "NO_AXIS_CAL"
            payload["t_s"] = time.time() - t0
            return payload
        if gw is None:
            payload["reason"] = "NO_GAUGE"
            payload["t_s"] = time.time() - t0
            return payload

        try:
            z_min, z_max, _travel = self.app._get_ax0_z_disp_limits()
        except Exception:
            # safe fallback
            z_min, z_max = -1e9, 1e9

        # parameters
        abs_low_approach = float(getattr(recipe, "len_low_approach_abs", 0.0) or 0.0)
        # Convert to Z_disp for travel checks / reporting
        z_low_approach = float(cal.abs_to_z_disp(0, abs_low_approach))
        d_low = float(getattr(recipe, "len_low_search_dist", 60.0) or 0.0)
        d_low = max(0.0, d_low)
        d_high = float(getattr(recipe, "len_high_search_dist", 60.0) or 0.0)
        d_high = max(0.0, d_high)
        # NOTE: Recipe uses `pipe_len_mm` (not `pipe_len`). Using the wrong field
        # will silently make auto length measurement run with 0.0mm and always fail.
        pipe_len = float(getattr(recipe, "pipe_len_mm", 0.0) or 0.0)
        hi_margin = float(getattr(recipe, "len_high_margin", 0.0) or 0.0)
        v_z = abs(float(getattr(recipe, "len_search_vel", 10.0) or 10.0))
        timeout_s = max(1.0, float(getattr(recipe, "len_search_timeout_s", 8.0) or 8.0))
        backoff_mm = max(0.0, float(getattr(recipe, "len_backoff_mm", 0.0) or 0.0))
        deb_k = max(1, int(float(getattr(recipe, "len_debounce_k", 2) or 2)))

        # feasibility check: rough Lmax based on travel + max search windows
        try:
            z_low_edge_max = min(float(z_max), float(z_low_approach) + float(d_low))
            lmax = float(z_low_edge_max) + float(hi_margin) - float(d_high) - float(z_min)
            if pipe_len > (lmax + 1.0):
                payload["skipped"] = True
                payload["reason"] = f"TOO_LONG(max≈{lmax:.1f}mm)"
                payload["t_s"] = time.time() - t0
                return payload
        except Exception:
            pass

        # axis state check
        ac0 = self.app.get_axis_copy(0)
        if (not self._is_enabled(int(getattr(ac0, "sts", 0)))) or self._is_fault(
            int(getattr(ac0, "sts", 0)), int(getattr(ac0, "err", 0))
        ):
            payload["reason"] = f"AX0_NOT_READY(err={int(getattr(ac0,'err',0) or 0)})"
            payload["t_s"] = time.time() - t0
            return payload

        def _wait_new_gauge(ts0: float, tmax: float = 1.5) -> Optional[Tuple[float, object]]:
            """Wait for a new gauge sample (valid numeric)."""
            t_wait0 = time.time()
            last_ts = float(ts0)
            while (not self._should_stop()) and ((time.time() - t_wait0) < float(tmax)):
                try:
                    gw.send_request()
                except Exception:
                    pass
                time.sleep(0.06)
                s = None
                try:
                    s = gw.get_last()
                except Exception:
                    s = None
                if s is None:
                    continue
                ts = float(getattr(s, "ts", 0.0) or 0.0)
                if ts > last_ts:
                    return ts, s
            return None

        
        # ---------------- bottom edge: GO -> HI -> GO (bidirectional average) ----------------
        def _scan_edge_bidirectional(
            z_start: float,
            dir_sign: int,
            max_dist: float,
            last_ts0: float,
            label: str,
        ) -> Tuple[Optional[float], float, str]:
            """Scan in Z_disp direction to find GO->HI, then reverse to find HI->GO; return averaged edge.

            dir_sign: +1 means +Z_disp, -1 means -Z_disp
            Returns: (edge_avg, last_ts, reason)
            """
            last_ts = float(last_ts0)
            # ---------- Pass 1: GO -> HI ----------
            edge1: Optional[float] = None
            t_search0 = time.time()
            unk_cnt = 0
            hi_cnt = 0
            last_go_z = float(z_start)

            # Motion guard (avoid waiting to timeout if axis is stuck)
            last_move_z: Optional[float] = None
            last_move_ts: float = time.time()

            def _axis_not_moving(z_cur: float) -> bool:
                nonlocal last_move_z, last_move_ts
                if last_move_z is None:
                    last_move_z = float(z_cur)
                    last_move_ts = time.time()
                    return False
                if abs(float(z_cur) - float(last_move_z)) >= 0.15:
                    last_move_z = float(z_cur)
                    last_move_ts = time.time()
                    return False
                return (time.time() - float(last_move_ts)) >= 1.0


            def _wait_axis_settled(tmax: float = 1.5) -> bool:
                """Wait until AX0 is not in stopping transient (warn=1003) and position is stable."""
                t_s0 = time.time()
                last_p = None
                stable = 0
                while (not self._should_stop()) and ((time.time() - t_s0) < float(tmax)):
                    ac = self.app.get_axis_copy(0)
                    try:
                        warn = int(getattr(ac, 'warn', 0) or 0)
                    except Exception:
                        warn = 0
                    try:
                        p = float(getattr(ac, 'act_pos', 0.0) or 0.0)
                    except Exception:
                        p = 0.0
                    if warn == 1003:
                        stable = 0
                        last_p = p
                        time.sleep(0.05)
                        continue
                    if last_p is None:
                        last_p = p
                        stable = 0
                        time.sleep(0.05)
                        continue
                    if abs(p - last_p) <= 0.02:
                        stable += 1
                    else:
                        stable = 0
                    last_p = p
                    if stable >= 8:
                        return True
                    time.sleep(0.05)
                return False

            vel_abs = float(v_z) * float(dir_sign) * float(cal.sign_eff(0))
            try:
                self.app._velmove_start_axis(0, vel_abs, acc=80.0, dec=80.0, jerk=300.0)
            except Exception:
                self._write_fp64(0, OFF_VEL_VELMOVE, vel_abs)
                self.app.set_cmd_bits(0, set_mask=CMD_VELMOVE_REQ, clr_mask=0)

            while not self._should_stop():
                ac0 = self.app.get_axis_copy(0)
                z_cur = float(cal.abs_to_z_disp(0, ac0.act_pos))

                dist = (z_cur - float(z_start)) if int(dir_sign) > 0 else (float(z_start) - z_cur)
                if float(max_dist) > 0.0 and dist >= (float(max_dist) - 1e-6):
                    return None, last_ts, f"{label}_NOT_FOUND_MAXDIST_P1"
                if (time.time() - t_search0) >= float(timeout_s):
                    return None, last_ts, f"{label}_NOT_FOUND_TIMEOUT_P1"
                if self._is_fault(int(getattr(ac0, 'sts', 0)), int(getattr(ac0, 'err', 0))):
                    return None, last_ts, f"{label}_AX0_FAULT(err={int(getattr(ac0,'err',0) or 0)})"
                if _axis_not_moving(z_cur):
                    return None, last_ts, f"{label}_AX0_NOT_MOVING"

                r = _wait_new_gauge(last_ts, 0.35)
                if r is None:
                    continue
                last_ts, s = r
                j = str(getattr(s, "judge", "UNK") or "UNK").strip().upper()
                if j == "UNK":
                    unk_cnt += 1
                    if unk_cnt >= 8:
                        return None, last_ts, f"{label}_JUDGE_UNK"
                    continue
                unk_cnt = 0

                if j == "GO":
                    last_go_z = float(z_cur)
                    hi_cnt = 0
                    continue

                if j in ("HI", "HH"):
                    hi_cnt += 1
                    if hi_cnt >= int(deb_k):
                        edge1 = float(last_go_z)
                        break
                else:
                    hi_cnt = 0

            # stop always
            try:
                self.app._velmove_stop_axis(0)
            except Exception:
                try:
                    self.app.set_cmd_bits(0, set_mask=0, clr_mask=CMD_VELMOVE_REQ)
                    self.app._pulse_cmd_bits(0, CMD_STOP_REQ)
                except Exception:
                    pass

            if self._should_stop():
                return None, last_ts, "ABORT"

            if edge1 is None:
                return None, last_ts, f"{label}_NOT_FOUND_P1"

            # ---------- Pass 2: HI -> GO ----------
            _wait_axis_settled(1.5)
            time.sleep(0.05)
            z_start2 = float(cal.abs_to_z_disp(0, self.app.get_axis_copy(0).act_pos))
            vel_abs2 = -float(v_z) * float(dir_sign) * float(cal.sign_eff(0))
            try:
                self.app._velmove_start_axis(0, vel_abs2, acc=80.0, dec=80.0, jerk=300.0)
            except Exception:
                self._write_fp64(0, OFF_VEL_VELMOVE, vel_abs2)
                self.app.set_cmd_bits(0, set_mask=CMD_VELMOVE_REQ, clr_mask=0)

            t_search1 = time.time()
            unk_cnt = 0
            go_cnt = 0
            seen_hi = False
            last_hi_z = float(z_start2)
            edge2: Optional[float] = None
            last_move_z = None
            last_move_ts = time.time()

            while not self._should_stop():
                ac0 = self.app.get_axis_copy(0)
                z_cur = float(cal.abs_to_z_disp(0, ac0.act_pos))

                dist = (float(z_start2) - z_cur) if int(dir_sign) > 0 else (z_cur - float(z_start2))
                if float(max_dist) > 0.0 and dist >= (float(max_dist) - 1e-6):
                    return None, last_ts, f"{label}_NOT_FOUND_MAXDIST_P2"
                if (time.time() - t_search1) >= float(timeout_s):
                    return None, last_ts, f"{label}_NOT_FOUND_TIMEOUT_P2"
                if self._is_fault(int(getattr(ac0, 'sts', 0)), int(getattr(ac0, 'err', 0))):
                    return None, last_ts, f"{label}_AX0_FAULT(err={int(getattr(ac0,'err',0) or 0)})"
                if _axis_not_moving(z_cur):
                    return None, last_ts, f"{label}_AX0_NOT_MOVING"

                r = _wait_new_gauge(last_ts, 0.35)
                if r is None:
                    continue
                last_ts, s = r
                j = str(getattr(s, "judge", "UNK") or "UNK").strip().upper()
                if j == "UNK":
                    unk_cnt += 1
                    if unk_cnt >= 8:
                        return None, last_ts, f"{label}_JUDGE_UNK"
                    continue
                unk_cnt = 0

                if not seen_hi:
                    if j in ("HI", "HH"):
                        seen_hi = True
                        last_hi_z = float(z_cur)
                    continue

                if j in ("HI", "HH"):
                    last_hi_z = float(z_cur)
                    go_cnt = 0
                    continue

                if j == "GO":
                    go_cnt += 1
                    if go_cnt >= int(deb_k):
                        edge2 = float(last_hi_z)
                        break
                else:
                    go_cnt = 0

            # stop always
            try:
                self.app._velmove_stop_axis(0)
            except Exception:
                try:
                    self.app.set_cmd_bits(0, set_mask=0, clr_mask=CMD_VELMOVE_REQ)
                    self.app._pulse_cmd_bits(0, CMD_STOP_REQ)
                except Exception:
                    pass

            if self._should_stop():
                return None, last_ts, "ABORT"

            if edge2 is None:
                return None, last_ts, f"{label}_NOT_FOUND_P2"

            edge_avg = 0.5 * (float(edge1) + float(edge2))
            return edge_avg, last_ts, "OK"

        z_low_edge: Optional[float] = None
        try:
            z_appr = max(float(z_min), min(float(z_max), float(z_low_approach)))
            abs_tgt = float(cal.z_disp_to_abs(0, z_appr))
            self.app.movea_abs(0, abs_tgt, context="AutoLenLowAppr")
            if not self._wait_in_position(0, abs_tgt, pos_tol=1.0, timeout_s=15.0):
                payload["reason"] = "LOW_APPR_TIMEOUT"
                payload["t_s"] = time.time() - t0
                return payload

            # make sure gauge is updating
            r = _wait_new_gauge(0.0, 1.5)
            if r is None:
                payload["reason"] = "LOW_NO_GAUGE"
                payload["t_s"] = time.time() - t0
                return payload
            last_ts, last_s = r
            judge0 = str(getattr(last_s, "judge", "UNK") or "UNK").strip().upper()
            if judge0 == "UNK":
                payload["reason"] = "LOW_JUDGE_UNK"
                payload["t_s"] = time.time() - t0
                return payload
            if judge0 != "GO":
                payload["reason"] = f"LOW_START_NOT_GO({judge0})"
                payload["t_s"] = time.time() - t0
                return payload

            z_start = float(cal.abs_to_z_disp(0, self.app.get_axis_copy(0).act_pos))
            edge_avg, last_ts, reason_scan = _scan_edge_bidirectional(
                z_start=z_start,
                dir_sign=+1,
                max_dist=float(d_low),
                last_ts0=float(last_ts),
                label="LOW",
            )
            if reason_scan != "OK" or edge_avg is None:
                payload["reason"] = str(reason_scan)
                payload["t_s"] = time.time() - t0
                return payload

            z_low_edge = float(edge_avg)
            payload["z_low"] = z_low_edge

            # optional backoff towards -Z_disp (inside tube)
            if backoff_mm > 1e-6:
                try:
                    z_back = max(float(z_min), min(float(z_max), float(z_low_edge) - float(backoff_mm)))
                    self.app.movea_abs(0, float(cal.z_disp_to_abs(0, z_back)), context="AutoLenLowBackoff")
                    self._wait_in_position(0, float(cal.z_disp_to_abs(0, z_back)), pos_tol=1.2, timeout_s=10.0)
                except Exception:
                    pass

            if self._should_stop():
                payload["reason"] = "ABORT"
                payload["t_s"] = time.time() - t0
                return payload

        except Exception as e:
            payload["reason"] = f"LOW_EXC({e})"
            try:
                self.app._velmove_stop_axis(0)
            except Exception:
                pass
            payload["t_s"] = time.time() - t0
            return payload


        # ---------------- top edge: GO -> HI -> GO (bidirectional average) ----------------
        if pipe_len <= 1e-6:
            payload["reason"] = "PIPE_LEN_ZERO"
            payload["t_s"] = time.time() - t0
            return payload

        z_high_edge: Optional[float] = None
        try:
            z_appr = float(z_low_edge - pipe_len + hi_margin)
            z_appr = max(float(z_min), min(float(z_max), float(z_appr)))
            abs_tgt = float(cal.z_disp_to_abs(0, z_appr))
            self.app.movea_abs(0, abs_tgt, context="AutoLenHighAppr")
            if not self._wait_in_position(0, abs_tgt, pos_tol=1.0, timeout_s=15.0):
                payload["reason"] = "HIGH_APPR_TIMEOUT"
                payload["t_s"] = time.time() - t0
                return payload

            # pre-check: gauge must be updating and should be GO at approach (inside tube)
            r = _wait_new_gauge(0.0, 1.5)
            if r is None:
                payload["reason"] = "HIGH_NO_GAUGE"
                payload["t_s"] = time.time() - t0
                return payload
            last_ts, _s = r
            j0 = str(getattr(_s, "judge", "UNK") or "UNK").strip().upper()
            if j0 == "UNK":
                payload["reason"] = "HIGH_JUDGE_UNK"
                payload["t_s"] = time.time() - t0
                return payload
            if j0 != "GO":
                payload["reason"] = f"HIGH_START_NOT_GO({j0})"
                payload["t_s"] = time.time() - t0
                return payload

            z_start = float(cal.abs_to_z_disp(0, self.app.get_axis_copy(0).act_pos))
            edge_avg, last_ts, reason_scan = _scan_edge_bidirectional(
                z_start=z_start,
                dir_sign=-1,
                max_dist=float(d_high),
                last_ts0=float(last_ts),
                label="HIGH",
            )
            if reason_scan != "OK" or edge_avg is None:
                payload["reason"] = str(reason_scan)
                payload["t_s"] = time.time() - t0
                return payload

            z_high_edge = float(edge_avg)
            payload["z_high"] = z_high_edge

            # optional backoff towards +Z_disp (inside tube)
            if backoff_mm > 1e-6:
                try:
                    z_back = max(float(z_min), min(float(z_max), float(z_high_edge) + float(backoff_mm)))
                    self.app.movea_abs(0, float(cal.z_disp_to_abs(0, z_back)), context="AutoLenHighBackoff")
                    self._wait_in_position(0, float(cal.z_disp_to_abs(0, z_back)), pos_tol=1.2, timeout_s=10.0)
                except Exception:
                    pass

            if self._should_stop():
                payload["reason"] = "ABORT"
                payload["t_s"] = time.time() - t0
                return payload

        except Exception as e:
            payload["reason"] = f"HIGH_EXC({e})"
            try:
                self.app._velmove_stop_axis(0)
            except Exception:
                pass
            payload["t_s"] = time.time() - t0
            return payload

# compute length
        try:
            length_mm = float(z_low_edge - z_high_edge)
            payload["length_mm"] = length_mm
            payload["ok"] = True
            payload["reason"] = "OK"
        except Exception:
            payload["reason"] = "LEN_CALC_FAIL"

        payload["t_s"] = time.time() - t0

        # lightweight telemetry for debugging
        try:
            log(
                "AUTO_LEN_RESULT",
                ok=bool(payload.get("ok")),
                reason=str(payload.get("reason", "")),
                z_low=payload.get("z_low"),
                z_high=payload.get("z_high"),
                length_mm=payload.get("length_mm"),
                t_s=payload.get("t_s"),
            )
        except Exception:
            pass
        return payload

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
            try:
                r0 = self.app.get_recipe_copy()
                log("AUTO_FLOW_START", section_count=getattr(r0,"section_count",None), points_per_rev=getattr(r0,"points_per_rev",None), min_bin_coverage=getattr(r0,"min_bin_coverage",None), timeout_s=getattr(r0,"sample_timeout_s",None), max_revolutions=getattr(r0,"max_revolutions",None))
            except Exception as e:
                log("AUTO_FLOW_START", err=str(e))

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



            # ---------------------
            # S20/S50: Clamp outputs + AX2 positioning + operator confirm (no clamp feedback)
            # ---------------------
            ax_clamp = 2  # AX2 center clamp
            try:
                ac2 = self.app.get_axis_copy(ax_clamp)
                if self._is_fault(int(ac2.sts), int(ac2.err)):
                    raise RuntimeError(f"中心架 AX2 故障，Err={int(ac2.err)}")
                if not self._is_enabled(int(ac2.sts)):
                    self.app.set_cmd_bits(ax_clamp, set_mask=CMD_EN_REQ, clr_mask=0)
                    time.sleep(0.15)
            except Exception as e:
                raise RuntimeError(f"中心架 AX2 使能失败：{e}")

            # Clamp prep for length/rotate:
            # - main clamp (Y10) must stay clamped during rotate measurement
            # - sub clamp (Y11) released for length step, then clamped for rotate step
            try:
                self.app.ui_q.put(("auto_state", {"state": "PREP", "msg": "夹爪准备：主爪夹紧、从爪松开"}))
            except Exception:
                pass
            try:
                self.app.plc_write_y_point(10, 1)  # Y10 主爪夹紧
                self.app.plc_write_y_point(11, 0)  # Y11 从爪松开
            except Exception:
                pass
            time.sleep(0.25)

            # Optional: move AX2 to length measurement position
            if bool(getattr(recipe, 'len_enable', False)):
                if bool(getattr(recipe, 'ax2_len_valid', False)):
                    try:
                        tgt2 = float(getattr(recipe, 'ax2_len_abs', 0.0))
                        tgt2 = self.app.apply_soft_limits_abs(ax_clamp, tgt2, strict=True, context='AUTO_AX2_LEN')
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
                        tgt0 = self.app.apply_soft_limits_abs(0, tgt0, strict=True, context='AUTO_AX0_STANDBY_AFTER_LEN')
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

            # Move AX2 to rotate measurement position (required for rotate stage)
            if bool(getattr(recipe, 'ax2_rot_valid', False)):
                try:
                    tgt2r = float(getattr(recipe, 'ax2_rot_abs', 0.0))
                    tgt2r = self.app.apply_soft_limits_abs(ax_clamp, tgt2r, strict=True, context='AUTO_AX2_ROT')
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

            # Clamp sub jaw for rotate stage, then wait operator confirm
            try:
                self.app.plc_write_y_point(11, 1)  # Y11 从爪夹紧
            except Exception:
                pass
            time.sleep(0.25)

            if self._should_stop():
                self.app.ui_q.put(("auto_state", {"state": "STOP", "msg": "用户停止"}))
                return

            try:
                msg = "请确认从爪已夹紧。\n\n- 按 X3 或点击‘确认’继续\n- 点击‘停止流程’可中断"
                res = self.app.operator_confirm("夹紧确认", msg, allow_stop=True, timeout_s=60.0)
            except Exception:
                res = 'timeout'

            if res != 'confirm' or self._should_stop():
                try:
                    self.app.ui_q.put(("auto_state", {"state": "STOP", "msg": f"操作员取消/超时：{res}"}))
                except Exception:
                    pass
                return
            # Prepare rotate axis (AX3): enable + ensure velmove params
            a3 = self.app.get_axis_copy(3)
            if self._is_fault(int(a3.sts), int(a3.err)):
                raise RuntimeError(f"旋转轴 AX3 故障，Err={int(a3.err)}")

            if not self._is_enabled(int(a3.sts)):
                self.app.set_cmd_bits(3, set_mask=CMD_EN_REQ, clr_mask=0)
                time.sleep(0.25)

            # Apply rotation speed from recipe every time (AX3 VelMove speed),
            # to make behavior deterministic and not rely on previous manual settings.
            try:
                rot_v = float(getattr(recipe, "rot_vel_velmove", getattr(recipe, "rot_speed", 200.0)) or 0.0)
            except Exception:
                rot_v = 200.0
            if abs(rot_v) <= 1e-9:
                rot_v = 200.0
            try:
                self._write_fp64(3, OFF_VEL_VELMOVE, float(rot_v))
            except Exception:
                pass

            self._ensure_velmove_setpoints(3)
            time.sleep(0.05)

            # start rotate (AX3) - level command
            self.app.set_cmd_bits(3, set_mask=CMD_VELMOVE_REQ, clr_mask=0)
            time.sleep(0.20)

            # Clear results first
            # NOTE: UI 清空（含 RunId/开始时间）已在 App._auto_start() 中完成。
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
                    0: (float(self.app.get_axis_copy(0).softlim_pos), float(self.app.get_axis_copy(0).softlim_neg)),
                    1: (float(self.app.get_axis_copy(1).softlim_pos), float(self.app.get_axis_copy(1).softlim_neg)),
                    4: (float(self.app.get_axis_copy(4).softlim_pos), float(self.app.get_axis_copy(4).softlim_neg)),
                }

                try:
                    # For section planning, keepout should be referenced to AX2 rotation measurement position if available.
                    ax2_abs = float(self.app._get_ax2_keepout_ref_abs(prefer_rot=True))
                except Exception:
                    ax2_abs = None
                tg = cal.od_z_disp_to_targets(z_od_disp, ax2_abs=ax2_abs, softlims_abs=softlims)
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
                    targets[ax] = self.app.apply_soft_limits_abs(
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


                keep_spinning = bool(getattr(recipe, 'split_keep_spinning', True))


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
                            self.app.set_cmd_bits(3, set_mask=0, clr_mask=CMD_VELMOVE_REQ)
                            self.app._pulse_cmd_bits(3, CMD_STOP_REQ)
                            t_stop0 = time.time()
                            while (time.time() - t_stop0) < 10.0:
                                if self._should_stop():
                                    break
                                ac3s = self.app.get_axis_copy(3)
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
                pp_mode = str(getattr(recipe, "pp_mode", "p99_p1") or "p99_p1")

                def _pp_strict(a: np.ndarray) -> float:
                    return float(_robust_span(a, "strict"))

                def _pp_robust(a: np.ndarray, **_kw) -> float:
                    """Robust peak-to-peak/span.

                    Compatibility: some older call-sites pass `trim_ratio=`.
                    Robustness is controlled by recipe.pp_mode, so we ignore
                    extra keywords.
                    """
                    return float(_robust_span(a, pp_mode))

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

                # Legacy path: derive OD stats from fitted circle (coords_od)
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
                self.app.set_cmd_bits(3, set_mask=0, clr_mask=CMD_VELMOVE_REQ)
                self.app._pulse_cmd_bits(3, CMD_STOP_REQ)
                t0 = time.time()
                while (time.time() - t0) < 10.0:
                    if self._should_stop():
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

                    # Soft limits for standby return: clamp if needed (do not block completion).
                    for ax, tgt in list(targets2.items()):
                        targets2[ax] = self.app.apply_soft_limits_abs(
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

            # ---------------------
            # Step5: Wait operator to release clamps (X4) for a short window.
            # Policy: timeout => do NOT release (keep clamped).
            # ---------------------
            try:
                wait_s = 15.0
                try:
                    self.app.ui_q.put(
                        (
                            "auto_state",
                            {
                                "state": "DONE",
                                "msg": f"测量完成，等待X4松爪({int(wait_s)}s)",
                            },
                        )
                    )
                except Exception:
                    pass

                t_wait0 = time.time()
                try:
                    last = 1 if bool(self.app.get_x_point(4)) else 0
                except Exception:
                    last = 0

                released = False
                while (time.time() - t_wait0) < float(wait_s):
                    if self._should_stop():
                        break
                    try:
                        cur = 1 if bool(self.app.get_x_point(4)) else 0
                    except Exception:
                        cur = 0

                    # rising edge X4
                    if (last == 0) and (cur == 1):
                        try:
                            self.app.plc_write_y_point(10, 0)  # Y10 主爪松开
                            self.app.plc_write_y_point(11, 0)  # Y11 从爪松开
                        except Exception:
                            pass
                        released = True
                        try:
                            self.app.ui_q.put(
                                (
                                    "auto_state",
                                    {
                                        "state": "DONE",
                                        "msg": "已执行松爪（Y10=0,Y11=0）",
                                    },
                                )
                            )
                        except Exception:
                            pass
                        break

                    last = cur
                    time.sleep(0.05)

                if (not released) and (not self._should_stop()):
                    try:
                        self.app.ui_q.put(
                            (
                                "auto_state",
                                {
                                    "state": "DONE",
                                    "msg": "松爪超时未执行（保持夹紧）",
                                },
                            )
                        )
                    except Exception:
                        pass
            except Exception:
                # never block completion on post-wait failures
                pass

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
            self.app.set_cmd_bits(3, set_mask=0, clr_mask=CMD_VELMOVE_REQ)
            # 保险起见给一个 STOP 脉冲（可选，但建议）
            self.app._pulse_cmd_bits(3, CMD_STOP_REQ)

            # 若用户停止：对所有轴发一次 STOP/HALT，避免继续运动
            if self._should_stop():
                try:
                    self.app.abort_motion()
                except Exception:
                    pass

    def _wait_in_position(
        self, axis: int, tgt_abs: float, pos_tol: float, timeout_s: float
    ) -> bool:
        t0 = time.time()
        while (time.time() - t0) < float(timeout_s):
            if self._should_stop():
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


    # ID fit helpers (Chord OUT4 + m OUT5)
    # ------------------------------
    def _idcal_get_delta_c_active(self) -> float:
        """Get active delta_c(mm) for ID chord correction.

        Priority:
        1) Read from app calibration file (./calibration/id_calibration.json)
        2) Fallback: parse app.idcal_delta_active_var if present (stringvar)
        Returns 0.0 if not available.
        """
        # 1) file is source of truth
        try:
            p = None
            if hasattr(self.app, "_idcal_file"):
                p = self.app._idcal_file()  # type: ignore[attr-defined]
            elif hasattr(self.app, "_app_root_dir"):
                # fallback path
                try:
                    from pathlib import Path
                    p = Path(self.app._app_root_dir()) / "calibration" / "id_calibration.json"  # type: ignore
                except Exception:
                    p = None
            if p is not None:
                import json as _json
                import os as _os
                if _os.path.exists(str(p)):
                    with open(p, "r", encoding="utf-8") as f:
                        data = _json.load(f) or {}
                    v = data.get("delta_c_mm", None)
                    if v is not None and math.isfinite(float(v)):
                        return float(v)
        except Exception:
            pass

        # 2) UI var fallback
        try:
            v = getattr(self.app, "idcal_delta_active_var", None)
            if v is not None:
                s = v.get() if hasattr(v, "get") else str(v)
                s = str(s).strip()
                if s and s not in ("--", "None", "nan", "NaN"):
                    return float(s)
        except Exception:
            pass
        return 0.0

    def _idcal_fit_diameter_local(self, theta_deg: np.ndarray, c_mm: np.ndarray, m_mm: np.ndarray, delta_c: float) -> dict:
        """Local copy of App._idcal_fit_diameter, used as fallback."""
        th = np.deg2rad(theta_deg.astype(float))
        m = m_mm.astype(float)

        # fit m = x0 + A*cos(th) + B*sin(th)
        X = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th)])
        beta, *_ = np.linalg.lstsq(X, m, rcond=None)
        x0, A, B = float(beta[0]), float(beta[1]), float(beta[2])

        e = float(math.hypot(A, B))
        phi = float(math.atan2(-B, A))  # m = x0 + e*cos(theta+phi)

        s = np.sin(th + phi)
        c_corr = np.clip(c_mm.astype(float) + float(delta_c), 0.001, None)
        Z = (0.5 * c_corr) ** 2 + (e * s) ** 2
        X2 = np.column_stack([np.ones_like(s), (-2.0 * e * s)])
        beta2, *_ = np.linalg.lstsq(X2, Z, rcond=None)
        R2p = float(beta2[0])
        y0 = float(beta2[1])
        R2 = float(R2p + y0 * y0)
        R = float(math.sqrt(max(R2, 0.0)))

        pred_R2 = (0.5 * c_corr) ** 2 + (y0 + e * s) ** 2
        rmse_R2 = float(math.sqrt(max(0.0, float(np.mean((pred_R2 - R2) ** 2)))))
        return {"R": R, "diam": 2.0 * R, "e": e, "phi_rad": phi, "x0": x0, "y0": y0, "rmse_R2": rmse_R2}

    def _id_fit_from_raw_points(self, raw_points: list[dict], delta_c: float, *, theta_delay_s: float = 0.0) -> tuple[Optional[dict], Optional[np.ndarray]]:
        """Fit ID diameter from raw points (needs theta_deg + id_c_mm + id_m_mm).

        Returns: (fit_dict, diam_series_Di)
        - fit_dict includes: diam,e,phi_rad,y0,rmse_R2...
        - diam_series_Di: per-sample diameter reconstructed from (c_corr, y0+e*sin(th+phi))
        """
        try:
            th_list = []
            ts_list = []
            c_list = []
            m_list = []
            for p in raw_points:
                if not isinstance(p, dict):
                    continue
                th = p.get("theta_deg", None)
                ts = p.get("ts", None)
                c = p.get("id_c_mm", None)
                mm = p.get("id_m_mm", None)
                if th is None or ts is None or c is None or mm is None:
                    continue
                thf = float(th)
                tsf = float(ts)
                cf = float(c)
                mf = float(mm)
                if (not math.isfinite(thf)) or (not math.isfinite(tsf)) or (not math.isfinite(cf)) or (not math.isfinite(mf)):
                    continue
                th_list.append(thf)
                ts_list.append(tsf)
                c_list.append(cf)
                m_list.append(mf)

            if len(th_list) < 8:
                return None, None

            th_arr = np.asarray(th_list, dtype=float)
            c_arr = np.asarray(c_list, dtype=float)
            m_arr = np.asarray(m_list, dtype=float)
            ts_arr = np.asarray(ts_list, dtype=float)

            # Optional theta delay compensation (shift theta by omega*delay)
            try:
                delay_s = float(theta_delay_s or 0.0)
            except Exception:
                delay_s = 0.0
            if abs(delay_s) > 1e-9:
                omega = _estimate_omega_deg_s(th_list, ts_list)
                th_arr = np.asarray([_theta_apply_delay(float(th), float(omega), float(delay_s)) for th in th_arr], dtype=float)

            # prefer App implementation if present (keeps consistent with calibration page)
            fit = None
            if hasattr(self.app, "_idcal_fit_diameter"):
                try:
                    fit = self.app._idcal_fit_diameter(th_arr, c_arr, m_arr, float(delta_c))  # type: ignore[attr-defined]
                except Exception:
                    fit = None
            if fit is None:
                fit = self._idcal_fit_diameter_local(th_arr, c_arr, m_arr, float(delta_c))

            # derive ID center vector from m(theta)=x0 + ex*cos(theta) + ey*sin(theta)
            # (ex,ey) share the same (cos,sin) basis with OD edge-based eccentricity.
            try:
                th_rad1 = np.deg2rad(th_arr.astype(float))
                X1 = np.column_stack([np.ones_like(th_rad1), np.cos(th_rad1), np.sin(th_rad1)])
                beta1, *_ = np.linalg.lstsq(X1, m_arr.astype(float), rcond=None)
                _x0, _ex, _ey = float(beta1[0]), float(beta1[1]), float(beta1[2])
                if isinstance(fit, dict):
                    fit.setdefault("x0", _x0)
                    fit["ex"] = _ex
                    fit["ey"] = _ey
                    fit["phi_xy_rad"] = float(math.atan2(_ey, _ex))
            except Exception:
                pass

            diam = float(fit.get("diam", 0.0) or 0.0)
            if (not math.isfinite(diam)) or diam <= 0.0:
                return None, None

            # reconstruct per-sample diameter series
            th_rad = np.deg2rad(th_arr.astype(float))
            phi = float(fit.get("phi_rad", 0.0) or 0.0)
            e = float(fit.get("e", 0.0) or 0.0)
            y0 = float(fit.get("y0", 0.0) or 0.0)
            s = np.sin(th_rad + phi)
            c_corr = np.clip(c_arr + float(delta_c), 0.001, None)
            pred_R2 = (0.5 * c_corr) ** 2 + (y0 + e * s) ** 2
            Di = 2.0 * np.sqrt(np.clip(pred_R2, 0.0, None))
            return fit, Di
        except Exception:
            return None, None

    def _sample_circle_points_dual(
        self,
        recipe: Recipe,
        section_idx: int = 0,
        *,
        sample_od: bool = True,
        sample_id: bool = True,
        phase: str = "SYNC",
    ) -> Tuple[np.ndarray, np.ndarray, str, str, list]:
        """Equal-angle sampling (OD/ID can be sampled independently).

        - Angle source: AX3 act_pos (deg)
        - OD source: gauge (real or simulated)
        - ID source: CL-3000 OUT3 (mapped via PLC) or simulated displacement meter

        Args:
            sample_od: sample OD in this pass
            sample_id: sample ID in this pass
            phase: tag written into raw_points for export/diagnostics (e.g. 'OD','ID','SYNC')

        Returns:
            (coords_od, coords_id, raw_last_od, raw_last_id, raw_points)
        """
        if (not sample_od) and (not sample_id):
            raise ValueError("sample_od and sample_id cannot both be False")

        n = max(3, int(getattr(recipe, "points_per_rev", 120)))
        min_cov = float(getattr(recipe, "min_bin_coverage", 0.95))
        min_cov = max(0.0, min(1.0, min_cov))
        timeout_s = float(getattr(recipe, "sample_timeout_s", 5.0))
        timeout_s = max(0.5, timeout_s)
        max_revs = float(getattr(recipe, "max_revolutions", 2.0))
        max_revs = max(0.25, max_revs)

        # speedtest: optionally skip ID reads to improve sampling throughput
        disable_id_modbus = (
            (bool(getattr(recipe, 'disable_id_modbus', False)) or bool(SPEEDTEST_DISABLE_ID_MODBUS))
            and (not bool(getattr(self.app, "sim_disp_enabled", False)))
            and bool(sample_id)
        )
        # In SPLIT mode, ID pass must not be disabled by OD-only switch.
        try:
            sm2 = str(getattr(recipe, "scan_mode", "sync") or "sync").strip().lower()
            if bool(sample_id) and sm2.startswith("split"):
                disable_id_modbus = False
        except Exception:
            pass
        # Single-probe ID rescue requires OUT2, so never disable ID reads in that mode.
        try:
            id_single_enable = bool(getattr(recipe, "id_single_enable", False))
        except Exception:
            id_single_enable = False
        if bool(sample_id) and id_single_enable:
            disable_id_modbus = False

        # Fit strategy affects sampling stop criteria.
        # - a: raw-point fit, keep all accepted raw points, bins only for coverage statistics.
        #      Sampling should run until max_revs (unless timeout/stop).
        # - b: raw-point fit with per-bin balancing weights. Can stop when coverage reached.
        # - c: per-bin radius averaging. Can stop when coverage reached.
        fs = str(getattr(recipe, "fit_strategy", "b 原始点按bin权重均衡") or "").strip().lower()
        mode = "b"
        if fs.startswith("a"):
            mode = "a"
        elif fs.startswith("b"):
            mode = "b"
        elif fs.startswith("c"):
            mode = "c"

        # Reduce background polling during sampling to improve sync-read latency.
        self.app.set_plc_poll_profile("sampling")
        try:
            # 等角bin：将 0~360° 划分为 n 个bin
            sum_x_od = [0.0] * n
            sum_y_od = [0.0] * n
            sum_x_id = [0.0] * n
            sum_y_id = [0.0] * n
            cnt = [0] * n
            sum_r_od = [0.0] * n
            sum_r_id = [0.0] * n

            # debug counters
            iters = 0
            skip_no_new_od = 0
            skip_od_outlier = 0
            skip_id_none = 0
            skip_id_outlier = 0
            bin_fill_logs = 0

            od_min = None
            od_max = None
            id_min = None
            id_max = None
            od_min_meta = None
            od_max_meta = None
            id_min_meta = None
            id_max_meta = None

            filled = 0
            raw_last_od = ""
            raw_last_id = ""
            last_id_cnt4 = None  # gate duplicate CL OUT4 samples when using id_use_fit
            last_id_cnt2 = None  # gate duplicate CL OUT2 samples in single-probe mode

            # Raw sample points for export/diagnostics
            raw_points: list[dict] = []

            t_start = time.time()
            need = max(3, int(math.ceil(min_cov * n)))
            reason = "COV"  # COV / TIMEOUT / REV
            prev_theta = None
            unwrapped_deg = 0.0
            revs = 0.0

            while True:
                if self._should_stop():
                    raise RuntimeError("测量被用户停止")

                iters += 1

                if mode != "a" and filled >= need:
                    reason = "COV"
                    break
                if revs >= max_revs:
                    reason = "REV"
                    break
                if (time.time() - t_start) >= timeout_s:
                    reason = "TIMEOUT"
                    break

                # Angle snapshot first (deg)
                theta_deg = None
                try:
                    theta_deg = self.app.read_axis_act_pos_deg_sync(axis=3, timeout_s=0.5)
                except Exception:
                    theta_deg = None
                if theta_deg is None:
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

                # map to bin
                b = int((theta_deg / 360.0) * n)
                if b >= n:
                    b = 0

                # ---------------- OD ----------------
                od = None
                od_out1 = None
                od_out2 = None
                od_B = None
                od_map_out1 = 'L'
                od_L = None
                od_R = None
                od_delta = None

                if sample_od:
                    if self.app.sim_gauge_enabled:
                        od_val, raw = self.app.simulate_gauge_once(recipe)
                        od = float(od_val)
                        od_out1 = float(od)
                        raw_last_od = raw
                    else:
                        gw = self.app.gauge_worker
                        if gw is None:
                            raise RuntimeError("测径仪未启用：请勾选“模拟测径仪”或连接真实串口。")

                        # Ensure OUT1/OUT2 are available for edge-based OD algorithm
                        if bool(getattr(recipe, 'od_use_edges', False)):
                            try:
                                req_cmd = str(getattr(gw, 'request_cmd', '') or '').strip().upper()
                                if not req_cmd.startswith('M0'):
                                    gw.configure(
                                        enabled=gw.enabled,
                                        port=gw.port,
                                        baud=gw.baud,
                                        timeout_s=gw.timeout_s,
                                        eol=gw.eol,
                                        request_cmd='M0,1',
                                        bytesize=gw.bytesize,
                                        parity=gw.parity,
                                        stopbits=gw.stopbits,
                                    )
                            except Exception:
                                pass

                        t_req = time.time()
                        gw.send_request()
                        t0 = time.time()
                        while (time.time() - t0) < 1.2:
                            if self._should_stop():
                                raise RuntimeError("测量被用户停止")
                            s = gw.get_last()
                            if s and s.ts >= t_req:
                                od_use_edges = bool(getattr(recipe, 'od_use_edges', False))
                                od_std = float(getattr(recipe, 'od_std_mm', 0.0) or 0.0)
                                od_tol = float(getattr(recipe, 'od_tol_mm', 0.0) or 0.0)
                                out1 = float(s.od)
                                out2 = None
                                try:
                                    out2 = None if getattr(s, 'od2', None) is None else float(getattr(s, 'od2'))
                                except Exception:
                                    out2 = None

                                B_active = None
                                map_out1 = 'L'
                                L_val = None
                                R_val = None
                                delta_val = None

                                if od_use_edges:
                                    try:
                                        b_txt = str(getattr(self.app, 'odcal_B_active_var', None).get()).strip()
                                        if b_txt and b_txt != '--':
                                            B_active = float(b_txt)
                                    except Exception:
                                        B_active = None
                                    try:
                                        map_out1 = str(getattr(self.app, 'odcal_map_out1_var', None).get()).strip().upper() or 'L'
                                        if map_out1 not in ('L', 'R'):
                                            map_out1 = 'L'
                                    except Exception:
                                        map_out1 = 'L'

                                    if B_active is None:
                                        raise RuntimeError('新外径算法(边缘距离)需要先标定B值')
                                    if out2 is None:
                                        raise RuntimeError('新外径算法(边缘距离)需要同时读取OUT1/OUT2 (建议选择 M0,1)')

                                    if map_out1 == 'L':
                                        L_val, R_val = out1, out2
                                    else:
                                        L_val, R_val = out2, out1
                                    od_calc = float(B_active) - (float(L_val) + float(R_val))
                                    delta_val = 0.5 * (float(L_val) - float(R_val))
                                else:
                                    od_calc = out1

                                # plausibility filter: drop extreme outliers
                                if od_std > 0.0:
                                    margin = max(5.0, 10.0 * max(od_tol, 0.1), 0.05 * od_std)
                                    if (not math.isfinite(od_calc)) or abs(float(od_calc) - od_std) > margin:
                                        skip_od_outlier += 1
                                        od = None
                                        break

                                od = float(od_calc)
                                od_out1 = out1
                                od_out2 = out2
                                od_B = B_active
                                od_map_out1 = map_out1
                                od_L = L_val
                                od_R = R_val
                                od_delta = delta_val
                                raw_last_od = s.raw
                                break
                            time.sleep(0.02)
                        else:
                            skip_no_new_od += 1
                            od = None

                else:
                    raw_last_od = "OD_SKIPPED"

                if sample_od and (od is None):
                    continue

                # ---------------- ID ----------------
                id_mm = None
                cnt_i = None
                id_x1_mm = None
                id_x2_mm = None
                id_c_mm = None
                id_m_mm = None
                id_cnt_out4 = None
                id_cnt_out5 = None
                id_out2_mm = None
                id_cnt_out2 = None
                ecc_x = 0.0
                ecc_y = 0.0

                if sample_id:
                    if id_single_enable:
                        if getattr(self.app, "sim_disp_enabled", False):
                            id_val, raw_id = self.app.simulate_disp_once(recipe)
                            id_out2_mm = float(id_val) if id_val is not None else None
                            raw_last_id = raw_id
                            cnt_i = None
                        else:
                            x1_mm, x2_mm, c_mm, m_mm, raw_dict, cnt_dict = self.app.read_cl_out145_sync(timeout_s=0.5)
                            id_out2_mm = x2_mm
                            try:
                                id_cnt_out2 = int(cnt_dict.get("out2", 0)) if isinstance(cnt_dict, dict) else None
                            except Exception:
                                id_cnt_out2 = None
                            # gate duplicates by OUT2 update counter if available
                            if id_cnt_out2 is not None:
                                if last_id_cnt2 is not None and int(id_cnt_out2) == int(last_id_cnt2):
                                    skip_id_none += 1
                                    continue
                                last_id_cnt2 = int(id_cnt_out2)
                            cnt_i = id_cnt_out2
                            raw_last_id = f"OUT2={raw_dict.get('out2', None)} cnt2={id_cnt_out2}"
                        if id_out2_mm is None:
                            skip_id_none += 1
                            continue
                    elif disable_id_modbus:
                        # placeholder ID so downstream code stays intact
                        id_mm = float(getattr(recipe, "id_std_mm", 0.0) or 0.0)
                        if (not math.isfinite(id_mm)) or id_mm <= 0.0:
                            id_mm = 80.0
                        raw_last_id = "ID_DISABLED"
                    elif getattr(self.app, "sim_disp_enabled", False):
                        id_val, raw_id = self.app.simulate_disp_once(recipe)
                        id_mm = float(id_val) if id_val is not None else None
                        raw_last_id = raw_id
                        ecc_x = 0.05 * math.sin(0.9 * float(section_idx))
                        ecc_y = 0.05 * math.cos(0.7 * float(section_idx))
                    else:
                        if bool(getattr(recipe, "id_use_fit", False)):
                            id_x1_mm, id_x2_mm, id_c_mm, id_m_mm, raw_dict, cnt_dict = self.app.read_cl_out145_sync(timeout_s=0.5)
                            try:
                                id_cnt_out4 = int(cnt_dict.get("out4", 0)) if isinstance(cnt_dict, dict) else None
                                id_cnt_out5 = int(cnt_dict.get("out5", 0)) if isinstance(cnt_dict, dict) else None
                            except Exception:
                                id_cnt_out4 = None
                                id_cnt_out5 = None

                            # gate duplicates by OUT4 update counter if available
                            if id_cnt_out4 is not None:
                                if last_id_cnt4 is not None and int(id_cnt_out4) == int(last_id_cnt4):
                                    skip_id_none += 1
                                    continue
                                last_id_cnt4 = int(id_cnt_out4)

                            cnt_i = id_cnt_out4
                            raw_last_id = f"OUT4={raw_dict.get('out4', None)} OUT5={raw_dict.get('out5', None)} cnt4={id_cnt_out4} cnt5={id_cnt_out5}"
                            if id_c_mm is None:
                                skip_id_none += 1
                                continue

                            # Use OUT5 as m if valid, else derive from x1/x2.
                            m_used = id_m_mm
                            try:
                                if m_used is None or (not math.isfinite(float(m_used))):
                                    if id_x1_mm is not None and id_x2_mm is not None and math.isfinite(float(id_x1_mm)) and math.isfinite(float(id_x2_mm)):
                                        m_used = 0.5 * (float(id_x1_mm) - float(id_x2_mm))
                            except Exception:
                                pass
                            if m_used is None or (not math.isfinite(float(m_used))):
                                skip_id_none += 1
                                continue
                            id_m_mm = float(m_used)

                            # For legacy coords_id synthesis, treat chord length as "diameter-like" scalar.
                            id_mm = float(id_c_mm)

                            # plausibility filter on chord (must be positive and not wildly larger than nominal D)
                            id_std = float(getattr(recipe, "id_std_mm", 0.0) or 0.0)
                            od_tol = float(getattr(recipe, "od_tol_mm", 0.0) or 0.0)
                            if id_std > 0.0:
                                margin = max(10.0, 20.0 * max(od_tol, 0.1), 0.10 * id_std)
                                if (not math.isfinite(float(id_mm))) or (float(id_mm) <= 0.0) or (float(id_mm) > (id_std + margin)):
                                    skip_id_outlier += 1
                                    continue

                        else:
                            id_val, raw_i, cnt_i = self.app.read_cl_out3_sync(timeout_s=0.5)
                            raw_last_id = f"OUT3={raw_i} cnt={cnt_i}"
                            if id_val is None:
                                skip_id_none += 1
                                continue
                            id_mm = float(id_val)

                            id_std = float(getattr(recipe, "id_std_mm", 0.0) or 0.0)
                            od_tol = float(getattr(recipe, "od_tol_mm", 0.0) or 0.0)
                            if id_std > 0.0:
                                margin = max(5.0, 10.0 * max(od_tol, 0.1), 0.05 * id_std)
                                if (not math.isfinite(float(id_mm))) or abs(float(id_mm) - id_std) > margin:
                                    skip_id_outlier += 1
                                    continue

                else:
                    raw_last_id = "ID_SKIPPED"

                if sample_id and (not id_single_enable) and (id_mm is None):
                    continue
                if sample_id and id_single_enable and (id_out2_mm is None):
                    continue

                # ---------------- accepted sample ----------------
                # track extremes on accepted samples
                try:
                    if sample_od and od is not None:
                        if od_min is None or float(od) < float(od_min):
                            od_min = float(od)
                            od_min_meta = (float(theta_deg), raw_last_od)
                            log("SAMPLE_OD_MIN", section=section_idx+1, theta_deg=float(theta_deg), od=float(od), raw=raw_last_od)
                        if od_max is None or float(od) > float(od_max):
                            od_max = float(od)
                            od_max_meta = (float(theta_deg), raw_last_od)
                            log("SAMPLE_OD_MAX", section=section_idx+1, theta_deg=float(theta_deg), od=float(od), raw=raw_last_od)
                    if sample_id and id_mm is not None:
                        if id_min is None or float(id_mm) < float(id_min):
                            id_min = float(id_mm)
                            id_min_meta = (float(theta_deg), raw_last_id)
                            log("SAMPLE_ID_MIN", section=section_idx+1, theta_deg=float(theta_deg), id=float(id_mm), raw=raw_last_id)
                        if id_max is None or float(id_mm) > float(id_max):
                            id_max = float(id_mm)
                            id_max_meta = (float(theta_deg), raw_last_id)
                            log("SAMPLE_ID_MAX", section=section_idx+1, theta_deg=float(theta_deg), id=float(id_mm), raw=raw_last_id)
                except Exception:
                    pass

                # record raw point
                try:
                    raw_points.append({
                        "phase": str(phase),
                        "ts": float(time.time()),
                        "theta_deg": float(theta_deg),
                        "od_mm": None if (od is None) else float(od),
                        "id_mm": None if (id_mm is None) else float(id_mm),
                        "id_out2_mm": None if (id_out2_mm is None) else float(id_out2_mm),
                        "id_c_mm": None if id_c_mm is None else float(id_c_mm),
                        "id_m_mm": None if id_m_mm is None else float(id_m_mm),
                        "id_x1_mm": None if id_x1_mm is None else float(id_x1_mm),
                        "id_x2_mm": None if id_x2_mm is None else float(id_x2_mm),
                        "id_cnt_out4": None if id_cnt_out4 is None else int(id_cnt_out4),
                        "id_cnt_out5": None if id_cnt_out5 is None else int(id_cnt_out5),
                        "id_cnt_out2": None if id_cnt_out2 is None else int(id_cnt_out2),
                        "od_out1": None if od_out1 is None else float(od_out1),
                        "od_out2": None if od_out2 is None else float(od_out2),
                        "od_B": None if od_B is None else float(od_B),
                        "od_map_out1": str(od_map_out1),
                        "od_L": None if od_L is None else float(od_L),
                        "od_R": None if od_R is None else float(od_R),
                        "od_delta": None if od_delta is None else float(od_delta),
                        "cl_cnt": None if cnt_i is None else int(cnt_i),
                        "bin": int(b),
                        "raw_od": str(raw_last_od),
                        "raw_id": str(raw_last_id),
                    })
                except Exception:
                    pass

                if cnt[b] == 0:
                    filled += 1
                    try:
                        bin_fill_logs += 1
                        log("SAMPLE_BIN_FILL", section=section_idx+1, bin=b, theta_deg=float(theta_deg),
                            od=None if od is None else float(od),
                            id=None if id_mm is None else float(id_mm),
                            raw_od=raw_last_od, raw_id=raw_last_id)
                    except Exception:
                        pass
                cnt[b] += 1

                if sample_od and od is not None:
                    r_od = 0.5 * float(od)
                    x_od = r_od * math.cos(theta)
                    y_od = r_od * math.sin(theta)
                    sum_x_od[b] += x_od
                    sum_y_od[b] += y_od
                    sum_r_od[b] += float(r_od)

                if sample_id and (not id_single_enable) and id_mm is not None:
                    r_id = 0.5 * float(id_mm)
                    x_id = r_id * math.cos(theta) + float(ecc_x)
                    y_id = r_id * math.sin(theta) + float(ecc_y)
                    sum_x_id[b] += x_id
                    sum_y_id[b] += y_id
                    sum_r_id[b] += float(r_id)

                time.sleep(0.005)

            # build coords according to fit strategy (mode already determined above)

            max_gap_deg = _max_gap_deg_from_bins(cnt, n)
            self._last_sample_max_gap_deg = float(max_gap_deg)

            coords_od: List[Tuple[float, float]] = []
            coords_id: List[Tuple[float, float]] = []
            self._last_fit_weights_od = None
            self._last_fit_weights_id = None

            if mode == "c":
                miss = 0
                for i in range(n):
                    if cnt[i] > 0:
                        th = math.radians((float(i) + 0.5) * (360.0 / float(n)))
                        if sample_od:
                            r_od_bin = float(sum_r_od[i]) / float(cnt[i])
                            coords_od.append((r_od_bin * math.cos(th), r_od_bin * math.sin(th)))
                        if sample_id and (not id_single_enable):
                            r_id_bin = float(sum_r_id[i]) / float(cnt[i])
                            coords_id.append((r_id_bin * math.cos(th), r_id_bin * math.sin(th)))
                    else:
                        miss += 1
            else:
                # Raw-point based strategies
                for p in raw_points:
                    if not isinstance(p, dict):
                        continue
                    try:
                        th_deg = float(p.get("theta_deg"))
                        th = math.radians(th_deg)
                    except Exception:
                        continue
                    if sample_od:
                        od_v = p.get("od_mm", None)
                        if od_v is not None:
                            try:
                                od_f = float(od_v)
                                r_od = 0.5 * od_f
                                coords_od.append((r_od * math.cos(th), r_od * math.sin(th)))
                            except Exception:
                                pass
                    if sample_id and (not id_single_enable):
                        id_v = p.get("id_mm", None)
                        if id_v is not None:
                            try:
                                id_f = float(id_v)
                                r_id = 0.5 * id_f
                                coords_id.append((r_id * math.cos(th), r_id * math.sin(th)))
                            except Exception:
                                pass

                if mode == "b":
                    try:
                        if sample_od:
                            w_od = []
                            for p in raw_points:
                                if p.get("od_mm", None) is None:
                                    continue
                                bidx = int(p.get("bin"))
                                c = int(cnt[bidx]) if 0 <= bidx < n else 0
                                w_od.append(1.0 / float(c) if c > 0 else 0.0)
                            w_od = np.asarray(w_od, dtype=float)
                            if w_od.size == len(coords_od) and w_od.size > 0:
                                self._last_fit_weights_od = w_od
                        if sample_id and (not id_single_enable):
                            w_id = []
                            for p in raw_points:
                                if p.get("id_mm", None) is None:
                                    continue
                                bidx = int(p.get("bin"))
                                c = int(cnt[bidx]) if 0 <= bidx < n else 0
                                w_id.append(1.0 / float(c) if c > 0 else 0.0)
                            w_id = np.asarray(w_id, dtype=float)
                            if w_id.size == len(coords_id) and w_id.size > 0:
                                self._last_fit_weights_id = w_id
                    except Exception:
                        self._last_fit_weights_od = None
                        self._last_fit_weights_id = None

                miss = int(n - sum(1 for i in range(n) if cnt[i] > 0))

            # store last coverage/reason for UI/debug
            elapsed = float(time.time() - t_start)
            self._last_sample_cov = (n, n - miss, miss)
            self._last_sample_reason = (reason, revs, elapsed)

            # debug summary (radius-based, per-bin averages)
            try:
                rbin_od = [sum_r_od[i] / cnt[i] for i in range(n) if cnt[i] > 0] if sample_od else []
                rbin_id = [sum_r_id[i] / cnt[i] for i in range(n) if cnt[i] > 0] if sample_id else []

                def _pp_trim_list(lst, trim_ratio: float = 0.01) -> float:
                    if not lst or len(lst) < 2:
                        return 0.0
                    b0 = sorted([float(x) for x in lst])
                    m0 = len(b0)
                    k0 = int(max(0, math.floor(float(trim_ratio) * m0)))
                    if (2 * k0) >= (m0 - 1):
                        k0 = 0
                    return float(b0[m0 - 1 - k0] - b0[k0])

                od_r_pp = float(max(rbin_od) - min(rbin_od)) if len(rbin_od) >= 2 else 0.0
                id_r_pp = float(max(rbin_id) - min(rbin_id)) if len(rbin_id) >= 2 else 0.0
                od_r_pp_t = _pp_trim_list(rbin_od, 0.01)
                id_r_pp_t = _pp_trim_list(rbin_id, 0.01)

                self._last_sample_debug = {
                    "phase": str(phase),
                    "iters": int(iters),
                    "skip_no_new_od": int(skip_no_new_od),
                    "skip_od_outlier": int(skip_od_outlier),
                    "skip_id_none": int(skip_id_none),
                    "skip_id_outlier": int(skip_id_outlier),
                    "od_min": od_min, "od_max": od_max,
                    "id_min": id_min, "id_max": id_max,
                    "od_r_pp": od_r_pp, "id_r_pp": id_r_pp,
                    "od_r_pp_trim": od_r_pp_t, "id_r_pp_trim": id_r_pp_t,
                    "filled": int(n - miss), "miss": int(miss),
                    "need": int(need), "n": int(n),
                    "reason": str(reason), "revs": float(revs), "elapsed": float(elapsed),
                }

                log(
                    "SAMPLE_DONE",
                    section=section_idx + 1,
                    phase=str(phase),
                    n=n,
                    need=need,
                    filled=(n - miss),
                    miss=miss,
                    reason=reason,
                    revs=revs,
                    elapsed=elapsed,
                    iters=iters,
                    skip_no_new_od=skip_no_new_od,
                    skip_od_outlier=skip_od_outlier,
                    skip_id_none=skip_id_none,
                    skip_id_outlier=skip_id_outlier,
                    od_min=od_min,
                    od_max=od_max,
                    id_min=id_min,
                    id_max=id_max,
                    od_r_pp=od_r_pp,
                    od_r_pp_trim=od_r_pp_t,
                    id_r_pp=id_r_pp,
                    id_r_pp_trim=id_r_pp_t,
                )
            except Exception as e:
                try:
                    log("SAMPLE_DONE_ERR", section=section_idx + 1, err=str(e))
                except Exception:
                    pass

            if sample_od and len(coords_od) < 3:
                raise RuntimeError("等角采样覆盖不足：外径有效点数 < 3，无法拟合圆。")
            if sample_id and (not id_single_enable) and len(coords_id) < 3:
                raise RuntimeError("等角采样覆盖不足：内径有效点数 < 3，无法拟合圆。")

            # expose per-channel sample counts for UI diagnostics

            try:

                self._last_sample_n_od = int(sum(1 for _p in (raw_points or []) if isinstance(_p, dict) and _p.get("od_mm", None) is not None))

                if id_single_enable:
                    self._last_sample_n_id = int(
                        sum(1 for _p in (raw_points or []) if isinstance(_p, dict) and _p.get("id_out2_mm", None) is not None)
                    )
                else:
                    self._last_sample_n_id = int(
                        sum(1 for _p in (raw_points or []) if isinstance(_p, dict) and _p.get("id_mm", None) is not None)
                    )

            except Exception:

                self._last_sample_n_od = None

                self._last_sample_n_id = None


            return (
                np.asarray(coords_od, dtype=float),
                np.asarray(coords_id, dtype=float),
                raw_last_od,
                raw_last_id,
                raw_points,
            )

        finally:
            self.app.set_plc_poll_profile("normal")


    def _sample_circle_points(self, recipe: Recipe) -> Tuple[np.ndarray, str]:
        """Backward-compatible OD-only sampling wrapper."""
        coords_od, _coords_id, raw_od, _raw_id, _raw_pts = self._sample_circle_points_dual(
            recipe,
            section_idx=0,
            sample_od=True,
            sample_id=False,
            phase="OD",
        )
        return coords_od, raw_od

    
    def _fit_circle(self, coords: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[float, float, float, float]:
        """圆拟合：优先使用 circle-fit；不可用则用最小二乘兜底。

        Returns: (xc, yc, r, sigma)
        """
        if coords is None or len(coords) < 3:
            raise ValueError("圆拟合需要至少3个点")

        pts = np.asarray(coords, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
            raise ValueError("圆拟合输入坐标形状错误")

        # If weights are provided, use weighted algebraic least squares (Kåsa).
        if weights is not None:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.size != pts.shape[0]:
                raise ValueError("weights length mismatch")
            w = np.clip(w, 1e-12, float("inf"))
            x = pts[:, 0]
            y = pts[:, 1]
            A = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
            b = x * x + y * y
            sw = np.sqrt(w)
            Aw = A * sw[:, None]
            bw = b * sw
            sol, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
            xc, yc, c = sol
            r = math.sqrt(max(0.0, float(c) + float(xc) * float(xc) + float(yc) * float(yc)))
            rr = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
            sigma = float(np.sqrt(np.average((rr - r) ** 2, weights=w))) if rr.size else 0.0
            return float(xc), float(yc), float(r), float(sigma)

        # circle-fit library (AlliedToasters/circle-fit)
        if cf is not None:
            try:
                xc, yc, r, sigma = cf.hyper_fit(pts)
                return float(xc), float(yc), float(r), float(sigma)
            except Exception:
                pass

        # fallback: algebraic least squares (Kåsa)
        x = pts[:, 0]
        y = pts[:, 1]
        A = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
        b = x * x + y * y
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        xc, yc, c = sol
        r = math.sqrt(max(0.0, float(c) + float(xc) * float(xc) + float(yc) * float(yc)))
        rr = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
        sigma = float(np.sqrt(np.mean((rr - r) ** 2))) if rr.size else 0.0
        return float(xc), float(yc), float(r), float(sigma)
    def _od_round_fit_from_raw_points(
        self,
        raw_points: List[dict],
        *,
        calc_input_mode: str = "bin",
        bin_count: int = 90,
        bin_method: str = "median",
        pp_mode: str = "p99_p1",
        theta_delay_s: float = 0.0,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute OD roundness by circle-fit residual (diameter mm).
    
        This uses the synthesized OD boundary points based on (od_mm, od_delta, theta_deg).
    
        calc_input_mode:
          - "raw": every raw sample contributes 2 boundary points (right/left edge)
          - "bin": bin by angle then reduce per-bin (median/mean) to 2 points/bin
    
        Returns: (od_round_fit_mm, od_round_fit_rob_mm) in diameter mm.
        """
        try:
            mode = (calc_input_mode or "bin").strip().lower()
            req_n = max(3, int(bin_count))
            n = req_n
            # Extract valid samples first
            th_list: list[float] = []
            ts_list: list[float] = []
            r_list: list[float] = []
            dlt_raw_list: list[float] = []
            for pnt in raw_points or []:
                if not isinstance(pnt, dict):
                    continue
                th = pnt.get("theta_deg", None)
                ts = pnt.get("ts", None)
                od_mm = pnt.get("od_mm", None)
                od_delta = pnt.get("od_delta", 0.0)
                if th is None or ts is None or od_mm is None:
                    continue
                try:
                    thf = float(th)
                    tsf = float(ts)
                    df = float(od_mm)
                    dlt = float(od_delta or 0.0)
                except Exception:
                    continue
                if (not math.isfinite(thf)) or (not math.isfinite(tsf)) or (not math.isfinite(df)) or df <= 0.0:
                    continue
                if not math.isfinite(dlt):
                    dlt = 0.0
                r = 0.5 * df
                th_list.append(thf % 360.0)
                ts_list.append(tsf)
                r_list.append(float(r))
                dlt_raw_list.append(float(dlt))

            if len(th_list) < 3:
                return None, None

            # Remove DC bias from od_delta before synthesizing OD boundary points.
            # If od_delta carries a large constant offset (e.g. sensor installation bias),
            # it will distort the synthesized point cloud and explode fit residuals.
            try:
                dlt_arr = np.asarray(dlt_raw_list, dtype=float)
                dlt_arr = dlt_arr[np.isfinite(dlt_arr)]
                dlt_bias = float(np.median(dlt_arr)) if dlt_arr.size else 0.0
            except Exception:
                dlt_bias = 0.0

            rr_list: list[float] = []
            rl_list: list[float] = []
            for r, dlt_raw in zip(r_list, dlt_raw_list):
                d = float(dlt_raw) - float(dlt_bias)
                rr_list.append(float(r) + d)
                rl_list.append(float(r) - d)
            # adapt bin_count per available samples
            if not mode.startswith("raw"):
                n = _adaptive_bin_count(req_n, len(th_list))

            try:
                delay_s = float(theta_delay_s or 0.0)
            except Exception:
                delay_s = 0.0
            omega = _estimate_omega_deg_s(th_list, ts_list) if abs(delay_s) > 1e-9 else 0.0
    
            pts: list[tuple[float, float]] = []
    
            if mode.startswith("raw"):
                for th_deg, rr, rl in zip(th_list, rr_list, rl_list):
                    th_corr = _theta_apply_delay(th_deg, omega, delay_s)
                    th = math.radians(float(th_corr))
                    c = math.cos(th)
                    s = math.sin(th)
                    pts.append((float(rr) * c, float(rr) * s))
                    pts.append((-float(rl) * c, -float(rl) * s))
            else:
                rr_bins: list[list[float]] = [[] for _ in range(n)]
                rl_bins: list[list[float]] = [[] for _ in range(n)]
                for th_deg, rr, rl in zip(th_list, rr_list, rl_list):
                    th_corr = _theta_apply_delay(th_deg, omega, delay_s)
                    b = int((float(th_corr) / 360.0) * n)
                    if b >= n:
                        b = 0
                    rr_bins[b].append(float(rr))
                    rl_bins[b].append(float(rl))
    
                used = 0
                for i in range(n):
                    if (not rr_bins[i]) or (not rl_bins[i]):
                        continue
                    used += 1
                    th = math.radians((float(i) + 0.5) * (360.0 / float(n)))
                    c = math.cos(th)
                    s = math.sin(th)
                    rr = _reduce_bin(rr_bins[i], bin_method)
                    rl = _reduce_bin(rl_bins[i], bin_method)
                    if (not math.isfinite(rr)) or (not math.isfinite(rl)):
                        continue
                    pts.append((rr * c, rr * s))
                    pts.append((-rl * c, -rl * s))
    
                if used < 3:
                    return None, None
    
            if len(pts) < 6:
                return None, None
    
            coords = np.asarray(pts, dtype=float)
            xc, yc, r_fit, _sigma = self._fit_circle(coords)
            dx = coords[:, 0] - float(xc)
            dy = coords[:, 1] - float(yc)
            rr = np.sqrt(dx * dx + dy * dy)
            e = rr - float(r_fit)
            if e.size < 2:
                return None, None
    
            pp = 2.0 * float(np.max(e) - np.min(e))
            rob = 2.0 * float(_robust_span(e, pp_mode))
    
            if not math.isfinite(pp):
                pp = None
            if not math.isfinite(rob):
                rob = None
            return (None if pp is None else float(pp), None if rob is None else float(rob))
        except Exception:
            return None, None
    def _id_round_fit_from_raw_points(
        self,
        raw_points: List[dict],
        use_fit: bool = False,
        delta_c: float = 0.0,
        *,
        calc_input_mode: str = "bin",
        bin_count: int = 90,
        bin_method: str = "median",
        pp_mode: str = "p99_p1",
        theta_delay_s: float = 0.0,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute ID roundness by circle-fit residual (diameter mm).
    
        - If use_fit=True: reconstruct per-sample diameter from OUT4 chord + OUT5 m series.
        - Else: use id_mm series directly.
    
        calc_input_mode:
          - "raw": every raw sample contributes 2 boundary points
          - "bin": bin by angle then reduce per-bin (median/mean)
    
        Returns: (id_round_fit_mm, id_round_fit_rob_mm) in diameter mm.
        """
        try:
            mode = (calc_input_mode or "bin").strip().lower()
            req_n = max(3, int(bin_count))
            n = req_n
    
            try:
                delay_s = float(theta_delay_s or 0.0)
            except Exception:
                delay_s = 0.0
    
            pts: list[tuple[float, float]] = []
    
            if bool(use_fit):
                th_list: list[float] = []
                ts_list: list[float] = []
                c_list: list[float] = []
                m_list: list[float] = []
    
                for pnt in raw_points or []:
                    if not isinstance(pnt, dict):
                        continue
                    th = pnt.get("theta_deg", None)
                    ts = pnt.get("ts", None)
                    c = pnt.get("id_c_mm", None)
                    mm = pnt.get("id_m_mm", None)
                    if th is None or ts is None or c is None or mm is None:
                        continue
                    try:
                        thf = float(th) % 360.0
                        tsf = float(ts)
                        cf = float(c)
                        mf = float(mm)
                    except Exception:
                        continue
                    if (not math.isfinite(thf)) or (not math.isfinite(tsf)) or (not math.isfinite(cf)) or (not math.isfinite(mf)):
                        continue
                    th_list.append(thf)
                    ts_list.append(tsf)
                    c_list.append(cf)
                    m_list.append(mf)
    
                if len(th_list) < 8:
                    return None, None
    
                omega = _estimate_omega_deg_s(th_list, ts_list) if abs(delay_s) > 1e-9 else 0.0
                th_arr = np.asarray([_theta_apply_delay(th, omega, delay_s) for th in th_list], dtype=float)
                c_arr = np.asarray(c_list, dtype=float)
                m_arr = np.asarray(m_list, dtype=float)
    
                fit = None
                if hasattr(self.app, "_idcal_fit_diameter"):
                    try:
                        fit = self.app._idcal_fit_diameter(th_arr, c_arr, m_arr, float(delta_c))  # type: ignore[attr-defined]
                    except Exception:
                        fit = None
                if fit is None:
                    fit = self._idcal_fit_diameter_local(th_arr, c_arr, m_arr, float(delta_c))
    
                diam = float((fit.get("diam", 0.0) if isinstance(fit, dict) else 0.0) or 0.0)
                if (not math.isfinite(diam)) or diam <= 0.0:
                    return None, None
    
                th_rad = np.deg2rad(th_arr.astype(float))
                phi = float((fit.get("phi_rad", 0.0) if isinstance(fit, dict) else 0.0) or 0.0)
                e = float((fit.get("e", 0.0) if isinstance(fit, dict) else 0.0) or 0.0)
                y0 = float((fit.get("y0", 0.0) if isinstance(fit, dict) else 0.0) or 0.0)
                s = np.sin(th_rad + phi)
                c_corr = np.clip(c_arr + float(delta_c), 0.001, None)
                pred_R2 = (0.5 * c_corr) ** 2 + (y0 + e * s) ** 2
                Di = 2.0 * np.sqrt(np.clip(pred_R2, 0.0, None))
    
                # adapt bin_count per available samples
    
                if not mode.startswith("raw"):
    
                    n = _adaptive_bin_count(req_n, len(th_list) if "th_list" in locals() else 0)

    
                if mode.startswith("raw"):
                    for th_deg, d in zip(th_arr.tolist(), Di.tolist()):
                        df = float(d)
                        if (not math.isfinite(df)) or df <= 0.0:
                            continue
                        th = math.radians(float(th_deg) % 360.0)
                        c = math.cos(th)
                        s = math.sin(th)
                        r = 0.5 * df
                        pts.append((r * c, r * s))
                        pts.append((-r * c, -r * s))
                else:
                    r_bins: list[list[float]] = [[] for _ in range(n)]
                    for th_deg, d in zip(th_arr.tolist(), Di.tolist()):
                        df = float(d)
                        if (not math.isfinite(df)) or df <= 0.0:
                            continue
                        thf = float(th_deg) % 360.0
                        b = int((thf / 360.0) * n)
                        if b >= n:
                            b = 0
                        r_bins[b].append(0.5 * df)
    
                    used = 0
                    for i in range(n):
                        if not r_bins[i]:
                            continue
                        used += 1
                        th = math.radians((float(i) + 0.5) * (360.0 / float(n)))
                        c = math.cos(th)
                        s = math.sin(th)
                        r = _reduce_bin(r_bins[i], bin_method)
                        if not math.isfinite(r):
                            continue
                        pts.append((r * c, r * s))
                        pts.append((-r * c, -r * s))
    
                    if used < 3:
                        return None, None
    
            else:
                th_list: list[float] = []
                ts_list: list[float] = []
                d_list: list[float] = []
                for pnt in raw_points or []:
                    if not isinstance(pnt, dict):
                        continue
                    th_deg = pnt.get("theta_deg", None)
                    ts = pnt.get("ts", None)
                    id_mm = pnt.get("id_mm", None)
                    if th_deg is None or ts is None or id_mm is None:
                        continue
                    try:
                        thf = float(th_deg) % 360.0
                        tsf = float(ts)
                        df = float(id_mm)
                    except Exception:
                        continue
                    if (not math.isfinite(thf)) or (not math.isfinite(tsf)) or (not math.isfinite(df)) or df <= 0.0:
                        continue
                    th_list.append(thf)
                    ts_list.append(tsf)
                    d_list.append(df)
    
                if len(th_list) < 3:
                    return None, None
    
                omega = _estimate_omega_deg_s(th_list, ts_list) if abs(delay_s) > 1e-9 else 0.0
    
                if mode.startswith("raw"):
                    for th_deg, d in zip(th_list, d_list):
                        th_corr = _theta_apply_delay(th_deg, omega, delay_s)
                        th = math.radians(float(th_corr))
                        c = math.cos(th)
                        s = math.sin(th)
                        r = 0.5 * float(d)
                        pts.append((r * c, r * s))
                        pts.append((-r * c, -r * s))
                else:
                    r_bins: list[list[float]] = [[] for _ in range(n)]
                    for th_deg, d in zip(th_list, d_list):
                        th_corr = _theta_apply_delay(th_deg, omega, delay_s)
                        b = int((float(th_corr) / 360.0) * n)
                        if b >= n:
                            b = 0
                        r_bins[b].append(0.5 * float(d))
    
                    used = 0
                    for i in range(n):
                        if not r_bins[i]:
                            continue
                        used += 1
                        th = math.radians((float(i) + 0.5) * (360.0 / float(n)))
                        c = math.cos(th)
                        s = math.sin(th)
                        r = _reduce_bin(r_bins[i], bin_method)
                        if not math.isfinite(r):
                            continue
                        pts.append((r * c, r * s))
                        pts.append((-r * c, -r * s))
    
                    if used < 3:
                        return None, None
    
            if len(pts) < 6:
                return None, None
    
            coords = np.asarray(pts, dtype=float)
            xc, yc, r_fit, _sigma = self._fit_circle(coords)
            dx = coords[:, 0] - float(xc)
            dy = coords[:, 1] - float(yc)
            rr = np.sqrt(dx * dx + dy * dy)
            e = rr - float(r_fit)
            if e.size < 2:
                return None, None
    
            pp = 2.0 * float(np.max(e) - np.min(e))
            rob = 2.0 * float(_robust_span(e, pp_mode))
    
            if not math.isfinite(pp):
                pp = None
            if not math.isfinite(rob):
                rob = None
            return (None if pp is None else float(pp), None if rob is None else float(rob))
        except Exception:
            return None, None
