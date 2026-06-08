from __future__ import annotations

"""Pure sampling and circle-fitting algorithms for the measurement flow.

This module contains zero side-effect functions — pure math/numpy computation
with no PLC, queue, threading, or UI dependencies.  Extracted from
``services/autoflow_service.py`` to break the ``services/ → application/``
dependency cycle.
"""

import math
import re
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Angular utilities
# ---------------------------------------------------------------------------

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


def _wrap_deg_180(d: float) -> float:
    try:
        x = float(d)
        if not math.isfinite(x):
            return float('nan')
        return ((x + 180.0) % 360.0) - 180.0
    except Exception:
        return float('nan')


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


# ---------------------------------------------------------------------------
# Statistics / reduction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Angular speed estimation
# ---------------------------------------------------------------------------

def _estimate_omega_deg_s(theta_deg: list[float], ts: list[float]) -> float:
    """Estimate average angular speed (deg/s) from (theta_deg, ts) pairs.

    theta_deg is assumed wrapped to [0,360). This function unwraps by the
    shortest jump (±180 rule).
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


# ---------------------------------------------------------------------------
# Split-scan diagnostics
# ---------------------------------------------------------------------------

def _split_slip_diag(
    raw_points_od: list[dict],
    raw_points_id: list[dict],
    slip_max_deg: float = 5.0,
    omega_cv_max: float = 0.25,
) -> tuple[float | None, bool | None]:
    """Lightweight split-scan diagnostics.

    Computes:
      - split_shift_deg: phase discontinuity between OD pass end and ID pass
        start (deg, wrapped to [-180,180)).
      - coax_unreliable: True if shift or speed stability exceeds thresholds;
        None if cannot evaluate.

    Notes:
      This does NOT prove mechanical slip; it's a sanity check to flag
      potentially unreliable coax metrics.
    """

    def _float_values(points: list[dict], key: str) -> list[float]:
        values: list[float] = []
        for point in points or []:
            if not isinstance(point, dict):
                continue
            value = point.get(key)
            if value is None:
                continue
            values.append(float(value))
        return values

    try:
        th_od = _float_values(raw_points_od, 'theta_deg')
        ts_od = _float_values(raw_points_od, 'ts')
        th_id = _float_values(raw_points_id, 'theta_deg')
        ts_id = _float_values(raw_points_id, 'ts')
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


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

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


__all__ = [
    "_adaptive_bin_count",
    "_estimate_omega_deg_s",
    "_max_gap_deg_from_bins",
    "_omega_cv_deg_s",
    "_reduce_bin",
    "_robust_span",
    "_split_slip_diag",
    "_theta_apply_delay",
    "_wrap_deg_180",
]
