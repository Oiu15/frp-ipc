from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable

import numpy as np

from core.models import Recipe


@dataclass(frozen=True, slots=True)
class OdBCandidateResult:
    ok: bool
    reason: str
    d_ref: float
    b_candidate: float | None
    mean_sum: float | None
    n_used: int


@dataclass(frozen=True, slots=True)
class IdDiameterFitResult:
    radius: float
    diam: float
    e: float
    phi_rad: float
    x0: float
    y0: float
    rmse_r2: float

    def to_legacy_dict(self) -> dict[str, float]:
        return {
            'R': float(self.radius),
            'diam': float(self.diam),
            'e': float(self.e),
            'phi_rad': float(self.phi_rad),
            'x0': float(self.x0),
            'y0': float(self.y0),
            'rmse_R2': float(self.rmse_r2),
        }


@dataclass(frozen=True, slots=True)
class IdDeltaCandidateResult:
    ok: bool
    reason: str
    delta_candidate: float | None
    cmax: float | None
    fit: IdDiameterFitResult | None
    sample_count: int
    fallback_used: bool


@dataclass(frozen=True, slots=True)
class IdVerifyResult:
    ok: bool
    diam: float
    err_mm: float
    cov_pct: float
    sample_count: int
    dtheta_max_deg: float


@dataclass(frozen=True, slots=True)
class IdSingleCalibrationResult:
    ok: bool
    reason: str | None = None
    mean_L2_decenter: float | None = None
    id_est_mm: float | None = None
    id_ecc_amp_mm: float | None = None
    id_ecc_ang_deg: float | None = None
    id_pp_mm: float | None = None
    id_pp_rob_mm: float | None = None
    cov: float | None = None
    n_used: int = 0
    n_bins: int = 0
    a: float | None = None
    b: float | None = None
    c: float | None = None

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            'ok': bool(self.ok),
            'reason': self.reason,
            'mean_L2_decenter': self.mean_L2_decenter,
            'id_est_mm': self.id_est_mm,
            'id_ecc_amp_mm': self.id_ecc_amp_mm,
            'id_ecc_ang_deg': self.id_ecc_ang_deg,
            'id_pp_mm': self.id_pp_mm,
            'id_pp_rob_mm': self.id_pp_rob_mm,
            'cov': self.cov,
            'n_used': int(self.n_used),
            'n_bins': int(self.n_bins),
            'a': self.a,
            'b': self.b,
            'c': self.c,
        }


def compute_od_b_candidate(sums: Iterable[float], d_ref: float) -> OdBCandidateResult:
    valid: list[float] = []
    for item in sums or []:
        try:
            value = float(item)
            if math.isfinite(value):
                valid.append(value)
        except Exception:
            continue
    if not valid:
        return OdBCandidateResult(
            ok=False,
            reason='empty_sums',
            d_ref=float(d_ref),
            b_candidate=None,
            mean_sum=None,
            n_used=0,
        )
    mean_sum = float(sum(valid) / len(valid))
    return OdBCandidateResult(
        ok=True,
        reason='',
        d_ref=float(d_ref),
        b_candidate=float(d_ref) + mean_sum,
        mean_sum=mean_sum,
        n_used=len(valid),
    )


def lsq_fit_cos_sin(theta_rad: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.column_stack([np.ones_like(theta_rad), np.cos(theta_rad), np.sin(theta_rad)])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(beta[0]), float(beta[1]), float(beta[2])


def robust_span(values: np.ndarray, mode: str = 'p99_p1') -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    normalized = (mode or '').strip().lower()
    if normalized in ('strict', 'maxmin', 'pp'):
        return float(np.max(arr) - np.min(arr))
    if normalized.startswith('trim'):
        ratio = 0.01
        mm = re.search(r'trim[_-]?([0-9]+(?:\.[0-9]+)?|[0-9]+p[0-9]+)', normalized)
        if mm:
            try:
                ratio = float(mm.group(1).replace('p', '.'))
            except Exception:
                ratio = 0.01
        ratio = max(0.0, min(0.49, float(ratio)))
        sorted_vals = np.sort(arr)
        n = int(sorted_vals.size)
        k = int(max(0, math.floor(ratio * n)))
        if (2 * k) >= (n - 1):
            k = 0
        return float(sorted_vals[n - 1 - k] - sorted_vals[k])
    mm = re.match(r'p(\d+(?:\.\d+)?)_p(\d+(?:\.\d+)?)$', normalized)
    if mm:
        try:
            hi = float(mm.group(1))
            lo = float(mm.group(2))
            if hi < lo:
                hi, lo = lo, hi
            hi = max(0.0, min(100.0, hi))
            lo = max(0.0, min(100.0, lo))
            return float(np.percentile(arr, hi) - np.percentile(arr, lo))
        except Exception:
            return float(np.max(arr) - np.min(arr))
    try:
        return float(np.percentile(arr, 99.0) - np.percentile(arr, 1.0))
    except Exception:
        return float(np.max(arr) - np.min(arr))


def fit_id_single_from_out2(theta_deg: Iterable[float], out2_mm: Iterable[float], recipe: Recipe) -> IdSingleCalibrationResult:
    try:
        th = np.asarray(list(theta_deg or []), dtype=float)
        y = np.asarray(list(out2_mm or []), dtype=float)
    except Exception:
        return IdSingleCalibrationResult(ok=False, reason='bad_input')
    if th.size == 0 or y.size == 0:
        return IdSingleCalibrationResult(ok=False, reason='empty')
    mask = np.isfinite(th) & np.isfinite(y)
    th = th[mask]
    y = y[mask]
    if th.size < 3:
        return IdSingleCalibrationResult(ok=False, reason='too_few')
    try:
        n_bins = max(3, int(getattr(recipe, 'bin_count', 90)))
    except Exception:
        n_bins = 90
    method = str(getattr(recipe, 'bin_method', 'median') or 'median').strip().lower()
    pp_mode = str(getattr(recipe, 'pp_mode', 'p99_p1') or 'p99_p1').strip().lower()
    bins: list[list[float]] = [[] for _ in range(int(n_bins))]
    for t, v in zip(th.tolist(), y.tolist()):
        b = int((float(t) % 360.0) / 360.0 * float(n_bins))
        if b >= int(n_bins):
            b = 0
        bins[b].append(float(v))

    def _reduce(vals: list[float]) -> float:
        if not vals:
            return float('nan')
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float('nan')
        if method in ('mean', 'avg', 'average'):
            return float(np.mean(arr))
        return float(np.median(arr))

    th_bin: list[float] = []
    y_bin: list[float] = []
    for idx, vals in enumerate(bins):
        if not vals:
            continue
        reduced = _reduce(vals)
        if not math.isfinite(reduced):
            continue
        th_bin.append((float(idx) + 0.5) * (360.0 / float(n_bins)))
        y_bin.append(float(reduced))
    used = len(y_bin)
    cov = float(used) / float(n_bins) if n_bins else 0.0
    if used < 3:
        return IdSingleCalibrationResult(ok=False, reason='too_few_bins', cov=cov, n_used=used, n_bins=int(n_bins))

    th_rad = np.deg2rad(np.asarray(th_bin, dtype=float))
    yb = np.asarray(y_bin, dtype=float)
    x0, a_coef, b_coef = lsq_fit_cos_sin(th_rad, yb)
    dec = yb - (float(a_coef) * np.cos(th_rad) + float(b_coef) * np.sin(th_rad))
    mean_dec = float(np.mean(dec)) if dec.size else float('nan')
    pp_strict = robust_span(dec, 'strict')
    pp_rob = robust_span(dec, pp_mode)
    try:
        k_val = float(getattr(recipe, 'id_single_k', 1.0) or 1.0)
    except Exception:
        k_val = 1.0
    try:
        b0 = float(getattr(recipe, 'id_single_b', 0.0) or 0.0)
    except Exception:
        b0 = 0.0
    id_est = (float(k_val) * float(mean_dec) + float(b0)) if math.isfinite(mean_dec) else float('nan')
    ecc_amp = float(k_val) * float(math.hypot(float(a_coef), float(b_coef)))
    ecc_ang = float(math.degrees(math.atan2(float(b_coef), float(a_coef)))) % 360.0
    id_pp_mm = float(k_val) * float(pp_strict)
    id_pp_rob = float(k_val) * float(pp_rob)
    return IdSingleCalibrationResult(
        ok=True,
        reason=None,
        mean_L2_decenter=_finite_or_none(mean_dec),
        id_est_mm=_finite_or_none(id_est),
        id_ecc_amp_mm=_finite_or_none(ecc_amp),
        id_ecc_ang_deg=_finite_or_none(ecc_ang),
        id_pp_mm=_finite_or_none(id_pp_mm),
        id_pp_rob_mm=_finite_or_none(id_pp_rob),
        cov=float(cov),
        n_used=int(used),
        n_bins=int(n_bins),
        a=_finite_or_none(a_coef),
        b=_finite_or_none(b_coef),
        c=_finite_or_none(x0),
    )


def fit_id_diameter(theta_deg: np.ndarray, c_mm: np.ndarray, m_mm: np.ndarray, delta_c: float) -> IdDiameterFitResult:
    th = np.deg2rad(theta_deg.astype(float))
    m = m_mm.astype(float)
    x0, a_coef, b_coef = lsq_fit_cos_sin(th, m)
    e = float(math.hypot(a_coef, b_coef))
    phi = float(math.atan2(-b_coef, a_coef))
    s = np.sin(th + phi)
    c_corr = np.clip(c_mm.astype(float) + float(delta_c), 0.001, None)
    z = (0.5 * c_corr) ** 2 + (e * s) ** 2
    x2 = np.column_stack([np.ones_like(s), (-2.0 * e * s)])
    beta2, *_ = np.linalg.lstsq(x2, z, rcond=None)
    r2p = float(beta2[0])
    y0 = float(beta2[1])
    r2 = float(r2p + y0 * y0)
    radius = float(math.sqrt(max(r2, 0.0)))
    pred_r2 = (0.5 * c_corr) ** 2 + (y0 + e * s) ** 2
    rmse_r2 = float(math.sqrt(max(0.0, float(np.mean((pred_r2 - r2) ** 2)))))
    return IdDiameterFitResult(
        radius=radius,
        diam=2.0 * radius,
        e=e,
        phi_rad=phi,
        x0=float(x0),
        y0=y0,
        rmse_r2=rmse_r2,
    )


def solve_id_delta_candidate(theta_deg: np.ndarray, c_mm: np.ndarray, m_mm: np.ndarray, d_ref: float) -> IdDeltaCandidateResult:
    c = np.asarray(c_mm, dtype=float)
    theta = np.asarray(theta_deg, dtype=float)
    m = np.asarray(m_mm, dtype=float)
    if theta.size < 20 or c.size < 20 or m.size < 20:
        if c.size == 0:
            return IdDeltaCandidateResult(False, 'no_valid_out4', None, None, None, int(theta.size), False)
        cmax = float(np.max(c))
        return IdDeltaCandidateResult(
            ok=True,
            reason='fallback_cmax',
            delta_candidate=float(d_ref - cmax),
            cmax=cmax,
            fit=None,
            sample_count=int(theta.size),
            fallback_used=True,
        )

    cmax = float(np.max(c))
    delta0 = float(d_ref - cmax)

    def f(delta: float) -> tuple[float, IdDiameterFitResult | None]:
        try:
            result = fit_id_diameter(theta, c, m, delta)
            return float(result.diam - d_ref), result
        except Exception:
            return float('nan'), None

    lo, hi = delta0 - 5.0, delta0 + 5.0
    flo, _ = f(lo)
    fhi, _ = f(hi)
    expand = 0
    while (not math.isfinite(flo) or not math.isfinite(fhi) or (flo * fhi > 0.0)) and expand < 6:
        lo -= 5.0
        hi += 5.0
        flo, _ = f(lo)
        fhi, _ = f(hi)
        expand += 1

    best_delta = delta0
    best: IdDiameterFitResult | None = None
    fallback_used = False
    reason = ''
    if math.isfinite(flo) and math.isfinite(fhi) and (flo * fhi <= 0.0):
        a, b = lo, hi
        fa, fb = flo, fhi
        for _ in range(28):
            mid = 0.5 * (a + b)
            fm, rm = f(mid)
            if (not math.isfinite(fm)) or (rm is None):
                break
            if fa * fm <= 0.0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
        best_delta = 0.5 * (a + b)
        _, best = f(best_delta)
    else:
        _, best = f(best_delta)
        fallback_used = True
        reason = 'fit_fallback_cmax'

    if best is None:
        return IdDeltaCandidateResult(False, 'fit_failed', None, cmax, None, int(theta.size), fallback_used)
    return IdDeltaCandidateResult(
        ok=True,
        reason=reason,
        delta_candidate=float(best_delta),
        cmax=cmax,
        fit=best,
        sample_count=int(theta.size),
        fallback_used=fallback_used,
    )


def verify_id_calibration(
    theta_deg: np.ndarray,
    c_mm: np.ndarray,
    m_mm: np.ndarray,
    *,
    delta_c: float,
    d_ref: float,
    min_cov_pct: float = 95.0,
    max_abs_err_mm: float = 0.020,
) -> IdVerifyResult:
    theta = np.asarray(theta_deg, dtype=float)
    c = np.asarray(c_mm, dtype=float)
    m = np.asarray(m_mm, dtype=float)
    if theta.size < 30 or c.size < 30 or m.size < 30:
        raise ValueError(f'verify sample too small: N={theta.size}')
    th_rad = np.deg2rad(theta)
    th_unw = np.unwrap(th_rad)
    th_deg_unw = np.rad2deg(th_unw)
    span = float(abs(th_deg_unw[-1] - th_deg_unw[0]))
    dth = np.abs(np.diff(th_deg_unw))
    dth_max = float(np.max(dth)) if len(dth) else 0.0
    cov_pct = 100.0 * min(1.0, span / 360.0)
    result = fit_id_diameter(theta, c, m, float(delta_c))
    err = float(result.diam - float(d_ref))
    ok = (cov_pct >= float(min_cov_pct)) and (abs(err) <= float(max_abs_err_mm))
    return IdVerifyResult(
        ok=bool(ok),
        diam=float(result.diam),
        err_mm=err,
        cov_pct=float(cov_pct),
        sample_count=int(theta.size),
        dtheta_max_deg=float(dth_max),
    )


def _finite_or_none(x: Any) -> float | None:
    try:
        return None if x is None or not math.isfinite(float(x)) else float(x)
    except Exception:
        return None


__all__ = [
    'IdDeltaCandidateResult',
    'IdDiameterFitResult',
    'IdSingleCalibrationResult',
    'IdVerifyResult',
    'OdBCandidateResult',
    'compute_od_b_candidate',
    'fit_id_diameter',
    'fit_id_single_from_out2',
    'lsq_fit_cos_sin',
    'robust_span',
    'solve_id_delta_candidate',
    'verify_id_calibration',
]
