from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    import circle_fit as cf  # type: ignore
except Exception:  # pragma: no cover
    cf = None  # type: ignore

from domain.sampling import (
    _adaptive_bin_count,
    _estimate_omega_deg_s,
    _reduce_bin,
    _robust_span,
    _theta_apply_delay,
)

__all__ = ["ExecutorFittingMixin"]


class ExecutorFittingMixin:

    # Typed port accessors — set by ExecutorCoreMixin.__init__, shared via MRO
    _typed_motion: Any = None  # type: ignore[assignment]
    _typed_sensors: Any = None  # type: ignore[assignment]
    _typed_operator: Any = None  # type: ignore[assignment]
    _typed_plc: Any = None  # type: ignore[assignment]
    _calibration_snapshot: Any

    # Methods called from other mixins (cooperative MRO)
    _get_calibration_snapshot: Any

    def get_active_id_delta_c(self) -> float:
        return self._idcal_get_delta_c_active()

    def _idcal_get_delta_c_active(self) -> float:
        """Get active delta_c(mm) for ID chord correction.

        Returns the current measurement-flow calibration snapshot value.
        """
        try:
            delta_c = float(self._get_calibration_snapshot().id_delta_c_mm)
            if math.isfinite(delta_c):
                return delta_c
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

            # Optional theta delay compensation (shift theta by omega*delay)
            try:
                delay_s = float(theta_delay_s or 0.0)
            except Exception:
                delay_s = 0.0
            if abs(delay_s) > 1e-9:
                omega = _estimate_omega_deg_s(th_list, ts_list)
                th_arr = np.asarray([_theta_apply_delay(float(th), float(omega), float(delay_s)) for th in th_arr], dtype=float)

            # Prefer the sensor port implementation (keeps consistency with the
            # calibration page); fall back to local fitting if unavailable.
            fit = self._typed_sensors.fit_id_diameter(th_arr, c_arr, m_arr, float(delta_c))
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

    def id_round_fit_from_raw_points(
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
        return self._id_round_fit_from_raw_points(
            raw_points,
            use_fit=use_fit,
            delta_c=delta_c,
            calc_input_mode=calc_input_mode,
            bin_count=bin_count,
            bin_method=bin_method,
            pp_mode=pp_mode,
            theta_delay_s=theta_delay_s,
        )

    def fit_id_from_raw_points(
        self, raw_points: list[dict], delta_c: float, *, theta_delay_s: float = 0.0,
    ) -> tuple[Optional[dict], Optional[np.ndarray]]:
        return self._id_fit_from_raw_points(raw_points, delta_c, theta_delay_s=theta_delay_s)

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

    def fit_circle(
        self, coords: np.ndarray, weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float, float]:
        return self._fit_circle(coords, weights=weights)

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

    def od_round_fit_from_raw_points(
        self,
        raw_points: List[dict],
        *,
        calc_input_mode: str = "bin",
        bin_count: int = 90,
        bin_method: str = "median",
        pp_mode: str = "p99_p1",
        theta_delay_s: float = 0.0,
    ) -> Tuple[Optional[float], Optional[float]]:
        return self._od_round_fit_from_raw_points(
            raw_points,
            calc_input_mode=calc_input_mode,
            bin_count=bin_count,
            bin_method=bin_method,
            pp_mode=pp_mode,
            theta_delay_s=theta_delay_s,
        )

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

                fit = self._typed_sensors.fit_id_diameter(th_arr, c_arr, m_arr, float(delta_c))
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
