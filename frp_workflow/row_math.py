from __future__ import annotations

"""Row-level measurement calculations for AutoFlow."""

import math
from typing import Any, Protocol, cast

import numpy as np

from domain.sampling import _robust_span
from frp_workflow.autoflow_executor import perf_logger
from frp_workflow.steps.measure_row_build_inputs import MeasureRowBuildInputs
from frp_workflow.steps.measure_row_computation_result import MeasureRowComputationResult


class LegacyFitPort(Protocol):
    def fit_circle(self, coords: Any, *, weights: Any | None = None) -> Any:
        ...

    def get_active_id_delta_c(self) -> float:
        ...

    def fit_id_from_raw_points(
        self,
        raw_points: Any,
        delta_c: float,
        *,
        theta_delay_s: float = 0.0,
    ) -> Any:
        ...

    def od_round_fit_from_raw_points(self, raw_points: Any, **kwargs: Any) -> Any:
        ...

    def id_round_fit_from_raw_points(self, raw_points: Any, **kwargs: Any) -> Any:
        ...


def _optional_finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return float(numeric)


def _point_float_values(raw_points: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for point in raw_points:
        raw_value = point.get(key)
        if raw_value is not None:
            values.append(float(cast(Any, raw_value)))
    return values


def _compute_section_v2(raw_points: Any, tooling: Any) -> dict[str, float]:
    """geometry_v2 单截面 ID 重建(去旋转→拟合圆),返回 v2 指标 dict。

    纯加性、best-effort:任何缺数据/异常都返回 {}(主链回退 legacy)。
    OD v2 暂留空(待 batch 5 标定 OD 工装后接入)。
    """
    out: dict[str, float] = {}
    try:
        if tooling is None or not tooling.id_calibrated():
            return out
        from domain.geometry_calibration import id_points_from_readings
        from domain.geometry_fit import fit_circle_geometric, roundness_from_points

        th: list[float] = []
        l1: list[float] = []
        l2: list[float] = []
        for p in raw_points or []:
            if not isinstance(p, dict):
                continue
            t = p.get("theta_deg")
            a = p.get("id_x1_mm")
            b = p.get("id_x2_mm")
            if t is None or a is None or b is None:
                continue
            th.append(float(t))
            l1.append(float(a))
            l2.append(float(b))
        if len(th) < 8:
            return out
        theta = np.deg2rad(np.asarray(th, dtype=float))
        pts = id_points_from_readings(
            tooling.id_tooling(), theta, np.asarray(l1, dtype=float), np.asarray(l2, dtype=float)
        )
        cf = fit_circle_geometric(pts[:, 0], pts[:, 1])
        rnd = roundness_from_points(pts[:, 0], pts[:, 1], cf.cx, cf.cy)
        out["id_diam_v2"] = float(2.0 * cf.r)
        out["id_round_v2"] = float(rnd.roundness_lsc)
        out["id_cx_v2"] = float(cf.cx)
        out["id_cy_v2"] = float(cf.cy)
    except Exception:
        return {}
    return out


def _compute_measure_row_result(inputs: MeasureRowBuildInputs) -> MeasureRowComputationResult:
    legacy = cast(LegacyFitPort, inputs.legacy)
    recipe = inputs.recipe
    sensors = inputs.sensors
    section_index = inputs.section_index
    z_pos_mm = inputs.z_pos_mm
    coords_od = inputs.coords_od
    coords_id = inputs.coords_id
    raw_points = inputs.raw_points
    fit_weights_od = inputs.fit_weights_od
    fit_weights_id = inputs.fit_weights_id
    scan_mode = inputs.scan_mode

    try:
        id_single_enable = bool(getattr(recipe, "id_single_enable", False))
    except Exception:
        id_single_enable = False

    try:
        raw_total = int(len(raw_points or []))
        od_raw_in = int(
            sum(1 for p in (raw_points or []) if isinstance(p, dict) and p.get("od_mm", None) is not None)
        )
        if id_single_enable:
            id_raw_in = int(
                sum(1 for p in (raw_points or []) if isinstance(p, dict) and p.get("id_out2_mm", None) is not None)
            )
        else:
            id_raw_in = int(
                sum(1 for p in (raw_points or []) if isinstance(p, dict) and p.get("id_mm", None) is not None)
            )
        od_fit_in = int(len(coords_od))
        id_fit_in = 0 if id_single_enable else int(len(coords_id))
        perf_logger.info(
            "[FIT_INPUT] section=%d scan_mode=%s raw_total=%d od_raw_in=%d id_raw_in=%d od_fit_in=%d id_fit_in=%d calc_input_mode=%s fit_strategy=%s",
            int(section_index),
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

    xc, yc, _r_fit, _sigma = legacy.fit_circle(coords_od, weights=fit_weights_od)
    xci = yci = _r_fit_i = _sigma_i = 0.0
    if not id_single_enable:
        xci, yci, _r_fit_i, _sigma_i = legacy.fit_circle(coords_id, weights=fit_weights_id)

    center_od_x = float(xc)
    center_od_y = float(yc)
    od_radius_fit_mm = _optional_finite_float(_r_fit)
    od_diameter_fit_mm = (
        None if od_radius_fit_mm is None else float(2.0 * od_radius_fit_mm)
    )
    od_ex = None
    od_ey = None
    center_id_x: float | None = None
    center_id_y: float | None = None
    id_radius_fit_mm: float | None = None
    id_diameter_fit_mm: float | None = None
    pp_mode = str(getattr(recipe, "pp_mode", "p99_p1") or "p99_p1")

    def _pp_strict(a: np.ndarray) -> float:
        return float(_robust_span(a, "strict"))

    def _pp_robust(a: np.ndarray, **_kw: Any) -> float:
        return float(_robust_span(a, pp_mode))

    try:
        od_vals = np.asarray(_point_float_values(raw_points, "od_mm"), dtype=float)
    except Exception:
        od_vals = np.asarray([], dtype=float)
    od_pp_mm = _pp_strict(od_vals)
    od_pp_rob_mm = _pp_robust(od_vals)
    od_runout = float(od_pp_rob_mm)

    if not id_single_enable:
        try:
            id_vals = np.asarray(_point_float_values(raw_points, "id_mm"), dtype=float)
        except Exception:
            id_vals = np.asarray([], dtype=float)
        id_pp_mm = _pp_strict(id_vals)
        id_pp_rob_mm = _pp_robust(id_vals)
    else:
        id_vals = np.asarray([], dtype=float)
        id_pp_mm = 0.0
        id_pp_rob_mm = 0.0

    id_fit = None
    id_fit_diam = None
    id_fit_vals = None
    sim_disp_enabled = bool(getattr(sensors, "sim_disp_enabled", False)) if sensors is not None else False
    if (not id_single_enable) and bool(getattr(recipe, "id_use_fit", False)) and (not sim_disp_enabled):
        delta_c = float(legacy.get_active_id_delta_c())
        id_fit, id_fit_vals = legacy.fit_id_from_raw_points(
            raw_points,
            delta_c,
            theta_delay_s=float(getattr(recipe, "theta_delay_s", 0.0) or 0.0),
        )
        if id_fit is not None:
            try:
                raw_id_fit_diam = id_fit.get("diam", None)
                id_fit_diam = None if raw_id_fit_diam is None else float(cast(Any, raw_id_fit_diam))
            except Exception:
                id_fit_diam = None

        if id_fit_vals is None:
            try:
                c_list = _point_float_values(raw_points, "id_c_mm")
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

    od_use_edges = bool(getattr(recipe, "od_use_edges", False))
    dx = coords_od[:, 0] - float(xc)
    dy = coords_od[:, 1] - float(yc)
    r_list = np.sqrt(dx * dx + dy * dy)
    od_list = 2.0 * r_list

    if od_use_edges and od_vals.size:
        od_avg = float(np.mean(od_vals))
        od_round = _pp_robust(od_vals)
        od_radius_fit_mm = _optional_finite_float(0.5 * od_avg)
        od_diameter_fit_mm = _optional_finite_float(od_avg)
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
                    if od_phi_deg <= -180.0:
                        od_phi_deg += 360.0
                    elif od_phi_deg > 180.0:
                        od_phi_deg -= 360.0
                except Exception:
                    od_phi_deg = None
        except Exception:
            od_e = 0.0
            od_phi_deg = None

        od_runout = float(2.0 * od_e)
        if (od_ex is not None) and (od_ey is not None):
            center_od_x = float(od_ex)
            center_od_y = float(od_ey)
    else:
        od_avg = float(np.mean(od_list)) if od_list.size else 0.0
        od_round = float(np.max(od_list) - np.min(od_list)) if od_list.size >= 2 else 0.0
        od_e = 0.0
        od_phi_deg = None

    od_dev = float(od_avg) - float(recipe.od_std_mm)

    od_round_fit_mm = None
    od_round_fit_rob_mm = None
    try:
        od_round_fit_mm, od_round_fit_rob_mm = legacy.od_round_fit_from_raw_points(
            raw_points,
            calc_input_mode=str(getattr(recipe, "calc_input_mode", "bin")),
            bin_count=int(getattr(recipe, "bin_count", 90)),
            bin_method=str(getattr(recipe, "bin_method", "median")),
            pp_mode=str(getattr(recipe, "pp_mode", "p99_p1")),
            theta_delay_s=float(getattr(recipe, "theta_delay_s", 0.0) or 0.0),
        )
    except Exception:
        od_round_fit_mm, od_round_fit_rob_mm = None, None

    id_round_fit_mm = None
    id_round_fit_rob_mm = None
    try:
        delta_c = float(legacy.get_active_id_delta_c())
    except Exception:
        delta_c = 0.0
    if not id_single_enable:
        try:
            id_round_fit_mm, id_round_fit_rob_mm = legacy.id_round_fit_from_raw_points(
                raw_points,
                use_fit=bool(getattr(recipe, "id_use_fit", False)),
                delta_c=float(delta_c),
                calc_input_mode=str(getattr(recipe, "calc_input_mode", "bin")),
                bin_count=int(getattr(recipe, "bin_count", 90)),
                bin_method=str(getattr(recipe, "bin_method", "median")),
                pp_mode=str(getattr(recipe, "pp_mode", "p99_p1")),
                theta_delay_s=float(getattr(recipe, "theta_delay_s", 0.0) or 0.0),
            )
        except Exception:
            id_round_fit_mm, id_round_fit_rob_mm = None, None

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

        center_id_x = float(xci)
        center_id_y = float(yci)
        id_radius_fit_mm = _optional_finite_float(_r_fit_i)
        id_diameter_fit_mm = (
            None if id_radius_fit_mm is None else float(2.0 * id_radius_fit_mm)
        )
        try:
            if bool(getattr(recipe, "id_use_fit", False)) and (id_fit is not None):
                _ex = id_fit.get("ex", None) if isinstance(id_fit, dict) else None
                _ey = id_fit.get("ey", None) if isinstance(id_fit, dict) else None
                if _ex is not None and _ey is not None and math.isfinite(float(_ex)) and math.isfinite(float(_ey)):
                    center_id_x = float(_ex)
                    center_id_y = float(_ey)
                _id_fit_radius = id_fit.get("R", None) if isinstance(id_fit, dict) else None
                if _id_fit_radius is not None:
                    id_radius_fit_mm = _optional_finite_float(_id_fit_radius)
                _id_fit_diameter = id_fit.get("diam", None) if isinstance(id_fit, dict) else None
                if _id_fit_diameter is not None:
                    id_diameter_fit_mm = _optional_finite_float(_id_fit_diameter)
        except Exception:
            pass

        concentricity = float(math.hypot(float(center_id_x) - float(center_od_x), float(center_id_y) - float(center_od_y)))

        try:
            if bool(getattr(recipe, "id_use_fit", False)) and (id_fit is not None):
                _e = id_fit.get("e", None)
                _phi = id_fit.get("phi_rad", None)
                if _e is not None and math.isfinite(float(_e)):
                    id_e = float(_e)
                if _phi is not None and math.isfinite(float(_phi)):
                    id_phi_deg = float(np.rad2deg(float(_phi)))
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
            if sensors is not None:
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
                    id_single_res = sensors.calc_id_single_from_out2(th_list, out2_list, recipe)
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

    try:
        if bool(getattr(recipe, "id_use_fit", False)) and (id_e is not None) and math.isfinite(float(id_e)):
            id_runout = float(2.0 * float(id_e))
    except Exception:
        pass

    v2: dict[str, float] = {}
    if str(getattr(recipe, "algo_version", "legacy") or "legacy") == "geometry_v2":
        tooling = getattr(inputs.calibration, "tooling", None) if inputs.calibration is not None else None
        v2 = _compute_section_v2(raw_points, tooling)

    return MeasureRowComputationResult(
        od_center=(float(center_od_x), float(center_od_y), float(z_pos_mm)),
        id_center=(
            None
            if id_single_enable
            else (float(cast(float, center_id_x)), float(cast(float, center_id_y)), float(z_pos_mm))
        ),
        center_od_x=float(center_od_x),
        center_od_y=float(center_od_y),
        center_id_x=center_id_x,
        center_id_y=center_id_y,
        od_radius_fit_mm=od_radius_fit_mm,
        od_diameter_fit_mm=od_diameter_fit_mm,
        id_radius_fit_mm=id_radius_fit_mm,
        id_diameter_fit_mm=id_diameter_fit_mm,
        od_avg=od_avg,
        od_dev=od_dev,
        od_runout=od_runout,
        od_round=od_round,
        od_round_fit_mm=od_round_fit_mm,
        od_round_fit_rob_mm=od_round_fit_rob_mm,
        od_pp_mm=od_pp_mm,
        od_pp_rob_mm=od_pp_rob_mm,
        id_avg=id_avg,
        id_dev=id_dev,
        id_runout=id_runout,
        id_round=id_round,
        id_round_fit_mm=id_round_fit_mm,
        id_round_fit_rob_mm=id_round_fit_rob_mm,
        id_pp_mm=id_pp_mm,
        id_pp_rob_mm=id_pp_rob_mm,
        od_e=od_e,
        od_phi_deg=od_phi_deg,
        id_e=id_e,
        id_phi_deg=id_phi_deg,
        id_mode=("single" if id_single_enable else "dual"),
        concentricity=concentricity,
        od_diam_v2=v2.get("od_diam_v2"),
        od_round_v2=v2.get("od_round_v2"),
        od_cx_v2=v2.get("od_cx_v2"),
        od_cy_v2=v2.get("od_cy_v2"),
        id_diam_v2=v2.get("id_diam_v2"),
        id_round_v2=v2.get("id_round_v2"),
        id_cx_v2=v2.get("id_cx_v2"),
        id_cy_v2=v2.get("id_cy_v2"),
        concentricity_v2=v2.get("concentricity_v2"),
    )


__all__ = ["LegacyFitPort", "_compute_measure_row_result"]
