from __future__ import annotations

"""Workflow-level orchestrator for the formal measurement flow.

Current scope is still intentionally staged:
- keep the constructor dependency boundary explicit
- own start/stop, outer state transitions, and section loop sequencing
- move measurement-chain logic out of App/AutoFlow incrementally
- reuse existing sampling/fit helpers instead of rewriting algorithms
"""

import math
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from application.contracts import EventSink, MachineGateway, RunRepositoryProtocol
from application.state import CalibrationSnapshot, RunSession, RuntimeState
from core.models import MeasureRow, Recipe
from domain.planning import (
    build_recipe_section_plan,
    plan_section_positions,
    require_ax2_rotate_target_abs,
    resolve_ax2_keepout_reference_abs,
    resolve_ax2_position_plan,
    resolve_standby_plan,
    resolve_start_anchor_plan,
)
from domain.summaries import compute_postcalc_result
from frp_workflow.production_workflow import ProductionWorkflow, RunResult

from services.autoflow_service import (
    AutoFlow,
    _robust_span,
    _split_slip_diag,
    log as legacy_log,
    perf_logger,
)

if TYPE_CHECKING:  # pragma: no cover
    from app import App
    from core.models import AxisCal


class _StopRequested(RuntimeError):
    """Internal sentinel used to unwind the workflow loop cleanly."""

    def __init__(self, message: str = "User stopped") -> None:
        super().__init__(message)
        self.message = message


def _unwrap_theta_span_deg(theta_values_deg: list[float]) -> float:
    if len(theta_values_deg) < 2:
        return 0.0

    prev = float(theta_values_deg[0])
    cursor = prev
    min_cursor = cursor
    max_cursor = cursor
    for raw_theta in theta_values_deg[1:]:
        theta = float(raw_theta)
        delta = theta - prev
        while delta <= -180.0:
            delta += 360.0
        while delta > 180.0:
            delta -= 360.0
        cursor += delta
        min_cursor = min(min_cursor, cursor)
        max_cursor = max(max_cursor, cursor)
        prev = theta
    return float(max_cursor - min_cursor)


def _optional_finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return float(numeric)


def _resolve_recipe_sampling_mode(recipe: Recipe) -> str:
    mode = str(
        getattr(
            recipe,
            "section_sampling_mode",
            getattr(recipe, "scan_mode", "sync"),
        )
        or "sync"
    ).strip().lower()
    if mode not in {"sync", "split"}:
        return "SYNC"
    return mode.upper()


def _annotate_validation_raw_points(
    *,
    raw_points: list[dict],
    section_index: int,
    z_pos_mm: float,
    window_index: int,
    window_role: str,
) -> list[dict]:
    annotated: list[dict] = []
    for point_index, point in enumerate(raw_points or []):
        copied = dict(point) if isinstance(point, dict) else {}
        copied["section_idx"] = int(section_index)
        copied["z_pos_mm"] = float(z_pos_mm)
        copied["window_index"] = int(window_index)
        copied["window_role"] = str(window_role)
        copied["point_index_in_window"] = int(point_index)
        annotated.append(copied)
    return annotated


def _build_validation_window_payload(
    *,
    window_index: int,
    window_role: str,
    raw_points: list[dict],
    sample_cov: tuple[int, int, int] | Any,
    sample_reason: tuple[str, float, float] | Any,
    n_od: int | None,
    n_id: int | None,
    max_gap_deg: float | None,
) -> dict[str, Any]:
    total_bins = filled_bins = miss_bins = None
    try:
        total_bins, filled_bins, miss_bins = sample_cov
    except Exception:
        pass

    reason = ""
    revs = None
    elapsed = None
    try:
        reason, revs, elapsed = sample_reason
    except Exception:
        pass

    ts_values = [
        float(point.get("ts"))
        for point in (raw_points or [])
        if isinstance(point, dict) and point.get("ts") is not None
    ]
    theta_values = [
        float(point.get("theta_deg"))
        for point in (raw_points or [])
        if isinstance(point, dict) and point.get("theta_deg") is not None
    ]

    theta_start_deg = theta_values[0] if theta_values else None
    theta_end_deg = theta_values[-1] if theta_values else None
    ts_start = min(ts_values) if ts_values else None
    ts_end = max(ts_values) if ts_values else None
    return {
        "window_index": int(window_index),
        "window_role": str(window_role),
        "point_start_index": (0 if raw_points else None),
        "point_end_index": ((len(raw_points) - 1) if raw_points else None),
        "point_count": int(len(raw_points or [])),
        "ts_start": ts_start,
        "ts_end": ts_end,
        "theta_start_deg": theta_start_deg,
        "theta_end_deg": theta_end_deg,
        "theta_span_deg": _unwrap_theta_span_deg(theta_values),
        "filled_bins": (None if filled_bins is None else int(filled_bins)),
        "total_bins": (None if total_bins is None else int(total_bins)),
        "miss_bins": (None if miss_bins is None else int(miss_bins)),
        "n_od": (None if n_od is None else int(n_od)),
        "n_id": (None if n_id is None else int(n_id)),
        "reason": str(reason or ""),
        "revs": (None if revs is None else float(revs)),
        "elapsed_s": (None if elapsed is None else float(elapsed)),
        "max_gap_deg": (None if max_gap_deg is None else float(max_gap_deg)),
    }


def _build_validation_coverage_payload(
    *,
    legacy: AutoFlow,
    section_index: int,
    scan_mode: str,
    split_shift_deg: float | None,
    coax_unreliable: bool | None,
    keep_spinning: bool,
    n_od_pass: int | None,
    n_id_pass: int | None,
) -> dict[str, Any]:
    n_total, n_hit, n_miss = getattr(legacy, "_last_sample_cov", (0, 0, 0))
    cov = (float(n_hit) / float(n_total)) if n_total else None
    reason, revs, elapsed = getattr(legacy, "_last_sample_reason", ("-", 0.0, 0.0))

    payload: dict[str, Any] = {
        "idx": int(section_index),
        "cov": cov,
        "cov_od": cov,
        "n_od": getattr(legacy, "_last_sample_n_od", None),
        "n_id": getattr(legacy, "_last_sample_n_id", None),
        "miss": n_miss,
        "max_gap_deg": getattr(legacy, "_last_sample_max_gap_deg", None),
        "reason": reason,
        "revs": revs,
        "elapsed": elapsed,
    }
    if str(scan_mode or "").upper() == "SPLIT":
        n_total_i, n_hit_i, n_miss_i = getattr(legacy, "_last_sample_cov_id", (0, 0, 0))
        cov_i = (float(n_hit_i) / float(n_total_i)) if n_total_i else None
        reason_i, revs_i, elapsed_i = getattr(legacy, "_last_sample_reason_id", ("-", 0.0, 0.0))
        payload.update(
            {
                "cov_id": cov_i,
                "n_od": n_od_pass,
                "n_id": n_id_pass,
                "miss_id": n_miss_i,
                "max_gap_deg_id": getattr(legacy, "_last_sample_max_gap_deg_id", None),
                "reason_id": reason_i,
                "revs_id": revs_i,
                "elapsed_id": elapsed_i,
                "split_shift_deg": split_shift_deg,
                "coax_unreliable": coax_unreliable,
                "keep_spinning": keep_spinning,
            }
        )
    return payload


def _build_measure_row_from_sampling(
    *,
    legacy: AutoFlow,
    recipe: Recipe,
    app: Any,
    section_index: int,
    z_pos_mm: float,
    x_abs: float,
    coords_od: np.ndarray,
    coords_id: np.ndarray,
    raw_od: str,
    raw_id: str,
    raw_points: list[dict],
    scan_mode: str,
    split_shift_deg: float | None,
    coax_unreliable: bool | None,
    centers_xyz: list[tuple[float, float, float]],
    centers_xyz_id: list[tuple[float, float, float]],
    concentricity_list: list[float],
    validation_fit_payload: dict[str, Any] | None = None,
) -> MeasureRow:
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

    xc, yc, _r_fit, _sigma = legacy._fit_circle(
        coords_od, weights=getattr(legacy, "_last_fit_weights_od", None)
    )
    xci = yci = _r_fit_i = _sigma_i = 0.0
    if not id_single_enable:
        xci, yci, _r_fit_i, _sigma_i = legacy._fit_circle(
            coords_id, weights=getattr(legacy, "_last_fit_weights_id", None)
        )

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
        od_vals = np.asarray([float(p.get("od_mm")) for p in raw_points if p.get("od_mm") is not None], dtype=float)
    except Exception:
        od_vals = np.asarray([], dtype=float)
    od_pp_mm = _pp_strict(od_vals)
    od_pp_rob_mm = _pp_robust(od_vals)
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

    id_fit = None
    id_fit_diam = None
    id_fit_vals = None
    sim_disp_enabled = bool(getattr(app, "sim_disp_enabled", False)) if app is not None else False
    if (not id_single_enable) and bool(getattr(recipe, "id_use_fit", False)) and (not sim_disp_enabled):
        delta_c = float(legacy._idcal_get_delta_c_active())
        id_fit, id_fit_vals = legacy._id_fit_from_raw_points(
            raw_points,
            delta_c,
            theta_delay_s=float(getattr(recipe, "theta_delay_s", 0.0) or 0.0),
        )
        if id_fit is not None:
            try:
                id_fit_diam = float(id_fit.get("diam", None))
            except Exception:
                id_fit_diam = None

        if id_fit_vals is None:
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
        od_round_fit_mm, od_round_fit_rob_mm = legacy._od_round_fit_from_raw_points(
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
        delta_c = float(legacy._idcal_get_delta_c_active())
    except Exception:
        delta_c = 0.0
    if not id_single_enable:
        try:
            id_round_fit_mm, id_round_fit_rob_mm = legacy._id_round_fit_from_raw_points(
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

    centers_xyz.append((float(center_od_x), float(center_od_y), float(z_pos_mm)))

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
        concentricity_list.append(float(concentricity))
        centers_xyz_id.append((float(center_id_x), float(center_id_y), float(z_pos_mm)))

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
            if app is not None:
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
                    id_single_res = app.calc_id_single_from_out2(th_list, out2_list, recipe)
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

    if validation_fit_payload is not None:
        validation_fit_payload.clear()
        validation_fit_payload.update(
            {
                "od_center_x_mm": _optional_finite_float(center_od_x),
                "od_center_y_mm": _optional_finite_float(center_od_y),
                "od_radius_mm": _optional_finite_float(od_radius_fit_mm),
                "od_diameter_fit_mm": _optional_finite_float(od_diameter_fit_mm),
                "id_center_x_mm": _optional_finite_float(center_id_x),
                "id_center_y_mm": _optional_finite_float(center_id_y),
                "id_radius_mm": _optional_finite_float(id_radius_fit_mm),
                "id_diameter_fit_mm": _optional_finite_float(id_diameter_fit_mm),
                "od_ecc_mm": (
                    _optional_finite_float(od_e)
                    if od_use_edges
                    else None
                ),
                "id_ecc_mm": _optional_finite_float(id_e),
                "concentricity_mm": _optional_finite_float(concentricity),
            }
        )

    try:
        od_tol_v = float(recipe.od_tol_mm)
    except Exception:
        od_tol_v = 0.0
    if id_dev is None:
        ok_flag = abs(od_dev) <= float(od_tol_v)
    else:
        ok_flag = (abs(od_dev) <= float(od_tol_v)) and (abs(id_dev) <= float(od_tol_v))

    return MeasureRow(
        idx=int(section_index),
        x_ui=float(z_pos_mm),
        x_abs=float(x_abs),
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


def measure_current_position_section_capture(
    *,
    gateway: MachineGateway,
    recipe: Recipe,
    calibration: CalibrationSnapshot,
) -> tuple[MeasureRow, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    runtime_app = getattr(gateway, "app", None)
    if runtime_app is None:
        raise RuntimeError("measure_current_position_section_capture requires gateway.app")

    legacy = AutoFlow(runtime_app)
    legacy._current_recipe = recipe
    legacy._calibration_snapshot = calibration

    section_index = 1
    try:
        x_abs = float(getattr(gateway.get_axis_copy(0), "act_pos", 0.0) or 0.0)
    except Exception:
        x_abs = 0.0
    try:
        axis_cal = getattr(runtime_app, "axis_cal", None)
        if axis_cal is not None and hasattr(axis_cal, "abs_to_z_disp"):
            z_pos_mm = float(axis_cal.abs_to_z_disp(0, x_abs))
        else:
            z_pos_mm = float(x_abs)
    except Exception:
        z_pos_mm = float(x_abs)

    scan_mode = _resolve_recipe_sampling_mode(recipe)
    split_shift_deg = None
    coax_unreliable = None
    keep_spinning = bool(getattr(recipe, "split_keep_spinning", True))
    slip_check = bool(getattr(recipe, "split_slip_check", True))
    slip_max_deg = float(getattr(recipe, "split_slip_max_deg", 5.0) or 5.0)
    omega_cv_max = float(getattr(recipe, "split_omega_cv_max", 0.25) or 0.25)

    windows: list[dict[str, Any]] = []
    centers_xyz: list[tuple[float, float, float]] = []
    centers_xyz_id: list[tuple[float, float, float]] = []
    concentricity_list: list[float] = []
    fit_payload: dict[str, Any] = {}

    if scan_mode == "SPLIT":
        coords_od, _coords_id0, raw_od, _raw_id0, raw_points_od = legacy._sample_circle_points_dual(
            recipe,
            section_idx=0,
            sample_od=True,
            sample_id=False,
            phase="OD",
        )
        cov_od = getattr(legacy, "_last_sample_cov", (0, 0, 0))
        reason_od = getattr(legacy, "_last_sample_reason", ("-", 0.0, 0.0))
        max_gap_od = getattr(legacy, "_last_sample_max_gap_deg", None)
        w_od = getattr(legacy, "_last_fit_weights_od", None)
        n_od_pass = getattr(legacy, "_last_sample_n_od", None)

        _coords_od0, coords_id, _raw_od0, raw_id, raw_points_id = legacy._sample_circle_points_dual(
            recipe,
            section_idx=0,
            sample_od=False,
            sample_id=True,
            phase="ID",
        )
        cov_id = getattr(legacy, "_last_sample_cov", (0, 0, 0))
        reason_id = getattr(legacy, "_last_sample_reason", ("-", 0.0, 0.0))
        max_gap_id = getattr(legacy, "_last_sample_max_gap_deg", None)
        w_id = getattr(legacy, "_last_fit_weights_id", None)
        n_id_pass = getattr(legacy, "_last_sample_n_id", None)

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

        annotated_od = _annotate_validation_raw_points(
            raw_points=list(raw_points_od or []),
            section_index=section_index,
            z_pos_mm=z_pos_mm,
            window_index=1,
            window_role="OD",
        )
        annotated_id = _annotate_validation_raw_points(
            raw_points=list(raw_points_id or []),
            section_index=section_index,
            z_pos_mm=z_pos_mm,
            window_index=2,
            window_role="ID",
        )
        raw_points = list(annotated_od) + list(annotated_id)
        for sample_idx, point in enumerate(raw_points):
            if isinstance(point, dict):
                point["sample_idx"] = int(sample_idx)

        windows.append(
            _build_validation_window_payload(
                window_index=1,
                window_role="OD",
                raw_points=annotated_od,
                sample_cov=cov_od,
                sample_reason=reason_od,
                n_od=n_od_pass,
                n_id=None,
                max_gap_deg=max_gap_od,
            )
        )
        windows.append(
            _build_validation_window_payload(
                window_index=2,
                window_role="ID",
                raw_points=annotated_id,
                sample_cov=cov_id,
                sample_reason=reason_id,
                n_od=None,
                n_id=n_id_pass,
                max_gap_deg=max_gap_id,
            )
        )

        legacy._last_fit_weights_od = w_od
        legacy._last_fit_weights_id = w_id
        legacy._last_sample_cov = cov_od
        legacy._last_sample_reason = reason_od
        legacy._last_sample_max_gap_deg = max_gap_od
        legacy._last_sample_cov_id = cov_id
        legacy._last_sample_n_od_pass = n_od_pass
        legacy._last_sample_n_id_pass = n_id_pass
        legacy._last_sample_reason_id = reason_id
        legacy._last_sample_max_gap_deg_id = max_gap_id
    else:
        coords_od, coords_id, raw_od, raw_id, raw_points_sync = legacy._sample_circle_points_dual(
            recipe,
            section_idx=0,
            sample_od=True,
            sample_id=True,
            phase="SYNC",
        )
        n_od_pass = None
        n_id_pass = None
        raw_points = _annotate_validation_raw_points(
            raw_points=list(raw_points_sync or []),
            section_index=section_index,
            z_pos_mm=z_pos_mm,
            window_index=1,
            window_role="SYNC",
        )
        for sample_idx, point in enumerate(raw_points):
            if isinstance(point, dict):
                point["sample_idx"] = int(sample_idx)
        windows.append(
            _build_validation_window_payload(
                window_index=1,
                window_role="SYNC",
                raw_points=raw_points,
                sample_cov=getattr(legacy, "_last_sample_cov", (0, 0, 0)),
                sample_reason=getattr(legacy, "_last_sample_reason", ("-", 0.0, 0.0)),
                n_od=getattr(legacy, "_last_sample_n_od", None),
                n_id=getattr(legacy, "_last_sample_n_id", None),
                max_gap_deg=getattr(legacy, "_last_sample_max_gap_deg", None),
            )
        )

    coverage_payload = _build_validation_coverage_payload(
        legacy=legacy,
        section_index=section_index,
        scan_mode=scan_mode,
        split_shift_deg=split_shift_deg,
        coax_unreliable=coax_unreliable,
        keep_spinning=keep_spinning,
        n_od_pass=n_od_pass,
        n_id_pass=n_id_pass,
    )
    row = _build_measure_row_from_sampling(
        legacy=legacy,
        recipe=recipe,
        app=runtime_app,
        section_index=section_index,
        z_pos_mm=float(z_pos_mm),
        x_abs=float(x_abs),
        coords_od=coords_od,
        coords_id=coords_id,
        raw_od=str(raw_od),
        raw_id=str(raw_id),
        raw_points=raw_points,
        scan_mode=scan_mode,
        split_shift_deg=split_shift_deg,
        coax_unreliable=coax_unreliable,
        centers_xyz=centers_xyz,
        centers_xyz_id=centers_xyz_id,
        concentricity_list=concentricity_list,
        validation_fit_payload=fit_payload,
    )
    return row, raw_points, windows, coverage_payload, dict(fit_payload)


def measure_current_position_od_avg(
    *,
    gateway: MachineGateway,
    recipe: Recipe,
    calibration: CalibrationSnapshot,
) -> float:
    """Sample OD once at the current machine position and return od_avg."""
    capture = measure_current_position_section_capture(
        gateway=gateway,
        recipe=recipe,
        calibration=calibration,
    )
    row = capture[0]
    return float(row.od_avg)


class AutoFlowOrchestrator:
    """Explicit dependency shell for the formal measurement workflow."""

    def __init__(
        self,
        gateway: MachineGateway,
        recipe: Recipe,
        calibration: CalibrationSnapshot,
        run_session: RunSession,
        event_sink: EventSink,
        *,
        runtime_state: RuntimeState | None = None,
        run_repository: RunRepositoryProtocol | None = None,
    ) -> None:
        self.gateway = gateway
        self.recipe = recipe
        self.calibration = calibration
        self.run_session = run_session
        self.event_sink = event_sink
        self.runtime_state = runtime_state or RuntimeState.from_run_session(run_session)
        self.run_repository = run_repository
        self.production_workflow = (
            ProductionWorkflow(
                recipe=recipe,
                calibration=calibration,
                runtime_state=self.runtime_state,
                gateway=gateway,
                run_repository=run_repository,
            )
            if run_repository is not None
            else None
        )
        self.run_result: RunResult | None = None
        self.state = "IDLE"
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._runtime_app: App | None = getattr(gateway, "app", None)
        self._legacy_flow: AutoFlow | None = None
        if self._runtime_app is not None:
            self._legacy_flow = AutoFlow(self._runtime_app)
            self._legacy_flow.stop_event = self._stop_event
            self._legacy_flow._current_recipe = recipe
            self._legacy_flow._calibration_snapshot = calibration

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    def is_alive(self) -> bool:
        """Compatibility helper so App can A/B old and new runners easily."""
        return self.is_running

    def start(self) -> None:
        """Start the orchestrator on a background thread."""
        if self.is_running:
            return
        self._stop_event.clear()
        self.run_session.end_ts = None
        self._set_internal_state("STARTING")
        self._thread = threading.Thread(
            target=self.run,
            name="AutoFlowOrchestrator",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Request the orchestrator to stop gracefully."""
        if not self.is_running:
            return
        self._stop_event.set()
        self._set_internal_state("STOPPING")
        self._emit_state("STOPPING", "Stop request received")

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def run(self) -> None:
        """Workflow entrypoint for the staged measurement orchestrator."""
        if self.run_session.start_ts is None:
            self.run_session.start_ts = time.time()
        self.run_session.end_ts = None
        self.runtime_state.started_at_ts = self.run_session.start_ts
        self.runtime_state.finished_at_ts = None
        if self.production_workflow is not None:
            try:
                self.production_workflow.ensure_identity()
            except Exception:
                pass
        self._set_internal_state("RUNNING")
        self._emit_state("RUN", "Auto measurement started")

        status = "DONE"
        message = "Measurement completed"
        try:
            self._run_main_loop()
        except _StopRequested as exc:
            status = "STOP"
            message = str(exc) or "User stopped"
            self._set_internal_state("STOPPED")
        except Exception as exc:
            status = "ERR"
            message = str(exc)
            self._set_internal_state("ERROR")
        finally:
            self.run_session.end_ts = time.time()
            try:
                self.gateway.stop(3)
            except Exception:
                pass
            if self._stop_event.is_set():
                try:
                    self.gateway.abort_motion()
                except Exception:
                    pass

        if status == "DONE":
            self._set_internal_state("DONE")
        if self.production_workflow is not None:
            try:
                self.run_result = self.production_workflow.build_run_result(
                    status=status,
                    message=message,
                    finished_at_ts=self.run_session.end_ts,
                )
            except Exception:
                self.run_result = None
        self._emit_state(status, message)

    def _run_main_loop(self) -> None:
        centers_xyz: list[tuple[float, float, float]] = []
        centers_xyz_id: list[tuple[float, float, float]] = []
        concentricity_list: list[float] = []

        self._apply_start_anchor_if_available()
        self._prepare_linear_axes()
        self._prepare_ax2_and_clamps()
        self._run_optional_length_stage()
        self._move_ax2_to_rotate_position()
        section_plan = self._build_section_plan()
        if not section_plan.sections:
            raise ValueError("section_count must be > 0")
        self._confirm_rotate_clamp()
        self._prepare_ax3_rotation()
        self._run_section_loop(
            section_plan,
            centers_xyz=centers_xyz,
            centers_xyz_id=centers_xyz_id,
            concentricity_list=concentricity_list,
        )
        self._run_postcalc(
            centers_xyz=centers_xyz,
            centers_xyz_id=centers_xyz_id,
            concentricity_list=concentricity_list,
        )
        self._stop_ax3_rotation()
        self._return_to_standby()

    def _prepare_linear_axes(self) -> None:
        for axis in (0, 1, 4):
            self._ensure_axis_ready(axis)

    def _prepare_ax2_and_clamps(self) -> None:
        self._ensure_axis_ready(2)
        self._emit_state("PREP", "Clamp prepare: main on, sub off")
        self._write_y_point(10, 1)
        self._write_y_point(11, 0)
        time.sleep(0.25)

    def _run_optional_length_stage(self) -> None:
        if not bool(getattr(self.recipe, "len_enable", False)):
            return

        current_ax2_abs = float(getattr(self.gateway.get_axis_copy(2), "act_pos", 0.0) or 0.0)
        ax2_plan = resolve_ax2_position_plan(self.recipe, current_ax2_abs=current_ax2_abs)
        if ax2_plan.has_length_target:
            self._move_axis_abs(
                2,
                float(ax2_plan.length_target_abs),
                strict=True,
                context="AUTO_AX2_LEN",
                state="PREP",
                message_template="AX2 -> length position: {target:.3f}",
            )
        else:
            self._emit_state("WARN", "Length enabled but AX2 length position is not saved")

        payload = self._measure_length_legacy()
        if self.production_workflow is not None:
            self.production_workflow.record_length(payload)
        self.event_sink.publish_length(payload)
        app = self._runtime_app
        if app is not None:
            try:
                setattr(app, "_run_len_result", payload)
            except Exception:
                pass

        standby_plan = resolve_standby_plan(self.recipe)
        if standby_plan.enabled and 0 in standby_plan.targets_abs:
            try:
                self._move_axis_abs(
                    0,
                    float(standby_plan.targets_abs[0]),
                    strict=True,
                    context="AUTO_AX0_STANDBY_AFTER_LEN",
                    state="PREP",
                    message_template="AX0 -> standby: {target:.3f}",
                )
            except Exception as exc:
                self._emit_state("WARN", f"AX0 standby move failed: {exc}")

        self._raise_if_stop_requested()

    def _move_ax2_to_rotate_position(self) -> None:
        self._move_axis_abs(
            2,
            require_ax2_rotate_target_abs(self.recipe),
            strict=True,
            context="AUTO_AX2_ROT",
            state="PREP",
            message_template="AX2 -> rotate position: {target:.3f}",
        )

    def _confirm_rotate_clamp(self) -> None:
        self._write_y_point(11, 1)
        time.sleep(0.25)
        self._raise_if_stop_requested()

        app = self._runtime_app
        if app is None or not hasattr(app, "operator_confirm"):
            return
        result = "timeout"
        try:
            result = app.operator_confirm(
                "Clamp Confirm",
                "Confirm sub clamp is ready.\n\n- Press X3 or click confirm to continue\n- Stop to abort",
                allow_stop=True,
                timeout_s=60.0,
            )
        except Exception:
            result = "timeout"
        if result != "confirm":
            raise _StopRequested(f"Operator canceled: {result}")

    def _prepare_ax3_rotation(self) -> None:
        self._ensure_axis_ready(3)
        self._start_ax3_rotation(emit_state=True)

    def _run_section_loop(
        self,
        section_plan,
        *,
        centers_xyz: list[tuple[float, float, float]],
        centers_xyz_id: list[tuple[float, float, float]],
        concentricity_list: list[float],
    ) -> None:
        section_total = len(section_plan.sections)
        for row in section_plan.sections:
            self._raise_if_stop_requested()
            section_index = int(row.section_index)
            z_pos_mm = float(row.z_od_disp)
            targets = row.linear_targets()
            self._emit_progress(
                section_index=section_index,
                section_total=section_total,
                z_pos_mm=float(z_pos_mm),
                ax0_abs=float(row.ax0_abs),
            )
            self._emit_state("RUN", f"Section {section_index}/{section_total} positioning")
            self._move_linear_axes_to_targets(
                targets,
                context=f"AUTO_SEC_{section_index}",
            )
            self._wait_before_section_capture(
                section_index=section_index,
                section_total=section_total,
                delay_s=float(getattr(self.recipe, "sample_delay_s", 0.0) or 0.0),
            )
            self._measure_section(
                section_index=section_index,
                z_pos_mm=float(z_pos_mm),
                x_abs=float(row.ax0_abs),
                centers_xyz=centers_xyz,
                centers_xyz_id=centers_xyz_id,
                concentricity_list=concentricity_list,
            )

    def _wait_before_section_capture(
        self,
        *,
        section_index: int,
        section_total: int,
        delay_s: float,
    ) -> None:
        delay = float(delay_s or 0.0)
        if delay <= 0.0:
            return
        self._emit_state(
            "RUN",
            f"Section {int(section_index)}/{int(section_total)} wait sample delay: {delay:.3f}s",
        )
        deadline = time.monotonic() + delay
        while True:
            self._raise_if_stop_requested()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            time.sleep(min(0.05, remaining))

    def _measure_section(
        self,
        *,
        section_index: int,
        z_pos_mm: float,
        x_abs: float,
        centers_xyz: list[tuple[float, float, float]],
        centers_xyz_id: list[tuple[float, float, float]],
        concentricity_list: list[float],
    ) -> None:
        legacy = self._require_legacy_flow()
        recipe = self.recipe
        i = int(section_index) - 1

        scan_mode = _resolve_recipe_sampling_mode(recipe)
        split_shift_deg = None
        coax_unreliable = None
        keep_spinning = bool(getattr(recipe, "split_keep_spinning", True))
        slip_check = bool(getattr(recipe, "split_slip_check", True))
        slip_max_deg = float(getattr(recipe, "split_slip_max_deg", 5.0) or 5.0)
        omega_cv_max = float(getattr(recipe, "split_omega_cv_max", 0.25) or 0.25)

        try:
            legacy_log(
                "SECTION_START",
                section=section_index,
                z_disp=z_pos_mm,
                ax0_abs=x_abs,
            )
        except Exception:
            pass

        if scan_mode == "SPLIT":
            coords_od, _coords_id0, raw_od, _raw_id0, raw_points_od = legacy._sample_circle_points_dual(
                recipe,
                section_idx=i,
                sample_od=True,
                sample_id=False,
                phase="OD",
            )
            cov_od = getattr(legacy, "_last_sample_cov", (0, 0, 0))
            reason_od = getattr(legacy, "_last_sample_reason", ("-", 0.0, 0.0))
            max_gap_od = getattr(legacy, "_last_sample_max_gap_deg", None)
            w_od = getattr(legacy, "_last_fit_weights_od", None)
            n_od_pass = getattr(legacy, "_last_sample_n_od", None)

            if not keep_spinning:
                try:
                    self._stop_ax3_rotation()
                except Exception:
                    pass
                try:
                    self._start_ax3_rotation(emit_state=False)
                except Exception:
                    pass

            _coords_od0, coords_id, _raw_od0, raw_id, raw_points_id = legacy._sample_circle_points_dual(
                recipe,
                section_idx=i,
                sample_od=False,
                sample_id=True,
                phase="ID",
            )
            cov_id = getattr(legacy, "_last_sample_cov", (0, 0, 0))
            reason_id = getattr(legacy, "_last_sample_reason", ("-", 0.0, 0.0))
            max_gap_id = getattr(legacy, "_last_sample_max_gap_deg", None)
            w_id = getattr(legacy, "_last_fit_weights_id", None)
            n_id_pass = getattr(legacy, "_last_sample_n_id", None)

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

            raw_points = list(raw_points_od or []) + list(raw_points_id or [])
            legacy._last_fit_weights_od = w_od
            legacy._last_fit_weights_id = w_id
            legacy._last_sample_cov = cov_od
            legacy._last_sample_reason = reason_od
            legacy._last_sample_max_gap_deg = max_gap_od
            legacy._last_sample_cov_id = cov_id
            legacy._last_sample_n_od_pass = n_od_pass
            legacy._last_sample_n_id_pass = n_id_pass
            legacy._last_sample_reason_id = reason_id
            legacy._last_sample_max_gap_deg_id = max_gap_id
        else:
            coords_od, coords_id, raw_od, raw_id, raw_points = legacy._sample_circle_points_dual(
                recipe,
                section_idx=i,
                sample_od=True,
                sample_id=True,
                phase="SYNC",
            )
            n_od_pass = None
            n_id_pass = None

        self._publish_section_raw_points(
            raw_points=raw_points,
            section_index=section_index,
            z_pos_mm=float(z_pos_mm),
        )
        self._publish_section_coverage(
            section_index=section_index,
            scan_mode=scan_mode,
            split_shift_deg=split_shift_deg,
            coax_unreliable=coax_unreliable,
            keep_spinning=keep_spinning,
            n_od_pass=n_od_pass,
            n_id_pass=n_id_pass,
        )

        row = self._build_section_row(
            section_index=section_index,
            z_pos_mm=float(z_pos_mm),
            x_abs=float(x_abs),
            coords_od=coords_od,
            coords_id=coords_id,
            raw_od=str(raw_od),
            raw_id=str(raw_id),
            raw_points=raw_points,
            scan_mode=scan_mode,
            split_shift_deg=split_shift_deg,
            coax_unreliable=coax_unreliable,
            centers_xyz=centers_xyz,
            centers_xyz_id=centers_xyz_id,
            concentricity_list=concentricity_list,
        )
        if self.production_workflow is not None:
            self.production_workflow.record_row(row)
        self.event_sink.publish_row(row)

    def _publish_section_raw_points(
        self,
        *,
        raw_points: list[dict],
        section_index: int,
        z_pos_mm: float,
    ) -> None:
        try:
            for j, point in enumerate(raw_points):
                if isinstance(point, dict):
                    point["section_idx"] = int(section_index)
                    point["z_pos_mm"] = float(z_pos_mm)
                    point["sample_idx"] = int(j)
        except Exception:
            pass
        if self.production_workflow is not None:
            self.production_workflow.record_raw_points(raw_points)
        self.event_sink.publish_raw_points(raw_points)

    def _publish_section_coverage(
        self,
        *,
        section_index: int,
        scan_mode: str,
        split_shift_deg: float | None,
        coax_unreliable: bool | None,
        keep_spinning: bool,
        n_od_pass: int | None,
        n_id_pass: int | None,
    ) -> None:
        legacy = self._require_legacy_flow()
        try:
            n_total, n_hit, n_miss = getattr(legacy, "_last_sample_cov", (0, 0, 0))
            cov = (float(n_hit) / float(n_total)) if n_total else None
            reason, revs, elapsed = getattr(legacy, "_last_sample_reason", ("-", 0.0, 0.0))

            payload: dict[str, Any] = {
                "idx": int(section_index),
                "cov": cov,
                "cov_od": cov,
                "n_od": getattr(legacy, "_last_sample_n_od", None),
                "n_id": getattr(legacy, "_last_sample_n_id", None),
                "miss": n_miss,
                "max_gap_deg": getattr(legacy, "_last_sample_max_gap_deg", None),
                "reason": reason,
                "revs": revs,
                "elapsed": elapsed,
            }

            if scan_mode == "SPLIT":
                n_total_i, n_hit_i, n_miss_i = getattr(legacy, "_last_sample_cov_id", (0, 0, 0))
                cov_i = (float(n_hit_i) / float(n_total_i)) if n_total_i else None
                reason_i, revs_i, elapsed_i = getattr(legacy, "_last_sample_reason_id", ("-", 0.0, 0.0))
                payload.update(
                    {
                        "cov_id": cov_i,
                        "n_od": n_od_pass,
                        "n_id": n_id_pass,
                        "miss_id": n_miss_i,
                        "max_gap_deg_id": getattr(legacy, "_last_sample_max_gap_deg_id", None),
                        "reason_id": reason_i,
                        "revs_id": revs_i,
                        "elapsed_id": elapsed_i,
                        "split_shift_deg": split_shift_deg,
                        "coax_unreliable": coax_unreliable,
                        "keep_spinning": keep_spinning,
                    }
                )

            if self.production_workflow is not None:
                self.production_workflow.record_coverage(payload)
            self.event_sink.publish_coverage(payload)
        except Exception:
            pass

    def _build_section_row(
        self,
        *,
        section_index: int,
        z_pos_mm: float,
        x_abs: float,
        coords_od: np.ndarray,
        coords_id: np.ndarray,
        raw_od: str,
        raw_id: str,
        raw_points: list[dict],
        scan_mode: str,
        split_shift_deg: float | None,
        coax_unreliable: bool | None,
        centers_xyz: list[tuple[float, float, float]],
        centers_xyz_id: list[tuple[float, float, float]],
        concentricity_list: list[float],
    ) -> MeasureRow:
        return _build_measure_row_from_sampling(
            legacy=self._require_legacy_flow(),
            recipe=self.recipe,
            app=self._runtime_app,
            section_index=section_index,
            z_pos_mm=z_pos_mm,
            x_abs=x_abs,
            coords_od=coords_od,
            coords_id=coords_id,
            raw_od=raw_od,
            raw_id=raw_id,
            raw_points=raw_points,
            scan_mode=scan_mode,
            split_shift_deg=split_shift_deg,
            coax_unreliable=coax_unreliable,
            centers_xyz=centers_xyz,
            centers_xyz_id=centers_xyz_id,
            concentricity_list=concentricity_list,
        )

    def _run_postcalc(
        self,
        *,
        centers_xyz: list[tuple[float, float, float]],
        centers_xyz_id: list[tuple[float, float, float]],
        concentricity_list: list[float],
    ) -> None:
        recipe = self.recipe
        try:
            result = compute_postcalc_result(
                centers_xyz,
                centers_xyz_id,
                concentricity_list=concentricity_list,
                id_single_enable=bool(getattr(recipe, "id_single_enable", False)),
            )
            if self.production_workflow is not None:
                self.production_workflow.record_summary(result.straightness_payload, source="straightness")
            self.event_sink.publish_straightness(result.straightness_payload)
            if self.production_workflow is not None:
                self.production_workflow.record_summary(result.postcalc_payload, source="postcalc")
            self.event_sink.publish_postcalc(result.postcalc_payload)
        except Exception:
            if self.production_workflow is not None:
                self.production_workflow.record_summary(
                    {
                        "straight_od": None,
                        "straight_id": None,
                        "axis_dist": None,
                        "conc_max": None,
                        "axis_span_max": None,
                    },
                    source="straightness",
                )
            self.event_sink.publish_straightness(
                {
                    "straight_od": None,
                    "straight_id": None,
                    "axis_dist": None,
                    "conc_max": None,
                    "axis_span_max": None,
                }
            )

    def _return_to_standby(self) -> None:
        standby_plan = resolve_standby_plan(self.recipe)
        if not standby_plan.enabled:
            return
        try:
            self._move_linear_axes_to_targets(
                dict(standby_plan.targets_abs),
                context="AUTO_STANDBY",
                strict=False,
            )
        except Exception:
            pass

    def _measure_length_legacy(self) -> dict[str, Any]:
        if not bool(getattr(self.recipe, "len_enable", False)):
            return {
                "enabled": False,
                "skipped": True,
                "ok": False,
                "reason": "DISABLED",
                "z_low": None,
                "z_high": None,
                "length_mm": None,
                "t_s": 0.0,
            }
        if not bool(getattr(self.recipe, "ax2_len_valid", False)):
            return {
                "enabled": True,
                "skipped": True,
                "ok": False,
                "reason": "NO_AX2_LEN_POS",
                "z_low": None,
                "z_high": None,
                "length_mm": None,
                "t_s": 0.0,
            }
        if self._legacy_flow is None:
            return {
                "enabled": True,
                "skipped": True,
                "ok": False,
                "reason": "ORCHESTRATOR_STAGE_ONLY",
                "z_low": None,
                "z_high": None,
                "length_mm": None,
                "t_s": 0.0,
            }

        self._emit_state("LEN", "Auto length measurement")
        try:
            return dict(self._legacy_flow._auto_measure_length(self.recipe))
        except Exception as exc:
            return {
                "enabled": True,
                "skipped": False,
                "ok": False,
                "reason": f"EXC({exc})",
                "z_low": None,
                "z_high": None,
                "length_mm": None,
                "t_s": 0.0,
            }

    def _resolve_section_positions(self) -> list[float]:
        return list(plan_section_positions(self.recipe).positions_z)

    def _build_section_plan(self):
        axis_cal = self._require_axis_cal()
        soft_limits = {
            0: self._soft_limits_from_axis(0),
            1: self._soft_limits_from_axis(1),
            4: self._soft_limits_from_axis(4),
        }
        return build_recipe_section_plan(
            self.recipe,
            axis_cal,
            ax2_abs=float(self._get_ax2_keepout_ref_abs()),
            soft_limits_abs=soft_limits,
        )

    def _resolve_section_targets(
        self,
        *,
        axis_cal: AxisCal,
        section_index: int,
        z_pos_mm: float,
    ) -> dict[int, float]:
        del axis_cal
        del z_pos_mm
        row = self._build_section_plan().section_at(section_index)
        targets = row.linear_targets()
        for axis, target in list(targets.items()):
            targets[axis] = self.gateway.apply_soft_limits_abs(
                int(axis),
                float(target),
                strict=True,
                context=f"AUTO_SEC_{section_index}",
            )
        return targets

    def _move_linear_axes_to_targets(
        self,
        targets: dict[int, float],
        *,
        context: str,
        strict: bool = True,
    ) -> None:
        resolved: dict[int, float] = {}
        for axis, target in targets.items():
            resolved[int(axis)] = self.gateway.apply_soft_limits_abs(
                int(axis),
                float(target),
                strict=bool(strict),
                context=context,
            )
        for axis, target in resolved.items():
            self.gateway.movea_abs(int(axis), float(target), context=context)
        for axis, target in resolved.items():
            ok = self._wait_in_position(int(axis), float(target), pos_tol=0.05, timeout_s=30.0)
            if not ok:
                self._raise_if_stop_requested()
                raise TimeoutError(f"AX{axis} in-position timeout: {target:.3f}")

    def _move_axis_abs(
        self,
        axis: int,
        target: float,
        *,
        strict: bool,
        context: str,
        state: str,
        message_template: str,
    ) -> None:
        target_resolved = self.gateway.apply_soft_limits_abs(
            int(axis),
            float(target),
            strict=bool(strict),
            context=context,
        )
        self._emit_state(state, message_template.format(target=float(target_resolved)))
        self.gateway.movea_abs(int(axis), float(target_resolved), context=context)
        ok = self._wait_in_position(int(axis), float(target_resolved), pos_tol=0.05, timeout_s=25.0)
        if not ok:
            self._raise_if_stop_requested()
            raise TimeoutError(f"AX{axis} in-position timeout: {target_resolved:.3f}")

    def _start_ax3_rotation(self, *, emit_state: bool) -> None:
        velocity = self._get_ax3_velocity()
        if emit_state:
            self._emit_state("PREP", f"AX3 rotate start: {velocity:.3f}")
        self.gateway.velmove(3, float(velocity))
        time.sleep(0.20)

    def _stop_ax3_rotation(self) -> None:
        try:
            self.gateway.stop(3)
            t0 = time.time()
            while (time.time() - t0) < 10.0:
                self._raise_if_stop_requested()
                ac3 = self.gateway.get_axis_copy(3)
                if not self._is_moving(int(getattr(ac3, "sts", 0))):
                    break
                time.sleep(0.08)
        except _StopRequested:
            raise
        except Exception:
            pass

    def _get_ax3_velocity(self) -> float:
        try:
            velocity = float(getattr(self.recipe, "rot_vel_velmove", 0.0) or 0.0)
        except Exception:
            velocity = 0.0
        if abs(velocity) <= 1e-9:
            velocity = 200.0
        return float(velocity)

    def _fit_line_and_dist(
        self, points_xyz: list[tuple[float, float, float]]
    ) -> tuple[float, list[float], np.ndarray, np.ndarray]:
        if len(points_xyz) < 2:
            return 0.0, [0.0 for _ in points_xyz], np.zeros(3, dtype=float), np.array([0.0, 0.0, 1.0], dtype=float)
        P = np.array(points_xyz, dtype=float)
        p0 = P.mean(axis=0)
        Q = P - p0
        C = (Q.T @ Q) / max(1, Q.shape[0])
        w, v = np.linalg.eigh(C)
        d = v[:, int(np.argmax(w))]
        d = d / (np.linalg.norm(d) + 1e-12)
        t = Q @ d
        proj = np.outer(t, d)
        R = Q - proj
        dist = np.linalg.norm(R, axis=1)
        straight = float(dist.max() - dist.min()) if dist.size else 0.0
        return straight, [float(x) for x in dist.tolist()], p0, d

    def _line_distance(self, p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> float:
        d1n = d1 / (np.linalg.norm(d1) + 1e-12)
        d2n = d2 / (np.linalg.norm(d2) + 1e-12)
        n = np.cross(d1n, d2n)
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            v = p2 - p1
            return float(np.linalg.norm(np.cross(v, d1n)))
        return float(abs(np.dot((p2 - p1), n)) / nn)

    def _tilt_and_end_offset(
        self,
        p0: np.ndarray,
        d: np.ndarray,
        pts_xyz: list[tuple[float, float, float]],
    ) -> tuple[float | None, float | None, float | None]:
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
            slope = float(math.hypot(sx, sy))
            tilt_deg = float(math.degrees(math.atan(slope)))
            t_min = (z_min - float(p0[2])) / dz
            t_max = (z_max - float(p0[2])) / dz
            p_min = p0 + t_min * d
            p_max = p0 + t_max * d
            end_off = float(math.hypot(float(p_max[0] - p_min[0]), float(p_max[1] - p_min[1])))
            return tilt_deg, end_off, slope
        except Exception:
            return None, None, None

    def _apply_start_anchor_if_available(self) -> None:
        plan = resolve_start_anchor_plan(self.recipe)
        if not plan.enabled:
            return
        app = self._runtime_app
        if app is None:
            return
        apply_start = getattr(app, "_apply_start_anchor_from_recipe", None)
        if callable(apply_start):
            apply_start()

    def _ensure_axis_ready(self, axis: int) -> None:
        snapshot = self.gateway.get_axis_copy(int(axis))
        sts = int(getattr(snapshot, "sts", 0) or 0)
        err = int(getattr(snapshot, "err", 0) or 0)
        if self._is_fault(sts, err):
            raise RuntimeError(f"AX{axis} fault, err={err}")
        if not self._is_enabled(sts):
            self.gateway.enable(int(axis))
            time.sleep(0.15)

    def _wait_in_position(self, axis: int, target_abs: float, *, pos_tol: float, timeout_s: float) -> bool:
        if self._legacy_flow is not None:
            return bool(self._legacy_flow._wait_in_position(int(axis), float(target_abs), float(pos_tol), float(timeout_s)))

        t0 = time.time()
        while (time.time() - t0) < float(timeout_s):
            self._raise_if_stop_requested()
            snapshot = self.gateway.get_axis_copy(int(axis))
            sts = int(getattr(snapshot, "sts", 0) or 0)
            err = int(getattr(snapshot, "err", 0) or 0)
            if self._is_fault(sts, err):
                raise RuntimeError(f"AX{axis} fault, err={err}")
            pos_err = abs(float(getattr(snapshot, "act_pos", 0.0) or 0.0) - float(target_abs))
            if pos_err <= float(pos_tol) and (not self._is_moving(sts)):
                return True
            time.sleep(0.08)
        return False

    def _is_fault(self, sts: int, err: int) -> bool:
        if self._legacy_flow is not None:
            return bool(self._legacy_flow._is_fault(int(sts), int(err)))
        return int(err) != 0

    def _is_enabled(self, sts: int) -> bool:
        if self._legacy_flow is not None:
            return bool(self._legacy_flow._is_enabled(int(sts)))
        return int(sts) != 0

    def _is_moving(self, sts: int) -> bool:
        if self._legacy_flow is not None:
            return bool(self._legacy_flow._is_moving(int(sts)))
        return False

    def _require_axis_cal(self) -> AxisCal:
        app = self._runtime_app
        if app is None:
            raise RuntimeError("Legacy runtime host is required for axis calibration")
        axis_cal = getattr(app, "axis_cal", None)
        if axis_cal is None:
            raise RuntimeError("AxisCal is not available")
        return axis_cal

    def _require_legacy_flow(self) -> AutoFlow:
        if self._legacy_flow is None:
            raise RuntimeError("Legacy AutoFlow helpers are not available")
        self._legacy_flow._current_recipe = self.recipe
        self._legacy_flow._calibration_snapshot = self.calibration
        return self._legacy_flow

    def _get_ax2_keepout_ref_abs(self) -> float:
        current_ax2_abs = float(getattr(self.gateway.get_axis_copy(2), "act_pos", 0.0) or 0.0)
        return resolve_ax2_keepout_reference_abs(
            self.recipe,
            current_ax2_abs=current_ax2_abs,
        )

    def _soft_limits_from_axis(self, axis: int) -> tuple[float, float]:
        snapshot = self.gateway.get_axis_copy(int(axis))
        return (
            float(getattr(snapshot, "softlim_pos", 0.0) or 0.0),
            float(getattr(snapshot, "softlim_neg", 0.0) or 0.0),
        )

    def _write_y_point(self, point: int, value: int) -> None:
        app = self._runtime_app
        if app is None:
            raise RuntimeError("Legacy runtime host is required for clamp outputs")
        app.plc_write_y_point(int(point), int(value))

    def _emit_progress(
        self,
        *,
        section_index: int,
        section_total: int,
        z_pos_mm: float,
        ax0_abs: float,
    ) -> None:
        if self.production_workflow is not None:
            self.production_workflow.record_progress(
                section_index=section_index,
                section_total=section_total,
                z_pos_mm=z_pos_mm,
                ax0_abs=ax0_abs,
            )
        self.event_sink.publish_progress(
            section_index=section_index,
            section_total=section_total,
            z_pos_mm=z_pos_mm,
            ax0_abs=ax0_abs,
        )

    def _emit_state(self, state: str, message: str) -> None:
        if self.production_workflow is not None:
            self.production_workflow.record_state(state, message)
        self.event_sink.publish_state(state, message)

    def _raise_if_stop_requested(self) -> None:
        if self._stop_event.is_set():
            raise _StopRequested("User stopped")
        app = self._runtime_app
        if app is None:
            return
        try:
            if int(app.get_x_point(0)) == 0:
                self._stop_event.set()
                raise _StopRequested("E-stop triggered")
        except _StopRequested:
            raise
        except Exception:
            return

    def _set_internal_state(self, state: str) -> None:
        with self._state_lock:
            self.state = state


__all__ = ["AutoFlowOrchestrator", "measure_current_position_od_avg"]
