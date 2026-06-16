from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from collections.abc import Iterable

from core.models import AxisCal


class LengthPlanStatus(Enum):
    OK = "ok"
    DISABLED = "disabled"
    LOW_APPROACH_OUT_OF_TRAVEL = "low_approach_out_of_travel"
    LOW_SEARCH_OUT_OF_TRAVEL = "low_search_out_of_travel"
    INSUFFICIENT_TRAVEL = "insufficient_travel"
    PIPE_TOO_LONG = "pipe_too_long"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True, slots=True)
class SoftLimitPlan:
    abs_min: float
    abs_max: float
    used_fallback: bool


@dataclass(frozen=True, slots=True)
class LengthRange:
    z_min: float
    z_max: float
    travel: float


@dataclass(frozen=True, slots=True)
class LengthSetupPlan:
    status: LengthPlanStatus
    lmax: float


@dataclass(frozen=True, slots=True)
class TopEdgeApproachPlan:
    z_approach: float
    clamped: bool


def normalize_soft_limits_abs(
    softlim_pos: float | int | None,
    softlim_neg: float | int | None,
    *,
    fallback_min: float,
    fallback_max: float,
) -> SoftLimitPlan:
    try:
        p = _finite_float("softlim_pos", softlim_pos)
        n = _finite_float("softlim_neg", softlim_neg)
        if abs(p) < 1e-6 and abs(n) < 1e-6:
            raise ValueError
        if abs(p - n) < 1e-6:
            raise ValueError
        return SoftLimitPlan(abs_min=min(p, n), abs_max=max(p, n), used_fallback=False)
    except Exception:
        lo = _finite_float("fallback_min", fallback_min)
        hi = _finite_float("fallback_max", fallback_max)
        return SoftLimitPlan(abs_min=min(lo, hi), abs_max=max(lo, hi), used_fallback=True)


def z_disp_range(axis_cal: AxisCal, *, abs_min: float, abs_max: float) -> LengthRange:
    z1 = _finite_float("abs_min_z", axis_cal.abs_to_z_disp(0, _finite_float("abs_min", abs_min)))
    z2 = _finite_float("abs_max_z", axis_cal.abs_to_z_disp(0, _finite_float("abs_max", abs_max)))
    z_min = min(z1, z2)
    z_max = max(z1, z2)
    return LengthRange(z_min=z_min, z_max=z_max, travel=max(0.0, z_max - z_min))


def plan_length_setup(
    *,
    enabled: bool,
    z_range: LengthRange,
    z_low_approach: float,
    low_search_dist: float,
    high_search_dist: float,
    high_margin: float,
    pipe_len: float,
) -> LengthSetupPlan:
    try:
        z_low = _finite_float("z_low_approach", z_low_approach)
        d_low = _finite_float("low_search_dist", low_search_dist)
        d_high = _finite_float("high_search_dist", high_search_dist)
        margin = _finite_float("high_margin", high_margin)
        pipe = _finite_float("pipe_len", pipe_len)
        z_min = _finite_float("z_min", z_range.z_min)
        z_max = _finite_float("z_max", z_range.z_max)
    except ValueError:
        return LengthSetupPlan(status=LengthPlanStatus.INVALID_INPUT, lmax=0.0)

    z_low_edge_max = min(z_max, z_low + d_low)
    lmax = max(0.0, z_low_edge_max + margin - d_high - z_min)

    if not enabled:
        return LengthSetupPlan(status=LengthPlanStatus.DISABLED, lmax=lmax)
    if not (z_min <= z_low <= z_max):
        return LengthSetupPlan(status=LengthPlanStatus.LOW_APPROACH_OUT_OF_TRAVEL, lmax=lmax)
    if z_low + d_low > z_max + 1e-6:
        return LengthSetupPlan(status=LengthPlanStatus.LOW_SEARCH_OUT_OF_TRAVEL, lmax=lmax)
    if lmax <= 1.0:
        return LengthSetupPlan(status=LengthPlanStatus.INSUFFICIENT_TRAVEL, lmax=lmax)
    if pipe > lmax + 1e-6:
        return LengthSetupPlan(status=LengthPlanStatus.PIPE_TOO_LONG, lmax=lmax)
    return LengthSetupPlan(status=LengthPlanStatus.OK, lmax=lmax)


def length_from_edges(z_low: float, z_high: float) -> float | None:
    try:
        length = _finite_float("z_low", z_low) - _finite_float("z_high", z_high)
    except ValueError:
        return None
    if length <= 0.0 or not math.isfinite(length):
        return None
    return float(length)


def average_edge_pair(edge1: float, edge2: float) -> float:
    return 0.5 * (_finite_float("edge1", edge1) + _finite_float("edge2", edge2))


def plan_top_edge_approach(
    *,
    z_low_edge: float,
    pipe_len: float,
    high_margin: float,
    z_range: LengthRange,
) -> TopEdgeApproachPlan:
    z_appr = (
        _finite_float("z_low_edge", z_low_edge)
        - _finite_float("pipe_len", pipe_len)
        + _finite_float("high_margin", high_margin)
    )
    z_clamped = clamp_z(z_appr, z_range)
    return TopEdgeApproachPlan(z_approach=z_clamped, clamped=abs(z_clamped - z_appr) > 1e-6)


def clamp_z(value: float, z_range: LengthRange) -> float:
    z = _finite_float("value", value)
    return max(float(z_range.z_min), min(float(z_range.z_max), z))


def reject_outliers_sigma(values: Iterable[float], *, sigma: float) -> tuple[float, ...]:
    finite_values = tuple(float(v) for v in values if math.isfinite(float(v)))
    if len(finite_values) < 2:
        return finite_values
    sigma_f = _finite_float("sigma", sigma)
    if sigma_f <= 0.0:
        return finite_values
    mean = sum(finite_values) / len(finite_values)
    variance = sum((v - mean) ** 2 for v in finite_values) / len(finite_values)
    std = math.sqrt(variance)
    if std <= 1e-12:
        return finite_values
    limit = sigma_f * std
    return tuple(v for v in finite_values if abs(v - mean) <= limit)


def _finite_float(name: str, value: float | int | None) -> float:
    if value is None:
        raise ValueError(f"{name} is not a valid float: {value!r}")
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{name} is not a valid float: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite: {value!r}")
    return result
