from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from core.models import AxisCal, Recipe

SoftLimitsAbs = Mapping[int, tuple[float, float]]


@dataclass(frozen=True, slots=True)
class SectionPlan:
    positions_z: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class StartAnchorPlan:
    ax0_abs: float | None

    @property
    def enabled(self) -> bool:
        return self.ax0_abs is not None


@dataclass(frozen=True, slots=True)
class StandbyPlan:
    targets_abs: dict[int, float]

    @property
    def enabled(self) -> bool:
        return bool(self.targets_abs)


@dataclass(frozen=True, slots=True)
class Ax2PositionPlan:
    length_target_abs: float | None
    rotate_target_abs: float | None
    keepout_reference_abs: float

    @property
    def has_length_target(self) -> bool:
        return self.length_target_abs is not None

    @property
    def has_rotate_target(self) -> bool:
        return self.rotate_target_abs is not None


@dataclass(frozen=True, slots=True)
class SectionTargets:
    ax0_abs: float
    ax1_abs: float
    ax4_abs: float
    z_id_disp: float
    z1_disp: float
    z4_disp: float

    def linear_targets(self) -> dict[int, float]:
        return {
            1: float(self.ax1_abs),
            4: float(self.ax4_abs),
            0: float(self.ax0_abs),
        }


def plan_section_positions(recipe: Recipe) -> SectionPlan:
    expected = max(0, int(getattr(recipe, 'section_count', 0) or 0))
    current = tuple(float(v) for v in list(getattr(recipe, 'section_pos_z', []) or []))
    if expected > 0 and len(current) == expected:
        _validate_positions(current)
        return SectionPlan(positions_z=current)
    if expected <= 0:
        return SectionPlan(positions_z=())
    computed = tuple(float(v) for v in recipe.compute_default_positions_z())
    _validate_positions(computed)
    return SectionPlan(positions_z=computed)


def resolve_start_anchor_plan(recipe: Recipe) -> StartAnchorPlan:
    return StartAnchorPlan(ax0_abs=_optional_target(bool(getattr(recipe, 'start_valid', False)), getattr(recipe, 'start_ax0_abs', 0.0)))


def resolve_standby_plan(recipe: Recipe) -> StandbyPlan:
    if not bool(getattr(recipe, 'standby_valid', False)):
        return StandbyPlan(targets_abs={})
    targets = {
        1: _required_float('standby_ax1_abs', getattr(recipe, 'standby_ax1_abs', 0.0)),
        4: _required_float('standby_ax4_abs', getattr(recipe, 'standby_ax4_abs', 0.0)),
        0: _required_float('standby_ax0_abs', getattr(recipe, 'standby_ax0_abs', 0.0)),
    }
    return StandbyPlan(targets_abs=targets)


def resolve_ax2_position_plan(recipe: Recipe, *, current_ax2_abs: float) -> Ax2PositionPlan:
    rotate_target = _optional_target(bool(getattr(recipe, 'ax2_rot_valid', False)), getattr(recipe, 'ax2_rot_abs', 0.0))
    length_target = _optional_target(bool(getattr(recipe, 'ax2_len_valid', False)), getattr(recipe, 'ax2_len_abs', 0.0))
    keepout_ref = float(rotate_target if rotate_target is not None else _required_float('current_ax2_abs', current_ax2_abs))
    return Ax2PositionPlan(
        length_target_abs=length_target,
        rotate_target_abs=rotate_target,
        keepout_reference_abs=keepout_ref,
    )


def resolve_ax2_keepout_reference_abs(recipe: Recipe, *, current_ax2_abs: float) -> float:
    return float(resolve_ax2_position_plan(recipe, current_ax2_abs=current_ax2_abs).keepout_reference_abs)


def require_ax2_rotate_target_abs(recipe: Recipe) -> float:
    rotate_target = _optional_target(bool(getattr(recipe, 'ax2_rot_valid', False)), getattr(recipe, 'ax2_rot_abs', 0.0))
    if rotate_target is None:
        raise ValueError('AX2 rotate position is not saved')
    return float(rotate_target)


def resolve_section_targets(
    axis_cal: AxisCal,
    z_pos_mm: float,
    *,
    ax2_abs: float,
    soft_limits_abs: SoftLimitsAbs | None = None,
) -> SectionTargets:
    resolved = axis_cal.od_z_disp_to_targets(
        float(_required_float('z_pos_mm', z_pos_mm)),
        ax2_abs=float(_required_float('ax2_abs', ax2_abs)),
        softlims_abs=dict(soft_limits_abs or {}),
    )
    return SectionTargets(
        ax0_abs=_required_float('ax0_abs', resolved['ax0_abs']),
        ax1_abs=_required_float('ax1_abs', resolved['ax1_abs']),
        ax4_abs=_required_float('ax4_abs', resolved['ax4_abs']),
        z_id_disp=_required_float('z_id_disp', resolved['z_id_disp']),
        z1_disp=_required_float('z1_disp', resolved['z1_disp']),
        z4_disp=_required_float('z4_disp', resolved['z4_disp']),
    )


def _optional_target(enabled: bool, value: float | int | None) -> float | None:
    if not enabled:
        return None
    return _required_float('target', value)


def _required_float(name: str, value: float | int | None) -> float:
    try:
        fv = float(value)
    except Exception as exc:  # pragma: no cover - defensive conversion guard
        raise ValueError(f'{name} is not a valid float: {value!r}') from exc
    if not math.isfinite(fv):
        raise ValueError(f'{name} must be finite: {value!r}')
    return float(fv)


def _validate_positions(values: tuple[float, ...]) -> None:
    if not values:
        return
    for idx, value in enumerate(values, start=1):
        _required_float(f'section_pos_z[{idx}]', value)
