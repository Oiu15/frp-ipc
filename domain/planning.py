from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

from core.models import AxisCal, Recipe, SectionPlanSnapshot, SectionTargetSnapshot

SoftLimitsAbs = Mapping[int, tuple[float, float]]


@dataclass(frozen=True, slots=True)
class SectionPlan:
    positions_z: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RecipeSectionPlanRow:
    section_index: int
    z_od_disp: float
    z_id_disp: float
    ax0_abs: float
    ax1_abs: float
    ax4_abs: float
    source: str = "computed"

    def linear_targets(self) -> dict[int, float]:
        return {
            1: float(self.ax1_abs),
            4: float(self.ax4_abs),
            0: float(self.ax0_abs),
        }


@dataclass(frozen=True, slots=True)
class RecipeSectionPlan:
    positions_z: tuple[float, ...]
    sections: tuple[RecipeSectionPlanRow, ...]

    def section_for_recipe_index(self, recipe_index: int) -> RecipeSectionPlanRow:
        index = int(recipe_index)
        if index < 0 or index >= len(self.sections):
            raise ValueError(f"recipe_index must be between 0 and {len(self.sections) - 1}")
        return self.sections[index]

    def section_at(self, section_index: int) -> RecipeSectionPlanRow:
        index = int(section_index)
        if index < 1 or index > len(self.sections):
            raise ValueError(f"section_index must be between 1 and {len(self.sections)}")
        return self.sections[index - 1]


@dataclass(frozen=True, slots=True)
class MeasuredSection:
    measure_section_index: int | None
    measure_section_name: str
    measured_z_pos_mm: float


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


def format_recipe_section_name(section_index: int, z_pos_mm: float) -> str:
    return f"{int(section_index)}: {float(z_pos_mm):.3f}"


def format_current_measure_section_name(z_pos_mm: float) -> str:
    return f"current: {float(z_pos_mm):.3f}"


def resolve_recipe_section(recipe: Recipe, *, section_index: int) -> MeasuredSection:
    positions = plan_section_positions(recipe).positions_z
    index = int(section_index)
    if index < 1 or index > len(positions):
        raise ValueError(f"section_index must be between 1 and {len(positions)}")
    z_pos_mm = float(positions[index - 1])
    return MeasuredSection(
        measure_section_index=index,
        measure_section_name=format_recipe_section_name(index, z_pos_mm),
        measured_z_pos_mm=z_pos_mm,
    )


def resolve_measured_section(
    recipe: Recipe,
    *,
    measured_z_pos_mm: float,
    measure_section_index: int | None = None,
) -> MeasuredSection:
    z_pos_mm = _required_float('measured_z_pos_mm', measured_z_pos_mm)
    if measure_section_index is not None:
        resolved = resolve_recipe_section(recipe, section_index=int(measure_section_index))
        return MeasuredSection(
            measure_section_index=resolved.measure_section_index,
            measure_section_name=resolved.measure_section_name,
            measured_z_pos_mm=z_pos_mm,
        )
    return MeasuredSection(
        measure_section_index=None,
        measure_section_name=format_current_measure_section_name(z_pos_mm),
        measured_z_pos_mm=z_pos_mm,
    )


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


def build_recipe_section_plan(
    recipe: Recipe,
    axis_cal: AxisCal,
    *,
    ax2_abs: float,
    soft_limits_abs: SoftLimitsAbs | None = None,
) -> RecipeSectionPlan:
    positions = plan_section_positions(recipe).positions_z
    sections = tuple(
        _build_recipe_section_plan_row(
            axis_cal,
            section_index=index,
            z_pos_mm=z_pos_mm,
            ax2_abs=ax2_abs,
            soft_limits_abs=soft_limits_abs,
        )
        for index, z_pos_mm in enumerate(positions, start=1)
    )
    return RecipeSectionPlan(positions_z=positions, sections=sections)


def section_plan_snapshot_from_plan(plan: RecipeSectionPlan) -> SectionPlanSnapshot:
    return SectionPlanSnapshot(
        sections=[
            SectionTargetSnapshot(
                section_index=int(row.section_index),
                z_od_disp=float(row.z_od_disp),
                z_id_disp=float(row.z_id_disp),
                ax0_abs=float(row.ax0_abs),
                ax1_abs=float(row.ax1_abs),
                ax4_abs=float(row.ax4_abs),
                source=str(getattr(row, 'source', 'computed') or 'computed'),
            )
            for row in plan.sections
        ]
    )


def section_plan_from_snapshot(snapshot: SectionPlanSnapshot) -> RecipeSectionPlan:
    rows = tuple(
        RecipeSectionPlanRow(
            section_index=int(section.section_index),
            z_od_disp=float(section.z_od_disp),
            z_id_disp=float(section.z_id_disp),
            ax0_abs=float(section.ax0_abs),
            ax1_abs=float(section.ax1_abs),
            ax4_abs=float(section.ax4_abs),
            source=str(section.source),
        )
        for section in snapshot.sections
    )
    return RecipeSectionPlan(
        positions_z=tuple(float(row.z_od_disp) for row in rows),
        sections=rows,
    )


def section_plan_is_compatible(recipe: Recipe, snapshot: SectionPlanSnapshot | None) -> bool:
    if snapshot is None:
        return False
    expected = max(0, int(getattr(recipe, 'section_count', 0) or 0))
    if expected <= 0 or len(snapshot.sections) != expected:
        return False
    expected_indexes = list(range(1, expected + 1))
    actual_indexes = [int(section.section_index) for section in snapshot.sections]
    return actual_indexes == expected_indexes


def rebuild_recipe_section_plan(
    recipe: Recipe,
    axis_cal: AxisCal,
    *,
    ax2_abs: float,
    soft_limits_abs: SoftLimitsAbs | None = None,
    previous_snapshot: SectionPlanSnapshot | None = None,
    preserve_taught: bool = False,
) -> RecipeSectionPlan:
    computed_positions = list(recipe.compute_default_positions_z())
    compute_recipe = replace(recipe, section_pos_z=computed_positions, section_pos_ui=list(computed_positions), section_plan=None)
    computed = build_recipe_section_plan(
        compute_recipe,
        axis_cal,
        ax2_abs=ax2_abs,
        soft_limits_abs=soft_limits_abs,
    )
    if not preserve_taught or not section_plan_is_compatible(recipe, previous_snapshot):
        return computed

    assert previous_snapshot is not None
    taught_by_index = {
        int(section.section_index): section
        for section in previous_snapshot.sections
        if str(section.source).lower() == "taught"
    }
    merged_rows: list[RecipeSectionPlanRow] = []
    for row in computed.sections:
        taught = taught_by_index.get(int(row.section_index))
        if taught is None:
            merged_rows.append(row)
            continue
        merged_rows.append(
            RecipeSectionPlanRow(
                section_index=int(taught.section_index),
                z_od_disp=float(taught.z_od_disp),
                z_id_disp=float(taught.z_id_disp),
                ax0_abs=float(taught.ax0_abs),
                ax1_abs=float(taught.ax1_abs),
                ax4_abs=float(taught.ax4_abs),
                source="taught",
            )
        )
    return RecipeSectionPlan(
        positions_z=tuple(float(row.z_od_disp) for row in merged_rows),
        sections=tuple(merged_rows),
    )


def _optional_target(enabled: bool, value: float | int | None) -> float | None:
    if not enabled:
        return None
    return _required_float('target', value)


def _required_float(name: str, value: float | int | None) -> float:
    if value is None:
        raise ValueError(f'{name} is not a valid float: {value!r}')
    try:
        fv = float(value)
    except Exception as exc:  # pragma: no cover - defensive conversion guard
        raise ValueError(f'{name} is not a valid float: {value!r}') from exc
    if not math.isfinite(fv):
        raise ValueError(f'{name} must be finite: {value!r}')
    return float(fv)


def _build_recipe_section_plan_row(
    axis_cal: AxisCal,
    *,
    section_index: int,
    z_pos_mm: float,
    ax2_abs: float,
    soft_limits_abs: SoftLimitsAbs | None = None,
) -> RecipeSectionPlanRow:
    targets = resolve_section_targets(
        axis_cal,
        float(z_pos_mm),
        ax2_abs=ax2_abs,
        soft_limits_abs=soft_limits_abs,
    )
    return RecipeSectionPlanRow(
        section_index=int(section_index),
        z_od_disp=float(z_pos_mm),
        z_id_disp=float(targets.z_id_disp),
        ax0_abs=float(targets.ax0_abs),
        ax1_abs=float(targets.ax1_abs),
        ax4_abs=float(targets.ax4_abs),
        source="computed",
    )


def _validate_positions(values: tuple[float, ...]) -> None:
    if not values:
        return
    for idx, value in enumerate(values, start=1):
        _required_float(f'section_pos_z[{idx}]', value)
