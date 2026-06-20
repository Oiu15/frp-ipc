from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Protocol

from core.models import AxisCal, Recipe


class TeachWarning(Enum):
    OD_ID_NOT_ALIGNED = "od_id_not_aligned"


@dataclass(frozen=True, slots=True)
class TeachTargets:
    ax0_abs: float | None = None
    ax1_abs: float | None = None
    ax2_abs: float | None = None
    ax4_abs: float | None = None

    def axis_items(self) -> tuple[tuple[int, float], ...]:
        items: list[tuple[int, float]] = []
        if self.ax0_abs is not None:
            items.append((0, float(self.ax0_abs)))
        if self.ax1_abs is not None:
            items.append((1, float(self.ax1_abs)))
        if self.ax2_abs is not None:
            items.append((2, float(self.ax2_abs)))
        if self.ax4_abs is not None:
            items.append((4, float(self.ax4_abs)))
        return tuple(items)


@dataclass(frozen=True, slots=True)
class TeachSavePlan:
    z_od_disp: float
    warnings: tuple[TeachWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class StandbyAlignmentPlan:
    z_od_disp: float
    z1_raw: float
    z4_raw: float
    z_id_disp: float
    z_id_expected_disp: float
    delta: float
    aligned: bool


@dataclass(frozen=True, slots=True)
class TeachPositionPlan:
    z0_raw: float
    z1_raw: float
    z2_raw: float
    z4_raw: float
    zid_raw: float
    z0_disp: float
    z2_disp: float
    zid_disp: float
    z_id_expected_disp: float
    delta: float
    aligned: bool


class SectionTeachRow(Protocol):
    z_od_disp: float
    ax0_abs: float
    ax1_abs: float
    ax4_abs: float


def selected_section_targets(
    axis_cal: AxisCal,
    *,
    mode: int,
    selected_row: SectionTeachRow,
) -> TeachTargets:
    normalized = normalize_teach_mode(mode)
    if normalized == 3:
        return TeachTargets(ax2_abs=float(axis_cal.z_disp_to_abs(2, selected_row.z_od_disp)))
    return TeachTargets(
        ax0_abs=float(selected_row.ax0_abs) if normalized in (0, 2) else None,
        ax1_abs=float(selected_row.ax1_abs) if normalized in (1, 2) else None,
        ax4_abs=float(selected_row.ax4_abs) if normalized in (1, 2) else None,
    )


def save_section_plan(
    axis_cal: AxisCal,
    *,
    mode: int,
    ax0_abs: float,
    ax1_abs: float,
    ax2_abs: float,
    ax4_abs: float,
    tolerance: float = 0.50,
) -> TeachSavePlan:
    normalized = normalize_teach_mode(mode)
    if normalized == 3:
        return TeachSavePlan(z_od_disp=float(axis_cal.abs_to_z_disp(2, ax2_abs)))

    z_od_from_od = float(axis_cal.abs_to_z_disp(0, ax0_abs))
    z1_raw = float(axis_cal.abs_to_z_raw(1, ax1_abs))
    z4_raw = float(axis_cal.abs_to_z_raw(4, ax4_abs))
    zid_disp = float(axis_cal.z_raw_to_z_disp(z1_raw + z4_raw))
    z_od_from_id = float(zid_disp) - float(axis_cal.b14)

    if normalized == 0:
        return TeachSavePlan(z_od_disp=z_od_from_od)
    if normalized == 1:
        return TeachSavePlan(z_od_disp=z_od_from_id)
    warnings: tuple[TeachWarning, ...] = ()
    if abs(z_od_from_od - z_od_from_id) > float(tolerance):
        warnings = (TeachWarning.OD_ID_NOT_ALIGNED,)
    return TeachSavePlan(z_od_disp=z_od_from_od, warnings=warnings)


def align_by_od_targets(
    axis_cal: AxisCal,
    *,
    ax0_abs: float,
    ax1_softlim_pos: float | None,
    ax1_softlim_neg: float | None,
) -> TeachTargets:
    z0_raw = float(axis_cal.abs_to_z_raw(0, ax0_abs))
    z_id_raw_tgt = z0_raw + float(axis_cal.b14)
    lo1, hi1 = raw_range_from_soft_limits(
        axis_cal,
        axis=1,
        softlim_pos=ax1_softlim_pos,
        softlim_neg=ax1_softlim_neg,
    )
    z1_raw_tgt = min(max(z_id_raw_tgt, lo1), hi1)
    z4_raw_tgt = z_id_raw_tgt - z1_raw_tgt
    if z4_raw_tgt < 0.0:
        z4_raw_tgt = 0.0
    return TeachTargets(
        ax1_abs=float(axis_cal.z_raw_to_abs(1, z1_raw_tgt)),
        ax4_abs=float(axis_cal.z_raw_to_abs(4, z4_raw_tgt)),
    )


def align_by_id_target(axis_cal: AxisCal, *, ax1_abs: float, ax4_abs: float) -> float:
    z1_raw = float(axis_cal.abs_to_z_raw(1, ax1_abs))
    z4_raw = float(axis_cal.abs_to_z_raw(4, ax4_abs))
    z_od_raw_tgt = z1_raw + z4_raw - float(axis_cal.b14)
    return float(axis_cal.z_raw_to_abs(0, z_od_raw_tgt))


def start_anchor_z_pos(axis_cal: AxisCal, recipe: Recipe) -> float:
    if not bool(getattr(recipe, "start_valid", False)):
        return 0.0
    return float(axis_cal.abs_to_z_raw(0, float(getattr(recipe, "start_ax0_abs", 0.0))))


def end_z_disp(recipe: Recipe) -> float:
    total = float(getattr(recipe, "meas_total_len_mm", 0.0) or 0.0)
    if total <= 1e-6:
        total = float(getattr(recipe, "pipe_len_mm", 0.0) or 0.0) - float(getattr(recipe, "clamp_occupy_mm", 0.0) or 0.0)
    return max(0.0, float(total))


def standby_alignment_plan(
    axis_cal: AxisCal,
    *,
    ax0_abs: float,
    ax1_abs: float,
    ax4_abs: float,
    tolerance: float = 0.50,
) -> StandbyAlignmentPlan:
    z0_disp = float(axis_cal.abs_to_z_disp(0, ax0_abs))
    z1_raw = float(axis_cal.abs_to_z_raw(1, ax1_abs))
    z4_raw = float(axis_cal.abs_to_z_raw(4, ax4_abs))
    zid_disp = float(axis_cal.z_raw_to_z_disp(z1_raw + z4_raw))
    zid_exp = z0_disp + float(axis_cal.b14)
    dz = zid_disp - zid_exp
    return StandbyAlignmentPlan(
        z_od_disp=z0_disp,
        z1_raw=z1_raw,
        z4_raw=z4_raw,
        z_id_disp=zid_disp,
        z_id_expected_disp=zid_exp,
        delta=dz,
        aligned=abs(dz) <= float(tolerance),
    )


def center_position_z_disp(axis_cal: AxisCal, *, ax2_abs: float) -> float:
    return float(axis_cal.abs_to_z_disp(2, ax2_abs))


def relative_move_targets(
    axis_cal: AxisCal,
    *,
    mode: int,
    dz: float,
    ax0_abs: float,
    ax1_abs: float,
    ax2_abs: float,
    ax4_abs: float,
    ax1_softlim_pos: float | None,
    ax1_softlim_neg: float | None,
    ax4_softlim_pos: float | None,
    ax4_softlim_neg: float | None,
) -> TeachTargets:
    normalized = normalize_teach_mode(mode)
    delta = float(dz)
    ax0_target = None
    ax1_target = None
    ax2_target = None
    ax4_target = None

    if normalized == 3:
        z2_disp = float(axis_cal.abs_to_z_disp(2, ax2_abs))
        ax2_target = float(axis_cal.z_disp_to_abs(2, z2_disp + delta))

    if normalized in (0, 2):
        z0_disp = float(axis_cal.abs_to_z_disp(0, ax0_abs))
        ax0_target = float(axis_cal.z_disp_to_abs(0, z0_disp + delta))

    if normalized in (1, 2):
        z1_raw = float(axis_cal.abs_to_z_raw(1, ax1_abs))
        z4_raw = float(axis_cal.abs_to_z_raw(4, ax4_abs))
        zid_raw = z1_raw + z4_raw
        zid_disp = float(axis_cal.z_raw_to_z_disp(zid_raw))
        zid_tgt_raw = float(axis_cal.z_disp_to_z_raw(zid_disp + delta))

        lo1, hi1 = raw_range_from_soft_limits(axis_cal, axis=1, softlim_pos=ax1_softlim_pos, softlim_neg=ax1_softlim_neg)
        lo4, hi4 = raw_range_from_soft_limits(axis_cal, axis=4, softlim_pos=ax4_softlim_pos, softlim_neg=ax4_softlim_neg)

        delta_raw = zid_tgt_raw - zid_raw
        z1_des = z1_raw + 0.5 * delta_raw
        z1_tgt = max(lo1, min(hi1, z1_des))
        used1 = z1_tgt - z1_raw

        rem = delta_raw - used1
        z4_des = z4_raw + rem
        z4_tgt = max(lo4, min(hi4, z4_des))

        ax1_target = float(axis_cal.z_raw_to_abs(1, z1_tgt))
        ax4_target = float(axis_cal.z_raw_to_abs(4, z4_tgt))

    return TeachTargets(ax0_abs=ax0_target, ax1_abs=ax1_target, ax2_abs=ax2_target, ax4_abs=ax4_target)


def teach_position_plan(
    axis_cal: AxisCal,
    *,
    ax0_abs: float,
    ax1_abs: float,
    ax2_abs: float,
    ax4_abs: float,
    tolerance: float = 0.50,
) -> TeachPositionPlan:
    z0_raw = float(axis_cal.abs_to_z_raw(0, ax0_abs))
    z1_raw = float(axis_cal.abs_to_z_raw(1, ax1_abs))
    z2_raw = float(axis_cal.abs_to_z_raw(2, ax2_abs))
    z4_raw = float(axis_cal.abs_to_z_raw(4, ax4_abs))
    zid_raw = z1_raw + z4_raw
    z_id_expect_raw = z0_raw + float(axis_cal.b14)
    delta = zid_raw - z_id_expect_raw
    return TeachPositionPlan(
        z0_raw=z0_raw,
        z1_raw=z1_raw,
        z2_raw=z2_raw,
        z4_raw=z4_raw,
        zid_raw=zid_raw,
        z0_disp=float(axis_cal.z_raw_to_z_disp(z0_raw)),
        z2_disp=float(axis_cal.z_raw_to_z_disp(z2_raw)),
        zid_disp=float(axis_cal.z_raw_to_z_disp(zid_raw)),
        z_id_expected_disp=float(axis_cal.z_raw_to_z_disp(z_id_expect_raw)),
        delta=delta,
        aligned=abs(delta) <= float(tolerance),
    )


def raw_range_from_soft_limits(
    axis_cal: AxisCal,
    *,
    axis: int,
    softlim_pos: float | None,
    softlim_neg: float | None,
) -> tuple[float, float]:
    if softlim_pos is None or softlim_neg is None:
        return (-math.inf, math.inf)
    try:
        p = float(softlim_pos)
        n = float(softlim_neg)
    except Exception:
        return (-math.inf, math.inf)
    if not (math.isfinite(p) and math.isfinite(n)):
        return (-math.inf, math.inf)
    if abs(p) + abs(n) < 1e-6:
        return (-math.inf, math.inf)
    r1 = float(axis_cal.abs_to_z_raw(int(axis), p))
    r2 = float(axis_cal.abs_to_z_raw(int(axis), n))
    return (min(r1, r2), max(r1, r2))


def normalize_teach_mode(mode: int) -> int:
    try:
        value = int(mode)
    except Exception:
        return 2
    return max(0, min(3, value))
