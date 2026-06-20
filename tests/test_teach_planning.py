from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from core.models import AxisCal, Recipe
from domain.teach_planning import (
    TeachWarning,
    align_by_id_target,
    align_by_od_targets,
    center_position_z_disp,
    end_z_disp,
    relative_move_targets,
    save_section_plan,
    selected_section_targets,
    standby_alignment_plan,
    start_anchor_z_pos,
)


ROOT = Path(__file__).resolve().parents[1]


class _Row:
    z_od_disp = 12.0
    ax0_abs = 1.0
    ax1_abs = 2.0
    ax4_abs = 3.0


class TestTeachPlanning:
    def test_selected_section_targets_support_ax2_and_linear_modes(self) -> None:
        axis_cal = AxisCal(sign=1)

        center = selected_section_targets(axis_cal, mode=3, selected_row=_Row())
        assert center.axis_items() == ((2, 12.0),)

        linear = selected_section_targets(axis_cal, mode=2, selected_row=_Row())
        assert linear.axis_items() == ((0, 1.0), (1, 2.0), (4, 3.0))

    def test_save_section_plan_returns_recipe_write_intent_without_mutating_inputs(self) -> None:
        axis_cal = AxisCal(sign=1, b14=3.0)
        before = replace(axis_cal)

        od = save_section_plan(axis_cal, mode=0, ax0_abs=10.0, ax1_abs=13.0, ax2_abs=5.0, ax4_abs=0.0)
        center = save_section_plan(axis_cal, mode=3, ax0_abs=10.0, ax1_abs=13.0, ax2_abs=5.0, ax4_abs=0.0)
        mismatched = save_section_plan(axis_cal, mode=2, ax0_abs=10.0, ax1_abs=1.0, ax2_abs=5.0, ax4_abs=1.0)

        assert od.z_od_disp == pytest.approx(10.0)
        assert center.z_od_disp == pytest.approx(5.0)
        assert TeachWarning.OD_ID_NOT_ALIGNED in mismatched.warnings
        assert axis_cal == before

    def test_alignment_targets_match_existing_od_id_semantics(self) -> None:
        axis_cal = AxisCal(sign=1, b14=3.0)

        od_targets = align_by_od_targets(
            axis_cal,
            ax0_abs=10.0,
            ax1_softlim_pos=5.0,
            ax1_softlim_neg=-100.0,
        )
        assert od_targets.ax1_abs == pytest.approx(5.0)
        assert od_targets.ax4_abs == pytest.approx(8.0)

        ax0_target = align_by_id_target(axis_cal, ax1_abs=4.0, ax4_abs=5.0)
        assert ax0_target == pytest.approx(6.0)

    def test_start_end_standby_and_center_helpers_are_read_only(self) -> None:
        axis_cal = AxisCal(sign=1, b14=3.0)
        recipe = Recipe(
            start_valid=True,
            start_ax0_abs=7.0,
            meas_total_len_mm=0.0,
            pipe_len_mm=100.0,
            clamp_occupy_mm=25.0,
        )
        recipe_before = replace(recipe)
        axis_before = replace(axis_cal)

        assert start_anchor_z_pos(axis_cal, recipe) == pytest.approx(7.0)
        assert end_z_disp(recipe) == pytest.approx(75.0)
        assert center_position_z_disp(axis_cal, ax2_abs=9.0) == pytest.approx(9.0)

        standby = standby_alignment_plan(axis_cal, ax0_abs=10.0, ax1_abs=6.0, ax4_abs=7.0)
        assert standby.aligned
        assert standby.delta == pytest.approx(0.0)
        assert recipe == recipe_before
        assert axis_cal == axis_before

    def test_relative_move_targets_split_id_motion_with_soft_limit_overflow(self) -> None:
        axis_cal = AxisCal(sign=1)

        targets = relative_move_targets(
            axis_cal,
            mode=2,
            dz=10.0,
            ax0_abs=0.0,
            ax1_abs=0.0,
            ax2_abs=0.0,
            ax4_abs=0.0,
            ax1_softlim_pos=4.0,
            ax1_softlim_neg=-100.0,
            ax4_softlim_pos=100.0,
            ax4_softlim_neg=-100.0,
        )

        assert targets.ax0_abs == pytest.approx(10.0)
        assert targets.ax1_abs == pytest.approx(4.0)
        assert targets.ax4_abs == pytest.approx(6.0)


def test_teach_planning_stays_out_of_ui_application_and_io_layers() -> None:
    source = (ROOT / "domain" / "teach_planning.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned_roots = {"application", "ui", "tkinter", "drivers", "repositories", "services"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in banned_roots
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in banned_roots
    assert "messagebox" not in source
    assert "movea_abs" not in source
