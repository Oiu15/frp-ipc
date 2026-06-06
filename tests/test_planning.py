import pytest
from dataclasses import replace

from core.models import AxisCal, Recipe
from domain.planning import (
    build_recipe_section_plan,
    format_current_measure_section_name,
    format_recipe_section_name,
    plan_section_positions,
    rebuild_recipe_section_plan,
    require_ax2_rotate_target_abs,
    resolve_measured_section,
    resolve_ax2_position_plan,
    resolve_recipe_section,
    resolve_section_targets,
    resolve_standby_plan,
    resolve_start_anchor_plan,
    section_plan_from_snapshot,
    section_plan_snapshot_from_plan,
)


# ---------------------------------------------------------------------------
# plan_section_positions — table-driven
# ---------------------------------------------------------------------------

_PLAN_POSITIONS_CASES = [
    # (recipe_kwargs, raises, explicit_expected)
    (
        {"section_count": 3, "section_pos_z": [10.0, 20.0, 30.0]},
        None,
        (10.0, 20.0, 30.0),
    ),
    (
        {
            "section_count": 3,
            "section_pos_z": [10.0, 20.0],
            "meas_total_len_mm": 300.0,
            "margin_head_mm": 10.0,
            "margin_tail_mm": 20.0,
        },
        None,
        None,  # computed from recipe defaults
    ),
    (
        {"section_count": 2, "section_pos_z": [10.0, float("nan")]},
        ValueError,
        None,
    ),
]


@pytest.mark.parametrize(
    ("recipe_kwargs", "raises", "explicit_expected"), _PLAN_POSITIONS_CASES
)
def test_plan_section_positions(
    recipe_kwargs: dict,
    raises: type[Exception] | None,
    explicit_expected: tuple[float, ...] | None,
) -> None:
    recipe = Recipe(**recipe_kwargs)

    if raises is not None:
        with pytest.raises(raises):
            plan_section_positions(recipe)
        return

    plan = plan_section_positions(recipe)
    expected = explicit_expected if explicit_expected is not None else tuple(
        recipe.compute_default_positions_z()
    )
    assert plan.positions_z == expected


class TestPlanning:

    def test_resolve_recipe_section_uses_recipe_index_and_position(self) -> None:
        recipe = Recipe(section_count=3, section_pos_z=[10.0, 20.0, 30.0])

        resolved = resolve_recipe_section(recipe, section_index=2)

        assert resolved.measure_section_index == 2
        assert resolved.measure_section_name == '2: 20.000'
        assert resolved.measured_z_pos_mm == 20.0

    def test_resolve_measured_section_falls_back_to_current_position(self) -> None:
        recipe = Recipe(section_count=3, section_pos_z=[10.0, 20.0, 30.0])

        resolved = resolve_measured_section(recipe, measured_z_pos_mm=12.5)

        assert resolved.measure_section_index is None
        assert resolved.measure_section_name == 'current: 12.500'
        assert resolved.measured_z_pos_mm == 12.5

    def test_section_name_formatters_use_shared_label_style(self) -> None:
        assert format_recipe_section_name(3, 45.6789) == '3: 45.679'
        assert format_current_measure_section_name(12.0) == 'current: 12.000'

    def test_resolve_start_anchor_plan_requires_finite_target(self) -> None:
        with pytest.raises(ValueError):
            resolve_start_anchor_plan(Recipe(start_valid=True, start_ax0_abs=float('inf')))

    def test_resolve_standby_plan_returns_expected_targets(self) -> None:
        recipe = Recipe(
            standby_valid=True,
            standby_ax0_abs=100.0,
            standby_ax1_abs=200.0,
            standby_ax4_abs=300.0,
        )

        plan = resolve_standby_plan(recipe)

        assert plan.enabled
        assert plan.targets_abs == {1: 200.0, 4: 300.0, 0: 100.0}

    def test_require_ax2_rotate_target_abs_raises_when_missing(self) -> None:
        with pytest.raises(ValueError):
            require_ax2_rotate_target_abs(Recipe(ax2_rot_valid=False))

    def test_resolve_ax2_position_plan_returns_saved_targets(self) -> None:
        recipe = Recipe(
            ax2_len_valid=True,
            ax2_len_abs=12.5,
            ax2_rot_valid=True,
            ax2_rot_abs=34.5,
        )

        plan = resolve_ax2_position_plan(recipe, current_ax2_abs=99.0)

        assert plan.length_target_abs == 12.5
        assert plan.rotate_target_abs == 34.5

    def test_resolve_section_targets_returns_soft_limit_safe_linear_targets(self) -> None:
        axis_cal = AxisCal(
            sign=1,
            off_ax0=0.0,
            off_ax1=0.0,
            off_ax2=0.0,
            off_ax4=0.0,
            b14=3.0,
            b2=10.0,
            keepout_w=5.0,
            z_pos=0.0,
        )

        targets = resolve_section_targets(
            axis_cal,
            0.0,
            ax2_abs=20.0,
            soft_limits_abs={
                0: (100.0, -100.0),
                1: (100.0, -100.0),
                4: (100.0, -100.0),
            },
        )

        assert targets.ax0_abs == pytest.approx(0.0)
        assert targets.ax1_abs == pytest.approx(1.5)
        assert targets.ax4_abs == pytest.approx(1.5)
        assert targets.z_id_disp == pytest.approx(3.0)
        assert set(targets.linear_targets().keys()) == {0, 1, 4}

    def test_build_recipe_section_plan_reuses_section_target_resolution(self) -> None:
        axis_cal = AxisCal(
            sign=1,
            off_ax0=0.0,
            off_ax1=0.0,
            off_ax2=0.0,
            off_ax4=0.0,
            b14=3.0,
            b2=10.0,
            keepout_w=5.0,
            z_pos=0.0,
        )
        recipe = Recipe(section_count=2, section_pos_z=[0.0, 12.5])
        soft_limits = {
            0: (100.0, -100.0),
            1: (100.0, -100.0),
            4: (100.0, -100.0),
        }

        plan = build_recipe_section_plan(
            recipe,
            axis_cal,
            ax2_abs=20.0,
            soft_limits_abs=soft_limits,
        )

        assert plan.positions_z == (0.0, 12.5)
        assert len(plan.sections) == 2
        assert plan.section_at(2).section_index == 2
        assert plan.section_at(1).ax0_abs == pytest.approx(0.0)

        second_targets = resolve_section_targets(
            axis_cal,
            12.5,
            ax2_abs=20.0,
            soft_limits_abs=soft_limits,
        )
        second_row = plan.section_at(2)
        assert second_row.z_od_disp == pytest.approx(12.5)
        assert second_row.z_id_disp == pytest.approx(second_targets.z_id_disp)
        assert second_row.ax0_abs == pytest.approx(second_targets.ax0_abs)
        assert second_row.ax1_abs == pytest.approx(second_targets.ax1_abs)
        assert second_row.ax4_abs == pytest.approx(second_targets.ax4_abs)

    def test_section_plan_snapshot_round_trips_sources_and_targets(self) -> None:
        axis_cal = AxisCal(b14=3.0, b2=10.0, keepout_w=5.0)
        recipe = Recipe(section_count=2, section_pos_z=[0.0, 12.5])
        plan = build_recipe_section_plan(
            recipe,
            axis_cal,
            ax2_abs=20.0,
            soft_limits_abs={0: (100.0, -100.0), 1: (100.0, -100.0), 4: (100.0, -100.0)},
        )
        plan = plan.__class__(
            positions_z=plan.positions_z,
            sections=(plan.sections[0], replace(plan.sections[1], source="taught")),
        )

        snapshot = section_plan_snapshot_from_plan(plan)
        restored = section_plan_from_snapshot(snapshot)

        assert snapshot.positions_z == [0.0, 12.5]
        assert restored.positions_z == plan.positions_z
        assert restored.section_at(2).source == "taught"
        assert restored.section_at(1).ax0_abs == pytest.approx(plan.section_at(1).ax0_abs)

    def test_rebuild_recipe_section_plan_preserves_taught_rows_when_requested(self) -> None:
        axis_cal = AxisCal(b14=3.0, b2=10.0, keepout_w=5.0)
        recipe = Recipe(section_count=2, meas_total_len_mm=100.0, section_pos_z=[9.0, 19.0])
        previous = build_recipe_section_plan(
            recipe,
            axis_cal,
            ax2_abs=20.0,
            soft_limits_abs={0: (100.0, -100.0), 1: (100.0, -100.0), 4: (100.0, -100.0)},
        )
        previous = previous.__class__(
            positions_z=previous.positions_z,
            sections=(replace(previous.sections[0], source="taught"), previous.sections[1]),
        )
        snapshot = section_plan_snapshot_from_plan(previous)

        rebuilt = rebuild_recipe_section_plan(
            Recipe(section_count=2, meas_total_len_mm=200.0, margin_head_mm=10.0, margin_tail_mm=10.0),
            axis_cal,
            ax2_abs=20.0,
            soft_limits_abs={0: (100.0, -100.0), 1: (100.0, -100.0), 4: (100.0, -100.0)},
            previous_snapshot=snapshot,
            preserve_taught=True,
        )

        assert rebuilt.section_at(1).source == "taught"
        assert rebuilt.section_at(1).z_od_disp == pytest.approx(9.0)
        assert rebuilt.section_at(2).source == "computed"
        assert rebuilt.section_at(2).z_od_disp != pytest.approx(19.0)

    def test_rebuild_recipe_section_plan_recomputes_all_rows_when_not_preserving(self) -> None:
        axis_cal = AxisCal(b14=3.0, b2=10.0, keepout_w=5.0)
        old_recipe = Recipe(section_count=2, section_pos_z=[9.0, 19.0])
        previous = build_recipe_section_plan(
            old_recipe,
            axis_cal,
            ax2_abs=20.0,
            soft_limits_abs={0: (100.0, -100.0), 1: (100.0, -100.0), 4: (100.0, -100.0)},
        )
        previous = previous.__class__(
            positions_z=previous.positions_z,
            sections=(replace(previous.sections[0], source="taught"), previous.sections[1]),
        )

        rebuilt = rebuild_recipe_section_plan(
            Recipe(section_count=2, meas_total_len_mm=200.0, margin_head_mm=10.0, margin_tail_mm=10.0),
            axis_cal,
            ax2_abs=20.0,
            soft_limits_abs={0: (100.0, -100.0), 1: (100.0, -100.0), 4: (100.0, -100.0)},
            previous_snapshot=section_plan_snapshot_from_plan(previous),
            preserve_taught=False,
        )

        assert [row.source for row in rebuilt.sections] == ["computed", "computed"]
        assert rebuilt.section_at(1).z_od_disp != pytest.approx(9.0)
