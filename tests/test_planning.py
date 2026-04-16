import unittest

from core.models import AxisCal, Recipe
from domain.planning import (
    format_current_measure_section_name,
    format_recipe_section_name,
    plan_section_positions,
    require_ax2_rotate_target_abs,
    resolve_measured_section,
    resolve_ax2_keepout_reference_abs,
    resolve_ax2_position_plan,
    resolve_recipe_section,
    resolve_section_targets,
    resolve_standby_plan,
    resolve_start_anchor_plan,
)


class PlanningTest(unittest.TestCase):
    def test_plan_section_positions_uses_recipe_positions_when_count_matches(self) -> None:
        recipe = Recipe(section_count=3, section_pos_z=[10.0, 20.0, 30.0])

        plan = plan_section_positions(recipe)

        self.assertEqual(plan.positions_z, (10.0, 20.0, 30.0))

    def test_plan_section_positions_falls_back_to_default_positions(self) -> None:
        recipe = Recipe(
            section_count=3,
            section_pos_z=[10.0, 20.0],
            meas_total_len_mm=300.0,
            margin_head_mm=10.0,
            margin_tail_mm=20.0,
        )

        plan = plan_section_positions(recipe)

        self.assertEqual(plan.positions_z, tuple(recipe.compute_default_positions_z()))

    def test_plan_section_positions_rejects_non_finite_values(self) -> None:
        recipe = Recipe(section_count=2, section_pos_z=[10.0, float('nan')])

        with self.assertRaises(ValueError):
            plan_section_positions(recipe)

    def test_resolve_recipe_section_uses_recipe_index_and_position(self) -> None:
        recipe = Recipe(section_count=3, section_pos_z=[10.0, 20.0, 30.0])

        resolved = resolve_recipe_section(recipe, section_index=2)

        self.assertEqual(resolved.measure_section_index, 2)
        self.assertEqual(resolved.measure_section_name, '2: 20.000')
        self.assertEqual(resolved.measured_z_pos_mm, 20.0)

    def test_resolve_measured_section_falls_back_to_current_position(self) -> None:
        recipe = Recipe(section_count=3, section_pos_z=[10.0, 20.0, 30.0])

        resolved = resolve_measured_section(recipe, measured_z_pos_mm=12.5)

        self.assertIsNone(resolved.measure_section_index)
        self.assertEqual(resolved.measure_section_name, 'current: 12.500')
        self.assertEqual(resolved.measured_z_pos_mm, 12.5)

    def test_section_name_formatters_use_shared_label_style(self) -> None:
        self.assertEqual(format_recipe_section_name(3, 45.6789), '3: 45.679')
        self.assertEqual(format_current_measure_section_name(12.0), 'current: 12.000')

    def test_resolve_start_anchor_plan_requires_finite_target(self) -> None:
        with self.assertRaises(ValueError):
            resolve_start_anchor_plan(Recipe(start_valid=True, start_ax0_abs=float('inf')))

    def test_resolve_standby_plan_returns_expected_targets(self) -> None:
        recipe = Recipe(
            standby_valid=True,
            standby_ax0_abs=100.0,
            standby_ax1_abs=200.0,
            standby_ax4_abs=300.0,
        )

        plan = resolve_standby_plan(recipe)

        self.assertTrue(plan.enabled)
        self.assertEqual(plan.targets_abs, {1: 200.0, 4: 300.0, 0: 100.0})

    def test_require_ax2_rotate_target_abs_raises_when_missing(self) -> None:
        with self.assertRaises(ValueError):
            require_ax2_rotate_target_abs(Recipe(ax2_rot_valid=False))

    def test_resolve_ax2_position_plan_prefers_rotate_target_for_keepout_reference(self) -> None:
        recipe = Recipe(
            ax2_len_valid=True,
            ax2_len_abs=12.5,
            ax2_rot_valid=True,
            ax2_rot_abs=34.5,
        )

        plan = resolve_ax2_position_plan(recipe, current_ax2_abs=99.0)

        self.assertEqual(plan.length_target_abs, 12.5)
        self.assertEqual(plan.rotate_target_abs, 34.5)
        self.assertEqual(plan.keepout_reference_abs, 34.5)
        self.assertEqual(resolve_ax2_keepout_reference_abs(recipe, current_ax2_abs=99.0), 34.5)

    def test_resolve_ax2_keepout_reference_abs_falls_back_to_current_position(self) -> None:
        recipe = Recipe(ax2_rot_valid=False)

        self.assertEqual(resolve_ax2_keepout_reference_abs(recipe, current_ax2_abs=88.0), 88.0)

    def test_resolve_section_targets_returns_keepout_safe_linear_targets(self) -> None:
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

        self.assertAlmostEqual(targets.ax0_abs, 25.0)
        self.assertAlmostEqual(targets.z_id_disp, 3.0)
        self.assertEqual(set(targets.linear_targets().keys()), {0, 1, 4})


if __name__ == '__main__':
    unittest.main()
