from __future__ import annotations

import types
import unittest

from tests.fakes import FakeVar

from application.app_host import AppHost
from core.models import AxisCal, Recipe
from domain.planning import RecipeSectionPlan, RecipeSectionPlanRow


class _FakeRecipeTree:
    def __init__(self) -> None:
        self.rows: list[tuple] = []
        self._selection: list[str] = []

    def get_children(self):
        return [str(i) for i in range(len(self.rows))]

    def delete(self, *items) -> None:
        self.rows.clear()

    def insert(self, parent, index, values):
        self.rows.append(tuple(values))
        item_id = str(len(self.rows) - 1)
        return item_id

    def selection(self):
        return tuple(self._selection)

    def item(self, item_id, option=None):
        values = self.rows[int(item_id)]
        if option == "values":
            return values
        return {"values": values}

    def select_index(self, row_index: int) -> None:
        self._selection = [str(row_index)]


class _FakeRecipeSectionHost:
    _refresh_recipe_table = AppHost._refresh_recipe_table
    _teach_move_to_selected = AppHost._teach_move_to_selected
    _get_selected_recipe_idx = AppHost._get_selected_recipe_idx

    def __init__(self) -> None:
        self.recipe = Recipe(section_count=2, section_pos_z=[10.0, 20.0], teach_axes_mode=2)
        self.axis_cal = AxisCal(b14=3.0)
        self._tree = _FakeRecipeTree()
        self.moves: list[tuple[int, float, str]] = []
        self.plan_requests: list[Recipe] = []

    def _recipe_ui_widget(self, name: str):
        if name == 'recipe_tree':
            return self._tree
        return None

    def _recipe_apply_from_ui(self) -> Recipe:
        return self.recipe

    def _ensure_recipe_section_plan(self, recipe=None):
        recipe_obj = self.recipe if recipe is None else recipe
        plan = self._build_recipe_section_plan(recipe_obj)
        recipe_obj.section_pos_z = list(plan.positions_z)
        recipe_obj.section_pos_ui = list(plan.positions_z)
        return plan

    def _build_recipe_section_plan(self, recipe=None):
        recipe_obj = self.recipe if recipe is None else recipe
        self.plan_requests.append(recipe_obj)
        positions = tuple(float(value) for value in getattr(recipe_obj, "section_pos_z", []))
        return RecipeSectionPlan(
            positions_z=positions,
            sections=tuple(
                RecipeSectionPlanRow(
                    section_index=index + 1,
                    z_od_disp=z_pos,
                    z_id_disp=z_pos + 3.0,
                    ax0_abs=101.0 + index,
                    ax1_abs=201.0 + index,
                    ax4_abs=401.0 + index,
                    source="taught" if getattr(recipe_obj, "name", "") == "taught-source" and index == 0 else "computed",
                )
                for index, z_pos in enumerate(positions)
            ),
        )

    def movea_abs(self, axis: int, target_abs: float, *, context: str = '') -> None:
        self.moves.append((int(axis), float(target_abs), str(context)))


class _FakeStartAnchorHost:
    _apply_start_anchor_from_recipe = AppHost._apply_start_anchor_from_recipe

    def __init__(self) -> None:
        self.recipe = Recipe(start_valid=False)
        self.axis_cal = AxisCal(sign=1, off_ax0=10.0, z_pos=123.0)
        self.axis_cal_vars = {'z_pos': FakeVar()}
        self.axis_cal_field_status_vars = {'z_pos': FakeVar()}
        self.refresh_start_calls = 0
        self.refresh_teach_calls = 0

    def _refresh_start_pos(self) -> None:
        self.refresh_start_calls += 1

    def _refresh_teach_pos(self) -> None:
        self.refresh_teach_calls += 1


class _FakeGotoStartHost(_FakeStartAnchorHost):
    _teach_goto_start = AppHost._teach_goto_start

    def __init__(self) -> None:
        super().__init__()
        self.recipe = Recipe(start_valid=True, start_ax0_abs=42.0, teach_axes_mode=2)
        self.axis_cal = AxisCal(sign=1, off_ax0=10.0, off_ax1=0.0, off_ax4=0.0, b14=4.0, z_pos=-999.0)
        self.moves: list[tuple[int, float, str]] = []
        self._axis_snapshots = {
            0: types.SimpleNamespace(softlim_pos=1000.0, softlim_neg=-1000.0),
            1: types.SimpleNamespace(softlim_pos=1000.0, softlim_neg=-1000.0),
            4: types.SimpleNamespace(softlim_pos=1000.0, softlim_neg=-1000.0),
        }

    def get_axis_copy(self, axis: int):
        return self._axis_snapshots[int(axis)]

    def movea_abs(self, axis: int, target_abs: float, *, context: str = '') -> None:
        self.moves.append((int(axis), float(target_abs), str(context)))


class AppHostRecipeSectionsTest(unittest.TestCase):
    def test_apply_start_anchor_clears_previous_zpos_when_recipe_has_no_start(self) -> None:
        host = _FakeStartAnchorHost()

        host._apply_start_anchor_from_recipe()

        self.assertEqual(host.axis_cal.z_pos, 0.0)
        self.assertEqual(host.axis_cal_vars['z_pos'].get(), "0.000000")
        self.assertEqual(host.axis_cal_field_status_vars['z_pos'].get(), "无配方Start")
        self.assertEqual(host.refresh_start_calls, 1)
        self.assertEqual(host.refresh_teach_calls, 1)

    def test_apply_start_anchor_updates_zpos_from_recipe_start(self) -> None:
        host = _FakeStartAnchorHost()
        host.recipe = Recipe(start_valid=True, start_ax0_abs=42.0)

        host._apply_start_anchor_from_recipe()

        self.assertEqual(host.axis_cal.z_pos, 32.0)
        self.assertEqual(host.axis_cal_vars['z_pos'].get(), "32.000000")
        self.assertEqual(host.axis_cal_field_status_vars['z_pos'].get(), "配方Start")
        self.assertEqual(host.refresh_start_calls, 1)
        self.assertEqual(host.refresh_teach_calls, 1)

    def test_teach_goto_start_moves_ax0_to_recipe_start_abs(self) -> None:
        host = _FakeGotoStartHost()

        host._teach_goto_start()

        self.assertEqual(host.axis_cal.z_pos, 32.0)
        self.assertIn((0, 42.0, "GotoStart"), host.moves)

    def test_refresh_recipe_table_uses_section_plan_targets(self) -> None:
        host = _FakeRecipeSectionHost()

        host._refresh_recipe_table()

        self.assertEqual(len(host.plan_requests), 1)
        self.assertEqual(host._tree.rows[0][:6], (0, '10.000', '13.000', '101.000', '201.000', '401.000'))
        self.assertEqual(host._tree.rows[1][:6], (1, '20.000', '23.000', '102.000', '202.000', '402.000'))
        self.assertEqual(host._tree.rows[0][6], 'computed')
        self.assertEqual(host.recipe.section_pos_z, [10.0, 20.0])
        self.assertEqual(host.recipe.section_pos_ui, [10.0, 20.0])

    def test_refresh_recipe_table_uses_current_plan_source(self) -> None:
        host = _FakeRecipeSectionHost()
        host.recipe.name = "taught-source"

        host._refresh_recipe_table()

        self.assertEqual(host._tree.rows[0][6], 'taught')
        self.assertEqual(host._tree.rows[1][6], 'computed')

    def test_refresh_recipe_table_does_not_retain_previous_recipe_rows(self) -> None:
        host = _FakeRecipeSectionHost()
        host._refresh_recipe_table()

        host.recipe = Recipe(name='recipe-b', section_count=1, section_pos_z=[77.0], teach_axes_mode=2)
        host._refresh_recipe_table()

        self.assertEqual(host._tree.rows, [(0, '77.000', '80.000', '101.000', '201.000', '401.000', 'computed')])

    def test_teach_move_to_selected_reuses_section_plan_targets(self) -> None:
        host = _FakeRecipeSectionHost()
        host._refresh_recipe_table()
        host._tree.select_index(1)

        host._teach_move_to_selected()

        self.assertEqual(len(host.plan_requests), 2)
        self.assertEqual(
            host.moves,
            [
                (0, 102.0, 'SectionMove'),
                (1, 202.0, 'SectionMove'),
                (4, 402.0, 'SectionMove'),
            ],
        )

    def test_teach_move_to_selected_uses_current_recipe_plan_not_stale_table(self) -> None:
        host = _FakeRecipeSectionHost()
        host._refresh_recipe_table()
        host._tree.select_index(0)
        host.recipe = Recipe(name='recipe-b', section_count=1, section_pos_z=[77.0], teach_axes_mode=2)

        host._teach_move_to_selected()

        self.assertEqual(
            host.moves,
            [
                (0, 101.0, 'SectionMove'),
                (1, 201.0, 'SectionMove'),
                (4, 401.0, 'SectionMove'),
            ],
        )
        self.assertEqual(host.plan_requests[-1].name, 'recipe-b')


if __name__ == '__main__':
    unittest.main()
