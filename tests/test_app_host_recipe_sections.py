from __future__ import annotations

import sys
import types
import unittest

_pymodbus = types.ModuleType("pymodbus")
_pymodbus_client = types.ModuleType("pymodbus.client")


class _FakeModbusTcpClient:
    pass


_pymodbus_client.ModbusTcpClient = _FakeModbusTcpClient
_pymodbus.client = _pymodbus_client
sys.modules.setdefault("pymodbus", _pymodbus)
sys.modules.setdefault("pymodbus.client", _pymodbus_client)

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

    def _build_recipe_section_plan(self, recipe=None):
        self.plan_requests.append(self.recipe if recipe is None else recipe)
        return RecipeSectionPlan(
            positions_z=(10.0, 20.0),
            sections=(
                RecipeSectionPlanRow(
                    section_index=1,
                    z_od_disp=10.0,
                    z_id_disp=13.0,
                    ax0_abs=101.0,
                    ax1_abs=201.0,
                    ax4_abs=401.0,
                ),
                RecipeSectionPlanRow(
                    section_index=2,
                    z_od_disp=20.0,
                    z_id_disp=23.0,
                    ax0_abs=102.0,
                    ax1_abs=202.0,
                    ax4_abs=402.0,
                ),
            ),
        )

    def movea_abs(self, axis: int, target_abs: float, *, context: str = '') -> None:
        self.moves.append((int(axis), float(target_abs), str(context)))


class AppHostRecipeSectionsTest(unittest.TestCase):
    def test_refresh_recipe_table_uses_section_plan_targets(self) -> None:
        host = _FakeRecipeSectionHost()

        host._refresh_recipe_table()

        self.assertEqual(len(host.plan_requests), 1)
        self.assertEqual(host._tree.rows[0][:6], (0, '10.000', '13.000', '101.000', '201.000', '401.000'))
        self.assertEqual(host._tree.rows[1][:6], (1, '20.000', '23.000', '102.000', '202.000', '402.000'))
        self.assertEqual(host.recipe.section_pos_z, [10.0, 20.0])
        self.assertEqual(host.recipe.section_pos_ui, [10.0, 20.0])

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


if __name__ == '__main__':
    unittest.main()
