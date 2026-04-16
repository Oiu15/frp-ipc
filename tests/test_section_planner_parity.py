from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace

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
from frp_workflow.autoflow_orchestrator import AutoFlowOrchestrator
from frp_workflow.validation_workflow import ValidationWorkflow


class _FakeAppHostPlanner:
    _build_recipe_section_plan = AppHost._build_recipe_section_plan

    def __init__(self, recipe: Recipe, axis_cal: AxisCal, *, ax2_abs: float, soft_limits: dict[int, tuple[float, float]]) -> None:
        self.recipe = recipe
        self.axis_cal = axis_cal
        self._ax2_abs = float(ax2_abs)
        self._axis_snapshots = {
            int(axis): SimpleNamespace(softlim_pos=float(pos), softlim_neg=float(neg))
            for axis, (pos, neg) in soft_limits.items()
        }

    def _get_ax2_keepout_ref_abs(self, prefer_rot: bool = True) -> float:
        return float(self._ax2_abs)

    def get_axis_copy(self, axis: int):
        return self._axis_snapshots[int(axis)]


class _FakeGateway:
    def __init__(self, *, ax2_abs: float, soft_limits: dict[int, tuple[float, float]]) -> None:
        self._ax2_abs = float(ax2_abs)
        self._soft_limits = dict(soft_limits)

    def get_axis_copy(self, axis: int):
        pos, neg = self._soft_limits.get(int(axis), (0.0, 0.0))
        act_pos = self._ax2_abs if int(axis) == 2 else 0.0
        return SimpleNamespace(
            softlim_pos=float(pos),
            softlim_neg=float(neg),
            act_pos=float(act_pos),
        )

    def apply_soft_limits_abs(self, axis: int, target: float, *, strict: bool = False):
        return float(target)


class _FakeOrchestratorPlanner:
    _build_section_plan = AutoFlowOrchestrator._build_section_plan
    _soft_limits_from_axis = AutoFlowOrchestrator._soft_limits_from_axis

    def __init__(self, recipe: Recipe, axis_cal: AxisCal, *, ax2_abs: float, soft_limits: dict[int, tuple[float, float]]) -> None:
        self.recipe = recipe
        self.gateway = _FakeGateway(ax2_abs=ax2_abs, soft_limits=soft_limits)
        self._axis_cal = axis_cal
        self._ax2_abs = float(ax2_abs)

    def _require_axis_cal(self) -> AxisCal:
        return self._axis_cal

    def _get_ax2_keepout_ref_abs(self) -> float:
        return float(self._ax2_abs)


class _FakeValidationPlanner:
    _build_validation_recipe_section_plan = ValidationWorkflow._build_validation_recipe_section_plan

    def __init__(self, recipe: Recipe, *, ax2_abs: float, soft_limits: dict[int, tuple[float, float]]) -> None:
        self.recipe = recipe
        self._ax2_abs = float(ax2_abs)
        self._soft_limits = dict(soft_limits)

    def _get_validation_ax2_keepout_reference_abs(self) -> float:
        return float(self._ax2_abs)

    def _get_validation_soft_limits_abs(self, axes):
        return {int(axis): self._soft_limits[int(axis)] for axis in axes}


class SectionPlannerParityTest(unittest.TestCase):
    def test_recipe_production_and_validation_share_same_section_plan(self) -> None:
        recipe = Recipe(
            section_count=2,
            section_pos_z=[10.0, 45.0],
            ax2_rot_valid=True,
            ax2_rot_abs=30.0,
        )
        axis_cal = AxisCal(
            sign=1,
            off_ax0=0.0,
            off_ax1=0.0,
            off_ax2=0.0,
            off_ax4=0.0,
            b14=3.0,
            b2=8.0,
            keepout_w=5.0,
            z_pos=0.0,
        )
        soft_limits = {
            0: (100.0, -100.0),
            1: (100.0, -100.0),
            4: (100.0, -100.0),
        }

        recipe_plan = _FakeAppHostPlanner(recipe, axis_cal, ax2_abs=30.0, soft_limits=soft_limits)._build_recipe_section_plan()
        production_plan = _FakeOrchestratorPlanner(recipe, axis_cal, ax2_abs=30.0, soft_limits=soft_limits)._build_section_plan()
        validation_plan = _FakeValidationPlanner(recipe, ax2_abs=30.0, soft_limits=soft_limits)._build_validation_recipe_section_plan(axis_cal)

        self.assertEqual(recipe_plan.positions_z, production_plan.positions_z)
        self.assertEqual(recipe_plan.positions_z, validation_plan.positions_z)

        for recipe_index in range(len(recipe_plan.sections)):
            recipe_row = recipe_plan.section_for_recipe_index(recipe_index)
            production_row = production_plan.section_for_recipe_index(recipe_index)
            validation_row = validation_plan.section_for_recipe_index(recipe_index)
            self.assertEqual(recipe_row.section_index, recipe_index + 1)
            self.assertAlmostEqual(recipe_row.z_od_disp, production_row.z_od_disp)
            self.assertAlmostEqual(recipe_row.z_od_disp, validation_row.z_od_disp)
            self.assertAlmostEqual(recipe_row.z_id_disp, production_row.z_id_disp)
            self.assertAlmostEqual(recipe_row.z_id_disp, validation_row.z_id_disp)
            self.assertEqual(recipe_row.linear_targets(), production_row.linear_targets())
            self.assertEqual(recipe_row.linear_targets(), validation_row.linear_targets())


if __name__ == '__main__':
    unittest.main()
