from __future__ import annotations

import pytest
from types import SimpleNamespace

from application.app_host import AppHost
from core.models import AxisCal, Recipe
from frp_workflow.autoflow_orchestrator import AutoFlowOrchestrator
from frp_workflow.validation_workflow import ValidationWorkflow


class _FakeAppHostPlanner:
    _section_plan_context = AppHost._section_plan_context
    _compute_recipe_section_plan = AppHost._compute_recipe_section_plan
    _bind_section_plan_to_recipe = AppHost._bind_section_plan_to_recipe
    _ensure_recipe_section_plan = AppHost._ensure_recipe_section_plan
    _build_recipe_section_plan = AppHost._build_recipe_section_plan

    def __init__(self, recipe: Recipe, axis_cal: AxisCal, *, ax2_abs: float, soft_limits: dict[int, tuple[float, float]]) -> None:
        self.recipe = recipe
        self.axis_cal = axis_cal
        self._ax2_abs = float(ax2_abs)
        self._axis_snapshots = {
            int(axis): SimpleNamespace(softlim_pos=float(pos), softlim_neg=float(neg))
            for axis, (pos, neg) in soft_limits.items()
        }

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
        gw = _FakeGateway(ax2_abs=ax2_abs, soft_limits=soft_limits)
        self.gateway = gw
        self.motion = gw
        self.sensors = gw  # type: ignore[assignment]
        self.operator = gw  # type: ignore[assignment]
        self._axis_cal = axis_cal
        self._ax2_abs = float(ax2_abs)

    def _require_axis_cal(self) -> AxisCal:
        return self._axis_cal


class _FakeValidationPlanner:
    _build_validation_recipe_section_plan = ValidationWorkflow._build_validation_recipe_section_plan

    def __init__(self, recipe: Recipe, *, ax2_abs: float, soft_limits: dict[int, tuple[float, float]]) -> None:
        self.recipe = recipe
        self._ax2_abs = float(ax2_abs)
        self._soft_limits = dict(soft_limits)

    def _get_validation_soft_limits_abs(self, axes):
        return {int(axis): self._soft_limits[int(axis)] for axis in axes}


class TestSectionPlannerParity:
    def test_recipe_production_and_validation_share_same_section_plan(self) -> None:
        recipe = Recipe(
            section_count=2,
            section_pos_z=[10.0, 45.0],
            ax2_rot_valid=True,
            ax2_rot_abs=30.0,
        )
        axis_cal = AxisCal(
            sign=1,
            off_ax0=0.0, off_ax1=0.0, off_ax2=0.0, off_ax4=0.0,
            b14=3.0, b2=8.0, keepout_w=5.0, z_pos=0.0,
        )
        soft_limits = {
            0: (100.0, -100.0),
            1: (100.0, -100.0),
            4: (100.0, -100.0),
        }

        recipe_plan = _FakeAppHostPlanner(recipe, axis_cal, ax2_abs=30.0, soft_limits=soft_limits)._build_recipe_section_plan()
        production_plan = _FakeOrchestratorPlanner(recipe, axis_cal, ax2_abs=30.0, soft_limits=soft_limits)._build_section_plan()
        validation_plan = _FakeValidationPlanner(recipe, ax2_abs=30.0, soft_limits=soft_limits)._build_validation_recipe_section_plan(axis_cal)

        assert recipe_plan.positions_z == production_plan.positions_z
        assert recipe_plan.positions_z == validation_plan.positions_z

        for recipe_index in range(len(recipe_plan.sections)):
            recipe_row = recipe_plan.section_for_recipe_index(recipe_index)
            production_row = production_plan.section_for_recipe_index(recipe_index)
            validation_row = validation_plan.section_for_recipe_index(recipe_index)
            assert recipe_row.section_index == recipe_index + 1
            assert recipe_row.z_od_disp == pytest.approx(production_row.z_od_disp)
            assert recipe_row.z_od_disp == pytest.approx(validation_row.z_od_disp)
            assert recipe_row.z_id_disp == pytest.approx(production_row.z_id_disp)
            assert recipe_row.z_id_disp == pytest.approx(validation_row.z_id_disp)
            assert recipe_row.linear_targets() == production_row.linear_targets()
            assert recipe_row.linear_targets() == validation_row.linear_targets()
