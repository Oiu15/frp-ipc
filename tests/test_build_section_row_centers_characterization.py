from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from frp_workflow.autoflow_orchestrator import AutoFlowOrchestrator


@dataclass
class _FakeRecipe:
    od_std_mm: float = 2.0
    id_std_mm: float = 2.0
    od_tol_mm: float = 10.0
    id_single_enable: bool = False
    id_use_fit: bool = False
    od_use_edges: bool = False
    pp_mode: str = "strict"
    calc_input_mode: str = "bin"
    fit_strategy: str = ""
    theta_delay_s: float = 0.0
    bin_count: int = 90
    bin_method: str = "median"


class _FakeLegacyFlow:
    def __init__(self) -> None:
        self.fit_circle_calls: list[tuple[np.ndarray, Any]] = []

    def fit_circle(self, coords: np.ndarray, *, weights: Any) -> tuple[float, float, float, float]:
        self.fit_circle_calls.append((coords, weights))
        if len(self.fit_circle_calls) == 1:
            return 10.0, 20.0, 1.0, 0.01
        return 13.0, 24.0, 1.0, 0.02

    def get_active_id_delta_c(self) -> float:
        return 0.0

    def od_round_fit_from_raw_points(self, *_args: Any, **_kwargs: Any) -> tuple[None, None]:
        return None, None

    def id_round_fit_from_raw_points(self, *_args: Any, **_kwargs: Any) -> tuple[None, None]:
        return None, None


def test_build_section_row_appends_centers_and_concentricity(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = cast(Any, AutoFlowOrchestrator.__new__(AutoFlowOrchestrator))
    legacy = _FakeLegacyFlow()
    orchestrator.recipe = _FakeRecipe()
    orchestrator.sensors = None
    monkeypatch.setattr(orchestrator, "_require_legacy_flow", lambda: legacy)
    centers_xyz: list[tuple[float, float, float]] = []
    centers_xyz_id: list[tuple[float, float, float]] = []
    concentricity_list: list[float] = []
    coords_od = np.array([[11.0, 20.0, 0.0], [9.0, 20.0, 0.0]])
    coords_id = np.array([[14.0, 24.0, 0.0], [12.0, 24.0, 0.0]])
    fit_weights_od = object()
    fit_weights_id = object()

    row = AutoFlowOrchestrator._build_section_row(
        orchestrator,
        section_index=2,
        z_pos_mm=7.5,
        x_abs=123.0,
        coords_od=coords_od,
        coords_id=coords_id,
        raw_od="od-raw",
        raw_id="id-raw",
        raw_points=[
            {"theta_deg": 0.0, "od_mm": 2.0, "id_mm": 2.0},
            {"theta_deg": 180.0, "od_mm": 2.0, "id_mm": 2.0},
        ],
        fit_weights_od=fit_weights_od,
        fit_weights_id=fit_weights_id,
        scan_mode="SYNC",
        split_shift_deg=None,
        coax_unreliable=None,
        centers_xyz=centers_xyz,
        centers_xyz_id=centers_xyz_id,
        concentricity_list=concentricity_list,
    )

    assert centers_xyz == [(10.0, 20.0, 7.5)]
    assert centers_xyz_id == [(13.0, 24.0, 7.5)]
    assert concentricity_list == [5.0]
    assert row.idx == 2
    assert row.x_ui == 7.5
    assert row.x_abs == 123.0
    assert row.concentricity == 5.0
    assert legacy.fit_circle_calls == [
        (coords_od, fit_weights_od),
        (coords_id, fit_weights_id),
    ]
