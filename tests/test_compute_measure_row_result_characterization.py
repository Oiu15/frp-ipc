from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from frp_workflow.row_math import _compute_measure_row_result
from frp_workflow.steps.measure_row_build_inputs import MeasureRowBuildInputs
from frp_workflow.steps.measure_row_computation_result import MeasureRowComputationResult


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
        self.round_fit_calls: list[str] = []

    def fit_circle(self, coords: np.ndarray, *, weights: Any) -> tuple[float, float, float, float]:
        self.fit_circle_calls.append((coords, weights))
        if len(self.fit_circle_calls) == 1:
            return 10.0, 20.0, 1.0, 0.01
        return 13.0, 24.0, 1.0, 0.02

    def get_active_id_delta_c(self) -> float:
        return 0.0

    def od_round_fit_from_raw_points(self, *_args: Any, **_kwargs: Any) -> tuple[float, float]:
        self.round_fit_calls.append("od")
        return 0.3, 0.4

    def id_round_fit_from_raw_points(self, *_args: Any, **_kwargs: Any) -> tuple[float, float]:
        self.round_fit_calls.append("id")
        return 0.5, 0.6


def test_compute_measure_row_result_characterizes_od_id_path_without_side_effects() -> None:
    legacy = _FakeLegacyFlow()
    recipe = _FakeRecipe()
    centers_xyz: list[tuple[float, float, float]] = []
    centers_xyz_id: list[tuple[float, float, float]] = []
    concentricity_list: list[float] = []
    validation_fit_payload: dict[str, object] = {"old": "value"}
    coords_od = np.array([[11.0, 20.0, 0.0], [9.0, 20.0, 0.0]])
    coords_id = np.array([[14.0, 24.0, 0.0], [12.0, 24.0, 0.0]])
    fit_weights_od = object()
    fit_weights_id = object()

    result = _compute_measure_row_result(
        MeasureRowBuildInputs(
            legacy=cast(Any, legacy),
            recipe=cast(Any, recipe),
            sensors=None,
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
            validation_fit_payload=validation_fit_payload,
        )
    )

    assert isinstance(result, MeasureRowComputationResult)
    assert result.od_center == (10.0, 20.0, 7.5)
    assert result.id_center == (13.0, 24.0, 7.5)
    assert result.center_od_x == 10.0
    assert result.center_od_y == 20.0
    assert result.center_id_x == 13.0
    assert result.center_id_y == 24.0
    assert result.od_radius_fit_mm == 1.0
    assert result.od_diameter_fit_mm == 2.0
    assert result.id_radius_fit_mm == 1.0
    assert result.id_diameter_fit_mm == 2.0
    assert result.od_avg == 2.0
    assert result.od_dev == 0.0
    assert result.id_avg == 2.0
    assert result.id_dev == 0.0
    assert result.concentricity == 5.0
    assert result.id_mode == "dual"
    assert centers_xyz == []
    assert centers_xyz_id == []
    assert concentricity_list == []
    assert validation_fit_payload == {"old": "value"}
    assert legacy.fit_circle_calls == [
        (coords_od, fit_weights_od),
        (coords_id, fit_weights_id),
    ]
    assert legacy.round_fit_calls == ["od", "id"]


def test_compute_measure_row_result_characterizes_id_single_fallback_without_side_effects() -> None:
    legacy = _FakeLegacyFlow()
    recipe = _FakeRecipe(id_single_enable=True)
    centers_xyz: list[tuple[float, float, float]] = []
    centers_xyz_id: list[tuple[float, float, float]] = []
    concentricity_list: list[float] = []
    validation_fit_payload: dict[str, object] = {"old": "value"}
    coords_od = np.array([[11.0, 20.0, 0.0], [9.0, 20.0, 0.0]])
    fit_weights_od = object()
    fit_weights_id = object()

    result = _compute_measure_row_result(
        MeasureRowBuildInputs(
            legacy=cast(Any, legacy),
            recipe=cast(Any, recipe),
            sensors=None,
            section_index=3,
            z_pos_mm=8.5,
            x_abs=124.0,
            coords_od=coords_od,
            coords_id=np.array([]),
            raw_od="od-raw",
            raw_id="missing-id",
            raw_points=[
                {"theta_deg": 0.0, "od_mm": 2.0},
                {"theta_deg": 180.0, "od_mm": 2.0},
            ],
            fit_weights_od=fit_weights_od,
            fit_weights_id=fit_weights_id,
            scan_mode="SYNC",
            split_shift_deg=None,
            coax_unreliable=None,
            centers_xyz=centers_xyz,
            centers_xyz_id=centers_xyz_id,
            concentricity_list=concentricity_list,
            validation_fit_payload=validation_fit_payload,
        )
    )

    assert isinstance(result, MeasureRowComputationResult)
    assert result.od_center == (10.0, 20.0, 8.5)
    assert result.id_center is None
    assert result.center_id_x is None
    assert result.center_id_y is None
    assert result.id_radius_fit_mm is None
    assert result.id_diameter_fit_mm is None
    assert result.id_avg is None
    assert result.id_dev is None
    assert result.id_round is None
    assert result.id_runout is None
    assert result.concentricity is None
    assert result.id_mode == "single"
    assert centers_xyz == []
    assert centers_xyz_id == []
    assert concentricity_list == []
    assert validation_fit_payload == {"old": "value"}
    assert legacy.fit_circle_calls == [(coords_od, fit_weights_od)]
    assert legacy.round_fit_calls == ["od"]
