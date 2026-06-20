from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from frp_workflow.autoflow_orchestrator import _build_measure_row_from_sampling
from frp_workflow.steps.measure_row_build_inputs import MeasureRowBuildInputs


_PAYLOAD_KEYS = {
    "od_center_x_mm",
    "od_center_y_mm",
    "od_radius_mm",
    "od_diameter_fit_mm",
    "id_center_x_mm",
    "id_center_y_mm",
    "id_radius_mm",
    "id_diameter_fit_mm",
    "od_ecc_mm",
    "id_ecc_mm",
    "concentricity_mm",
}


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

    def od_round_fit_from_raw_points(self, *_args: Any, **_kwargs: Any) -> tuple[float, float]:
        return 0.3, 0.4

    def id_round_fit_from_raw_points(self, *_args: Any, **_kwargs: Any) -> tuple[float, float]:
        return 0.5, 0.6


def test_validation_fit_payload_characterizes_od_id_path() -> None:
    legacy = _FakeLegacyFlow()
    recipe = _FakeRecipe()
    centers_xyz: list[tuple[float, float, float]] = []
    centers_xyz_id: list[tuple[float, float, float]] = []
    concentricity_list: list[float] = []
    validation_fit_payload: dict[str, object] = {"old": "value"}
    coords_od = np.array([[11.0, 20.0, 0.0], [9.0, 20.0, 0.0]])
    coords_id = np.array([[14.0, 24.0, 0.0], [12.0, 24.0, 0.0]])

    row = _build_measure_row_from_sampling(
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
            fit_weights_od=object(),
            fit_weights_id=object(),
            scan_mode="SYNC",
            split_shift_deg=None,
            coax_unreliable=None,
            centers_xyz=centers_xyz,
            centers_xyz_id=centers_xyz_id,
            concentricity_list=concentricity_list,
            validation_fit_payload=validation_fit_payload,
        )
    )

    assert set(validation_fit_payload) == _PAYLOAD_KEYS
    assert "old" not in validation_fit_payload
    assert validation_fit_payload == {
        "od_center_x_mm": 10.0,
        "od_center_y_mm": 20.0,
        "od_radius_mm": 1.0,
        "od_diameter_fit_mm": 2.0,
        "id_center_x_mm": 13.0,
        "id_center_y_mm": 24.0,
        "id_radius_mm": 1.0,
        "id_diameter_fit_mm": 2.0,
        "od_ecc_mm": None,
        "id_ecc_mm": None,
        "concentricity_mm": row.concentricity,
    }
    assert validation_fit_payload["concentricity_mm"] == concentricity_list[-1]
    assert centers_xyz == [(10.0, 20.0, 7.5)]
    assert centers_xyz_id == [(13.0, 24.0, 7.5)]
    assert concentricity_list == [5.0]


def test_validation_fit_payload_characterizes_id_single_fallback_path() -> None:
    legacy = _FakeLegacyFlow()
    recipe = _FakeRecipe(id_single_enable=True)
    centers_xyz: list[tuple[float, float, float]] = []
    centers_xyz_id: list[tuple[float, float, float]] = []
    concentricity_list: list[float] = []
    validation_fit_payload: dict[str, object] = {"old": "value"}
    coords_od = np.array([[11.0, 20.0, 0.0], [9.0, 20.0, 0.0]])

    row = _build_measure_row_from_sampling(
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
            fit_weights_od=object(),
            fit_weights_id=object(),
            scan_mode="SYNC",
            split_shift_deg=None,
            coax_unreliable=None,
            centers_xyz=centers_xyz,
            centers_xyz_id=centers_xyz_id,
            concentricity_list=concentricity_list,
            validation_fit_payload=validation_fit_payload,
        )
    )

    assert set(validation_fit_payload) == _PAYLOAD_KEYS
    assert "old" not in validation_fit_payload
    assert validation_fit_payload == {
        "od_center_x_mm": 10.0,
        "od_center_y_mm": 20.0,
        "od_radius_mm": 1.0,
        "od_diameter_fit_mm": 2.0,
        "id_center_x_mm": None,
        "id_center_y_mm": None,
        "id_radius_mm": None,
        "id_diameter_fit_mm": None,
        "od_ecc_mm": None,
        "id_ecc_mm": None,
        "concentricity_mm": None,
    }
    assert row.id_mode == "single"
    assert row.concentricity is None
    assert validation_fit_payload["concentricity_mm"] is row.concentricity
    assert centers_xyz == [(10.0, 20.0, 8.5)]
    assert centers_xyz_id == []
    assert concentricity_list == []
