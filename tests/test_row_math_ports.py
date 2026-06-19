from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from frp_workflow.row_math import _compute_measure_row_result
from frp_workflow.steps.measure_row_build_inputs import MeasureRowBuildInputs


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


class _MinimalLegacyFit:
    def __init__(self) -> None:
        self.fit_circle_calls = 0

    def fit_circle(self, coords: Any, *, weights: Any | None = None) -> tuple[float, float, float, float]:
        self.fit_circle_calls += 1
        if self.fit_circle_calls == 1:
            return 10.0, 20.0, 1.0, 0.01
        return 13.0, 24.0, 1.0, 0.02

    def get_active_id_delta_c(self) -> float:
        return 0.0

    def fit_id_from_raw_points(
        self,
        raw_points: Any,
        delta_c: float,
        *,
        theta_delay_s: float = 0.0,
    ) -> tuple[None, None]:
        return None, None

    def od_round_fit_from_raw_points(self, raw_points: Any, **kwargs: Any) -> tuple[float, float]:
        return 0.3, 0.4

    def id_round_fit_from_raw_points(self, raw_points: Any, **kwargs: Any) -> tuple[float, float]:
        return 0.5, 0.6


def test_row_math_uses_minimal_legacy_fit_port() -> None:
    legacy = _MinimalLegacyFit()

    result = _compute_measure_row_result(
        MeasureRowBuildInputs(
            legacy=cast(Any, legacy),
            recipe=cast(Any, _FakeRecipe()),
            sensors=None,
            section_index=2,
            z_pos_mm=7.5,
            x_abs=123.0,
            coords_od=np.array([[11.0, 20.0, 0.0], [9.0, 20.0, 0.0]]),
            coords_id=np.array([[14.0, 24.0, 0.0], [12.0, 24.0, 0.0]]),
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
            centers_xyz=[],
            centers_xyz_id=[],
            concentricity_list=[],
        )
    )

    assert result.od_center == (10.0, 20.0, 7.5)
    assert result.id_center == (13.0, 24.0, 7.5)
    assert result.concentricity == 5.0
