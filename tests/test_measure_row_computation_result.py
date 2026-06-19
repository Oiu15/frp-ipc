from __future__ import annotations

from frp_workflow.steps.measure_row_computation_result import MeasureRowComputationResult


def test_measure_row_computation_result_keeps_values() -> None:
    od_center = (1.0, 2.0, 3.0)
    id_center = (4.0, 5.0, 6.0)

    result = MeasureRowComputationResult(
        od_center=od_center,
        id_center=id_center,
        center_od_x=1.0,
        center_od_y=2.0,
        center_id_x=4.0,
        center_id_y=5.0,
        od_radius_fit_mm=7.0,
        od_diameter_fit_mm=14.0,
        id_radius_fit_mm=8.0,
        id_diameter_fit_mm=16.0,
        od_avg=14.2,
        od_dev=0.2,
        od_runout=0.3,
        od_round=0.4,
        od_round_fit_mm=0.5,
        od_round_fit_rob_mm=0.6,
        od_pp_mm=0.7,
        od_pp_rob_mm=0.8,
        id_avg=16.2,
        id_dev=0.2,
        id_runout=0.9,
        id_round=1.0,
        id_round_fit_mm=1.1,
        id_round_fit_rob_mm=1.2,
        id_pp_mm=1.3,
        id_pp_rob_mm=1.4,
        od_e=1.5,
        od_phi_deg=10.0,
        id_e=1.6,
        id_phi_deg=20.0,
        id_mode="dual",
        concentricity=2.5,
    )

    assert result.od_center is od_center
    assert result.id_center is id_center
    assert result.center_od_x == 1.0
    assert result.center_id_y == 5.0
    assert result.od_avg == 14.2
    assert result.id_avg == 16.2
    assert result.id_mode == "dual"
    assert result.concentricity == 2.5
