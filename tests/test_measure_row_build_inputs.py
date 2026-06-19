from __future__ import annotations

from frp_workflow.steps.measure_row_build_inputs import MeasureRowBuildInputs


def test_measure_row_build_inputs_keeps_values_and_identity() -> None:
    legacy = object()
    recipe = object()
    sensors = object()
    coords_od = object()
    coords_id = object()
    raw_points = [{"theta_deg": 0.0}]
    fit_weights_od = object()
    fit_weights_id = object()
    centers_xyz = [(1.0, 2.0, 3.0)]
    centers_xyz_id = [(4.0, 5.0, 6.0)]
    concentricity_list = [0.25]
    validation_fit_payload: dict[str, object] = {"existing": True}

    inputs = MeasureRowBuildInputs(
        legacy=legacy,
        recipe=recipe,
        sensors=sensors,
        section_index=2,
        z_pos_mm=7.5,
        x_abs=123.0,
        coords_od=coords_od,
        coords_id=coords_id,
        raw_od="od-raw",
        raw_id="id-raw",
        raw_points=raw_points,
        fit_weights_od=fit_weights_od,
        fit_weights_id=fit_weights_id,
        scan_mode="SYNC",
        split_shift_deg=1.25,
        coax_unreliable=False,
        centers_xyz=centers_xyz,
        centers_xyz_id=centers_xyz_id,
        concentricity_list=concentricity_list,
        validation_fit_payload=validation_fit_payload,
    )

    assert inputs.legacy is legacy
    assert inputs.recipe is recipe
    assert inputs.sensors is sensors
    assert inputs.section_index == 2
    assert inputs.z_pos_mm == 7.5
    assert inputs.x_abs == 123.0
    assert inputs.coords_od is coords_od
    assert inputs.coords_id is coords_id
    assert inputs.raw_points is raw_points
    assert inputs.fit_weights_od is fit_weights_od
    assert inputs.fit_weights_id is fit_weights_id
    assert inputs.scan_mode == "SYNC"
    assert inputs.split_shift_deg == 1.25
    assert inputs.coax_unreliable is False
    assert inputs.centers_xyz is centers_xyz
    assert inputs.centers_xyz_id is centers_xyz_id
    assert inputs.concentricity_list is concentricity_list
    assert inputs.validation_fit_payload is validation_fit_payload
