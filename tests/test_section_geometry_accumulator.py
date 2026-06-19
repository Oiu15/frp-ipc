from __future__ import annotations

from frp_workflow.steps.section_geometry_accumulator import SectionGeometryAccumulator


def test_section_geometry_accumulator_appends_to_original_lists() -> None:
    centers_xyz: list[tuple[float, float, float]] = []
    centers_xyz_id: list[tuple[float, float, float]] = []
    concentricity_list: list[float] = []
    accumulator = SectionGeometryAccumulator(
        centers_xyz=centers_xyz,
        centers_xyz_id=centers_xyz_id,
        concentricity_list=concentricity_list,
    )

    od_center = (1.0, 2.0, 3.0)
    id_center = (4.0, 5.0, 6.0)
    accumulator.append_od_center(od_center)
    accumulator.append_id_center(id_center)
    accumulator.append_concentricity(0.25)

    assert accumulator.centers_xyz is centers_xyz
    assert accumulator.centers_xyz_id is centers_xyz_id
    assert accumulator.concentricity_list is concentricity_list
    assert centers_xyz == [od_center]
    assert centers_xyz_id == [id_center]
    assert concentricity_list == [0.25]
