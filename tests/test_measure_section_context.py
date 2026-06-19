from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from frp_workflow.steps.measure_section_context import MeasureSectionContext


def test_measure_section_context_keeps_entry_values() -> None:
    centers_xyz = [(1.0, 2.0, 3.0)]
    centers_xyz_id = [(4.0, 5.0, 6.0)]
    concentricity_list = [0.25]

    context = MeasureSectionContext(
        section_index=2,
        z_pos_mm=12.5,
        x_abs=101.0,
        centers_xyz=centers_xyz,
        centers_xyz_id=centers_xyz_id,
        concentricity_list=concentricity_list,
    )

    assert context.section_index == 2
    assert context.z_pos_mm == 12.5
    assert context.x_abs == 101.0
    assert context.centers_xyz is centers_xyz
    assert context.centers_xyz_id is centers_xyz_id
    assert context.concentricity_list is concentricity_list


def test_measure_section_context_is_frozen() -> None:
    context = MeasureSectionContext(
        section_index=1,
        z_pos_mm=0.0,
        x_abs=0.0,
        centers_xyz=[],
        centers_xyz_id=[],
        concentricity_list=[],
    )

    with pytest.raises(FrozenInstanceError):
        setattr(context, "section_index", 3)
