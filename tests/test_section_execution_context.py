from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from frp_workflow.steps.section_context import SectionExecutionContext


def test_section_execution_context_keeps_entry_values() -> None:
    section = object()
    centers_xyz = [(1.0, 2.0, 3.0)]
    centers_xyz_id = [(4.0, 5.0, 6.0)]
    concentricity_list = [0.25]

    context = SectionExecutionContext(
        section=section,
        section_index=2,
        total_sections=5,
        centers_xyz=centers_xyz,
        centers_xyz_id=centers_xyz_id,
        concentricity_list=concentricity_list,
    )

    assert context.section is section
    assert context.section_index == 2
    assert context.total_sections == 5
    assert context.centers_xyz is centers_xyz
    assert context.centers_xyz_id is centers_xyz_id
    assert context.concentricity_list is concentricity_list


def test_section_execution_context_is_frozen() -> None:
    context = SectionExecutionContext(
        section=object(),
        section_index=1,
        total_sections=1,
        centers_xyz=[],
        centers_xyz_id=[],
        concentricity_list=[],
    )

    with pytest.raises(FrozenInstanceError):
        setattr(context, "section_index", 3)
