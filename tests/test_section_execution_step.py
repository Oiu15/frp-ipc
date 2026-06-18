from __future__ import annotations

from typing import Any

from frp_workflow.steps.section_execution import SectionExecutionStep


class FakePort:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = object()

    def _execute_section_impl(
        self,
        section: object,
        *,
        section_total: int,
        centers_xyz: list[tuple[float, float, float]],
        centers_xyz_id: list[tuple[float, float, float]],
        concentricity_list: list[float],
    ) -> object:
        self.calls.append(
            {
                "section": section,
                "section_total": section_total,
                "centers_xyz": centers_xyz,
                "centers_xyz_id": centers_xyz_id,
                "concentricity_list": concentricity_list,
            }
        )
        return self.result


def test_section_execution_step_delegates_args_and_returns_result() -> None:
    port = FakePort()
    section = object()
    centers_xyz = [(1.0, 2.0, 3.0)]
    centers_xyz_id = [(4.0, 5.0, 6.0)]
    concentricity_list = [0.12]
    step = SectionExecutionStep(port)

    result = step.execute(
        section,
        section_total=7,
        centers_xyz=centers_xyz,
        centers_xyz_id=centers_xyz_id,
        concentricity_list=concentricity_list,
    )

    assert result is port.result
    assert step.name == "section_execution"
    assert port.calls == [
        {
            "section": section,
            "section_total": 7,
            "centers_xyz": centers_xyz,
            "centers_xyz_id": centers_xyz_id,
            "concentricity_list": concentricity_list,
        }
    ]
