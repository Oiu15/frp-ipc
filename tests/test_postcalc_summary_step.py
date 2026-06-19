from __future__ import annotations

from frp_workflow.steps.postcalc_summary import PostcalcSummaryStep


class FakePort:
    def __init__(self) -> None:
        self.calls: list[tuple[
            list[tuple[float, float, float]],
            list[tuple[float, float, float]],
            list[float],
        ]] = []

    def _run_postcalc_impl(
        self,
        *,
        centers_xyz: list[tuple[float, float, float]],
        centers_xyz_id: list[tuple[float, float, float]],
        concentricity_list: list[float],
    ) -> None:
        self.calls.append((centers_xyz, centers_xyz_id, concentricity_list))


def test_postcalc_summary_step_delegates_to_impl() -> None:
    centers_xyz = [(1.0, 2.0, 3.0)]
    centers_xyz_id = [(4.0, 5.0, 6.0)]
    concentricity_list = [0.25]
    port = FakePort()
    step = PostcalcSummaryStep(port)

    result = step.execute(
        centers_xyz=centers_xyz,
        centers_xyz_id=centers_xyz_id,
        concentricity_list=concentricity_list,
    )

    assert result is None
    assert port.calls == [(centers_xyz, centers_xyz_id, concentricity_list)]
    assert port.calls[0][0] is centers_xyz
    assert port.calls[0][1] is centers_xyz_id
    assert port.calls[0][2] is concentricity_list
    assert step.name == "postcalc_summary"
