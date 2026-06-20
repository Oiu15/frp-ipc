from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

import frp_workflow.autoflow_orchestrator as orchestrator_module
from frp_workflow.autoflow_orchestrator import AutoFlowOrchestrator


@dataclass
class _FakeRecipe:
    id_single_enable: bool = False


@dataclass
class _FakePostcalcResult:
    straightness_payload: dict[str, Any]
    postcalc_payload: dict[str, Any]


class _FakeProductionWorkflow:
    def __init__(self, calls: list[tuple[str, str, Any]]) -> None:
        self._calls = calls

    def record_summary(self, payload: dict[str, Any], *, source: str) -> None:
        self._calls.append(("record_summary", source, payload))


class _FakeEventSink:
    def __init__(self, calls: list[tuple[str, str, Any]]) -> None:
        self._calls = calls

    def publish_straightness(self, payload: dict[str, Any]) -> None:
        self._calls.append(("publish", "straightness", payload))

    def publish_postcalc(self, payload: dict[str, Any]) -> None:
        self._calls.append(("publish", "postcalc", payload))


def test_run_postcalc_impl_records_and_publishes_computed_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, Any]] = []
    centers_xyz = [(1.0, 2.0, 3.0)]
    centers_xyz_id = [(4.0, 5.0, 6.0)]
    concentricity_list = [0.25]
    straightness_payload = {"straight_od": 0.1}
    postcalc_payload = {"ecc_od": 0.2}
    result = _FakePostcalcResult(
        straightness_payload=straightness_payload,
        postcalc_payload=postcalc_payload,
    )
    orchestrator = cast(Any, AutoFlowOrchestrator.__new__(AutoFlowOrchestrator))
    orchestrator.recipe = _FakeRecipe(id_single_enable=True)
    orchestrator.production_workflow = _FakeProductionWorkflow(calls)
    orchestrator.event_sink = _FakeEventSink(calls)
    compute_calls: list[dict[str, Any]] = []

    def fake_compute_postcalc_result(
        actual_centers_xyz: list[tuple[float, float, float]],
        actual_centers_xyz_id: list[tuple[float, float, float]],
        *,
        concentricity_list: list[float],
        id_single_enable: bool,
    ) -> _FakePostcalcResult:
        compute_calls.append(
            {
                "centers_xyz": actual_centers_xyz,
                "centers_xyz_id": actual_centers_xyz_id,
                "concentricity_list": concentricity_list,
                "id_single_enable": id_single_enable,
            }
        )
        return result

    monkeypatch.setattr(orchestrator_module, "compute_postcalc_result", fake_compute_postcalc_result)

    AutoFlowOrchestrator._run_postcalc_impl(
        orchestrator,
        centers_xyz=centers_xyz,
        centers_xyz_id=centers_xyz_id,
        concentricity_list=concentricity_list,
    )

    assert compute_calls == [
        {
            "centers_xyz": centers_xyz,
            "centers_xyz_id": centers_xyz_id,
            "concentricity_list": concentricity_list,
            "id_single_enable": True,
        }
    ]
    assert compute_calls[0]["centers_xyz"] is centers_xyz
    assert compute_calls[0]["centers_xyz_id"] is centers_xyz_id
    assert compute_calls[0]["concentricity_list"] is concentricity_list
    assert calls == [
        ("record_summary", "straightness", straightness_payload),
        ("publish", "straightness", straightness_payload),
        ("record_summary", "postcalc", postcalc_payload),
        ("publish", "postcalc", postcalc_payload),
    ]
