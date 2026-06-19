from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

import frp_workflow.autoflow_orchestrator as orchestrator_module
from frp_workflow.autoflow_orchestrator import AutoFlowOrchestrator
from frp_workflow.steps.measure_section_context import MeasureSectionContext
from frp_workflow.steps.sampling_result import SamplingResult


@dataclass
class _FakeRecipe:
    section_sampling_mode: str
    split_keep_spinning: bool = True
    split_slip_check: bool = True
    split_slip_max_deg: float = 5.0
    split_omega_cv_max: float = 0.25


@dataclass
class _FakeSample:
    coords_od: np.ndarray
    coords_id: np.ndarray
    raw_od: str
    raw_id: str
    raw_points: list[dict[str, Any]]


class _FakeLegacyFlow:
    def __init__(self, *, od_sample: _FakeSample, id_sample: _FakeSample, sync_sample: _FakeSample) -> None:
        self.od_sample = od_sample
        self.id_sample = id_sample
        self.sync_sample = sync_sample
        self.calls: list[dict[str, Any]] = []
        self.runtime_contexts: list[tuple[Any, Any]] = []

    def set_runtime_context(self, recipe: Any, calibration: Any) -> None:
        self.runtime_contexts.append((recipe, calibration))

    def sample_circle_points_result(
        self,
        recipe: Any,
        *,
        section_idx: int,
        sample_od: bool,
        sample_id: bool,
        phase: str,
    ) -> _FakeSample:
        self.calls.append(
            {
                "recipe": recipe,
                "section_idx": section_idx,
                "sample_od": sample_od,
                "sample_id": sample_id,
                "phase": phase,
            }
        )
        if phase == "OD":
            return self.od_sample
        if phase == "ID":
            return self.id_sample
        if phase == "SYNC":
            return self.sync_sample
        raise AssertionError(f"unexpected phase: {phase}")


def _context() -> MeasureSectionContext:
    return MeasureSectionContext(
        section_index=3,
        z_pos_mm=12.5,
        x_abs=45.5,
        centers_xyz=[],
        centers_xyz_id=[],
        concentricity_list=[],
    )


def _sample(label: str, raw_points: list[dict[str, Any]]) -> _FakeSample:
    return _FakeSample(
        coords_od=np.array([[1.0, 2.0, 3.0]]),
        coords_id=np.array([[4.0, 5.0, 6.0]]),
        raw_od=f"{label}-raw-od",
        raw_id=f"{label}-raw-id",
        raw_points=raw_points,
    )


def _orchestrator_with_fakes(
    *,
    recipe: _FakeRecipe,
    legacy: _FakeLegacyFlow,
    stop_calls: list[str] | None = None,
) -> Any:
    orchestrator = cast(Any, AutoFlowOrchestrator.__new__(AutoFlowOrchestrator))
    orchestrator.recipe = recipe
    orchestrator.calibration = object()
    orchestrator._legacy_flow = legacy
    stop_log = stop_calls if stop_calls is not None else []
    orchestrator._stop_ax3_rotation = lambda: stop_log.append("stop")
    orchestrator._start_ax3_rotation = lambda *, emit_state=True: stop_log.append(f"start:{emit_state}")
    return orchestrator


def test_sample_section_impl_characterizes_split_path(monkeypatch: pytest.MonkeyPatch) -> None:
    od_points = [{"phase": "od", "theta_deg": 0.0}]
    id_points = [{"phase": "id", "theta_deg": 180.0}]
    od_sample = _sample("od", od_points)
    id_sample = _sample("id", id_points)
    sync_sample = _sample("sync", [{"phase": "sync"}])
    legacy = _FakeLegacyFlow(od_sample=od_sample, id_sample=id_sample, sync_sample=sync_sample)
    recipe = _FakeRecipe(
        section_sampling_mode="split",
        split_keep_spinning=False,
        split_slip_check=True,
        split_slip_max_deg=7.5,
        split_omega_cv_max=0.33,
    )
    rotation_calls: list[str] = []
    slip_calls: list[dict[str, Any]] = []

    def fake_split_slip_diag(**kwargs: Any) -> tuple[float, bool]:
        slip_calls.append(kwargs)
        return 2.5, True

    monkeypatch.setattr(orchestrator_module, "_split_slip_diag", fake_split_slip_diag)
    orchestrator = _orchestrator_with_fakes(
        recipe=recipe,
        legacy=legacy,
        stop_calls=rotation_calls,
    )

    result = AutoFlowOrchestrator._sample_section_impl(orchestrator, _context())

    assert isinstance(result, SamplingResult)
    assert result.scan_mode == "SPLIT"
    assert result.keep_spinning is False
    assert result.primary_sample is od_sample
    assert result.id_sample is id_sample
    assert result.coords_od is od_sample.coords_od
    assert result.coords_id is id_sample.coords_id
    assert result.raw_od == "od-raw-od"
    assert result.raw_id == "id-raw-id"
    assert result.raw_points == od_points + id_points
    assert result.split_shift_deg == 2.5
    assert result.coax_unreliable is True
    assert [call["phase"] for call in legacy.calls] == ["OD", "ID"]
    assert legacy.calls[0] == {
        "recipe": recipe,
        "section_idx": 2,
        "sample_od": True,
        "sample_id": False,
        "phase": "OD",
    }
    assert legacy.calls[1] == {
        "recipe": recipe,
        "section_idx": 2,
        "sample_od": False,
        "sample_id": True,
        "phase": "ID",
    }
    assert rotation_calls == ["stop", "start:False"]
    assert slip_calls == [
        {
            "raw_points_od": od_points,
            "raw_points_id": id_points,
            "slip_max_deg": 7.5,
            "omega_cv_max": 0.33,
        }
    ]


def test_sample_section_impl_characterizes_split_keep_spinning_true(monkeypatch: pytest.MonkeyPatch) -> None:
    od_points = [{"phase": "od", "theta_deg": 10.0}]
    id_points = [{"phase": "id", "theta_deg": 190.0}]
    od_sample = _sample("od", od_points)
    id_sample = _sample("id", id_points)
    sync_sample = _sample("sync", [{"phase": "sync"}])
    legacy = _FakeLegacyFlow(od_sample=od_sample, id_sample=id_sample, sync_sample=sync_sample)
    recipe = _FakeRecipe(
        section_sampling_mode="split",
        split_keep_spinning=True,
        split_slip_check=True,
        split_slip_max_deg=4.5,
        split_omega_cv_max=0.2,
    )
    rotation_calls: list[str] = []
    slip_calls: list[dict[str, Any]] = []

    def fake_split_slip_diag(**kwargs: Any) -> tuple[float, bool]:
        slip_calls.append(kwargs)
        return 1.25, False

    monkeypatch.setattr(orchestrator_module, "_split_slip_diag", fake_split_slip_diag)
    orchestrator = _orchestrator_with_fakes(
        recipe=recipe,
        legacy=legacy,
        stop_calls=rotation_calls,
    )

    result = AutoFlowOrchestrator._sample_section_impl(orchestrator, _context())

    assert isinstance(result, SamplingResult)
    assert result.scan_mode == "SPLIT"
    assert result.keep_spinning is True
    assert result.primary_sample is od_sample
    assert result.id_sample is id_sample
    assert result.raw_points == od_points + id_points
    assert result.split_shift_deg == 1.25
    assert result.coax_unreliable is False
    assert [call["phase"] for call in legacy.calls] == ["OD", "ID"]
    assert legacy.calls[0] == {
        "recipe": recipe,
        "section_idx": 2,
        "sample_od": True,
        "sample_id": False,
        "phase": "OD",
    }
    assert legacy.calls[1] == {
        "recipe": recipe,
        "section_idx": 2,
        "sample_od": False,
        "sample_id": True,
        "phase": "ID",
    }
    assert rotation_calls == []
    assert slip_calls == [
        {
            "raw_points_od": od_points,
            "raw_points_id": id_points,
            "slip_max_deg": 4.5,
            "omega_cv_max": 0.2,
        }
    ]


def test_sample_section_impl_characterizes_sync_path() -> None:
    od_sample = _sample("od", [{"phase": "od"}])
    id_sample = _sample("id", [{"phase": "id"}])
    sync_points = [{"phase": "sync", "theta_deg": 90.0}]
    sync_sample = _sample("sync", sync_points)
    legacy = _FakeLegacyFlow(od_sample=od_sample, id_sample=id_sample, sync_sample=sync_sample)
    recipe = _FakeRecipe(section_sampling_mode="sync", split_keep_spinning=True)
    rotation_calls: list[str] = []
    orchestrator = _orchestrator_with_fakes(
        recipe=recipe,
        legacy=legacy,
        stop_calls=rotation_calls,
    )

    result = AutoFlowOrchestrator._sample_section_impl(orchestrator, _context())

    assert isinstance(result, SamplingResult)
    assert result.scan_mode == "SYNC"
    assert result.keep_spinning is True
    assert result.primary_sample is sync_sample
    assert result.id_sample is None
    assert result.coords_od is sync_sample.coords_od
    assert result.coords_id is sync_sample.coords_id
    assert result.raw_od == "sync-raw-od"
    assert result.raw_id == "sync-raw-id"
    assert result.raw_points is sync_points
    assert result.split_shift_deg is None
    assert result.coax_unreliable is None
    assert legacy.calls == [
        {
            "recipe": recipe,
            "section_idx": 2,
            "sample_od": True,
            "sample_id": True,
            "phase": "SYNC",
        }
    ]
    assert rotation_calls == []
