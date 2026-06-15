from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from core.models import MeasureRow, Recipe
from domain.state import CalibrationSnapshot, RunContext, RunSession
from services.results_service import ResultsService
from services.run_export_coordinator import (
    ExportKind,
    ExportStatus,
    RunExportCoordinator,
)


class _RecordingRepository:
    def __init__(self) -> None:
        self.exported: list[RunContext] = []
        self.daily: list[RunContext] = []
        self.raise_export = False

    def prepare_run(self, recipe_name: str) -> Any:
        raise AssertionError("prepare_run is not part of export coordination")

    def export_run(self, context: RunContext) -> str:
        if self.raise_export:
            raise RuntimeError("disk failed")
        self.exported.append(context)
        return str(Path("/virtual/exports") / context.identity.serial)

    def export_daily_summary(self, context: RunContext) -> None:
        self.daily.append(context)


class _SummaryService:
    def __init__(self) -> None:
        self.raise_summary = False
        self.calls: list[dict[str, Any]] = []

    def compute_run_summary(self, **kwargs: Any) -> dict[str, Any]:
        if self.raise_summary:
            raise RuntimeError("summary failed")
        self.calls.append(dict(kwargs))
        return {"ok": True, "from_summary_service": True}


def _row(idx: int = 1) -> MeasureRow:
    return MeasureRow(
        idx=idx,
        x_ui=float(idx),
        x_abs=float(idx),
        od_avg=100.0,
        od_dev=0.0,
        od_runout=0.0,
        od_round=0.1,
        id_avg=80.0,
        id_dev=0.0,
        id_runout=0.0,
        id_round=0.1,
        concentricity=0.01,
    )


def _session(*, status: str = "DONE", row_count: int = 2) -> RunSession:
    return RunSession(
        serial="serial-001",
        run_id="run-001",
        start_ts=100.0,
        end_ts=120.0,
        status=status,
        rows=[_row(i + 1) for i in range(row_count)],
        raw_points=[{"section_idx": 1, "theta_deg": 0.0}],
        summary_cache={"straight_od": 0.2},
    )


def _make(
    *,
    repo: _RecordingRepository | None = None,
    summary: _SummaryService | None = None,
    recipe: Recipe | None = None,
) -> tuple[RunExportCoordinator, _RecordingRepository, _SummaryService]:
    repository = repo or _RecordingRepository()
    summary_service = summary or _SummaryService()
    coordinator = RunExportCoordinator(
        repository=repository,
        results_service=cast(ResultsService, summary_service),
        recipe_provider=lambda: recipe or Recipe(section_count=2),
        calibration_provider=CalibrationSnapshot,
        coverage_provider=lambda: {1: {"cov": 1.0}},
        clock=lambda: 130.0,
    )
    return coordinator, repository, summary_service


def test_missing_identity_returns_pending_without_repository_call() -> None:
    coordinator, repo, _summary = _make()
    session = RunSession(status="DONE")

    result = coordinator.try_export_terminal_run(session)

    assert result.status is ExportStatus.PENDING
    assert repo.exported == []


def test_done_with_zero_expected_sections_is_pending() -> None:
    coordinator, repo, _summary = _make(recipe=Recipe(section_count=0))

    result = coordinator.try_export_terminal_run(_session(status="DONE", row_count=2))

    assert result.status is ExportStatus.PENDING
    assert repo.exported == []


def test_done_with_incomplete_sections_is_pending() -> None:
    coordinator, repo, _summary = _make(recipe=Recipe(section_count=3))

    result = coordinator.try_export_terminal_run(_session(status="DONE", row_count=2))

    assert result.status is ExportStatus.PENDING
    assert repo.exported == []


def test_done_complete_exports_completed_and_daily_summary() -> None:
    coordinator, repo, summary = _make(recipe=Recipe(section_count=2))

    result = coordinator.try_export_terminal_run(_session(status="DONE", row_count=2))

    assert result.status is ExportStatus.EXPORTED
    assert result.kind is ExportKind.COMPLETED
    assert result.completed is True
    assert len(repo.exported) == 1
    assert len(repo.daily) == 1
    context = repo.exported[0]
    assert context.completed is True
    assert context.completed_sections == 2
    assert context.expected_sections == 2
    assert context.summary["from_summary_service"] is True
    assert summary.calls[0]["summary_cache"] == {"straight_od": 0.2}


def test_repeated_done_is_skipped() -> None:
    coordinator, repo, _summary = _make()
    session = _session(status="DONE", row_count=2)

    assert coordinator.try_export_terminal_run(session).status is ExportStatus.EXPORTED
    result = coordinator.try_export_terminal_run(session)

    assert result.status is ExportStatus.SKIPPED
    assert len(repo.exported) == 1


def test_stop_partial_exports_once_then_skips() -> None:
    coordinator, repo, _summary = _make()
    session = _session(status="STOP", row_count=1)

    first = coordinator.try_export_terminal_run(session)
    second = coordinator.try_export_terminal_run(session)

    assert first.status is ExportStatus.EXPORTED
    assert first.kind is ExportKind.PARTIAL
    assert first.completed is False
    assert second.status is ExportStatus.SKIPPED
    assert len(repo.exported) == 1
    assert repo.exported[0].abort_reason == "user_cancel"


def test_partial_then_done_can_export_completed() -> None:
    coordinator, repo, _summary = _make(recipe=Recipe(section_count=2))
    session = _session(status="STOP", row_count=1)
    assert coordinator.try_export_terminal_run(session).status is ExportStatus.EXPORTED

    session.status = "DONE"
    session.rows.append(_row(2))
    result = coordinator.try_export_terminal_run(session)

    assert result.status is ExportStatus.EXPORTED
    assert result.kind is ExportKind.COMPLETED
    assert [ctx.completed for ctx in repo.exported] == [False, True]


def test_late_postcalc_pending_then_exported_when_rows_complete() -> None:
    coordinator, repo, _summary = _make(recipe=Recipe(section_count=2))
    session = _session(status="DONE", row_count=1)

    pending = coordinator.try_export_terminal_run(session)
    session.rows.append(_row(2))
    exported = coordinator.try_export_terminal_run(session)

    assert pending.status is ExportStatus.PENDING
    assert exported.status is ExportStatus.EXPORTED
    assert len(repo.exported) == 1


def test_failed_export_does_not_mark_exported_key() -> None:
    repo = _RecordingRepository()
    repo.raise_export = True
    coordinator, _repo, _summary = _make(repo=repo)
    session = _session(status="DONE", row_count=2)

    failed = coordinator.try_export_terminal_run(session)
    repo.raise_export = False
    exported = coordinator.try_export_terminal_run(session)

    assert failed.status is ExportStatus.FAILED
    assert exported.status is ExportStatus.EXPORTED
    assert len(repo.exported) == 1


def test_summary_failure_does_not_mark_exported_key() -> None:
    summary = _SummaryService()
    summary.raise_summary = True
    coordinator, repo, _summary = _make(summary=summary)
    session = _session(status="DONE", row_count=2)

    failed = coordinator.try_export_terminal_run(session)
    summary.raise_summary = False
    exported = coordinator.try_export_terminal_run(session)

    assert failed.status is ExportStatus.FAILED
    assert exported.status is ExportStatus.EXPORTED
    assert len(repo.exported) == 1


def test_provider_failure_returns_failed_without_exported_key() -> None:
    repo = _RecordingRepository()
    summary = _SummaryService()
    coordinator = RunExportCoordinator(
        repository=repo,
        results_service=cast(ResultsService, summary),
        recipe_provider=lambda: (_ for _ in ()).throw(RuntimeError("recipe failed")),
        calibration_provider=CalibrationSnapshot,
        coverage_provider=lambda: {},
    )

    result = coordinator.try_export_terminal_run(_session(status="DONE", row_count=2))

    assert result.status is ExportStatus.FAILED
    assert repo.exported == []


def test_manual_export_ignores_auto_export_dedupe() -> None:
    coordinator, repo, _summary = _make()
    session = _session(status="DONE", row_count=2)
    assert coordinator.try_export_terminal_run(session).status is ExportStatus.EXPORTED

    first_manual = coordinator.export_manual(session)
    second_manual = coordinator.export_manual(session)

    assert first_manual.status is ExportStatus.EXPORTED
    assert second_manual.status is ExportStatus.EXPORTED
    assert [ctx.status for ctx in repo.exported] == ["DONE", "DONE", "DONE"]


def test_result_status_can_supply_terminal_done() -> None:
    coordinator, repo, _summary = _make()
    session = _session(status="RUN", row_count=2)
    result_obj = SimpleNamespace(status="DONE", finished_at_ts=125.0)

    result = coordinator.try_export_terminal_run(session, result_obj)

    assert result.status is ExportStatus.EXPORTED
    assert repo.exported[0].finished_at_ts == 120.0
