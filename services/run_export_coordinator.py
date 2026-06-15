from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from core.models import Recipe
from domain.protocols import RunRepositoryProtocol
from domain.state import CalibrationSnapshot, RunContext, RunIdentity, RunSession
from services.results_service import ResultsService

RecipeProvider = Callable[[], Recipe]
CalibrationProvider = Callable[[], CalibrationSnapshot]
CoverageProvider = Callable[[], Mapping[int, Mapping[str, Any]]]


class ExportStatus(str, Enum):
    EXPORTED = "exported"
    SKIPPED = "skipped"
    PENDING = "pending"
    FAILED = "failed"


class ExportKind(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class ExportResult:
    status: ExportStatus
    kind: ExportKind
    completed: bool
    message: str
    path: Path | None = None
    error: str | None = None


class RunExportCoordinator:
    """Coordinates production run export decisions outside of AppHost."""

    _PARTIAL_STATUSES = {"STOP", "ERR", "ABORTED"}

    def __init__(
        self,
        *,
        repository: RunRepositoryProtocol | Callable[[], RunRepositoryProtocol],
        results_service: ResultsService,
        recipe_provider: RecipeProvider,
        calibration_provider: CalibrationProvider,
        coverage_provider: CoverageProvider,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._repository = repository
        self._results_service = results_service
        self._recipe_provider = recipe_provider
        self._calibration_provider = calibration_provider
        self._coverage_provider = coverage_provider
        self._clock = clock
        self._exported: set[tuple[str, ExportKind]] = set()

    def can_export(self, session: RunSession) -> bool:
        return self._has_identity(session)

    def try_export_terminal_run(
        self,
        session: RunSession,
        result: Any | None = None,
    ) -> ExportResult:
        status = self._terminal_status(session, result)
        if status == "DONE":
            return self._try_export(session, result, kind=ExportKind.COMPLETED, status=status)
        if status in self._PARTIAL_STATUSES:
            return self._try_export(session, result, kind=ExportKind.PARTIAL, status=status)
        return ExportResult(
            status=ExportStatus.PENDING,
            kind=ExportKind.PARTIAL,
            completed=False,
            message=f"run is not terminal: {status or 'UNKNOWN'}",
        )

    def export_manual(self, session: RunSession) -> ExportResult:
        status = self._session_status(session)
        completed = status == "DONE"
        if not self._has_identity(session):
            return self._pending(ExportKind.MANUAL, completed, "missing run identity")
        if not self._has_any_export_data(session):
            return self._pending(ExportKind.MANUAL, completed, "no exportable run data")
        try:
            context = self._build_context(
                session=session,
                result=None,
                status=status or "MANUAL",
                kind=ExportKind.MANUAL,
                completed=completed,
            )
            path = Path(self._repo().export_run(context))
            return ExportResult(
                status=ExportStatus.EXPORTED,
                kind=ExportKind.MANUAL,
                completed=completed,
                message=f"exported: {path}",
                path=path,
            )
        except Exception as exc:
            return self._failed(ExportKind.MANUAL, completed, exc)

    def _try_export(
        self,
        session: RunSession,
        result: Any | None,
        *,
        kind: ExportKind,
        status: str,
    ) -> ExportResult:
        completed = kind is ExportKind.COMPLETED
        if not self._has_identity(session):
            return self._pending(kind, completed, "missing run identity")

        run_id = str(session.run_id or "")
        export_key = (run_id, kind)
        if export_key in self._exported:
            return ExportResult(
                status=ExportStatus.SKIPPED,
                kind=kind,
                completed=completed,
                message=f"{kind.value} export already completed for run_id={run_id}",
            )

        try:
            recipe = self._recipe_provider()
            expected_sections = self._expected_sections(recipe)
            completed_sections = len(list(session.rows or []))

            if kind is ExportKind.COMPLETED:
                if expected_sections <= 0:
                    return self._pending(kind, completed, "expected section count is zero")
                if completed_sections < expected_sections:
                    return self._pending(
                        kind,
                        completed,
                        f"completed sections {completed_sections} < expected {expected_sections}",
                    )
            elif not self._has_any_export_data(session):
                return self._pending(kind, completed, "no exportable partial data")

            context = self._build_context(
                session=session,
                result=result,
                status=status,
                kind=kind,
                completed=completed,
                recipe=recipe,
                expected_sections=expected_sections,
                completed_sections=completed_sections,
            )
            repo = self._repo()
            path = Path(repo.export_run(context))
            if kind is ExportKind.COMPLETED:
                repo.export_daily_summary(context)
            self._exported.add(export_key)
            return ExportResult(
                status=ExportStatus.EXPORTED,
                kind=kind,
                completed=completed,
                message=f"exported: {path}",
                path=path,
            )
        except Exception as exc:
            return self._failed(kind, completed, exc)

    def _build_context(
        self,
        *,
        session: RunSession,
        result: Any | None,
        status: str,
        kind: ExportKind,
        completed: bool,
        recipe: Recipe | None = None,
        expected_sections: int | None = None,
        completed_sections: int | None = None,
    ) -> RunContext:
        if session.start_ts is None:
            raise ValueError("missing run start timestamp")
        recipe = recipe if recipe is not None else self._recipe_provider()
        expected = self._expected_sections(recipe) if expected_sections is None else int(expected_sections)
        completed_count = len(list(session.rows or [])) if completed_sections is None else int(completed_sections)
        summary = self._results_service.compute_run_summary(
            recipe=recipe,
            rows=list(session.rows or []),
            raw_points=list(session.raw_points or []),
            summary_cache=dict(session.summary_cache or {}),
        )
        summary = dict(summary or {})
        summary["completed"] = bool(completed)
        summary["abort_reason"] = self._abort_reason(status, kind)
        summary["completed_sections"] = int(completed_count)
        summary["expected_sections"] = int(expected)

        return RunContext(
            identity=RunIdentity(
                serial=str(session.serial),
                run_id=str(session.run_id),
                started_at_ts=float(session.start_ts),
            ),
            recipe=recipe,
            calibration=self._calibration_provider(),
            rows=list(session.rows or []),
            raw_points=list(session.raw_points or []),
            section_coverage={int(k): dict(v) for k, v in dict(self._coverage_provider() or {}).items()},
            length_result=dict(session.length_result or {}) if isinstance(session.length_result, dict) else None,
            summary=summary,
            finished_at_ts=self._finished_at(session, result),
            status=str(status or ""),
            completed=bool(completed),
            abort_reason=self._abort_reason(status, kind),
            completed_sections=int(completed_count),
            expected_sections=int(expected),
        )

    def _repo(self) -> RunRepositoryProtocol:
        return self._repository() if callable(self._repository) else self._repository

    def _has_identity(self, session: RunSession) -> bool:
        return bool(session.serial and session.run_id and session.start_ts is not None)

    def _has_any_export_data(self, session: RunSession) -> bool:
        return bool(
            list(session.rows or [])
            or list(session.raw_points or [])
            or isinstance(session.length_result, dict)
            or dict(session.summary_cache or {})
        )

    def _session_status(self, session: RunSession) -> str:
        return str(session.status or "").strip().upper()

    def _terminal_status(self, session: RunSession, result: Any | None) -> str:
        result_status = getattr(result, "status", None) if result is not None else None
        return str(result_status or session.status or "").strip().upper()

    def _expected_sections(self, recipe: Recipe) -> int:
        try:
            return int(getattr(recipe, "section_count", 0) or 0)
        except Exception:
            return 0

    def _finished_at(self, session: RunSession, result: Any | None) -> float:
        result_finished = getattr(result, "finished_at_ts", None) if result is not None else None
        value = session.end_ts if session.end_ts is not None else result_finished
        return float(value if value is not None else self._clock())

    def _abort_reason(self, status: str, kind: ExportKind) -> str | None:
        if kind is ExportKind.COMPLETED:
            return None
        st = str(status or "").upper()
        if st == "ERR":
            return "error"
        if st == "STOP":
            return "user_cancel"
        if st == "ABORTED":
            return "aborted"
        if kind is ExportKind.MANUAL:
            return None
        return st.lower() or None

    def _pending(self, kind: ExportKind, completed: bool, message: str) -> ExportResult:
        return ExportResult(
            status=ExportStatus.PENDING,
            kind=kind,
            completed=completed,
            message=message,
        )

    def _failed(self, kind: ExportKind, completed: bool, exc: Exception) -> ExportResult:
        return ExportResult(
            status=ExportStatus.FAILED,
            kind=kind,
            completed=completed,
            message=f"export failed: {exc}",
            error=str(exc),
        )


__all__ = [
    "CalibrationProvider",
    "CoverageProvider",
    "ExportKind",
    "ExportResult",
    "ExportStatus",
    "RecipeProvider",
    "RunExportCoordinator",
]
