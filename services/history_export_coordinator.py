from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Protocol

from services.history_result_export_service import HistoryExportEntry


class HistoryExportServiceProtocol(Protocol):
    def list_exportable_entries(self) -> list[HistoryExportEntry]: ...

    def export_detection_summary(self, entries: list[HistoryExportEntry], output_path: Path) -> Path: ...


HistoryExportResultQueue = queue.Queue[tuple[str, object]]


class HistoryExportCoordinator:
    """Coordinate non-UI work for manual history exports."""

    def list_exportable_entries(self, service: HistoryExportServiceProtocol) -> list[HistoryExportEntry]:
        return service.list_exportable_entries()

    def start_export(
        self,
        service: HistoryExportServiceProtocol,
        entries: list[HistoryExportEntry],
        output_path: Path,
    ) -> HistoryExportResultQueue:
        result_q: HistoryExportResultQueue = queue.Queue()

        def _worker() -> None:
            try:
                result_q.put(("ok", service.export_detection_summary(entries, output_path)))
            except Exception as exc:
                result_q.put(("error", str(exc)))

        threading.Thread(target=_worker, daemon=True, name="history-result-export").start()
        return result_q


__all__ = [
    "HistoryExportCoordinator",
    "HistoryExportResultQueue",
    "HistoryExportServiceProtocol",
]
