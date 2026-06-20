from __future__ import annotations

"""Record-row step boundary for one measured section."""

from dataclasses import dataclass
from typing import Any, Protocol


class RecordRowPort(Protocol):
    """Narrow surface that the record-row step needs."""

    def _record_row_impl(self, row: Any) -> None: ...


@dataclass(slots=True)
class RecordRowStep:
    """Record the existing measurement row via the orchestrator boundary."""

    port: RecordRowPort
    name: str = "record_row"

    def execute(self, row: Any) -> None:
        self.port._record_row_impl(row)


__all__ = ["RecordRowPort", "RecordRowStep"]
