from __future__ import annotations

"""Finalize-run step - Phase 5 first extraction.

This first step boundary deliberately delegates to the orchestrator's
legacy implementation method. It proves the step/wiring shape without
rewriting the formal measurement cleanup behavior.

No UI host, no PLC connection or gauge serial I/O.
"""

from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias


FinalizeRunStatus: TypeAlias = Literal["DONE", "STOP", "ERR"]


class FinalizeRunPort(Protocol):
    """Narrow surface that the finalize step needs from its orchestrator."""

    def _finalize_run_impl(self, status: FinalizeRunStatus, message: str) -> Any: ...


@dataclass(slots=True)
class FinalizeRunStep:
    """Post-run finalization step."""

    port: FinalizeRunPort
    status: FinalizeRunStatus
    message: str
    name: str = "finalize_run"

    def execute(self) -> Any:
        """Run the finalize phase via the legacy implementation boundary."""
        return self.port._finalize_run_impl(self.status, self.message)


__all__ = ["FinalizeRunPort", "FinalizeRunStatus", "FinalizeRunStep"]
