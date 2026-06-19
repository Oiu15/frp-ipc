from __future__ import annotations

"""Prepare-run-context step - Phase 5 second extraction.

This step delegates the formal measurement run prelude to the orchestrator's
legacy implementation method. It keeps the startup behavior unchanged while
adding a second workflow step boundary.

No UI host, no PLC connection or gauge serial I/O.
"""

from dataclasses import dataclass
from typing import Any, Protocol


class PrepareRunContextPort(Protocol):
    """Narrow surface that the prepare-run-context step needs."""

    def _prepare_run_context_impl(self) -> Any: ...


@dataclass(slots=True)
class PrepareRunContextStep:
    """Prepare the formal measurement run context before the main loop."""

    port: PrepareRunContextPort
    name: str = "prepare_run_context"

    def execute(self) -> Any:
        """Run the prepare phase via the legacy implementation boundary."""
        return self.port._prepare_run_context_impl()


__all__ = ["PrepareRunContextPort", "PrepareRunContextStep"]
