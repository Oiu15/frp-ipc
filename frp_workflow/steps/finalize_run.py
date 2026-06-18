from __future__ import annotations

"""Finalize-run step — Phase 5 first extraction.

Encapsulates the post-measurement cleanup that happens in the ``finally``
block and trailing lines of ``AutoFlowOrchestrator.run()``:

* stamp session end time
* stop AX3 rotation
* optionally abort motion (user stop)
* optionally return to standby (user stop)
* set internal DONE state
* build run result via production workflow
* emit final state

No Tk, no AppHost, no PLC connection or gauge serial I/O.
"""

from dataclasses import dataclass
from typing import Any, Protocol


class FinalizeRunPort(Protocol):
    """Narrow surface that the finalize step needs from its orchestrator."""

    run_session: Any        # has end_ts
    motion: Any             # has stop(axis), abort_motion()
    _stop_event: Any        # has is_set()
    _return_standby_after_stop: bool
    production_workflow: Any | None
    run_result: Any | None

    def _finalize_run_impl(self, status: str, message: str) -> None: ...
    def _set_internal_state(self, state: str) -> None: ...
    def _emit_state(self, state: str, message: str) -> None: ...
    def _return_to_standby_after_user_stop(self) -> None: ...


@dataclass(slots=True)
class FinalizeRunStep:
    """Post-run finalization step.

    Delegates to ``orchestrator._finalize_run_impl(status, message)``
    which contains the original inline logic extracted from ``run()``.
    """

    orchestrator: FinalizeRunPort
    name: str = "finalize_run"

    def execute(self, status: str, message: str) -> None:
        """Run the finalize phase.

        Args:
            status:  "DONE", "STOP", or "ERR"
            message: human-readable completion message
        """
        self.orchestrator._finalize_run_impl(status, message)


__all__ = ["FinalizeRunPort", "FinalizeRunStep"]
