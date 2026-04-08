from __future__ import annotations

"""Workflow-level orchestrator skeleton for the formal measurement flow.

Current scope is intentionally minimal:
- define the constructor dependency boundary
- keep the object graph explicit for future AutoFlow migration

The legacy threaded AutoFlow implementation still owns the real runtime
logic today. This orchestrator exists as the next landing zone so we can
move behavior incrementally without changing dependency signatures again.
"""

from application.contracts import EventSink, MachineGateway
from application.state import CalibrationSnapshot, RunSession
from core.models import Recipe


class AutoFlowOrchestrator:
    """Explicit dependency shell for the formal measurement workflow."""

    def __init__(
        self,
        gateway: MachineGateway,
        recipe: Recipe,
        calibration: CalibrationSnapshot,
        run_session: RunSession,
        event_sink: EventSink,
    ) -> None:
        self.gateway = gateway
        self.recipe = recipe
        self.calibration = calibration
        self.run_session = run_session
        self.event_sink = event_sink

    def run(self) -> None:
        """Workflow entrypoint placeholder for future migration."""
        raise NotImplementedError(
            "AutoFlow orchestration has not been migrated yet; "
            "legacy services.autoflow_service.AutoFlow still owns execution."
        )


__all__ = ["AutoFlowOrchestrator"]
