from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable


logger = logging.getLogger("frp.app.mode")

MeasurementAction = Callable[[], Any]
RunnerGetter = Callable[[], Any | None]
NotificationAction = Callable[[], Any]


class ProductionModeState(StrEnum):
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass(slots=True)
class ProductionMode:
    """Application-level production measurement mode.

    This wrapper keeps the explicit mode/state model outside the controller,
    while continuing to reuse the current legacy start/stop implementations.
    """

    start_impl: MeasurementAction
    stop_impl: MeasurementAction
    runner_getter: RunnerGetter
    already_running_handler: NotificationAction | None = None
    state: ProductionModeState = field(default=ProductionModeState.IDLE, init=False)
    last_workflow_state: str = field(default="IDLE", init=False)
    last_error: str | None = field(default=None, init=False)

    @property
    def state_name(self) -> str:
        return str(self.state.value)

    def start(self) -> Any:
        if self._runner_is_alive():
            self._set_state(ProductionModeState.RUNNING)
            if self.already_running_handler is not None:
                try:
                    self.already_running_handler()
                except Exception:
                    logger.exception("PRODUCTION_MODE_ALREADY_RUNNING_HANDLER_FAILED")
            return None

        self.last_error = None
        self._set_state(ProductionModeState.PREPARING)
        result = self.start_impl()
        if not self._runner_is_alive() and self.state == ProductionModeState.PREPARING:
            self.last_error = self.last_error or "Measurement runner did not start"
            self._set_state(ProductionModeState.ERROR)
        return result

    def stop(self) -> Any:
        if not self._runner_is_alive():
            if self.state == ProductionModeState.STOPPING:
                self._set_state(ProductionModeState.IDLE)
            return None

        self._set_state(ProductionModeState.STOPPING)
        return self.stop_impl()

    def sync_from_workflow_state(self, workflow_state: str, message: str = "") -> ProductionModeState:
        normalized = str(workflow_state or "IDLE").upper()
        self.last_workflow_state = normalized

        if normalized in {"PREP", "LEN"}:
            self._set_state(ProductionModeState.PREPARING)
        elif normalized == "RUN":
            self._set_state(ProductionModeState.RUNNING)
        elif normalized == "STOPPING":
            self._set_state(ProductionModeState.STOPPING)
        elif normalized == "DONE":
            self._set_state(ProductionModeState.COMPLETED)
        elif normalized == "ERR":
            self.last_error = str(message or "Workflow error")
            self._set_state(ProductionModeState.ERROR)
        elif normalized == "STOP":
            self._set_state(ProductionModeState.IDLE)
        elif normalized == "IDLE":
            self._set_state(ProductionModeState.IDLE)
        elif normalized == "WARN":
            logger.warning("PRODUCTION_MODE_WORKFLOW_WARN message=%s", str(message))
        return self.state

    def reset(self) -> ProductionModeState:
        self.last_error = None
        self.last_workflow_state = "IDLE"
        self._set_state(ProductionModeState.IDLE)
        return self.state

    def _runner_is_alive(self) -> bool:
        runner = self.runner_getter()
        return bool(runner and getattr(runner, "is_alive", lambda: False)())

    def _set_state(self, state: ProductionModeState) -> None:
        if self.state == state:
            return
        previous = self.state
        self.state = state
        logger.info(
            "PRODUCTION_MODE_STATE from=%s to=%s workflow_state=%s",
            previous.value,
            state.value,
            self.last_workflow_state,
        )


__all__ = ["ProductionMode", "ProductionModeState"]
