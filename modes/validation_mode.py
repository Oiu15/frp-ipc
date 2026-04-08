from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable


logger = logging.getLogger("frp.app.mode")

ValidationAction = Callable[[], Any]
RunnerGetter = Callable[[], Any | None]
NotificationAction = Callable[[], Any]


class ValidationModeState(StrEnum):
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass(slots=True)
class ValidationMode:
    """Application-level validation mode shell.

    This mirrors the production-mode state model, but intentionally stays
    lightweight until validation workflow behavior is migrated out of the
    legacy host.
    """

    start_impl: ValidationAction | None = None
    stop_impl: ValidationAction | None = None
    runner_getter: RunnerGetter | None = None
    already_running_handler: NotificationAction | None = None
    state: ValidationModeState = field(default=ValidationModeState.IDLE, init=False)
    last_workflow_state: str = field(default="IDLE", init=False)
    last_error: str | None = field(default=None, init=False)

    @property
    def state_name(self) -> str:
        return str(self.state.value)

    def start(self) -> Any:
        if self._runner_is_alive():
            self._set_state(ValidationModeState.RUNNING)
            if self.already_running_handler is not None:
                try:
                    self.already_running_handler()
                except Exception:
                    logger.exception("VALIDATION_MODE_ALREADY_RUNNING_HANDLER_FAILED")
            return None

        self.last_error = None
        self._set_state(ValidationModeState.PREPARING)
        if self.start_impl is None:
            self.last_error = "Validation start is not wired"
            self._set_state(ValidationModeState.ERROR)
            return None
        result = self.start_impl()
        if not self._runner_is_alive() and self.state == ValidationModeState.PREPARING:
            self.last_error = self.last_error or "Validation runner did not start"
            self._set_state(ValidationModeState.ERROR)
        return result

    def stop(self) -> Any:
        if not self._runner_is_alive():
            if self.state == ValidationModeState.STOPPING:
                self._set_state(ValidationModeState.IDLE)
            return None

        self._set_state(ValidationModeState.STOPPING)
        if self.stop_impl is None:
            self._set_state(ValidationModeState.IDLE)
            return None
        return self.stop_impl()

    def sync_from_workflow_state(self, workflow_state: str, message: str = "") -> ValidationModeState:
        normalized = str(workflow_state or "IDLE").upper()
        self.last_workflow_state = normalized

        if normalized in {"PREP", "LEN"}:
            self._set_state(ValidationModeState.PREPARING)
        elif normalized == "RUN":
            self._set_state(ValidationModeState.RUNNING)
        elif normalized == "STOPPING":
            self._set_state(ValidationModeState.STOPPING)
        elif normalized == "DONE":
            self._set_state(ValidationModeState.COMPLETED)
        elif normalized == "ERR":
            self.last_error = str(message or "Validation workflow error")
            self._set_state(ValidationModeState.ERROR)
        elif normalized in {"STOP", "IDLE"}:
            self._set_state(ValidationModeState.IDLE)
        elif normalized == "WARN":
            logger.warning("VALIDATION_MODE_WORKFLOW_WARN message=%s", str(message))
        return self.state

    def reset(self) -> ValidationModeState:
        self.last_error = None
        self.last_workflow_state = "IDLE"
        self._set_state(ValidationModeState.IDLE)
        return self.state

    def _runner_is_alive(self) -> bool:
        if self.runner_getter is None:
            return False
        runner = self.runner_getter()
        return bool(runner and getattr(runner, "is_alive", lambda: False)())

    def _set_state(self, state: ValidationModeState) -> None:
        if self.state == state:
            return
        previous = self.state
        self.state = state
        logger.info(
            "VALIDATION_MODE_STATE from=%s to=%s workflow_state=%s",
            previous.value,
            state.value,
            self.last_workflow_state,
        )


__all__ = ["ValidationMode", "ValidationModeState"]
