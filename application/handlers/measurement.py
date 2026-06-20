from __future__ import annotations

"""Measurement typed UI event handlers."""

from dataclasses import dataclass

from application.handlers.actions import ExportActions, RunStateActions, RunViewActions, WorkflowStatusActions
from events.types import (
    AutoCoverageEvent,
    AutoLenEvent,
    AutoPostcalcEvent,
    AutoProgressEvent,
    AutoRowEvent,
    AutoStateEvent,
)


@dataclass(frozen=True, slots=True)
class AutoProgressEventHandler:
    run_state: RunStateActions
    run_view: RunViewActions

    def handle(self, event: AutoProgressEvent) -> None:
        section_index = int(event.idx) + 1
        self.run_state.set_current_section_index(section_index)
        self.run_view.set_auto_progress(int(event.idx), int(event.total))
        self.run_view.set_auto_done(False)


@dataclass(frozen=True, slots=True)
class AutoCoverageEventHandler:
    run_state: RunStateActions
    run_view: RunViewActions

    def handle(self, event: AutoCoverageEvent) -> None:
        section_index, info = self.run_state.cache_section_coverage(event.to_payload())
        if self.run_state.should_show_section_coverage(section_index):
            self.run_view.show_section_coverage(info)


@dataclass(frozen=True, slots=True)
class AutoLenEventHandler:
    run_state: RunStateActions
    run_view: RunViewActions

    def handle(self, event: AutoLenEvent) -> None:
        payload = self.run_state.cache_auto_len_result(event.to_payload())
        self.run_view.project_auto_len_result(payload)


@dataclass(frozen=True, slots=True)
class AutoStateEventHandler:
    run_state: RunStateActions
    run_view: RunViewActions
    workflow_status: WorkflowStatusActions
    export_actions: ExportActions

    def handle(self, event: AutoStateEvent) -> None:
        state = str(event.state or "IDLE")
        message = str(event.msg or "-")
        self.run_state.update_run_status(state, message)
        self.workflow_status.sync_production_workflow_state(state, message)
        self.run_view.set_auto_state(state, message)
        self.workflow_status.refresh_stack_light_for_state(state)

        if state == "DONE":
            self.run_view.set_auto_done(True)
            self.export_actions.trigger_terminal_export(status="DONE", completed=True)
        elif state in {"ERR", "STOP", "ABORTED"}:
            self.run_view.set_auto_done(False)
            self.run_state.freeze_run_end_ts_if_missing()
            self.export_actions.trigger_terminal_export(status=state, completed=False)


@dataclass(frozen=True, slots=True)
class AutoRowEventHandler:
    run_state: RunStateActions

    def handle(self, event: AutoRowEvent) -> None:
        self.run_state.append_result_row(event.row)


@dataclass(frozen=True, slots=True)
class AutoPostcalcEventHandler:
    run_state: RunStateActions
    run_view: RunViewActions
    export_actions: ExportActions

    def handle(self, event: AutoPostcalcEvent) -> None:
        self.run_state.apply_postcalc_result(event.to_payload())
        self.export_actions.maybe_retry_terminal_export()
        self.run_view.refresh_done_run_summary()


__all__ = [
    "AutoCoverageEventHandler",
    "AutoLenEventHandler",
    "AutoPostcalcEventHandler",
    "AutoProgressEventHandler",
    "AutoRowEventHandler",
    "AutoStateEventHandler",
]
