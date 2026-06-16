from __future__ import annotations

"""Small action protocols and callback adapters used by UI event handlers."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from core.models import MeasureRow
from events.types import PlcOkEvent


class RunViewActions(Protocol):
    def set_auto_progress(self, idx: int, total: int) -> None: ...
    def set_auto_done(self, completed: bool) -> None: ...
    def project_auto_len_result(self, payload: dict[str, Any]) -> None: ...
    def show_section_coverage(self, info: dict[str, Any]) -> None: ...
    def set_auto_state(self, state: str, message: str) -> None: ...
    def refresh_done_run_summary(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CallbackRunViewActions:
    set_auto_progress_cb: Callable[[int, int], None]
    set_auto_done_cb: Callable[[bool], None]
    project_auto_len_result_cb: Callable[[dict[str, Any]], None]
    show_section_coverage_cb: Callable[[dict[str, Any]], None]
    set_auto_state_cb: Callable[[str, str], None]
    refresh_done_run_summary_cb: Callable[[], None]

    def set_auto_progress(self, idx: int, total: int) -> None:
        self.set_auto_progress_cb(idx, total)

    def set_auto_done(self, completed: bool) -> None:
        self.set_auto_done_cb(completed)

    def project_auto_len_result(self, payload: dict[str, Any]) -> None:
        self.project_auto_len_result_cb(payload)

    def show_section_coverage(self, info: dict[str, Any]) -> None:
        self.show_section_coverage_cb(info)

    def set_auto_state(self, state: str, message: str) -> None:
        self.set_auto_state_cb(state, message)

    def refresh_done_run_summary(self) -> None:
        self.refresh_done_run_summary_cb()


class RunStateActions(Protocol):
    def set_current_section_index(self, section_index: int) -> None: ...
    def cache_auto_len_result(self, payload: Any) -> dict[str, Any]: ...
    def cache_section_coverage(self, payload: Any) -> tuple[int | None, dict[str, Any]]: ...
    def should_show_section_coverage(self, section_index: int | None) -> bool: ...
    def update_run_status(self, state: str, message: str) -> None: ...
    def freeze_run_end_ts_if_missing(self) -> None: ...
    def append_result_row(self, row: MeasureRow) -> None: ...
    def apply_postcalc_result(self, payload: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class CallbackRunStateActions:
    set_current_section_index_cb: Callable[[int], None]
    cache_auto_len_result_cb: Callable[[Any], dict[str, Any]]
    cache_section_coverage_cb: Callable[[Any], tuple[int | None, dict[str, Any]]]
    should_show_section_coverage_cb: Callable[[int | None], bool]
    update_run_status_cb: Callable[[str, str], None]
    freeze_run_end_ts_if_missing_cb: Callable[[], None]
    append_result_row_cb: Callable[[MeasureRow], None]
    apply_postcalc_result_cb: Callable[[Any], None]

    def set_current_section_index(self, section_index: int) -> None:
        self.set_current_section_index_cb(section_index)

    def cache_auto_len_result(self, payload: Any) -> dict[str, Any]:
        return self.cache_auto_len_result_cb(payload)

    def cache_section_coverage(self, payload: Any) -> tuple[int | None, dict[str, Any]]:
        return self.cache_section_coverage_cb(payload)

    def should_show_section_coverage(self, section_index: int | None) -> bool:
        return self.should_show_section_coverage_cb(section_index)

    def update_run_status(self, state: str, message: str) -> None:
        self.update_run_status_cb(state, message)

    def freeze_run_end_ts_if_missing(self) -> None:
        self.freeze_run_end_ts_if_missing_cb()

    def append_result_row(self, row: MeasureRow) -> None:
        self.append_result_row_cb(row)

    def apply_postcalc_result(self, payload: Any) -> None:
        self.apply_postcalc_result_cb(payload)


class WorkflowStatusActions(Protocol):
    def sync_production_workflow_state(self, state: str, message: str) -> None: ...
    def refresh_stack_light_for_state(self, state: str | None = None) -> None: ...


@dataclass(frozen=True, slots=True)
class CallbackWorkflowStatusActions:
    sync_production_workflow_state_cb: Callable[[str, str], None]
    refresh_stack_light_for_state_cb: Callable[[str | None], None]

    def sync_production_workflow_state(self, state: str, message: str) -> None:
        self.sync_production_workflow_state_cb(state, message)

    def refresh_stack_light_for_state(self, state: str | None = None) -> None:
        self.refresh_stack_light_for_state_cb(state)


class ExportActions(Protocol):
    def trigger_terminal_export(self, status: str, completed: bool) -> None: ...
    def maybe_retry_terminal_export(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CallbackExportActions:
    trigger_terminal_export_cb: Callable[[str, bool], None]
    maybe_retry_terminal_export_cb: Callable[[], None]

    def trigger_terminal_export(self, status: str, completed: bool) -> None:
        self.trigger_terminal_export_cb(status, completed)

    def maybe_retry_terminal_export(self) -> None:
        self.maybe_retry_terminal_export_cb()


class DeviceStateActions(Protocol):
    def set_plc_ok_status(self) -> None: ...
    def set_plc_error_status(
        self, err: str, retry: int | None, max_attempts: int | None, backoff_s: float | None
    ) -> None: ...
    def update_axis_snapshot_from_plc(self, event: PlcOkEvent) -> None: ...
    def update_cl_cache_and_ui(self, event: PlcOkEvent) -> None: ...
    def set_gauge_error(self, message: str) -> None: ...


@dataclass(frozen=True, slots=True)
class CallbackDeviceStateActions:
    set_plc_ok_status_cb: Callable[[], None]
    set_plc_error_status_cb: Callable[[str, int | None, int | None, float | None], None]
    update_axis_snapshot_from_plc_cb: Callable[[PlcOkEvent], None]
    update_cl_cache_and_ui_cb: Callable[[PlcOkEvent], None]
    set_gauge_error_cb: Callable[[str], None]

    def set_plc_ok_status(self) -> None:
        self.set_plc_ok_status_cb()

    def set_plc_error_status(
        self, err: str, retry: int | None, max_attempts: int | None, backoff_s: float | None
    ) -> None:
        self.set_plc_error_status_cb(err, retry, max_attempts, backoff_s)

    def update_axis_snapshot_from_plc(self, event: PlcOkEvent) -> None:
        self.update_axis_snapshot_from_plc_cb(event)

    def update_cl_cache_and_ui(self, event: PlcOkEvent) -> None:
        self.update_cl_cache_and_ui_cb(event)

    def set_gauge_error(self, message: str) -> None:
        self.set_gauge_error_cb(message)


class AxisViewActions(Protocol):
    def update_keytest_from_plc(self, event: PlcOkEvent) -> None: ...
    def refresh_axis_panel_from_snapshot(self) -> None: ...
    def handle_axis_cal_one_shot_read(self) -> None: ...
    def refresh_axis_cal_status(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CallbackAxisViewActions:
    update_keytest_from_plc_cb: Callable[[PlcOkEvent], None]
    refresh_axis_panel_from_snapshot_cb: Callable[[], None]
    handle_axis_cal_one_shot_read_cb: Callable[[], None]
    refresh_axis_cal_status_cb: Callable[[], None]

    def update_keytest_from_plc(self, event: PlcOkEvent) -> None:
        self.update_keytest_from_plc_cb(event)

    def refresh_axis_panel_from_snapshot(self) -> None:
        self.refresh_axis_panel_from_snapshot_cb()

    def handle_axis_cal_one_shot_read(self) -> None:
        self.handle_axis_cal_one_shot_read_cb()

    def refresh_axis_cal_status(self) -> None:
        self.refresh_axis_cal_status_cb()


__all__ = [
    "AxisViewActions",
    "CallbackAxisViewActions",
    "CallbackDeviceStateActions",
    "CallbackExportActions",
    "CallbackRunStateActions",
    "CallbackRunViewActions",
    "CallbackWorkflowStatusActions",
    "DeviceStateActions",
    "ExportActions",
    "RunStateActions",
    "RunViewActions",
    "WorkflowStatusActions",
]
