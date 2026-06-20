from __future__ import annotations

"""Device typed UI event handlers."""

from dataclasses import dataclass

from application.handlers.actions import AxisViewActions, DeviceStateActions, WorkflowStatusActions
from events.types import GaugeErrEvent, PlcErrEvent, PlcOkEvent


@dataclass(frozen=True, slots=True)
class GaugeErrEventHandler:
    device_state: DeviceStateActions

    def handle(self, event: GaugeErrEvent) -> None:
        self.device_state.set_gauge_error(f"Gauge ERROR: {event.err}")


@dataclass(frozen=True, slots=True)
class PlcErrEventHandler:
    device_state: DeviceStateActions

    def handle(self, event: PlcErrEvent) -> None:
        self.device_state.set_plc_error_status(event.err, event.retry, event.max, event.backoff_s)


@dataclass(frozen=True, slots=True)
class PlcOkEventHandler:
    device_state: DeviceStateActions
    axis_view: AxisViewActions
    workflow_status: WorkflowStatusActions

    def handle(self, event: PlcOkEvent) -> None:
        self.device_state.set_plc_ok_status()
        self.device_state.update_axis_snapshot_from_plc(event)
        self.device_state.update_cl_cache_and_ui(event)
        self.axis_view.update_keytest_from_plc(event)
        self.workflow_status.refresh_stack_light_for_state()
        self.axis_view.handle_axis_cal_one_shot_read()
        self.axis_view.refresh_axis_panel_from_snapshot()
        self.axis_view.refresh_axis_cal_status()


__all__ = ["GaugeErrEventHandler", "PlcErrEventHandler", "PlcOkEventHandler"]
