from __future__ import annotations

from typing import Any, Protocol

from application.controllers.validation_controller import ValidationController, ValidationHostPort


class GaugeHostPort(ValidationHostPort, Protocol):
    calibration_controller: Any

    def apply_plc_connection(self) -> Any: ...

    def toggle_sim_gauge(self) -> Any: ...

    def refresh_gauge_ports(self) -> Any: ...

    def connect_gauge(self) -> Any: ...

    def disconnect_gauge(self) -> Any: ...

    def request_gauge_once(self) -> Any: ...

    def set_gauge_request_command(self, cmd: str) -> str: ...

    def learn_odcal_defect_a(self) -> Any: ...

    def learn_odcal_defect_b(self) -> Any: ...

    def clear_odcal_defect_template(self) -> Any: ...

    def open_validation_screen(self) -> Any: ...


class GaugeController(ValidationController):
    def __init__(self, host: GaugeHostPort) -> None:
        super().__init__(host)
        self._gauge_host = host

    @property
    def _calibration(self) -> Any:
        return self._gauge_host.calibration_controller

    def apply_plc_connection(self) -> Any:
        return self._gauge_host.apply_plc_connection()

    def toggle_sim_gauge(self) -> Any:
        return self._gauge_host.toggle_sim_gauge()

    def refresh_gauge_ports(self) -> Any:
        return self._gauge_host.refresh_gauge_ports()

    def connect_gauge(self) -> Any:
        return self._gauge_host.connect_gauge()

    def disconnect_gauge(self) -> Any:
        return self._gauge_host.disconnect_gauge()

    def request_gauge_once(self) -> Any:
        return self._gauge_host.request_gauge_once()

    def set_gauge_request_command(self, cmd: str) -> str:
        return self._gauge_host.set_gauge_request_command(cmd)

    def learn_odcal_defect_a(self) -> Any:
        return self._gauge_host.learn_odcal_defect_a()

    def learn_odcal_defect_b(self) -> Any:
        return self._gauge_host.learn_odcal_defect_b()

    def clear_odcal_defect_template(self) -> Any:
        return self._gauge_host.clear_odcal_defect_template()

    def open_validation_screen(self) -> Any:
        return self._gauge_host.open_validation_screen()

    def start_od_b_capture(self) -> Any:
        return self._calibration.start_od_b_capture()

    def stop_od_b_capture(self, reason: str = "manual") -> Any:
        return self._calibration.stop_od_b_capture(reason)

    def compute_od_b(self) -> Any:
        return self._calibration.compute_od_b()

    def apply_od_b(self) -> Any:
        return self._calibration.apply_od_b()

    def export_od_b_raw(self) -> Any:
        return self._calibration.export_od_b_raw()

    def clear_od_b_capture(self) -> Any:
        return self._calibration.clear_od_b_capture()

    def start_id_capture(self) -> Any:
        return self._calibration.start_id_capture()

    def stop_id_capture(self) -> Any:
        return self._calibration.stop_id_capture()

    def clear_id_capture(self) -> Any:
        return self._calibration.clear_id_capture()

    def compute_id_calibration(self) -> Any:
        return self._calibration.compute_id_calibration()

    def apply_id_calibration(self) -> Any:
        return self._calibration.apply_id_calibration()

    def export_id_raw(self) -> Any:
        return self._calibration.export_id_raw()

    def verify_id_calibration(self) -> Any:
        return self._calibration.verify_id_calibration()

    def start_id_single_capture(self) -> Any:
        return self._calibration.start_id_single_capture()

    def stop_id_single_capture(self, reason: str = "manual") -> Any:
        return self._calibration.stop_id_single_capture(reason)

    def compute_and_write_id_single_calibration(self) -> Any:
        return self._calibration.compute_and_write_id_single_calibration()


__all__ = ["GaugeController", "GaugeHostPort"]
