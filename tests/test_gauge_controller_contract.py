from __future__ import annotations

from typing import Any

from application.controllers.gauge_controller import GaugeController


class _FakeCalibrationController:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def start_od_b_capture(self) -> str:
        self.calls.append(("start_od_b_capture",))
        return "start-od"

    def stop_od_b_capture(self, reason: str = "manual") -> str:
        self.calls.append(("stop_od_b_capture", reason))
        return "stop-od"

    def compute_od_b(self) -> None:
        self.calls.append(("compute_od_b",))

    def apply_od_b(self) -> None:
        self.calls.append(("apply_od_b",))

    def export_od_b_raw(self) -> None:
        self.calls.append(("export_od_b_raw",))

    def clear_od_b_capture(self) -> None:
        self.calls.append(("clear_od_b_capture",))

    def start_id_capture(self) -> None:
        self.calls.append(("start_id_capture",))

    def stop_id_capture(self) -> None:
        self.calls.append(("stop_id_capture",))

    def clear_id_capture(self) -> None:
        self.calls.append(("clear_id_capture",))

    def compute_id_calibration(self) -> None:
        self.calls.append(("compute_id_calibration",))

    def apply_id_calibration(self) -> None:
        self.calls.append(("apply_id_calibration",))

    def export_id_raw(self) -> None:
        self.calls.append(("export_id_raw",))

    def verify_id_calibration(self) -> None:
        self.calls.append(("verify_id_calibration",))

    def start_id_single_capture(self) -> None:
        self.calls.append(("start_id_single_capture",))

    def stop_id_single_capture(self, reason: str = "manual") -> None:
        self.calls.append(("stop_id_single_capture", reason))

    def compute_and_write_id_single_calibration(self) -> None:
        self.calls.append(("compute_and_write_id_single_calibration",))


class _FakeGaugeHost:
    def __init__(self) -> None:
        self.calibration_controller = _FakeCalibrationController()
        self.calls: list[tuple[Any, ...]] = []

    def apply_plc_connection(self) -> None:
        self.calls.append(("apply_plc_connection",))

    def toggle_sim_gauge(self) -> None:
        self.calls.append(("toggle_sim_gauge",))

    def refresh_gauge_ports(self) -> None:
        self.calls.append(("refresh_gauge_ports",))

    def connect_gauge(self) -> None:
        self.calls.append(("connect_gauge",))

    def disconnect_gauge(self) -> None:
        self.calls.append(("disconnect_gauge",))

    def request_gauge_once(self) -> None:
        self.calls.append(("request_gauge_once",))

    def set_gauge_request_command(self, cmd: str) -> str:
        self.calls.append(("set_gauge_request_command", cmd))
        return cmd

    def learn_odcal_defect_a(self) -> None:
        self.calls.append(("learn_odcal_defect_a",))

    def learn_odcal_defect_b(self) -> None:
        self.calls.append(("learn_odcal_defect_b",))

    def clear_odcal_defect_template(self) -> None:
        self.calls.append(("clear_odcal_defect_template",))

    def open_validation_screen(self) -> None:
        self.calls.append(("open_validation_screen",))

    def list_validation_section_choices(self) -> list[str]:
        self.calls.append(("list_validation_section_choices",))
        return ["1"]

    def start_validation_run(self, **kwargs: Any) -> str:
        self.calls.append(("start_validation_run", dict(kwargs)))
        return "validation-started"

    def stop_validation_run(self) -> None:
        self.calls.append(("stop_validation_run",))


def test_gauge_controller_delegates_gauge_commands() -> None:
    host = _FakeGaugeHost()
    controller = GaugeController(host)

    controller.apply_plc_connection()
    controller.toggle_sim_gauge()
    controller.refresh_gauge_ports()
    controller.connect_gauge()
    controller.disconnect_gauge()
    controller.request_gauge_once()
    assert controller.set_gauge_request_command("M0,1") == "M0,1"
    controller.learn_odcal_defect_a()
    controller.learn_odcal_defect_b()
    controller.clear_odcal_defect_template()
    controller.open_validation_screen()

    assert host.calls == [
        ("apply_plc_connection",),
        ("toggle_sim_gauge",),
        ("refresh_gauge_ports",),
        ("connect_gauge",),
        ("disconnect_gauge",),
        ("request_gauge_once",),
        ("set_gauge_request_command", "M0,1"),
        ("learn_odcal_defect_a",),
        ("learn_odcal_defect_b",),
        ("clear_odcal_defect_template",),
        ("open_validation_screen",),
    ]


def test_gauge_controller_delegates_calibration_commands() -> None:
    host = _FakeGaugeHost()
    controller = GaugeController(host)

    assert controller.start_od_b_capture() == "start-od"
    assert controller.stop_od_b_capture("manual") == "stop-od"
    controller.compute_od_b()
    controller.apply_od_b()
    controller.export_od_b_raw()
    controller.clear_od_b_capture()
    controller.start_id_capture()
    controller.stop_id_capture()
    controller.clear_id_capture()
    controller.compute_id_calibration()
    controller.apply_id_calibration()
    controller.export_id_raw()
    controller.verify_id_calibration()
    controller.start_id_single_capture()
    controller.stop_id_single_capture("manual")
    controller.compute_and_write_id_single_calibration()

    assert host.calibration_controller.calls == [
        ("start_od_b_capture",),
        ("stop_od_b_capture", "manual"),
        ("compute_od_b",),
        ("apply_od_b",),
        ("export_od_b_raw",),
        ("clear_od_b_capture",),
        ("start_id_capture",),
        ("stop_id_capture",),
        ("clear_id_capture",),
        ("compute_id_calibration",),
        ("apply_id_calibration",),
        ("export_id_raw",),
        ("verify_id_calibration",),
        ("start_id_single_capture",),
        ("stop_id_single_capture", "manual"),
        ("compute_and_write_id_single_calibration",),
    ]
