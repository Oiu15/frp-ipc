from __future__ import annotations

from typing import Any

from application.controllers.main_controller import MainController


class _FakeHost:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def start_measurement(self) -> str:
        self.calls.append(("start_measurement", ()))
        return "start"

    def stop_measurement(self) -> str:
        self.calls.append(("stop_measurement", ()))
        return "stop"

    def clear_measurement_results(self) -> str:
        self.calls.append(("clear_measurement_results", ()))
        return "clear"

    def export_history_results(self) -> str:
        self.calls.append(("export_history_results", ()))
        return "export"

    def open_serial_template_settings(self) -> str:
        self.calls.append(("open_serial_template_settings", ()))
        return "settings"

    def handle_main_result_selection(self, event: Any = None) -> tuple[str, Any]:
        self.calls.append(("handle_main_result_selection", (event,)))
        return ("select", event)

    def refresh_main_summary_panel(self) -> str:
        self.calls.append(("refresh_main_summary_panel", ()))
        return "refresh"


def test_main_controller_delegates_commands_to_host() -> None:
    host = _FakeHost()
    controller = MainController(host)
    event = object()

    assert controller.start_measurement() == "start"
    assert controller.stop_measurement() == "stop"
    assert controller.clear_measurement_results() == "clear"
    assert controller.export_history_results() == "export"
    assert controller.open_serial_template_settings() == "settings"
    assert controller.handle_main_result_selection(event) == ("select", event)
    assert controller.refresh_main_summary_panel() == "refresh"

    assert host.calls == [
        ("start_measurement", ()),
        ("stop_measurement", ()),
        ("clear_measurement_results", ()),
        ("export_history_results", ()),
        ("open_serial_template_settings", ()),
        ("handle_main_result_selection", (event,)),
        ("refresh_main_summary_panel", ()),
    ]
