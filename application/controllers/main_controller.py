from __future__ import annotations

from typing import Any, Protocol


class MainHostPort(Protocol):
    def start_measurement(self) -> Any: ...

    def stop_measurement(self) -> Any: ...

    def clear_measurement_results(self) -> Any: ...

    def export_history_results(self) -> Any: ...

    def open_serial_template_settings(self) -> Any: ...

    def handle_main_result_selection(self, event: Any = None) -> Any: ...

    def refresh_main_summary_panel(self) -> Any: ...


class MainController:
    def __init__(self, host: MainHostPort) -> None:
        self._host = host

    def start_measurement(self) -> Any:
        return self._host.start_measurement()

    def stop_measurement(self) -> Any:
        return self._host.stop_measurement()

    def clear_measurement_results(self) -> Any:
        return self._host.clear_measurement_results()

    def export_history_results(self) -> Any:
        return self._host.export_history_results()

    def open_serial_template_settings(self) -> Any:
        return self._host.open_serial_template_settings()

    def handle_main_result_selection(self, event: Any = None) -> Any:
        return self._host.handle_main_result_selection(event)

    def refresh_main_summary_panel(self) -> Any:
        return self._host.refresh_main_summary_panel()


__all__ = ["MainController", "MainHostPort"]
