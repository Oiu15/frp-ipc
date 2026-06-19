from __future__ import annotations

import types
from typing import Any, cast

from domain.state import RunSession
from tests.fakes import FakeVar

from ui.presenters.gauge_presenter import GaugeScreenPresenter
from application.app_host import AppHost


class _FakeModeMachine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def sync_production_workflow_state(self, state: str, message: str) -> None:
        self.calls.append((state, message))


class _FakeHost:
    def __init__(self) -> None:
        self.gauge_conn_var = FakeVar("")
        self.gauge_err_var = FakeVar("")
        self.plc_status_var = FakeVar("")
        self.auto_progress_var = FakeVar("")
        self.auto_done_var = FakeVar("")
        self.auto_state_var = FakeVar("IDLE")
        self.auto_msg_var = FakeVar("-")
        self._auto_cur_sec_idx = None
        self._selected_sec_idx = None
        self._run_session = RunSession()
        self.mode_machine = _FakeModeMachine()
        self._trigger_run_export_calls = 0
        self._freeze_run_end_ts_if_missing_calls = 0

    def _trigger_run_export(self, **_kwargs: object) -> None:
        self._trigger_run_export_calls += 1

    def _freeze_run_end_ts_if_missing(self) -> None:
        self._freeze_run_end_ts_if_missing_calls += 1

    def _refresh_stack_light_for_state(self, _state: str | None = None) -> None:
        pass

    def axis_cal_refresh_status(self) -> None:
        pass

    def _format_cov_info(self, info: dict[str, object]) -> str:
        return str(info)

    def _cache_auto_len_result(self, payload: object) -> dict[str, object]:
        return dict(payload) if isinstance(payload, dict) else {}

    def _cache_section_cov_info(self, payload: object):
        return None, dict(payload) if isinstance(payload, dict) else {}

    def _apply_run_summary_payload(self, _payload: object) -> None:
        pass

    def _apply_postcalc_eccentricity(self, _payload: object) -> None:
        pass

    def _refresh_done_run_summary_and_export(self) -> None:
        pass

    def _append_result_row(self, _row: object) -> None:
        pass


def _bind_routing_methods(host: _FakeHost) -> None:
    names = [
        "_get_run_view_actions",
        "_get_run_state_actions",
        "_get_workflow_status_actions",
        "_get_export_actions",
        "_get_device_state_actions",
        "_get_axis_view_actions",
        "_get_auto_progress_event_handler",
        "_get_auto_coverage_event_handler",
        "_get_auto_len_event_handler",
        "_get_auto_state_event_handler",
        "_get_auto_row_event_handler",
        "_get_auto_postcalc_event_handler",
        "_get_gauge_err_event_handler",
        "_get_plc_err_event_handler",
        "_get_plc_ok_event_handler",
        "_set_auto_current_section_index",
        "_set_auto_progress_view",
        "_set_auto_done_view",
        "_project_auto_len_result",
        "_should_show_section_coverage",
        "_show_section_coverage",
        "_set_auto_state_view",
        "_update_run_status_message",
        "_sync_production_workflow_state",
        "_trigger_terminal_export",
        "_maybe_retry_terminal_export",
        "_apply_postcalc_result_payload",
        "_set_gauge_error",
        "_set_plc_error_status",
        "_set_plc_ok_status",
        "_update_axis_snapshot_from_plc",
        "_update_cl_cache_and_ui",
        "_update_keytest_from_plc",
        "_refresh_axis_panel_from_snapshot",
        "_handle_axis_cal_one_shot_read",
        "_handle_plc_ok_event",
        "_handle_plc_err_event",
        "_handle_plc_giveup_event",
        "_handle_plc_manual_event",
        "_handle_plc_read_event",
        "_handle_gauge_conn_event",
        "_handle_gauge_tx_event",
        "_handle_gauge_ok_event",
        "_handle_gauge_raw_event",
        "_handle_gauge_err_event",
        "_handle_op_confirm_show_event",
        "_handle_op_confirm_close_event",
        "_handle_auto_clear_event",
        "_handle_auto_len_event",
        "_handle_auto_progress_event",
        "_handle_auto_coverage_event",
        "_handle_auto_straightness_event",
        "_handle_auto_postcalc_event",
        "_handle_auto_raw_points_event",
        "_handle_auto_row_event",
        "_handle_auto_state_event",
    ]
    for name in names:
        setattr(host, name, types.MethodType(getattr(AppHost, name), host))

    host._refresh_axis_panel = lambda: None  # type: ignore[attr-defined]


class TestTypedUiEventRouting:
    def test_gauge_conn_event_routes_to_device_handler_and_presenter_state(self) -> None:
        host = _FakeHost()
        _bind_routing_methods(host)
        dispatcher = AppHost._build_device_ui_event_dispatcher(cast(AppHost, host))
        presenter = GaugeScreenPresenter(host, controller=object())

        handled = dispatcher.dispatch("gauge_conn", {"ts": 1.0, "connected": True, "port": "COM3", "baud": 115200})

        assert handled is True
        gauge_err_handler = dispatcher.get_handler("gauge_err")
        assert getattr(gauge_err_handler, "__self__", None) is getattr(host, "_get_gauge_err_event_handler")()
        assert "COM3@115200" in presenter.gauge_conn_var.get()

    def test_auto_progress_event_routes_to_measurement_handler_and_presenter_state(self) -> None:
        host = _FakeHost()
        _bind_routing_methods(host)
        dispatcher = AppHost._build_measurement_ui_event_dispatcher(cast(AppHost, host))

        handled = dispatcher.dispatch("auto_progress", {"idx": 1, "total": 5, "x_ui": 100.0, "x_abs": 200.0})

        assert handled is True
        auto_progress_handler = dispatcher.get_handler("auto_progress")
        assert getattr(auto_progress_handler, "__self__", None) is getattr(host, "_get_auto_progress_event_handler")()
        assert host._auto_cur_sec_idx == 2
        assert "2" in str(host.auto_progress_var.get())
        assert host.auto_done_var.get() is not None

    def test_auto_state_done_routes_to_state_handler_and_done_side_effect(self) -> None:
        host = _FakeHost()
        _bind_routing_methods(host)
        dispatcher = AppHost._build_measurement_ui_event_dispatcher(cast(AppHost, host))

        handled = dispatcher.dispatch("auto_state", {"state": "DONE", "msg": "completed"})

        assert handled is True
        assert host.mode_machine.calls == [("DONE", "completed")]
        assert host.auto_state_var.get() == "DONE"
        assert host.auto_msg_var.get() == "completed"
        assert host._trigger_run_export_calls == 1
        assert host._freeze_run_end_ts_if_missing_calls == 0

    def test_plc_err_event_routes_to_plc_status_presenter(self) -> None:
        host = _FakeHost()
        _bind_routing_methods(host)
        dispatcher = AppHost._build_device_ui_event_dispatcher(cast(AppHost, host))

        handled = dispatcher.dispatch("plc_err", {"err": "connect failed", "retry": 2, "max": 5, "backoff_s": 10.0})

        assert handled is True
        plc_status = str(host.plc_status_var.get())
        assert "connect failed" in plc_status
        assert "2/5" in plc_status
