from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8-sig")


def test_main_screen_does_not_add_dynamic_fallback_tokens() -> None:
    source = _read("ui/screens/main_screen.py")

    for forbidden in (
        "getattr(controller",
        "getattr(presenter",
        "getattr(ui",
        "controller._host",
        "presenter._host",
        "ui._host",
    ):
        assert forbidden not in source


def test_main_screen_is_wired_with_explicit_objects() -> None:
    source = _read("application/host/ui.py")

    assert "build_main_screen(tab_main, presenter=self.main_ui, controller=self.main_controller, ui=self.main_ui)" in source
    assert "build_main_screen(tab_main, presenter=self._screen_presenter" not in source
    assert "build_main_screen(tab_main, presenter=self.main_ui, controller=self._screen_controller" not in source
    assert "build_main_screen(tab_main, presenter=self.main_ui, controller=self.main_controller, ui=self._screen_ui_context" not in source
    assert "build_gauge_screen(tab_gauge, presenter=self._gauge_screen_presenter, controller=self.gauge_controller" in source
    assert "build_validation_screen(tab_validation, presenter=self.validation_ui, controller=self.validation_controller" in source
    assert "build_axis_cal_screen(tab_axis_cal, presenter=self.axis_cal_ui, controller=self.axis_cal_controller" in source
    assert "build_key_test_screen(tab_keytest, presenter=self.key_test_ui, controller=self.key_test_controller" in source


def test_main_screen_state_inventory_is_stable() -> None:
    source = _read("ui/screens/main_screen.py")

    presenter_attrs = (
        "pipe_sn_var",
        "meas_seq_var",
        "meas_start_var",
        "meas_elapsed_var",
        "auto_progress_var",
        "auto_done_var",
        "auto_state_var",
        "ui_meas_mode_var",
        "auto_msg_var",
        "od_mean_var",
        "od_range_var",
        "max_od_pp_var",
        "max_od_pp_rob_var",
        "max_od_fit_res_var",
        "od_tilt_var",
        "od_slope_var",
        "od_endoff_var",
        "id_mean_var",
        "id_range_var",
        "id_tilt_var",
        "id_slope_var",
        "id_endoff_var",
        "max_id_round_var",
        "axis_dist_var",
        "conc_max_var",
        "axis_span_max_var",
        "len_meas_var",
        "cov_var",
        "remember_widget",
        "remember_view_state",
    )
    for attr in presenter_attrs:
        assert f"presenter.{attr}" in source


def test_main_screen_command_inventory_is_stable() -> None:
    source = _read("ui/screens/main_screen.py")

    controller_methods = (
        "start_measurement",
        "stop_measurement",
        "clear_measurement_results",
        "export_history_results",
        "open_serial_template_settings",
        "handle_main_result_selection",
        "refresh_main_summary_panel",
    )
    for method in controller_methods:
        assert f"controller.{method}" in source
