from __future__ import annotations

import ast
from pathlib import Path
from typing import cast


ROOT = Path(__file__).resolve().parents[1]
DEVICE_GATEWAY = ROOT / "application" / "adapters" / "device_gateway.py"


def _source() -> str:
    return DEVICE_GATEWAY.read_text(encoding="utf-8-sig")


def _module() -> ast.Module:
    return ast.parse(_source())


def _literal_strings(name: str) -> tuple[str, ...]:
    for node in _module().body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    value = ast.literal_eval(node.value)
                    if isinstance(value, set):
                        return tuple(sorted(cast(set[str], value)))
                    if isinstance(value, tuple):
                        return cast(tuple[str, ...], value)
    raise AssertionError(f"missing constant {name}")


def _class_method_names(class_name: str) -> list[str]:
    for node in _module().body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [
                item.name
                for item in node.body
                if isinstance(item, ast.FunctionDef)
            ]
    raise AssertionError(f"missing class {class_name}")


def test_screen_controller_fallback_allowlists_do_not_expand() -> None:
    assert _literal_strings("_SCREEN_CONTROLLER_HOST_CALL_ALLOWLIST") == (
        "apply_id_calibration",
        "apply_od_b",
        "apply_plc_connection",
        "clear_measurement_results",
        "connect_gauge",
        "disconnect_gauge",
        "export_history_results",
        "handle_main_result_selection",
        "learn_odcal_defect_a",
        "learn_odcal_defect_b",
        "list_validation_section_choices",
        "open_serial_template_settings",
        "open_validation_screen",
        "refresh_main_summary_panel",
        "request_gauge_once",
        "set_gauge_request_command",
        "start_measurement",
        "start_validation_run",
        "stop_measurement",
        "stop_validation_run",
        "toggle_sim_gauge",
        "verify_id_calibration",
    )
    assert _literal_strings("_SCREEN_CONTROLLER_HOST_CALL_PREFIX_ALLOWLIST") == (
        "_kv_row",
        "_on_recipe",
        "_on_teach",
        "_recipe",
        "_refresh",
        "_save",
        "_teach",
        "axis_cal_",
        "clear_",
        "compute_",
        "export_",
        "handle_",
        "open_",
        "refresh_",
        "start_",
        "stop_",
        "write_keytest_",
    )


def test_screen_presenter_fallback_allowlists_do_not_expand() -> None:
    assert _literal_strings("_SCREEN_PRESENTER_HOST_ATTR_ALLOWLIST") == (
        "auto_done_var",
        "auto_msg_var",
        "auto_progress_var",
        "auto_state_var",
        "axis_dist_var",
        "axis_span_max_var",
        "conc_max_var",
        "cov_var",
        "id_endoff_var",
        "id_mean_var",
        "id_range_var",
        "id_slope_var",
        "id_tilt_var",
        "len_meas_var",
        "max_id_round_var",
        "max_od_fit_res_var",
        "max_od_pp_rob_var",
        "max_od_pp_var",
        "meas_elapsed_var",
        "meas_seq_var",
        "meas_start_var",
        "od_endoff_var",
        "od_mean_var",
        "od_range_var",
        "od_slope_var",
        "od_tilt_var",
        "pipe_sn_var",
        "plc_status_var",
        "ui_meas_mode_var",
    )
    assert _literal_strings("_SCREEN_PRESENTER_HOST_ATTR_PREFIX_ALLOWLIST") == (
        "axis_cal_",
        "keytest_",
        "validation_",
    )
    assert _literal_strings("_SCREEN_PRESENTER_HOST_CALL_ALLOWLIST") == (
        "list_validation_section_choices",
    )
    assert _literal_strings("_SCREEN_PRESENTER_HOST_CALL_PREFIX_ALLOWLIST") == (
        "_list",
        "_refresh",
    )


def test_screen_ui_context_fallback_allowlists_do_not_expand() -> None:
    assert _literal_strings("_SCREEN_UI_CONTEXT_ATTR_ALLOWLIST") == (
        "app",
        "axis_cal",
        "axis_idx",
        "recipe",
        "root",
        "style",
        "ui",
    )
    assert _literal_strings("_SCREEN_UI_CONTEXT_ATTR_PREFIX_ALLOWLIST") == (
        "axis_",
        "keytest_",
        "validation_",
    )


def test_generic_fallback_classes_do_not_gain_more_getattr_methods() -> None:
    for class_name in ("ScreenController", "ScreenPresenter", "ScreenUiContext"):
        methods = _class_method_names(class_name)
        assert methods.count("__getattr__") == 1
