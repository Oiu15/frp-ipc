from __future__ import annotations

import ast
from pathlib import Path
import tkinter as tk

from application.app_host import AppHost
from core.models import AxisCal, Recipe
from ui.presenters.gauge_presenter import GaugeScreenPresenter
from ui.presenters.recipe_presenter import RecipeScreenPresenter
from ui.state import UiState


PHASE_7A_DEFAULTS = {
    "pipe_sn_var": "--",
    "meas_seq_var": "--",
    "meas_start_var": "--",
    "meas_elapsed_var": "--",
    "ui_meas_mode_var": "检测模式：--",
    "auto_state_var": "IDLE",
    "auto_msg_var": "-",
    "auto_progress_var": "当前截面: - / 总截面: -",
    "auto_done_var": "测量完成: 否",
    "cov_var": "采样覆盖率：--",
    "straight_var": "直线度   --（外圆） | --（内圆）",
    "straight_od_var": "--",
    "straight_id_var": "--",
    "conc_var": "整体同心度   --",
    "conc_max_var": "--",
    "axis_dist_var": "--",
    "axis_span_max_var": "--",
}

PHASE_7B_VALIDATION_DEFAULTS = {
    "validation_section_name_var": "",
    "validation_metric_name_var": "od_avg",
    "validation_repeat_count_var": "3",
    "validation_reclamp_between_repeats_var": False,
    "validation_reclamp_enabled_var": False,
    "validation_rotation_stop_before_measure_var": False,
    "validation_release_settle_s_var": "0.0",
    "validation_clamp_settle_s_var": "0.0",
    "validation_position_settle_s_var": "0.0",
    "validation_sample_delay_s_var": "0.0",
    "validation_ax3_speed_dps_var": "60.0",
    "validation_move_enabled_var": False,
    "validation_move_channel_var": "od_channel",
    "validation_move_away_delta_mm_var": "0.0",
    "validation_move_scenario_var": "distance_round_trip",
    "validation_move_from_section_var": "1",
    "validation_move_target_section_var": "1",
    "validation_move_return_section_var": "1",
    "validation_move_target_pos_var": "",
    "validation_move_actual_pos_var": "",
    "validation_status_var": "IDLE",
    "validation_phase_var": "IDLE",
    "validation_wait_phase_var": "",
    "validation_wait_remaining_s_var": "",
    "validation_current_repeat_var": "0/0",
    "validation_result_var": "",
    "validation_error_var": "",
    "validation_export_path_var": "",
    "validation_current_metric_value_var": "",
    "validation_current_section_var": "",
    "validation_current_z_pos_var": "",
    "validation_current_concentricity_var": "",
    "validation_summary_count_var": "0",
    "validation_summary_mean_var": "",
    "validation_summary_std_var": "",
    "validation_summary_min_var": "",
    "validation_summary_max_var": "",
    "validation_summary_range_var": "",
}

VALIDATION_ALIASES = {
    "validation_section_name_var": "validation_debug_section_name_var",
    "validation_metric_name_var": "validation_debug_metric_name_var",
    "validation_repeat_count_var": "validation_debug_repeat_count_var",
    "validation_reclamp_between_repeats_var": "validation_debug_reclamp_between_repeats_var",
    "validation_reclamp_enabled_var": "validation_debug_reclamp_enabled_var",
    "validation_rotation_stop_before_measure_var": "validation_debug_rotation_stop_before_measure_var",
    "validation_release_settle_s_var": "validation_debug_release_settle_s_var",
    "validation_clamp_settle_s_var": "validation_debug_clamp_settle_s_var",
    "validation_position_settle_s_var": "validation_debug_position_settle_s_var",
    "validation_sample_delay_s_var": "validation_debug_sample_delay_s_var",
    "validation_ax3_speed_dps_var": "validation_debug_ax3_speed_dps_var",
    "validation_move_enabled_var": "validation_debug_move_enabled_var",
    "validation_move_channel_var": "validation_debug_move_channel_var",
    "validation_move_away_delta_mm_var": "validation_debug_move_away_delta_mm_var",
    "validation_move_scenario_var": "validation_debug_move_scenario_var",
    "validation_move_from_section_var": "validation_debug_move_from_section_var",
    "validation_move_target_section_var": "validation_debug_move_target_section_var",
    "validation_move_return_section_var": "validation_debug_move_return_section_var",
    "validation_move_target_pos_var": "validation_debug_move_target_pos_var",
    "validation_move_actual_pos_var": "validation_debug_move_actual_pos_var",
    "validation_status_var": "validation_debug_status_var",
    "validation_phase_var": "validation_debug_phase_var",
    "validation_wait_phase_var": "validation_debug_wait_phase_var",
    "validation_wait_remaining_s_var": "validation_debug_wait_remaining_s_var",
    "validation_current_repeat_var": "validation_debug_current_repeat_var",
    "validation_result_var": "validation_debug_result_var",
    "validation_error_var": "validation_debug_error_var",
    "validation_export_path_var": "validation_debug_export_path_var",
}

PHASE_7B_LENGTH_DEFAULTS = {
    "pipe_len_var": "0.0",
    "len_enable_var": False,
    "len_z_low_approach_var": "0.0",
    "len_low_search_dist_var": "220.0",
    "len_high_search_dist_var": "220.0",
    "len_search_vel_var": "5.0",
    "len_search_timeout_var": "12.0",
    "len_tol_var": "20.0",
    "len_high_margin_var": "20.0",
    "len_debounce_k_var": "6",
    "len_max_stale_ms_var": "300",
    "len_backoff_var": "2.0",
    "len_info_var": "--",
    "len_status_var": "--",
    "len_edge_state_var": "--",
    "len_edge_low_var": "--",
    "len_edge_high_var": "--",
    "len_edge_len_var": "--",
}

PHASE_7B_TEACH_DEFAULTS = {
    "center_pos_var": "--",
    "teach_axes_mode_var": 2,
    "teach_rel_dist_var": "10",
    "teach_abs_var": "--",
    "teach_z_var": "--",
    "teach_align_var": "--",
    "teach_mode_var": "--",
    "teach_axes_var": "--",
    "start_info_var": "Start: 未设置",
    "standby_info_var": "未设置",
    "standby_state_var": "-",
}

PHASE_7C_GAUGE_DEFAULTS = {
    "sim_gauge_var": 0,
    "baud_var": "115200",
    "req_cmd_var": "M1,1",
    "gauge_conn_var": "未连接",
    "gauge_last_var": "Gauge: --",
    "gauge_err_var": "",
    "odcal_out2_hint_var": "OUT2→R",
    "odcal_duration_label_var": "时长(s)",
    "odcal_adv_open_var": False,
}

PHASE_7C_ODCAL_DEFAULTS = {
    "odcal_cmd_var": "M0,1",
    "odcal_dref_var": "180.000",
    "odcal_map_out1_var": "L",
    "odcal_mode_var": "timed",
    "odcal_hz_var": "20",
    "odcal_duration_var": "10",
    "odcal_rot_degps_var": "10",
    "odcal_angle_src_var": "AX3",
    "odcal_filter_var": "无",
    "odcal_outlier_sigma_var": "3.0",
    "odcal_defect_dyn_enable_var": 1,
    "odcal_state_var": "IDLE",
    "odcal_msg_var": "-",
    "odcal_defect_mode_var": "OFF",
    "odcal_defect_shift_var": "--",
    "odcal_defects_var": "--",
    "odcal_B_candidate_var": "--",
    "odcal_B_active_var": "--",
    "odcal_n_var": "0",
    "odcal_elapsed_var": "--",
    "odcal_sum_mean_var": "--",
    "odcal_sum_std_var": "--",
    "odcal_sum_min_var": "--",
    "odcal_sum_max_var": "--",
    "odcal_drop_rate_var": "--",
}

PHASE_7B_VAR_NAMES = frozenset(
    set(PHASE_7B_VALIDATION_DEFAULTS)
    | set(VALIDATION_ALIASES.values())
    | set(PHASE_7B_LENGTH_DEFAULTS)
    | set(PHASE_7B_TEACH_DEFAULTS)
)

PHASE_7C_VAR_NAMES = frozenset(set(PHASE_7C_GAUGE_DEFAULTS) | set(PHASE_7C_ODCAL_DEFAULTS))


def test_ui_state_create_uses_phase_7a_defaults() -> None:
    root = tk.Tcl()
    ui = UiState.create(root)

    for name, expected in PHASE_7A_DEFAULTS.items():
        assert getattr(ui, name).get() == expected


def test_ui_state_create_uses_phase_7b_defaults() -> None:
    root = tk.Tcl()
    ui = UiState.create(root)

    for defaults in (PHASE_7B_VALIDATION_DEFAULTS, PHASE_7B_LENGTH_DEFAULTS, PHASE_7B_TEACH_DEFAULTS):
        for name, expected in defaults.items():
            assert getattr(ui, name).get() == expected


def test_ui_state_create_uses_phase_7c_defaults() -> None:
    root = tk.Tcl()
    ui = UiState.create(root)

    for defaults in (PHASE_7C_GAUGE_DEFAULTS, PHASE_7C_ODCAL_DEFAULTS):
        for name, expected in defaults.items():
            assert getattr(ui, name).get() == expected


def test_validation_debug_aliases_share_variable_identity_and_values() -> None:
    root = tk.Tcl()
    ui = UiState.create(root)

    for canonical_name, alias_name in VALIDATION_ALIASES.items():
        canonical = getattr(ui, canonical_name)
        alias = getattr(ui, alias_name)
        assert canonical is alias
        if isinstance(canonical.get(), bool):
            canonical.set(True)
            assert alias.get() is True
            alias.set(False)
            assert canonical.get() is False
        else:
            canonical.set("from-canonical")
            assert alias.get() == "from-canonical"
            alias.set("from-alias")
            assert canonical.get() == "from-alias"


def test_app_host_compat_properties_return_ui_state_variables() -> None:
    root = tk.Tcl()
    host = object.__new__(AppHost)
    host.ui = UiState.create(root)

    for name in set(PHASE_7A_DEFAULTS) | PHASE_7B_VAR_NAMES | PHASE_7C_VAR_NAMES:
        assert getattr(host, name) is getattr(host.ui, name)


def test_app_host_compat_setter_supports_legacy_test_fakes() -> None:
    class _FakeVar:
        pass

    host = object.__new__(AppHost)
    fake = _FakeVar()

    host.auto_msg_var = fake

    assert host.auto_msg_var is fake


def test_gauge_presenter_binds_validation_vars_to_ui_state() -> None:
    class _Host:
        def __init__(self, root: tk.Misc) -> None:
            self.ui = UiState.create(root)
            self.sim_gauge_enabled = False

    root = tk.Tcl()
    host = _Host(root)
    presenter = GaugeScreenPresenter(host, object())

    before = host.ui.validation_status_var
    presenter.ensure_vars(root)
    presenter.validation_status_var.set("BUSY")
    presenter.ensure_vars(root)

    assert presenter.validation_status_var is before
    assert presenter.validation_status_var is host.ui.validation_status_var
    assert presenter.validation_debug_status_var is host.ui.validation_debug_status_var
    assert host.ui.validation_status_var.get() == "BUSY"


def test_gauge_presenter_binds_phase_7c_vars_to_ui_state_without_overwrite() -> None:
    class _Host:
        def __init__(self, root: tk.Misc) -> None:
            self.ui = UiState.create(root)
            self.sim_gauge_enabled = False

        def __getattr__(self, name: str) -> object:
            ui = self.__dict__.get("ui")
            if ui is not None and hasattr(ui, name):
                return getattr(ui, name)
            raise AttributeError(name)

    root = tk.Tcl()
    host = _Host(root)
    presenter = GaugeScreenPresenter(host, object())
    before = {name: getattr(host.ui, name) for name in PHASE_7C_VAR_NAMES}

    presenter.ensure_vars(root)
    host.ui.baud_var.set("57600")
    host.ui.odcal_cmd_var.set("M9,1")
    host.ui.odcal_defect_dyn_enable_var.set(0)
    host.ui.odcal_adv_open_var.set(True)
    presenter.ensure_vars(root)

    for name, variable in before.items():
        assert getattr(presenter, name) is variable
        assert getattr(host.ui, name) is variable
    assert host.ui.baud_var.get() == "57600"
    assert host.ui.odcal_cmd_var.get() == "M9,1"
    assert host.ui.odcal_defect_dyn_enable_var.get() == 0
    assert host.ui.odcal_adv_open_var.get() is True


def test_gauge_presenter_refreshes_odcal_derived_vars_from_ui_state() -> None:
    class _Host:
        def __init__(self, root: tk.Misc) -> None:
            self.ui = UiState.create(root)
            self.sim_gauge_enabled = False

        def __getattr__(self, name: str) -> object:
            ui = self.__dict__.get("ui")
            if ui is not None and hasattr(ui, name):
                return getattr(ui, name)
            raise AttributeError(name)

    root = tk.Tcl()
    host = _Host(root)
    presenter = GaugeScreenPresenter(host, object())
    presenter.ensure_vars(root)

    host.ui.odcal_map_out1_var.set("R")
    presenter.refresh_out2_hint()
    assert presenter.odcal_out2_hint_var.get().endswith("L")

    host.ui.odcal_mode_var.set("one_rev")
    presenter.refresh_odcal_duration_label()
    one_rev_label = presenter.odcal_duration_label_var.get()
    host.ui.odcal_mode_var.set("timed")
    presenter.refresh_odcal_duration_label()
    timed_label = presenter.odcal_duration_label_var.get()
    assert one_rev_label != timed_label
    assert timed_label.endswith("(s)")


def test_recipe_presenter_binds_length_and_teach_vars_to_ui_state_without_overwrite() -> None:
    class _Host:
        def __init__(self, root: tk.Misc) -> None:
            self.ui = UiState.create(root)
            self.recipe = Recipe()
            self.axis_cal = AxisCal()

    root = tk.Tcl()
    host = _Host(root)
    presenter = RecipeScreenPresenter(host)

    before_len = host.ui.len_low_search_dist_var
    before_teach = host.ui.teach_rel_dist_var
    presenter.ensure_vars(root)
    host.ui.len_low_search_dist_var.set("999")
    host.ui.teach_rel_dist_var.set("42")
    presenter.ensure_vars(root)

    assert presenter.len_low_search_dist_var is before_len
    assert presenter.teach_rel_dist_var is before_teach
    assert presenter.len_low_search_dist_var is host.ui.len_low_search_dist_var
    assert presenter.teach_rel_dist_var is host.ui.teach_rel_dist_var
    assert host.ui.len_low_search_dist_var.get() == "999"
    assert host.ui.teach_rel_dist_var.get() == "42"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_python_files(*relative_roots: str) -> list[tuple[Path, ast.Module]]:
    parsed: list[tuple[Path, ast.Module]] = []
    root = _repo_root()
    for relative_root in relative_roots:
        for path in sorted((root / relative_root).rglob("*.py")):
            parsed.append((path, ast.parse(path.read_text(encoding="utf-8-sig"))))
    return parsed


def test_forbidden_layers_do_not_import_ui_state() -> None:
    offenders: list[str] = []
    for path, tree in _parse_python_files("domain", "frp_workflow", "services", "repositories", "machine"):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "ui.state":
                        offenders.append(f"{path}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported_names = {alias.name for alias in node.names}
                if module == "ui.state" or "UiState" in imported_names:
                    offenders.append(f"{path}:{node.lineno}: from {module} import {', '.join(sorted(imported_names))}")

    assert not offenders, "\n".join(offenders)


def test_forbidden_layers_do_not_create_tk_variables() -> None:
    forbidden_calls = {"StringVar", "BooleanVar", "IntVar", "DoubleVar"}
    offenders: list[str] = []
    for path, tree in _parse_python_files("domain", "frp_workflow", "services", "repositories", "machine"):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in forbidden_calls:
                offenders.append(f"{path}:{node.lineno}: {ast.unparse(node)}")
            elif isinstance(func, ast.Name) and func.id in forbidden_calls:
                offenders.append(f"{path}:{node.lineno}: {ast.unparse(node)}")

    assert not offenders, "\n".join(offenders)


def test_application_host_layers_do_not_create_migrated_ui_state_variables_directly() -> None:
    offenders: list[str] = []
    migrated_names = PHASE_7B_VAR_NAMES | PHASE_7C_VAR_NAMES
    for path, tree in _parse_python_files("application"):
        if "presenters" in path.parts:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            target_names = {
                target.attr
                for target in node.targets
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"
            }
            if not (target_names & migrated_names):
                continue
            value = node.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr in {"StringVar", "BooleanVar", "IntVar", "DoubleVar"}
            ):
                offenders.append(f"{path}:{node.lineno}: {ast.unparse(node)}")

    assert not offenders, "\n".join(offenders)


def test_gauge_presenter_uses_ui_state_binding_for_phase_7c_variables() -> None:
    offenders: list[str] = []
    for path, tree in _parse_python_files("ui/presenters"):
        if path.name != "gauge_presenter.py":
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"
                and func.attr == "_ensure_var"
            ):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                continue
            if node.args[0].value in PHASE_7C_VAR_NAMES:
                offenders.append(f"{path}:{node.lineno}: {ast.unparse(node)}")

    assert not offenders, "\n".join(offenders)
