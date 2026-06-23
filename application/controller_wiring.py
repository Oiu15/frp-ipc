from __future__ import annotations

"""Screen controller / presenter wiring (Phase 2).

This module owns the creation of ``ScreenPresenter``, ``ScreenController``,
``ScreenUiContext``, and the per-tab presenters (axis, recipe, gauge).
The presenter *classes* stay where they are; only the assembly is moved
here.
"""

from typing import Any

from application.adapters.device_gateway import (
    ScreenController,
    ScreenPresenter,
    ScreenUiContext,
)
from application.controllers.axis_controller import AxisController
from application.controllers.axis_cal_controller import AxisCalController
from application.controllers.gauge_controller import GaugeController
from application.controllers.key_test_controller import KeyTestController
from application.controllers.main_controller import MainController
from application.controllers.recipe_controller import RecipeController
from application.controllers.validation_controller import ValidationController
from ui.presenters.axis_cal_presenter_deps import AxisCalUiState
from ui.presenters.axis_presenter import AxisScreenPresenter
from ui.presenters.gauge_presenter import GaugeScreenPresenter
from ui.presenters.gauge_presenter_deps import GaugeUiState
from ui.presenters.key_test_presenter import KeyTestUiState
from ui.presenters.main_presenter_deps import MainUiState
from ui.presenters.recipe_presenter import RecipeScreenPresenter
from ui.presenters.recipe_presenter_deps import RecipePresenterDeps
from ui.presenters.validation_presenter_deps import ValidationUiState


def _build_recipe_presenter_deps(host: Any) -> RecipePresenterDeps:
    def set_recipe(value: Any) -> None:
        host.recipe = value

    def log_ax3_speed_trace(location_name: str, recipe_obj: Any) -> None:
        host._log_ax3_speed_trace(location_name, recipe_obj=recipe_obj)

    def set_len_low_approach_legacy_z(value: float | None) -> None:
        host._len_low_appr_legacy_z = value

    callbacks = []
    for method_name in (
        "_apply_start_anchor_from_recipe",
        "_refresh_recipe_table",
        "_refresh_auto_std_panel",
        "_refresh_standby_pos",
        "_refresh_center_positions",
    ):
        callback = getattr(host, method_name, None)
        if callable(callback):
            callbacks.append(callback)

    return RecipePresenterDeps(
        get_recipe=lambda: host.recipe,
        set_recipe=set_recipe,
        axis_cal=host.axis_cal,
        ui_state=getattr(host, "ui", None),
        log_ax3_speed_trace=log_ax3_speed_trace,
        refresh_length_info=getattr(host, "_refresh_length_info", None),
        set_len_low_approach_legacy_z=set_len_low_approach_legacy_z,
        after_recipe_data_applied=tuple(callbacks),
    )


def _build_gauge_ui_state(host: Any) -> GaugeUiState:
    return GaugeUiState(
        list_serial_ports_callback=host._list_serial_ports,
        sim_gauge_enabled=bool(host.sim_gauge_enabled),
        ip_var=host.ip_var,
        port_var=host.port_var,
        plc_status_var=host.plc_status_var,
        sim_gauge_var=host.ui.sim_gauge_var,
        baud_var=host.ui.baud_var,
        req_cmd_var=host.ui.req_cmd_var,
        gauge_conn_var=host.ui.gauge_conn_var,
        gauge_last_var=host.ui.gauge_last_var,
        gauge_err_var=host.ui.gauge_err_var,
        odcal_out2_hint_var=host.ui.odcal_out2_hint_var,
        odcal_duration_label_var=host.ui.odcal_duration_label_var,
        odcal_adv_open_var=host.ui.odcal_adv_open_var,
        odcal_cmd_var=host.ui.odcal_cmd_var,
        odcal_dref_var=host.ui.odcal_dref_var,
        odcal_map_out1_var=host.ui.odcal_map_out1_var,
        odcal_mode_var=host.ui.odcal_mode_var,
        odcal_hz_var=host.ui.odcal_hz_var,
        odcal_duration_var=host.ui.odcal_duration_var,
        odcal_rot_degps_var=host.ui.odcal_rot_degps_var,
        odcal_angle_src_var=host.ui.odcal_angle_src_var,
        odcal_filter_var=host.ui.odcal_filter_var,
        odcal_outlier_sigma_var=host.ui.odcal_outlier_sigma_var,
        odcal_defect_dyn_enable_var=host.ui.odcal_defect_dyn_enable_var,
        odcal_state_var=host.ui.odcal_state_var,
        odcal_msg_var=host.ui.odcal_msg_var,
        odcal_defect_mode_var=host.ui.odcal_defect_mode_var,
        odcal_defect_shift_var=host.ui.odcal_defect_shift_var,
        odcal_defects_var=host.ui.odcal_defects_var,
        odcal_B_candidate_var=host.ui.odcal_B_candidate_var,
        odcal_B_active_var=host.ui.odcal_B_active_var,
        odcal_n_var=host.ui.odcal_n_var,
        odcal_elapsed_var=host.ui.odcal_elapsed_var,
        odcal_sum_mean_var=host.ui.odcal_sum_mean_var,
        odcal_sum_std_var=host.ui.odcal_sum_std_var,
        odcal_sum_min_var=host.ui.odcal_sum_min_var,
        odcal_sum_max_var=host.ui.odcal_sum_max_var,
        odcal_drop_rate_var=host.ui.odcal_drop_rate_var,
        cl_out1_var=host.cl_out1_var,
        cl_out1_cnt_var=host.cl_out1_cnt_var,
        cl_out2_var=host.cl_out2_var,
        cl_out2_cnt_var=host.cl_out2_cnt_var,
        cl_out3_var=host.cl_out3_var,
        cl_out3_cnt_var=host.cl_out3_cnt_var,
        cl_out4_var=host.cl_out4_var,
        cl_out4_cnt_var=host.cl_out4_cnt_var,
        cl_out5_var=host.cl_out5_var,
        cl_out5_cnt_var=host.cl_out5_cnt_var,
        cl_m_calc_var=host.cl_m_calc_var,
        cl_m_diff_var=host.cl_m_diff_var,
        idcal_state_var=host.idcal_state_var,
        idcal_msg_var=host.idcal_msg_var,
        idcal_dref_var=host.idcal_dref_var,
        idcal_mode_var=host.idcal_mode_var,
        idcal_hz_var=host.idcal_hz_var,
        idcal_duration_var=host.idcal_duration_var,
        idcal_rot_degps_var=host.idcal_rot_degps_var,
        idcal_delta_candidate_var=host.idcal_delta_candidate_var,
        idcal_delta_active_var=host.idcal_delta_active_var,
        idcal_cmax_var=host.idcal_cmax_var,
        idcal_mmean_var=host.idcal_mmean_var,
        idcal_mpp_var=host.idcal_mpp_var,
        idcal_fit_diam_var=host.idcal_fit_diam_var,
        idcal_fit_e_var=host.idcal_fit_e_var,
        idcal_fit_y0_var=host.idcal_fit_y0_var,
        idcal_fit_rmse_var=host.idcal_fit_rmse_var,
        idcal_chk_err_var=host.idcal_chk_err_var,
        idcal_chk_cov_var=host.idcal_chk_cov_var,
        idcal_chk_n_var=host.idcal_chk_n_var,
        idcal_chk_dtheta_var=host.idcal_chk_dtheta_var,
        id_single_cal_state_var=host.id_single_cal_state_var,
        id_single_cal_msg_var=host.id_single_cal_msg_var,
        id_single_cal_dref_var=host.id_single_cal_dref_var,
        id_single_cal_mean_var=host.id_single_cal_mean_var,
        id_single_cal_B_var=host.id_single_cal_B_var,
        id_single_cal_ecc_amp_var=host.id_single_cal_ecc_amp_var,
        id_single_cal_ecc_ang_var=host.id_single_cal_ecc_ang_var,
        id_single_cal_cov_var=host.id_single_cal_cov_var,
        id_single_cal_warn_var=host.id_single_cal_warn_var,
        validation_section_name_var=host.ui.validation_section_name_var,
        validation_metric_name_var=host.ui.validation_metric_name_var,
        validation_repeat_count_var=host.ui.validation_repeat_count_var,
        validation_reclamp_between_repeats_var=host.ui.validation_reclamp_between_repeats_var,
        validation_reclamp_enabled_var=host.ui.validation_reclamp_enabled_var,
        validation_rotation_stop_before_measure_var=host.ui.validation_rotation_stop_before_measure_var,
        validation_release_settle_s_var=host.ui.validation_release_settle_s_var,
        validation_clamp_settle_s_var=host.ui.validation_clamp_settle_s_var,
        validation_position_settle_s_var=host.ui.validation_position_settle_s_var,
        validation_sample_delay_s_var=host.ui.validation_sample_delay_s_var,
        validation_ax3_speed_dps_var=host.ui.validation_ax3_speed_dps_var,
        validation_move_enabled_var=host.ui.validation_move_enabled_var,
        validation_move_channel_var=host.ui.validation_move_channel_var,
        validation_move_away_delta_mm_var=host.ui.validation_move_away_delta_mm_var,
        validation_move_scenario_var=host.ui.validation_move_scenario_var,
        validation_move_from_section_var=host.ui.validation_move_from_section_var,
        validation_move_target_section_var=host.ui.validation_move_target_section_var,
        validation_move_return_section_var=host.ui.validation_move_return_section_var,
        validation_move_target_pos_var=host.ui.validation_move_target_pos_var,
        validation_move_actual_pos_var=host.ui.validation_move_actual_pos_var,
        validation_status_var=host.ui.validation_status_var,
        validation_phase_var=host.ui.validation_phase_var,
        validation_wait_phase_var=host.ui.validation_wait_phase_var,
        validation_wait_remaining_s_var=host.ui.validation_wait_remaining_s_var,
        validation_current_repeat_var=host.ui.validation_current_repeat_var,
        validation_result_var=host.ui.validation_result_var,
        validation_error_var=host.ui.validation_error_var,
        validation_export_path_var=host.ui.validation_export_path_var,
        validation_current_metric_value_var=host.ui.validation_current_metric_value_var,
        validation_current_section_var=host.ui.validation_current_section_var,
        validation_current_z_pos_var=host.ui.validation_current_z_pos_var,
        validation_current_concentricity_var=host.ui.validation_current_concentricity_var,
        validation_summary_count_var=host.ui.validation_summary_count_var,
        validation_summary_mean_var=host.ui.validation_summary_mean_var,
        validation_summary_std_var=host.ui.validation_summary_std_var,
        validation_summary_min_var=host.ui.validation_summary_min_var,
        validation_summary_max_var=host.ui.validation_summary_max_var,
        validation_summary_range_var=host.ui.validation_summary_range_var,
    )


def wire_screen_controllers(host: Any) -> None:
    """Create screen presenters and controllers, attaching each to *host*.

    This replaces the former ``HostUIMixin._init_presenters`` inline
    wiring.  The caller (AppHost.__init__) invokes this once, then calls
    ``host._build_ui()``.
    """
    controller = ScreenController(host)
    host._screen_controller = controller

    presenter = ScreenPresenter(host)
    host._screen_presenter = presenter

    main_controller = MainController(host)
    host.main_controller = main_controller

    main_ui = MainUiState(
        pipe_sn_var=host.pipe_sn_var,
        meas_seq_var=host.meas_seq_var,
        meas_start_var=host.meas_start_var,
        meas_elapsed_var=host.meas_elapsed_var,
        auto_progress_var=host.auto_progress_var,
        auto_done_var=host.auto_done_var,
        auto_state_var=host.auto_state_var,
        ui_meas_mode_var=host.ui_meas_mode_var,
        auto_msg_var=host.auto_msg_var,
        od_mean_var=host.od_mean_var,
        od_range_var=host.od_range_var,
        max_od_pp_var=host.max_od_pp_var,
        max_od_pp_rob_var=host.max_od_pp_rob_var,
        max_od_fit_res_var=host.max_od_fit_res_var,
        od_tilt_var=host.od_tilt_var,
        od_slope_var=host.od_slope_var,
        od_endoff_var=host.od_endoff_var,
        id_mean_var=host.id_mean_var,
        id_range_var=host.id_range_var,
        id_tilt_var=host.id_tilt_var,
        id_slope_var=host.id_slope_var,
        id_endoff_var=host.id_endoff_var,
        max_id_round_var=host.max_id_round_var,
        axis_dist_var=host.axis_dist_var,
        conc_max_var=host.conc_max_var,
        axis_span_max_var=host.axis_span_max_var,
        geom_v2_tau_var=host.geom_v2_tau_var,
        len_meas_var=host.len_meas_var,
        cov_var=host.cov_var,
    )
    host.main_ui = main_ui

    recipe_presenter = RecipeScreenPresenter(_build_recipe_presenter_deps(host))
    host._recipe_screen_presenter = recipe_presenter

    recipe_controller = RecipeController(host)
    host.recipe_controller = recipe_controller

    axis_cal_controller = AxisCalController(host)
    host.axis_cal_controller = axis_cal_controller

    axis_cal_ui = AxisCalUiState(
        axis_cal_vars=host.axis_cal_vars,
        axis_cal_field_status_vars=host.axis_cal_field_status_vars,
        axis_cal_status_vars=host.axis_cal_status_vars,
    )
    host.axis_cal_ui = axis_cal_ui

    validation_controller = ValidationController(host)
    host.validation_controller = validation_controller

    validation_ui = ValidationUiState(
        validation_section_name_var=host.ui.validation_section_name_var,
        validation_metric_name_var=host.ui.validation_metric_name_var,
        validation_repeat_count_var=host.ui.validation_repeat_count_var,
        validation_reclamp_between_repeats_var=host.ui.validation_reclamp_between_repeats_var,
        validation_reclamp_enabled_var=host.ui.validation_reclamp_enabled_var,
        validation_rotation_stop_before_measure_var=host.ui.validation_rotation_stop_before_measure_var,
        validation_release_settle_s_var=host.ui.validation_release_settle_s_var,
        validation_clamp_settle_s_var=host.ui.validation_clamp_settle_s_var,
        validation_position_settle_s_var=host.ui.validation_position_settle_s_var,
        validation_sample_delay_s_var=host.ui.validation_sample_delay_s_var,
        validation_ax3_speed_dps_var=host.ui.validation_ax3_speed_dps_var,
        validation_move_enabled_var=host.ui.validation_move_enabled_var,
        validation_move_channel_var=host.ui.validation_move_channel_var,
        validation_move_away_delta_mm_var=host.ui.validation_move_away_delta_mm_var,
        validation_move_scenario_var=host.ui.validation_move_scenario_var,
        validation_move_from_section_var=host.ui.validation_move_from_section_var,
        validation_move_target_section_var=host.ui.validation_move_target_section_var,
        validation_move_return_section_var=host.ui.validation_move_return_section_var,
        validation_move_target_pos_var=host.ui.validation_move_target_pos_var,
        validation_move_actual_pos_var=host.ui.validation_move_actual_pos_var,
        validation_status_var=host.ui.validation_status_var,
        validation_phase_var=host.ui.validation_phase_var,
        validation_wait_phase_var=host.ui.validation_wait_phase_var,
        validation_wait_remaining_s_var=host.ui.validation_wait_remaining_s_var,
        validation_current_repeat_var=host.ui.validation_current_repeat_var,
        validation_result_var=host.ui.validation_result_var,
        validation_error_var=host.ui.validation_error_var,
        validation_export_path_var=host.ui.validation_export_path_var,
        validation_current_metric_value_var=host.ui.validation_current_metric_value_var,
        validation_current_section_var=host.ui.validation_current_section_var,
        validation_current_z_pos_var=host.ui.validation_current_z_pos_var,
        validation_current_concentricity_var=host.ui.validation_current_concentricity_var,
        validation_summary_count_var=host.ui.validation_summary_count_var,
        validation_summary_mean_var=host.ui.validation_summary_mean_var,
        validation_summary_std_var=host.ui.validation_summary_std_var,
        validation_summary_min_var=host.ui.validation_summary_min_var,
        validation_summary_max_var=host.ui.validation_summary_max_var,
        validation_summary_range_var=host.ui.validation_summary_range_var,
    )
    host.validation_ui = validation_ui

    key_test_controller = KeyTestController(host)
    host.key_test_controller = key_test_controller

    key_test_ui = KeyTestUiState(
        keytest_x_vars=host.keytest_x_vars,
        keytest_y_vars=host.keytest_y_vars,
        keytest_y_lastcmd_vars=host.keytest_y_lastcmd_vars,
    )
    host.key_test_ui = key_test_ui

    axis_controller = AxisController(host)
    host.axis_controller = axis_controller

    axis_presenter = AxisScreenPresenter(host, axis_controller)
    host._axis_screen_presenter = axis_presenter

    gauge_controller = GaugeController(host)
    host.gauge_controller = gauge_controller

    gauge_ui = _build_gauge_ui_state(host)
    host.gauge_ui = gauge_ui

    gauge_presenter = GaugeScreenPresenter(gauge_ui, gauge_controller)
    host._gauge_screen_presenter = gauge_presenter

    ui_context = ScreenUiContext(host)
    host._screen_ui_context = ui_context


__all__ = ["wire_screen_controllers"]
