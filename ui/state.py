from __future__ import annotations

"""Tk variable container for UI-only display state."""

from dataclasses import dataclass
import tkinter as tk


@dataclass(slots=True)
class UiStateDefaults:
    # Phase 7A: main/run display defaults
    pipe_sn: str = "--"
    meas_seq: str = "--"
    meas_start: str = "--"
    meas_elapsed: str = "--"
    ui_meas_mode: str = "检测模式：--"
    auto_state: str = "IDLE"
    auto_msg: str = "-"
    auto_progress: str = "当前截面: - / 总截面: -"
    auto_done: str = "测量完成: 否"
    cov: str = "采样覆盖率：--"
    straight: str = "直线度   --（外圆） | --（内圆）"
    straight_od: str = "--"
    straight_id: str = "--"
    conc: str = "整体同心度   --"
    conc_max: str = "--"
    axis_dist: str = "--"
    axis_span_max: str = "--"
    geom_v2_tau: str = "--"

    # Phase 7B-1: validation defaults
    validation_section_name: str = ""
    validation_metric_name: str = "od_avg"
    validation_repeat_count: str = "3"
    validation_reclamp_between_repeats: bool = False
    validation_reclamp_enabled: bool = False
    validation_rotation_stop_before_measure: bool = False
    validation_release_settle_s: str = "0.0"
    validation_clamp_settle_s: str = "0.0"
    validation_position_settle_s: str = "0.0"
    validation_sample_delay_s: str = "0.0"
    validation_ax3_speed_dps: str = "60.0"
    validation_move_enabled: bool = False
    validation_move_channel: str = "od_channel"
    validation_move_away_delta_mm: str = "0.0"
    validation_move_scenario: str = "distance_round_trip"
    validation_move_from_section: str = "1"
    validation_move_target_section: str = "1"
    validation_move_return_section: str = "1"
    validation_move_target_pos: str = ""
    validation_move_actual_pos: str = ""
    validation_status: str = "IDLE"
    validation_phase: str = "IDLE"
    validation_wait_phase: str = ""
    validation_wait_remaining_s: str = ""
    validation_current_repeat: str = "0/0"
    validation_result: str = ""
    validation_error: str = ""
    validation_export_path: str = ""
    validation_current_metric_value: str = ""
    validation_current_section: str = ""
    validation_current_z_pos: str = ""
    validation_current_concentricity: str = ""
    validation_summary_count: str = "0"
    validation_summary_mean: str = ""
    validation_summary_std: str = ""
    validation_summary_min: str = ""
    validation_summary_max: str = ""
    validation_summary_range: str = ""

    # Phase 7B-2: length config/input defaults
    pipe_len: str = "0.0"
    len_enable: bool = False
    len_z_low_approach: str = "0.0"
    len_low_search_dist: str = "220.0"
    len_high_search_dist: str = "220.0"
    len_search_vel: str = "5.0"
    len_search_timeout: str = "12.0"
    len_tol: str = "20.0"
    len_high_margin: str = "20.0"
    len_debounce_k: str = "6"
    len_max_stale_ms: str = "300"
    len_backoff: str = "2.0"

    # Phase 7B-2: length display defaults
    len_info: str = "--"
    len_status: str = "--"
    len_edge_state: str = "--"
    len_edge_low: str = "--"
    len_edge_high: str = "--"
    len_edge_len: str = "--"

    # Phase 7B-2: teach defaults
    center_pos: str = "--"
    teach_axes_mode: int = 2
    teach_rel_dist: str = "10"
    teach_abs: str = "--"
    teach_z: str = "--"
    teach_align: str = "--"
    teach_mode: str = "--"
    teach_axes: str = "--"
    start_info: str = "Start: 未设置"
    standby_info: str = "未设置"
    standby_state: str = "-"

    # Phase 7C-1: gauge defaults
    sim_gauge: int = 0
    baud: str = "115200"
    req_cmd: str = "M1,1"
    gauge_conn: str = "未连接"
    gauge_last: str = "Gauge: --"
    gauge_err: str = ""
    odcal_out2_hint: str = "OUT2→R"
    odcal_duration_label: str = "时长(s)"
    odcal_adv_open: bool = False

    # Phase 7C-2: OD calibration config/input defaults
    odcal_cmd: str = "M0,1"
    odcal_dref: str = "180.000"
    odcal_map_out1: str = "L"
    odcal_mode: str = "timed"
    odcal_hz: str = "20"
    odcal_duration: str = "10"
    odcal_rot_degps: str = "10"
    odcal_angle_src: str = "AX3"
    odcal_filter: str = "无"
    odcal_outlier_sigma: str = "3.0"
    odcal_defect_dyn_enable: int = 1

    # Phase 7C-2: OD calibration state/display defaults
    odcal_state: str = "IDLE"
    odcal_msg: str = "-"
    odcal_defect_mode: str = "OFF"
    odcal_defect_shift: str = "--"
    odcal_defects: str = "--"
    odcal_B_candidate: str = "--"
    odcal_B_active: str = "--"
    odcal_n: str = "0"
    odcal_elapsed: str = "--"
    odcal_sum_mean: str = "--"
    odcal_sum_std: str = "--"
    odcal_sum_min: str = "--"
    odcal_sum_max: str = "--"
    odcal_drop_rate: str = "--"


@dataclass(slots=True)
class UiState:
    pipe_sn_var: tk.StringVar
    meas_seq_var: tk.StringVar
    meas_start_var: tk.StringVar
    meas_elapsed_var: tk.StringVar
    ui_meas_mode_var: tk.StringVar
    auto_state_var: tk.StringVar
    auto_msg_var: tk.StringVar
    auto_progress_var: tk.StringVar
    auto_done_var: tk.StringVar
    cov_var: tk.StringVar
    straight_var: tk.StringVar
    straight_od_var: tk.StringVar
    straight_id_var: tk.StringVar
    conc_var: tk.StringVar
    conc_max_var: tk.StringVar
    axis_dist_var: tk.StringVar
    axis_span_max_var: tk.StringVar
    geom_v2_tau_var: tk.StringVar
    validation_section_name_var: tk.StringVar
    validation_debug_section_name_var: tk.StringVar
    validation_metric_name_var: tk.StringVar
    validation_debug_metric_name_var: tk.StringVar
    validation_repeat_count_var: tk.StringVar
    validation_debug_repeat_count_var: tk.StringVar
    validation_reclamp_between_repeats_var: tk.BooleanVar
    validation_debug_reclamp_between_repeats_var: tk.BooleanVar
    validation_reclamp_enabled_var: tk.BooleanVar
    validation_debug_reclamp_enabled_var: tk.BooleanVar
    validation_rotation_stop_before_measure_var: tk.BooleanVar
    validation_debug_rotation_stop_before_measure_var: tk.BooleanVar
    validation_release_settle_s_var: tk.StringVar
    validation_debug_release_settle_s_var: tk.StringVar
    validation_clamp_settle_s_var: tk.StringVar
    validation_debug_clamp_settle_s_var: tk.StringVar
    validation_position_settle_s_var: tk.StringVar
    validation_debug_position_settle_s_var: tk.StringVar
    validation_sample_delay_s_var: tk.StringVar
    validation_debug_sample_delay_s_var: tk.StringVar
    validation_ax3_speed_dps_var: tk.StringVar
    validation_debug_ax3_speed_dps_var: tk.StringVar
    validation_move_enabled_var: tk.BooleanVar
    validation_debug_move_enabled_var: tk.BooleanVar
    validation_move_channel_var: tk.StringVar
    validation_debug_move_channel_var: tk.StringVar
    validation_move_away_delta_mm_var: tk.StringVar
    validation_debug_move_away_delta_mm_var: tk.StringVar
    validation_move_scenario_var: tk.StringVar
    validation_debug_move_scenario_var: tk.StringVar
    validation_move_from_section_var: tk.StringVar
    validation_debug_move_from_section_var: tk.StringVar
    validation_move_target_section_var: tk.StringVar
    validation_debug_move_target_section_var: tk.StringVar
    validation_move_return_section_var: tk.StringVar
    validation_debug_move_return_section_var: tk.StringVar
    validation_move_target_pos_var: tk.StringVar
    validation_debug_move_target_pos_var: tk.StringVar
    validation_move_actual_pos_var: tk.StringVar
    validation_debug_move_actual_pos_var: tk.StringVar
    validation_status_var: tk.StringVar
    validation_debug_status_var: tk.StringVar
    validation_phase_var: tk.StringVar
    validation_debug_phase_var: tk.StringVar
    validation_wait_phase_var: tk.StringVar
    validation_debug_wait_phase_var: tk.StringVar
    validation_wait_remaining_s_var: tk.StringVar
    validation_debug_wait_remaining_s_var: tk.StringVar
    validation_current_repeat_var: tk.StringVar
    validation_debug_current_repeat_var: tk.StringVar
    validation_result_var: tk.StringVar
    validation_debug_result_var: tk.StringVar
    validation_error_var: tk.StringVar
    validation_debug_error_var: tk.StringVar
    validation_export_path_var: tk.StringVar
    validation_debug_export_path_var: tk.StringVar
    validation_current_metric_value_var: tk.StringVar
    validation_current_section_var: tk.StringVar
    validation_current_z_pos_var: tk.StringVar
    validation_current_concentricity_var: tk.StringVar
    validation_summary_count_var: tk.StringVar
    validation_summary_mean_var: tk.StringVar
    validation_summary_std_var: tk.StringVar
    validation_summary_min_var: tk.StringVar
    validation_summary_max_var: tk.StringVar
    validation_summary_range_var: tk.StringVar
    pipe_len_var: tk.StringVar
    len_enable_var: tk.BooleanVar
    len_z_low_approach_var: tk.StringVar
    len_low_search_dist_var: tk.StringVar
    len_high_search_dist_var: tk.StringVar
    len_search_vel_var: tk.StringVar
    len_search_timeout_var: tk.StringVar
    len_tol_var: tk.StringVar
    len_high_margin_var: tk.StringVar
    len_debounce_k_var: tk.StringVar
    len_max_stale_ms_var: tk.StringVar
    len_backoff_var: tk.StringVar
    len_info_var: tk.StringVar
    len_status_var: tk.StringVar
    len_edge_state_var: tk.StringVar
    len_edge_low_var: tk.StringVar
    len_edge_high_var: tk.StringVar
    len_edge_len_var: tk.StringVar
    center_pos_var: tk.StringVar
    teach_axes_mode_var: tk.IntVar
    teach_rel_dist_var: tk.StringVar
    teach_abs_var: tk.StringVar
    teach_z_var: tk.StringVar
    teach_align_var: tk.StringVar
    teach_mode_var: tk.StringVar
    teach_axes_var: tk.StringVar
    start_info_var: tk.StringVar
    standby_info_var: tk.StringVar
    standby_state_var: tk.StringVar
    sim_gauge_var: tk.IntVar
    baud_var: tk.StringVar
    req_cmd_var: tk.StringVar
    gauge_conn_var: tk.StringVar
    gauge_last_var: tk.StringVar
    gauge_err_var: tk.StringVar
    odcal_out2_hint_var: tk.StringVar
    odcal_duration_label_var: tk.StringVar
    odcal_adv_open_var: tk.BooleanVar
    odcal_cmd_var: tk.StringVar
    odcal_dref_var: tk.StringVar
    odcal_map_out1_var: tk.StringVar
    odcal_mode_var: tk.StringVar
    odcal_hz_var: tk.StringVar
    odcal_duration_var: tk.StringVar
    odcal_rot_degps_var: tk.StringVar
    odcal_angle_src_var: tk.StringVar
    odcal_filter_var: tk.StringVar
    odcal_outlier_sigma_var: tk.StringVar
    odcal_defect_dyn_enable_var: tk.IntVar
    odcal_state_var: tk.StringVar
    odcal_msg_var: tk.StringVar
    odcal_defect_mode_var: tk.StringVar
    odcal_defect_shift_var: tk.StringVar
    odcal_defects_var: tk.StringVar
    odcal_B_candidate_var: tk.StringVar
    odcal_B_active_var: tk.StringVar
    odcal_n_var: tk.StringVar
    odcal_elapsed_var: tk.StringVar
    odcal_sum_mean_var: tk.StringVar
    odcal_sum_std_var: tk.StringVar
    odcal_sum_min_var: tk.StringVar
    odcal_sum_max_var: tk.StringVar
    odcal_drop_rate_var: tk.StringVar

    @classmethod
    def create(cls, root: tk.Misc, defaults: UiStateDefaults | None = None) -> UiState:
        d = defaults or UiStateDefaults()
        validation_section_name_var = tk.StringVar(master=root, value=d.validation_section_name)
        validation_metric_name_var = tk.StringVar(master=root, value=d.validation_metric_name)
        validation_repeat_count_var = tk.StringVar(master=root, value=d.validation_repeat_count)
        validation_reclamp_between_repeats_var = tk.BooleanVar(
            master=root, value=d.validation_reclamp_between_repeats
        )
        validation_reclamp_enabled_var = tk.BooleanVar(master=root, value=d.validation_reclamp_enabled)
        validation_rotation_stop_before_measure_var = tk.BooleanVar(
            master=root, value=d.validation_rotation_stop_before_measure
        )
        validation_release_settle_s_var = tk.StringVar(master=root, value=d.validation_release_settle_s)
        validation_clamp_settle_s_var = tk.StringVar(master=root, value=d.validation_clamp_settle_s)
        validation_position_settle_s_var = tk.StringVar(master=root, value=d.validation_position_settle_s)
        validation_sample_delay_s_var = tk.StringVar(master=root, value=d.validation_sample_delay_s)
        validation_ax3_speed_dps_var = tk.StringVar(master=root, value=d.validation_ax3_speed_dps)
        validation_move_enabled_var = tk.BooleanVar(master=root, value=d.validation_move_enabled)
        validation_move_channel_var = tk.StringVar(master=root, value=d.validation_move_channel)
        validation_move_away_delta_mm_var = tk.StringVar(master=root, value=d.validation_move_away_delta_mm)
        validation_move_scenario_var = tk.StringVar(master=root, value=d.validation_move_scenario)
        validation_move_from_section_var = tk.StringVar(master=root, value=d.validation_move_from_section)
        validation_move_target_section_var = tk.StringVar(master=root, value=d.validation_move_target_section)
        validation_move_return_section_var = tk.StringVar(master=root, value=d.validation_move_return_section)
        validation_move_target_pos_var = tk.StringVar(master=root, value=d.validation_move_target_pos)
        validation_move_actual_pos_var = tk.StringVar(master=root, value=d.validation_move_actual_pos)
        validation_status_var = tk.StringVar(master=root, value=d.validation_status)
        validation_phase_var = tk.StringVar(master=root, value=d.validation_phase)
        validation_wait_phase_var = tk.StringVar(master=root, value=d.validation_wait_phase)
        validation_wait_remaining_s_var = tk.StringVar(master=root, value=d.validation_wait_remaining_s)
        validation_current_repeat_var = tk.StringVar(master=root, value=d.validation_current_repeat)
        validation_result_var = tk.StringVar(master=root, value=d.validation_result)
        validation_error_var = tk.StringVar(master=root, value=d.validation_error)
        validation_export_path_var = tk.StringVar(master=root, value=d.validation_export_path)
        return cls(
            pipe_sn_var=tk.StringVar(master=root, value=d.pipe_sn),
            meas_seq_var=tk.StringVar(master=root, value=d.meas_seq),
            meas_start_var=tk.StringVar(master=root, value=d.meas_start),
            meas_elapsed_var=tk.StringVar(master=root, value=d.meas_elapsed),
            ui_meas_mode_var=tk.StringVar(master=root, value=d.ui_meas_mode),
            auto_state_var=tk.StringVar(master=root, value=d.auto_state),
            auto_msg_var=tk.StringVar(master=root, value=d.auto_msg),
            auto_progress_var=tk.StringVar(master=root, value=d.auto_progress),
            auto_done_var=tk.StringVar(master=root, value=d.auto_done),
            cov_var=tk.StringVar(master=root, value=d.cov),
            straight_var=tk.StringVar(master=root, value=d.straight),
            straight_od_var=tk.StringVar(master=root, value=d.straight_od),
            straight_id_var=tk.StringVar(master=root, value=d.straight_id),
            conc_var=tk.StringVar(master=root, value=d.conc),
            conc_max_var=tk.StringVar(master=root, value=d.conc_max),
            axis_dist_var=tk.StringVar(master=root, value=d.axis_dist),
            axis_span_max_var=tk.StringVar(master=root, value=d.axis_span_max),
            geom_v2_tau_var=tk.StringVar(master=root, value=d.geom_v2_tau),
            validation_section_name_var=validation_section_name_var,
            validation_debug_section_name_var=validation_section_name_var,
            validation_metric_name_var=validation_metric_name_var,
            validation_debug_metric_name_var=validation_metric_name_var,
            validation_repeat_count_var=validation_repeat_count_var,
            validation_debug_repeat_count_var=validation_repeat_count_var,
            validation_reclamp_between_repeats_var=validation_reclamp_between_repeats_var,
            validation_debug_reclamp_between_repeats_var=validation_reclamp_between_repeats_var,
            validation_reclamp_enabled_var=validation_reclamp_enabled_var,
            validation_debug_reclamp_enabled_var=validation_reclamp_enabled_var,
            validation_rotation_stop_before_measure_var=validation_rotation_stop_before_measure_var,
            validation_debug_rotation_stop_before_measure_var=validation_rotation_stop_before_measure_var,
            validation_release_settle_s_var=validation_release_settle_s_var,
            validation_debug_release_settle_s_var=validation_release_settle_s_var,
            validation_clamp_settle_s_var=validation_clamp_settle_s_var,
            validation_debug_clamp_settle_s_var=validation_clamp_settle_s_var,
            validation_position_settle_s_var=validation_position_settle_s_var,
            validation_debug_position_settle_s_var=validation_position_settle_s_var,
            validation_sample_delay_s_var=validation_sample_delay_s_var,
            validation_debug_sample_delay_s_var=validation_sample_delay_s_var,
            validation_ax3_speed_dps_var=validation_ax3_speed_dps_var,
            validation_debug_ax3_speed_dps_var=validation_ax3_speed_dps_var,
            validation_move_enabled_var=validation_move_enabled_var,
            validation_debug_move_enabled_var=validation_move_enabled_var,
            validation_move_channel_var=validation_move_channel_var,
            validation_debug_move_channel_var=validation_move_channel_var,
            validation_move_away_delta_mm_var=validation_move_away_delta_mm_var,
            validation_debug_move_away_delta_mm_var=validation_move_away_delta_mm_var,
            validation_move_scenario_var=validation_move_scenario_var,
            validation_debug_move_scenario_var=validation_move_scenario_var,
            validation_move_from_section_var=validation_move_from_section_var,
            validation_debug_move_from_section_var=validation_move_from_section_var,
            validation_move_target_section_var=validation_move_target_section_var,
            validation_debug_move_target_section_var=validation_move_target_section_var,
            validation_move_return_section_var=validation_move_return_section_var,
            validation_debug_move_return_section_var=validation_move_return_section_var,
            validation_move_target_pos_var=validation_move_target_pos_var,
            validation_debug_move_target_pos_var=validation_move_target_pos_var,
            validation_move_actual_pos_var=validation_move_actual_pos_var,
            validation_debug_move_actual_pos_var=validation_move_actual_pos_var,
            validation_status_var=validation_status_var,
            validation_debug_status_var=validation_status_var,
            validation_phase_var=validation_phase_var,
            validation_debug_phase_var=validation_phase_var,
            validation_wait_phase_var=validation_wait_phase_var,
            validation_debug_wait_phase_var=validation_wait_phase_var,
            validation_wait_remaining_s_var=validation_wait_remaining_s_var,
            validation_debug_wait_remaining_s_var=validation_wait_remaining_s_var,
            validation_current_repeat_var=validation_current_repeat_var,
            validation_debug_current_repeat_var=validation_current_repeat_var,
            validation_result_var=validation_result_var,
            validation_debug_result_var=validation_result_var,
            validation_error_var=validation_error_var,
            validation_debug_error_var=validation_error_var,
            validation_export_path_var=validation_export_path_var,
            validation_debug_export_path_var=validation_export_path_var,
            validation_current_metric_value_var=tk.StringVar(master=root, value=d.validation_current_metric_value),
            validation_current_section_var=tk.StringVar(master=root, value=d.validation_current_section),
            validation_current_z_pos_var=tk.StringVar(master=root, value=d.validation_current_z_pos),
            validation_current_concentricity_var=tk.StringVar(master=root, value=d.validation_current_concentricity),
            validation_summary_count_var=tk.StringVar(master=root, value=d.validation_summary_count),
            validation_summary_mean_var=tk.StringVar(master=root, value=d.validation_summary_mean),
            validation_summary_std_var=tk.StringVar(master=root, value=d.validation_summary_std),
            validation_summary_min_var=tk.StringVar(master=root, value=d.validation_summary_min),
            validation_summary_max_var=tk.StringVar(master=root, value=d.validation_summary_max),
            validation_summary_range_var=tk.StringVar(master=root, value=d.validation_summary_range),
            pipe_len_var=tk.StringVar(master=root, value=d.pipe_len),
            len_enable_var=tk.BooleanVar(master=root, value=d.len_enable),
            len_z_low_approach_var=tk.StringVar(master=root, value=d.len_z_low_approach),
            len_low_search_dist_var=tk.StringVar(master=root, value=d.len_low_search_dist),
            len_high_search_dist_var=tk.StringVar(master=root, value=d.len_high_search_dist),
            len_search_vel_var=tk.StringVar(master=root, value=d.len_search_vel),
            len_search_timeout_var=tk.StringVar(master=root, value=d.len_search_timeout),
            len_tol_var=tk.StringVar(master=root, value=d.len_tol),
            len_high_margin_var=tk.StringVar(master=root, value=d.len_high_margin),
            len_debounce_k_var=tk.StringVar(master=root, value=d.len_debounce_k),
            len_max_stale_ms_var=tk.StringVar(master=root, value=d.len_max_stale_ms),
            len_backoff_var=tk.StringVar(master=root, value=d.len_backoff),
            len_info_var=tk.StringVar(master=root, value=d.len_info),
            len_status_var=tk.StringVar(master=root, value=d.len_status),
            len_edge_state_var=tk.StringVar(master=root, value=d.len_edge_state),
            len_edge_low_var=tk.StringVar(master=root, value=d.len_edge_low),
            len_edge_high_var=tk.StringVar(master=root, value=d.len_edge_high),
            len_edge_len_var=tk.StringVar(master=root, value=d.len_edge_len),
            center_pos_var=tk.StringVar(master=root, value=d.center_pos),
            teach_axes_mode_var=tk.IntVar(master=root, value=d.teach_axes_mode),
            teach_rel_dist_var=tk.StringVar(master=root, value=d.teach_rel_dist),
            teach_abs_var=tk.StringVar(master=root, value=d.teach_abs),
            teach_z_var=tk.StringVar(master=root, value=d.teach_z),
            teach_align_var=tk.StringVar(master=root, value=d.teach_align),
            teach_mode_var=tk.StringVar(master=root, value=d.teach_mode),
            teach_axes_var=tk.StringVar(master=root, value=d.teach_axes),
            start_info_var=tk.StringVar(master=root, value=d.start_info),
            standby_info_var=tk.StringVar(master=root, value=d.standby_info),
            standby_state_var=tk.StringVar(master=root, value=d.standby_state),
            sim_gauge_var=tk.IntVar(master=root, value=d.sim_gauge),
            baud_var=tk.StringVar(master=root, value=d.baud),
            req_cmd_var=tk.StringVar(master=root, value=d.req_cmd),
            gauge_conn_var=tk.StringVar(master=root, value=d.gauge_conn),
            gauge_last_var=tk.StringVar(master=root, value=d.gauge_last),
            gauge_err_var=tk.StringVar(master=root, value=d.gauge_err),
            odcal_out2_hint_var=tk.StringVar(master=root, value=d.odcal_out2_hint),
            odcal_duration_label_var=tk.StringVar(master=root, value=d.odcal_duration_label),
            odcal_adv_open_var=tk.BooleanVar(master=root, value=d.odcal_adv_open),
            odcal_cmd_var=tk.StringVar(master=root, value=d.odcal_cmd),
            odcal_dref_var=tk.StringVar(master=root, value=d.odcal_dref),
            odcal_map_out1_var=tk.StringVar(master=root, value=d.odcal_map_out1),
            odcal_mode_var=tk.StringVar(master=root, value=d.odcal_mode),
            odcal_hz_var=tk.StringVar(master=root, value=d.odcal_hz),
            odcal_duration_var=tk.StringVar(master=root, value=d.odcal_duration),
            odcal_rot_degps_var=tk.StringVar(master=root, value=d.odcal_rot_degps),
            odcal_angle_src_var=tk.StringVar(master=root, value=d.odcal_angle_src),
            odcal_filter_var=tk.StringVar(master=root, value=d.odcal_filter),
            odcal_outlier_sigma_var=tk.StringVar(master=root, value=d.odcal_outlier_sigma),
            odcal_defect_dyn_enable_var=tk.IntVar(master=root, value=d.odcal_defect_dyn_enable),
            odcal_state_var=tk.StringVar(master=root, value=d.odcal_state),
            odcal_msg_var=tk.StringVar(master=root, value=d.odcal_msg),
            odcal_defect_mode_var=tk.StringVar(master=root, value=d.odcal_defect_mode),
            odcal_defect_shift_var=tk.StringVar(master=root, value=d.odcal_defect_shift),
            odcal_defects_var=tk.StringVar(master=root, value=d.odcal_defects),
            odcal_B_candidate_var=tk.StringVar(master=root, value=d.odcal_B_candidate),
            odcal_B_active_var=tk.StringVar(master=root, value=d.odcal_B_active),
            odcal_n_var=tk.StringVar(master=root, value=d.odcal_n),
            odcal_elapsed_var=tk.StringVar(master=root, value=d.odcal_elapsed),
            odcal_sum_mean_var=tk.StringVar(master=root, value=d.odcal_sum_mean),
            odcal_sum_std_var=tk.StringVar(master=root, value=d.odcal_sum_std),
            odcal_sum_min_var=tk.StringVar(master=root, value=d.odcal_sum_min),
            odcal_sum_max_var=tk.StringVar(master=root, value=d.odcal_sum_max),
            odcal_drop_rate_var=tk.StringVar(master=root, value=d.odcal_drop_rate),
        )


__all__ = ["UiState", "UiStateDefaults"]
