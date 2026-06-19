from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


_VALIDATION_ALIASES = {
    "validation_debug_section_name_var": "validation_section_name_var",
    "validation_debug_metric_name_var": "validation_metric_name_var",
    "validation_debug_repeat_count_var": "validation_repeat_count_var",
    "validation_debug_reclamp_between_repeats_var": "validation_reclamp_between_repeats_var",
    "validation_debug_reclamp_enabled_var": "validation_reclamp_enabled_var",
    "validation_debug_rotation_stop_before_measure_var": "validation_rotation_stop_before_measure_var",
    "validation_debug_release_settle_s_var": "validation_release_settle_s_var",
    "validation_debug_clamp_settle_s_var": "validation_clamp_settle_s_var",
    "validation_debug_position_settle_s_var": "validation_position_settle_s_var",
    "validation_debug_sample_delay_s_var": "validation_sample_delay_s_var",
    "validation_debug_ax3_speed_dps_var": "validation_ax3_speed_dps_var",
    "validation_debug_move_enabled_var": "validation_move_enabled_var",
    "validation_debug_move_channel_var": "validation_move_channel_var",
    "validation_debug_move_away_delta_mm_var": "validation_move_away_delta_mm_var",
    "validation_debug_move_scenario_var": "validation_move_scenario_var",
    "validation_debug_move_from_section_var": "validation_move_from_section_var",
    "validation_debug_move_target_section_var": "validation_move_target_section_var",
    "validation_debug_move_return_section_var": "validation_move_return_section_var",
    "validation_debug_move_target_pos_var": "validation_move_target_pos_var",
    "validation_debug_move_actual_pos_var": "validation_move_actual_pos_var",
    "validation_debug_status_var": "validation_status_var",
    "validation_debug_phase_var": "validation_phase_var",
    "validation_debug_wait_phase_var": "validation_wait_phase_var",
    "validation_debug_wait_remaining_s_var": "validation_wait_remaining_s_var",
    "validation_debug_current_repeat_var": "validation_current_repeat_var",
    "validation_debug_result_var": "validation_result_var",
    "validation_debug_error_var": "validation_error_var",
    "validation_debug_export_path_var": "validation_export_path_var",
}


@dataclass(slots=True)
class GaugeUiState:
    list_serial_ports_callback: Callable[[], list[str]]
    sim_gauge_enabled: bool
    ip_var: Any
    port_var: Any
    plc_status_var: Any
    sim_gauge_var: Any
    baud_var: Any
    req_cmd_var: Any
    gauge_conn_var: Any
    gauge_last_var: Any
    gauge_err_var: Any
    odcal_out2_hint_var: Any
    odcal_duration_label_var: Any
    odcal_adv_open_var: Any
    odcal_cmd_var: Any
    odcal_dref_var: Any
    odcal_map_out1_var: Any
    odcal_mode_var: Any
    odcal_hz_var: Any
    odcal_duration_var: Any
    odcal_rot_degps_var: Any
    odcal_angle_src_var: Any
    odcal_filter_var: Any
    odcal_outlier_sigma_var: Any
    odcal_defect_dyn_enable_var: Any
    odcal_state_var: Any
    odcal_msg_var: Any
    odcal_defect_mode_var: Any
    odcal_defect_shift_var: Any
    odcal_defects_var: Any
    odcal_B_candidate_var: Any
    odcal_B_active_var: Any
    odcal_n_var: Any
    odcal_elapsed_var: Any
    odcal_sum_mean_var: Any
    odcal_sum_std_var: Any
    odcal_sum_min_var: Any
    odcal_sum_max_var: Any
    odcal_drop_rate_var: Any
    cl_out1_var: Any
    cl_out1_cnt_var: Any
    cl_out2_var: Any
    cl_out2_cnt_var: Any
    cl_out3_var: Any
    cl_out3_cnt_var: Any
    cl_out4_var: Any
    cl_out4_cnt_var: Any
    cl_out5_var: Any
    cl_out5_cnt_var: Any
    cl_m_calc_var: Any
    cl_m_diff_var: Any
    idcal_state_var: Any
    idcal_msg_var: Any
    idcal_dref_var: Any
    idcal_mode_var: Any
    idcal_hz_var: Any
    idcal_duration_var: Any
    idcal_rot_degps_var: Any
    idcal_delta_candidate_var: Any
    idcal_delta_active_var: Any
    idcal_cmax_var: Any
    idcal_mmean_var: Any
    idcal_mpp_var: Any
    idcal_fit_diam_var: Any
    idcal_fit_e_var: Any
    idcal_fit_y0_var: Any
    idcal_fit_rmse_var: Any
    idcal_chk_err_var: Any
    idcal_chk_cov_var: Any
    idcal_chk_n_var: Any
    idcal_chk_dtheta_var: Any
    id_single_cal_state_var: Any
    id_single_cal_msg_var: Any
    id_single_cal_dref_var: Any
    id_single_cal_mean_var: Any
    id_single_cal_B_var: Any
    id_single_cal_ecc_amp_var: Any
    id_single_cal_ecc_ang_var: Any
    id_single_cal_cov_var: Any
    id_single_cal_warn_var: Any
    validation_section_name_var: Any
    validation_metric_name_var: Any
    validation_repeat_count_var: Any
    validation_reclamp_between_repeats_var: Any
    validation_reclamp_enabled_var: Any
    validation_rotation_stop_before_measure_var: Any
    validation_release_settle_s_var: Any
    validation_clamp_settle_s_var: Any
    validation_position_settle_s_var: Any
    validation_sample_delay_s_var: Any
    validation_ax3_speed_dps_var: Any
    validation_move_enabled_var: Any
    validation_move_channel_var: Any
    validation_move_away_delta_mm_var: Any
    validation_move_scenario_var: Any
    validation_move_from_section_var: Any
    validation_move_target_section_var: Any
    validation_move_return_section_var: Any
    validation_move_target_pos_var: Any
    validation_move_actual_pos_var: Any
    validation_status_var: Any
    validation_phase_var: Any
    validation_wait_phase_var: Any
    validation_wait_remaining_s_var: Any
    validation_current_repeat_var: Any
    validation_result_var: Any
    validation_error_var: Any
    validation_export_path_var: Any
    validation_current_metric_value_var: Any
    validation_current_section_var: Any
    validation_current_z_pos_var: Any
    validation_current_concentricity_var: Any
    validation_summary_count_var: Any
    validation_summary_mean_var: Any
    validation_summary_std_var: Any
    validation_summary_min_var: Any
    validation_summary_max_var: Any
    validation_summary_range_var: Any

    def get_var(self, name: str) -> Any:
        field_name = _VALIDATION_ALIASES.get(name, name)
        try:
            return getattr(self, field_name)
        except AttributeError as exc:
            raise AttributeError(name) from exc

    def get_flag(self, name: str, default: bool = False) -> bool:
        if name == "sim_gauge_enabled":
            return bool(self.sim_gauge_enabled)
        return default

    def list_serial_ports(self) -> list[str]:
        return list(self.list_serial_ports_callback())


__all__ = ["GaugeUiState"]
