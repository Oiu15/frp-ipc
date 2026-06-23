from __future__ import annotations

"""Compatibility accessors for UiState-backed Tk variables."""

import tkinter as tk
from typing import Any


def _ui_state_property(name: str) -> property:
    def fget(self: UiStateCompatMixin) -> Any:
        return self._get_ui_state_var(name)

    return property(fget)


class UiStateCompatMixin:
    """Expose legacy AppHost variable attributes while UiState is introduced."""

    def _get_ui_state_var(self, name: str) -> Any:
        ui = self.__dict__.get("ui")
        if ui is not None and hasattr(ui, name):
            return getattr(ui, name)
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(name)

    def _set_ui_state_var(self, name: str, value: Any) -> None:
        ui = self.__dict__.get("ui")
        if ui is not None and hasattr(ui, name):
            setattr(ui, name, value)
            return
        self.__dict__[name] = value

    @property
    def pipe_sn_var(self) -> tk.StringVar:
        return self._get_ui_state_var("pipe_sn_var")

    @pipe_sn_var.setter
    def pipe_sn_var(self, value: Any) -> None:
        self._set_ui_state_var("pipe_sn_var", value)

    @property
    def meas_seq_var(self) -> tk.StringVar:
        return self._get_ui_state_var("meas_seq_var")

    @meas_seq_var.setter
    def meas_seq_var(self, value: Any) -> None:
        self._set_ui_state_var("meas_seq_var", value)

    @property
    def meas_start_var(self) -> tk.StringVar:
        return self._get_ui_state_var("meas_start_var")

    @meas_start_var.setter
    def meas_start_var(self, value: Any) -> None:
        self._set_ui_state_var("meas_start_var", value)

    @property
    def meas_elapsed_var(self) -> tk.StringVar:
        return self._get_ui_state_var("meas_elapsed_var")

    @meas_elapsed_var.setter
    def meas_elapsed_var(self, value: Any) -> None:
        self._set_ui_state_var("meas_elapsed_var", value)

    @property
    def ui_meas_mode_var(self) -> tk.StringVar:
        return self._get_ui_state_var("ui_meas_mode_var")

    @ui_meas_mode_var.setter
    def ui_meas_mode_var(self, value: Any) -> None:
        self._set_ui_state_var("ui_meas_mode_var", value)

    @property
    def auto_state_var(self) -> tk.StringVar:
        return self._get_ui_state_var("auto_state_var")

    @auto_state_var.setter
    def auto_state_var(self, value: Any) -> None:
        self._set_ui_state_var("auto_state_var", value)

    @property
    def auto_msg_var(self) -> tk.StringVar:
        return self._get_ui_state_var("auto_msg_var")

    @auto_msg_var.setter
    def auto_msg_var(self, value: Any) -> None:
        self._set_ui_state_var("auto_msg_var", value)

    @property
    def auto_progress_var(self) -> tk.StringVar:
        return self._get_ui_state_var("auto_progress_var")

    @auto_progress_var.setter
    def auto_progress_var(self, value: Any) -> None:
        self._set_ui_state_var("auto_progress_var", value)

    @property
    def auto_done_var(self) -> tk.StringVar:
        return self._get_ui_state_var("auto_done_var")

    @auto_done_var.setter
    def auto_done_var(self, value: Any) -> None:
        self._set_ui_state_var("auto_done_var", value)

    @property
    def cov_var(self) -> tk.StringVar:
        return self._get_ui_state_var("cov_var")

    @cov_var.setter
    def cov_var(self, value: Any) -> None:
        self._set_ui_state_var("cov_var", value)

    @property
    def straight_var(self) -> tk.StringVar:
        return self._get_ui_state_var("straight_var")

    @straight_var.setter
    def straight_var(self, value: Any) -> None:
        self._set_ui_state_var("straight_var", value)

    @property
    def straight_od_var(self) -> tk.StringVar:
        return self._get_ui_state_var("straight_od_var")

    @straight_od_var.setter
    def straight_od_var(self, value: Any) -> None:
        self._set_ui_state_var("straight_od_var", value)

    @property
    def straight_id_var(self) -> tk.StringVar:
        return self._get_ui_state_var("straight_id_var")

    @straight_id_var.setter
    def straight_id_var(self, value: Any) -> None:
        self._set_ui_state_var("straight_id_var", value)

    @property
    def conc_var(self) -> tk.StringVar:
        return self._get_ui_state_var("conc_var")

    @conc_var.setter
    def conc_var(self, value: Any) -> None:
        self._set_ui_state_var("conc_var", value)

    @property
    def conc_max_var(self) -> tk.StringVar:
        return self._get_ui_state_var("conc_max_var")

    @conc_max_var.setter
    def conc_max_var(self, value: Any) -> None:
        self._set_ui_state_var("conc_max_var", value)

    @property
    def axis_dist_var(self) -> tk.StringVar:
        return self._get_ui_state_var("axis_dist_var")

    @axis_dist_var.setter
    def axis_dist_var(self, value: Any) -> None:
        self._set_ui_state_var("axis_dist_var", value)

    @property
    def axis_span_max_var(self) -> tk.StringVar:
        return self._get_ui_state_var("axis_span_max_var")

    @axis_span_max_var.setter
    def axis_span_max_var(self, value: Any) -> None:
        self._set_ui_state_var("axis_span_max_var", value)

    @property
    def geom_v2_tau_var(self) -> tk.StringVar:
        return self._get_ui_state_var("geom_v2_tau_var")

    @geom_v2_tau_var.setter
    def geom_v2_tau_var(self, value: Any) -> None:
        self._set_ui_state_var("geom_v2_tau_var", value)

    validation_section_name_var = _ui_state_property("validation_section_name_var")
    validation_debug_section_name_var = _ui_state_property("validation_debug_section_name_var")
    validation_metric_name_var = _ui_state_property("validation_metric_name_var")
    validation_debug_metric_name_var = _ui_state_property("validation_debug_metric_name_var")
    validation_repeat_count_var = _ui_state_property("validation_repeat_count_var")
    validation_debug_repeat_count_var = _ui_state_property("validation_debug_repeat_count_var")
    validation_reclamp_between_repeats_var = _ui_state_property("validation_reclamp_between_repeats_var")
    validation_debug_reclamp_between_repeats_var = _ui_state_property("validation_debug_reclamp_between_repeats_var")
    validation_reclamp_enabled_var = _ui_state_property("validation_reclamp_enabled_var")
    validation_debug_reclamp_enabled_var = _ui_state_property("validation_debug_reclamp_enabled_var")
    validation_rotation_stop_before_measure_var = _ui_state_property("validation_rotation_stop_before_measure_var")
    validation_debug_rotation_stop_before_measure_var = _ui_state_property(
        "validation_debug_rotation_stop_before_measure_var"
    )
    validation_release_settle_s_var = _ui_state_property("validation_release_settle_s_var")
    validation_debug_release_settle_s_var = _ui_state_property("validation_debug_release_settle_s_var")
    validation_clamp_settle_s_var = _ui_state_property("validation_clamp_settle_s_var")
    validation_debug_clamp_settle_s_var = _ui_state_property("validation_debug_clamp_settle_s_var")
    validation_position_settle_s_var = _ui_state_property("validation_position_settle_s_var")
    validation_debug_position_settle_s_var = _ui_state_property("validation_debug_position_settle_s_var")
    validation_sample_delay_s_var = _ui_state_property("validation_sample_delay_s_var")
    validation_debug_sample_delay_s_var = _ui_state_property("validation_debug_sample_delay_s_var")
    validation_ax3_speed_dps_var = _ui_state_property("validation_ax3_speed_dps_var")
    validation_debug_ax3_speed_dps_var = _ui_state_property("validation_debug_ax3_speed_dps_var")
    validation_move_enabled_var = _ui_state_property("validation_move_enabled_var")
    validation_debug_move_enabled_var = _ui_state_property("validation_debug_move_enabled_var")
    validation_move_channel_var = _ui_state_property("validation_move_channel_var")
    validation_debug_move_channel_var = _ui_state_property("validation_debug_move_channel_var")
    validation_move_away_delta_mm_var = _ui_state_property("validation_move_away_delta_mm_var")
    validation_debug_move_away_delta_mm_var = _ui_state_property("validation_debug_move_away_delta_mm_var")
    validation_move_scenario_var = _ui_state_property("validation_move_scenario_var")
    validation_debug_move_scenario_var = _ui_state_property("validation_debug_move_scenario_var")
    validation_move_from_section_var = _ui_state_property("validation_move_from_section_var")
    validation_debug_move_from_section_var = _ui_state_property("validation_debug_move_from_section_var")
    validation_move_target_section_var = _ui_state_property("validation_move_target_section_var")
    validation_debug_move_target_section_var = _ui_state_property("validation_debug_move_target_section_var")
    validation_move_return_section_var = _ui_state_property("validation_move_return_section_var")
    validation_debug_move_return_section_var = _ui_state_property("validation_debug_move_return_section_var")
    validation_move_target_pos_var = _ui_state_property("validation_move_target_pos_var")
    validation_debug_move_target_pos_var = _ui_state_property("validation_debug_move_target_pos_var")
    validation_move_actual_pos_var = _ui_state_property("validation_move_actual_pos_var")
    validation_debug_move_actual_pos_var = _ui_state_property("validation_debug_move_actual_pos_var")
    validation_status_var = _ui_state_property("validation_status_var")
    validation_debug_status_var = _ui_state_property("validation_debug_status_var")
    validation_phase_var = _ui_state_property("validation_phase_var")
    validation_debug_phase_var = _ui_state_property("validation_debug_phase_var")
    validation_wait_phase_var = _ui_state_property("validation_wait_phase_var")
    validation_debug_wait_phase_var = _ui_state_property("validation_debug_wait_phase_var")
    validation_wait_remaining_s_var = _ui_state_property("validation_wait_remaining_s_var")
    validation_debug_wait_remaining_s_var = _ui_state_property("validation_debug_wait_remaining_s_var")
    validation_current_repeat_var = _ui_state_property("validation_current_repeat_var")
    validation_debug_current_repeat_var = _ui_state_property("validation_debug_current_repeat_var")
    validation_result_var = _ui_state_property("validation_result_var")
    validation_debug_result_var = _ui_state_property("validation_debug_result_var")
    validation_error_var = _ui_state_property("validation_error_var")
    validation_debug_error_var = _ui_state_property("validation_debug_error_var")
    validation_export_path_var = _ui_state_property("validation_export_path_var")
    validation_debug_export_path_var = _ui_state_property("validation_debug_export_path_var")
    validation_current_metric_value_var = _ui_state_property("validation_current_metric_value_var")
    validation_current_section_var = _ui_state_property("validation_current_section_var")
    validation_current_z_pos_var = _ui_state_property("validation_current_z_pos_var")
    validation_current_concentricity_var = _ui_state_property("validation_current_concentricity_var")
    validation_summary_count_var = _ui_state_property("validation_summary_count_var")
    validation_summary_mean_var = _ui_state_property("validation_summary_mean_var")
    validation_summary_std_var = _ui_state_property("validation_summary_std_var")
    validation_summary_min_var = _ui_state_property("validation_summary_min_var")
    validation_summary_max_var = _ui_state_property("validation_summary_max_var")
    validation_summary_range_var = _ui_state_property("validation_summary_range_var")
    pipe_len_var = _ui_state_property("pipe_len_var")
    len_enable_var = _ui_state_property("len_enable_var")
    len_z_low_approach_var = _ui_state_property("len_z_low_approach_var")
    len_low_search_dist_var = _ui_state_property("len_low_search_dist_var")
    len_high_search_dist_var = _ui_state_property("len_high_search_dist_var")
    len_search_vel_var = _ui_state_property("len_search_vel_var")
    len_search_timeout_var = _ui_state_property("len_search_timeout_var")
    len_tol_var = _ui_state_property("len_tol_var")
    len_high_margin_var = _ui_state_property("len_high_margin_var")
    len_debounce_k_var = _ui_state_property("len_debounce_k_var")
    len_max_stale_ms_var = _ui_state_property("len_max_stale_ms_var")
    len_backoff_var = _ui_state_property("len_backoff_var")
    len_info_var = _ui_state_property("len_info_var")
    len_status_var = _ui_state_property("len_status_var")
    len_edge_state_var = _ui_state_property("len_edge_state_var")
    len_edge_low_var = _ui_state_property("len_edge_low_var")
    len_edge_high_var = _ui_state_property("len_edge_high_var")
    len_edge_len_var = _ui_state_property("len_edge_len_var")
    center_pos_var = _ui_state_property("center_pos_var")
    teach_axes_mode_var = _ui_state_property("teach_axes_mode_var")
    teach_rel_dist_var = _ui_state_property("teach_rel_dist_var")
    teach_abs_var = _ui_state_property("teach_abs_var")
    teach_z_var = _ui_state_property("teach_z_var")
    teach_align_var = _ui_state_property("teach_align_var")
    teach_mode_var = _ui_state_property("teach_mode_var")
    teach_axes_var = _ui_state_property("teach_axes_var")
    start_info_var = _ui_state_property("start_info_var")
    standby_info_var = _ui_state_property("standby_info_var")
    standby_state_var = _ui_state_property("standby_state_var")
    sim_gauge_var = _ui_state_property("sim_gauge_var")
    baud_var = _ui_state_property("baud_var")
    req_cmd_var = _ui_state_property("req_cmd_var")
    gauge_conn_var = _ui_state_property("gauge_conn_var")
    gauge_last_var = _ui_state_property("gauge_last_var")
    gauge_err_var = _ui_state_property("gauge_err_var")
    odcal_out2_hint_var = _ui_state_property("odcal_out2_hint_var")
    odcal_duration_label_var = _ui_state_property("odcal_duration_label_var")
    odcal_adv_open_var = _ui_state_property("odcal_adv_open_var")
    odcal_cmd_var = _ui_state_property("odcal_cmd_var")
    odcal_dref_var = _ui_state_property("odcal_dref_var")
    odcal_map_out1_var = _ui_state_property("odcal_map_out1_var")
    odcal_mode_var = _ui_state_property("odcal_mode_var")
    odcal_hz_var = _ui_state_property("odcal_hz_var")
    odcal_duration_var = _ui_state_property("odcal_duration_var")
    odcal_rot_degps_var = _ui_state_property("odcal_rot_degps_var")
    odcal_angle_src_var = _ui_state_property("odcal_angle_src_var")
    odcal_filter_var = _ui_state_property("odcal_filter_var")
    odcal_outlier_sigma_var = _ui_state_property("odcal_outlier_sigma_var")
    odcal_defect_dyn_enable_var = _ui_state_property("odcal_defect_dyn_enable_var")
    odcal_state_var = _ui_state_property("odcal_state_var")
    odcal_msg_var = _ui_state_property("odcal_msg_var")
    odcal_defect_mode_var = _ui_state_property("odcal_defect_mode_var")
    odcal_defect_shift_var = _ui_state_property("odcal_defect_shift_var")
    odcal_defects_var = _ui_state_property("odcal_defects_var")
    odcal_B_candidate_var = _ui_state_property("odcal_B_candidate_var")
    odcal_B_active_var = _ui_state_property("odcal_B_active_var")
    odcal_n_var = _ui_state_property("odcal_n_var")
    odcal_elapsed_var = _ui_state_property("odcal_elapsed_var")
    odcal_sum_mean_var = _ui_state_property("odcal_sum_mean_var")
    odcal_sum_std_var = _ui_state_property("odcal_sum_std_var")
    odcal_sum_min_var = _ui_state_property("odcal_sum_min_var")
    odcal_sum_max_var = _ui_state_property("odcal_sum_max_var")
    odcal_drop_rate_var = _ui_state_property("odcal_drop_rate_var")


__all__ = ["UiStateCompatMixin"]
