from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ValidationUiState:
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
    _widgets: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def ensure_vars(self, _master: Any) -> None:
        return None

    def remember_widget(self, name: str, widget: Any) -> Any:
        self._widgets[name] = widget
        return widget

    def widget(self, name: str) -> Any:
        return self._widgets.get(name)


__all__ = ["ValidationUiState"]
