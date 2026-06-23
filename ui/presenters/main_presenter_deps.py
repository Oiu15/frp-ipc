from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MainUiState:
    pipe_sn_var: Any
    meas_seq_var: Any
    meas_start_var: Any
    meas_elapsed_var: Any
    auto_progress_var: Any
    auto_done_var: Any
    auto_state_var: Any
    ui_meas_mode_var: Any
    auto_msg_var: Any
    od_mean_var: Any
    od_range_var: Any
    max_od_pp_var: Any
    max_od_pp_rob_var: Any
    max_od_fit_res_var: Any
    od_tilt_var: Any
    od_slope_var: Any
    od_endoff_var: Any
    id_mean_var: Any
    id_range_var: Any
    id_tilt_var: Any
    id_slope_var: Any
    id_endoff_var: Any
    max_id_round_var: Any
    axis_dist_var: Any
    conc_max_var: Any
    axis_span_max_var: Any
    geom_v2_tau_var: Any
    len_meas_var: Any
    cov_var: Any
    _widgets: dict[str, Any] = field(default_factory=dict)
    _view_state: dict[str, Any] = field(default_factory=dict)

    def remember_widget(self, name: str, widget: Any) -> Any:
        self._widgets[name] = widget
        return widget

    def widget(self, name: str) -> Any:
        return self._widgets.get(name)

    def remember_view_state(self, name: str, value: Any) -> Any:
        self._view_state[name] = value
        return value

    def view_state(self, name: str, default: Any = None) -> Any:
        return self._view_state.get(name, default)


__all__ = ["MainUiState"]
