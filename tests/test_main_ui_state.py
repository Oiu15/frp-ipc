from __future__ import annotations

from typing import Any

from ui.presenters.main_presenter_deps import MainUiState


def _make_state() -> tuple[MainUiState, dict[str, Any]]:
    values: dict[str, Any] = {
        "pipe_sn_var": object(),
        "meas_seq_var": object(),
        "meas_start_var": object(),
        "meas_elapsed_var": object(),
        "auto_progress_var": object(),
        "auto_done_var": object(),
        "auto_state_var": object(),
        "ui_meas_mode_var": object(),
        "auto_msg_var": object(),
        "od_mean_var": object(),
        "od_range_var": object(),
        "max_od_pp_var": object(),
        "max_od_pp_rob_var": object(),
        "max_od_fit_res_var": object(),
        "od_tilt_var": object(),
        "od_slope_var": object(),
        "od_endoff_var": object(),
        "id_mean_var": object(),
        "id_range_var": object(),
        "id_tilt_var": object(),
        "id_slope_var": object(),
        "id_endoff_var": object(),
        "max_id_round_var": object(),
        "axis_dist_var": object(),
        "conc_max_var": object(),
        "axis_span_max_var": object(),
        "len_meas_var": object(),
        "cov_var": object(),
    }
    return MainUiState(**values), values


def test_main_ui_state_keeps_explicit_field_identity() -> None:
    state, values = _make_state()

    for name, value in values.items():
        assert getattr(state, name) is value


def test_main_ui_state_remembers_widgets_and_view_state() -> None:
    state, _values = _make_state()
    widget = object()
    columns = ("idx", "x_ui")

    assert state.remember_widget("result_tree", widget) is widget
    assert state.widget("result_tree") is widget
    assert state.widget("missing") is None

    assert state.remember_view_state("tree_displaycols_sync", columns) is columns
    assert state.view_state("tree_displaycols_sync") is columns
    assert state.view_state("missing", "fallback") == "fallback"
