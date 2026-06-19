from __future__ import annotations

from ui.presenters.axis_cal_presenter_deps import AxisCalUiState


def test_axis_cal_ui_state_keeps_field_identity() -> None:
    vars_by_key = {"sign": object()}
    field_status = {"sign": object()}
    status = {"off_abs": object()}

    state = AxisCalUiState(
        axis_cal_vars=vars_by_key,
        axis_cal_field_status_vars=field_status,
        axis_cal_status_vars=status,
    )

    assert state.axis_cal_vars is vars_by_key
    assert state.axis_cal_field_status_vars is field_status
    assert state.axis_cal_status_vars is status
