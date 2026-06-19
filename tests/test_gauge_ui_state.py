from __future__ import annotations

from dataclasses import fields

import pytest

from ui.presenters.gauge_presenter_deps import GaugeUiState


def test_gauge_ui_state_keeps_field_identity_and_lists_ports() -> None:
    ports = ["COM1", "COM2"]
    values = {
        field.name: object()
        for field in fields(GaugeUiState)
        if field.init and field.name not in {"list_serial_ports_callback", "sim_gauge_enabled"}
    }
    state = GaugeUiState(
        list_serial_ports_callback=lambda: ports,
        sim_gauge_enabled=True,
        **values,
    )

    for name, value in values.items():
        assert getattr(state, name) is value
        assert state.get_var(name) is value

    assert state.get_flag("sim_gauge_enabled") is True
    assert state.get_flag("other", default=True) is True
    assert state.list_serial_ports() == ["COM1", "COM2"]

    ports.append("COM3")
    assert state.list_serial_ports() == ["COM1", "COM2", "COM3"]


def test_gauge_ui_state_maps_validation_debug_aliases_to_canonical_vars() -> None:
    values = {
        field.name: object()
        for field in fields(GaugeUiState)
        if field.init and field.name not in {"list_serial_ports_callback", "sim_gauge_enabled"}
    }
    state = GaugeUiState(
        list_serial_ports_callback=list,
        sim_gauge_enabled=False,
        **values,
    )

    assert state.get_var("validation_debug_status_var") is values["validation_status_var"]
    assert state.get_var("validation_debug_section_name_var") is values["validation_section_name_var"]
    assert state.get_var("validation_debug_export_path_var") is values["validation_export_path_var"]
    with pytest.raises(AttributeError):
        state.get_var("unknown_var")
