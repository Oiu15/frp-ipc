from __future__ import annotations

import inspect
from pathlib import Path
import tkinter as tk
from typing import Any, cast

import pytest

from tests.fakes import FakeVar

from application.form_mapper import RecipeFormMapper
from application.adapters.device_gateway import ScreenController, ScreenPresenter, ScreenUiContext
from core.models import AxisCal, Recipe
from ui.presenters.axis_presenter import AxisScreenPresenter
from ui.presenters.gauge_presenter import GaugeScreenPresenter
from ui.presenters.recipe_presenter import RecipeScreenPresenter


class _FakeHost:
    def __init__(self) -> None:
        self.axis_idx = FakeVar(0)
        self._axis_snapshot = [object() for _ in range(5)]
        self.some_state = "ok"

    def _refresh_axis_panel(self) -> str:
        return "host-refresh"

    def secret_method(self) -> None:
        pass


class _FakeAxisController:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def _refresh_axis_panel(self) -> None:
        self.calls.append(("_refresh_axis_panel",))

    def _do_movea(self) -> None:
        self.calls.append(("_do_movea",))

    def _jog_hold(self, direction: str, on: bool) -> None:
        self.calls.append(("_jog_hold", direction, on))


class _FakeGaugeController:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def set_gauge_request_command(self, cmd: str) -> str:
        self.commands.append(cmd)
        return cmd


class _FakeRecipeHost:
    def __init__(self) -> None:
        self.recipe = Recipe()
        self.axis_cal = AxisCal()
        self.secret_state = "hidden"
        self.calls = 0

    def _refresh_recipe_panel(self) -> str:
        self.calls += 1
        return "refreshed"

    def secret_method(self) -> None:
        self.calls += 1


class _FakeCombo:
    def __init__(self, values: list[str]) -> None:
        self._values = list(values)
        self.current_index: int | None = None

    def cget(self, key: str) -> Any:
        if key == "values":
            return tuple(self._values)
        return None

    def current(self, index: int) -> None:
        self.current_index = int(index)


class _FakeValidationHost:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.feedback: list[dict] = []
        self.stop_calls = 0
        self.navigation_calls = 0

    def start_validation_run(self, **kwargs):
        self.calls.append(dict(kwargs))
        return "started"

    def stop_validation_run(self):
        self.stop_calls += 1
        return "stopped"

    def _set_validation_feedback(self, **kwargs) -> None:
        self.feedback.append(dict(kwargs))

    def open_validation_screen(self):
        self.navigation_calls += 1
        return None

    start_fixed_section_repeatability_debug = start_validation_run
    stop_fixed_section_repeatability_debug = stop_validation_run
    _set_validation_debug_feedback = _set_validation_feedback


class _FakePresenterHost:
    def __init__(self) -> None:
        self.validation_status_var = FakeVar("IDLE")
        self.secret_state = "hidden"
        self._private_state = "private"
        self.calls = 0

    def _refresh_main_summary_panel(self) -> str:
        self.calls += 1
        return "refreshed"

    def secret_method(self) -> None:
        self.calls += 1


class TestScreenPresenter:
    def test_axis_presenter_tracks_current_axis_and_forwards_intent(self) -> None:
        host = _FakeHost()
        controller = _FakeAxisController()
        presenter = AxisScreenPresenter(host, controller)
        presenter.register_axis_widgets(2, {"ent_pos": object()}, FakeVar(0))

        presenter.handle_axis_selected(2)
        presenter.handle_action(2, "_do_movea")
        presenter.handle_jog(2, "fwd", True)

        assert host.axis_idx.get() == 2
        assert presenter.current_axis == 2
        assert presenter.current_widget("ent_pos") is not None
        assert controller.calls == [
            ("_refresh_axis_panel",), ("_do_movea",), ("_jog_hold", "fwd", True)
        ]

    def test_axis_presenter_blocks_undeclared_host_state_and_methods(self) -> None:
        host = _FakeHost()
        presenter = AxisScreenPresenter(host, _FakeAxisController())

        assert presenter.axis_idx is host.axis_idx
        assert presenter._refresh_axis_panel() == "host-refresh"

        with pytest.raises(AttributeError):
            _ = presenter.some_state
        with pytest.raises(AttributeError):
            _ = presenter._axis_snapshot
        with pytest.raises(AttributeError):
            presenter.secret_method()

    def test_gauge_presenter_translates_request_change_to_controller_intent(self) -> None:
        host = _FakeHost()
        controller = _FakeGaugeController()
        presenter = GaugeScreenPresenter(host, controller)

        presenter.handle_request_command_changed("M0,1")
        presenter.handle_request_command_changed("")

        assert controller.commands == ["M0,1", "M1,1"]

    def test_gauge_presenter_blocks_undeclared_host_state_and_methods(self) -> None:
        host = _FakeHost()
        host.gauge_conn_var = FakeVar("connected")
        host.calibration_controller = object()
        presenter = GaugeScreenPresenter(host, _FakeGaugeController())

        assert presenter.gauge_conn_var is host.gauge_conn_var
        assert presenter.calibration_controller is host.calibration_controller

        with pytest.raises(AttributeError):
            _ = presenter.some_state
        with pytest.raises(AttributeError):
            presenter.secret_method()

    def test_recipe_presenter_allows_declared_calls_and_blocks_unknown_host_access(self) -> None:
        host = _FakeRecipeHost()
        presenter = RecipeScreenPresenter(host)

        assert presenter._refresh_recipe_panel() == "refreshed"

        with pytest.raises(AttributeError):
            _ = presenter.secret_state
        with pytest.raises(AttributeError):
            presenter.secret_method()

    def test_gauge_presenter_initializes_validation_progress_vars(self) -> None:
        host = _FakeHost()
        controller = _FakeGaugeController()
        presenter = GaugeScreenPresenter(host, controller)
        root = tk.Tcl()

        presenter.ensure_vars(master=root)

        assert presenter.validation_phase_var.get() == "IDLE"
        assert presenter.validation_wait_phase_var.get() == ""
        assert presenter.validation_wait_remaining_s_var.get() == ""
        assert presenter.validation_current_repeat_var.get() == "0/0"
        assert presenter.validation_current_metric_value_var.get() == ""
        assert presenter.validation_current_section_var.get() == ""
        assert presenter.validation_summary_count_var.get() == "0"
        assert presenter.validation_summary_mean_var.get() == ""
        assert presenter.validation_phase_var is presenter.validation_debug_phase_var
        assert presenter.validation_section_name_var is presenter.validation_debug_section_name_var
        assert presenter.validation_status_var is presenter.validation_debug_status_var

    def test_gauge_presenter_owned_vars_do_not_write_back_to_host(self) -> None:
        host = _FakeHost()
        presenter = GaugeScreenPresenter(host, _FakeGaugeController())
        root = tk.Tcl()

        presenter.ensure_vars(master=root)
        presenter.local_only_var = FakeVar("presenter")

        assert presenter.baud_var.get() == "115200"
        assert presenter.local_only_var.get() == "presenter"
        assert "baud_var" not in host.__dict__
        assert "odcal_cmd_var" not in host.__dict__
        assert "local_only_var" not in host.__dict__

    def test_recipe_presenter_owned_vars_do_not_write_back_to_host(self) -> None:
        host = _FakeRecipeHost()
        presenter = RecipeScreenPresenter(host)
        root = tk.Tcl()

        presenter.ensure_vars(master=root)
        presenter.local_only_var = FakeVar("presenter")

        assert presenter.recipe is host.recipe
        assert presenter.recipe_name_var.get() == host.recipe.name
        assert presenter.local_only_var.get() == "presenter"
        assert "recipe_name_var" not in host.__dict__
        assert "pipe_len_var" not in host.__dict__
        assert "local_only_var" not in host.__dict__

    def test_recipe_form_mapper_reads_and_writes_presenter_owned_vars(self) -> None:
        host = _FakeRecipeHost()
        presenter = RecipeScreenPresenter(host)
        root = tk.Tcl()
        presenter.ensure_vars(master=root)
        combo = _FakeCombo(["sync", "split"])
        presenter.remember_widget("section_sampling_mode_combo", combo)
        mapper = RecipeFormMapper(presenter)

        presenter.recipe_name_var.set("presenter-recipe")
        presenter.pipe_len_var.set("1888")
        recipe = mapper.ui_vars_to_recipe()
        mapper.apply_data_to_ui(
            {
                "name": "loaded",
                "pipe_len_mm": 1700.0,
                "clamp_occupy_mm": 300.0,
                "margin_head_mm": 20.0,
                "margin_tail_mm": 20.0,
                "section_count": 2,
                "section_sampling_mode": "split",
                "points_per_rev": 180,
                "sample_coverage": 0.9,
                "section_timeout_s": 8.0,
                "max_revs": 3.0,
                "section_pos_z": [25.0, 50.0],
            }
        )

        assert recipe.name == "presenter-recipe"
        assert recipe.pipe_len_mm == pytest.approx(1888.0)
        assert presenter.recipe_name_var.get() == "loaded"
        assert presenter.section_sampling_mode_var.get() == "split"
        assert combo.current_index == 1
        assert "recipe_name_var" not in host.__dict__

    def test_screen_controller_forwards_validation_motion_options(self) -> None:
        host = _FakeValidationHost()
        controller = ScreenController(cast(Any, host))

        result = controller.start_validation_run(
            section_name=" S1 ",
            metric_name="od_avg",
            repeat_count="2",
            reclamp_enabled="true",
            rotation_stop_before_measure=True,
            release_settle_s="0.25",
            clamp_settle_s="0.5",
            position_settle_s="0.75",
            sample_delay_s="0.125",
            validation_ax3_speed_dps="45",
            move_enabled="true",
            move_channel="id_channel",
            move_away_delta_mm="12.5",
            move_scenario="switch_and_return",
            move_from_section_index="1: 100.000",
            move_target_section_index="2: 200.000",
            move_return_section_index="1: 100.000",
        )

        assert result == "started"
        assert host.calls == [
            {
                "section_name": "S1",
                "metric_name": "od_avg",
                "repeat_count": 2,
                "reclamp_between_repeats": False,
                "reclamp_enabled": True,
                "rotation_stop_before_measure": True,
                "release_settle_s": 0.25,
                "clamp_settle_s": 0.5,
                "position_settle_s": 0.75,
                "sample_delay_s": 0.125,
                "validation_ax3_speed_dps": 45.0,
                "move_enabled": True,
                "move_channel": "id_channel",
                "move_away_delta_mm": 12.5,
                "move_scenario": "switch_and_return",
                "move_from_section_index": 1,
                "move_target_section_index": 2,
                "move_return_section_index": 1,
            }
        ]

    def test_screen_controller_forwards_validation_stop(self) -> None:
        host = _FakeValidationHost()
        controller = ScreenController(cast(Any, host))

        result = controller.stop_validation_run()

        assert result == "stopped"
        assert host.stop_calls == 1

    def test_screen_controller_exposes_validation_screen_navigation(self) -> None:
        host = _FakeValidationHost()
        controller = ScreenController(cast(Any, host))

        result = controller.open_validation_screen()

        assert result is None
        assert host.navigation_calls == 1

    def test_screen_controller_validation_debug_aliases_forward_to_existing_chain(self) -> None:
        host = _FakeValidationHost()
        controller = ScreenController(cast(Any, host))

        result = controller.start_fixed_section_repeatability_debug(
            section_name="S1",
            metric_name="od_avg",
            repeat_count="1",
            move_enabled=False,
            move_channel="od_channel",
            move_away_delta_mm="0.0",
            move_scenario="distance_round_trip",
            move_from_section_index="1",
            move_target_section_index="1",
            move_return_section_index="1",
        )
        stop_result = controller.stop_fixed_section_repeatability_debug()

        assert result == "started"
        assert stop_result == "stopped"
        assert len(host.calls) == 1
        assert host.calls[0]["section_name"] == "S1"
        assert host.calls[0]["metric_name"] == "od_avg"
        assert host.stop_calls == 1

    def test_screen_presenter_allows_declared_view_state_and_blocks_unknown_host_state(self) -> None:
        host = _FakePresenterHost()
        presenter = ScreenPresenter(cast(Any, host))

        assert presenter.validation_status_var is host.validation_status_var
        assert presenter._refresh_main_summary_panel() == "refreshed"

        with pytest.raises(AttributeError):
            _ = presenter.secret_state
        with pytest.raises(AttributeError):
            _ = presenter._private_state
        with pytest.raises(AttributeError):
            presenter.secret_method()

    def test_screen_controller_blocks_undeclared_host_methods(self) -> None:
        host = _FakePresenterHost()
        controller = ScreenController(cast(Any, host))

        with pytest.raises(AttributeError):
            controller.secret_method()

    def test_screen_ui_context_blocks_undeclared_host_state_and_callables(self) -> None:
        host = _FakePresenterHost()
        host.recipe = object()
        ui = ScreenUiContext(cast(Any, host))

        assert ui.recipe is host.recipe
        with pytest.raises(AttributeError):
            _ = ui.secret_state
        with pytest.raises(AttributeError):
            ui.secret_method()


def test_presenter_getattr_fallbacks_are_guarded_by_allowlists() -> None:
    presenter_types = [
        ScreenPresenter,
        ScreenController,
        ScreenUiContext,
        AxisScreenPresenter,
        GaugeScreenPresenter,
        RecipeScreenPresenter,
    ]

    offenders: list[str] = []
    for presenter_type in presenter_types:
        source = inspect.getsource(presenter_type.__getattr__)
        guard_index = source.find("raise AttributeError")
        host_getattr_indexes = [
            idx
            for idx in (
                source.find("getattr(self.host"),
                source.find("getattr(self.host_app"),
            )
            if idx >= 0
        ]
        host_getattr_index = min(host_getattr_indexes) if host_getattr_indexes else -1
        if guard_index < 0 or host_getattr_index < 0 or guard_index > host_getattr_index:
            offenders.append(presenter_type.__name__)

    assert offenders == []


def test_presenters_do_not_write_owned_state_back_to_host() -> None:
    root = Path(__file__).resolve().parents[1]
    presenter_paths = sorted((root / "ui" / "presenters").glob("*_presenter.py"))

    offenders: list[str] = []
    for path in presenter_paths:
        source = path.read_text(encoding="utf-8-sig")
        for forbidden in ("setattr(self.host", "setattr(self.host_app"):
            if forbidden in source:
                offenders.append(f"{path.name}: {forbidden}")

    assert offenders == []
