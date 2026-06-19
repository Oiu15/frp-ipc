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
from ui.presenters.recipe_presenter_deps import RecipePresenterDeps


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

    def refresh_axis_panel(self) -> None:
        self.calls.append(("refresh_axis_panel",))

    def dispatch_axis_action(self, action_name: str) -> None:
        self.calls.append(("dispatch_axis_action", action_name))

    def jog_hold(self, direction: str, on: bool) -> None:
        self.calls.append(("jog_hold", direction, on))


class _FakeAxisView:
    def __init__(self, *, axis_count: int = 5) -> None:
        self.axis_idx = FakeVar(0)
        self._axis_count = int(axis_count)
        self.refresh_calls = 0

    def axis_count(self) -> int:
        return self._axis_count

    def get_axis_index_var(self) -> FakeVar:
        return self.axis_idx

    def refresh_axis_panel(self) -> None:
        self.refresh_calls += 1


class _FakeGaugeController:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def set_gauge_request_command(self, cmd: str) -> str:
        self.commands.append(cmd)
        return cmd


class _FakeGaugeView:
    def __init__(self, *, sim_enabled: bool = False, ports: list[str] | None = None, calibration_controller: Any = None) -> None:
        self.sim_gauge_enabled = bool(sim_enabled)
        self.ports = list(ports or [])
        self._calibration_controller = object() if calibration_controller is None else calibration_controller
        self.vars: dict[str, tk.Variable] = {}

    def get_var(self, name: str) -> Any:
        if name in self.vars:
            return self.vars[name]
        raise AttributeError(name)

    def get_flag(self, name: str, default: bool = False) -> bool:
        if name == "sim_gauge_enabled":
            return self.sim_gauge_enabled
        return default

    def list_serial_ports(self) -> list[str]:
        return list(self.ports)

    def calibration_controller(self) -> Any:
        return self._calibration_controller


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


def _recipe_deps(host: _FakeRecipeHost) -> RecipePresenterDeps:
    def set_recipe(value: Any) -> None:
        host.recipe = value

    def refresh_recipe_table() -> None:
        host.calls += 1

    return RecipePresenterDeps(
        get_recipe=lambda: host.recipe,
        set_recipe=set_recipe,
        axis_cal=host.axis_cal,
        after_recipe_data_applied=(refresh_recipe_table,),
    )


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
        self.axis_cal_vars = {"sign": FakeVar("1")}
        self.axis_cal_field_status_vars = {"sign": FakeVar("未读取")}
        self.axis_cal_status_vars = {"off_abs": FakeVar("-")}
        self.axis_span_max_var = FakeVar("--")
        self.keytest_x_vars = [FakeVar(0)]
        self.keytest_y_vars = [FakeVar(0)]
        self.keytest_y_lastcmd_vars = [FakeVar("--")]
        self.validation_status_var = FakeVar("IDLE")
        self.secret_state = "hidden"
        self._private_state = "private"
        self.calls = 0
        self.recipe: object | None = None

    def _refresh_main_summary_panel(self) -> str:
        self.calls += 1
        return "refreshed"

    def secret_method(self) -> None:
        self.calls += 1


class TestScreenPresenter:
    def test_axis_presenter_tracks_current_axis_and_forwards_intent(self) -> None:
        view = _FakeAxisView(axis_count=5)
        controller = _FakeAxisController()
        presenter = AxisScreenPresenter(view, controller)
        presenter.register_axis_widgets(2, {"ent_pos": object()}, FakeVar(0))

        presenter.handle_axis_selected(2)
        presenter.handle_action(2, "_do_movea")
        presenter.handle_jog(2, "fwd", True)

        assert view.axis_idx.get() == 2
        assert presenter.current_axis == 2
        assert presenter.current_widget("ent_pos") is not None
        assert controller.calls == [
            ("refresh_axis_panel",),
            ("dispatch_axis_action", "_do_movea"),
            ("jog_hold", "fwd", True),
        ]

    def test_axis_presenter_blocks_undeclared_host_state_and_methods(self) -> None:
        view = _FakeAxisView()
        presenter = AxisScreenPresenter(view, _FakeAxisController())

        assert presenter.axis_idx is view.axis_idx

        with pytest.raises(AttributeError):
            _ = presenter.some_state
        with pytest.raises(AttributeError):
            _ = presenter._axis_snapshot
        with pytest.raises(AttributeError):
            presenter.secret_method()

    def test_axis_presenter_clamps_axis_and_refreshes_via_controller(self) -> None:
        view = _FakeAxisView(axis_count=3)
        controller = _FakeAxisController()
        presenter = AxisScreenPresenter(view, controller)

        selected = presenter.activate_axis(99)
        presenter.handle_axis_selected(-10)

        assert selected == 2
        assert view.axis_idx.get() == 0
        assert presenter.current_axis == 0
        assert controller.calls == [("refresh_axis_panel",)]

    def test_gauge_presenter_translates_request_change_to_controller_intent(self) -> None:
        view = _FakeGaugeView()
        controller = _FakeGaugeController()
        presenter = GaugeScreenPresenter(view, controller)

        presenter.handle_request_command_changed("M0,1")
        presenter.handle_request_command_changed("")

        assert controller.commands == ["M0,1", "M1,1"]

    def test_gauge_presenter_blocks_undeclared_host_state_and_methods(self) -> None:
        root = tk.Tcl()
        calibration_controller = object()
        view = _FakeGaugeView(calibration_controller=calibration_controller)
        view.vars["gauge_conn_var"] = tk.StringVar(master=root, value="connected")
        presenter = GaugeScreenPresenter(view, _FakeGaugeController())

        presenter.ensure_vars(root)

        assert presenter.gauge_conn_var is view.vars["gauge_conn_var"]
        assert presenter.calibration_controller is calibration_controller

        with pytest.raises(AttributeError):
            _ = presenter.some_state
        with pytest.raises(AttributeError):
            presenter.secret_method()

    def test_gauge_presenter_allows_declared_host_backed_view_state(self) -> None:
        root = tk.Tcl()
        view = _FakeGaugeView()
        for name in (
            "ip_var",
            "port_var",
            "plc_status_var",
            "cl_out1_var",
            "cl_out1_cnt_var",
            "cl_m_calc_var",
            "cl_m_diff_var",
        ):
            view.vars[name] = tk.StringVar(master=root, value=name)
        presenter = GaugeScreenPresenter(view, _FakeGaugeController())

        assert presenter.ip_var is view.vars["ip_var"]
        assert presenter.port_var is view.vars["port_var"]
        assert presenter.plc_status_var is view.vars["plc_status_var"]
        assert presenter.cl_out1_var is view.vars["cl_out1_var"]
        assert presenter.cl_out1_cnt_var is view.vars["cl_out1_cnt_var"]
        assert presenter.cl_m_calc_var is view.vars["cl_m_calc_var"]
        assert presenter.cl_m_diff_var is view.vars["cl_m_diff_var"]

    def test_recipe_presenter_allows_declared_calls_and_blocks_unknown_host_access(self) -> None:
        host = _FakeRecipeHost()
        presenter = RecipeScreenPresenter(_recipe_deps(host))

        presenter.after_recipe_data_applied()
        assert host.calls == 1

        dynamic_presenter = cast(Any, presenter)
        with pytest.raises(AttributeError):
            _ = dynamic_presenter.secret_state
        with pytest.raises(AttributeError):
            dynamic_presenter.secret_method()

    def test_gauge_presenter_initializes_validation_progress_vars(self) -> None:
        view = _FakeGaugeView()
        controller = _FakeGaugeController()
        presenter = GaugeScreenPresenter(view, controller)
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

    def test_gauge_presenter_initializes_id_calibration_vars(self) -> None:
        root = tk.Tcl()
        view = _FakeGaugeView()
        view.vars["idcal_dref_var"] = tk.StringVar(master=root, value="151.000")
        presenter = GaugeScreenPresenter(view, _FakeGaugeController())

        presenter.ensure_vars(master=root)

        assert presenter.idcal_dref_var is view.vars["idcal_dref_var"]
        assert presenter.idcal_mode_var.get() == "one_rev"
        assert presenter.idcal_state_var.get() == "IDLE"
        assert presenter.idcal_delta_candidate_var.get() == "--"
        assert presenter.idcal_chk_dtheta_var.get() == "--"
        assert presenter.id_single_cal_dref_var.get() == "150.000"
        assert presenter.id_single_cal_state_var.get() == "IDLE"
        assert presenter.id_single_cal_warn_var.get() == ""

    def test_gauge_presenter_owned_vars_do_not_write_back_to_host(self) -> None:
        view = _FakeGaugeView()
        presenter = GaugeScreenPresenter(view, _FakeGaugeController())
        root = tk.Tcl()

        presenter.ensure_vars(master=root)
        presenter.local_only_var = FakeVar("presenter")

        assert presenter.baud_var.get() == "115200"
        assert presenter.local_only_var.get() == "presenter"
        assert "baud_var" not in view.__dict__
        assert "odcal_cmd_var" not in view.__dict__
        assert "local_only_var" not in view.__dict__

    def test_gauge_presenter_uses_view_for_flags_ports_and_odcal_derived_vars(self) -> None:
        view = _FakeGaugeView(sim_enabled=True, ports=["COM1", "COM2"])
        presenter = GaugeScreenPresenter(view, _FakeGaugeController())
        root = tk.Tcl()

        presenter.ensure_vars(master=root)
        presenter.odcal_map_out1_var.set("R")
        presenter.refresh_out2_hint()
        presenter.odcal_mode_var.set("one_rev")
        presenter.refresh_odcal_duration_label()
        one_rev_label = presenter.odcal_duration_label_var.get()
        presenter.odcal_angle_src_var.set("无")
        presenter.handle_odcal_angle_source_changed()

        assert presenter.sim_gauge_var.get() == 1
        assert presenter.list_serial_ports() == ["COM1", "COM2"]
        assert presenter.odcal_out2_hint_var.get().endswith("L")
        assert one_rev_label != presenter.odcal_duration_label_var.get()
        assert presenter.odcal_mode_var.get() == "timed"

    def test_recipe_presenter_owned_vars_do_not_write_back_to_host(self) -> None:
        host = _FakeRecipeHost()
        presenter = RecipeScreenPresenter(_recipe_deps(host))
        dynamic_presenter = cast(Any, presenter)
        root = tk.Tcl()

        presenter.ensure_vars(master=root)
        dynamic_presenter.local_only_var = FakeVar("presenter")

        assert presenter.recipe is host.recipe
        assert dynamic_presenter.recipe_name_var.get() == host.recipe.name
        assert dynamic_presenter.local_only_var.get() == "presenter"
        assert "recipe_name_var" not in host.__dict__
        assert "pipe_len_var" not in host.__dict__
        assert "local_only_var" not in host.__dict__

    def test_recipe_form_mapper_reads_and_writes_presenter_owned_vars(self) -> None:
        host = _FakeRecipeHost()
        presenter = RecipeScreenPresenter(_recipe_deps(host))
        dynamic_presenter = cast(Any, presenter)
        root = tk.Tcl()
        presenter.ensure_vars(master=root)
        combo = _FakeCombo(["sync", "split"])
        presenter.remember_widget("section_sampling_mode_combo", combo)
        mapper = RecipeFormMapper(presenter)

        dynamic_presenter.recipe_name_var.set("presenter-recipe")
        dynamic_presenter.pipe_len_var.set("1888")
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
        assert dynamic_presenter.recipe_name_var.get() == "loaded"
        assert dynamic_presenter.section_sampling_mode_var.get() == "split"
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

    def test_screen_controller_blocks_migrated_validation_screen_navigation(self) -> None:
        host = _FakeValidationHost()
        controller = ScreenController(cast(Any, host))
        dynamic_controller: Any = controller

        with pytest.raises(AttributeError):
            dynamic_controller.open_validation_screen()

        assert host.navigation_calls == 0

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

    def test_screen_presenter_keeps_explicit_local_registry(self) -> None:
        host = _FakePresenterHost()
        presenter = ScreenPresenter(cast(Any, host))
        dynamic_presenter: Any = presenter

        widget = object()
        state = object()

        assert presenter.remember_widget("result_tree", widget) is widget
        assert presenter.widget("result_tree") is widget
        assert presenter.remember_view_state("selected_row", state) is state
        assert presenter.view_state("selected_row") is state

        with pytest.raises(AttributeError):
            _ = dynamic_presenter.result_tree
        with pytest.raises(AttributeError):
            _ = dynamic_presenter.selected_row
        with pytest.raises(AttributeError):
            _ = dynamic_presenter.axis_cal_vars
        with pytest.raises(AttributeError):
            _ = dynamic_presenter.axis_span_max_var
        with pytest.raises(AttributeError):
            _ = dynamic_presenter.validation_status_var
        with pytest.raises(AttributeError):
            _ = dynamic_presenter.secret_state
        with pytest.raises(AttributeError):
            dynamic_presenter.secret_method()

    def test_screen_controller_blocks_undeclared_host_methods(self) -> None:
        host = _FakePresenterHost()
        controller = ScreenController(cast(Any, host))
        dynamic_controller: Any = controller

        with pytest.raises(AttributeError):
            dynamic_controller.secret_method()

    def test_screen_controller_blocks_migrated_screen_commands(self) -> None:
        class _GaugeCommandHost:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def apply_id_calibration(self) -> None:
                self.calls.append("apply_id_calibration")

            def apply_od_b(self) -> None:
                self.calls.append("apply_od_b")

            def apply_plc_connection(self) -> None:
                self.calls.append("apply_plc_connection")

            def connect_gauge(self) -> None:
                self.calls.append("connect_gauge")

            def disconnect_gauge(self) -> None:
                self.calls.append("disconnect_gauge")

            def learn_odcal_defect_a(self) -> None:
                self.calls.append("learn_odcal_defect_a")

            def learn_odcal_defect_b(self) -> None:
                self.calls.append("learn_odcal_defect_b")

            def request_gauge_once(self) -> None:
                self.calls.append("request_gauge_once")

            def toggle_sim_gauge(self) -> None:
                self.calls.append("toggle_sim_gauge")

            def verify_id_calibration(self) -> None:
                self.calls.append("verify_id_calibration")

        names = [
            "apply_id_calibration",
            "apply_od_b",
            "apply_plc_connection",
            "connect_gauge",
            "disconnect_gauge",
            "learn_odcal_defect_a",
            "learn_odcal_defect_b",
            "request_gauge_once",
            "toggle_sim_gauge",
            "verify_id_calibration",
        ]
        host = _GaugeCommandHost()
        controller = ScreenController(cast(Any, host))

        for name in names:
            with pytest.raises(AttributeError):
                getattr(controller, name)

        assert host.calls == []

    def test_screen_ui_context_has_no_dynamic_host_state_or_callables(self) -> None:
        host = _FakePresenterHost()
        host.recipe = object()
        ui = ScreenUiContext(cast(Any, host))
        dynamic_ui: Any = ui

        with pytest.raises(AttributeError):
            _ = dynamic_ui.recipe
        with pytest.raises(AttributeError):
            _ = dynamic_ui.secret_state
        with pytest.raises(AttributeError):
            dynamic_ui.secret_method()


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
        if "__getattr__" not in presenter_type.__dict__:
            continue
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
        if guard_index < 0:
            offenders.append(presenter_type.__name__)
            continue
        if host_getattr_index >= 0 and guard_index > host_getattr_index:
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


def test_recipe_form_mapper_uses_explicit_view_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "application" / "form_mapper.py").read_text(encoding="utf-8-sig")

    for forbidden in ("host: Any", "self.host", "getattr(self.host", "_recipe_ui_widget"):
        assert forbidden not in source


def test_gauge_presenter_uses_explicit_view_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "ui" / "presenters" / "gauge_presenter.py").read_text(encoding="utf-8-sig")

    for forbidden in ("self.host.", "getattr(self.host"):
        assert forbidden not in source


def test_axis_presenter_uses_explicit_view_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "ui" / "presenters" / "axis_presenter.py").read_text(encoding="utf-8-sig")

    for forbidden in ("self.host", "getattr(self.host", "_axis_snapshot"):
        assert forbidden not in source
