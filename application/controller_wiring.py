from __future__ import annotations

"""Screen controller / presenter wiring (Phase 2).

This module owns the creation of ``ScreenPresenter``, ``ScreenController``,
``ScreenUiContext``, and the per-tab presenters (axis, recipe, gauge).
The presenter *classes* stay where they are; only the assembly is moved
here.
"""

from typing import Any

from application.adapters.device_gateway import (
    ScreenController,
    ScreenPresenter,
    ScreenUiContext,
)
from application.controllers.key_test_controller import KeyTestController
from application.controllers.recipe_controller import RecipeController
from ui.presenters.axis_presenter import AxisScreenPresenter
from ui.presenters.gauge_presenter import GaugeScreenPresenter
from ui.presenters.key_test_presenter import KeyTestUiState
from ui.presenters.recipe_presenter import RecipeScreenPresenter
from ui.presenters.recipe_presenter_deps import RecipePresenterDeps


def _build_recipe_presenter_deps(host: Any) -> RecipePresenterDeps:
    def set_recipe(value: Any) -> None:
        host.recipe = value

    def log_ax3_speed_trace(location_name: str, recipe_obj: Any) -> None:
        host._log_ax3_speed_trace(location_name, recipe_obj=recipe_obj)

    def set_len_low_approach_legacy_z(value: float | None) -> None:
        host._len_low_appr_legacy_z = value

    callbacks = []
    for method_name in (
        "_apply_start_anchor_from_recipe",
        "_refresh_recipe_table",
        "_refresh_auto_std_panel",
        "_refresh_standby_pos",
        "_refresh_center_positions",
    ):
        callback = getattr(host, method_name, None)
        if callable(callback):
            callbacks.append(callback)

    return RecipePresenterDeps(
        get_recipe=lambda: host.recipe,
        set_recipe=set_recipe,
        axis_cal=host.axis_cal,
        ui_state=getattr(host, "ui", None),
        log_ax3_speed_trace=log_ax3_speed_trace,
        refresh_length_info=getattr(host, "_refresh_length_info", None),
        set_len_low_approach_legacy_z=set_len_low_approach_legacy_z,
        after_recipe_data_applied=tuple(callbacks),
    )


def wire_screen_controllers(host: Any) -> None:
    """Create screen presenters and controllers, attaching each to *host*.

    This replaces the former ``HostUIMixin._init_presenters`` inline
    wiring.  The caller (AppHost.__init__) invokes this once, then calls
    ``host._build_ui()``.
    """
    controller = ScreenController(host)
    host._screen_controller = controller

    presenter = ScreenPresenter(host)
    host._screen_presenter = presenter

    recipe_presenter = RecipeScreenPresenter(_build_recipe_presenter_deps(host))
    host._recipe_screen_presenter = recipe_presenter

    recipe_controller = RecipeController(host)
    host.recipe_controller = recipe_controller

    key_test_controller = KeyTestController(host)
    host.key_test_controller = key_test_controller

    key_test_ui = KeyTestUiState(
        keytest_x_vars=host.keytest_x_vars,
        keytest_y_vars=host.keytest_y_vars,
        keytest_y_lastcmd_vars=host.keytest_y_lastcmd_vars,
    )
    host.key_test_ui = key_test_ui

    axis_presenter = AxisScreenPresenter(host, controller)
    host._axis_screen_presenter = axis_presenter

    gauge_presenter = GaugeScreenPresenter(host, controller)
    host._gauge_screen_presenter = gauge_presenter

    ui_context = ScreenUiContext(host)
    host._screen_ui_context = ui_context


__all__ = ["wire_screen_controllers"]
