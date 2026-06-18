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
from application.controllers.recipe_controller import RecipeController
from ui.presenters.axis_presenter import AxisScreenPresenter
from ui.presenters.gauge_presenter import GaugeScreenPresenter
from ui.presenters.recipe_presenter import RecipeScreenPresenter


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

    recipe_presenter = RecipeScreenPresenter(host)
    host._recipe_screen_presenter = recipe_presenter

    recipe_controller = RecipeController(host)
    host.recipe_controller = recipe_controller

    axis_presenter = AxisScreenPresenter(host, controller)
    host._axis_screen_presenter = axis_presenter

    gauge_presenter = GaugeScreenPresenter(host, controller)
    host._gauge_screen_presenter = gauge_presenter

    ui_context = ScreenUiContext(host)
    host._screen_ui_context = ui_context


__all__ = ["wire_screen_controllers"]
