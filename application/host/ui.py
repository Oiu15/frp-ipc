from __future__ import annotations

"""UI construction mixin for AppHost.

Extracted from ``app_host.py`` to eliminate the application -> ui
module-level dependency (violation V9).  All Tkinter screen assembly
and presenter initialisation lives here.
"""

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Any, cast

from application.adapters.device_gateway import ScreenController, ScreenPresenter, ScreenUiContext
from application.controllers.axis_cal_controller import AxisCalController
from application.controllers.key_test_controller import KeyTestController
from application.controllers.validation_controller import ValidationController
from ui.presenters.axis_cal_presenter_deps import AxisCalUiState
from ui.presenters.axis_presenter import AxisScreenPresenter
from ui.presenters.gauge_presenter import GaugeScreenPresenter
from ui.presenters.key_test_presenter import KeyTestUiState
from ui.presenters.recipe_presenter import RecipeScreenPresenter
from ui.presenters.validation_presenter_deps import ValidationUiState
from ui.screens.axis_cal_screen import build_axis_cal_screen
from ui.screens.axis_screen import build_axis_screen
from ui.screens.gauge_screen import build_gauge_screen
from ui.screens.key_test_screen import build_key_test_screen
from ui.screens.main_screen import build_main_screen
from ui.screens.recipe_screen import build_recipe_screen
from ui.screens.validation_screen import build_validation_screen


class HostUIMixin:
    """Mixin providing UI construction and presenter initialisation.

    Requires the AppHost to have ``self._recipe_store_init``,
    ``self.plc_status_var``, ``self.err_banner_var``, and the
    ``_screen_*`` attributes initialised before ``_build_ui`` is called.
    """

    _screen_controller: ScreenController
    _screen_presenter: ScreenPresenter
    _recipe_screen_presenter: RecipeScreenPresenter
    _axis_screen_presenter: AxisScreenPresenter
    _gauge_screen_presenter: GaugeScreenPresenter
    _screen_ui_context: ScreenUiContext
    recipe_controller: Any
    axis_cal_controller: AxisCalController
    axis_cal_ui: AxisCalUiState
    validation_controller: ValidationController
    validation_ui: ValidationUiState
    key_test_controller: KeyTestController
    key_test_ui: KeyTestUiState
    plc_status_var: tk.StringVar
    err_banner_var: tk.StringVar

    if TYPE_CHECKING:
        def _recipe_store_init(self) -> None: ...

    # -- presenter initialisation ---------------------------------------

    def _init_presenters(self) -> None:
        from application.controller_wiring import wire_screen_controllers

        wire_screen_controllers(self)

    # -- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        host = cast(tk.Tk, self)
        top = ttk.Frame(host)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        # Top bar: left = PLC status; right = rolling error banner.
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)
        ttk.Label(top, textvariable=self.plc_status_var).grid(row=0, column=0, sticky="w")
        self._err_banner_lbl = tk.Label(
            top,
            textvariable=self.err_banner_var,
            fg="red",
            anchor="e",
            justify="right",
        )
        self._err_banner_lbl.grid(row=0, column=1, sticky="e")

        nb = ttk.Notebook(host)
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        # keep a reference for future extensions
        self._notebook = nb

        tab_main = ttk.Frame(nb)
        tab_axis_cal = ttk.Frame(nb)
        tab_axis = ttk.Frame(nb)
        tab_recipe = ttk.Frame(nb)
        tab_validation = ttk.Frame(nb)
        tab_gauge = ttk.Frame(nb)
        tab_keytest = ttk.Frame(nb)
        self._tab_main = tab_main

        # Main operation tab first (left-most) and selected by default.
        nb.add(tab_main, text="主操作/自动测量")
        nb.add(tab_axis_cal, text="轴位标定")
        nb.add(tab_axis, text="轴参数/调试")
        nb.add(tab_recipe, text="配方/示教")
        nb.add(tab_gauge, text="外设通信")
        nb.add(tab_keytest, text="按键测试")

        build_main_screen(tab_main, presenter=self._screen_presenter, controller=self._screen_controller, ui=self._screen_ui_context)
        build_axis_cal_screen(tab_axis_cal, presenter=self.axis_cal_ui, controller=self.axis_cal_controller, ui=self.axis_cal_ui)
        build_axis_screen(tab_axis, presenter=self._axis_screen_presenter, controller=self._screen_controller, ui=self._screen_ui_context)
        build_recipe_screen(tab_recipe, presenter=self._recipe_screen_presenter, controller=self.recipe_controller, ui=self._screen_ui_context)
        build_validation_screen(tab_validation, presenter=self.validation_ui, controller=self.validation_controller, ui=self.validation_ui)
        build_gauge_screen(tab_gauge, presenter=self._gauge_screen_presenter, controller=self._screen_controller, ui=self._screen_ui_context)
        build_key_test_screen(tab_keytest, presenter=self.key_test_ui, controller=self.key_test_controller, ui=self.key_test_ui)
        nb.insert(tab_gauge, tab_validation, text="Validation")
        self._tab_validation = tab_validation

        try:
            nb.select(tab_main)
        except Exception:
            pass

        # init recipe store UI (dropdown, last recipe)
        try:
            self._recipe_store_init()
        except Exception:
            pass
