from __future__ import annotations

import tkinter as tk
from collections.abc import Iterable
from typing import Any, Callable

from ui.presenters.recipe_presenter_deps import RecipePresenterDeps


_MISSING = object()


def _read_attr(obj: Any, name: str, default: Any = _MISSING) -> Any:
    if obj is None:
        if default is _MISSING:
            raise AttributeError(name)
        return default
    try:
        return object.__getattribute__(obj, name)
    except AttributeError:
        if default is _MISSING:
            raise
        return default


class RecipeScreenPresenter:
    """Own the recipe-screen Tk variables behind an explicit dependency set."""

    def __init__(self, deps: RecipePresenterDeps) -> None:
        object.__setattr__(self, "_deps", deps)
        object.__setattr__(self, "_owned_attrs", {})
        object.__setattr__(self, "_widgets", {})

    @property
    def recipe(self) -> Any:
        deps = object.__getattribute__(self, "_deps")
        return deps.get_recipe()

    @recipe.setter
    def recipe(self, value: Any) -> None:
        deps = object.__getattribute__(self, "_deps")
        deps.set_recipe(value)

    def _remember(self, name: str, value: Any) -> Any:
        owned = object.__getattribute__(self, "_owned_attrs")
        owned[name] = value
        object.__setattr__(self, name, value)
        return value

    def remember_widget(self, name: str, widget: Any) -> Any:
        widgets = object.__getattribute__(self, "_widgets")
        widgets[name] = widget
        object.__setattr__(self, name, widget)
        return widget

    def widget(self, name: str) -> Any:
        return object.__getattribute__(self, "_widgets").get(name)

    def _registered_var(self, name: str, default: Any = _MISSING) -> Any:
        deps = object.__getattribute__(self, "_deps")
        registry = deps.variable_registry
        if name in registry:
            return registry[name]
        return _read_attr(deps.ui_state, name, default)

    def get_var_value(self, name: str, default: Any = _MISSING) -> Any:
        owned = object.__getattribute__(self, "_owned_attrs")
        var = owned.get(name, _MISSING)
        if var is _MISSING:
            var = self._registered_var(name, _MISSING)
        if var is _MISSING:
            if default is _MISSING:
                raise AttributeError(name)
            return default
        getter = _read_attr(var, "get", None)
        if not callable(getter):
            if default is _MISSING:
                raise AttributeError(name)
            return default
        try:
            return getter()
        except Exception:
            if default is _MISSING:
                raise
            return default

    def set_var_value(self, name: str, value: Any) -> None:
        owned = object.__getattribute__(self, "_owned_attrs")
        var = owned.get(name)
        if var is None:
            var = self._registered_var(name, None)
        setter = _read_attr(var, "set", None)
        if not callable(setter):
            return
        try:
            setter(value)
        except Exception:
            pass

    def sync_combo_value(self, combo_name: str, value: str) -> None:
        combo = self.widget(combo_name)
        cget = _read_attr(combo, "cget", None)
        current = _read_attr(combo, "current", None)
        if not callable(cget) or not callable(current):
            return
        try:
            raw_values = cget("values")
            vals = list(raw_values) if isinstance(raw_values, Iterable) else []
            if value in vals:
                current(vals.index(value))
        except Exception:
            pass

    def log_ax3_speed_trace(self, location_name: str, *, recipe_obj: Any = None) -> None:
        deps = object.__getattribute__(self, "_deps")
        if deps.log_ax3_speed_trace is not None:
            deps.log_ax3_speed_trace(location_name, recipe_obj)

    def refresh_length_info(self) -> None:
        deps = object.__getattribute__(self, "_deps")
        if deps.refresh_length_info is not None:
            deps.refresh_length_info()

    def set_len_low_approach_legacy_z(self, value: float | None) -> None:
        deps = object.__getattribute__(self, "_deps")
        if deps.set_len_low_approach_legacy_z is not None:
            deps.set_len_low_approach_legacy_z(value)

    def after_recipe_data_applied(self) -> None:
        deps = object.__getattribute__(self, "_deps")
        for callback in deps.after_recipe_data_applied:
            try:
                callback()
            except Exception:
                pass

    def _ensure_var(self, name: str, factory: Callable[[], tk.Variable]) -> tk.Variable:
        owned = object.__getattribute__(self, "_owned_attrs")
        if name in owned:
            return owned[name]
        existing = self._registered_var(name, None)
        if isinstance(existing, tk.Variable):
            return self._remember(name, existing)
        return self._remember(name, factory())

    def _ensure_ui_var(self, name: str, factory: Callable[[], tk.Variable]) -> tk.Variable:
        return self._ensure_var(name, factory)

    def ensure_vars(self, master: tk.Misc) -> None:
        recipe = self.recipe
        deps = object.__getattribute__(self, "_deps")
        axis_cal = deps.axis_cal
        scan_mode = str(
            _read_attr(
                recipe,
                "section_sampling_mode",
                _read_attr(recipe, "scan_mode", "sync"),
            )
            or "sync"
        ).strip().lower()
        legacy_z = float(_read_attr(recipe, "len_z_low_approach", 1300.0))
        abs_appr = float(_read_attr(recipe, "len_low_approach_abs", 0.0) or 0.0)
        if abs_appr == 0.0:
            try:
                abs_appr = float(axis_cal.z_disp_to_abs(0, legacy_z))
            except Exception:
                abs_appr = 0.0

        self._ensure_var("recipe_name_var", lambda: tk.StringVar(master=master, value=recipe.name))
        self._ensure_ui_var("pipe_len_var", lambda: tk.StringVar(master=master, value=str(recipe.pipe_len_mm)))
        self._ensure_var("clamp_var", lambda: tk.StringVar(master=master, value=str(recipe.clamp_occupy_mm)))
        self._ensure_var("clamp_confirm_wait_s_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "clamp_confirm_wait_s", 3.0))))
        self._ensure_var("margin_h_var", lambda: tk.StringVar(master=master, value=str(recipe.margin_head_mm)))
        self._ensure_var("margin_t_var", lambda: tk.StringVar(master=master, value=str(recipe.margin_tail_mm)))
        self._ensure_var("meas_total_len_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "meas_total_len_mm", 0.0))))
        self._ensure_var("section_n_var", lambda: tk.StringVar(master=master, value=str(recipe.section_count)))
        self._ensure_ui_var("teach_axes_mode_var", lambda: tk.IntVar(master=master, value=int(_read_attr(recipe, "teach_axes_mode", 2))))
        self._ensure_var("od_std_var", lambda: tk.StringVar(master=master, value=str(recipe.od_std_mm)))
        self._ensure_var("id_std_var", lambda: tk.StringVar(master=master, value=str(recipe.id_std_mm)))
        self._ensure_var("od_tol_var", lambda: tk.StringVar(master=master, value=str(recipe.od_tol_mm)))
        self._ensure_var("od_use_edges_var", lambda: tk.BooleanVar(master=master, value=bool(_read_attr(recipe, "od_use_edges", False))))
        self._ensure_var("id_use_fit_var", lambda: tk.BooleanVar(master=master, value=bool(_read_attr(recipe, "id_use_fit", False))))
        self._ensure_var(
            "algo_version_var",
            lambda: tk.StringVar(
                master=master,
                value=(
                    "geometry_v2 几何重建"
                    if str(_read_attr(recipe, "algo_version", "legacy")) == "geometry_v2"
                    else "legacy 旧链(默认)"
                ),
            ),
        )
        self._ensure_var("id_single_enable_var", lambda: tk.BooleanVar(master=master, value=False))
        self._ensure_var("id_single_k_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "id_single_k", 1.0))))
        self._ensure_var("id_single_b_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "id_single_b", 0.0))))
        self._ensure_var("section_sampling_mode_var", lambda: tk.StringVar(master=master, value=scan_mode))
        self._ensure_var("split_scan_var", lambda: tk.BooleanVar(master=master, value=scan_mode.startswith("split")))
        self._ensure_var("disable_id_modbus_var", lambda: tk.BooleanVar(master=master, value=bool(_read_attr(recipe, "disable_id_modbus", False))))
        self._ensure_var("split_keep_spinning_var", lambda: tk.BooleanVar(master=master, value=True))
        self._ensure_var("split_slip_check_var", lambda: tk.BooleanVar(master=master, value=bool(_read_attr(recipe, "split_slip_check", True))))
        self._ensure_var("points_per_rev_var", lambda: tk.StringVar(master=master, value=str(recipe.points_per_rev)))
        self._ensure_var("min_cov_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "min_bin_coverage", 0.95))))
        self._ensure_var("sample_timeout_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "sample_timeout_s", 5.0))))
        self._ensure_var("max_revs_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "max_revolutions", 2.0))))
        self._ensure_var("sample_delay_s_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "sample_delay_s", 0.0))))
        self._ensure_var("rot_vel_velmove_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "rot_vel_velmove", 200.0))))
        self._ensure_var("fit_strategy_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "fit_strategy", "b 原始点按bin权重均衡"))))
        self._ensure_var("calc_input_mode_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "calc_input_mode", "bin"))))
        self._ensure_var("bin_count_var", lambda: tk.StringVar(master=master, value=str(int(_read_attr(recipe, "bin_count", 90)))))
        self._ensure_var("bin_method_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "bin_method", "median"))))
        self._ensure_var("pp_mode_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "pp_mode", "p99_p1"))))
        self._ensure_var("theta_delay_s_var", lambda: tk.StringVar(master=master, value=str(float(_read_attr(recipe, "theta_delay_s", 0.0) or 0.0))))
        self._ensure_ui_var("len_enable_var", lambda: tk.BooleanVar(master=master, value=bool(_read_attr(recipe, "len_enable", False))))
        self._ensure_ui_var("len_z_low_approach_var", lambda: tk.StringVar(master=master, value=str(abs_appr)))
        self._ensure_ui_var("len_low_search_dist_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "len_low_search_dist", 220.0))))
        self._ensure_ui_var("len_high_search_dist_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "len_high_search_dist", 220.0))))
        self._ensure_ui_var("len_search_vel_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "len_search_vel", 5.0))))
        self._ensure_ui_var("len_search_timeout_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "len_search_timeout_s", 12.0))))
        self._ensure_ui_var("len_tol_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "len_tol_mm", 20.0))))
        self._ensure_ui_var("len_high_margin_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "len_high_margin", 20.0))))
        self._ensure_ui_var("len_debounce_k_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "len_debounce_k", 6))))
        self._ensure_ui_var("len_max_stale_ms_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "len_max_stale_ms", 300))))
        self._ensure_ui_var("len_backoff_var", lambda: tk.StringVar(master=master, value=str(_read_attr(recipe, "len_backoff_mm", 2.0))))
        self._ensure_ui_var("center_pos_var", lambda: tk.StringVar(master=master, value="--"))
        self._ensure_ui_var("len_info_var", lambda: tk.StringVar(master=master, value="--"))
        self._ensure_ui_var("len_status_var", lambda: tk.StringVar(master=master, value="--"))
        self._ensure_ui_var("teach_rel_dist_var", lambda: tk.StringVar(master=master, value="10"))
        self._ensure_ui_var("teach_abs_var", lambda: tk.StringVar(master=master, value="--"))
        self._ensure_ui_var("teach_z_var", lambda: tk.StringVar(master=master, value="--"))
        self._ensure_ui_var("teach_align_var", lambda: tk.StringVar(master=master, value="--"))
        self._ensure_ui_var("teach_mode_var", lambda: tk.StringVar(master=master, value="--"))
        self._ensure_ui_var("teach_axes_var", lambda: tk.StringVar(master=master, value="--"))
        self._ensure_ui_var("start_info_var", lambda: tk.StringVar(master=master, value="Start: 未设置"))
        self._ensure_ui_var("standby_info_var", lambda: tk.StringVar(master=master, value="未设置"))
        self._ensure_ui_var("standby_state_var", lambda: tk.StringVar(master=master, value="-"))
        self._ensure_ui_var("len_edge_state_var", lambda: tk.StringVar(master=master, value="--"))
        self._ensure_ui_var("len_edge_low_var", lambda: tk.StringVar(master=master, value="--"))
        self._ensure_ui_var("len_edge_high_var", lambda: tk.StringVar(master=master, value="--"))
        self._ensure_ui_var("len_edge_len_var", lambda: tk.StringVar(master=master, value="--"))
        self._ensure_var("recipe_len_adv_open_var", lambda: tk.BooleanVar(master=master, value=False))
        self._ensure_var("recipe_algo_open_var", lambda: tk.BooleanVar(master=master, value=False))
        self._ensure_var("recipe_algo_btn_text_var", lambda: tk.StringVar(master=master, value="算法参数 ▸"))


__all__ = ["RecipeScreenPresenter"]
