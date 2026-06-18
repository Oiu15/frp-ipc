from __future__ import annotations

import tkinter as tk
from collections.abc import Iterable
from typing import Any


_MISSING = object()


class RecipeScreenPresenter:
    """Own the recipe-screen Tk variables while preserving legacy host compatibility."""

    _HOST_ATTR_ALLOWLIST = {
        'recipe',
    }
    _HOST_CALL_PREFIX_ALLOWLIST = (
        '_list',
        '_refresh',
    )

    def __init__(self, host: Any) -> None:
        object.__setattr__(self, '_host', host)
        object.__setattr__(self, '_owned_attrs', {})
        object.__setattr__(self, '_widgets', {})

    @property
    def host_app(self) -> Any:
        return object.__getattribute__(self, '_host')

    @property
    def recipe(self) -> Any:
        return self.host_app.recipe

    @recipe.setter
    def recipe(self, value: Any) -> None:
        self.host_app.recipe = value

    def _remember(self, name: str, value: Any) -> Any:
        owned = object.__getattribute__(self, '_owned_attrs')
        owned[name] = value
        return value

    def remember_widget(self, name: str, widget: Any) -> Any:
        widgets = object.__getattribute__(self, '_widgets')
        widgets[name] = widget
        return widget

    def widget(self, name: str) -> Any:
        return object.__getattribute__(self, '_widgets').get(name)

    def get_var_value(self, name: str, default: Any = _MISSING) -> Any:
        owned = object.__getattribute__(self, '_owned_attrs')
        var = owned.get(name, _MISSING)
        if var is _MISSING:
            ui = getattr(self.host_app, "ui", None)
            var = getattr(ui, name, _MISSING)
        if var is _MISSING:
            if default is _MISSING:
                raise AttributeError(name)
            return default
        getter = getattr(var, "get", None)
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
        owned = object.__getattribute__(self, '_owned_attrs')
        var = owned.get(name)
        if var is None:
            ui = getattr(self.host_app, "ui", None)
            var = getattr(ui, name, None)
        setter = getattr(var, "set", None)
        if not callable(setter):
            return
        try:
            setter(value)
        except Exception:
            pass

    def sync_combo_value(self, combo_name: str, value: str) -> None:
        combo = self.widget(combo_name)
        cget = getattr(combo, "cget", None)
        current = getattr(combo, "current", None)
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
        fn = getattr(self.host_app, "_log_ax3_speed_trace", None)
        if callable(fn):
            fn(location_name, recipe_obj=recipe_obj)

    def refresh_length_info(self) -> None:
        fn = getattr(self.host_app, "_refresh_length_info", None)
        if callable(fn):
            fn()

    def set_len_low_approach_legacy_z(self, value: float | None) -> None:
        self.host_app._len_low_appr_legacy_z = value

    def after_recipe_data_applied(self) -> None:
        for method_name in (
            "_apply_start_anchor_from_recipe",
            "_refresh_recipe_table",
            "_refresh_auto_std_panel",
            "_refresh_standby_pos",
            "_refresh_center_positions",
        ):
            fn = getattr(self.host_app, method_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    def _ensure_var(self, name: str, factory) -> tk.Variable:
        owned = object.__getattribute__(self, '_owned_attrs')
        if name in owned:
            return owned[name]
        try:
            existing = getattr(self.host_app, name)
        except AttributeError:
            existing = None
        if isinstance(existing, tk.Variable):
            owned[name] = existing
            return existing
        return self._remember(name, factory())

    def _ensure_ui_var(self, name: str, factory) -> tk.Variable:
        ui = getattr(self.host_app, "ui", None)
        existing = getattr(ui, name, None)
        if isinstance(existing, tk.Variable):
            object.__getattribute__(self, "_owned_attrs")[name] = existing
            return existing
        return self._ensure_var(name, factory)

    def ensure_vars(self, master: tk.Misc) -> None:
        recipe = self.host_app.recipe
        axis_cal = self.host_app.axis_cal
        scan_mode = str(
            getattr(
                recipe,
                'section_sampling_mode',
                getattr(recipe, 'scan_mode', 'sync'),
            )
            or 'sync'
        ).strip().lower()
        legacy_z = float(getattr(recipe, 'len_z_low_approach', 1300.0))
        abs_appr = float(getattr(recipe, 'len_low_approach_abs', 0.0) or 0.0)
        if abs_appr == 0.0:
            try:
                abs_appr = float(axis_cal.z_disp_to_abs(0, legacy_z))
            except Exception:
                abs_appr = 0.0

        self._ensure_var('recipe_name_var', lambda: tk.StringVar(master=master, value=recipe.name))
        self._ensure_ui_var('pipe_len_var', lambda: tk.StringVar(master=master, value=str(recipe.pipe_len_mm)))
        self._ensure_var('clamp_var', lambda: tk.StringVar(master=master, value=str(recipe.clamp_occupy_mm)))
        self._ensure_var('clamp_confirm_wait_s_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'clamp_confirm_wait_s', 3.0))))
        self._ensure_var('margin_h_var', lambda: tk.StringVar(master=master, value=str(recipe.margin_head_mm)))
        self._ensure_var('margin_t_var', lambda: tk.StringVar(master=master, value=str(recipe.margin_tail_mm)))
        self._ensure_var('meas_total_len_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'meas_total_len_mm', 0.0))))
        self._ensure_var('section_n_var', lambda: tk.StringVar(master=master, value=str(recipe.section_count)))
        self._ensure_ui_var('teach_axes_mode_var', lambda: tk.IntVar(master=master, value=int(getattr(recipe, 'teach_axes_mode', 2))))
        self._ensure_var('od_std_var', lambda: tk.StringVar(master=master, value=str(recipe.od_std_mm)))
        self._ensure_var('id_std_var', lambda: tk.StringVar(master=master, value=str(recipe.id_std_mm)))
        self._ensure_var('od_tol_var', lambda: tk.StringVar(master=master, value=str(recipe.od_tol_mm)))
        self._ensure_var('od_use_edges_var', lambda: tk.BooleanVar(master=master, value=bool(getattr(recipe, 'od_use_edges', False))))
        self._ensure_var('id_use_fit_var', lambda: tk.BooleanVar(master=master, value=bool(getattr(recipe, 'id_use_fit', False))))
        self._ensure_var('id_single_enable_var', lambda: tk.BooleanVar(master=master, value=False))
        self._ensure_var('id_single_k_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'id_single_k', 1.0))))
        self._ensure_var('id_single_b_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'id_single_b', 0.0))))
        self._ensure_var('section_sampling_mode_var', lambda: tk.StringVar(master=master, value=scan_mode))
        self._ensure_var('split_scan_var', lambda: tk.BooleanVar(master=master, value=scan_mode.startswith('split')))
        self._ensure_var('disable_id_modbus_var', lambda: tk.BooleanVar(master=master, value=bool(getattr(recipe, 'disable_id_modbus', False))))
        self._ensure_var('split_keep_spinning_var', lambda: tk.BooleanVar(master=master, value=True))
        self._ensure_var('split_slip_check_var', lambda: tk.BooleanVar(master=master, value=bool(getattr(recipe, 'split_slip_check', True))))
        self._ensure_var('points_per_rev_var', lambda: tk.StringVar(master=master, value=str(recipe.points_per_rev)))
        self._ensure_var('min_cov_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'min_bin_coverage', 0.95))))
        self._ensure_var('sample_timeout_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'sample_timeout_s', 5.0))))
        self._ensure_var('max_revs_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'max_revolutions', 2.0))))
        self._ensure_var('sample_delay_s_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'sample_delay_s', 0.0))))
        self._ensure_var('rot_vel_velmove_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'rot_vel_velmove', 200.0))))
        self._ensure_var('fit_strategy_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'fit_strategy', 'b 原始点按bin权重均衡'))))
        self._ensure_var('calc_input_mode_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'calc_input_mode', 'bin'))))
        self._ensure_var('bin_count_var', lambda: tk.StringVar(master=master, value=str(int(getattr(recipe, 'bin_count', 90)))))
        self._ensure_var('bin_method_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'bin_method', 'median'))))
        self._ensure_var('pp_mode_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'pp_mode', 'p99_p1'))))
        self._ensure_var('theta_delay_s_var', lambda: tk.StringVar(master=master, value=str(float(getattr(recipe, 'theta_delay_s', 0.0) or 0.0))))
        self._ensure_ui_var('len_enable_var', lambda: tk.BooleanVar(master=master, value=bool(getattr(recipe, 'len_enable', False))))
        self._ensure_ui_var('len_z_low_approach_var', lambda: tk.StringVar(master=master, value=str(abs_appr)))
        self._ensure_ui_var('len_low_search_dist_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'len_low_search_dist', 220.0))))
        self._ensure_ui_var('len_high_search_dist_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'len_high_search_dist', 220.0))))
        self._ensure_ui_var('len_search_vel_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'len_search_vel', 5.0))))
        self._ensure_ui_var('len_search_timeout_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'len_search_timeout_s', 12.0))))
        self._ensure_ui_var('len_tol_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'len_tol_mm', 20.0))))
        self._ensure_ui_var('len_high_margin_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'len_high_margin', 20.0))))
        self._ensure_ui_var('len_debounce_k_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'len_debounce_k', 6))))
        self._ensure_ui_var('len_max_stale_ms_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'len_max_stale_ms', 300))))
        self._ensure_ui_var('len_backoff_var', lambda: tk.StringVar(master=master, value=str(getattr(recipe, 'len_backoff_mm', 2.0))))
        self._ensure_ui_var('center_pos_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('len_info_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('len_status_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('teach_rel_dist_var', lambda: tk.StringVar(master=master, value='10'))
        self._ensure_ui_var('teach_abs_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('teach_z_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('teach_align_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('teach_mode_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('teach_axes_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('start_info_var', lambda: tk.StringVar(master=master, value='Start: 未设置'))
        self._ensure_ui_var('standby_info_var', lambda: tk.StringVar(master=master, value='未设置'))
        self._ensure_ui_var('standby_state_var', lambda: tk.StringVar(master=master, value='-'))
        self._ensure_ui_var('len_edge_state_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('len_edge_low_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('len_edge_high_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('len_edge_len_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_var('recipe_len_adv_open_var', lambda: tk.BooleanVar(master=master, value=False))
        self._ensure_var('recipe_algo_open_var', lambda: tk.BooleanVar(master=master, value=False))
        self._ensure_var('recipe_algo_btn_text_var', lambda: tk.StringVar(master=master, value='算法参数 ▸'))

    def __getattr__(self, name: str) -> Any:
        owned = object.__getattribute__(self, '_owned_attrs')
        if name in owned:
            return owned[name]
        widgets = object.__getattribute__(self, '_widgets')
        if name in widgets:
            return widgets[name]
        if not (
            name in self._HOST_ATTR_ALLOWLIST
            or any(name.startswith(prefix) for prefix in self._HOST_CALL_PREFIX_ALLOWLIST)
        ):
            raise AttributeError(name)
        attr = getattr(self.host_app, name)
        if callable(attr) and not any(
            name.startswith(prefix) for prefix in self._HOST_CALL_PREFIX_ALLOWLIST
        ):
            raise AttributeError(name)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {'_host', '_owned_attrs', '_widgets'}:
            object.__setattr__(self, name, value)
            return
        if name == 'recipe':
            self.host_app.recipe = value
            return
        self._remember(name, value)


__all__ = ['RecipeScreenPresenter']
