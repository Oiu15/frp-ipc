from __future__ import annotations

"""Map between legacy recipe UI vars, Recipe objects, and persisted dict data."""

import logging
from typing import Any, Mapping

from core.models import Recipe, SectionPlanSnapshot

recipe_logger = logging.getLogger("frp.recipe")


class RecipeFormMapper:
    """Compatibility mapper for the current Tk form and Recipe model."""

    def __init__(self, host: Any) -> None:
        self.host = host

    def _host_recipe(self) -> Recipe:
        return getattr(self.host, "recipe")

    def _fallback(self, attr: str, default: Any) -> Any:
        return getattr(self._host_recipe(), attr, default)

    def _get_var(self, name: str) -> Any:
        return getattr(self.host, name).get()

    def _set_var_if_exists(self, name: str, value: Any) -> None:
        try:
            getattr(self.host, name).set(value)
        except Exception:
            pass

    def _sync_combo(self, combo_name: str, value: str) -> None:
        try:
            widget_getter = getattr(self.host, '_recipe_ui_widget', None)
            combo = widget_getter(combo_name) if callable(widget_getter) else getattr(self.host, combo_name)
            vals = list(combo.cget("values") or [])
            if value in vals:
                combo.current(vals.index(value))
        except Exception:
            pass

    def _reset_planning_state_defaults(self) -> None:
        defaults = {
            "standby_valid": False,
            "standby_ax0_abs": 0.0,
            "standby_ax1_abs": 0.0,
            "standby_ax4_abs": 0.0,
            "start_valid": False,
            "start_ax0_abs": 0.0,
            "ax2_len_valid": False,
            "ax2_len_abs": 0.0,
            "ax2_rot_valid": False,
            "ax2_rot_abs": 0.0,
        }
        host_recipe = self._host_recipe()
        for attr, default in defaults.items():
            try:
                setattr(host_recipe, attr, default)
            except Exception:
                pass

    def _norm_choice(self, value: str, default: str, mapping: dict[str, str]) -> str:
        vv = str(value or "").strip()
        if vv in mapping:
            return mapping[vv]
        if vv in set(mapping.values()):
            return vv
        for k, out in mapping.items():
            if vv.startswith(str(k).split(" ")[0]):
                return out
        return default

    def recipe_to_dict(self, recipe: Recipe) -> dict:
        section_plan = getattr(recipe, "section_plan", None)
        return {
            "name": recipe.name,
            "pipe_len_mm": recipe.pipe_len_mm,
            "clamp_occupy_mm": recipe.clamp_occupy_mm,
            "clamp_confirm_wait_s": float(getattr(recipe, "clamp_confirm_wait_s", 3.0)),
            "margin_head_mm": recipe.margin_head_mm,
            "margin_tail_mm": recipe.margin_tail_mm,
            "meas_total_len_mm": float(getattr(recipe, "meas_total_len_mm", 0.0) or 0.0),
            "section_count": recipe.section_count,
            "scan_axis": recipe.scan_axis,
            "section_sampling_mode": str(getattr(recipe, "section_sampling_mode", getattr(recipe, "scan_mode", "sync")) or "sync"),
            "sampling_window_mode": str(getattr(recipe, "sampling_window_mode", "shared") or "shared"),
            "scan_mode": str(getattr(recipe, "scan_mode", "sync") or "sync"),
            "disable_id_modbus": bool(getattr(recipe, "disable_id_modbus", False)),
            "split_keep_spinning": True,
            "split_slip_check": bool(getattr(recipe, "split_slip_check", True)),
            "split_slip_max_deg": float(getattr(recipe, "split_slip_max_deg", 5.0) or 5.0),
            "split_omega_cv_max": float(getattr(recipe, "split_omega_cv_max", 0.25) or 0.25),
            "teach_axes_mode": int(getattr(recipe, "teach_axes_mode", 2)),
            "od_std_mm": recipe.od_std_mm,
            "id_std_mm": recipe.id_std_mm,
            "od_tol_mm": recipe.od_tol_mm,
            "points_per_rev": recipe.points_per_rev,
            "sample_coverage": recipe.min_bin_coverage,
            "section_timeout_s": recipe.sample_timeout_s,
            "max_revs": recipe.max_revolutions,
            "sample_delay_s": float(getattr(recipe, "sample_delay_s", 0.0) or 0.0),
            "rot_vel_velmove": float(getattr(recipe, "rot_vel_velmove", 200.0) or 200.0),
            "fit_strategy": str(getattr(recipe, "fit_strategy", "b 原始点按bin权重均衡")),
            "calc_input_mode": str(getattr(recipe, "calc_input_mode", "bin")),
            "bin_count": int(getattr(recipe, "bin_count", 90)),
            "bin_method": str(getattr(recipe, "bin_method", "median")),
            "pp_mode": str(getattr(recipe, "pp_mode", "p99_p1")),
            "theta_delay_s": float(getattr(recipe, "theta_delay_s", 0.0) or 0.0),
            "od_use_edges": bool(getattr(recipe, "od_use_edges", False)),
            "id_use_fit": bool(getattr(recipe, "id_use_fit", False)),
            "id_single_enable": False,
            "id_single_k": float(getattr(recipe, "id_single_k", 1.0) or 1.0),
            "id_single_b": float(getattr(recipe, "id_single_b", 0.0) or 0.0),
            "id_single_show_debug": bool(getattr(recipe, "id_single_show_debug", False)),
            "len_enable": bool(getattr(recipe, "len_enable", False)),
            "len_low_approach_abs": float(getattr(recipe, "len_low_approach_abs", 0.0) or 0.0),
            "len_low_search_dist": float(getattr(recipe, "len_low_search_dist", 220.0)),
            "len_high_search_dist": float(getattr(recipe, "len_high_search_dist", 220.0)),
            "len_search_vel": float(getattr(recipe, "len_search_vel", 5.0)),
            "len_search_timeout_s": float(getattr(recipe, "len_search_timeout_s", 12.0)),
            "len_tol_mm": float(getattr(recipe, "len_tol_mm", 20.0)),
            "len_high_margin": float(getattr(recipe, "len_high_margin", 20.0)),
            "len_debounce_k": int(getattr(recipe, "len_debounce_k", 6)),
            "len_max_stale_ms": int(getattr(recipe, "len_max_stale_ms", 300)),
            "len_backoff_mm": float(getattr(recipe, "len_backoff_mm", 2.0)),
            "section_pos_z": getattr(recipe, "section_pos_z", []),
            "section_plan": (
                section_plan.to_dict()
                if isinstance(section_plan, SectionPlanSnapshot)
                else None
            ),
            "standby_valid": bool(getattr(recipe, "standby_valid", False)),
            "standby_ax0_abs": float(getattr(recipe, "standby_ax0_abs", 0.0)),
            "standby_ax1_abs": float(getattr(recipe, "standby_ax1_abs", 0.0)),
            "standby_ax4_abs": float(getattr(recipe, "standby_ax4_abs", 0.0)),
            "start_valid": bool(getattr(recipe, "start_valid", False)),
            "start_ax0_abs": float(getattr(recipe, "start_ax0_abs", 0.0)),
            "ax2_len_valid": bool(getattr(recipe, "ax2_len_valid", False)),
            "ax2_len_abs": float(getattr(recipe, "ax2_len_abs", 0.0)),
            "ax2_rot_valid": bool(getattr(recipe, "ax2_rot_valid", False)),
            "ax2_rot_abs": float(getattr(recipe, "ax2_rot_abs", 0.0)),
        }

    def ui_vars_to_recipe(self) -> Recipe:
        host = self.host
        recipe = Recipe()
        recipe.name = str(self._get_var("recipe_name_var")).strip() or "默认配方"
        recipe.pipe_len_mm = float(self._get_var("pipe_len_var"))
        recipe.clamp_occupy_mm = float(self._get_var("clamp_var"))
        try:
            recipe.clamp_confirm_wait_s = float(self._get_var("clamp_confirm_wait_s_var"))
        except Exception:
            recipe.clamp_confirm_wait_s = float(self._fallback("clamp_confirm_wait_s", 3.0))
        recipe.margin_head_mm = float(self._get_var("margin_h_var"))
        recipe.margin_tail_mm = float(self._get_var("margin_t_var"))
        recipe.meas_total_len_mm = float(getattr(host, "meas_total_len_var", type("", (), {"get": lambda *_: self._fallback("meas_total_len_mm", 0.0)})()).get())
        recipe.section_count = int(float(self._get_var("section_n_var")))
        recipe.scan_axis = 0
        recipe.teach_axes_mode = int(getattr(host, "teach_axes_mode_var", type("", (), {"get": lambda *_: self._fallback("teach_axes_mode", 2)})()).get())
        recipe.od_std_mm = float(self._get_var("od_std_var"))
        recipe.id_std_mm = float(self._get_var("id_std_var"))
        recipe.od_tol_mm = float(self._get_var("od_tol_var"))
        recipe.points_per_rev = int(float(self._get_var("points_per_rev_var")))
        recipe.min_bin_coverage = float(self._get_var("min_cov_var"))
        recipe.sample_timeout_s = float(self._get_var("sample_timeout_var"))
        recipe.max_revolutions = float(self._get_var("max_revs_var"))
        recipe.sample_delay_s = float(getattr(host, "sample_delay_s_var", type("", (), {"get": lambda *_: self._fallback("sample_delay_s", 0.0)})()).get())
        recipe.rot_vel_velmove = float(getattr(host, "rot_vel_velmove_var", type("", (), {"get": lambda *_: self._fallback("rot_vel_velmove", 200.0)})()).get())
        recipe.fit_strategy = str(getattr(host, "fit_strategy_var", type("", (), {"get": lambda *_: self._fallback("fit_strategy", "b 原始点按bin权重均衡")})()).get())
        recipe.od_use_edges = bool(getattr(host, "od_use_edges_var", type("", (), {"get": lambda *_: self._fallback("od_use_edges", False)})()).get())
        recipe.id_use_fit = bool(getattr(host, "id_use_fit_var", type("", (), {"get": lambda *_: self._fallback("id_use_fit", False)})()).get())
        recipe.id_single_enable = False
        recipe.id_single_k = float(getattr(host, "id_single_k_var", type("", (), {"get": lambda *_: self._fallback("id_single_k", 1.0)})()).get())
        recipe.id_single_b = float(getattr(host, "id_single_b_var", type("", (), {"get": lambda *_: self._fallback("id_single_b", 0.0)})()).get())
        recipe.id_single_show_debug = bool(self._fallback("id_single_show_debug", False))
        sampling_mode_value = getattr(
            host,
            "section_sampling_mode_var",
            type("", (), {"get": lambda *_: self._fallback("section_sampling_mode", self._fallback("scan_mode", "sync"))})(),
        ).get()
        sampling_mode = str(sampling_mode_value or "").strip().lower()
        if sampling_mode not in {"sync", "split"}:
            sampling_mode = "split" if bool(getattr(host, "split_scan_var", type("", (), {"get": lambda *_: False})()).get()) else "sync"
        recipe.section_sampling_mode = sampling_mode
        recipe.sampling_window_mode = "separate_channels" if sampling_mode == "split" else "shared"
        recipe.scan_mode = sampling_mode
        recipe.disable_id_modbus = bool(getattr(host, "disable_id_modbus_var", type("", (), {"get": lambda *_: self._fallback("disable_id_modbus", False)})()).get())
        if recipe.id_single_enable:
            recipe.disable_id_modbus = False
        recipe.split_keep_spinning = True
        recipe.split_slip_check = bool(getattr(host, "split_slip_check_var", type("", (), {"get": lambda *_: self._fallback("split_slip_check", True)})()).get())
        recipe.split_slip_max_deg = float(self._fallback("split_slip_max_deg", 5.0) or 5.0)
        recipe.split_omega_cv_max = float(self._fallback("split_omega_cv_max", 0.25) or 0.25)

        try:
            recipe.len_enable = bool(self._get_var("len_enable_var"))
            recipe.len_low_approach_abs = float(self._get_var("len_z_low_approach_var"))
            recipe.len_low_search_dist = float(self._get_var("len_low_search_dist_var"))
            recipe.len_high_search_dist = float(self._get_var("len_high_search_dist_var"))
            recipe.len_search_vel = float(self._get_var("len_search_vel_var"))
            recipe.len_search_timeout_s = float(self._get_var("len_search_timeout_var"))
            recipe.len_tol_mm = float(self._get_var("len_tol_var"))
            recipe.len_high_margin = float(self._get_var("len_high_margin_var"))
            recipe.len_debounce_k = int(float(self._get_var("len_debounce_k_var")))
            recipe.len_max_stale_ms = int(float(self._get_var("len_max_stale_ms_var")))
            recipe.len_backoff_mm = float(self._get_var("len_backoff_var"))
        except Exception:
            for attr in (
                "len_enable",
                "len_low_approach_abs",
                "len_z_low_approach",
                "len_low_search_dist",
                "len_high_search_dist",
                "len_search_vel",
                "len_search_timeout_s",
                "len_tol_mm",
                "len_high_margin",
                "len_debounce_k",
                "len_max_stale_ms",
                "len_backoff_mm",
            ):
                if hasattr(self._host_recipe(), attr):
                    setattr(recipe, attr, getattr(self._host_recipe(), attr))

        if len(getattr(self._host_recipe(), "section_pos_z", [])) == recipe.section_count:
            recipe.section_pos_z = list(self._host_recipe().section_pos_z)
        else:
            recipe.section_pos_z = recipe.compute_default_positions_z()
        recipe.section_pos_ui = list(recipe.section_pos_z)
        host_plan = getattr(self._host_recipe(), "section_plan", None)
        if isinstance(host_plan, SectionPlanSnapshot) and len(host_plan.sections) == int(recipe.section_count):
            recipe.section_plan = host_plan
            recipe.section_pos_z = list(host_plan.positions_z)
            recipe.section_pos_ui = list(recipe.section_pos_z)

        for attr in (
            "start_valid",
            "start_ax0_abs",
            "standby_valid",
            "standby_ax0_abs",
            "standby_ax1_abs",
            "standby_ax4_abs",
            "ax2_len_valid",
            "ax2_len_abs",
            "ax2_rot_valid",
            "ax2_rot_abs",
        ):
            if hasattr(self._host_recipe(), attr):
                setattr(recipe, attr, getattr(self._host_recipe(), attr))

        try:
            recipe.calc_input_mode = self._norm_choice(
                getattr(host, "calc_input_mode_var").get(),
                default=str(self._fallback("calc_input_mode", "bin")),
                mapping={"raw 保留全部原始点": "raw", "bin 按角度分bin再降采样": "bin"},
            )
            recipe.bin_count = int(float(getattr(host, "bin_count_var").get()))
            recipe.bin_method = self._norm_choice(
                getattr(host, "bin_method_var").get(),
                default=str(self._fallback("bin_method", "median")),
                mapping={"median 中值": "median", "mean 均值": "mean"},
            )
            recipe.pp_mode = self._norm_choice(
                getattr(host, "pp_mode_var").get(),
                default=str(self._fallback("pp_mode", "p99_p1")),
                mapping={"strict max-min": "strict", "trim_0p01 剪裁1%": "trim_0p01", "p99_p1 百分位99-1": "p99_p1"},
            )
            recipe.theta_delay_s = float(getattr(host, "theta_delay_s_var").get())
        except Exception:
            recipe.calc_input_mode = str(self._fallback("calc_input_mode", "bin"))
            recipe.bin_count = int(self._fallback("bin_count", 90))
            recipe.bin_method = str(self._fallback("bin_method", "median"))
            recipe.pp_mode = str(self._fallback("pp_mode", "p99_p1"))
            recipe.theta_delay_s = float(self._fallback("theta_delay_s", 0.0) or 0.0)

        try:
            recipe_logger.debug(
                "RECIPE_APPLY name=%s section_count=%s rot_vel_velmove=%s",
                getattr(recipe, "name", None),
                getattr(recipe, "section_count", None),
                getattr(recipe, "rot_vel_velmove", None),
            )
        except Exception:
            pass
        try:
            host._log_ax3_speed_trace("recipe_apply_from_ui_commit", recipe_obj=recipe)
        except Exception:
            pass
        host.recipe = recipe
        return recipe

    def apply_data_to_ui(self, data: Mapping[str, Any]) -> None:
        host = self.host
        self._set_var_if_exists("recipe_name_var", str(data.get("name", "默认配方")))
        self._set_var_if_exists("pipe_len_var", str(data.get("pipe_len_mm", 1700.0)))
        self._set_var_if_exists("clamp_var", str(data.get("clamp_occupy_mm", 300.0)))
        clamp_wait = float(data.get("clamp_confirm_wait_s", self._fallback("clamp_confirm_wait_s", 3.0)))
        host.recipe.clamp_confirm_wait_s = clamp_wait
        self._set_var_if_exists("clamp_confirm_wait_s_var", str(clamp_wait))
        self._set_var_if_exists("margin_h_var", str(data.get("margin_head_mm", 20.0)))
        self._set_var_if_exists("margin_t_var", str(data.get("margin_tail_mm", 20.0)))
        self._set_var_if_exists("meas_total_len_var", str(data.get("meas_total_len_mm", 0.0)))
        self._set_var_if_exists("section_n_var", str(data.get("section_count", 12)))
        host.recipe.scan_axis = 0

        teach_mode = max(0, min(3, int(data.get("teach_axes_mode", getattr(host.recipe, "teach_axes_mode", 2)))))
        self._set_var_if_exists("teach_axes_mode_var", teach_mode)
        self._sync_combo('teach_axes_combo', str(teach_mode))

        self._set_var_if_exists("od_std_var", str(data.get("od_std_mm", 187.3)))
        self._set_var_if_exists("id_std_var", str(data.get("id_std_mm", 152.7)))
        self._set_var_if_exists("od_tol_var", str(data.get("od_tol_mm", 0.1)))
        self._set_var_if_exists("points_per_rev_var", str(data.get("points_per_rev", data.get("sample_count", 120))))
        self._set_var_if_exists("min_cov_var", str(data.get("sample_coverage", data.get("min_bin_coverage", self._fallback("min_bin_coverage", 0.95)))))
        self._set_var_if_exists("sample_timeout_var", str(data.get("section_timeout_s", data.get("sample_timeout_s", self._fallback("sample_timeout_s", 5.0)))))
        self._set_var_if_exists("max_revs_var", str(data.get("max_revs", data.get("max_revolutions", self._fallback("max_revolutions", 2.0)))))
        self._set_var_if_exists("sample_delay_s_var", str(data.get("sample_delay_s", self._fallback("sample_delay_s", 0.0))))

        rot_vel = float(data.get("rot_vel_velmove", data.get("rot_speed", self._fallback("rot_vel_velmove", 200.0))))
        host.recipe.rot_vel_velmove = rot_vel
        self._set_var_if_exists("rot_vel_velmove_var", str(rot_vel))

        fit_strategy = str(data.get("fit_strategy", self._fallback("fit_strategy", "b 原始点按bin权重均衡")))
        self._set_var_if_exists("fit_strategy_var", fit_strategy)
        self._sync_combo("fit_strategy_combo", fit_strategy)

        host.recipe.od_use_edges = bool(data.get("od_use_edges", data.get("od_algo_edges", self._fallback("od_use_edges", False))))
        self._set_var_if_exists("od_use_edges_var", host.recipe.od_use_edges)

        host.recipe.id_use_fit = bool(data.get("id_use_fit", data.get("id_algo_fit", self._fallback("id_use_fit", False))))
        self._set_var_if_exists("id_use_fit_var", host.recipe.id_use_fit)

        host.recipe.id_single_enable = False
        host.recipe.id_single_k = float(data.get("id_single_k", self._fallback("id_single_k", 1.0)))
        host.recipe.id_single_b = float(data.get("id_single_b", self._fallback("id_single_b", 0.0)))
        host.recipe.id_single_show_debug = bool(data.get("id_single_show_debug", self._fallback("id_single_show_debug", False)))
        self._set_var_if_exists("id_single_enable_var", host.recipe.id_single_enable)
        self._set_var_if_exists("id_single_k_var", str(host.recipe.id_single_k))
        self._set_var_if_exists("id_single_b_var", str(host.recipe.id_single_b))

        section_sampling_mode = str(
            data.get(
                "section_sampling_mode",
                data.get("scan_mode", self._fallback("section_sampling_mode", self._fallback("scan_mode", "sync"))),
            )
            or "sync"
        ).strip().lower() or "sync"
        if section_sampling_mode not in {"sync", "split"}:
            section_sampling_mode = "sync"
        host.recipe.section_sampling_mode = section_sampling_mode
        host.recipe.sampling_window_mode = str(
            data.get(
                "sampling_window_mode",
                "separate_channels" if section_sampling_mode == "split" else "shared",
            )
            or ("separate_channels" if section_sampling_mode == "split" else "shared")
        )
        host.recipe.scan_mode = section_sampling_mode
        self._set_var_if_exists("section_sampling_mode_var", host.recipe.section_sampling_mode)
        self._sync_combo("section_sampling_mode_combo", host.recipe.section_sampling_mode)
        self._set_var_if_exists("split_scan_var", host.recipe.scan_mode.startswith("split"))

        host.recipe.disable_id_modbus = bool(data.get("disable_id_modbus", self._fallback("disable_id_modbus", False)))
        if host.recipe.id_single_enable:
            host.recipe.disable_id_modbus = False
        self._set_var_if_exists("disable_id_modbus_var", host.recipe.disable_id_modbus)

        host.recipe.split_keep_spinning = True
        host.recipe.split_slip_check = bool(data.get("split_slip_check", self._fallback("split_slip_check", True)))
        host.recipe.split_slip_max_deg = float(data.get("split_slip_max_deg", self._fallback("split_slip_max_deg", 5.0)) or 5.0)
        host.recipe.split_omega_cv_max = float(data.get("split_omega_cv_max", self._fallback("split_omega_cv_max", 0.25)) or 0.25)
        self._set_var_if_exists("split_keep_spinning_var", host.recipe.split_keep_spinning)
        self._set_var_if_exists("split_slip_check_var", host.recipe.split_slip_check)

        host.recipe.calc_input_mode = str(data.get("calc_input_mode", self._fallback("calc_input_mode", "bin")))
        host.recipe.bin_count = int(data.get("bin_count", self._fallback("bin_count", 90)))
        host.recipe.bin_method = str(data.get("bin_method", self._fallback("bin_method", "median")))
        host.recipe.pp_mode = str(data.get("pp_mode", self._fallback("pp_mode", "p99_p1")))
        host.recipe.theta_delay_s = float(data.get("theta_delay_s", self._fallback("theta_delay_s", 0.0)) or 0.0)
        self._set_var_if_exists("calc_input_mode_var", "raw 保留全部原始点" if host.recipe.calc_input_mode == "raw" else "bin 按角度分bin再降采样")
        self._sync_combo("calc_input_mode_combo", getattr(host, "calc_input_mode_var").get() if hasattr(host, "calc_input_mode_var") else "")
        self._set_var_if_exists("bin_count_var", str(host.recipe.bin_count))
        self._set_var_if_exists("bin_method_var", "mean 均值" if host.recipe.bin_method == "mean" else "median 中值")
        self._sync_combo("bin_method_combo", getattr(host, "bin_method_var").get() if hasattr(host, "bin_method_var") else "")
        if host.recipe.pp_mode == "strict":
            pp_mode_display = "strict max-min"
        elif host.recipe.pp_mode == "trim_0p01":
            pp_mode_display = "trim_0p01 剪裁1%"
        else:
            pp_mode_display = "p99_p1 百分位99-1"
        self._set_var_if_exists("pp_mode_var", pp_mode_display)
        self._sync_combo("pp_mode_combo", pp_mode_display)
        self._set_var_if_exists("theta_delay_s_var", str(float(host.recipe.theta_delay_s)))

        try:
            host.recipe.len_enable = bool(data.get("len_enable", self._fallback("len_enable", False)))
            host._len_low_appr_legacy_z = None
            if "len_low_approach_abs" in data:
                host.recipe.len_low_approach_abs = float(data.get("len_low_approach_abs", self._fallback("len_low_approach_abs", 0.0)))
                self._set_var_if_exists("len_z_low_approach_var", str(float(host.recipe.len_low_approach_abs or 0.0)))
            elif "len_z_low_approach" in data:
                host._len_low_appr_legacy_z = float(data.get("len_z_low_approach"))
                host.recipe.len_z_low_approach = float(host._len_low_appr_legacy_z)
                self._set_var_if_exists("len_z_low_approach_var", str(host.recipe.len_z_low_approach))
            host.recipe.len_low_search_dist = float(data.get("len_low_search_dist", self._fallback("len_low_search_dist", 220.0)))
            host.recipe.len_high_search_dist = float(data.get("len_high_search_dist", self._fallback("len_high_search_dist", 220.0)))
            host.recipe.len_search_vel = float(data.get("len_search_vel", self._fallback("len_search_vel", 5.0)))
            host.recipe.len_search_timeout_s = float(data.get("len_search_timeout_s", self._fallback("len_search_timeout_s", 12.0)))
            host.recipe.len_tol_mm = float(data.get("len_tol_mm", self._fallback("len_tol_mm", 20.0)))
            host.recipe.len_high_margin = float(data.get("len_high_margin", self._fallback("len_high_margin", 20.0)))
            host.recipe.len_debounce_k = int(data.get("len_debounce_k", self._fallback("len_debounce_k", 6)))
            host.recipe.len_max_stale_ms = int(data.get("len_max_stale_ms", self._fallback("len_max_stale_ms", 300)))
            host.recipe.len_backoff_mm = float(data.get("len_backoff_mm", self._fallback("len_backoff_mm", 2.0)))
            self._set_var_if_exists("len_enable_var", host.recipe.len_enable)
            self._set_var_if_exists("len_low_search_dist_var", str(host.recipe.len_low_search_dist))
            self._set_var_if_exists("len_high_search_dist_var", str(host.recipe.len_high_search_dist))
            self._set_var_if_exists("len_search_vel_var", str(host.recipe.len_search_vel))
            self._set_var_if_exists("len_search_timeout_var", str(host.recipe.len_search_timeout_s))
            self._set_var_if_exists("len_tol_var", str(host.recipe.len_tol_mm))
            self._set_var_if_exists("len_high_margin_var", str(host.recipe.len_high_margin))
            self._set_var_if_exists("len_debounce_k_var", str(host.recipe.len_debounce_k))
            self._set_var_if_exists("len_max_stale_ms_var", str(host.recipe.len_max_stale_ms))
            self._set_var_if_exists("len_backoff_var", str(host.recipe.len_backoff_mm))
            try:
                host._refresh_length_info()
            except Exception:
                pass
        except Exception:
            pass

        section_plan_data = data.get("section_plan")
        section_plan_snapshot = None
        if isinstance(section_plan_data, Mapping):
            try:
                section_plan_snapshot = SectionPlanSnapshot.from_mapping(section_plan_data)
            except Exception:
                section_plan_snapshot = None

        pos_z = data.get("section_pos_z", [])
        pos_ui = data.get("section_pos_ui", [])
        if isinstance(section_plan_snapshot, SectionPlanSnapshot):
            host.recipe.section_plan = section_plan_snapshot
            host.recipe.section_pos_z = list(section_plan_snapshot.positions_z)
        elif isinstance(pos_z, list) and pos_z:
            host.recipe.section_plan = None
            host.recipe.section_pos_z = [float(x) for x in pos_z]
        elif isinstance(pos_ui, list) and pos_ui:
            host.recipe.section_plan = None
            host.recipe.section_pos_z = [float(x) for x in pos_ui]
        else:
            host.recipe.section_plan = None
            host.recipe.section_pos_z = host.recipe.compute_default_positions_z()
        host.recipe.section_pos_ui = list(host.recipe.section_pos_z)

        # Old recipe files may omit section-planning state fields. Reset them
        # up front so the new recipe cannot inherit the previous host state.
        self._reset_planning_state_defaults()
        for attr in (
            "standby_valid",
            "standby_ax0_abs",
            "standby_ax1_abs",
            "standby_ax4_abs",
            "start_valid",
            "start_ax0_abs",
            "ax2_len_valid",
            "ax2_len_abs",
            "ax2_rot_valid",
            "ax2_rot_abs",
        ):
            if attr in data:
                try:
                    setattr(host.recipe, attr, data[attr])
                except Exception:
                    pass

        self.ui_vars_to_recipe()
        for method_name in (
            "_apply_start_anchor_from_recipe",
            "_refresh_recipe_table",
            "_refresh_auto_std_panel",
            "_refresh_standby_pos",
            "_refresh_center_positions",
        ):
            try:
                getattr(host, method_name)()
            except Exception:
                pass

    def recipe_to_ui_vars(self, recipe: Recipe) -> None:
        self.apply_data_to_ui(self.recipe_to_dict(recipe))

    def apply_from_ui(self) -> Recipe:
        return self.ui_vars_to_recipe()

    def dump_dict(self, recipe: Recipe) -> dict:
        return self.recipe_to_dict(recipe)


__all__ = ["RecipeFormMapper"]
