from __future__ import annotations

"""Map between legacy recipe UI vars, Recipe objects, and persisted dict data."""

import logging
from typing import Any, Mapping, Protocol

from core.models import Recipe, SectionPlanSnapshot

recipe_logger = logging.getLogger("frp.recipe")


class RecipeFormViewPort(Protocol):
    @property
    def recipe(self) -> Recipe: ...

    @recipe.setter
    def recipe(self, value: Recipe) -> None: ...

    def get_var_value(self, name: str, default: Any = None) -> Any: ...
    def set_var_value(self, name: str, value: Any) -> None: ...
    def sync_combo_value(self, combo_name: str, value: str) -> None: ...
    def log_ax3_speed_trace(self, location_name: str, *, recipe_obj: Any = None) -> None: ...
    def refresh_length_info(self) -> None: ...
    def set_len_low_approach_legacy_z(self, value: float | None) -> None: ...
    def after_recipe_data_applied(self) -> None: ...


class RecipeFormMapper:
    """Compatibility mapper for the current Tk form and Recipe model."""

    def __init__(self, view: RecipeFormViewPort) -> None:
        self.view = view

    def _host_recipe(self) -> Recipe:
        return self.view.recipe

    def _fallback(self, attr: str, default: Any) -> Any:
        return getattr(self._host_recipe(), attr, default)

    def _get_var(self, name: str) -> Any:
        return self.view.get_var_value(name)

    def _get_var_or_fallback(self, name: str, fallback_attr: str, default: Any) -> Any:
        return self.view.get_var_value(name, self._fallback(fallback_attr, default))

    def _set_var_if_exists(self, name: str, value: Any) -> None:
        self.view.set_var_value(name, value)

    def _sync_combo(self, combo_name: str, value: str) -> None:
        self.view.sync_combo_value(combo_name, value)

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
        recipe.meas_total_len_mm = float(self._get_var_or_fallback("meas_total_len_var", "meas_total_len_mm", 0.0))
        recipe.section_count = int(float(self._get_var("section_n_var")))
        recipe.scan_axis = 0
        recipe.teach_axes_mode = int(self._get_var_or_fallback("teach_axes_mode_var", "teach_axes_mode", 2))
        recipe.od_std_mm = float(self._get_var("od_std_var"))
        recipe.id_std_mm = float(self._get_var("id_std_var"))
        recipe.od_tol_mm = float(self._get_var("od_tol_var"))
        recipe.points_per_rev = int(float(self._get_var("points_per_rev_var")))
        recipe.min_bin_coverage = float(self._get_var("min_cov_var"))
        recipe.sample_timeout_s = float(self._get_var("sample_timeout_var"))
        recipe.max_revolutions = float(self._get_var("max_revs_var"))
        recipe.sample_delay_s = float(self._get_var_or_fallback("sample_delay_s_var", "sample_delay_s", 0.0))
        recipe.rot_vel_velmove = float(self._get_var_or_fallback("rot_vel_velmove_var", "rot_vel_velmove", 200.0))
        recipe.fit_strategy = str(self._get_var_or_fallback("fit_strategy_var", "fit_strategy", "b 原始点按bin权重均衡"))
        recipe.od_use_edges = bool(self._get_var_or_fallback("od_use_edges_var", "od_use_edges", False))
        recipe.id_use_fit = bool(self._get_var_or_fallback("id_use_fit_var", "id_use_fit", False))
        recipe.id_single_enable = False
        recipe.id_single_k = float(self._get_var_or_fallback("id_single_k_var", "id_single_k", 1.0))
        recipe.id_single_b = float(self._get_var_or_fallback("id_single_b_var", "id_single_b", 0.0))
        recipe.id_single_show_debug = bool(self._fallback("id_single_show_debug", False))
        sampling_mode_value = self._get_var_or_fallback(
            "section_sampling_mode_var",
            "section_sampling_mode",
            self._fallback("scan_mode", "sync"),
        )
        sampling_mode = str(sampling_mode_value or "").strip().lower()
        if sampling_mode not in {"sync", "split"}:
            sampling_mode = "split" if bool(self._get_var_or_fallback("split_scan_var", "scan_mode", False)) else "sync"
        recipe.section_sampling_mode = sampling_mode
        recipe.sampling_window_mode = "separate_channels" if sampling_mode == "split" else "shared"
        recipe.scan_mode = sampling_mode
        recipe.disable_id_modbus = bool(self._get_var_or_fallback("disable_id_modbus_var", "disable_id_modbus", False))
        if recipe.id_single_enable:
            recipe.disable_id_modbus = False
        recipe.split_keep_spinning = True
        recipe.split_slip_check = bool(self._get_var_or_fallback("split_slip_check_var", "split_slip_check", True))
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
                self._get_var("calc_input_mode_var"),
                default=str(self._fallback("calc_input_mode", "bin")),
                mapping={"raw 保留全部原始点": "raw", "bin 按角度分bin再降采样": "bin"},
            )
            recipe.bin_count = int(float(self._get_var("bin_count_var")))
            recipe.bin_method = self._norm_choice(
                self._get_var("bin_method_var"),
                default=str(self._fallback("bin_method", "median")),
                mapping={"median 中值": "median", "mean 均值": "mean"},
            )
            recipe.pp_mode = self._norm_choice(
                self._get_var("pp_mode_var"),
                default=str(self._fallback("pp_mode", "p99_p1")),
                mapping={"strict max-min": "strict", "trim_0p01 剪裁1%": "trim_0p01", "p99_p1 百分位99-1": "p99_p1"},
            )
            recipe.theta_delay_s = float(self._get_var("theta_delay_s_var"))
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
            self.view.log_ax3_speed_trace("recipe_apply_from_ui_commit", recipe_obj=recipe)
        except Exception:
            pass
        self.view.recipe = recipe
        return recipe

    def apply_data_to_ui(self, data: Mapping[str, Any]) -> None:
        recipe = self.view.recipe
        self._set_var_if_exists("recipe_name_var", str(data.get("name", "默认配方")))
        self._set_var_if_exists("pipe_len_var", str(data.get("pipe_len_mm", 1700.0)))
        self._set_var_if_exists("clamp_var", str(data.get("clamp_occupy_mm", 300.0)))
        clamp_wait = float(data.get("clamp_confirm_wait_s", self._fallback("clamp_confirm_wait_s", 3.0)))
        recipe.clamp_confirm_wait_s = clamp_wait
        self._set_var_if_exists("clamp_confirm_wait_s_var", str(clamp_wait))
        self._set_var_if_exists("margin_h_var", str(data.get("margin_head_mm", 20.0)))
        self._set_var_if_exists("margin_t_var", str(data.get("margin_tail_mm", 20.0)))
        self._set_var_if_exists("meas_total_len_var", str(data.get("meas_total_len_mm", 0.0)))
        self._set_var_if_exists("section_n_var", str(data.get("section_count", 12)))
        recipe.scan_axis = 0

        teach_mode = max(0, min(3, int(data.get("teach_axes_mode", getattr(recipe, "teach_axes_mode", 2)))))
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
        recipe.rot_vel_velmove = rot_vel
        self._set_var_if_exists("rot_vel_velmove_var", str(rot_vel))

        fit_strategy = str(data.get("fit_strategy", self._fallback("fit_strategy", "b 原始点按bin权重均衡")))
        self._set_var_if_exists("fit_strategy_var", fit_strategy)
        self._sync_combo("fit_strategy_combo", fit_strategy)

        recipe.od_use_edges = bool(data.get("od_use_edges", data.get("od_algo_edges", self._fallback("od_use_edges", False))))
        self._set_var_if_exists("od_use_edges_var", recipe.od_use_edges)

        recipe.id_use_fit = bool(data.get("id_use_fit", data.get("id_algo_fit", self._fallback("id_use_fit", False))))
        self._set_var_if_exists("id_use_fit_var", recipe.id_use_fit)

        recipe.id_single_enable = False
        recipe.id_single_k = float(data.get("id_single_k", self._fallback("id_single_k", 1.0)))
        recipe.id_single_b = float(data.get("id_single_b", self._fallback("id_single_b", 0.0)))
        recipe.id_single_show_debug = bool(data.get("id_single_show_debug", self._fallback("id_single_show_debug", False)))
        self._set_var_if_exists("id_single_enable_var", recipe.id_single_enable)
        self._set_var_if_exists("id_single_k_var", str(recipe.id_single_k))
        self._set_var_if_exists("id_single_b_var", str(recipe.id_single_b))

        section_sampling_mode = str(
            data.get(
                "section_sampling_mode",
                data.get("scan_mode", self._fallback("section_sampling_mode", self._fallback("scan_mode", "sync"))),
            )
            or "sync"
        ).strip().lower() or "sync"
        if section_sampling_mode not in {"sync", "split"}:
            section_sampling_mode = "sync"
        recipe.section_sampling_mode = section_sampling_mode
        recipe.sampling_window_mode = str(
            data.get(
                "sampling_window_mode",
                "separate_channels" if section_sampling_mode == "split" else "shared",
            )
            or ("separate_channels" if section_sampling_mode == "split" else "shared")
        )
        recipe.scan_mode = section_sampling_mode
        self._set_var_if_exists("section_sampling_mode_var", recipe.section_sampling_mode)
        self._sync_combo("section_sampling_mode_combo", recipe.section_sampling_mode)
        self._set_var_if_exists("split_scan_var", recipe.scan_mode.startswith("split"))

        recipe.disable_id_modbus = bool(data.get("disable_id_modbus", self._fallback("disable_id_modbus", False)))
        if recipe.id_single_enable:
            recipe.disable_id_modbus = False
        self._set_var_if_exists("disable_id_modbus_var", recipe.disable_id_modbus)

        recipe.split_keep_spinning = True
        recipe.split_slip_check = bool(data.get("split_slip_check", self._fallback("split_slip_check", True)))
        recipe.split_slip_max_deg = float(data.get("split_slip_max_deg", self._fallback("split_slip_max_deg", 5.0)) or 5.0)
        recipe.split_omega_cv_max = float(data.get("split_omega_cv_max", self._fallback("split_omega_cv_max", 0.25)) or 0.25)
        self._set_var_if_exists("split_keep_spinning_var", recipe.split_keep_spinning)
        self._set_var_if_exists("split_slip_check_var", recipe.split_slip_check)

        recipe.calc_input_mode = str(data.get("calc_input_mode", self._fallback("calc_input_mode", "bin")))
        recipe.bin_count = int(data.get("bin_count", self._fallback("bin_count", 90)))
        recipe.bin_method = str(data.get("bin_method", self._fallback("bin_method", "median")))
        recipe.pp_mode = str(data.get("pp_mode", self._fallback("pp_mode", "p99_p1")))
        recipe.theta_delay_s = float(data.get("theta_delay_s", self._fallback("theta_delay_s", 0.0)) or 0.0)
        calc_mode_display = "raw 保留全部原始点" if recipe.calc_input_mode == "raw" else "bin 按角度分bin再降采样"
        self._set_var_if_exists("calc_input_mode_var", calc_mode_display)
        self._sync_combo("calc_input_mode_combo", calc_mode_display)
        self._set_var_if_exists("bin_count_var", str(recipe.bin_count))
        bin_method_display = "mean 均值" if recipe.bin_method == "mean" else "median 中值"
        self._set_var_if_exists("bin_method_var", bin_method_display)
        self._sync_combo("bin_method_combo", bin_method_display)
        if recipe.pp_mode == "strict":
            pp_mode_display = "strict max-min"
        elif recipe.pp_mode == "trim_0p01":
            pp_mode_display = "trim_0p01 剪裁1%"
        else:
            pp_mode_display = "p99_p1 百分位99-1"
        self._set_var_if_exists("pp_mode_var", pp_mode_display)
        self._sync_combo("pp_mode_combo", pp_mode_display)
        self._set_var_if_exists("theta_delay_s_var", str(float(recipe.theta_delay_s)))

        try:
            recipe.len_enable = bool(data.get("len_enable", self._fallback("len_enable", False)))
            self.view.set_len_low_approach_legacy_z(None)
            if "len_low_approach_abs" in data:
                recipe.len_low_approach_abs = float(data.get("len_low_approach_abs", self._fallback("len_low_approach_abs", 0.0)))
                self._set_var_if_exists("len_z_low_approach_var", str(float(recipe.len_low_approach_abs or 0.0)))
            elif "len_z_low_approach" in data:
                legacy_raw = data.get("len_z_low_approach")
                legacy_z = float(0.0 if legacy_raw is None else legacy_raw)
                self.view.set_len_low_approach_legacy_z(legacy_z)
                recipe.len_z_low_approach = legacy_z
                self._set_var_if_exists("len_z_low_approach_var", str(recipe.len_z_low_approach))
            recipe.len_low_search_dist = float(data.get("len_low_search_dist", self._fallback("len_low_search_dist", 220.0)))
            recipe.len_high_search_dist = float(data.get("len_high_search_dist", self._fallback("len_high_search_dist", 220.0)))
            recipe.len_search_vel = float(data.get("len_search_vel", self._fallback("len_search_vel", 5.0)))
            recipe.len_search_timeout_s = float(data.get("len_search_timeout_s", self._fallback("len_search_timeout_s", 12.0)))
            recipe.len_tol_mm = float(data.get("len_tol_mm", self._fallback("len_tol_mm", 20.0)))
            recipe.len_high_margin = float(data.get("len_high_margin", self._fallback("len_high_margin", 20.0)))
            recipe.len_debounce_k = int(data.get("len_debounce_k", self._fallback("len_debounce_k", 6)))
            recipe.len_max_stale_ms = int(data.get("len_max_stale_ms", self._fallback("len_max_stale_ms", 300)))
            recipe.len_backoff_mm = float(data.get("len_backoff_mm", self._fallback("len_backoff_mm", 2.0)))
            self._set_var_if_exists("len_enable_var", recipe.len_enable)
            self._set_var_if_exists("len_low_search_dist_var", str(recipe.len_low_search_dist))
            self._set_var_if_exists("len_high_search_dist_var", str(recipe.len_high_search_dist))
            self._set_var_if_exists("len_search_vel_var", str(recipe.len_search_vel))
            self._set_var_if_exists("len_search_timeout_var", str(recipe.len_search_timeout_s))
            self._set_var_if_exists("len_tol_var", str(recipe.len_tol_mm))
            self._set_var_if_exists("len_high_margin_var", str(recipe.len_high_margin))
            self._set_var_if_exists("len_debounce_k_var", str(recipe.len_debounce_k))
            self._set_var_if_exists("len_max_stale_ms_var", str(recipe.len_max_stale_ms))
            self._set_var_if_exists("len_backoff_var", str(recipe.len_backoff_mm))
            self.view.refresh_length_info()
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
            recipe.section_plan = section_plan_snapshot
            recipe.section_pos_z = list(section_plan_snapshot.positions_z)
        elif isinstance(pos_z, list) and pos_z:
            recipe.section_plan = None
            recipe.section_pos_z = [float(x) for x in pos_z]
        elif isinstance(pos_ui, list) and pos_ui:
            recipe.section_plan = None
            recipe.section_pos_z = [float(x) for x in pos_ui]
        else:
            recipe.section_plan = None
            recipe.section_pos_z = recipe.compute_default_positions_z()
        recipe.section_pos_ui = list(recipe.section_pos_z)

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
                    setattr(recipe, attr, data[attr])
                except Exception:
                    pass

        self.ui_vars_to_recipe()
        self.view.after_recipe_data_applied()

    def recipe_to_ui_vars(self, recipe: Recipe) -> None:
        self.apply_data_to_ui(self.recipe_to_dict(recipe))

    def apply_from_ui(self) -> Recipe:
        return self.ui_vars_to_recipe()

    def dump_dict(self, recipe: Recipe) -> dict:
        return self.recipe_to_dict(recipe)


__all__ = ["RecipeFormMapper", "RecipeFormViewPort"]
