from __future__ import annotations

"""Main-screen view-state mixin for AppHost."""

from typing import TYPE_CHECKING, Any

from core.models import Recipe


class HostMainViewMixin:
    """Mixin providing main-screen display refresh and layout helpers."""

    recipe: Recipe
    _id_samples: Any
    _run_serial: Any
    pipe_sn_var: Any
    meas_seq_var: Any
    id_n_var: Any
    id_avg_var: Any
    id_dev_var: Any
    id_round_var: Any
    ui_meas_mode_var: Any
    max_id_dev_var: Any
    max_id_round_var: Any
    id_mean_var: Any
    id_dpp_var: Any
    id_range_var: Any
    id_slope_var: Any
    id_tilt_var: Any
    id_endoff_var: Any
    axis_dist_var: Any
    conc_max_var: Any
    axis_span_max_var: Any

    if TYPE_CHECKING:
        def _auto_clear_ui(self, preserve_run: bool = False) -> None: ...
        def _on_result_select(self, event: Any = None) -> Any: ...
        def _main_ui_widget(self, name: str) -> Any: ...
        def _main_view_state(self, name: str, default: Any = None) -> Any: ...
        def after(self, ms: Any, func: Any | None = None, *args: Any) -> Any: ...

    def _refresh_measurement_display(self):
        self._auto_clear_ui(preserve_run=True)
        try:
            if getattr(self, "_run_serial", None):
                self.pipe_sn_var.set(str(self._run_serial))
                self.meas_seq_var.set(str(self._run_serial).split("-")[-1])
        except Exception:
            pass

    def handle_main_result_selection(self, event=None):
        return self._on_result_select(event)

    def refresh_main_summary_panel(self):
        return self._refresh_auto_std_panel()


    def _refresh_id_stats(self) -> None:
        """Compute ID metrics from recent CL OUT3 samples.

        Metrics (aligned with OD semantics):
        - 平均内径: mean(ID)
        - 内径偏差: mean(ID) - recipe.id_std_mm
        - 内径真圆度: max(ID) - min(ID)
        """
        try:
            samples = list(getattr(self, "_id_samples", []) or [])
            n = len(samples)
            self.id_n_var.set(str(n))
            if n <= 0:
                self.id_avg_var.set("--")
                self.id_dev_var.set("--")
                self.id_round_var.set("--")
                return

            avg = float(sum(samples) / n)
            mn = float(min(samples))
            mx = float(max(samples))
            roundness = mx - mn

            # deviation against recipe standard
            try:
                std = float(getattr(self, "recipe", None).id_std_mm)  # type: ignore[attr-defined]
            except Exception:
                std = float(getattr(getattr(self, "recipe", None), "id_std_mm", 0.0) or 0.0)

            dev = avg - std

            self.id_avg_var.set(f"{avg:.3f}")
            self.id_dev_var.set(f"{dev:+.3f}")
            self.id_round_var.set(f"{roundness:.3f}")
        except Exception:
            pass

    def _refresh_auto_std_panel(self):
        r = self.recipe
        # UI: keep the main-screen standard value concise; tolerance is shown/edited in recipe screen.
        lbl_od_std = self._main_ui_widget('lbl_od_std')
        lbl_id_std = self._main_ui_widget('lbl_id_std')
        if lbl_od_std is not None:
            lbl_od_std.config(text=f"{r.od_std_mm:.3f} mm")
        if lbl_id_std is not None:
            lbl_id_std.config(text=f"{r.id_std_mm:.3f} mm")

        # Apply main-screen UI mode (SYNC/SPLIT/OD_ONLY)
        try:
            self._apply_main_ui_mode()
        except Exception:
            pass

    def _ui_get_meas_mode(self) -> str:
        """Return one of: 'SYNC', 'SPLIT', 'OD_ONLY', 'ID_SINGLE', 'SPLIT_SINGLE'."""
        r = getattr(self, 'recipe', None)
        try:
            sm = str(
                getattr(
                    r,
                    'section_sampling_mode',
                    getattr(r, 'scan_mode', 'sync'),
                )
                or 'sync'
            ).strip().lower()
        except Exception:
            sm = 'sync'
        try:
            if bool(getattr(r, 'id_single_enable', False)):
                return 'SPLIT_SINGLE' if sm.startswith('split') else 'ID_SINGLE'
        except Exception:
            pass
        if sm.startswith('split'):
            return 'SPLIT'
        # OD-only / speedtest: disable ID Modbus reads (recipe)
        try:
            if bool(getattr(r, 'disable_id_modbus', False)):
                return 'OD_ONLY'
        except Exception:
            pass
        return 'SYNC'

    def _resize_result_tree_columns(self, tree: Any, columns: tuple[str, ...]) -> None:
        if tree is None or not columns:
            return
        try:
            preferred = dict(self._main_view_state('tree_column_widths', {}) or {})
            minimums = dict(self._main_view_state('tree_column_min_widths', {}) or {})
        except Exception:
            preferred, minimums = {}, {}

        pref: dict[str, int] = {}
        mins: dict[str, int] = {}
        for col in columns:
            try:
                current = int(tree.column(col, 'width') or 0)
            except Exception:
                current = 0
            pref[col] = max(1, int(preferred.get(col, current or 100) or 100))
            mins[col] = max(1, int(minimums.get(col, min(pref[col], 70)) or 70))
            if pref[col] < mins[col]:
                pref[col] = mins[col]

        try:
            available = max(0, int(tree.winfo_width() or 0) - 4)
        except Exception:
            available = 0

        total_pref = sum(pref.values())
        total_min = sum(mins.values())
        widths = dict(pref)

        if available > 0 and available < total_pref and total_pref > total_min:
            shrink_total = min(total_pref - available, total_pref - total_min)
            shrinkable = {col: max(0, pref[col] - mins[col]) for col in columns}
            shrink_base = sum(shrinkable.values())
            used = 0
            for col in columns:
                if shrink_base <= 0:
                    break
                shrink = int(round(shrink_total * shrinkable[col] / shrink_base))
                shrink = min(shrink, shrinkable[col])
                widths[col] = pref[col] - shrink
                used += shrink
            remainder = shrink_total - used
            for col in reversed(columns):
                if remainder <= 0:
                    break
                room = max(0, widths[col] - mins[col])
                if room <= 0:
                    continue
                delta = min(room, remainder)
                widths[col] -= delta
                remainder -= delta
        elif available > total_pref:
            extra = available - total_pref
            weights = {
                'x_ui': 2,
                'od_fit_res': 1,
                'od_pp_rob': 1,
                'id_round': 1,
                'concentricity': 1,
                'cov_reason': 2,
            }
            active_weights = {col: weights.get(col, 0) for col in columns}
            weight_sum = sum(active_weights.values())
            if weight_sum <= 0:
                active_weights = {col: 1 for col in columns}
                weight_sum = len(columns)
            used = 0
            for col in columns:
                add = int(extra * active_weights[col] / weight_sum)
                widths[col] = pref[col] + add
                used += add
            for col in columns:
                if used >= extra:
                    break
                widths[col] += 1
                used += 1

        for col in columns:
            try:
                tree.column(col, width=max(mins[col], int(widths[col])), minwidth=mins[col], stretch=False)
            except Exception:
                pass

    def _schedule_result_tree_column_resize(self, tree: Any, columns: tuple[str, ...]) -> None:
        try:
            self.after(0, lambda: self._resize_result_tree_columns(tree, tuple(columns)))
        except Exception:
            self._resize_result_tree_columns(tree, tuple(columns))

    def _apply_main_ui_mode(self) -> None:
        """Adjust main-screen widgets by measurement mode."""
        mode = self._ui_get_meas_mode()
        # Status line
        try:
            if mode == 'OD_ONLY':
                self.ui_meas_mode_var.set('检测模式：仅外径（OD Only）')
            elif mode in ('SPLIT', 'SPLIT_SINGLE'):
                if mode == 'SPLIT_SINGLE':
                    self.ui_meas_mode_var.set('检测模式：分圈（ID单探头）')
                else:
                    self.ui_meas_mode_var.set('检测模式：分圈（OD→ID）')
            elif mode == 'ID_SINGLE':
                self.ui_meas_mode_var.set('检测模式：同步（ID单探头）')
            else:
                self.ui_meas_mode_var.set('检测模式：同步（OD+ID）')
        except Exception:
            pass

        # Treeview displaycolumns presets (stored by main_screen.build)
        try:
            tree = self._main_ui_widget('result_tree')
            sync_cols = self._main_view_state('tree_displaycols_sync')
            split_cols = self._main_view_state('tree_displaycols_split')
            od_only_cols = self._main_view_state('tree_displaycols_od_only')
            if tree is not None and sync_cols:
                if mode == 'OD_ONLY':
                    active_cols = tuple(od_only_cols or ())
                elif mode in ('SPLIT', 'SPLIT_SINGLE'):
                    active_cols = tuple(split_cols or ())
                else:
                    active_cols = tuple(sync_cols or ())
                tree.configure(displaycolumns=active_cols)
                self._schedule_result_tree_column_resize(tree, active_cols)
        except Exception:
            pass

        # Summary panel placeholders: OD_ONLY disables ID/cross fields
        if mode == 'OD_ONLY':
            try:
                self.max_id_dev_var.set('--（OD Only）')
                self.max_id_round_var.set('--（OD Only）')
                self.id_mean_var.set('--（OD Only）')
                self.id_dpp_var.set('--（OD Only）')
                self.id_range_var.set('--（OD Only）')
                self.id_slope_var.set('--（OD Only）')
                self.id_tilt_var.set('--（OD Only）')
                self.id_endoff_var.set('--（OD Only）')
            except Exception:
                pass
            try:
                self.axis_dist_var.set('--（需要ID）')
                self.conc_max_var.set('--（需要ID）')
                self.axis_span_max_var.set('--（需要ID）')
            except Exception:
                pass
