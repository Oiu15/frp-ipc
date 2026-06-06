from __future__ import annotations

"""Recipe store and section-plan mixin for AppHost."""

import logging
import tkinter as tk
from dataclasses import replace
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Any, Mapping, Optional, cast

from application.recipe_form_mapper import RecipeFormMapper
from core.models import AxisCal, AxisComm, Recipe, SectionPlanSnapshot
from domain.planning import (
    build_recipe_section_plan,
    rebuild_recipe_section_plan,
    section_plan_from_snapshot,
    section_plan_is_compatible,
    section_plan_snapshot_from_plan,
)
from utils.logger import log

recipe_logger = logging.getLogger("frp.recipe")


class HostRecipeMixin:
    """Mixin providing recipe UI mapping, persistence, and section-plan table helpers."""

    axis_cal: AxisCal
    recipe: Recipe
    recipe_store: Any
    recipe_name_var: Any
    _recipe_screen_presenter: Any

    if TYPE_CHECKING:
        def _recipe_ui_widget(self, name: str) -> Any: ...
        def get_axis_copy(self, axis: int) -> AxisComm: ...
        def get_recipe_copy(self) -> Recipe: ...
        def _refresh_auto_std_panel(self): ...
        def _log_ax3_speed_trace(
            self,
            location_name: str,
            *,
            recipe_obj: Any = None,
            caller_name: Optional[str] = None,
        ) -> None: ...

    def _recipe_apply_from_ui(self) -> Recipe:
        """Read recipe fields from UI into self.recipe (and return a copy)."""
        return RecipeFormMapper(self._recipe_screen_presenter).ui_vars_to_recipe()

    def _recipe_compute(self):
        try:
            r = self._recipe_apply_from_ui()
            previous = getattr(r, "section_plan", None)
            taught_exists = (
                isinstance(previous, SectionPlanSnapshot)
                and any(str(section.source).lower() == "taught" for section in previous.sections)
            )
            preserve_taught = False
            if taught_exists:
                choice = self._ask_section_plan_recompute_choice()
                if choice == "cancel":
                    return
                preserve_taught = choice == "preserve"
            soft_limits = self._section_plan_context()
            section_plan = rebuild_recipe_section_plan(
                r,
                self.axis_cal,
                soft_limits_abs=soft_limits,
                previous_snapshot=previous if isinstance(previous, SectionPlanSnapshot) else None,
                preserve_taught=preserve_taught,
            )
            self._bind_section_plan_to_recipe(r, section_plan)
            self.recipe = r
            self._refresh_recipe_table()
            self._refresh_auto_std_panel()

            try:
                r = self.get_recipe_copy()
                log("AUTO_START", section_count=getattr(r,'section_count',None), points_per_rev=getattr(r,'points_per_rev',None), min_bin_coverage=getattr(r,'min_bin_coverage',None), timeout_s=getattr(r,'sample_timeout_s',None), max_revolutions=getattr(r,'max_revolutions',None))
            except Exception:
                log("AUTO_START", section_count="unknown")
        except Exception as e:
            messagebox.showerror("配方计算错误", str(e))

    
    # -------------------------
    # Recipe store (backend)
    # -------------------------
    def _recipe_store_init(self) -> None:
        """Initialize recipe dropdown and auto-load last recipe."""
        self._recipe_refresh_dropdown()
        # auto load last
        last = None
        try:
            idx = self.recipe_store.load_index()
            last = str(idx.get("last_recipe", "")).strip() or None
        except Exception:
            last = None

        # prefer last recipe if exists; otherwise keep current UI values
        if last:
            try:
                self._recipe_load_from_store(last, show_msg=False)
                return
            except Exception:
                pass

        # if current name exists in store, load it (for consistency)
        try:
            cur = str(self.recipe_name_var.get()).strip()
            if cur:
                self._recipe_load_from_store(cur, show_msg=False)
        except Exception:
            pass

    def _recipe_refresh_dropdown(self) -> None:
        """Refresh recipe name combobox values."""
        try:
            names = self.recipe_store.list_names()
        except Exception:
            names = []
        if "默认配方" not in names:
            names.insert(0, "默认配方")
        combo = self._recipe_ui_widget('recipe_name_combo')
        try:
            if combo is not None:
                combo["values"] = names
        except Exception:
            pass

    def _on_recipe_selected(self, _evt=None) -> None:
        """Recipe combobox selected -> auto load."""
        name = str(self.recipe_name_var.get()).strip()
        if not name:
            return
        try:
            self._recipe_load_from_store(name, show_msg=False)
        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def _on_recipe_enter(self, _evt=None) -> None:
        """Enter in recipe name: if exists, load; otherwise treat as new name."""
        name = str(self.recipe_name_var.get()).strip()
        if not name:
            return
        try:
            self._recipe_load_from_store(name, show_msg=False)
        except FileNotFoundError:
            return
        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def _recipe_dump_dict(self, r: Recipe) -> dict:
        return RecipeFormMapper(self._recipe_screen_presenter).recipe_to_dict(r)

    def _recipe_apply_data_to_ui(self, data: dict) -> None:
        """Apply recipe dict to UI vars and internal recipe object (no dialogs)."""
        RecipeFormMapper(self._recipe_screen_presenter).apply_data_to_ui(data)

    def _recipe_load_from_store(self, name: str, *, show_msg: bool = False) -> None:
        data = self.recipe_store.load(name)
        had_section_plan = isinstance(data.get("section_plan"), Mapping)
        tree = self._recipe_ui_widget('recipe_tree')
        if tree is not None:
            try:
                tree.delete(*tree.get_children())
            except Exception:
                pass
        self._recipe_apply_data_to_ui(data)
        if not had_section_plan:
            try:
                self._ensure_recipe_section_plan(self.recipe)
                self.recipe_store.save(self.recipe.name, self._recipe_dump_dict(self.recipe))
            except Exception:
                pass
        try:
            recipe_logger.info("RECIPE_LOAD name=%s", name)
        except Exception:
            pass
        self._log_ax3_speed_trace("recipe_load_complete")
        # refresh dropdown in case new files appear
        self._recipe_refresh_dropdown()
        # remember last
        try:
            self.recipe_store.save_index({"last_recipe": str(self.recipe_name_var.get()).strip()})
        except Exception:
            pass
        if show_msg:
            messagebox.showinfo("加载成功", f"已加载配方：{self.recipe_name_var.get()}")

    def _recipe_save_backend(self) -> None:
        try:
            r = self._recipe_apply_from_ui()
            self._ensure_recipe_section_plan(r)
            data = self._recipe_dump_dict(r)
            safe = self.recipe_store.save(r.name, data)
            # sync name if sanitized
            if safe != r.name:
                self.recipe_name_var.set(safe)
                try:
                    self.recipe.name = safe
                except Exception:
                    pass
            self._recipe_refresh_dropdown()
            try:
                self.recipe_store.save_index({"last_recipe": safe})
            except Exception:
                pass
            save_path = self.recipe_store.root / f"{safe}.json"
            try:
                recipe_logger.info("RECIPE_SAVE name=%s path=%s", safe, save_path)
            except Exception:
                pass
            messagebox.showinfo("保存成功", f"已保存：{save_path}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def _recipe_delete_backend(self) -> None:
        name = str(self.recipe_name_var.get()).strip()
        if not name:
            return
        if name == "默认配方":
            messagebox.showinfo("提示", "默认配方不允许删除。")
            return
        if not messagebox.askyesno("确认删除", f"确定删除配方“{name}”吗？此操作不可恢复。"):
            return
        try:
            self.recipe_store.delete(name)
            self._recipe_refresh_dropdown()
            # reset to default values (do not auto-create file)
            self.recipe_name_var.set("默认配方")
            self._recipe_apply_data_to_ui(self._recipe_dump_dict(Recipe()))
            try:
                self.recipe_store.save_index({"last_recipe": "默认配方"})
            except Exception:
                pass
            messagebox.showinfo("删除成功", f"已删除配方：{name}")
        except Exception as e:
            messagebox.showerror("删除失败", str(e))

    def _section_plan_context(self) -> dict[int, tuple[float, float]]:
        soft_limits = {
            0: (float(self.get_axis_copy(0).softlim_pos), float(self.get_axis_copy(0).softlim_neg)),
            1: (float(self.get_axis_copy(1).softlim_pos), float(self.get_axis_copy(1).softlim_neg)),
            4: (float(self.get_axis_copy(4).softlim_pos), float(self.get_axis_copy(4).softlim_neg)),
        }
        return soft_limits

    def _compute_recipe_section_plan(self, recipe: Recipe):
        soft_limits = self._section_plan_context()
        return build_recipe_section_plan(
            recipe,
            self.axis_cal,
            soft_limits_abs=soft_limits,
        )

    def _bind_section_plan_to_recipe(self, recipe: Recipe, section_plan) -> SectionPlanSnapshot:
        snapshot = section_plan_snapshot_from_plan(section_plan)
        recipe.section_plan = snapshot
        recipe.section_pos_z = list(snapshot.positions_z)
        recipe.section_pos_ui = list(recipe.section_pos_z)
        return snapshot

    def _save_taught_section_to_recipe(self, recipe: Recipe, recipe_index: int, z_od_disp: float) -> None:
        previous = getattr(recipe, "section_plan", None)
        sources: dict[int, str] = {}
        if isinstance(previous, SectionPlanSnapshot) and section_plan_is_compatible(recipe, previous):
            sources = {int(row.section_index) - 1: str(row.source) for row in previous.sections}

        positions = list(getattr(recipe, "section_pos_z", []))
        if len(positions) != int(recipe.section_count):
            positions = list(recipe.compute_default_positions_z())
        positions[int(recipe_index)] = float(z_od_disp)
        recipe.section_pos_z = positions
        recipe.section_pos_ui = list(positions)
        recipe.section_plan = None

        section_plan = self._compute_recipe_section_plan(recipe)
        rows = []
        for row in section_plan.sections:
            source = "taught" if int(row.section_index) - 1 == int(recipe_index) else sources.get(int(row.section_index) - 1, row.source)
            rows.append(replace(row, source=source))
        self._bind_section_plan_to_recipe(recipe, replace(section_plan, sections=tuple(rows)))

    def _ensure_recipe_section_plan(self, recipe: Optional[Recipe] = None):
        recipe_obj = self.recipe if recipe is None else recipe
        snapshot = getattr(recipe_obj, "section_plan", None)
        if isinstance(snapshot, SectionPlanSnapshot) and section_plan_is_compatible(recipe_obj, snapshot):
            recipe_obj.section_pos_z = list(snapshot.positions_z)
            recipe_obj.section_pos_ui = list(recipe_obj.section_pos_z)
            return section_plan_from_snapshot(snapshot)
        section_plan = self._compute_recipe_section_plan(recipe_obj)
        self._bind_section_plan_to_recipe(recipe_obj, section_plan)
        return section_plan

    def _build_recipe_section_plan(self, recipe: Optional[Recipe] = None):
        return self._ensure_recipe_section_plan(self.recipe if recipe is None else recipe)

    def _ask_section_plan_recompute_choice(self) -> str:
        result = {"choice": "cancel"}
        root = cast(tk.Tk, self)
        top = tk.Toplevel(root)
        top.title("截面位置计算")
        top.transient(root)
        try:
            top.grab_set()
        except Exception:
            pass
        frm = ttk.Frame(top, padding=12)
        frm.pack(fill="both", expand=True)
        ttk.Label(
            frm,
            text="当前配方存在示教截面。请选择如何处理这些示教位置。",
            wraplength=520,
            justify="left",
        ).pack(fill="x", pady=(0, 10))
        btn_row = ttk.Frame(frm)
        btn_row.pack(fill="x")

        def choose(value: str) -> None:
            result["choice"] = value
            try:
                top.destroy()
            except Exception:
                pass

        ttk.Button(btn_row, text="全部重新计算", command=lambda: choose("recompute")).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="保留示教位置", command=lambda: choose("preserve")).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="取消", command=lambda: choose("cancel")).pack(side="left")
        top.protocol("WM_DELETE_WINDOW", lambda: choose("cancel"))
        try:
            top.wait_window()
        except Exception:
            return "cancel"
        choice = str(result.get("choice", "cancel"))
        return choice if choice in {"recompute", "preserve", "cancel"} else "cancel"

    def _refresh_recipe_table(self):
        tree = self._recipe_ui_widget('recipe_tree')
        try:
            if tree is None:
                return
            tree.delete(*tree.get_children())
        except Exception:
            return

        try:
            section_plan = self._ensure_recipe_section_plan(self.recipe)
        except Exception:
            return

        for row in getattr(section_plan, 'sections', ()):
            tree.insert(
                "",
                "end",
                values=(
                    int(row.section_index) - 1,
                    f"{float(row.z_od_disp):.3f}",
                    f"{float(row.z_id_disp):.3f}",
                    f"{float(row.ax0_abs):.3f}",
                    f"{float(row.ax1_abs):.3f}",
                    f"{float(row.ax4_abs):.3f}",
                    str(row.source),
                ),
            )
        return

    def _get_selected_recipe_idx(self) -> Optional[int]:
        tree = self._recipe_ui_widget('recipe_tree')
        if tree is None:
            return None
        sel = tree.selection()
        if not sel:
            return None
        item = sel[0]
        vals = tree.item(item, "values")
        if not vals:
            return None
        try:
            return int(vals[0])
        except Exception:
            return None
