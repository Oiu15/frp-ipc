from __future__ import annotations

"""Explicit controller for ``recipe_screen.py`` (Phase 3).

Replaces the ``ScreenController`` dynamic ``__getattr__`` proxy for the
recipe tab.  Every method here corresponds to a ``controller.xxx`` call
in ``build_recipe_screen`` and delegates directly to the same-named
method on the underlying AppHost.
"""

from typing import Any


class RecipeController:
    """Controller surface for the recipe/teach tab.

    Each public method is a thin forwarding wrapper around the AppHost
    method of the same name.  No new business logic is introduced here.
    """

    def __init__(self, host: Any) -> None:
        self._host = host

    # -- axis / motion --------------------------------------------------

    def _save_ax2_len_pos(self) -> None:
        self._host._save_ax2_len_pos()

    def _save_ax2_rot_pos(self) -> None:
        self._host._save_ax2_rot_pos()

    def _teach_move_ax2_to_len_pos(self) -> None:
        self._host._teach_move_ax2_to_len_pos()

    def _teach_move_ax2_to_rot_pos(self) -> None:
        self._host._teach_move_ax2_to_rot_pos()

    # -- kv_row (UI helper) ---------------------------------------------

    def _kv_row(self, parent: Any, label: str, var: Any, row: int) -> None:
        self._host._kv_row(parent, label, var, row)

    # -- recipe selection -----------------------------------------------

    def _on_recipe_selected(self, event: Any = None) -> None:
        if event is not None:
            self._host._on_recipe_selected(event)
        else:
            self._host._on_recipe_selected()

    def _on_recipe_enter(self, event: Any = None) -> None:
        if event is not None:
            self._host._on_recipe_enter(event)
        else:
            self._host._on_recipe_enter()

    def _on_teach_axes_selected(self, event: Any = None) -> None:
        if event is not None:
            self._host._on_teach_axes_selected(event)
        else:
            self._host._on_teach_axes_selected()

    # -- teach actions --------------------------------------------------

    def _teach_move_relative(self) -> None:
        self._host._teach_move_relative()

    def _teach_align_by_od(self) -> None:
        self._host._teach_align_by_od()

    def _teach_align_by_id(self) -> None:
        self._host._teach_align_by_id()

    def _teach_move_to_selected(self) -> None:
        self._host._teach_move_to_selected()

    def _teach_save_current_to_selected(self) -> None:
        self._host._teach_save_current_to_selected()

    def _teach_save_start(self) -> None:
        self._host._teach_save_start()

    def _teach_goto_start(self) -> None:
        self._host._teach_goto_start()

    def _teach_goto_end(self) -> None:
        self._host._teach_goto_end()

    def _teach_save_standby(self) -> None:
        self._host._teach_save_standby()

    def _teach_go_standby(self) -> None:
        self._host._teach_go_standby()

    def _teach_start_from_standby(self) -> None:
        self._host._teach_start_from_standby()

    # -- recipe persistence ---------------------------------------------

    def _recipe_compute(self) -> None:
        self._host._recipe_compute()

    def _recipe_save_backend(self) -> None:
        self._host._recipe_save_backend()

    def _recipe_delete_backend(self) -> None:
        self._host._recipe_delete_backend()

    # -- refresh helpers ------------------------------------------------

    def _refresh_teach_action_buttons(self) -> None:
        self._host._refresh_teach_action_buttons()

    def _refresh_recipe_table(self) -> None:
        self._host._refresh_recipe_table()

    def _refresh_center_positions(self) -> None:
        self._host._refresh_center_positions()

    def _refresh_length_info(self) -> None:
        self._host._refresh_length_info()

    # -- length measurement helpers -------------------------------------

    def _len_pick_low_approach(self) -> None:
        self._host._len_pick_low_approach()

    def _teach_len_search_low_toggle(self) -> None:
        self._host._teach_len_search_low_toggle()

    def _teach_len_search_high_toggle(self) -> None:
        self._host._teach_len_search_high_toggle()


__all__ = ["RecipeController"]
