from __future__ import annotations

"""Application-layer port Protocols for AppHost.

These Protocols define explicit boundaries between AppHost mixins and the
concrete AppHost class.  Each port is a thin contract — AppHost already
has the required attributes/methods; the Protocols make the dependency
explicit so mixin code can be type-checked and tested independently.
"""

from typing import Any, Protocol

from core.models import AxisCal, AxisComm, Recipe


# ---------------------------------------------------------------------------
# MotionPort — axis motion & I/O
# ---------------------------------------------------------------------------


class MotionPort(Protocol):
    """Axis motion and command port."""

    axis_cal: AxisCal

    def get_axis_copy(self, axis: int) -> AxisComm: ...
    def movea_abs(
        self, axis: int, pos_abs: float, *, context: str = "MoveA"
    ) -> None: ...
    def apply_soft_limits_abs(
        self,
        axis: int,
        target_abs: float,
        *,
        strict: bool = False,
        context: str = "",
    ) -> float: ...
    def _write_axis_params(self, axis: int) -> None: ...
    def set_cmd_bits(
        self, axis: int, set_mask: int = 0, clr_mask: int = 0
    ) -> None: ...


# ---------------------------------------------------------------------------
# OperatorPort — user interaction (messagebox wrappers)
# ---------------------------------------------------------------------------


class OperatorPort(Protocol):
    """Wrappers around tkinter.messagebox so mixins don’t import tkinter directly."""

    def show_error(self, title: str, message: str) -> None: ...
    def show_info(self, title: str, message: str) -> None: ...
    def show_warning(self, title: str, message: str) -> None: ...
    def ask_ok_cancel(self, title: str, message: str) -> bool: ...


# ---------------------------------------------------------------------------
# RecipePort — recipe access
# ---------------------------------------------------------------------------


class RecipePort(Protocol):
    """Recipe read/write and UI-widget access port."""

    recipe: Recipe

    def _recipe_ui_widget(self, name: str) -> Any: ...
    def _recipe_apply_from_ui(self) -> Recipe: ...
    def _get_selected_recipe_idx(self) -> int | None: ...
    def _ensure_recipe_section_plan(
        self, recipe: Recipe | None = None
    ) -> Any: ...
    def _save_taught_section_to_recipe(
        self, recipe: Recipe, recipe_index: int, z_od_disp: float
    ) -> None: ...
    def _refresh_recipe_table(self) -> None: ...
    def _refresh_length_info(self) -> None: ...


# ---------------------------------------------------------------------------
# RuntimeStatePort — runtime / scheduling
# ---------------------------------------------------------------------------


class RuntimeStatePort(Protocol):
    """Runtime scheduling and state snapshot port."""

    def after(self, ms: int, func: Any | None = None, *args: Any) -> Any: ...


# ---------------------------------------------------------------------------
# EventPort — event publishing
# ---------------------------------------------------------------------------


class EventPort(Protocol):
    """Lightweight event publishing port (placeholder for future expansion)."""

    pass


# ---------------------------------------------------------------------------
# Composite host types used by mixin type annotations
# ---------------------------------------------------------------------------


class TeachHost(MotionPort, OperatorPort, RecipePort, RuntimeStatePort, Protocol):
    """Composite port required by HostTeachMixin.

    AppHost satisfies this implicitly — the Protocol is used for
    ``self: TeachHost`` annotations on mixin helper methods so pyright
    can verify the dependency surface without importing AppHost.
    """

    pass


__all__ = [
    "EventPort",
    "MotionPort",
    "OperatorPort",
    "RecipePort",
    "RuntimeStatePort",
    "TeachHost",
]
