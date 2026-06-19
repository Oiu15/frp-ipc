from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(slots=True)
class RecipePresenterDeps:
    get_recipe: Callable[[], Any]
    set_recipe: Callable[[Any], None]
    axis_cal: Any
    ui_state: Any | None = None
    variable_registry: Mapping[str, Any] = field(default_factory=dict)
    log_ax3_speed_trace: Callable[[str, Any], None] | None = None
    refresh_length_info: Callable[[], None] | None = None
    set_len_low_approach_legacy_z: Callable[[float | None], None] | None = None
    after_recipe_data_applied: tuple[Callable[[], None], ...] = ()


__all__ = ["RecipePresenterDeps"]
