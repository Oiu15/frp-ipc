from __future__ import annotations

"""Contract test: RecipeController delegates every method to the underlying host."""

from typing import Any

from application.controllers.recipe_controller import RecipeController


class _FakeHost:
    """Host that records every method call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str) -> Any:
        def _record(*args: Any, **kwargs: Any) -> None:
            self.calls.append((name, args, kwargs))

        return _record


# Methods with non-zero-argument signatures — skipped in the zero-arg sweep
_PASSTHROUGH_NO_ARG = sorted(
    [
        "_len_pick_low_approach",
        "_on_recipe_enter",
        "_on_recipe_selected",
        "_on_teach_axes_selected",
        "_recipe_compute",
        "_recipe_delete_backend",
        "_recipe_save_backend",
        "_refresh_center_positions",
        "_refresh_length_info",
        "_refresh_recipe_table",
        "_refresh_teach_action_buttons",
        "_save_ax2_len_pos",
        "_save_ax2_rot_pos",
        "_teach_align_by_id",
        "_teach_align_by_od",
        "_teach_go_standby",
        "_teach_goto_end",
        "_teach_goto_start",
        "_teach_len_search_high_toggle",
        "_teach_len_search_low_toggle",
        "_teach_move_ax2_to_len_pos",
        "_teach_move_ax2_to_rot_pos",
        "_teach_move_relative",
        "_teach_move_to_selected",
        "_teach_save_current_to_selected",
        "_teach_save_standby",
        "_teach_save_start",
        "_teach_start_from_standby",
    ]
)


class TestRecipeControllerContract:
    def test_every_method_delegates_to_host(self) -> None:
        host = _FakeHost()
        ctrl = RecipeController(host)

        for method_name in _PASSTHROUGH_NO_ARG:
            method = getattr(ctrl, method_name)
            method()
            assert host.calls[-1] == (method_name, (), {}), f"{method_name} did not delegate"

        assert len(host.calls) == len(_PASSTHROUGH_NO_ARG)

    def test_methods_are_callable(self) -> None:
        host = _FakeHost()
        ctrl = RecipeController(host)

        for method_name in _PASSTHROUGH_NO_ARG + ["_kv_row"]:
            method = getattr(ctrl, method_name)
            assert callable(method), f"{method_name} is not callable"

    def test_kv_row_forwards_args(self) -> None:
        host = _FakeHost()
        ctrl = RecipeController(host)

        ctrl._kv_row("parent", "label", "var", 3)

        assert host.calls == [("_kv_row", ("parent", "label", "var", 3), {})]

    def test_on_recipe_selected_forwards_event(self) -> None:
        host = _FakeHost()
        ctrl = RecipeController(host)

        ctrl._on_recipe_selected("evt")

        assert host.calls == [("_on_recipe_selected", ("evt",), {})]
