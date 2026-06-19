from __future__ import annotations

from ui.presenters.recipe_presenter_deps import RecipePresenterDeps


def test_recipe_presenter_deps_keeps_field_identity() -> None:
    recipe = object()
    axis_cal = object()
    ui_state = object()
    registry = {"recipe_name_var": object()}
    calls: list[tuple] = []

    def get_recipe() -> object:
        return recipe

    def set_recipe(value: object) -> None:
        calls.append(("set_recipe", value))

    def log_trace(location_name: str, recipe_obj: object) -> None:
        calls.append(("log", location_name, recipe_obj))

    def refresh_length_info() -> None:
        calls.append(("refresh",))

    def set_legacy_z(value: float | None) -> None:
        calls.append(("legacy_z", value))

    def after_applied() -> None:
        calls.append(("after",))

    deps = RecipePresenterDeps(
        get_recipe=get_recipe,
        set_recipe=set_recipe,
        axis_cal=axis_cal,
        ui_state=ui_state,
        variable_registry=registry,
        log_ax3_speed_trace=log_trace,
        refresh_length_info=refresh_length_info,
        set_len_low_approach_legacy_z=set_legacy_z,
        after_recipe_data_applied=(after_applied,),
    )

    assert deps.get_recipe is get_recipe
    assert deps.set_recipe is set_recipe
    assert deps.axis_cal is axis_cal
    assert deps.ui_state is ui_state
    assert deps.variable_registry is registry
    assert deps.log_ax3_speed_trace is log_trace
    assert deps.refresh_length_info is refresh_length_info
    assert deps.set_len_low_approach_legacy_z is set_legacy_z
    assert deps.after_recipe_data_applied == (after_applied,)


def test_recipe_presenter_deps_defaults_are_narrow() -> None:
    deps = RecipePresenterDeps(
        get_recipe=lambda: object(),
        set_recipe=lambda _value: None,
        axis_cal=object(),
    )

    assert deps.ui_state is None
    assert deps.variable_registry == {}
    assert deps.log_ax3_speed_trace is None
    assert deps.refresh_length_info is None
    assert deps.set_len_low_approach_legacy_z is None
    assert deps.after_recipe_data_applied == ()
