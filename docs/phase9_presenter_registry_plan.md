# Phase 9 ScreenPresenter Registry Plan

Status: Phase 9.10 complete.

Phase 9.11 also removed the remaining runtime UI-state fallback in
`GaugeScreenPresenter`. Gauge state now uses explicit `get_var()` / `get_flag()`
calls, and its controller calls are explicit protocol methods.

## Current Responsibilities

`ScreenPresenter` is constructed in `application/controller_wiring.py` and
attached as `host._screen_presenter`. It currently owns:

- `_app`: the original host reference exposed by `host_app`.
- `_widgets`: a local name-to-widget registry.
- `_view_state`: a local name-to-value registry.
- `remember_widget()` / `widget()`: explicit widget registry writes and reads.
- `remember_view_state()` / `view_state()`: explicit view-state writes and reads.

Phase 9.10 removed `ScreenPresenter.__getattr__`, its four empty host fallback
allowlists, and the helper used only by that fallback. Registry access is now
method-based only.

## Production Usage Inventory

No screen builder receives the generic `_screen_presenter`. Migrated screens use
their explicit presenter or UI-state object.

The only remaining production reads of `_screen_presenter` are in:

- `AppHost._main_ui_widget()`: second-level fallback through `widget(name)`.
- `AppHost._main_view_state()`: second-level fallback through
  `view_state(name, default)`.

These calls use explicit registry methods, not `__getattr__`.

The main screen registers its widgets and view state into `MainUiState`, which
already implements the same four explicit registry methods. No production code
registers data in the generic `_screen_presenter`, so the AppHost fallback is
currently an empty compatibility path.

No production call site was found that reads a generic registry entry as
`presenter.some_widget` or `presenter.some_view_state`.

## Explicit Registry Target

The target contract is method-based:

1. Writers call `remember_widget(name, widget)` or
   `remember_view_state(name, value)` on the owning screen state.
2. Readers call `widget(name)` or `view_state(name, default)` explicitly.
3. Screen-specific state owns its registry. A generic registry is introduced
   only if multiple screens demonstrate a real shared ownership requirement.
4. No registry uses `__getattr__`, host passthrough, or prefix allowlists.

If a standalone shared object is still useful after removing the dead fallback,
name it `ScreenRegistry`; do not retain the presenter name for an object that
contains no presentation state or behavior.

## Phase 9.10 Outcome

The deletion preconditions were:

- No production screen receives `_screen_presenter`.
- No production code uses attribute-style registry reads.
- Host attribute and callable allowlists remain empty or are removed.
- Explicit `widget()` and `view_state()` behavior remains covered.
- Tests that intentionally exercise `presenter.result_tree` or
  `presenter.selected_row` are changed to assert explicit method access instead.

All conditions are satisfied. Attribute-style compatibility tests now verify
that such access raises `AttributeError`, while explicit registry API tests
retain their original behavior coverage.

Removing the `ScreenPresenter` object itself is a separate change. Before that:

1. Remove `_screen_presenter` fallback branches from `_main_ui_widget()` and
   `_main_view_state()` after confirming `main_ui` remains the sole registry
   owner.
2. Remove its construction and host attachment from `wire_screen_controllers()`.
3. Remove the Host UI type declaration and obsolete registry compatibility
   tests.

## Recommended Migration Path

1. Complete in Phase 9.10: delete `ScreenPresenter.__getattr__`; retain explicit
   registry methods and update only compatibility tests.
2. Remove the dead AppHost `_screen_presenter` fallback branches.
3. Delete the unused generic presenter, or rename/extract it as `ScreenRegistry`
   only if a concrete shared-registry use appears.

## Risks And Rollback

- Risk: an untested plugin or external caller may use attribute-style registry
  access. Roll back by using explicit `widget()` / `view_state()` calls, not by
  restoring host passthrough.
- Risk: main-screen registry wiring regresses. Keep `MainUiState` registry tests
  and AppHost main-view behavior tests as the primary protection.
- Risk: a future screen starts using the generic object. Guard tests must reject
  `_screen_presenter` screen wiring and `getattr(presenter, ...)` additions.
