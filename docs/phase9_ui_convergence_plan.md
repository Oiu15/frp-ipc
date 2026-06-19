# Phase 9 UI Convergence Plan

## Scope

Phase 9 starts after the Phase 8 screen migrations. Recipe, key test, axis calibration, validation, and gauge screens are wired to explicit controller/state objects. The remaining hybrid UI path is the main measurement screen:

```text
build_main_screen(
    tab_main,
    presenter=self._screen_presenter,
    controller=self._screen_controller,
    ui=self._screen_ui_context,
)
```

This document is a planning artifact only. It does not authorize UI layout changes, workflow rewrites, export schema changes, or removal of the generic fallback classes before main screen migration is complete.

## Main Screen Dependency Map

| Category | Current dependency | Evidence | Phase 9 target |
| --- | --- | --- | --- |
| UI state dependency | `presenter.pipe_sn_var`, `meas_seq_var`, `meas_start_var`, `meas_elapsed_var`, `auto_progress_var`, `auto_done_var`, `auto_state_var`, `ui_meas_mode_var`, `auto_msg_var`, summary/result vars, and `cov_var` | `ui/screens/main_screen.py` reads these through `presenter.*`; `application/host/ui.py` injects `self._screen_presenter` | `MainUiState` or equivalent explicit presenter deps with named fields |
| Widget/view state dependency | `presenter.remember_widget(...)`, `presenter.remember_view_state(...)` | Result labels and `result_tree` are registered through the generic presenter | Explicit widget registry/view-state methods on `MainUiState`, preserving current host compatibility |
| Command routing dependency | `controller.start_measurement`, `stop_measurement`, `clear_measurement_results`, `export_history_results`, `open_serial_template_settings`, `handle_main_result_selection`, `refresh_main_summary_panel` | `main_screen.py` binds buttons and tree selection through `controller.*`; `application/host/ui.py` injects `self._screen_controller` | `MainController` with exactly these methods delegating to narrow host ports |
| Workflow orchestration dependency | `start_measurement`, `stop_measurement`, result selection, summary refresh | Commands currently reach AppHost workflow/main-view methods through `ScreenController.__getattr__` | Keep behavior, but make the dependency explicit behind `MainController` |
| Export/settings dependency | `export_history_results`, `open_serial_template_settings` | Button commands in main control panel | Explicit controller methods; no export schema or dialog behavior change |
| Gauge/validation crossover | No direct `gauge_*` or `validation_*` access in `main_screen.py` | Main screen consumes measurement summary state only | No crossover controller dependency should be introduced |

## Fallback Shrink Strategy

1. Add `MainController` in `application/controllers/` with the seven main-screen commands:
   `start_measurement`, `stop_measurement`, `clear_measurement_results`, `export_history_results`, `open_serial_template_settings`, `handle_main_result_selection`, and `refresh_main_summary_panel`.
2. Add `MainUiState` or equivalent deps object with all state variables and widget/view-state methods currently read from `ScreenPresenter`.
3. Wire `build_main_screen(...)` with explicit objects:

   ```text
   presenter=self.main_ui,
   controller=self.main_controller,
   ui=self.main_ui
   ```

4. Keep `main_screen.py` layout and command bindings unchanged; only change object types and construction.
5. Extend fallback guard tests so main screen cannot return to `ScreenController` or `ScreenPresenter`.
6. After main migration, shrink generic allowlists in `ScreenController`, `ScreenPresenter`, and `ScreenUiContext`.
7. Delete or hard-disable `ScreenController.__getattr__` only after inventory confirms no screen creation path still requires generic command routing.

## Conditions For Deleting Generic Fallback

`ScreenController.__getattr__` is not safe to delete yet. The deletion gate is:

- `main_screen` is wired to `MainController` and explicit `MainUiState`.
- All migrated screen guard tests pass.
- No `build_*_screen` call receives `self._screen_controller` for command routing.
- The remaining `ScreenPresenter` / `ScreenUiContext` allowlists are either empty or documented as compatibility-only.
- Startup smoke and UI inventory tests confirm no hidden generic command path remains.

Until then, the generic fallback should be treated as legacy compatibility only.

## Risk Control

- Do not change UI layout, button text, or event binding semantics.
- Keep AppHost method names and signatures stable during the first main migration.
- Preserve `result_tree` registration and `handle_main_result_selection` binding.
- Do not combine main workflow commands with export/settings commands in business logic.
- Add `MainController` contract tests before rewiring.
- Add `MainUiState` identity tests for all state vars and widget registry methods.

## Recommended Next Step

Proceed to Phase 9.2: implement `MainController` and `MainUiState`, then rewire only `main_screen`. Do not delete generic `__getattr__` methods in the same phase.
