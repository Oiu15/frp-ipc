# Phase 9 UI Convergence Plan

## Scope

Phase 9 starts after the Phase 8 screen migrations. Recipe, key test, axis calibration, validation, and gauge screens are wired to explicit controller/state objects. Phase 9.2 migrates the main measurement screen to the same explicit pattern:

```text
build_main_screen(
    tab_main,
    presenter=self.main_ui,
    controller=self.main_controller,
    ui=self.main_ui,
)
```

This document is a planning artifact only. It does not authorize UI layout changes, workflow rewrites, export schema changes, or removal of the generic fallback classes before main screen migration is complete.

## Main Screen Dependency Map

| Category | Current dependency | Evidence | Phase 9 target |
| --- | --- | --- | --- |
| UI state dependency | `presenter.pipe_sn_var`, `meas_seq_var`, `meas_start_var`, `meas_elapsed_var`, `auto_progress_var`, `auto_done_var`, `auto_state_var`, `ui_meas_mode_var`, `auto_msg_var`, summary/result vars, and `cov_var` | `ui/screens/main_screen.py` reads these through `presenter.*`; Phase 9.2 injects `self.main_ui` | `MainUiState` with named fields |
| Widget/view state dependency | `presenter.remember_widget(...)`, `presenter.remember_view_state(...)` | Result labels and `result_tree` are registered through `MainUiState` | Explicit widget registry/view-state methods on `MainUiState`, with AppHost lookup updated to prefer `main_ui` |
| Command routing dependency | `controller.start_measurement`, `stop_measurement`, `clear_measurement_results`, `export_history_results`, `open_serial_template_settings`, `handle_main_result_selection`, `refresh_main_summary_panel` | `main_screen.py` binds buttons and tree selection through `controller.*`; Phase 9.2 injects `self.main_controller` | `MainController` with exactly these methods delegating to narrow host ports |
| Workflow orchestration dependency | `start_measurement`, `stop_measurement`, result selection, summary refresh | Commands reach existing AppHost workflow/main-view methods through `MainController` | Behavior preserved; dependency is explicit |
| Export/settings dependency | `export_history_results`, `open_serial_template_settings` | Button commands in main control panel | Explicit controller methods; no export schema or dialog behavior change |
| Gauge/validation crossover | No direct `gauge_*` or `validation_*` access in `main_screen.py` | Main screen consumes measurement summary state only | No crossover controller dependency should be introduced |

## Fallback Shrink Strategy

1. Done in Phase 9.2: add `MainController` in `application/controllers/` with the seven main-screen commands:
   `start_measurement`, `stop_measurement`, `clear_measurement_results`, `export_history_results`, `open_serial_template_settings`, `handle_main_result_selection`, and `refresh_main_summary_panel`.
2. Done in Phase 9.2: add `MainUiState` with all state variables and widget/view-state methods previously read from `ScreenPresenter`.
3. Done in Phase 9.2: wire `build_main_screen(...)` with explicit objects:

   ```text
   presenter=self.main_ui,
   controller=self.main_controller,
   ui=self.main_ui
   ```

4. Keep `main_screen.py` layout and command bindings unchanged; only change object types and construction.
5. Done in Phase 9.2: extend fallback guard tests so main screen cannot return to `ScreenController` or `ScreenPresenter`.
6. Next: audit whether generic allowlists in `ScreenController`, `ScreenPresenter`, and `ScreenUiContext` can shrink.
7. Delete or hard-disable `ScreenController.__getattr__` only after inventory confirms no screen creation path still requires generic command routing.

## Conditions For Deleting Generic Fallback

`ScreenController.__getattr__` is closer to deletion after Phase 9.2, but should not be deleted without a final shrink/delete audit. The deletion gate is:

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

Proceed to Phase 9.3: fallback shrink/delete audit. Do not delete generic `__getattr__` methods until the audit confirms every remaining `_screen_controller` / `_screen_presenter` use is compatibility-only.
