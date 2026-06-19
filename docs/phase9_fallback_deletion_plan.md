# Phase 9 Fallback Deletion Plan

## Current Fallback State

Phase 9.5 shrank the generic host allowlists after the primary screen tabs moved to explicit controller/state objects:

| Screen | Current screen-level status |
| --- | --- |
| `recipe_screen.py` | Explicit `RecipeScreenPresenter` + `RecipeController`; generic `ui` argument is still passed but the screen source does not use dynamic `getattr(ui, ...)`. |
| `key_test_screen.py` | Explicit `KeyTestUiState` + `KeyTestController`. |
| `axis_cal_screen.py` | Explicit `AxisCalUiState` + `AxisCalController`. |
| `validation_screen.py` | Explicit `ValidationUiState` + `ValidationController`. |
| `gauge_screen.py` | Explicit `GaugeUiState`, `GaugeController`, and `GaugeScreenPresenter`. |
| `main_screen.py` | Explicit `MainUiState` + `MainController`. |

Generic fallback classes still exist in `application/adapters/device_gateway.py`:

- `ScreenController.__getattr__`
- `ScreenPresenter.__getattr__`
- `ScreenUiContext.__getattr__`

They are still constructed in `application/controller_wiring.py` and attached to AppHost:

- `host._screen_controller = ScreenController(host)`
- `host._screen_presenter = ScreenPresenter(host)`
- `host._screen_ui_context = ScreenUiContext(host)`

Their host allowlists are now empty. The classes remain as legacy inventory and still raise `AttributeError` for undeclared host passthrough.

## Production Usage Points

| Object | Production use | Deletion implication |
| --- | --- | --- |
| `_screen_controller` | Still constructed for legacy inventory; no longer passed to `build_axis_screen(...)` after Phase 9.4 | No longer the direct axis presenter blocker. |
| `_screen_presenter` | Still attached for compatibility; AppHost main widget/view state can fall back to its local widget/view-state registry after `main_ui` | Host attribute/call fallback is no longer allowed; deletion should wait until compatibility registry fallback is removed. |
| `_screen_ui_context` | Passed to `build_axis_screen(...)` and `build_recipe_screen(...)` as compatibility arg | Its host allowlist is empty; deletion should wait until wiring no longer passes it to migrated screens. |

`axis_screen.py` itself calls explicit `AxisScreenPresenter` methods. After Phase 9.4,
`AxisScreenPresenter` uses `AxisController` and no longer calls
`getattr(self.controller, ...)`. Axis action names are still string keys from the
screen, but they are resolved by an explicit `AxisController.dispatch_axis_action()`
mapping rather than generic host fallback.

## Allowlist Inventory

### ScreenController

Exact allowlist: empty.

Prefix allowlist: empty.

Removed in Phase 9.5:

- Main exact commands: `start_measurement`, `stop_measurement`, `clear_measurement_results`, `export_history_results`, `open_serial_template_settings`, `handle_main_result_selection`, `refresh_main_summary_panel`
- Gauge/calibration exact commands: `apply_plc_connection`, `connect_gauge`, `disconnect_gauge`, `request_gauge_once`, `set_gauge_request_command`, `toggle_sim_gauge`, `learn_odcal_defect_a`, `learn_odcal_defect_b`, `apply_od_b`, `apply_id_calibration`, `verify_id_calibration`, `open_validation_screen`
- Validation exact commands: `list_validation_section_choices`, `start_validation_run`, `stop_validation_run`
- Legacy prefixes: `_kv_row`, `_on_recipe`, `_on_teach`, `_recipe`, `_refresh`, `_save`, `_teach`, `axis_cal_`, `clear_`, `compute_`, `export_`, `handle_`, `open_`, `refresh_`, `start_`, `stop_`, `write_keytest_`

### ScreenPresenter

Exact attr allowlist: empty.

Attr prefix allowlist: empty.

Call allowlist: empty.

Call prefix allowlist: empty.

`ScreenPresenter` still owns local widget and view-state registries. It no longer proxies host attributes or host methods.

### ScreenUiContext

Exact allowlist: empty.

Prefix allowlist: empty.

`ScreenUiContext` remains constructed and passed to a small compatibility surface, but it no longer exposes host state.

## Deletion Stages

### 1. Shrink Allowlist

Status: complete in Phase 9.5.

Rollback:

- Restore only the specific allowlist item needed by a proven production path, then add an inventory test for that path.

### 2. Keep Fallback But Make It Test-Oriented

Goal: after allowlist shrink, fallback classes remain for compatibility tests but no screen creation path depends on them for commands/state.

Required conditions:

- Done in Phase 9.4: `build_axis_screen(...)` no longer receives `_screen_controller` for action dispatch.
- Done in Phase 9.4: `AxisScreenPresenter` no longer calls `getattr(self.controller, action_name, None)` against a generic controller.
- Done in Phase 9.5: generic fallback host allowlists are empty.
- Remaining: `build_recipe_screen(...)` and `build_axis_screen(...)` no longer receive `_screen_ui_context` unless needed by an explicit typed adapter.

Risk:

- Axis debug panel is action-heavy and may expose latent command names currently covered by broad prefixes.

Rollback:

- Reintroduce a narrow `AxisController` method rather than restoring broad generic fallback.

### 3. Delete `__getattr__`

Goal: remove dynamic routing entirely.

Deletion conditions:

- No production wiring passes `_screen_controller`, `_screen_presenter`, or `_screen_ui_context` to screen builders for command/state access.
- Guard tests prove all migrated screen files have no dynamic fallback tokens.
- Contract tests exist for each explicit controller/state object replacing generic fallback.
- Startup smoke passes.

Risk:

- Tests that intentionally validate legacy fallback behavior will need to be retired or converted into explicit controller/state tests.

Rollback:

- Prefer restoring a narrow explicit adapter over restoring generic `__getattr__`.

## Recommended Next Action

Proceed to Phase 9.6: delete-fallback candidate audit. First remove unused `_screen_ui_context` wiring arguments and confirm `_screen_presenter` local registry compatibility is no longer needed. Do not directly delete `__getattr__` until that audit passes.
