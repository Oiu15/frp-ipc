# Phase 9 Fallback Deletion Plan

## Current Fallback State

Phase 9.2 moved the main screen to explicit `MainController` and `MainUiState`. The primary screen tabs now use explicit controller/state objects for their screen code:

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

## Production Usage Points

| Object | Production use | Deletion implication |
| --- | --- | --- |
| `_screen_controller` | Still constructed for legacy inventory; no longer passed to `build_axis_screen(...)` after Phase 9.4 | No longer the direct axis presenter blocker. |
| `_screen_presenter` | Still attached for compatibility; AppHost main widget/view state falls back to it after `main_ui` | Can likely shrink after confirming no runtime path writes main widgets/view state to it. |
| `_screen_ui_context` | Passed to `build_axis_screen(...)` and `build_recipe_screen(...)` as compatibility arg | Screen source does not appear to use `getattr(ui, ...)`, but deletion should wait until wiring no longer passes it to migrated screens. |

`axis_screen.py` itself calls explicit `AxisScreenPresenter` methods. After Phase 9.4,
`AxisScreenPresenter` uses `AxisController` and no longer calls
`getattr(self.controller, ...)`. Axis action names are still string keys from the
screen, but they are resolved by an explicit `AxisController.dispatch_axis_action()`
mapping rather than generic host fallback.

## Allowlist Inventory

### ScreenController

Exact allowlist:

- Main screen legacy entries: `start_measurement`, `stop_measurement`, `clear_measurement_results`, `export_history_results`, `open_serial_template_settings`, `handle_main_result_selection`, `refresh_main_summary_panel`
- Gauge/calibration legacy entries: `apply_plc_connection`, `connect_gauge`, `disconnect_gauge`, `request_gauge_once`, `set_gauge_request_command`, `toggle_sim_gauge`, `learn_odcal_defect_a`, `learn_odcal_defect_b`, `apply_od_b`, `apply_id_calibration`, `verify_id_calibration`, `open_validation_screen`
- Validation legacy entries: `list_validation_section_choices`, `start_validation_run`, `stop_validation_run`

Prefix allowlist:

- Axis likely still active: `_refresh`, `axis_cal_`, `clear_`, `compute_`, `export_`, `handle_`, `open_`, `start_`, `stop_`
- Recipe/teach legacy candidates: `_kv_row`, `_on_recipe`, `_on_teach`, `_recipe`, `_save`, `_teach`
- Key test legacy candidate: `write_keytest_`

Likely immediately removable after a focused test pass:

- Main exact entries, because `main_screen` now uses `MainController`.
- Key test prefix `write_keytest_`, because `key_test_screen` uses `KeyTestController`.
- Validation exact entries, because validation/gauge validation entry points use explicit controllers.
- Gauge exact entries, because gauge screen uses `GaugeController`.

Must retain or replace first:

- Any recipe/teach private prefixes until a focused search confirms they are test-only or unused.
- Any axis prefixes still covered only by compatibility tests; production screen routing no longer requires generic controller fallback after Phase 9.4.

### ScreenPresenter

Exact attr allowlist mostly mirrors old main screen state (`pipe_sn_var`, `auto_*`, OD/ID summary vars, `cov_var`, and related state). After Phase 9.2 these are explicit fields on `MainUiState`.

Prefix allowlist:

- `axis_cal_`
- `keytest_`
- `validation_`

Call allowlist:

- `list_validation_section_choices`

Call prefix allowlist:

- `_list`
- `_refresh`

Likely immediately removable after a focused test pass:

- Main exact state entries, once `_main_ui_widget/_main_view_state` no longer need compatibility fallback to `_screen_presenter`.
- `keytest_`, `axis_cal_`, and `validation_` prefixes if no explicit presenter still uses `ScreenPresenter`.

Must retain or replace first:

- Compatibility widget/view-state fallback in AppHost should be removed only after confirming `main_ui` is always initialized before main view refresh calls.

### ScreenUiContext

Exact allowlist:

- `app`
- `axis_cal`
- `axis_idx`
- `recipe`
- `root`
- `style`
- `ui`

Prefix allowlist:

- `axis_`
- `keytest_`
- `validation_`

Likely immediately removable after a focused test pass:

- `keytest_`, `validation_`, and `axis_cal` screen state exposure, because migrated screens use explicit state objects.

Must retain or replace first:

- Any `axis_screen` compatibility dependency on `axis_idx` or `axis_` state.
- `recipe_screen` wiring still passes `self._screen_ui_context`; remove that arg from wiring only after confirming the screen does not depend on it.

## Deletion Stages

### 1. Shrink Allowlist

Goal: remove allowlist entries for already migrated screens while keeping fallback class shape.

Suggested order:

1. Remove main exact entries from `ScreenController` and main exact state entries from `ScreenPresenter`.
2. Remove key test and validation/gauge allowlist entries.
3. Leave axis-related entries until `AxisScreenPresenter` receives an explicit axis controller.

Risk:

- Hidden startup path may still call `_main_view_state` before `main_ui` is available.
- Existing tests may still assert old broad allowlists.

Rollback:

- Restore removed allowlist entries only for the failing path and add an inventory test explaining the remaining user.

### 2. Keep Fallback But Make It Test-Oriented

Goal: after allowlist shrink, fallback classes remain for compatibility tests but no screen creation path depends on them for commands/state.

Required conditions:

- Done in Phase 9.4: `build_axis_screen(...)` no longer receives `_screen_controller` for action dispatch.
- Done in Phase 9.4: `AxisScreenPresenter` no longer calls `getattr(self.controller, action_name, None)` against a generic controller.
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

Proceed to Phase 9.5: shrink allowlists for migrated screens and remove remaining unused generic wiring arguments. Do not directly delete `__getattr__` until shrink tests pass.
