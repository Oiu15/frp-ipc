# Phase 8 Fallback Audit

Scope: audit only. This document records the remaining dynamic fallback
surfaces after RecipePresenter, key-test, axis calibration, and the
standalone validation tab, and gauge tab were moved to explicit
dependencies.

## Fallback Definitions

| Class | File | Target | Guard | Risk |
| --- | --- | --- | --- | --- |
| `ScreenPresenter.__getattr__` | `application/adapters/device_gateway.py` | `host_app` attributes and selected callables | exact/prefix allowlists for state and calls | Medium. It still exposes host-backed view state to legacy screens, but callables are guarded. |
| `ScreenController.__getattr__` | `application/adapters/device_gateway.py` | callable attributes on `host_app` | exact/prefix callable allowlists | High. Button commands can still reach AppHost methods by naming convention. |
| `ScreenUiContext.__getattr__` | `application/adapters/device_gateway.py` | non-callable attributes on `host_app` | exact/prefix attribute allowlists | Medium-high. It can still expose broad UI/host state, including prefix groups. |

## Current Usage Points

| Area | Uses dynamic fallback? | Evidence | Notes |
| --- | --- | --- | --- |
| `recipe_screen.py` | No dynamic fallback after Phase 8.2 guard | uses `RecipeScreenPresenter` fields and `RecipeController` methods directly | Keep this path locked. |
| `main_screen.py` | Yes | uses `ScreenPresenter` state vars and `ScreenController` commands such as `start_measurement`, `stop_measurement`, `export_history_results` | Broadest runtime surface; main workflow buttons still proxy through `ScreenController`. |
| `key_test_screen.py` | No after Phase 8.3 | uses explicit `KeyTestUiState` and `KeyTestController` | Migrated out of generic `ScreenPresenter` / `ScreenController` / `ScreenUiContext` fallback. |
| `axis_cal_screen.py` | No after Phase 8.6 | uses explicit `AxisCalUiState` and `AxisCalController` | Migrated out of generic `ScreenPresenter` / `ScreenController` / `ScreenUiContext` fallback. |
| `validation_screen.py` | No after Phase 8.7 | uses explicit `ValidationUiState` and `ValidationController` | Standalone validation tab is migrated. |
| `gauge_screen.py` | No generic fallback after Phase 8.8 | uses explicit `GaugeUiState`, `GaugeController`, and `GaugeScreenPresenter(GaugeUiState, GaugeController)` | Device, calibration, and validation commands are explicit controller methods. |
| `AxisScreenPresenter` | Own fallback remains | `ui/presenters/axis_presenter.py` | Guarded by its own view/controller boundary; not migrated in this phase. |
| `GaugeScreenPresenter` | Own fallback remains | `ui/presenters/gauge_presenter.py` | Still has UI-state fallback through its view. |

## Already Explicit

| Path | Explicit boundary |
| --- | --- |
| Recipe tab presenter | `RecipePresenterDeps` |
| Recipe tab controller | `RecipeController` |
| Recipe form mapping | `RecipeFormViewPort` |
| Key-test tab state | `KeyTestUiState` |
| Key-test tab controller | `KeyTestController` |
| Axis calibration tab state | `AxisCalUiState` |
| Axis calibration tab controller | `AxisCalController` |
| Validation tab state | `ValidationUiState` |
| Validation tab controller | `ValidationController` |
| Gauge tab state | `GaugeUiState` |
| Gauge tab controller | `GaugeController` |
| Axis tab presenter | `AxisScreenPresenter(view, controller)` |
| Gauge tab presenter | `GaugeScreenPresenter(view, controller)` |

## Risk Ranking

1. `ScreenController.__getattr__`: highest coupling because it routes commands into `AppHost`.
2. `ScreenUiContext.__getattr__`: state exposure by allowlist and prefix, still host-shaped.
3. `ScreenPresenter.__getattr__`: mostly view state and legacy refresh/list calls.
4. Presenter-local fallbacks in `AxisScreenPresenter` / `GaugeScreenPresenter`: narrower than the generic screen proxies, but still worth later cleanup.

## Recommended Migration Order

1. `main_screen`: formal measurement, export, result table, and workflow commands.

Next recommended target: `main_screen`.
