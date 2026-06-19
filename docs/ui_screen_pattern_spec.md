# UI Screen Pattern Spec

Scope: architecture pattern only. This document defines the target UI
screen model used after Phase 8.5. It does not describe a completed
migration state for every screen.

## Target Model

```text
Screen =
    Controller: explicit commands only
    Presenter: pure UI state and view helpers only
    UIState/Deps: explicit injected state
    No fallback layer
```

The target screen receives explicit objects from controller wiring. A
screen can read its presenter/UI state and can call its controller methods,
but it must not discover behavior by asking a generic host proxy for an
attribute name.

## Controller Specification

Controllers expose command methods that a screen can bind to buttons,
menus, keyboard handlers, or widget events.

Rules:

1. A controller method must be explicitly declared on the concrete
   controller class.
2. A controller may delegate to a narrow host port or service, but it must
   not expose the full host object as an open-ended surface.
3. A controller must not use `__getattr__` to route command names.
4. A controller must not use string prefixes such as `start_`, `stop_`,
   `axis_cal_`, or `compute_` as a dispatch mechanism.
5. A controller should contain input normalization only when it is UI
   adapter behavior; business rules belong in services or workflow code.

Recommended shape:

```python
class AxisCalController:
    def __init__(self, host: AxisCalCommandPort) -> None:
        self._host = host

    def axis_cal_read(self) -> None:
        self._host.axis_cal_read()
```

## Presenter Specification

Presenters expose UI state and view helpers. They should not be command
routers.

Rules:

1. A presenter owns or receives explicit Tk variables, display state, and
   widget registry helpers.
2. A presenter must not keep a generic `host_app` passthrough.
3. A presenter must not use `__getattr__` to fetch host fields.
4. A presenter may call local view helper methods, but it should not call
   PLC/gauge/workflow services directly.
5. If a presenter needs a callback, inject it as a named `Callable`, not as
   a host object.

## UIState / Deps Specification

UIState or Deps dataclasses group explicit screen state.

Rules:

1. Field names must be explicit and screen-owned.
2. Do not store `AppHost`, generic host objects, widgets unrelated to the
   screen, repositories, services, or drivers.
3. Use `Any` only for Tk variables/widgets whose concrete type would create
   unnecessary local typing churn.
4. Prefer `slots=True` dataclasses for simple state containers.

Recommended shape:

```python
@dataclass(slots=True)
class AxisCalUiState:
    axis_cal_vars: Any
    axis_cal_field_status_vars: Any
    axis_cal_status_vars: Any
```

## Forbidden Patterns

New or migrated screens must not introduce:

- `__getattr__` routing
- `host_app` passthrough
- generic `_host` passthrough
- `getattr(controller, ...)`
- `getattr(presenter, ...)`
- `getattr(ui, ...)`
- command routing by string prefix
- presenter access to PLC/gauge drivers
- screen direct access to `AppHost`

Existing generic fallback in `ScreenPresenter`, `ScreenController`, and
`ScreenUiContext` is legacy-only. It remains until the remaining hybrid
screens are migrated.

## Controller Wiring Pattern

Controller wiring should construct explicit screen dependencies and assign
them to the host only as compatibility fields for the UI build phase.

Recommended shape:

```python
axis_cal_controller = AxisCalController(host)
host.axis_cal_controller = axis_cal_controller
host.axis_cal_ui_state = AxisCalUiState(
    axis_cal_vars=host.axis_cal_vars,
    axis_cal_field_status_vars=host.axis_cal_field_status_vars,
    axis_cal_status_vars=host.axis_cal_status_vars,
)
```

The UI build path should then pass explicit objects:

```python
build_axis_cal_screen(
    tab_axis_cal,
    presenter=host.axis_cal_ui_state,
    controller=host.axis_cal_controller,
    ui=host.axis_cal_ui_state,
)
```

## Hybrid Screen Alignment Analysis

| Screen | Current violation | Violation type | Hidden coupling | Migration difficulty |
| --- | --- | --- | --- | --- |
| `axis_cal_screen.py` | Aligned after Phase 8.6 with `AxisCalUiState` and `AxisCalController`. | None for generic proxies | No current generic fallback in screen wiring. | Done |
| `validation_screen.py` | Aligned after Phase 8.7 with `ValidationUiState` and `ValidationController`. | None for generic proxies | No current generic fallback in standalone validation tab wiring. | Done |
| `gauge_screen.py` | Uses explicit gauge presenter for much of the UI, but commands span PLC connection, gauge connection, OD/ID calibration, validation, and clear/compute/export actions. | Mixed | Device/control command routing, calibration state sharing, implicit command naming. | High |
| `main_screen.py` | Uses generic presenter for run summary/result variables and generic controller for measurement, export, result selection, and serial-template commands. | Mixed | Formal measurement host routing, result widget/view state coupling, implicit command naming. | High |

## Migration Order

1. `gauge_screen.py`
2. `main_screen.py`

Reasoning:

1. `gauge_screen.py` is larger and combines device connection,
   calibration, validation entry points, and presenter-local helpers. It is
   the next remaining hybrid screen after the standalone validation tab was
   made explicit.
2. `main_screen.py` should be last because it is the user-facing formal
   measurement surface and includes workflow commands, export commands,
   result selection, and result table view state.

## Deletion Policy

Do not delete `ScreenController.__getattr__`,
`ScreenPresenter.__getattr__`, or `ScreenUiContext.__getattr__` in Phase
8.5.

Deletion can begin only after the hybrid screens above are no longer wired
through generic fallback objects and their allowlist entries have been
shrunk or removed.
