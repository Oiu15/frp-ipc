# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project summary

FRP-IPC is a Python 3.12 Tkinter desktop application for automated pipe measurement (FRP 管检测) on factory floors. It communicates with a XINJE PLC over Modbus TCP, reads a Keyence CL-3000 laser gauge over serial, and orchestrates 5 servo axes (AX0–AX4) for production measurement, calibration, and validation workflows.

## Commands

```bash
# Development setup
python -m pip install -r requirements-dev.txt

# Lint (ruff)
python -m ruff check .

# Type check (pyright)
python -m pyright

# Compile-check all Python sources
python -m compileall _version.py app.py application config controllers core domain drivers events frp_workflow machine modes repositories services ui utils

# Run all tests
python -m pytest -q

# Run a single test file
python -m pytest tests/test_mode_machine.py

# Run a specific test
python -m pytest tests/test_mode_machine.py -k "test_enter_production"

# Version validation (CI gate)
python tools/check_version.py

# Bump version
python tools/bump_version.py 0.6.3-beta.3

# Build Windows distributable
python -m PyInstaller --noconfirm frp-ipc.spec
```

## Architecture

### Layered design (dependency direction: outer → inner)

```
ui/              Tkinter screens, widgets, and presenters
controllers/     UI intent entrypoints for production and calibration
application/     AppHost (Tk root), shell, application adapters, and compatibility boundaries
events/          Typed UI events, dispatchers, worker adapters, and queue pump
frp_workflow/    Production workflow orchestration and AutoFlow executor
modes/           Mode state machines (production, calibration, validation) + ModeMachine
services/        Calibration/results/export services and legacy AutoFlow import compatibility
repositories/    File-based persistence (JSON) — calibration, validation, recipes
drivers/         IO threads — PlcWorker (Modbus TCP), GaugeWorker (serial)
machine/         DeviceGateway and validation action protocols
domain/          Shared state/models/protocols and pure computation — sampling, summaries, calibration, postcalc
core/            Pure data models (AxisComm, Recipe, MeasureRow, GaugeSample) + Modbus codec
config/          Hardware addresses, PLC memory layout, app config schema
utils/           Logger, performance aggregator
```

### Threading model

Four threads communicate via two `queue.Queue` instances:

| Thread | Role | Queues |
|---|---|---|
| **Main (Tk)** | UI rendering, callbacks, state refresh | reads `ui_q`, writes `cmd_q` |
| **PlcWorker** | Modbus TCP polling at ~70 ms interval | writes `ui_q`, reads `cmd_q` |
| **GaugeWorker** | Serial gauge read loop | writes `ui_q` |
| **AutoFlow** | Background measurement state machine | writes `ui_q`, reads `cmd_q` |

The `UiEventDispatcher` bridges the worker threads to UI: workers push raw `(event_name, payload)` tuples onto `ui_q`; the main thread pumps them through `UiEventDispatcher.dispatch()`. Events are defined as strongly-typed dataclasses in `events/types.py` (subclasses of `UiEventBase`), with a registry in `UI_EVENT_TYPES`.

### Dependency assembly

`application/shell.py::ApplicationShell` owns lifecycle: it creates queues, starts workers, builds `AppDependencies`, and hands them to the Tk `AppHost` factory. Tests can substitute dependencies by calling `AppHost(dependencies=..., shell=...)` directly.

### Mode machine

`modes/mode_machine.py::ModeMachine` is the top-level mode orchestrator. It holds three mode objects (production, calibration, validation) and enforces mutual exclusion — only one mode is active at a time. Mode transitions are tracked in `RuntimeState.mode_kind` and `RuntimeState.mode_state`.

### Coordinate system

- **Z axis**: positive is **downwards** (display coordinate, `z_disp`).
- **Servo feedback** (abs): positive is upwards on AX0/AX1/AX4.
- **Axis calibration** (`core/models.py::AxisCal`): maps between abs and Z via `sign` (default -1), per-axis offsets, and a `z_pos` UI shift.
- ID axis is composed: AX1 + AX4 with split logic in `AxisCal.od_z_disp_to_targets()`.
- Keepout zone protection is **PLC-owned** (IPC no longer clamps motion).

### Measurement flow

1. Recipe defines section positions, tolerances, sampling parameters (`core/models.py::Recipe`).
2. `AutoFlowOrchestrator` and `frp_workflow/autoflow_executor.py` drive the measurement sequence: move axes, spin AX3, sample gauge, compute results.
3. Per-section results stored as `MeasureRow` list; post-processing computes straightness, concentricity, eccentricity via `domain/summaries.py`.
4. Results exported to CSV by repository layer (`repositories/`).

### PLC memory layout

Defined in `config/addresses.py`: 5 axes × 100 words each (D100–D599), plus a system comm block (D50–D99). Each axis block follows the `AXIS_Ctrl` struct with cmd/seq/sts/err words and LREAL (FP64) setpoints. FP64 uses little-endian 4-word encoding via `core/modbus_codec.py`.

Keyence CL-3000 measurement data arrives via Ethernet/IP mapped into PLC D2000–D2135, with per-output scaling (OUT1/2/5: 0.0001 mm/LSB; OUT3/4: 0.001 mm/LSB).

## Testing

- Tests live in `tests/`; run with `python -m pytest` from repo root.
- **Fakes** in `tests/fakes.py`:
  - `StrictDeviceGateway` — mock device gateway that only permits explicitly allowed calls (fails loudly on unexpected calls). Use `.allow("method_name", return_value=...)` to set up.
  - `SequentialRunRepository` — generates sequential `RunIdentity` objects for workflow tests.
  - `RecordingValidationRepository` — records exported contexts and paths without writing files.
- Tests that touch `AppHost` typically construct it with fake dependencies rather than starting real PLC/gauge workers.
- No mocking framework is used — tests use these hand-rolled fakes.

## Key conventions

- All Python files start with `from __future__ import annotations` for deferred evaluation.
- Dataclasses use `slots=True` where possible.
- Public APIs exposed via `__all__`.
- Logging uses the `"frp.…”` logger hierarchy (`"frp.app.mode"`, `"frp.modbus"`, etc.).
- Chinese strings in UI and comments are expected (the application is for a Chinese-speaking factory).
- Version is the single source of truth in `_version.py`. The CI gate (`tools/check_version.py`) validates `VERSION`, `VERSION_TAG`, and `SOFTWARE_VERSION` consistency plus git tag alignment.
