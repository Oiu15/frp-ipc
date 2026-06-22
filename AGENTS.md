# AGENTS.md

## Repository overview

This repository is a Tkinter-based IPC application for FRP tube dimension inspection.
Main concerns include:

- PLC communication
- Gauge / sensor communication
- Auto measurement workflow
- UI screens and widgets
- Recipe management
- Calibration-related logic
- Export and run records

## Primary entrypoints and reading order

When analyzing architecture, start from:

1. `app.py`
2. `PROJECT_OVERVIEW.md`
3. `services/`
4. `ui/`
5. `drivers/`
6. `core/`
7. `config/`

Do not assume the documentation fully matches the code.
Always verify real call relationships from the code.

## Repository access contract

Treat `c:/Users/11982/Projects/frp-ipc` as the preferred repository root.
If it is unavailable, use `d:/Users/11982/projects/frp-ipc` as the fallback repository root.
Do not rely on shell current directory or relative paths for repository-wide analysis.
Prefer absolute paths for repository scans and file reads whenever possible.

Before starting a repository-wide architecture scan, verify the root by checking:

- `app.py`
- `PROJECT_OVERVIEW.md`
- `services/`
- `ui/`
- `drivers/`
- `core/`
- `config/`

If repository-root directory listing and repository-file reads disagree, treat that as an environment problem rather than a codebase fact.
If both preferred and fallback root verification fail, stop the analysis and report an environment/path issue.
Do not continue with fallback guesses such as scanning unrelated roots, using filename-only reads, or inferring a sandbox mirror.

## Directories to ignore by default

Unless the task explicitly asks for them, skip:

- `build/`
- `dist/`
- `demo/`
- `.venv/`
- `__pycache__/`
- temporary packaging outputs
- generated files

## Test environment

Use the project virtual environment for pytest:

- Python: `C:\Users\11982\Projects\frp-ipc\.venv\Scripts\python.exe`
- Pytest command: `C:\Users\11982\Projects\frp-ipc\.venv\Scripts\python.exe -m pytest`

## Project-specific architecture cautions

`app.py` was historically a heavy "God object", but it has since been reduced
to a thin entry point (`App` factory + `main()` that calls
`ApplicationShell().run(App)`). The former host responsibilities now live in
`application/shell.py` (queues, worker lifecycle, dependency assembly) and
`application/app_host.py` (Tk root, screen mounting, `ui_q` event consumption).
See `PROJECT_OVERVIEW.md` for the current layout.

Do not assume that refactor is complete in every corner — always verify from
the code. When performing architecture review, explicitly check that these
couplings stay resolved (and watch for regressions):

- whether UI variables directly drive workflow logic
- whether workflow logic directly updates UI state
- whether export logic is split across UI and service layers
- whether AutoFlow runs through `AutoFlowOrchestrator` rather than depending on `app.py` / `AppHost` as a runtime host
- whether nominal layering exists only at file level but not object-boundary level

## Architecture scan order

For repository-wide architecture analysis, use this order:

1. Verify the repository root.
2. Build a symbol index from the host layer — `application/shell.py` (queues, worker/thread creation, dependency assembly) and `application/app_host.py` (imports, class definitions, queue/event usage, AutoFlow assembly). `app.py` itself is now a thin handoff.
3. Use targeted searches to locate runtime call chains.
4. Read only the relevant line ranges in `application/app_host.py` (the host that carries the former `app.py` weight).
5. Then inspect `services/`, `ui/`, `drivers/`, `core/`, and `config/`.
6. Use `PROJECT_OVERVIEW.md` only to compare documentation against code, not as proof of implementation.

Do not start by reporting file length, broad directory dumps, or large unscoped file reads.

## Analysis style

For architecture analysis:

- prefer actual runtime call chains over idealized layering
- identify hidden coupling points
- point out where documentation and implementation diverge
- distinguish relatively clean drivers/data objects from app-layer glue code
- focus on incremental refactor feasibility before proposing rewrite

## Encoding and text handling

Assume UTF-8 first for documentation and source files.
If text is garbled, retry once with explicit UTF-8 or byte-based reading.
If the text is still garbled, mark the document as unreliable and continue with code evidence.
Do not spend multiple rounds probing encodings unless the task is specifically about encoding.

## Environment anomaly handling

Report shell or path anomalies once, briefly.
After the first anomaly report, either continue with a validated absolute-path method or stop and report that environment validation failed.
Do not repeatedly narrate directory-resolution instability.

## Refactor guidance preferences

This repository should prefer pragmatic, staged refactoring.
Avoid recommending a full rewrite unless the current boundary violations make migration risk lower than direct evolution.

When proposing refactors, separate:

- structural changes required first
- second-stage module extraction
- future support for multi-mode operation
- areas that should remain untouched for now

## Output expectations for architecture audit tasks

When asked for architecture audit / refactor analysis:

- do not modify code
- do not generate large diffs
- give file-level and function-level evidence where possible
- explain thread model, queue model, state flow, and ownership boundaries
- highlight first-batch extraction candidates with lowest risk
- separate environment/tooling facts, verified code facts, and inferences
- do not present environment behavior as architecture evidence
