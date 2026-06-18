from __future__ import annotations

"""Pack the project source tree into a zip archive for AI/agent consumption.

Usage:
  python tools/pack_source.py                  # → frp-ipc-source.zip
  python tools/pack_source.py my-project.zip   # → my-project.zip
  python tools/pack_source.py --list           # show what would be included
  python tools/pack_source.py -o out.zip       # → out.zip

The resulting archive preserves the project tree structure, includes all
tracked source files (Python, JSON, Markdown, YAML, TOML, etc.), and
excludes generated artefacts (__pycache__, .pytest_cache, .git, etc.).
"""

import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Never

ROOT = Path(__file__).resolve().parents[1]

DESCRIPTION = """Source archive for the FRP-IPC project.

What's included:
  application/   AppHost, Shell, adapters, handlers, host/mixins, form mapper
  config/        Hardware addresses, PLC memory layout, app config schema
  core/          Pure data models (AxisComm, Recipe, MeasureRow, …), Modbus codec
  domain/        Pure computation (calibration, sampling, summaries, state)
  drivers/       PLC / Gauge IO threads
  docs/          Engineering baseline, refactoring plan
  events/        Typed UI event system (dispatcher, adapters, types)
  frp_workflow/  AutoFlow orchestrator + executor
  machine/       Hardware port protocols, DeviceGateway
  modes/         Mode state machine (production / calibration / validation)
  repositories/  JSON file persistence
  services/      Calibration, results, export services
  tests/         Full test suite including fixtures
  tools/         Dev tooling (check_version, bump_version, pack_source, …)
  ui/            UI layer (screens, presenters, state)
  utils/         Logging, performance aggregator

  Root files: _version.py, app.py, CLAUDE.md, AGENTS.md, PROJECT_OVERVIEW.md,
              pyproject.toml, requirements*.txt, frp-ipc.spec, CI YAML
"""

SOURCE_DIRS = [
    "application",
    "config",
    "core",
    "domain",
    "drivers",
    "docs",
    "events",
    "frp_workflow",
    "machine",
    "modes",
    "repositories",
    "services",
    "tests",
    "tools",
    "ui",
    "utils",
]

ROOT_FILES = [
    "_version.py",
    "app.py",
    "CLAUDE.md",
    "AGENTS.md",
    "PROJECT_OVERVIEW.md",
    "pyproject.toml",
    "pyrightconfig.json",
    "requirements.txt",
    "requirements-dev.txt",
    "frp-ipc.spec",
]

SOURCE_EXTENSIONS = frozenset(
    {".py", ".json", ".md", ".yaml", ".yml", ".toml", ".txt", ".spec", ".cfg"}
)

EXCLUDE_DIRS = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".import_linter",
        ".claude",
        ".git",
        ".venv",
        "venv",
        "dist",
        "build",
        ".egg-info",
        "node_modules",
        ".tmp",
    }
)

DEFAULT_OUTPUT = "frp-ipc-source.zip"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fail(message: str) -> Never:
    print(f"pack_source: {message}", file=sys.stderr)
    raise SystemExit(1)


def _collect_files() -> list[str]:
    """Walk the project tree and return sorted relative paths to include."""
    paths: list[str] = []

    for name in SOURCE_DIRS:
        directory = ROOT / name
        if not directory.is_dir():
            print(f"  (skip missing) {name}/")
            continue
        for fp in sorted(directory.rglob("*")):
            if not fp.is_file():
                continue
            if fp.suffix not in SOURCE_EXTENSIONS:
                continue
            parts = set(fp.relative_to(ROOT).parts)
            if EXCLUDE_DIRS & parts:
                continue
            paths.append(str(fp.relative_to(ROOT).as_posix()))

    for name in ROOT_FILES:
        fp = ROOT / name
        if fp.is_file():
            paths.append(str(fp.relative_to(ROOT).as_posix()))

    # Root-level CI YAML files
    for fp in sorted(ROOT.glob("*.yml")):
        paths.append(str(fp.relative_to(ROOT).as_posix()))
    for fp in sorted(ROOT.glob("*.yaml")):
        paths.append(str(fp.relative_to(ROOT).as_posix()))

    return sorted(set(paths))


def _add_files_to_zip(zf: zipfile.ZipFile, paths: Iterable[str]) -> None:
    for rel in paths:
        abs_path = ROOT / rel
        try:
            zf.write(abs_path, rel)
        except OSError as exc:
            print(f"  WARN  skip {rel}: {exc}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        raise SystemExit(0)

    # --list: print what would be packed and exit
    if "--list" in sys.argv:
        paths = _collect_files()
        for p in paths:
            print(p)
        print(f"\n{len(paths)} files would be included.")
        raise SystemExit(0)

    # resolve output path
    output: str | None = None
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "-o" in sys.argv:
        try:
            idx = sys.argv.index("-o")
            output = sys.argv[idx + 1]
        except (ValueError, IndexError):
            _fail("missing argument after -o")
    elif args:
        output = args[0]

    dest = ROOT / (output or DEFAULT_OUTPUT)

    # try git-archive first (clean, fewer surprises)
    try:
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "archive",
                "--format=zip",
                f"--output={dest}",
                "HEAD",
            ],
            cwd=str(ROOT),
            check=True,
        )
        print(f"Packed {dest.name}  (via git archive)")
        size_kb = dest.stat().st_size / 1024
        print(f"  {size_kb:.0f} KB  ({dest})")
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("git archive not available, falling back to manual collection...")

    # fallback: manual collection
    paths = _collect_files()
    if not paths:
        _fail("no files collected — check SOURCE_DIRS / ROOT_FILES")

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        _add_files_to_zip(zf, paths)

    size_kb = dest.stat().st_size / 1024
    print(f"Packed {dest.name}  (manual collection)")
    print(f"  {size_kb:.0f} KB  {len(paths)} files")


if __name__ == "__main__":
    main()
