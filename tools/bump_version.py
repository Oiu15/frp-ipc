from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "_version.py"
VERSION_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-(?:beta|rc)\.([1-9]\d*))?$")


def _replace_version(text: str, version: str) -> str:
    updated, count = re.subn(r'^VERSION = "[^"]+"$', f'VERSION = "{version}"', text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"expected exactly one VERSION assignment in {VERSION_FILE}")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Update the FRP-IPC software version source.")
    parser.add_argument("version", help="Target version, for example 0.6.3-beta.2, 0.6.3-rc.1, or 0.6.3")
    args = parser.parse_args()

    version = args.version.strip()
    if VERSION_RE.fullmatch(version) is None:
        raise SystemExit("version must match MAJOR.MINOR.PATCH, MAJOR.MINOR.PATCH-beta.N, or MAJOR.MINOR.PATCH-rc.N")

    text = VERSION_FILE.read_text(encoding="utf-8")
    VERSION_FILE.write_text(_replace_version(text, version), encoding="utf-8", newline="\n")
    print(f"updated {VERSION_FILE} to v{version}")


if __name__ == "__main__":
    main()
