from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Never

ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-(beta|rc)\.([1-9]\d*))?$")


def _fail(message: str) -> Never:
    print(f"version check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def _current_commit_tags() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "tag", "--points-at", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail(f"could not inspect current git tags: {exc}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> None:
    sys.path.insert(0, str(ROOT))
    try:
        from _version import SOFTWARE_VERSION, VERSION, VERSION_TAG
    except Exception as exc:  # pragma: no cover - import failure should be reported directly
        _fail(f"could not import _version: {exc}")

    match = VERSION_RE.fullmatch(VERSION)
    if match is None:
        _fail("VERSION must match MAJOR.MINOR.PATCH, MAJOR.MINOR.PATCH-beta.N, or MAJOR.MINOR.PATCH-rc.N")

    expected_tag = f"v{VERSION}"
    if VERSION_TAG != expected_tag:
        _fail(f"VERSION_TAG must be {expected_tag!r}, got {VERSION_TAG!r}")

    if SOFTWARE_VERSION != VERSION_TAG:
        _fail(f"SOFTWARE_VERSION must equal VERSION_TAG {VERSION_TAG!r}, got {SOFTWARE_VERSION!r}")

    prerelease_kind = match.group(4)
    tags = [tag for tag in _current_commit_tags() if tag.startswith("v")]
    if tags and tags != [VERSION_TAG]:
        _fail(f"current commit tag(s) {tags!r} must exactly match VERSION_TAG {VERSION_TAG!r}")

    if tags and prerelease_kind is None:
        prerelease_tags = [tag for tag in tags if "-beta." in tag or "-rc." in tag]
        if prerelease_tags:
            _fail(f"release VERSION {VERSION!r} must not be built from prerelease tag(s) {prerelease_tags!r}")

    print(f"version check ok: {VERSION_TAG}")


if __name__ == "__main__":
    main()
