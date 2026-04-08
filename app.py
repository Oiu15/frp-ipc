from __future__ import annotations

"""Thin entrypoint, shell handoff, and compatibility exports.

The legacy Tk host implementation now lives in
``application.legacy_app_host``. This module intentionally stays small so the
repository entrypoint no longer also carries the full runtime implementation.
"""

from application.legacy_app_host import LegacyAppHost, SOFTWARE_VERSION
from application.shell import ApplicationShell

App = LegacyAppHost


def main() -> None:
    ApplicationShell().run(App)


__all__ = ["App", "ApplicationShell", "LegacyAppHost", "SOFTWARE_VERSION", "main"]


if __name__ == "__main__":
    main()
