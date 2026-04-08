from __future__ import annotations

"""Thin entrypoint, shell handoff, and compatibility exports.

The legacy Tk host implementation now lives in
``application.legacy_app_host``. Runtime startup stays here, while the legacy
runtime host remains an implementation detail behind the factory-style ``App``
compat shim.
"""

from typing import TYPE_CHECKING

from application.legacy_app_host import LegacyAppHost, SOFTWARE_VERSION
from application.shell import ApplicationShell

if TYPE_CHECKING:  # pragma: no cover
    from application.legacy_app_host import LegacyAppHost as App
else:

    def App(*, dependencies=None, shell=None):
        """Thin compatibility factory for the legacy Tk runtime host."""

        return LegacyAppHost(dependencies=dependencies, shell=shell)


def main() -> None:
    ApplicationShell().run(App)


__all__ = ["App", "ApplicationShell", "LegacyAppHost", "SOFTWARE_VERSION", "main"]


if __name__ == "__main__":
    main()
