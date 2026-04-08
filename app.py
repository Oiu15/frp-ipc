from __future__ import annotations

"""Thin entrypoint, shell handoff, and compatibility exports.

The Tk app host implementation now lives in
``application.app_host``. Runtime startup stays here, while the
app host remains an implementation detail behind the factory-style ``App``
compat shim.
"""

from typing import TYPE_CHECKING

from application.app_host import AppHost, SOFTWARE_VERSION
from application.shell import ApplicationShell

if TYPE_CHECKING:  # pragma: no cover
    from application.app_host import AppHost as App
else:

    def App(*, dependencies=None, shell=None):
        """Thin compatibility factory for the Tk app host."""

        return AppHost(dependencies=dependencies, shell=shell)


def main() -> None:
    ApplicationShell().run(App)


__all__ = ["App", "ApplicationShell", "AppHost", "SOFTWARE_VERSION", "main"]


if __name__ == "__main__":
    main()
