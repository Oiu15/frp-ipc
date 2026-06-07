"""Tests for utils/logger.py — public API surface.

The logger module has filesystem-backed rotating handlers; these tests
verify the module is importable and the re-exported helpers are callable
without triggering filesystem side-effects.
"""

from __future__ import annotations

from utils.logger import log, log_exc


class TestLogHelpers:
    """log / log_exc — re-exported from the module's singleton logger."""

    def test_log_is_callable(self) -> None:
        assert callable(log)

    def test_log_exc_is_callable(self) -> None:
        assert callable(log_exc)
