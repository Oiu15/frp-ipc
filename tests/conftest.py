"""Shared pytest fixtures for the FRP-IPC test suite.

Shared fake / stub *classes* live in ``tests.fakes`` alongside the
existing ``StrictDeviceGateway`` and repository fakes.
"""

from __future__ import annotations

import sys
import types


# ---------------------------------------------------------------------------
# pymodbus import shim (module-level — MUST run before any test file)
# ---------------------------------------------------------------------------
# Pytest imports conftest.py *before* test modules in the same directory.
# Module-level code here therefore runs in time to satisfy top-level
# ``from pymodbus.client import ModbusTcpClient`` triggered by
# ``from drivers.plc_client import ...`` in test files.
#
# The ``if … not in sys.modules`` guard leaves a real pymodbus untouched
# when it is already installed (e.g. on a developer machine).

if "pymodbus" not in sys.modules:
    _pymodbus = types.ModuleType("pymodbus")
    _pymodbus_client = types.ModuleType("pymodbus.client")

    class _FakeModbusTcpClient:
        pass

    setattr(_pymodbus_client, "ModbusTcpClient", _FakeModbusTcpClient)
    setattr(_pymodbus, "client", _pymodbus_client)
    sys.modules["pymodbus"] = _pymodbus
    sys.modules["pymodbus.client"] = _pymodbus_client
