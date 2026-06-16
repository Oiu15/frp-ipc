from __future__ import annotations
# pyright: reportUnsupportedDunderAll=false

"""AutoFlow executor — backward-compatible re-export.

All implementation has moved to ``frp_workflow.executor.*``.
Importing from this module continues to work unchanged.
"""

import sys
from types import ModuleType

from frp_workflow.executor._executor_core import AutoFlow
from frp_workflow.executor import _executor_helpers

log = _executor_helpers.log
log_exc = _executor_helpers.log_exc
perf_logger = _executor_helpers.perf_logger


class _AutoFlowExecutorModule(ModuleType):
    def __getattr__(self, name: str):
        if name == "SPEEDTEST_DISABLE_ID_MODBUS":
            return _executor_helpers.SPEEDTEST_DISABLE_ID_MODBUS
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        if name == "SPEEDTEST_DISABLE_ID_MODBUS":
            _executor_helpers.SPEEDTEST_DISABLE_ID_MODBUS = bool(value)
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _AutoFlowExecutorModule

__all__ = ["AutoFlow", "SPEEDTEST_DISABLE_ID_MODBUS", "log", "log_exc"]
