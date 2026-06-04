"""Backward-compatible access to ``frp_workflow.autoflow_executor``."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from frp_workflow import autoflow_executor as _executor
from frp_workflow.autoflow_executor import *  # noqa: F401, F403

_SHARED_MUTABLE_NAMES = frozenset({"SPEEDTEST_DISABLE_ID_MODBUS"})


class _AutoflowCompatModule(ModuleType):
    """Keep mutable compatibility settings synchronized with the executor."""

    def __getattribute__(self, name: str) -> Any:
        if name in _SHARED_MUTABLE_NAMES:
            return getattr(_executor, name)
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _SHARED_MUTABLE_NAMES:
            setattr(_executor, name, value)
        super().__setattr__(name, value)


setattr(sys.modules[__name__], "__class__", _AutoflowCompatModule)
