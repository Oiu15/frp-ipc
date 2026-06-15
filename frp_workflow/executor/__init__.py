"""AutoFlow executor sub-package — mixin classes split from autoflow_executor."""
# pyright: reportUnsupportedDunderAll=false

import sys
from types import ModuleType

from frp_workflow.executor._executor_core import AutoFlow, ExecutorCoreMixin
from frp_workflow.executor import _executor_helpers
from frp_workflow.executor._executor_clamps import ExecutorClampsMixin
from frp_workflow.executor._executor_motion import ExecutorMotionMixin
from frp_workflow.executor._executor_length import ExecutorLengthMixin
from frp_workflow.executor._executor_sampling import ExecutorSamplingMixin, SamplingResult
from frp_workflow.executor._executor_fitting import ExecutorFittingMixin

log = _executor_helpers.log
log_exc = _executor_helpers.log_exc
perf_logger = _executor_helpers.perf_logger


class _ExecutorPackageModule(ModuleType):
    def __getattr__(self, name: str):
        if name == "SPEEDTEST_DISABLE_ID_MODBUS":
            return _executor_helpers.SPEEDTEST_DISABLE_ID_MODBUS
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        if name == "SPEEDTEST_DISABLE_ID_MODBUS":
            _executor_helpers.SPEEDTEST_DISABLE_ID_MODBUS = bool(value)
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _ExecutorPackageModule

__all__ = [
    "AutoFlow",
    "ExecutorClampsMixin",
    "ExecutorCoreMixin",
    "ExecutorFittingMixin",
    "ExecutorLengthMixin",
    "ExecutorMotionMixin",
    "ExecutorSamplingMixin",
    "SamplingResult",
    "SPEEDTEST_DISABLE_ID_MODBUS",
    "log",
    "log_exc",
]
