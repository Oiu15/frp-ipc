from __future__ import annotations

import frp_workflow.autoflow_executor as compat_executor
import frp_workflow.executor as executor_package
from frp_workflow.executor import _executor_helpers, _executor_sampling
from tests.fakes import NoOpEventSink


class _App:
    pass


class _Device:
    pass


def teardown_function() -> None:
    compat_executor.SPEEDTEST_DISABLE_ID_MODBUS = False  # pyright: ignore[reportAttributeAccessIssue]  -- module __getattr__


def test_autoflow_remains_daemon_thread_after_split() -> None:
    flow = compat_executor.AutoFlow(_App(), device=_Device(), event_sink=NoOpEventSink())  # pyright: ignore[reportArgumentType]

    assert flow.daemon is True


def test_speedtest_flag_assignment_reaches_sampling_module() -> None:
    assert _executor_helpers.SPEEDTEST_DISABLE_ID_MODBUS is False
    assert _executor_sampling._executor_helpers.SPEEDTEST_DISABLE_ID_MODBUS is False

    compat_executor.SPEEDTEST_DISABLE_ID_MODBUS = True  # pyright: ignore[reportAttributeAccessIssue]  -- module __getattr__

    assert compat_executor.SPEEDTEST_DISABLE_ID_MODBUS is True  # pyright: ignore[reportAttributeAccessIssue]  -- module __getattr__
    assert executor_package.SPEEDTEST_DISABLE_ID_MODBUS is True  # pyright: ignore[reportAttributeAccessIssue]  -- module __getattr__
    assert _executor_helpers.SPEEDTEST_DISABLE_ID_MODBUS is True
    assert _executor_sampling._executor_helpers.SPEEDTEST_DISABLE_ID_MODBUS is True

    executor_package.SPEEDTEST_DISABLE_ID_MODBUS = False  # pyright: ignore[reportAttributeAccessIssue]  -- module __getattr__

    assert compat_executor.SPEEDTEST_DISABLE_ID_MODBUS is False  # pyright: ignore[reportAttributeAccessIssue]  -- module __getattr__
    assert _executor_helpers.SPEEDTEST_DISABLE_ID_MODBUS is False
    assert _executor_sampling._executor_helpers.SPEEDTEST_DISABLE_ID_MODBUS is False
