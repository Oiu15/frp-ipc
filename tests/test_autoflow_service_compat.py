from __future__ import annotations

import services.autoflow_service as compat
from frp_workflow import autoflow_executor as executor


def test_speedtest_flag_stays_shared_across_compat_module() -> None:
    original = executor.SPEEDTEST_DISABLE_ID_MODBUS
    try:
        setattr(compat, "SPEEDTEST_DISABLE_ID_MODBUS", not original)
        assert executor.SPEEDTEST_DISABLE_ID_MODBUS is not original

        executor.SPEEDTEST_DISABLE_ID_MODBUS = original
        assert getattr(compat, "SPEEDTEST_DISABLE_ID_MODBUS") is original
    finally:
        executor.SPEEDTEST_DISABLE_ID_MODBUS = original
        setattr(compat, "SPEEDTEST_DISABLE_ID_MODBUS", original)
