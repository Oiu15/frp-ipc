from __future__ import annotations

"""自动测量流程辅助模块。

助手函数和日志工具。
"""

import logging

logger = logging.getLogger("frp.autoflow")
algo_logger = logging.getLogger("frp.algo")
data_logger = logging.getLogger("frp.data")
perf_logger = logging.getLogger("frp.autoflow.perf")


def _fmt_log_value(value) -> str:
    try:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)
    except Exception:
        return "<fmt_err>"


def log(event: str, **fields) -> None:
    """Module-local compatible log helper routed to standard logging."""
    try:
        event_s = str(event)
        if fields:
            parts = [f"{k}={_fmt_log_value(v)}" for k, v in fields.items()]
            msg = f"{event_s} " + " ".join(parts)
        else:
            msg = event_s
        evu = event_s.upper()
        if ("FIT" in evu) or evu.startswith("SAMPLE_") or ("ROUND" in evu):
            algo_logger.debug(msg)
        elif evu.startswith("SECTION_") or evu.startswith("AUTO_ROW"):
            data_logger.info(msg)
        else:
            logger.info(msg)
    except Exception:
        return


def log_exc(event: str, exc: BaseException) -> None:
    try:
        logger.exception("%s | exc=%s", str(event), str(exc))
    except Exception:
        return


# -------------------------
# Speedtest knobs
# -------------------------
# When True, AutoFlow will skip reading ID (CL Modbus) during sampling.
# This is useful to benchmark OD sampling rate and isolate comm bottlenecks.
SPEEDTEST_DISABLE_ID_MODBUS: bool = False

__all__ = ["SPEEDTEST_DISABLE_ID_MODBUS", "log", "log_exc"]
