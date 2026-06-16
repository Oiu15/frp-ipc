from __future__ import annotations

"""Typed UI event handlers for AppHost orchestration boundaries."""

from application.handlers.device import GaugeErrEventHandler, PlcErrEventHandler, PlcOkEventHandler
from application.handlers.measurement import (
    AutoCoverageEventHandler,
    AutoLenEventHandler,
    AutoPostcalcEventHandler,
    AutoProgressEventHandler,
    AutoRowEventHandler,
    AutoStateEventHandler,
)

__all__ = [
    "AutoCoverageEventHandler",
    "AutoLenEventHandler",
    "AutoPostcalcEventHandler",
    "AutoProgressEventHandler",
    "AutoRowEventHandler",
    "AutoStateEventHandler",
    "GaugeErrEventHandler",
    "PlcErrEventHandler",
    "PlcOkEventHandler",
]
