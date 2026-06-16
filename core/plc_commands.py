"""Pure-data PLC command dataclasses — no IO dependency.

Moved from ``drivers/plc_client.py`` so that host mixins and other
application-layer code can import them without depending on the IO layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union


@dataclass
class CmdWriteRegs:
    d_addr: int
    values: List[int]


@dataclass
class CmdWriteCoil:
    """Write a single Modbus coil (0/1)."""

    coil_addr: int
    value: int


@dataclass
class CmdReadRegs:
    """Read holding registers on demand.

    Used for one-shot reads that are not part of the regular polling loop,
    e.g. reading the axis calibration block stored in PLC HD area.
    """

    d_addr: int
    count: int
    tag: str = ""


@dataclass
class CmdSetPollProfile:
    """Change background polling profile.

    - profile="normal": poll all axes + CL + keytest(X/Y)
    - profile="sampling": poll only selected axes (default AX3), disable CL and Y background polling,
      but keep X background polling (E-Stop/footswitch need immediate response).
      (AutoFlow uses sync reads for angle/CL during sampling).
    """
    profile: str = "normal"


@dataclass
class CmdSetCmdMask:
    axis: int
    set_mask: int = 0
    clr_mask: int = 0


@dataclass
class CmdPulseCmdMask:
    axis: int
    pulse_mask: int
    pulse_ms: int = 120


WorkerCmd = Union[CmdWriteRegs, CmdWriteCoil, CmdReadRegs, CmdSetPollProfile, CmdSetCmdMask, CmdPulseCmdMask]


__all__ = [
    "CmdPulseCmdMask",
    "CmdReadRegs",
    "CmdSetCmdMask",
    "CmdSetPollProfile",
    "CmdWriteCoil",
    "CmdWriteRegs",
    "WorkerCmd",
]
