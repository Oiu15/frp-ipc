# -*- coding: utf-8 -*-
"""System communication block (SysCommD) definitions.

This file is intended to be shared by:
- IPC Python app (Modbus/TCP or other register-mapped protocol)
- PLC address/offset documentation

Design goals:
- SysCommD is independent from per-axis AxisCommD.
- One-shot command semantics via SEQ/ACK (recommended).
- SysCommD placed at end of 500-word COMM region to avoid shifting existing axis maps.

Default placement:
- COMM_BASE: the first holding-register word of your IPC<->PLC comm region (e.g., D100)
- SYS_BASE_OFF: word offset from COMM_BASE to the start of SysCommD
"""

from __future__ import annotations

# -------------------------
# Placement
# -------------------------
COMM_WORDS_TOTAL: int = 500
SYS_WORDS: int = 32
SYS_BASE_OFF: int = 468  # start word offset from COMM_BASE (default: last 32 words)

# -------------------------
# SysCommD word offsets (relative to SYS_BASE)
# -------------------------
OFF_SYS_CMD: int         = 0   # WORD  (IPC->PLC)
OFF_SYS_CMD_CLR: int     = 1   # WORD  (IPC->PLC, optional)
OFF_SYS_STS: int         = 2   # WORD  (PLC->IPC)
OFF_SYS_ERR: int         = 3   # UINT  (PLC->IPC)
OFF_SYS_WARN: int        = 4   # UINT  (PLC->IPC)
OFF_SYS_MODE_REQ: int    = 5   # UINT  (IPC->PLC)
OFF_SYS_MODE_ACT: int    = 6   # UINT  (PLC->IPC)
OFF_SYS_SEQ: int         = 7   # UINT  (IPC->PLC)
OFF_SYS_SEQ_ACK: int     = 8   # UINT  (PLC->IPC)
OFF_SYS_HB_PC: int       = 9   # UINT  (IPC->PLC)
OFF_SYS_HB_PLC: int      = 10  # UINT  (PLC->IPC)
OFF_SYS_CYCLE_ID: int    = 11  # UINT  (PLC->IPC)
OFF_SYS_PROGRESS: int    = 12  # UINT  (PLC->IPC, 0..1000)
OFF_SYS_RECIPE_REQ: int  = 13  # UINT  (IPC->PLC)
OFF_SYS_RECIPE_ACT: int  = 14  # UINT  (PLC->IPC)
OFF_SYS_STOP_REASON: int = 15  # UINT  (PLC->IPC)

# Spare words: OFF_SYS_SPARE0..15 = 16..31
OFF_SYS_SPARE0: int      = 16

def sys_reg(comm_base: int, off: int) -> int:
    """Convert SysCommD word offset to absolute holding-register word address."""
    return comm_base + SYS_BASE_OFF + off

# -------------------------
# SYS_CMD bit definitions (WORD, IPC->PLC)
# -------------------------
SYS_CMD_APPLY_REQ: int       = 0x0001  # apply/commit (copy UI edit area -> active)
SYS_CMD_START_CYCLE_REQ: int = 0x0002  # start auto cycle
SYS_CMD_STOP_CYCLE_REQ: int  = 0x0004  # stop auto cycle (graceful stop)
SYS_CMD_FAULT_RESET_REQ: int = 0x0008  # reset faults/alarms
SYS_CMD_ABORT_REQ: int       = 0x0010  # immediate abort (stronger than STOP; optional)
SYS_CMD_RESERVED_5: int      = 0x0020  # reserved

# -------------------------
# SYS_STS bit definitions (WORD, PLC->IPC)
# -------------------------
SYS_STS_READY: int       = 0x0001  # PLC in idle/ready state
SYS_STS_RUNNING: int     = 0x0002  # auto cycle running
SYS_STS_BUSY: int        = 0x0004  # executing a command / applying / transitioning
SYS_STS_DONE: int        = 0x0008  # last cycle done (latched until next start)
SYS_STS_FAULT: int       = 0x0010  # fault present
SYS_STS_ESTOP: int       = 0x0020  # estop active
SYS_STS_REMOTE: int      = 0x0040  # remote control allowed/active
SYS_STS_COMMS_OK: int    = 0x0080  # comms heartbeat OK (PLC perspective)
SYS_STS_APPLYING: int    = 0x0100  # apply in progress
SYS_STS_RESERVED_9: int  = 0x0200  # reserved

# -------------------------
# Mode enums (UINT)
# -------------------------
MODE_UNKNOWN: int = 0
MODE_LOCAL: int   = 1   # local only (HMI / panel)
MODE_REMOTE: int  = 2   # IPC remote allowed
MODE_MANUAL: int  = 10  # manual jog/inch/vel/move allowed
MODE_AUTO: int    = 20  # auto flow allowed

# -------------------------
# Stop reason enums (UINT, PLC->IPC)
# -------------------------
STOP_NONE: int           = 0
STOP_BY_OPERATOR: int    = 1
STOP_BY_FAULT: int       = 2
STOP_BY_ESTOP: int       = 3
STOP_BY_TIMEOUT: int     = 4
STOP_BY_INTERLOCK: int   = 5

"""
Example usage (IPC side):

comm_base = 100  # e.g., D100 as holding register base in your mapping

# send a one-shot START command:
seq = (seq + 1) & 0xFFFF
write_uint(sys_reg(comm_base, OFF_SYS_SEQ), seq)
write_word(sys_reg(comm_base, OFF_SYS_CMD), SYS_CMD_START_CYCLE_REQ)

# PLC will copy SYS_SEQ to SYS_SEQ_ACK after consuming it.
"""
