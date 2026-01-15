# ./frp_app/config/addresses.py
from __future__ import annotations

"""PLC/IPC 通信地址、位定义与布局常量（统一版，single source of truth）。

目标
- 用“统一通信基址 + 偏移”方式访问：axis 与 system 都通过 helper 计算绝对D地址（word address）。
- 固定总地址空间长度：COMM_WORDS_TOTAL = 500 words（D区连续映射）。
- 轴控制字/状态字与系统控制字/状态字分区；未使用的间隙明确标记为 RESERVED，便于后续扩展且不破坏兼容。

总览（默认）
- 通信总区：D{COMM_BASE_D} .. D{COMM_BASE_D + COMM_WORDS_TOTAL - 1}（共 500 words）
- 轴区：每轴一个槽位，槽位跨度 COMM_STRIDE_D=100 words（用于保持每轴基址对齐）
    * AXn 槽位：D(COMM_BASE_D + 100*n) .. D(COMM_BASE_D + 100*n + 99)
    * 其中前 AXIS_WORDS_USED=40 words 为 AxisCommD 已用字段，其余为每轴 RESERVED
- 系统区：放在通信总区尾部（SysCommD，32 words）
    * SYS_BASE_OFF = COMM_WORDS_TOTAL - SYS_WORDS = 468
    * SYS：D(COMM_BASE_D + 468) .. D(COMM_BASE_D + 499)

注意
- 你目前的 AxisCommD stride=100 + 5轴，刚好构成 500 words（100..599）。
- SysCommD 放在尾部，会占用 AX4 槽位的 RESERVED 区间（这是预期设计，不会覆盖 AX4 已用字段）。
"""

# =========================
# Basic config
# =========================
DEFAULT_PLC_IP = "192.168.6.6"
DEFAULT_PLC_PORT = 502
DEFAULT_UNIT_ID = 1
DEFAULT_GAUGE_PORT = "COM2"

POLL_INTERVAL_S = 0.15

AXIS_NAMES = [
    "AX0 测径仪支架",
    "AX1 滑台支架",
    "AX2 中心架",
    "AX3 旋转轴",
    "AX4 位移计滑台",
]
AXIS_COUNT = len(AXIS_NAMES)

# =========================
# Communication memory layout (word-addressed, D registers)
# =========================
COMM_BASE_D: int = 100            # 通信总区起始 D 地址（word）
COMM_WORDS_TOTAL: int = 500       # 固定：通信总区总长度（word）

COMM_STRIDE_D: int = 100          # 每轴槽位跨度（word）
AXIS_WORDS_USED: int = 40         # AxisCommD 已用字段长度（word）

SYS_WORDS: int = 32               # SysCommD 长度（word）
SYS_BASE_OFF: int = COMM_WORDS_TOTAL - SYS_WORDS  # 468（从COMM_BASE起的偏移）

# 校验：5轴×stride=500 words
assert AXIS_COUNT * COMM_STRIDE_D == COMM_WORDS_TOTAL, (
    "AXIS_COUNT * COMM_STRIDE_D must equal COMM_WORDS_TOTAL to keep a single continuous COMM region."
)

# =========================
# Helpers: absolute D-word address from base + offset
# =========================
def comm_reg(off: int) -> int:
    """COMM总区的绝对D地址：COMM_BASE_D + off"""
    return COMM_BASE_D + off

def axis_base(axis: int) -> int:
    """AXn 槽位起始D地址"""
    if not (0 <= axis < AXIS_COUNT):
        raise ValueError(f"axis out of range: {axis}")
    return COMM_BASE_D + COMM_STRIDE_D * axis

def axis_reg(axis: int, off: int) -> int:
    """AXn 内字段的绝对D地址（off 为 AxisCommD word offset）"""
    return axis_base(axis) + off

def sys_base() -> int:
    """SysCommD 起始D地址"""
    return COMM_BASE_D + SYS_BASE_OFF

def sys_reg(off: int) -> int:
    """SysCommD 内字段的绝对D地址（off 为 SysCommD word offset）"""
    return sys_base() + off

# =========================
# Reserved gaps (explicit)
# =========================
# Per-axis reserved range inside each 100-word slot:
#   D(axis_base + AXIS_WORDS_USED) .. D(axis_base + COMM_STRIDE_D - 1)
AXIS_RESERVED_OFF_START: int = AXIS_WORDS_USED       # 40
AXIS_RESERVED_OFF_END: int = COMM_STRIDE_D - 1       # 99

# Global reserved ranges (excluding axis used areas and SysCommD) can be derived if needed.

# =========================
# PLC-side bit definitions (axis protocol, fixed)
# =========================
# Cmd (WORD, IPC write)
CMD_EN_REQ      = 0x0001
CMD_RESET_REQ   = 0x0002
CMD_STOP_REQ    = 0x0004
CMD_HALT_REQ    = 0x0008
CMD_MOVEA_REQ   = 0x0020
CMD_VELMOVE_REQ = 0x0080
CMD_JOG_F_REQ   = 0x0100
CMD_JOG_B_REQ   = 0x0200

# Mode (UINT, IPC write)
MODE_INCH    = 0x0001
MODE_DIR_REV = 0x0002

# Sts (WORD, PLC write)
STS_READY     = 0x0001
STS_ENABLED   = 0x0002
STS_BUSY      = 0x0004
STS_DONE      = 0x0008
STS_FAULT     = 0x0020
STS_JOGGING   = 0x0100
STS_VELRUN    = 0x0200
STS_INTERLOCK = 0x4000

# =========================
# AxisCommD layout offsets (relative to AXn base)
# =========================
OFF_CMD      = 0     # WORD  IPC->PLC
OFF_CMD_CLR  = 1     # WORD  IPC->PLC (optional)
OFF_STS      = 2     # WORD  PLC->IPC
OFF_ERR      = 3     # UINT  PLC->IPC
OFF_WARN     = 4     # UINT  PLC->IPC (reserved extension)
OFF_SEQ      = 5     # UINT  IPC->PLC (optional, for jog/inch semantics)
OFF_SEQ_ACK  = 6     # UINT  PLC->IPC (optional)
# 7 reserved
OFF_TGT_POS  = 8     # LREAL FP64, 4 words: [8..11]
OFF_TGT_POS2 = 12    # LREAL FP64, 4 words: [12..15]
OFF_VEL      = 16    # UINT
OFF_ACC      = 17    # UINT
OFF_DEC      = 18    # UINT
OFF_JERK     = 19    # UINT (reserved)
OFF_MODE     = 20    # UINT
# 21..23 reserved
OFF_ACT_POS  = 24    # LREAL FP64, 4 words: [24..27]
OFF_ACT_VEL  = 28    # LREAL FP64, 4 words: [28..31]
OFF_ACT_TRQ  = 32    # LREAL FP64, 4 words: [32..35] (reserved)
OFF_DIAG     = 36    # WORD (reserved)
OFF_ST_ID    = 37    # UINT (debug)
# 38..39 reserved

# =========================
# SysCommD layout offsets (relative to SYS base)
# =========================
OFF_SYS_CMD: int         = 0   # WORD  IPC->PLC
OFF_SYS_CMD_CLR: int     = 1   # WORD  IPC->PLC (optional)
OFF_SYS_STS: int         = 2   # WORD  PLC->IPC
OFF_SYS_ERR: int         = 3   # UINT  PLC->IPC
OFF_SYS_WARN: int        = 4   # UINT  PLC->IPC
OFF_SYS_MODE_REQ: int    = 5   # UINT  IPC->PLC
OFF_SYS_MODE_ACT: int    = 6   # UINT  PLC->IPC
OFF_SYS_SEQ: int         = 7   # UINT  IPC->PLC (SEQ/ACK one-shot)
OFF_SYS_SEQ_ACK: int     = 8   # UINT  PLC->IPC
OFF_SYS_HB_PC: int       = 9   # UINT  IPC->PLC
OFF_SYS_HB_PLC: int      = 10  # UINT  PLC->IPC
OFF_SYS_CYCLE_ID: int    = 11  # UINT  PLC->IPC
OFF_SYS_PROGRESS: int    = 12  # UINT  PLC->IPC (0..1000)
OFF_SYS_RECIPE_REQ: int  = 13  # UINT  IPC->PLC
OFF_SYS_RECIPE_ACT: int  = 14  # UINT  PLC->IPC
OFF_SYS_STOP_REASON: int = 15  # UINT  PLC->IPC
OFF_SYS_SPARE0: int      = 16  # UINT  reserved (16..31)

# =========================
# SysCmd bit definitions (WORD, IPC->PLC)
# =========================
SYS_CMD_APPLY_REQ: int       = 0x0001
SYS_CMD_START_CYCLE_REQ: int = 0x0002
SYS_CMD_STOP_CYCLE_REQ: int  = 0x0004
SYS_CMD_FAULT_RESET_REQ: int = 0x0008
SYS_CMD_ABORT_REQ: int       = 0x0010  # immediate abort (optional)
SYS_CMD_RESERVED_5: int      = 0x0020

# =========================
# SysSts bit definitions (WORD, PLC->IPC)
# =========================
SYS_STS_READY: int       = 0x0001
SYS_STS_RUNNING: int     = 0x0002
SYS_STS_BUSY: int        = 0x0004
SYS_STS_DONE: int        = 0x0008
SYS_STS_FAULT: int       = 0x0010
SYS_STS_ESTOP: int       = 0x0020
SYS_STS_REMOTE: int      = 0x0040
SYS_STS_COMMS_OK: int    = 0x0080
SYS_STS_APPLYING: int    = 0x0100
SYS_STS_RESERVED_9: int  = 0x0200

# =========================
# System mode enums (UINT)
# =========================
MODE_UNKNOWN: int = 0
MODE_LOCAL: int   = 1   # local only (HMI/panel)
MODE_REMOTE: int  = 2   # IPC remote allowed
MODE_MANUAL: int  = 10  # manual allowed
MODE_AUTO: int    = 20  # auto flow allowed

# =========================
# Stop reason enums (UINT)
# =========================
STOP_NONE: int           = 0
STOP_BY_OPERATOR: int    = 1
STOP_BY_FAULT: int       = 2
STOP_BY_ESTOP: int       = 3
STOP_BY_TIMEOUT: int     = 4
STOP_BY_INTERLOCK: int   = 5

# =========================
# Float64 packing order (FP64 word order)
# =========================
FLOAT64_WORD_ORDER = "le"  # be / le / cdab / badc
