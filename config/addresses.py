# ./frp_app/config/addresses.py
from __future__ import annotations

"""PLC/IPC 通信地址与轴控制字（AXIS_Ctrl）。

本文件以 PLC 变量表为准（截图：g_axis_comm_ipc[0] : AXIS_Ctrl 映射到 D100..D199）。

通信总区（D寄存器，word addressing）
- COMM_BASE_D = 50
- SysCommD：D50..D99（50 words）
- 轴槽位：每轴 100 words，对齐方便扩展
    AX0：D100..D199
    AX1：D200..D299
    AX2：D300..D399
    AX3：D400..D499
    AX4：D500..D599

AXIS_Ctrl（每轴槽位内 offset，单位=word）
- 0   Cmd (WORD)
- 1   Seq (UINT)
- 2   Seq_Ack (UINT)
- 3   Sts (UINT)            raw_axis_state (0..8)
- 4   St_ID (UINT)
- 5   ErrCode (UINT)        raw_axis_err
- 6   Warn (UINT)           预留：放其他BMC错误/警告
- 7   reserved
- 8   Act_Pos (LREAL,FP64)  8..11
- 12  Pos_MoveA (LREAL)     12..15
- 16  Pos_MoveR (LREAL)     16..19
- 20  Dir_MoveR (UINT)
- 21..23 reserved
- 24  Vel_MoveA (LREAL)     24..27
- 28  Vel_MoveR (LREAL)     28..31
- 32  Vel_Jog (LREAL)       32..35
- 36  Vel_VelMove (LREAL)   36..39
- 40  Acceleration (LREAL)  40..43
- 44  Deceleration (LREAL)  44..47
- 48  Jerk (LREAL)          48..51
- 52..99 reserved

FP64 字序：PLC 侧 LREAL=FP64，按你要求使用 LE（低字在前）。
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
# Communication memory layout
# =========================
COMM_BASE_D: int = 50

SYS_WORDS: int = 50
SYS_BASE_OFF: int = 0

COMM_STRIDE_D: int = 100
AXIS_WORDS_USED: int = 52          # 0..51
COMM_WORDS: int = AXIS_WORDS_USED  # 轮询读取长度（word）

AXIS_BASE_OFF: int = SYS_WORDS     # AX0 从 D100 开始

COMM_WORDS_TOTAL: int = SYS_WORDS + AXIS_COUNT * COMM_STRIDE_D
assert COMM_WORDS_TOTAL == 50 + AXIS_COUNT * 100

# FP64 16-bit word order（PLC=LREAL=FP64）
FLOAT64_WORD_ORDER: str = "le"  # 低字在前


# =========================
# Helpers
# =========================
def comm_reg(off: int) -> int:
    return COMM_BASE_D + off


def axis_base(axis: int) -> int:
    if not (0 <= axis < AXIS_COUNT):
        raise ValueError(f"axis out of range: {axis}")
    return COMM_BASE_D + AXIS_BASE_OFF + COMM_STRIDE_D * axis


def axis_reg(axis: int, off: int) -> int:
    return axis_base(axis) + off


def sys_base() -> int:
    return COMM_BASE_D + SYS_BASE_OFF


def sys_reg(off: int) -> int:
    return sys_base() + off


# =========================
# Cmd bits (Axis_Ctrl.Cmd)
# =========================
CMD_EN_REQ      = 0x0001
CMD_RESET_REQ   = 0x0002
CMD_STOP_REQ    = 0x0004
CMD_HALT_REQ    = 0x0008
CMD_MOVEA_REQ   = 0x0020
CMD_MOVER_REQ   = 0x0040
CMD_VELMOVE_REQ = 0x0080
CMD_JOG_F_REQ   = 0x0100
CMD_JOG_B_REQ   = 0x0200


# =========================
# Dir_MoveR enum (Axis_Ctrl.Dir_MoveR)
# =========================
# 0：无方向 1：正向 2：负向 3：最短路径 4：当前方向（模轴模式下生效）
DIR_NONE     = 0
DIR_POS      = 1
DIR_NEG      = 2
DIR_SHORTEST = 3
DIR_CURRENT  = 4


# =========================
# Sts enum (Axis_Ctrl.Sts = raw_axis_state)
# =========================
STS_RAW_NOT_ENABLED = 0
STS_RAW_ENABLED_IDLE = 1
STS_RAW_MOVING = 2
STS_RAW_VELRUN = 3
STS_RAW_SYNC = 4
STS_RAW_HOMING = 5
STS_RAW_STOPPING = 6
STS_RAW_FAULT = 7
STS_RAW_GROUP = 8


# =========================
# Axis_Ctrl offsets (word)
# =========================
OFF_CMD      = 0
OFF_SEQ      = 1
OFF_SEQ_ACK  = 2
OFF_STS      = 3
OFF_ST_ID    = 4
OFF_ERR      = 5
OFF_WARN     = 6

OFF_ACT_POS  = 8      # FP64 (8..11)
OFF_POS_MOVEA = 12    # FP64 (12..15)
OFF_POS_MOVER = 16    # FP64 (16..19)
OFF_DIR_MOVER = 20    # UINT

OFF_VEL_MOVEA   = 24  # FP64 (24..27)
OFF_VEL_MOVER   = 28  # FP64 (28..31)
OFF_VEL_JOG     = 32  # FP64 (32..35)
OFF_VEL_VELMOVE = 36  # FP64 (36..39)

OFF_ACC   = 40        # FP64 (40..43)
OFF_DEC   = 44        # FP64 (44..47)
OFF_JERK  = 48        # FP64 (48..51)

# 兼容别名（旧代码中常用的命名）
OFF_TGT_POS  = OFF_POS_MOVEA
OFF_TGT_POS2 = OFF_POS_MOVER
OFF_VEL      = OFF_VEL_MOVEA


# =========================
# Axis calibration block (PLC HD1000..1027 mapped to Modbus holding regs)
# =========================
# Your rule: Modbus address = HD + 41088
# Therefore HD1000 -> 42088
AXISCAL_MB_BASE: int = 42088
AXISCAL_WORDS: int = 28

# Field offsets within the axis calibration block (word offsets)
AXISCAL_OFF_SIGN: int = 0
AXISCAL_OFF_OFF_AX0: int = 4
AXISCAL_OFF_OFF_AX1: int = 8
AXISCAL_OFF_OFF_AX2: int = 12
AXISCAL_OFF_OFF_AX4: int = 16
AXISCAL_OFF_B14: int = 20
AXISCAL_OFF_HANDOFF_Z: int = 24

