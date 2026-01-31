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
- 52  Softlim_pos (LREAL)   52..55   软限位正向（轴绝对位置）
- 56  Softlim_neg (LREAL)   56..59   软限位负向（轴绝对位置）
- 60..99 reserved

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

# =========================
# Key test (PLC X/Y points via Modbus coils)
# =========================
# 目的：
# - X 点：只读显示（物理输入），用于验证现场按键/传感器是否触发
# - Y 点：读状态 + 单次写入(0/1)，用于验证 IPC 写入输出是否生效
#
# 地址约定（按用户提供）：
# - X0..X17：线圈地址空间中 *没有* X8/X9（X 点采用“八进制标签”），
#   因此 X10 的线圈地址紧随 X7：
#     X0..X7   -> coil 20480..20487
#     X10..X17 -> coil 20488..20495
# - Y0..Y15：同理 *没有* Y8/Y9：
#     Y0..Y7   -> coil 24576..24583
#     Y10..Y15 -> coil 24584..24589

KEYTEST_X_BASE_COIL: int = 20480
KEYTEST_X_COUNT: int = 16  # X0..X7 + X10..X17

KEYTEST_Y_BASE_COIL: int = 24576
KEYTEST_Y_COUNT: int = 14  # Y0..Y7 + Y10..Y15

# 展示点位（不显示 8/9）
KEYTEST_X_POINTS = [
    0, 1, 2, 3, 4, 5, 6, 7,
    10, 11, 12, 13, 14, 15, 16, 17,
]

KEYTEST_Y_POINTS = [
    0, 1, 2, 3, 4, 5, 6, 7,
    10, 11, 12, 13, 14, 15,
]

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
AXIS_WORDS_USED: int = 60          # 0..59 (includes Softlim_pos/neg)
COMM_WORDS: int = AXIS_WORDS_USED  # 轮询读取长度（word）

AXIS_BASE_OFF: int = SYS_WORDS     # AX0 从 D100 开始

COMM_WORDS_TOTAL: int = SYS_WORDS + AXIS_COUNT * COMM_STRIDE_D
assert COMM_WORDS_TOTAL == 50 + AXIS_COUNT * 100

# FP64 16-bit word order（PLC=LREAL=FP64）
FLOAT64_WORD_ORDER: str = "le"  # 低字在前



# =========================
# Keyence CL-3000 (EIP input mapped into PLC D area)
# =========================
# NOTE: These are PLC D addresses (word addressing) for the CL input assembly (device->PLC).
# In your PLC project, map CL input (Assembly 0x64, 272 bytes = 136 words) to D2000..D2135.
CL_IN_BASE_D: int = 2000

# OUT3: byte 84..87 => word offset 42..43 (DWORD)
CL_OUT3_WORD_OFF: int = 42
# OUT3 update counter: byte 148..151 => word offset 74..75 (DWORD)
CL_OUT3_UPD_WORD_OFF: int = 74

# OUT1: byte 76..79 => word offset 38..39 (DINT32)
CL_OUT1_WORD_OFF: int = 38
# OUT2: byte 80..83 => word offset 40..41 (DINT32)
CL_OUT2_WORD_OFF: int = 40
# OUT4: byte 88..91 => word offset 44..45 (DINT32)
CL_OUT4_WORD_OFF: int = 44
# OUT5: byte 92..95 => word offset 46..47 (DINT32)
CL_OUT5_WORD_OFF: int = 46

# OUT1 update counter: byte 140..143 => word offset 70..71 (UINT32)
CL_OUT1_UPD_WORD_OFF: int = 70
# OUT2 update counter: byte 144..147 => word offset 72..73 (UINT32)
CL_OUT2_UPD_WORD_OFF: int = 72
# OUT4 update counter: byte 152..155 => word offset 76..77 (UINT32)
CL_OUT4_UPD_WORD_OFF: int = 76
# OUT5 update counter: byte 156..159 => word offset 78..79 (UINT32)
CL_OUT5_UPD_WORD_OFF: int = 78

# Convenience blocks (reduce Modbus reads)
# Measurements OUT1..OUT5 occupy word offsets 38..47 (10 words)
CL_OUT_MEAS_BLOCK_OFF: int = 38
CL_OUT_MEAS_BLOCK_WORDS: int = 10
# Update counters OUT1..OUT5 occupy word offsets 70..79 (10 words)
CL_OUT_CNT_BLOCK_OFF: int = 70
CL_OUT_CNT_BLOCK_WORDS: int = 10

# Project convention: use OUT4 as 'ID direct' output from CL.
CL_ID_WORD_OFF: int = CL_OUT4_WORD_OFF
CL_ID_UPD_WORD_OFF: int = CL_OUT4_UPD_WORD_OFF

# Measurement value scaling (DINT -> mm).
#
# 现象定位：ID（内径）结果约为标准值的 10 倍（例如应为 152.700mm，但显示 1527.0mm）。
# 这通常意味着 PLC 映射到 D 区的 OUT3 原始值单位更细（常见为 μm），
# 因此应按 0.001 mm / LSB 换算（即 raw=152700 -> 152.700mm）。
# 若你后续在 CL 侧调整最小显示单位/缩放，请同步调整此系数。
"""Keyence CL (EIP -> PLC D area) scaling.

现象（以 CL-NavigatorN 面板显示为准）：
- OUT1/OUT2/OUT5 显示 4 位小数（例如 -0.2924 mm / 1.1428 mm），对应 PLC 原始 DINT 的分辨率更细。
- OUT3/OUT4 显示 3 位小数（例如 147.208 mm / 152.791 mm），对应 PLC 原始 DINT 的分辨率为 0.001 mm。

因此：
- OUT1/OUT2/OUT5: 0.0001 mm/LSB
- OUT3/OUT4:       0.001  mm/LSB

历史版本只有一个统一的 CL_OUT_SCALE_MM；为兼容保留该常量（默认按 0.001）。
"""

# Backward-compatible default scale (kept for historical code paths)
CL_OUT_SCALE_MM: float = 0.001

# Per-output scales (recommended)
CL_OUT_SCALE_MM_FINE: float = 0.0001

CL_OUT1_SCALE_MM: float = CL_OUT_SCALE_MM_FINE
CL_OUT2_SCALE_MM: float = CL_OUT_SCALE_MM_FINE
CL_OUT5_SCALE_MM: float = CL_OUT_SCALE_MM_FINE

CL_OUT3_SCALE_MM: float = CL_OUT_SCALE_MM
CL_OUT4_SCALE_MM: float = CL_OUT_SCALE_MM

# Main ID mapping uses OUT4
CL_ID_SCALE_MM: float = CL_OUT4_SCALE_MM

# Special values (DINT) reported by CL
CL_OUT_INVALID: int = -999999
CL_OUT_STANDBY: int = -999998
CL_OUT_POS_OVER: int = 999999
CL_OUT_NEG_OVER: int = -999999

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

# 软限位（轴绝对位置，FP64）
OFF_SOFTLIM_POS = 52  # FP64 (52..55)
OFF_SOFTLIM_NEG = 56  # FP64 (56..59)

OFF_SOFTLIM_POS = 52  # FP64 (52..55)
OFF_SOFTLIM_NEG = 56  # FP64 (56..59)

# 兼容别名（旧代码中常用的命名）
OFF_TGT_POS  = OFF_POS_MOVEA
OFF_TGT_POS2 = OFF_POS_MOVER
OFF_VEL      = OFF_VEL_MOVEA




# =========================
# Soft limits (SFD, read-only)
# =========================
# User rule:
# - SFD base (Modbus holding regs): 58560
# - +soft limit: SFD8064 + 300*N  (FP64 / LREAL, 4 words)
# - -soft limit: SFD8068 + 300*N  (FP64 / LREAL, 4 words)
# Example: AX1 +soft = 58560 + 8064 + 300*1 = 66924
SFD_BASE_D: int = 58560
SFD_SOFT_LIM_POS_OFF: int = 8064
SFD_SOFT_LIM_NEG_OFF: int = 8068
SFD_AXIS_STRIDE_D: int = 300

LINEAR_AXES = (0, 1, 2, 4)

def sfd_soft_lim_pos_addr(axis: int) -> int:
    return SFD_BASE_D + SFD_SOFT_LIM_POS_OFF + SFD_AXIS_STRIDE_D * int(axis)

def sfd_soft_lim_neg_addr(axis: int) -> int:
    return SFD_BASE_D + SFD_SOFT_LIM_NEG_OFF + SFD_AXIS_STRIDE_D * int(axis)

# =========================
# Axis calibration block (PLC HD1000..1027 mapped to Modbus holding regs)
# =========================
# Your rule: Modbus address = HD + 41088
# Therefore HD1000 -> 42088
AXISCAL_MB_BASE: int = 42088
AXISCAL_WORDS: int = 32

# Field offsets within the axis calibration block (word offsets)
AXISCAL_OFF_SIGN: int = 0
AXISCAL_OFF_OFF_AX0: int = 4
AXISCAL_OFF_OFF_AX1: int = 8
AXISCAL_OFF_OFF_AX2: int = 12
AXISCAL_OFF_OFF_AX4: int = 16
AXISCAL_OFF_B14: int = 20
AXISCAL_OFF_B2: int = 24
AXISCAL_OFF_KEEPOUT_W: int = 28

