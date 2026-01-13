# ./frp_app/config/addresses.py
from __future__ import annotations

"""PLC/IPC 通信地址、位定义与 Axis_CommD 布局常量。

该文件应作为全项目唯一的“协议事实来源（single source of truth）”：
- PLC 通信区基地址/步进（COMM_BASE_D / COMM_STRIDE_D）
- Axis_CommD 的寄存器偏移（OFF_*）
- Cmd/Mode/Sts 位掩码（CMD_* / MODE_* / STS_*）
- 默认连接参数（DEFAULT_*）
"""

# =========================
# Basic config
# =========================
DEFAULT_PLC_IP = "192.168.6.6"
DEFAULT_PLC_PORT = 502
DEFAULT_UNIT_ID = 1
DEFAULT_GAUGE_PORT = "COM2"

AXIS_NAMES = [
    "AX0 测径仪支架",
    "AX1 滑台支架",
    "AX2 中心架",
    "AX3 旋转轴",
    "AX4 位移计滑台",
]
AXIS_COUNT = len(AXIS_NAMES)

COMM_BASE_D = 100        # Axis_CommD_0 at D100
COMM_STRIDE_D = 100      # Axis_CommD_n at D(100 + 100*n)
COMM_WORDS = 40          # D100..D139 (每轴通信区长度)

POLL_INTERVAL_S = 0.15

# =========================
# PLC-side bit definitions (your fixed protocol)
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
# Axis_CommD layout offsets (based on your screenshot)
# =========================
OFF_CMD      = 0     # D100
OFF_CMD_CLR  = 1     # D101
OFF_STS      = 2     # D102
OFF_ERR      = 3     # D103
OFF_WARN     = 4     # D104
OFF_SEQ      = 5     # D105
OFF_SEQ_ACK  = 6     # D106
OFF_TGT_POS  = 8     # D108..D111 (FP64, 4 words)
OFF_TGT_POS2 = 12    # D112..D115 (FP64, 4 words)
OFF_VEL      = 16    # D116
OFF_ACC      = 17    # D117
OFF_DEC      = 18    # D118
OFF_JERK     = 19    # D119
OFF_MODE     = 20    # D120
OFF_ACT_POS  = 24    # D124..D127 (FP64, 4 words)
OFF_ACT_VEL  = 28    # D128..D131 (FP64, 4 words)
OFF_ACT_TRQ  = 32    # D132..D135 (FP64, 4 words)
OFF_DIAG     = 36    # D136
OFF_ST_ID    = 37    # D137

# =========================
# Float64 packing order
# NOTE: if Act_Pos is wrong, switch here (or via UI dropdown).
# =========================
FLOAT64_WORD_ORDER = "le"  # be / le / cdab / badc
