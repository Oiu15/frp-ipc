# ./core/models.py
from __future__ import annotations

"""项目核心数据模型（纯数据，无 IO / 无 UI）。

约束：
- 该层不依赖 tkinter、pymodbus、serial 等外部 IO。
- 该层可被 drivers / services / ui 任意引用。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class AxisComm:
    """Axis comm snapshot used across IPC.

    Historical note:
    - Earlier versions used AxisCommD-style layout (cmd/mode/cmd_clr + bitfield sts + UINT setpoints).
    - Current PLC layout is AXIS_Ctrl (no mode/cmd_clr; sts is raw_axis_state 0..8;
      setpoints are LREAL/FP64).

    To keep the UI/services stable, this model keeps legacy aliases (tgt_pos/tgt_pos2/vel/acc/dec/jerk)
    while also exposing the new explicit AXIS_Ctrl fields (pos_movea/pos_mover/vel_movea/...).
    """

    # ---- core words (AXIS_Ctrl) ----
    cmd: int = 0
    seq: int = 0
    seq_ack: int = 0
    sts: int = 0          # raw_axis_state (0..8)
    st_id: int = 0
    err: int = 0          # raw_axis_err
    warn: int = 0         # non-axis errors (BMC errors, interlock, etc.)

    # ---- feedback (AXIS_Ctrl) ----
    act_pos: float = 0.0

    # ---- setpoints (AXIS_Ctrl, FP64) ----
    pos_movea: float = 0.0
    pos_mover: float = 0.0
    dir_mover: int = 0
    vel_movea: float = 0.0
    vel_mover: float = 0.0
    vel_jog: float = 0.0
    vel_velmove: float = 0.0
    acceleration: float = 0.0
    deceleration: float = 0.0
    jerk: float = 0.0

    # ---- legacy compatibility fields (may be unused in new protocol) ----
    # retained so old UI/service code doesn't break
    cmd_clr: int = 0
    mode: int = 0
    tgt_pos: float = 0.0
    tgt_pos2: float = 0.0
    vel: float = 0.0
    acc: float = 0.0
    dec: float = 0.0

    # legacy feedback
    act_vel: float = 0.0
    act_trq: float = 0.0
    diag: int = 0


@dataclass
class UiCoord:
    """A UI-only coordinate system anchored to a captured 'zero' based on servo feedback.

    x_ui = sign * (x_abs - zero_abs)
    x_abs = zero_abs + sign * x_ui
    """

    zero_abs: float = 0.0
    sign: int = +1  # +1: positive is increasing abs; -1: positive is decreasing abs

    def abs_to_ui(self, x_abs: float) -> float:
        return float(self.sign) * (float(x_abs) - float(self.zero_abs))

    def ui_to_abs(self, x_ui: float) -> float:
        return float(self.zero_abs) + float(self.sign) * float(x_ui)


@dataclass
class Recipe:
    name: str = "默认配方"
    pipe_len_mm: float = 1700.0
    clamp_occupy_mm: float = 300.0
    margin_head_mm: float = 20.0
    margin_tail_mm: float = 20.0
    section_count: int = 12
    scan_axis: int = 0
    od_std_mm: float = 187.3
    id_std_mm: float = 152.7  # v0.1: display only
    od_tol_mm: float = 0.100  # v0.1: default for deviation (abs)

    # 每个截面每圈采样点数（等角bin数量）
    points_per_rev: int = 120

    # 等角采样：最小角度覆盖率（0~1），达到即停止采样
    min_bin_coverage: float = 0.95

    # 等角采样：单截面采样最大等待时间(s)，超时则以已覆盖的bin拟合
    sample_timeout_s: float = 5.0

    # 等角采样：单截面采样最大圈数(转)，达到则停止（与覆盖率/超时共同构成退出条件）
    max_revolutions: float = 2.0

    # UI coordinate positions (final, after teaching). Length == section_count.
    section_pos_ui: List[float] = field(default_factory=list)

    def measurable_len(self) -> float:
        return max(0.0, float(self.pipe_len_mm) - float(self.clamp_occupy_mm))

    def compute_default_positions_ui(self) -> List[float]:
        """Compute default section positions in UI coordinates (0..L_meas) with margins."""
        n = max(1, int(self.section_count))
        l_meas = self.measurable_len()
        start = float(self.margin_head_mm)
        end = max(start, l_meas - float(self.margin_tail_mm))

        if n == 1:
            return [0.5 * (start + end)]

        step = (end - start) / float(n - 1)
        return [start + i * step for i in range(n)]


@dataclass
class MeasureRow:
    idx: int
    x_ui: float
    x_abs: float
    od_avg: float
    od_max: float
    od_min: float
    dev: float
    od_round: float  # 真圆度（按直径差：maxOD-minOD）
    ok: bool
    raw: str = ""


@dataclass
class GaugeSample:
    ts: float
    od: float
    raw: str = ""
