# ./core/models.py
from __future__ import annotations

"""项目核心数据模型（纯数据，无 IO / 无 UI）。

约束：
- 该层不依赖 tkinter、pymodbus、serial 等外部 IO。
- 该层可被 drivers / services / ui 任意引用。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from core.modbus_codec import decode_fp64_le, decode_int16, encode_fp64_le, encode_int16


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
class AxisCal:
    """Axis calibration & unified Z coordinate mapping (pure math).

    Coordinate conventions (as agreed for this project):
    - Z axis positive direction is *downwards*.
    - For AX0/AX1/AX4, servo feedback (abs) positive is upwards in your machine.
      Therefore default `sign` is -1 to make Z positive downwards.
    - AX2 feedback positive is downwards, so it uses an automatic inverted sign.

    Offsets (`off_ax*`) are defined as the servo feedback position (abs) at Z_raw == 0.
    Thus:
        z_raw = sign_eff(axis) * (abs - off_axis)
        abs   = off_axis + sign_eff(axis) * z_raw

    `z_pos` is an IPC-only temporary UI shift (not written to PLC):
        z_disp = z_raw - z_pos
        z_raw  = z_disp + z_pos

    Inner diameter (ID) axis is composed by AX1 + AX4 (AX4 is extension stage).
    `b14` defines the offset between OD section Z and ID section Z:
        z_id_disp = z_od_disp + b14

    `handoff_z` is the AX1/AX4 handoff point in Z_raw coordinates.
    """

    sign: int = -1

    off_ax0: float = 0.0
    off_ax1: float = 0.0
    off_ax2: float = 0.0
    off_ax4: float = 0.0

    b14: float = 0.0
    handoff_z: float = 0.0

    # IPC-only temporary UI shift
    z_pos: float = 0.0

    def sign_eff(self, axis: int) -> int:
        """Effective sign for given axis.

        - AX2 is mechanically opposite: uses -sign.
        - All other axes: uses sign.
        """
        s = int(self.sign)
        return -s if int(axis) == 2 else s

    def _off_for_axis(self, axis: int) -> float:
        a = int(axis)
        if a == 0:
            return float(self.off_ax0)
        if a == 1:
            return float(self.off_ax1)
        if a == 2:
            return float(self.off_ax2)
        if a == 4:
            return float(self.off_ax4)
        # Other axes (e.g. rotate AX3) are not part of Z mapping
        return 0.0

    def abs_to_z_raw(self, axis: int, abs_pos: float) -> float:
        """Convert servo feedback position (abs) to raw Z coordinate."""
        s = float(self.sign_eff(axis))
        off = self._off_for_axis(axis)
        return s * (float(abs_pos) - off)

    def z_raw_to_abs(self, axis: int, z_raw: float) -> float:
        """Convert raw Z coordinate to servo feedback position (abs)."""
        s = float(self.sign_eff(axis))
        off = self._off_for_axis(axis)
        return off + s * float(z_raw)

    def z_raw_to_z_disp(self, z_raw: float) -> float:
        """Raw Z to displayed Z (IPC UI) by applying `z_pos` shift."""
        return float(z_raw) - float(self.z_pos)

    def z_disp_to_z_raw(self, z_disp: float) -> float:
        """Displayed Z (IPC UI) back to raw Z by removing `z_pos` shift."""
        return float(z_disp) + float(self.z_pos)

    def od_z_disp_to_targets(
        self, z_od_disp: float
    ) -> Tuple[float, float, float, float, float, float]:
        """Given OD section Z (display coordinates), compute motion targets.

        Returns:
            (ax0_abs, ax1_abs, ax4_abs, z_id_disp, z1_disp, z4_disp)

        Where:
        - z_id_disp = z_od_disp + b14
        - z1/z4 are the split (AX1/AX4) targets in display coordinates.
        """
        z_od_disp_f = float(z_od_disp)

        # OD raw & AX0 abs
        z_od_raw = self.z_disp_to_z_raw(z_od_disp_f)
        ax0_abs = self.z_raw_to_abs(0, z_od_raw)

        # ID target in display coordinates (per requirement)
        z_id_disp = z_od_disp_f + float(self.b14)
        z_id_raw = self.z_disp_to_z_raw(z_id_disp)

        # Split in raw coordinates by handoff_z
        handoff = float(self.handoff_z)
        if z_id_raw <= handoff:
            z1_raw = z_id_raw
            z4_raw = 0.0
        else:
            z1_raw = handoff
            z4_raw = z_id_raw - handoff

        # Convert split back to display
        z1_disp = self.z_raw_to_z_disp(z1_raw)
        z4_disp = self.z_raw_to_z_disp(z4_raw)

        # Compute abs targets for AX1/AX4
        ax1_abs = self.z_raw_to_abs(1, z1_raw)
        ax4_abs = self.z_raw_to_abs(4, z4_raw)

        return ax0_abs, ax1_abs, ax4_abs, z_id_disp, z1_disp, z4_disp

    # ------------------- Modbus regs codec (pure) -------------------
    # AxisCal struct in PLC (HD1000 ..) mapped to Modbus (base 42088):
    #   Sign      : +0   (INT16, 1 word)
    #   padding   : +1..+3
    #   Off_ax0   : +4   (FP64, 4 words)
    #   Off_ax1   : +8
    #   Off_ax2   : +12
    #   Off_ax4   : +16
    #   B14       : +20
    #   Handoff_z : +24
    # Total: 28 words

    _REG_WORDS: int = 28
    _OFF_SIGN: int = 0
    _OFF_OFF_AX0: int = 4
    _OFF_OFF_AX1: int = 8
    _OFF_OFF_AX2: int = 12
    _OFF_OFF_AX4: int = 16
    _OFF_B14: int = 20
    _OFF_HANDOFF_Z: int = 24

    @classmethod
    def from_regs(cls, regs: List[int], base: int = 0) -> "AxisCal":
        """Create AxisCal from a Modbus register block.

        Args:
            regs: list of 16-bit registers that contains at least 28 words
                  starting at `base`.
            base: start index in `regs`.

        Notes:
            - `z_pos` is IPC-only; it is initialized to 0.0 here.
            - Uses little-endian FP64 decoding per project convention.
        """
        if regs is None:
            raise ValueError("regs is None")
        if len(regs) < base + cls._REG_WORDS:
            raise ValueError(
                f"from_regs needs >= {base + cls._REG_WORDS} regs, got {len(regs)}"
            )

        def fp64_at(off: int) -> float:
            i = base + off
            return decode_fp64_le(regs[i : i + 4])

        sign = decode_int16(regs[base + cls._OFF_SIGN])
        return cls(
            sign=sign,
            off_ax0=fp64_at(cls._OFF_OFF_AX0),
            off_ax1=fp64_at(cls._OFF_OFF_AX1),
            off_ax2=fp64_at(cls._OFF_OFF_AX2),
            off_ax4=fp64_at(cls._OFF_OFF_AX4),
            b14=fp64_at(cls._OFF_B14),
            handoff_z=fp64_at(cls._OFF_HANDOFF_Z),
            z_pos=0.0,
        )

    def to_regs(self) -> List[int]:
        """Encode AxisCal into a 28-word Modbus register block.

        Notes:
            - `z_pos` is IPC-only and is NOT encoded.
            - Unused/padding words are set to 0.
        """
        regs: List[int] = [0] * int(self._REG_WORDS)
        regs[self._OFF_SIGN] = encode_int16(int(self.sign))

        def put_fp64(off: int, value: float) -> None:
            enc = encode_fp64_le(float(value))
            regs[off : off + 4] = enc

        put_fp64(self._OFF_OFF_AX0, self.off_ax0)
        put_fp64(self._OFF_OFF_AX1, self.off_ax1)
        put_fp64(self._OFF_OFF_AX2, self.off_ax2)
        put_fp64(self._OFF_OFF_AX4, self.off_ax4)
        put_fp64(self._OFF_B14, self.b14)
        put_fp64(self._OFF_HANDOFF_Z, self.handoff_z)
        return regs


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
