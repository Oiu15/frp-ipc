# ./core/models.py
from __future__ import annotations

"""项目核心数据模型（纯数据，无 IO / 无 UI）。

约束：
- 该层不依赖 tkinter、pymodbus、serial 等外部 IO。
- 该层可被 drivers / services / ui 任意引用。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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

    # ---- soft limits (absolute position, FP64) ----
    # Mapped in PLC side from RSFD (SFD area) into AXIS_Ctrl for IPC polling.
    softlim_pos: float = 0.0
    softlim_neg: float = 0.0

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
    - AX2 follows the same sign convention as other axes (f1_11).

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

    Keepout zone (for AX2 center clamp) is parameterized by:
    - `b2`: keepout center offset relative to AX2 Z_raw
    - `keepout_w`: keepout half width in Z_raw
  (PLC-side `handoff_z` has been removed; ID split will be derived from keepout.)
    """

    sign: int = -1

    off_ax0: float = 0.0
    off_ax1: float = 0.0
    off_ax2: float = 0.0
    off_ax4: float = 0.0

    b14: float = 0.0

    b2: float = 0.0
    keepout_w: float = 0.0

    # IPC-only temporary UI shift
    z_pos: float = 0.0

    def sign_eff(self, axis: int) -> int:
        """Effective sign for given axis.

        Project note:
        - Earlier versions treated AX2 as "opposite" and auto-inverted its sign.
        - From f1_11, AX2 uses the same sign convention as other axes so that
          `z_raw = sign * (abs - off)` is consistent across AX0/AX1/AX2/AX4.
          This improves the monotonic behavior of keepout constraints vs AX2 abs.
        """
        s = int(self.sign)
        AX2_DIR_SIGN = 1  # set to -1 to invert AX2 only
        return s * (AX2_DIR_SIGN if int(axis) == 2 else 1)

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

    def abs_to_z_disp(self, axis: int, abs_pos: float) -> float:
        """Convert servo feedback position (abs) to displayed Z (IPC UI)."""
        return self.z_raw_to_z_disp(self.abs_to_z_raw(axis, abs_pos))

    def z_disp_to_abs(self, axis: int, z_disp: float) -> float:
        """Convert displayed Z (IPC UI) to servo feedback position (abs)."""
        return self.z_raw_to_abs(axis, self.z_disp_to_z_raw(z_disp))


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


    def od_z_disp_to_targets(self, z_od_disp: float, ax2_abs: Optional[float] = None, softlims_abs: Optional[Dict[int, Tuple[float, float]]] = None) -> Dict[str, float]:
        """Given OD section Z (display coordinates), compute motion targets.

        Returns a dict (stable, app-friendly):
            {
              "ax0_abs": <float>,
              "ax1_abs": <float>,
              "ax4_abs": <float>,
              "z_id_disp": <float>,
              "z1_disp": <float>,
              "z4_disp": <float>,
            }

        Where:
        - z_id_disp = z_od_disp + b14
        - z1/z4 are the split (AX1/AX4) targets in display coordinates.

        Keepout logic (f1_9, direction-aware):
        - AX0: collision direction is +Abs (=> Z_raw decreases). Constrain Z0_raw >= (Zc - W).
        - AX1: collision direction is -Abs (=> Z_raw increases). Constrain Z1_raw <= (Zc + W).

        Soft limits (if provided) are applied on top of keepout.
        """
        z_od_disp_f = float(z_od_disp)

        # ---------- Keepout bounds in Z_raw (derived from AX2) ----------
        try:
            if ax2_abs is not None:
                z2_raw = float(self.abs_to_z_raw(2, float(ax2_abs)))
            else:
                z2_raw = 0.0
        except Exception:
            z2_raw = 0.0

        z_center = z2_raw + float(self.b2)
        w = float(self.keepout_w)
        keepout_low = z_center - w
        keepout_high = z_center + w

        def _raw_range_from_softlims(axis: int):
            if not softlims_abs:
                return None
            v = softlims_abs.get(int(axis))
            if not v:
                return None
            try:
                sp, sn = float(v[0]), float(v[1])
                # Some PLC projects may report 0/0 when soft limits are not configured.
                if (sp == 0.0 and sn == 0.0):
                    return None
                r1 = float(self.abs_to_z_raw(int(axis), sp))
                r2 = float(self.abs_to_z_raw(int(axis), sn))
                lo, hi = (r1, r2) if r1 <= r2 else (r2, r1)
                return lo, hi
            except Exception:
                return None

        def _clamp(x: float, lo: float, hi: float) -> float:
            return lo if x < lo else (hi if x > hi else x)

        # ---------- OD raw & AX0 abs ----------
        z_od_raw = self.z_disp_to_z_raw(z_od_disp_f)

        r0 = _raw_range_from_softlims(0)
        if r0 is None:
            lo0, hi0 = (-1e12, 1e12)
        else:
            lo0, hi0 = float(r0[0]), float(r0[1])
        z_od_raw = _clamp(float(z_od_raw), float(lo0), float(hi0))

        # Keepout for AX0: do not allow Z_raw smaller than keepout_low (approach direction)
        z_od_raw = max(float(z_od_raw), float(keepout_low))
        ax0_abs = self.z_raw_to_abs(0, float(z_od_raw))

        # ---------- ID target (display) ----------
        z_id_disp = z_od_disp_f + float(self.b14)
        z_id_raw = self.z_disp_to_z_raw(z_id_disp)

        # ---------- Split ID (AX1 + AX4) in raw ----------
        # Strategy (f1_9):
        #   - Default: equal split (half to AX1, half to AX4) to minimize max travel time.
        #   - Constraints:
        #       * AX1 must stay within its soft limits (if provided)
        #       * AX1 must NOT exceed keepout_high (approach direction for AX1)
        #       * AX4 must stay within its soft limits (if provided)
        #   - If AX1 hits its constraint, the remaining travel is assigned to AX4.

        # AX1 raw constraints
        r1 = _raw_range_from_softlims(1)
        if r1 is None:
            lo1, hi1 = (-1e12, 1e12)
        else:
            lo1, hi1 = float(r1[0]), float(r1[1])
        hi1 = min(float(hi1), float(keepout_high))

        # AX4 raw constraints
        r4 = _raw_range_from_softlims(4)
        if r4 is None:
            lo4, hi4 = (-1e12, 1e12)
        else:
            lo4, hi4 = float(r4[0]), float(r4[1])

        # Default equal split
        z1_raw = float(z_id_raw) * 0.5
        z1_raw = _clamp(z1_raw, float(lo1), float(hi1))

        # Solve with constraints (two-pass is enough for 2 variables)
        for _ in range(2):
            z4_raw = float(z_id_raw) - float(z1_raw)
            z4_raw = _clamp(z4_raw, float(lo4), float(hi4))
            z1_raw = float(z_id_raw) - float(z4_raw)
            z1_raw = _clamp(z1_raw, float(lo1), float(hi1))

        # Convert split back to display
        z1_disp = self.z_raw_to_z_disp(z1_raw)
        z4_disp = self.z_raw_to_z_disp(z4_raw)

        # Compute abs targets for AX1/AX4
        ax1_abs = self.z_raw_to_abs(1, z1_raw)
        ax4_abs = self.z_raw_to_abs(4, z4_raw)

        return {
            "ax0_abs": float(ax0_abs),
            "ax1_abs": float(ax1_abs),
            "ax4_abs": float(ax4_abs),
            "z_id_disp": float(z_id_disp),
            "z1_disp": float(z1_disp),
            "z4_disp": float(z4_disp),
        }


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
    # Total: 32 words

    _REG_WORDS: int = 32
    _OFF_SIGN: int = 0
    _OFF_OFF_AX0: int = 4
    _OFF_OFF_AX1: int = 8
    _OFF_OFF_AX2: int = 12
    _OFF_OFF_AX4: int = 16
    _OFF_B14: int = 20
    _OFF_B2: int = 24
    _OFF_KEEPOUT_W: int = 28

    @classmethod
    def from_regs(cls, regs: List[int], base: int = 0) -> "AxisCal":
        """Create AxisCal from a Modbus register block.

        Args:
            regs: list of 16-bit registers that contains at least 32 words
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
            b2=fp64_at(cls._OFF_B2),
            keepout_w=fp64_at(cls._OFF_KEEPOUT_W),
            z_pos=0.0,
        )

    def to_regs(self) -> List[int]:
        """Encode AxisCal into a 32-word Modbus register block.

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
        put_fp64(self._OFF_B2, self.b2)
        put_fp64(self._OFF_KEEPOUT_W, self.keepout_w)
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
    # Teach axes selection (UI only): 0=OD(AX0), 1=ID(AX1+AX4), 2=OD+ID
    teach_axes_mode: int = 2
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

    
    # Circle-fit strategy:
    # a: raw points fit
    # b: raw points fit with per-bin weight balancing
    # c: bin-center angle + scalar r_bin average (route A)
    # Stored as a short tagged string, e.g. "b 原始点按bin权重均衡"
    fit_strategy: str = "b 原始点按bin权重均衡"

# Z coordinate positions (display Z_Pos, mm, positive downwards). Length == section_count.
    section_pos_z: List[float] = field(default_factory=list)

    # Legacy UI_Pos positions kept for backward compatibility (deprecated).
    section_pos_ui: List[float] = field(default_factory=list)

    # Standby (待定点) absolute positions for returning after auto-flow.
    # These are absolute axis positions (mm) in the servo's engineering units.
    standby_valid: bool = False
    standby_ax0_abs: float = 0.0
    standby_ax1_abs: float = 0.0
    standby_ax4_abs: float = 0.0

    # Center clamp (AX2) saved positions (absolute)
    ax2_len_valid: bool = False
    ax2_len_abs: float = 0.0
    ax2_rot_valid: bool = False
    ax2_rot_abs: float = 0.0

    def measurable_len(self) -> float:
        return max(0.0, float(self.pipe_len_mm) - float(self.clamp_occupy_mm))

    def compute_default_positions_z(self) -> List[float]:
        """Compute default section positions in Z_Pos coordinates (mm).

        Notes:
            - Z_Pos is the displayed Z coordinate (positive downwards).
            - By default we still distribute sections along the measurable length
              using head/tail margins, identical to the old UI_Pos behavior.
        """
        n = max(1, int(self.section_count))
        l_meas = self.measurable_len()
        start = float(self.margin_head_mm)
        end = max(start, l_meas - float(self.margin_tail_mm))

        if n == 1:
            return [0.5 * (start + end)]

        step = (end - start) / float(n - 1)
        return [start + i * step for i in range(n)]

    def compute_default_positions_ui(self) -> List[float]:
        """Deprecated: kept for backward compatibility. Use compute_default_positions_z."""
        return self.compute_default_positions_z()


@dataclass
class MeasureRow:
    idx: int
    # Z_Pos (display, mm, positive downwards). Kept as x_ui for backward compatibility.
    x_ui: float
    # AX0 absolute target/position used by AutoFlow. Kept as x_abs for backward compatibility.
    x_abs: float

    # OD
    od_avg: float
    od_dev: float
    od_runout: float  # 外径径向跳动（按半径差：maxR-minR）
    od_round: float  # 真圆度（按直径差：maxOD-minOD）

    # ID
    id_avg: float
    id_dev: float
    id_runout: float  # 内径径向跳动（按半径差：maxR-minR）
    id_round: float

    # Concentricity between fitted OD/ID circles (mm)
    concentricity: float

    # Eccentricity to fitted axis line (mm). Filled after all sections measured.
    od_ecc: Optional[float] = None
    id_ecc: Optional[float] = None

    # Section-level pass/fail flag (for future highlighting/export).
    ok: bool = True

    raw: str = ""



@dataclass
class GaugeSample:
    ts: float
    od: float
    raw: str = ""