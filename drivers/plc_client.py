# ./frp_app/drivers/plc_client.py
from __future__ import annotations

"""Modbus-TCP client worker for XINJE PLC.

This version matches the new AXIS_Ctrl layout (see config/addresses.py):
- No Mode word
- No Cmd_Clr word
- All motion setpoints (vel/acc/dec/jerk/pos) are FP64 (LREAL) words

The worker provides a small command queue API used by app.py.
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Union

from pymodbus.client import ModbusTcpClient

from config.addresses import (
    CL_IN_BASE_D,
    CL_OUT_MEAS_BLOCK_OFF,
    CL_OUT_MEAS_BLOCK_WORDS,
    CL_OUT_CNT_BLOCK_OFF,
    CL_OUT_CNT_BLOCK_WORDS,
    CL_OUT1_WORD_OFF,
    CL_OUT2_WORD_OFF,
    CL_OUT3_WORD_OFF,
    CL_OUT4_WORD_OFF,
    CL_OUT5_WORD_OFF,
    CL_OUT1_UPD_WORD_OFF,
    CL_OUT2_UPD_WORD_OFF,
    CL_OUT3_UPD_WORD_OFF,
    CL_OUT4_UPD_WORD_OFF,
    CL_OUT5_UPD_WORD_OFF,
    CL_OUT_SCALE_MM,
    CL_OUT1_SCALE_MM,
    CL_OUT2_SCALE_MM,
    CL_OUT3_SCALE_MM,
    CL_OUT4_SCALE_MM,
    CL_OUT5_SCALE_MM,
    CL_OUT_INVALID,
    CL_OUT_STANDBY,
    CL_OUT_POS_OVER,
    CL_OUT_NEG_OVER,
    DEFAULT_PLC_IP,
    DEFAULT_PLC_PORT,
    DEFAULT_UNIT_ID,
    KEYTEST_X_BASE_COIL,
    KEYTEST_X_COUNT,
    KEYTEST_Y_BASE_COIL,
    KEYTEST_Y_COUNT,
    AXIS_COUNT,
    COMM_WORDS,
    FLOAT64_WORD_ORDER,
    POLL_INTERVAL_S,
    axis_base,
    # cmd bits
    CMD_EN_REQ,
    CMD_JOG_F_REQ,
    CMD_JOG_B_REQ,
    CMD_VELMOVE_REQ,
    # offsets
    OFF_CMD,
    OFF_SEQ,
    OFF_SEQ_ACK,
    OFF_STS,
    OFF_ST_ID,
    OFF_ERR,
    OFF_WARN,
    OFF_ACT_POS,
    OFF_POS_MOVEA,
    OFF_POS_MOVER,
    OFF_DIR_MOVER,
    OFF_VEL_MOVEA,
    OFF_VEL_MOVER,
    OFF_VEL_JOG,
    OFF_VEL_VELMOVE,
    OFF_ACC,
    OFF_DEC,
    OFF_JERK,
    OFF_SOFTLIM_POS,
    OFF_SOFTLIM_NEG,
)

# core.models 在你的工程中提供 AxisComm 数据结构
from core.models import AxisComm


# =========================
# FP64 helpers (word-based)
# =========================

def _u16(x: int) -> int:
    return int(x) & 0xFFFF


def encode_float64_to_4regs(value: float, word_order: str = "le") -> List[int]:
    import struct

    b = struct.pack(">d", float(value))  # big-endian bytes
    regs = list(struct.unpack(">4H", b))
    if word_order.lower() == "le":
        regs = list(reversed(regs))
    return [_u16(r) for r in regs]


def decode_float64_from_4regs(regs: List[int], word_order: str = "le") -> float:
    import struct

    if regs is None or len(regs) < 4:
        return 0.0
    rr = [_u16(x) for x in regs[:4]]
    if word_order.lower() == "le":
        rr = list(reversed(rr))
    b = struct.pack(">4H", *rr)
    return float(struct.unpack(">d", b)[0])


# =========================
# Worker commands
# =========================

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


# =========================
# Axis parse
# =========================

def _safe_setattr(obj, name: str, value) -> None:
    try:
        setattr(obj, name, value)
    except Exception:
        pass


def parse_axis_ctrl(block: List[int], word_order: str = FLOAT64_WORD_ORDER) -> AxisComm:
    """Parse one AXIS_Ctrl block (COMM_WORDS words) into AxisComm.

    Note: AxisComm is your project dataclass. This function fills the commonly-used
    fields and additionally tries to attach new setpoint fields if AxisComm allows.
    """

    cmd = _u16(block[OFF_CMD])
    seq = _u16(block[OFF_SEQ])
    seq_ack = _u16(block[OFF_SEQ_ACK])
    sts = _u16(block[OFF_STS])
    st_id = _u16(block[OFF_ST_ID])
    err = _u16(block[OFF_ERR])
    warn = _u16(block[OFF_WARN])

    act_pos = decode_float64_from_4regs(block[OFF_ACT_POS : OFF_ACT_POS + 4], word_order)
    pos_movea = decode_float64_from_4regs(block[OFF_POS_MOVEA : OFF_POS_MOVEA + 4], word_order)
    pos_mover = decode_float64_from_4regs(block[OFF_POS_MOVER : OFF_POS_MOVER + 4], word_order)

    dir_mover = _u16(block[OFF_DIR_MOVER])

    vel_movea = decode_float64_from_4regs(block[OFF_VEL_MOVEA : OFF_VEL_MOVEA + 4], word_order)
    vel_mover = decode_float64_from_4regs(block[OFF_VEL_MOVER : OFF_VEL_MOVER + 4], word_order)
    vel_jog = decode_float64_from_4regs(block[OFF_VEL_JOG : OFF_VEL_JOG + 4], word_order)
    vel_velmove = decode_float64_from_4regs(block[OFF_VEL_VELMOVE : OFF_VEL_VELMOVE + 4], word_order)

    acc = decode_float64_from_4regs(block[OFF_ACC : OFF_ACC + 4], word_order)
    dec = decode_float64_from_4regs(block[OFF_DEC : OFF_DEC + 4], word_order)
    jerk = decode_float64_from_4regs(block[OFF_JERK : OFF_JERK + 4], word_order)

    softlim_pos = decode_float64_from_4regs(block[OFF_SOFTLIM_POS : OFF_SOFTLIM_POS + 4], word_order)
    softlim_neg = decode_float64_from_4regs(block[OFF_SOFTLIM_NEG : OFF_SOFTLIM_NEG + 4], word_order)

    # AxisComm: fill new AXIS_Ctrl fields + legacy aliases
    return AxisComm(
        # core
        cmd=cmd,
        seq=seq,
        seq_ack=seq_ack,
        sts=sts,
        st_id=st_id,
        err=err,
        warn=warn,
        # feedback
        act_pos=act_pos,
        # new setpoints
        pos_movea=pos_movea,
        pos_mover=pos_mover,
        dir_mover=dir_mover,
        vel_movea=vel_movea,
        vel_mover=vel_mover,
        vel_jog=vel_jog,
        vel_velmove=vel_velmove,
        acceleration=acc,
        deceleration=dec,
        jerk=jerk,
        softlim_pos=softlim_pos,
        softlim_neg=softlim_neg,
        # legacy aliases
        tgt_pos=pos_movea,
        tgt_pos2=pos_mover,
        vel=vel_movea,
        acc=acc,
        dec=dec,
        mode=0,
        cmd_clr=0,
        act_vel=0.0,
        act_trq=0.0,
        diag=0,
    )


# =========================
# Worker thread
# =========================

class PlcWorker(threading.Thread):
    """Background Modbus poller + command executor."""

    # Bits that should stay asserted as level signals
    LEVEL_BITS = CMD_EN_REQ | CMD_JOG_F_REQ | CMD_JOG_B_REQ | CMD_VELMOVE_REQ


    def __init__(
        self,
        ui_q: "queue.Queue",
        cmd_q: "queue.Queue[WorkerCmd]",
        ip: str = DEFAULT_PLC_IP,
        port: int = DEFAULT_PLC_PORT,
        unit_id: int = DEFAULT_UNIT_ID,
        poll_interval_s: float = POLL_INTERVAL_S,
        word_order: str = FLOAT64_WORD_ORDER,
        connect_on_start: bool = True,
        reconnect_backoff_s: Optional[List[int]] = None,
        reconnect_max_tries: int = 5,
        **_kw,
    ):
        """Create worker.

        This constructor is intentionally backward-compatible with earlier app.py
        that calls: PlcWorker(ui_q, cmd_q).

        Args:
            ui_q: UI event queue (worker -> UI)
            cmd_q: command queue (UI -> worker)
            ip/port/unit_id: Modbus TCP connection parameters
        """
        super().__init__(daemon=True)
        self.ip = str(ip)
        self.port = int(port)
        self.unit_id = int(unit_id)
        self.cmd_q = cmd_q
        self.ui_q = ui_q
        self.poll_interval_s = float(poll_interval_s)
        self.word_order = str(word_order)

        self._stop_evt = threading.Event()
        self._connect_evt = threading.Event()
        if connect_on_start:
            self._connect_evt.set()

        self._client: Optional[ModbusTcpClient] = None
        self._connected = False
        self._lock = threading.Lock()

        self.reconnect_backoff_s = reconnect_backoff_s or [5, 15, 30, 60]
        self.reconnect_max_tries = int(reconnect_max_tries)
        self._retry = 0
        self._giveup = False

        # polling profile
        # normal: poll all axes + CL + keytest
        # sampling: poll only selected axes (default AX3), and disable CL/keytest background polling
        self._poll_profile: str = "normal"
        self._poll_axes_sampling: List[int] = [3]
        self._poll_cl_enable: bool = True
        # Key-test coils polling: keep X (inputs) even in sampling profile for E-Stop/footswitch.
        self._poll_keytest_x_enable: bool = True
        self._poll_keytest_y_enable: bool = True
        self._last_keytest_x_bits = None
        self._last_keytest_y_bits = None
        self._last_axes: List[AxisComm] = [AxisComm() for _ in range(AXIS_COUNT)]

        # per-axis latched command word (level bits only)
        self.level_cmd_word = [0 for _ in range(AXIS_COUNT)]
        # Whether level_cmd_word has been seeded from PLC real cmd word.
        # If the PLC boots with axes already enabled, we MUST seed first, otherwise
        # the first IPC motion command would overwrite the Cmd word and drop EN.
        self._level_inited = [False for _ in range(AXIS_COUNT)]

    def _seed_level_from_plc(self, axis: int) -> None:
        """Seed level_cmd_word from PLC's current Cmd word (LEVEL_BITS only).

        This prevents a common failure mode:
        - PLC side axis is already enabled (Cmd.EN_REQ=1) when IPC starts;
        - IPC sends a pulse command (MoveA/MoveR/Reset/Stop...);
        - if IPC writes Cmd without EN_REQ, PLC will interpret it as disable.
        """
        try:
            if not self._client:
                return
            base = axis_base(axis)
            rr = self._client.read_holding_registers(int(base + OFF_CMD), count=1, device_id=self.unit_id)
            if rr.isError():
                return
            regs = list(getattr(rr, 'registers', []) or [])
            if not regs:
                return
            cmd = _u16(regs[0])
            self.level_cmd_word[axis] = cmd & self.LEVEL_BITS
            self._level_inited[axis] = True
        except Exception:
            return

    # -----------------
    # public control
    # -----------------
    def stop(self):
        self._stop_evt.set()

    def request_connect(self, *, ip: str | None = None, port: int | None = None, manual: bool = True):
        """Request (re)connect.

        - manual=True: reset giveup and retry counter (used by Apply button)
        - manual=False: do NOT clear giveup (auto-connect should stop after giveup)
        """
        if (not manual) and self._giveup:
            return
        if ip is not None:
            self.ip = str(ip)
        if port is not None:
            self.port = int(port)

        self._retry = 0
        if manual:
            self._giveup = False

        self._connect_evt.set()
        if manual:
            self.ui_q.put(("plc_manual", {"ip": self.ip, "port": self.port}))

    # -----------------
    # internal helpers
    # -----------------
    def _disconnect(self):
        try:
            if self._client:
                self._client.close()
        except Exception:
            pass
        self._client = None
        self._connected = False

    def _connect(self) -> bool:
        self._disconnect()
        self._client = ModbusTcpClient(self.ip, port=self.port)
        ok = bool(self._client.connect())
        self._connected = ok
        return ok

    def _read_axis_block(self, axis: int) -> List[int]:
        base = axis_base(axis)
        rr = self._client.read_holding_registers(base, count=COMM_WORDS, device_id=self.unit_id)
        if rr.isError():
            raise RuntimeError(f"read error: {rr}")
        regs = list(rr.registers)
        if len(regs) < COMM_WORDS:
            raise RuntimeError(f"short read: {len(regs)}")
        return regs

    def _write_regs(self, d_addr: int, values: List[int]):
        wr = self._client.write_registers(d_addr, [_u16(v) for v in values], device_id=self.unit_id)
        if wr.isError():
            raise RuntimeError(f"write error: {wr}")

    def _write_coil(self, coil_addr: int, value: int):
        """Write a single coil (0/1)."""
        vv = bool(int(value) != 0)
        wr = self._client.write_coil(int(coil_addr), vv, device_id=self.unit_id)
        if wr.isError():
            raise RuntimeError(f"write coil error: {wr}")

    def _write_axis_cmd_word(self, axis: int, word: int):
        base = axis_base(axis)
        self._write_regs(base + OFF_CMD, [_u16(word)])

    def _apply_cmd_level(self, axis: int, set_mask: int, clr_mask: int):
        # Ensure we don't accidentally drop an already-asserted EN_REQ at startup.
        if not self._level_inited[axis]:
            self._seed_level_from_plc(axis)
        lvl = self.level_cmd_word[axis] & self.LEVEL_BITS
        lvl |= (set_mask & self.LEVEL_BITS)
        lvl &= (~clr_mask) & 0xFFFF
        self.level_cmd_word[axis] = lvl
        self._write_axis_cmd_word(axis, lvl)

    def _pulse_cmd(self, axis: int, pulse_mask: int, pulse_ms: int):
        # Ensure we don't accidentally drop an already-asserted EN_REQ at startup.
        if not self._level_inited[axis]:
            self._seed_level_from_plc(axis)
        lvl = self.level_cmd_word[axis] & self.LEVEL_BITS
        word_on = _u16(lvl | (pulse_mask & 0xFFFF))
        self._write_axis_cmd_word(axis, word_on)
        time.sleep(max(0.02, float(pulse_ms) / 1000.0))
        self._write_axis_cmd_word(axis, lvl)

    # -----------------
    # main loop
    # -----------------
    def run(self):
        while not self._stop_evt.is_set():
            # (re)connect if requested
            if (not self._connected) and (not self._giveup) and self._connect_evt.is_set():
                try:
                    ok = self._connect()
                    if ok:
                        self._retry = 0
                        self._connect_evt.clear()
                    else:
                        raise RuntimeError("connect() failed")
                except Exception as e:
                    self._disconnect()
                    # backoff retry
                    self._retry += 1
                    if self._retry > self.reconnect_max_tries:
                        self._giveup = True
                        self.ui_q.put(("plc_giveup", {"retry": self._retry, "max": self.reconnect_max_tries}))
                    else:
                        backoff_s = self.reconnect_backoff_s[min(self._retry - 1, len(self.reconnect_backoff_s) - 1)]
                        self.ui_q.put(("plc_err", {"err": str(e), "retry": self._retry, "max": self.reconnect_max_tries, "backoff_s": backoff_s}))
                        time.sleep(float(backoff_s))
                    continue

            if not self._connected:
                time.sleep(0.2)
                continue

            try:
                with self._lock:
                    # 1) drain cmd queue (writes)
                    while True:
                        try:
                            cmd = self.cmd_q.get_nowait()
                        except queue.Empty:
                            break

                        if isinstance(cmd, CmdWriteRegs):
                            self._write_regs(cmd.d_addr, cmd.values)

                        elif isinstance(cmd, CmdWriteCoil):
                            self._write_coil(int(cmd.coil_addr), int(cmd.value))

                        elif isinstance(cmd, CmdReadRegs):
                            # On-demand read (e.g., HD axis calibration block)
                            d_addr = int(cmd.d_addr)
                            count = int(cmd.count)
                            tag = str(getattr(cmd, "tag", "") or "")
                            rr = self._client.read_holding_registers(
                                d_addr, count=count, device_id=self.unit_id
                            )
                            if rr.isError():
                                raise RuntimeError(f"read error: {rr}")
                            regs = list(rr.registers)
                            self.ui_q.put(
                                (
                                    "plc_read",
                                    {
                                        "tag": tag,
                                        "d_addr": d_addr,
                                        "count": count,
                                        "regs": regs,
                                    },
                                )
                            )

                        elif isinstance(cmd, CmdSetPollProfile):
                            prof = str(getattr(cmd, "profile", "normal") or "normal").strip().lower()
                            if prof not in ("normal", "sampling"):
                                prof = "normal"
                            self._poll_profile = prof
                            # enable/disable background polling by profile
                            if prof == "sampling":
                                # Reduce load: keep only AX3 (angle) + X inputs; disable CL and Y.
                                self._poll_cl_enable = False
                                self._poll_keytest_x_enable = True
                                self._poll_keytest_y_enable = False
                            else:
                                self._poll_cl_enable = True
                                self._poll_keytest_x_enable = True
                                self._poll_keytest_y_enable = True

                        elif isinstance(cmd, CmdSetCmdMask):
                            ax = max(0, min(AXIS_COUNT - 1, int(cmd.axis)))
                            self._apply_cmd_level(ax, int(cmd.set_mask), int(cmd.clr_mask))

                        elif isinstance(cmd, CmdPulseCmdMask):
                            ax = max(0, min(AXIS_COUNT - 1, int(cmd.axis)))
                            self._pulse_cmd(ax, int(cmd.pulse_mask), int(cmd.pulse_ms))

                # 2) poll axis data
                # During AutoFlow sampling we can greatly reduce polling load by only updating AX3,
                # while keeping last snapshots for other axes.
                with self._lock:
                    if self._poll_profile == "sampling":
                        axes: List[AxisComm] = list(self._last_axes)
                        for ax in self._poll_axes_sampling:
                            try:
                                ax_i = max(0, min(AXIS_COUNT - 1, int(ax)))
                                block = self._read_axis_block(ax_i)
                                axes[ax_i] = parse_axis_ctrl(block, self.word_order)
                            except Exception:
                                # keep last snapshot
                                pass
                        self._last_axes = axes
                    else:
                        axes = []
                        for ax in range(AXIS_COUNT):
                            block = self._read_axis_block(ax)
                            axes.append(parse_axis_ctrl(block, self.word_order))
                        self._last_axes = list(axes)

                # Sync local level bits from PLC snapshot (Cmd word).
                # This keeps IPC's "level" intentions aligned with PLC reality
                # and prevents EN_REQ drop when PLC powers on with enabled axes.
                for ax in range(AXIS_COUNT):
                    try:
                        self.level_cmd_word[ax] = int(getattr(axes[ax], 'cmd', 0)) & self.LEVEL_BITS
                        self._level_inited[ax] = True
                    except Exception:
                        pass

                # 3) poll CL (Keyence) input words if mapped (OUT1..OUT5 + update counters)
                #
                # 说明：
                # - OUT1..OUT5 均为 DINT32 (2 words)；更新计数为 UINT32 (2 words)
                # - 为降低 Modbus 读负载，按连续块一次读取：
                #     measurements: word off 38..47 (10 words)
                #     counters:      word off 70..79 (10 words)
                cl_out1_raw = None
                cl_out1_mm = None
                cl_out1_cnt = None
                cl_out2_raw = None
                cl_out2_mm = None
                cl_out2_cnt = None
                cl_out3_raw = None
                cl_out3_mm = None
                cl_out3_cnt = None
                cl_out4_raw = None
                cl_out4_mm = None
                cl_out4_cnt = None
                cl_out5_raw = None
                cl_out5_mm = None
                cl_out5_cnt = None

                if self._poll_cl_enable:
                    try:
                        def _s32_from_2regs(r0: int, r1: int) -> int:
                            u32 = int(r0 & 0xFFFF) | (int(r1 & 0xFFFF) << 16)
                            return int(u32 - 0x100000000) if (u32 & 0x80000000) else int(u32)

                        def _u32_from_2regs(r0: int, r1: int) -> int:
                            return int(r0 & 0xFFFF) | (int(r1 & 0xFFFF) << 16)

                        def _to_mm(raw: int | None, scale_mm: float) -> float | None:
                            if raw is None:
                                return None
                            if raw in {CL_OUT_INVALID, CL_OUT_STANDBY, CL_OUT_POS_OVER, CL_OUT_NEG_OVER}:
                                return None
                            return float(raw) * float(scale_mm)

                        # measurements block
                        rr = self._client.read_holding_registers(
                            int(CL_IN_BASE_D + CL_OUT_MEAS_BLOCK_OFF),
                            count=int(CL_OUT_MEAS_BLOCK_WORDS),
                            device_id=self.unit_id,
                        )
                        if not rr.isError():
                            regs = list(getattr(rr, 'registers', []) or [])
                            if len(regs) >= int(CL_OUT_MEAS_BLOCK_WORDS):
                                cl_out1_raw = _s32_from_2regs(regs[0], regs[1])
                                cl_out2_raw = _s32_from_2regs(regs[2], regs[3])
                                cl_out3_raw = _s32_from_2regs(regs[4], regs[5])
                                cl_out4_raw = _s32_from_2regs(regs[6], regs[7])
                                cl_out5_raw = _s32_from_2regs(regs[8], regs[9])

                                # NOTE: OUT1/OUT2/OUT5 use finer resolution (0.0001 mm/LSB) in current CL program.
                                #       OUT3/OUT4 keep 0.001 mm/LSB.
                                cl_out1_mm = _to_mm(cl_out1_raw, CL_OUT1_SCALE_MM)
                                cl_out2_mm = _to_mm(cl_out2_raw, CL_OUT2_SCALE_MM)
                                cl_out3_mm = _to_mm(cl_out3_raw, CL_OUT3_SCALE_MM)
                                cl_out4_mm = _to_mm(cl_out4_raw, CL_OUT4_SCALE_MM)
                                cl_out5_mm = _to_mm(cl_out5_raw, CL_OUT5_SCALE_MM)

                        # counters block
                        rr2 = self._client.read_holding_registers(
                            int(CL_IN_BASE_D + CL_OUT_CNT_BLOCK_OFF),
                            count=int(CL_OUT_CNT_BLOCK_WORDS),
                            device_id=self.unit_id,
                        )
                        if not rr2.isError():
                            regs2 = list(getattr(rr2, 'registers', []) or [])
                            if len(regs2) >= int(CL_OUT_CNT_BLOCK_WORDS):
                                cl_out1_cnt = _u32_from_2regs(regs2[0], regs2[1])
                                cl_out2_cnt = _u32_from_2regs(regs2[2], regs2[3])
                                cl_out3_cnt = _u32_from_2regs(regs2[4], regs2[5])
                                cl_out4_cnt = _u32_from_2regs(regs2[6], regs2[7])
                                cl_out5_cnt = _u32_from_2regs(regs2[8], regs2[9])
                    except Exception:
                        pass

                # 4) poll key-test coils (X/Y)
                # In sampling profile we keep X polling for E-Stop/footswitch, and freeze Y at last snapshot.
                keytest_x_bits = getattr(self, '_last_keytest_x_bits', None)
                keytest_y_bits = getattr(self, '_last_keytest_y_bits', None)

                if getattr(self, '_poll_keytest_x_enable', False):
                    try:
                        rrx = self._client.read_coils(int(KEYTEST_X_BASE_COIL), count=int(KEYTEST_X_COUNT), device_id=self.unit_id)
                        if not rrx.isError():
                            bits = list(getattr(rrx, 'bits', []) or [])
                            keytest_x_bits = [1 if bool(b) else 0 for b in bits[: int(KEYTEST_X_COUNT)]]
                            self._last_keytest_x_bits = keytest_x_bits
                    except Exception:
                        pass

                if getattr(self, '_poll_keytest_y_enable', False):
                    try:
                        rry = self._client.read_coils(int(KEYTEST_Y_BASE_COIL), count=int(KEYTEST_Y_COUNT), device_id=self.unit_id)
                        if not rry.isError():
                            bits = list(getattr(rry, 'bits', []) or [])
                            keytest_y_bits = [1 if bool(b) else 0 for b in bits[: int(KEYTEST_Y_COUNT)]]
                            self._last_keytest_y_bits = keytest_y_bits
                    except Exception:
                        pass



                self.ui_q.put(
                    (
                        "plc_ok",
                        {
                            "axes": axes,
                            "cl_out1_raw": cl_out1_raw,
                            "cl_out1_mm": cl_out1_mm,
                            "cl_out1_cnt": cl_out1_cnt,
                            "cl_out2_raw": cl_out2_raw,
                            "cl_out2_mm": cl_out2_mm,
                            "cl_out2_cnt": cl_out2_cnt,
                            "cl_out3_raw": cl_out3_raw,
                            "cl_out3_mm": cl_out3_mm,
                            "cl_out3_cnt": cl_out3_cnt,
                            "cl_out4_raw": cl_out4_raw,
                            "cl_out4_mm": cl_out4_mm,
                            "cl_out4_cnt": cl_out4_cnt,
                            "cl_out5_raw": cl_out5_raw,
                            "cl_out5_mm": cl_out5_mm,
                            "cl_out5_cnt": cl_out5_cnt,
                            "keytest_x_bits": keytest_x_bits,
                            "keytest_y_bits": keytest_y_bits,
                        },
                    )
                )

            except Exception as e:
                self.ui_q.put(("plc_err", {"err": str(e)}))
                self._disconnect()
                time.sleep(0.2)

            time.sleep(self.poll_interval_s)

        self._disconnect()
