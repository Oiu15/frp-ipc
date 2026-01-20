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
    DEFAULT_PLC_IP,
    DEFAULT_PLC_PORT,
    DEFAULT_UNIT_ID,
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
class CmdSetCmdMask:
    axis: int
    set_mask: int = 0
    clr_mask: int = 0


@dataclass
class CmdPulseCmdMask:
    axis: int
    pulse_mask: int
    pulse_ms: int = 120


WorkerCmd = Union[CmdWriteRegs, CmdSetCmdMask, CmdPulseCmdMask]


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

        # per-axis latched command word (level bits only)
        self.level_cmd_word = [0 for _ in range(AXIS_COUNT)]

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

    def _write_axis_cmd_word(self, axis: int, word: int):
        base = axis_base(axis)
        self._write_regs(base + OFF_CMD, [_u16(word)])

    def _apply_cmd_level(self, axis: int, set_mask: int, clr_mask: int):
        lvl = self.level_cmd_word[axis] & self.LEVEL_BITS
        lvl |= (set_mask & self.LEVEL_BITS)
        lvl &= (~clr_mask) & 0xFFFF
        self.level_cmd_word[axis] = lvl
        self._write_axis_cmd_word(axis, lvl)

    def _pulse_cmd(self, axis: int, pulse_mask: int, pulse_ms: int):
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

                        elif isinstance(cmd, CmdSetCmdMask):
                            ax = max(0, min(AXIS_COUNT - 1, int(cmd.axis)))
                            self._apply_cmd_level(ax, int(cmd.set_mask), int(cmd.clr_mask))

                        elif isinstance(cmd, CmdPulseCmdMask):
                            ax = max(0, min(AXIS_COUNT - 1, int(cmd.axis)))
                            self._pulse_cmd(ax, int(cmd.pulse_mask), int(cmd.pulse_ms))

                # 2) poll axis data
                axes: List[AxisComm] = []
                with self._lock:
                    for ax in range(AXIS_COUNT):
                        block = self._read_axis_block(ax)
                        axes.append(parse_axis_ctrl(block, self.word_order))

                self.ui_q.put(("plc_ok", {"axes": axes}))

            except Exception as e:
                self.ui_q.put(("plc_err", {"err": str(e)}))
                self._disconnect()
                time.sleep(0.2)

            time.sleep(self.poll_interval_s)

        self._disconnect()
