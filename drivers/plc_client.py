from __future__ import annotations

"""Modbus TCP PLC 通信层（轮询 + 指令队列）。

职责：
- 与 PLC 建立 Modbus TCP 连接
- 轮询读取每轴 Axis_CommD 区域并解析为 AxisComm
- 处理上层下发的写寄存器/置位/脉冲命令（cmd_q）
- 将最新状态通过 ui_q 回传给 UI 线程

重连策略（需求）：
- 软件启动自动连接
- 自动重连最多 5 次
- 退避间隔：5s / 15s / 30s / 60s（第5次仍用 60s）
- 达到 5 次后停止自动重连，直到 UI 手动触发（APPLY）
"""

import struct
import threading
import time
import queue
from dataclasses import dataclass
from typing import List, Optional

from pymodbus.client import ModbusTcpClient

from config.addresses import (
    DEFAULT_PLC_IP,
    DEFAULT_PLC_PORT,
    DEFAULT_UNIT_ID,
    AXIS_COUNT,
    COMM_BASE_D,
    COMM_STRIDE_D,
    COMM_WORDS,
    POLL_INTERVAL_S,
    OFF_CMD,
    OFF_CMD_CLR,
    OFF_STS,
    OFF_ERR,
    OFF_WARN,
    OFF_SEQ,
    OFF_SEQ_ACK,
    OFF_TGT_POS,
    OFF_TGT_POS2,
    OFF_VEL,
    OFF_ACC,
    OFF_DEC,
    OFF_JERK,
    OFF_MODE,
    OFF_ACT_POS,
    OFF_ACT_VEL,
    OFF_ACT_TRQ,
    OFF_DIAG,
    OFF_ST_ID,
    FLOAT64_WORD_ORDER,
)
from core.models import AxisComm


def decode_float64_from_4regs(regs: List[int], order: str) -> float:
    """Decode IEEE754 float64 from 4x16-bit regs with word-order variants."""
    if len(regs) != 4:
        raise ValueError("float64 needs exactly 4 regs")

    if order == "be":
        rr = regs
    elif order == "le":
        rr = [regs[3], regs[2], regs[1], regs[0]]
    elif order == "cdab":
        rr = [regs[2], regs[3], regs[0], regs[1]]
    elif order == "badc":
        rr = [regs[1], regs[0], regs[3], regs[2]]
    else:
        raise ValueError(f"unknown float64 order: {order}")

    b = b"".join(struct.pack(">H", r & 0xFFFF) for r in rr)
    return struct.unpack(">d", b)[0]


def encode_float64_to_4regs(value: float, order: str) -> List[int]:
    """Encode IEEE754 float64 into 4x16-bit regs with word-order variants."""
    b = struct.pack(">d", float(value))
    regs_be = list(struct.unpack(">4H", b))

    if order == "be":
        return regs_be
    if order == "le":
        return [regs_be[3], regs_be[2], regs_be[1], regs_be[0]]
    if order == "cdab":
        return [regs_be[2], regs_be[3], regs_be[0], regs_be[1]]
    if order == "badc":
        return [regs_be[1], regs_be[0], regs_be[3], regs_be[2]]

    raise ValueError(f"unknown float64 order: {order}")


def parse_axis_comm(block: List[int], f64_order: str) -> AxisComm:
    if len(block) < COMM_WORDS:
        raise ValueError("Axis_CommD block too short")

    a = AxisComm()
    a.cmd = block[OFF_CMD] & 0xFFFF
    a.cmd_clr = block[OFF_CMD_CLR] & 0xFFFF
    a.sts = block[OFF_STS] & 0xFFFF
    a.err = block[OFF_ERR] & 0xFFFF
    a.warn = block[OFF_WARN] & 0xFFFF
    a.seq = block[OFF_SEQ] & 0xFFFF
    a.seq_ack = block[OFF_SEQ_ACK] & 0xFFFF

    a.tgt_pos = decode_float64_from_4regs(block[OFF_TGT_POS : OFF_TGT_POS + 4], f64_order)
    a.tgt_pos2 = decode_float64_from_4regs(block[OFF_TGT_POS2 : OFF_TGT_POS2 + 4], f64_order)

    a.vel = block[OFF_VEL] & 0xFFFF
    a.acc = block[OFF_ACC] & 0xFFFF
    a.dec = block[OFF_DEC] & 0xFFFF
    a.jerk = block[OFF_JERK] & 0xFFFF
    a.mode = block[OFF_MODE] & 0xFFFF

    a.act_pos = decode_float64_from_4regs(block[OFF_ACT_POS : OFF_ACT_POS + 4], f64_order)
    a.act_vel = decode_float64_from_4regs(block[OFF_ACT_VEL : OFF_ACT_VEL + 4], f64_order)
    a.act_trq = decode_float64_from_4regs(block[OFF_ACT_TRQ : OFF_ACT_TRQ + 4], f64_order)

    a.diag = block[OFF_DIAG] & 0xFFFF
    a.st_id = block[OFF_ST_ID] & 0xFFFF
    return a


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
    clr_mask: int = 0  # will write to Cmd_Clr (safer clear)


@dataclass
class CmdPulseCmdMask:
    axis: int
    pulse_mask: int
    pulse_ms: int = 120


@dataclass
class CmdWriteModeWord:
    axis: int
    mode_word: int


WorkerCmd = CmdWriteRegs | CmdSetCmdMask | CmdPulseCmdMask | CmdWriteModeWord


class PlcWorker(threading.Thread):
    """PLC Modbus worker.

    Reconnect rules:
    - Auto tries up to max_retries times after failures, with backoff schedule.
    - After give-up, it will NOT auto reconnect until request_connect(manual=True) is called.
    """

    # 需求指定退避表（第5次继续用60s）
    BACKOFF_SCHEDULE_S = [5, 15, 30, 60]

    def __init__(self, ui_q: queue.Queue, cmd_q: queue.Queue):
        super().__init__(daemon=True)
        self.ui_q = ui_q
        self.cmd_q = cmd_q

        self.stop_event = threading.Event()
        self.client: Optional[ModbusTcpClient] = None
        self.connected = False

        self.ip = DEFAULT_PLC_IP
        self.port = DEFAULT_PLC_PORT

        self.unit_id = DEFAULT_UNIT_ID
        self.f64_order = FLOAT64_WORD_ORDER

        self.last_axis: List[AxisComm] = [AxisComm() for _ in range(AXIS_COUNT)]
        self.last_cmd_word: List[int] = [0 for _ in range(AXIS_COUNT)]
        self.last_mode_word: List[int] = [0 for _ in range(AXIS_COUNT)]

        #  重连状态
        self.max_retries = 5
        self.retry_count = 0              # 已发生的失败次数（仅统计“连接/通讯失败导致的断线”）
        self.give_up = False              # True 表示停止自动重连，等待手动触发
        self._manual_kick = False         # 手动触发标志（APPLY）

    # ---- public API ----
    def request_connect(self, ip: str, port: int, manual: bool = False):
        """Update connection target and trigger reconnect.
        - manual=True: UI APPLY触发，重置计数并解除give_up
        - manual=False: 启动时/自动触发（一般不需要反复调用）
        """
        self.ip = ip
        self.port = port

        # 固定协议参数（防止上层绕过）
        self.unit_id = DEFAULT_UNIT_ID
        self.f64_order = FLOAT64_WORD_ORDER

        # 手动连接：解除give_up，并重置计数
        if manual:
            self.retry_count = 0
            self.give_up = False
            self._manual_kick = True
            self.ui_q.put(("plc_manual", {"ts": time.time(), "ip": ip, "port": port}))

        # force reconnect
        self.connected = False
        try:
            if self.client:
                self.client.close()
        except Exception:
            pass
        self.client = None

    # 兼容旧调用名（如果别处还在调用 configure）
    def configure(self, ip: str, port: int):
        self.request_connect(ip=ip, port=port, manual=False)

    def stop(self):
        self.stop_event.set()
        try:
            if self.client:
                self.client.close()
        except Exception:
            pass

    # ---- internal ----
    def _ensure_connected(self):
        if self.client is None:
            # timeout 1s: 失败要快，退避由 run() 控制
            self.client = ModbusTcpClient(self.ip, port=self.port, timeout=1.0)

        if not self.connected:
            self.connected = bool(self.client.connect())

        if not self.connected:
            raise ConnectionError("Modbus TCP connect failed")

    def _read_block(self, axis: int) -> List[int]:
        base = COMM_BASE_D + COMM_STRIDE_D * axis
        rr = self.client.read_holding_registers(
            base,
            count=COMM_WORDS,
            device_id=self.unit_id,
        )
        if rr.isError():
            raise IOError(f"read axis{axis} failed: {rr}")
        return list(rr.registers)

    def _write_regs(self, d_addr: int, values: List[int]):
        ww = self.client.write_registers(
            d_addr,
            [v & 0xFFFF for v in values],
            device_id=self.unit_id,
        )
        if ww.isError():
            raise IOError(f"write regs failed at D{d_addr}: {ww}")

    def _write_single(self, d_addr: int, value: int):
        self._write_regs(d_addr, [value & 0xFFFF])

    def _flush_cmd_queue(self, limit: int = 2000):
        """Drop queued commands to avoid stale motions after reconnect."""
        dropped = 0
        try:
            while dropped < limit:
                self.cmd_q.get_nowait()
                dropped += 1
        except queue.Empty:
            pass
        return dropped

    def _handle_cmd(self, c: WorkerCmd) -> None:
        # 1) raw register writes
        if isinstance(c, CmdWriteRegs):
            self._write_regs(c.d_addr, c.values)
            return

        # 2) write Mode word
        if isinstance(c, CmdWriteModeWord):
            axis = int(c.axis)
            base_mode = COMM_BASE_D + COMM_STRIDE_D * axis + OFF_MODE
            self._write_single(base_mode, int(c.mode_word) & 0xFFFF)
            return

        # 3) set/clear command masks (level operations)
        if isinstance(c, CmdSetCmdMask):
            axis = int(c.axis)
            set_mask = int(c.set_mask) & 0xFFFF
            clr_mask = int(c.clr_mask) & 0xFFFF

            base_cmd = COMM_BASE_D + COMM_STRIDE_D * axis + OFF_CMD
            base_clr = COMM_BASE_D + COMM_STRIDE_D * axis + OFF_CMD_CLR

            cur = int(self.last_cmd_word[axis]) & 0xFFFF

            if set_mask:
                new_cmd = (cur | set_mask) & 0xFFFF
                self._write_single(base_cmd, new_cmd)
                self.last_cmd_word[axis] = new_cmd

            if clr_mask:
                self._write_single(base_clr, clr_mask)
                self.last_cmd_word[axis] = (int(self.last_cmd_word[axis]) & (~clr_mask)) & 0xFFFF

            return

        # 4) pulse command masks (edge-triggered)
        if isinstance(c, CmdPulseCmdMask):
            axis = int(c.axis)
            mask = int(c.pulse_mask) & 0xFFFF
            pulse_ms = int(c.pulse_ms)

            base_cmd = COMM_BASE_D + COMM_STRIDE_D * axis + OFF_CMD
            base_clr = COMM_BASE_D + COMM_STRIDE_D * axis + OFF_CMD_CLR

            cur = int(self.last_cmd_word[axis]) & 0xFFFF

            new_cmd = (cur | mask) & 0xFFFF
            self._write_single(base_cmd, new_cmd)
            self.last_cmd_word[axis] = new_cmd

            time.sleep(max(0, pulse_ms) / 1000.0)

            self._write_single(base_clr, mask)
            self.last_cmd_word[axis] = (int(self.last_cmd_word[axis]) & (~mask)) & 0xFFFF
            return

        raise ValueError(f"unknown WorkerCmd type: {type(c)}")

    def _next_backoff(self) -> int:
        """Return next backoff seconds based on current retry_count (after increment)."""
        idx = max(0, self.retry_count - 1)
        idx = min(idx, len(self.BACKOFF_SCHEDULE_S) - 1)
        return int(self.BACKOFF_SCHEDULE_S[idx])

    def run(self):
        while not self.stop_event.is_set():
            # Give-up: do not auto reconnect until manual kick happens
            if self.give_up and not self._manual_kick:
                # 轻睡眠，避免空转；不处理命令，避免积累危险动作
                time.sleep(0.5)
                continue

            try:
                self._ensure_connected()

                # connected OK: clear manual kick and retry counter
                self._manual_kick = False
                self.retry_count = 0

                # drain commands
                for _ in range(120):
                    try:
                        cmd = self.cmd_q.get_nowait()
                    except queue.Empty:
                        break
                    self._handle_cmd(cmd)

                # poll all axes
                for ax in range(AXIS_COUNT):
                    block = self._read_block(ax)
                    ac = parse_axis_comm(block, self.f64_order)
                    self.last_axis[ax] = ac
                    self.last_cmd_word[ax] = ac.cmd
                    self.last_mode_word[ax] = ac.mode

                self.ui_q.put(
                    ("plc_ok", {"ts": time.time(), "connected": True, "axes": self.last_axis})
                )
                time.sleep(POLL_INTERVAL_S)

            except Exception as e:
                # any comm/connect error => disconnected
                self.connected = False
                try:
                    if self.client:
                        self.client.close()
                except Exception:
                    pass
                self.client = None

                # drop pending commands to avoid stale execution later
                self._flush_cmd_queue()

                # bump retry count and decide whether to give up
                self.retry_count += 1
                backoff_s = self._next_backoff()

                # notify UI about error + retry status
                self.ui_q.put(
                    (
                        "plc_err",
                        {
                            "ts": time.time(),
                            "connected": False,
                            "err": str(e),
                            "retry": self.retry_count,
                            "max": self.max_retries,
                            "backoff_s": backoff_s,
                        },
                    )
                )

                if self.retry_count >= self.max_retries:
                    self.give_up = True
                    self._manual_kick = False
                    self.ui_q.put(
                        (
                            "plc_giveup",
                            {"ts": time.time(), "retry": self.retry_count, "max": self.max_retries},
                        )
                    )
                    # 进入 give-up 后，别再自动睡长时间；交给上面 giveup 分支
                    time.sleep(0.2)
                else:
                    time.sleep(backoff_s)
