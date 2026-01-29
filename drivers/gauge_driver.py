# ./frp_app/drivers/gauge_driver.py
from __future__ import annotations

"""测径仪串口通信（外径 OD）。

设计目标：
- 连接稳定：重复点击“连接”不会反复 open/close 串口，避免 Windows PermissionError(13)
- 请求/响应友好：上层可调用 send_request()，后台线程异步读取并解析
 - 对协议更稳健：严格解析完整帧（M1/M0 + 数值 [+ 鉴别]），拒绝半帧/乱码，避免极端离群值
"""

import queue
import re
import threading
import time
import math
from dataclasses import dataclass
from typing import Optional

try:
    import serial  # type: ignore
    import serial.tools.list_ports  # type: ignore
except Exception:  # pragma: no cover
    # Allow the app to run in "模拟测径仪" mode without pyserial installed.
    serial = None  # type: ignore

from core.models import GaugeSample


_JUDGE_SET = {"HH", "HI", "GO", "LO", "LL", "NG"}


def parse_gauge_out1(line: str) -> Optional[tuple[float, str]]:
    """解析测径仪返回字符串（只取 OUT1）。

    本项目常用：
    - 发送 "M1,0\r" -> 回 "M1,<value>\r"（仅测量值）
    - 发送 "M1,1\r" -> 回 "M1,<value>,<judge>\r"（测量值 + 鉴别）

    也兼容：
    - 发送 "M0,1\r" -> 回 "M0,<v1>,<j1>,<v2>,<j2>\r"（OUT1+OUT2）

    返回： (od_mm, judge_str)。若无鉴别字段，则 judge="UNK"。
    """
    if not isinstance(line, str):
        return None
    s = (line or "").strip()
    if not s:
        return None

    parts = [p.strip() for p in s.split(",")]
    if not parts:
        return None

    head = parts[0].upper()
    if head not in ("M1", "M0", "M2"):
        # some firmwares may omit the prefix; in this app we keep strict to avoid outliers
        return None

    if len(parts) < 2:
        return None

    # OUT1 value is always the first numeric field after the head
    try:
        od = float(parts[1])
    except Exception:
        return None

    judge = "UNK"
    if len(parts) >= 3:
        j = (parts[2] or "").strip().upper()
        if j in _JUDGE_SET:
            judge = j

    return float(od), judge


_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


class GaugeWorker(threading.Thread):
    """Serial reader for gauge (request/response friendly)."""

    def __init__(self, ui_q: queue.Queue):
        super().__init__(daemon=True)
        self.ui_q = ui_q
        self.stop_event = threading.Event()
        self._lock = threading.Lock()

        # runtime state
        self.enabled: bool = False
        self.port: str = ""
        self.baud: int = 9600
        self.timeout_s: float = 0.5
        self.eol: str = "\r"  # expected line terminator (CR)
        # default to include discrimination (GO/HI/LO...) for future length-edge detection
        self.request_cmd: str = "M1,1"  # without/with \r both ok

        # serial settings (default 8N1)
        self.bytesize: int = 8
        self.parity: str = "N"
        self.stopbits: int = 1

        self._ser: Optional["serial.Serial"] = None
        self.last: Optional[GaugeSample] = None

    def stop(self):
        self.stop_event.set()
        self._close()

    @property
    def is_connected(self) -> bool:
        try:
            return bool(self._ser) and bool(self._ser.is_open)
        except Exception:
            return False

    def configure(
        self,
        enabled: bool,
        port: str,
        baud: int,
        timeout_s: float,
        eol: str,
        request_cmd: str,
        bytesize: int = 8,
        parity: str = "N",
        stopbits: int = 1,
    ):
        """Update configuration.

        注意：不会无条件 close/open。只有端口/波特率等变化时才会重连。
        """
        enabled = bool(enabled)
        port = (port or "").strip()

        need_reopen = False
        if port != self.port or int(baud) != int(self.baud):
            need_reopen = True
        if (
            int(bytesize) != int(self.bytesize)
            or str(parity) != str(self.parity)
            or int(stopbits) != int(self.stopbits)
        ):
            need_reopen = True

        self.enabled = enabled
        self.port = port
        self.baud = int(baud)
        self.timeout_s = float(timeout_s)
        self.eol = eol or "\r"
        # Default to include discrimination result (M1,1) so the app can use GO/HI/LO for edge detection.
        self.request_cmd = (request_cmd or "M1,1").strip()
        self.bytesize = int(bytesize)
        self.parity = str(parity)
        self.stopbits = int(stopbits)

        if not self.enabled:
            self._close()
            return

        if need_reopen:
            self._close()
        self._ensure_open()

    def _ensure_open(self):
        if serial is None:
            raise RuntimeError(
                "pyserial 未安装，无法使用真实串口。请 pip install pyserial 或勾选“模拟测径仪”。"
            )
        if not self.port:
            raise ValueError("串口号为空")

        if self._ser is not None:
            try:
                if self._ser.is_open:
                    return
            except Exception:
                pass

        self._ser = serial.Serial(
            port=self.port,
            baudrate=self.baud,
            timeout=self.timeout_s,
            bytesize=self.bytesize,
            parity=self.parity,
            stopbits=self.stopbits,
        )

        try:
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
        except Exception:
            pass

        self.ui_q.put(
            (
                "gauge_conn",
                {
                    "ts": time.time(),
                    "connected": True,
                    "port": self.port,
                    "baud": self.baud,
                },
            )
        )

    def _close(self):
        if self._ser is None:
            return
        try:
            if getattr(self._ser, "is_open", False):
                self._ser.close()
        except Exception:
            pass
        finally:
            self._ser = None
            self.ui_q.put(("gauge_conn", {"ts": time.time(), "connected": False}))

    def _parse_line(self, line: str) -> Optional[tuple[float, str]]:
        """Strict parse: accept 'M1,<float>' or 'M1,<float>,<judge>' responses.

        NOTE:
        - We intentionally do NOT fallback to 'first float' parsing, because partial/garbled
          serial frames can otherwise create extreme outliers (e.g. 'M1,+' -> 1.0).
        """
        line = (line or "").strip()
        parsed = parse_gauge_out1(line)
        if parsed is None:
            return None
        od, judge = parsed
        # basic sanity
        if not math.isfinite(float(od)):
            return None
        return float(od), str(judge or "UNK")

    def send_request(self):
        """Send request command once (non-blocking)."""
        if not self.enabled:
            self.ui_q.put(
                ("gauge_err", {"ts": time.time(), "err": "gauge not enabled"})
            )
            return

        self._ensure_open()

        if not self._ser or not self.request_cmd:
            return

        try:
            cmd = self.request_cmd
            if not cmd.endswith("\r"):
                cmd = cmd + "\r"
            payload = cmd.encode("ascii", errors="ignore")

            try:
                self._ser.reset_input_buffer()
            except Exception:
                pass

            self._ser.write(payload)
            try:
                self._ser.flush()
            except Exception:
                pass

            self.ui_q.put(("gauge_tx", {"ts": time.time(), "cmd": cmd.strip()}))
        except Exception as e:
            self.ui_q.put(
                ("gauge_err", {"ts": time.time(), "err": f"gauge write failed: {e}"})
            )

    def get_last(self) -> Optional[GaugeSample]:
        with self._lock:
            return self.last

    def run(self):
        while not self.stop_event.is_set():
            try:
                if not self.enabled:
                    time.sleep(0.2)
                    continue

                self._ensure_open()

                raw = self._ser.read_until(b"\r") if self._ser else b""
                if not raw:
                    continue
                # If CR delimiter was not received before timeout, discard partial frame
                if not raw.endswith(b"\r"):
                    continue

                try:
                    line = raw.decode("ascii", errors="ignore").strip()
                except Exception:
                    line = ""

                parsed = self._parse_line(line)
                if parsed is None:
                    self.ui_q.put(("gauge_raw", {"ts": time.time(), "raw": line}))
                    continue

                od, judge = parsed
                s = GaugeSample(ts=time.time(), od=float(od), judge=str(judge or "UNK"), raw=line)
                with self._lock:
                    self.last = s

                self.ui_q.put(
                    (
                        "gauge_ok",
                        {"ts": s.ts, "od": float(od), "judge": s.judge, "raw": line},
                    )
                )

            except Exception as e:
                try:
                    with self._lock:
                        self.enabled = False
                except Exception:
                    self.enabled = False

                self._close()
                self.ui_q.put(("gauge_err", {"ts": time.time(), "err": str(e)}))
                time.sleep(0.5)


def list_serial_ports() -> list[str]:
    """Return list of available serial ports."""
    if serial is None:
        return []
    try:
        ports = [p.device for p in serial.tools.list_ports.comports()]
        return sorted(set(ports), key=lambda s: (len(s), s))
    except Exception:
        return []
