# ./frp_app/drivers/gauge_driver.py
from __future__ import annotations

"""测径仪串口通信（外径 OD）。

设计目标：
- 连接稳定：重复点击“连接”不会反复 open/close 串口，避免 Windows PermissionError(13)
- 请求/响应友好：上层可调用 send_request()，后台线程异步读取并解析
 - 对协议更稳健：严格解析完整帧（M1/M0 + 数值 [+ 鉴别]），拒绝半帧/乱码，避免极端离群值
"""

import queue
import logging
import re
import threading
import time
import math
from dataclasses import dataclass
from typing import Any, Optional
from utils.perf import PerfAggregator, ns_to_ms

from application.ui_queue_adapters import WorkerUiEventAdapter

try:
    import serial  # type: ignore
    import serial.tools.list_ports  # type: ignore
    from serial import Serial  # type: ignore
except Exception:  # pragma: no cover
    # Allow the app to run in "模拟测径仪" mode without pyserial installed.
    serial = None  # type: ignore
    Serial = object  # type: ignore

from core.models import GaugeSample

logger = logging.getLogger("frp.gauge")
perf_logger = logging.getLogger("frp.gauge.perf")


_JUDGE_SET = {"HH", "HI", "GO", "LO", "LL", "NG"}


def _is_judge_token(s: str) -> bool:
    try:
        return str(s or "").strip().upper() in _JUDGE_SET
    except Exception:
        return False


def parse_gauge_line(line: str) -> Optional[tuple[float, str, Optional[float], str]]:
    """解析测径仪返回字符串（OUT1 + 可选 OUT2）。

    常见格式：
    - M1,<v1>\r
    - M1,<v1>,<j1>\r
    - M0,<v1>,<j1>,<v2>,<j2>\r
    - M0,<v1>,<v2>\r  (部分固件在 r=0 时无鉴别字段)

    返回：(v1, j1, v2, j2)。当无 OUT2 时 v2=None。
    """
    if not isinstance(line, str):
        return None
    s = (line or "").strip()
    if not s:
        return None

    parts = [p.strip() for p in s.split(",") if p is not None]
    if len(parts) < 2:
        return None

    head = (parts[0] or "").strip().upper()
    if head not in ("M0", "M1", "M2"):
        # strict mode: reject frames without known prefix to avoid outliers
        return None

    # OUT1
    try:
        v1 = float(parts[1])
    except Exception:
        return None
    j1 = "UNK"
    idx = 2
    if idx < len(parts) and _is_judge_token(parts[idx]):
        j1 = str(parts[idx]).strip().upper()
        idx += 1

    # OUT2 (only for M0 output)
    v2: Optional[float] = None
    j2: str = "UNK"
    if head == "M0":
        if idx >= len(parts):
            return None
        try:
            v2 = float(parts[idx])
        except Exception:
            return None
        idx += 1
        if idx < len(parts) and _is_judge_token(parts[idx]):
            j2 = str(parts[idx]).strip().upper()

    return float(v1), str(j1 or "UNK"), (float(v2) if v2 is not None else None), str(j2 or "UNK")


_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


class GaugeWorker(threading.Thread):
    """Serial reader for gauge (request/response friendly)."""

    def __init__(self, ui_q: queue.Queue):
        super().__init__(daemon=True)
        self.ui_q = ui_q
        self.ui_events = WorkerUiEventAdapter(ui_q)
        self.stop_event = threading.Event()
        self._lock = threading.Lock()

        # runtime state
        self.enabled: bool = False
        self.port: str = ""
        self.baud: int = 115200
        self.timeout_s: float = 0.5
        self.eol: str = "\r"  # expected line terminator (CR)
        # default to include discrimination (GO/HI/LO...) for future length-edge detection
        self.request_cmd: str = "M1,1"  # without/with \r both ok

        # serial settings (default 8N1)
        self.bytesize: int = 8
        self.parity: str = "N"
        self.stopbits: int = 1

        # NOTE: do NOT annotate with "serial.Serial" because we intentionally
        # allow `serial=None` when pyserial isn't installed (模拟测径仪模式).
        # Using the imported `Serial` symbol keeps type checkers happy.
        self._ser: Any | None = None
        self.last: Optional[GaugeSample] = None
        self._perf = PerfAggregator()
        self._last_seq: int = 0
        self._last_consumed_seq: int = 0

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

        ser = serial.Serial(
            port=self.port,
            baudrate=self.baud,
            timeout=self.timeout_s,
            bytesize=self.bytesize,
            parity=self.parity,
            stopbits=self.stopbits,
        )
        self._ser = ser
        try:
            logger.info("GAUGE_CONNECT port=%s baud=%s", self.port, self.baud)
        except Exception:
            pass

        try:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            pass

        self.ui_events.publish_gauge_conn(
            ts=time.time(),
            connected=True,
            port=self.port,
            baud=self.baud,
        )

    def _close(self):
        if self._ser is None:
            return
        try:
            if getattr(self._ser, "is_open", False):
                self._ser.close()
                try:
                    logger.info("GAUGE_DISCONNECT port=%s", self.port)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self._ser = None
            self.ui_events.publish_gauge_conn(ts=time.time(), connected=False)

    def _parse_line(self, line: str) -> Optional[tuple[float, str, Optional[float], str]]:
        """Strict parse: accept M1/M0 responses.

        NOTE:
        - We intentionally do NOT fallback to 'first float' parsing, because partial/garbled
          serial frames can otherwise create extreme outliers (e.g. 'M1,+' -> 1.0).
        """
        line = (line or "").strip()
        parsed = parse_gauge_line(line)
        if parsed is None:
            return None
        v1, j1, v2, j2 = parsed
        if not math.isfinite(float(v1)):
            return None
        if v2 is not None and (not math.isfinite(float(v2))):
            return None
        return float(v1), str(j1 or "UNK"), (float(v2) if v2 is not None else None), str(j2 or "UNK")

    def send_request(self):
        """Send request command once (non-blocking)."""
        t_send0_ns = time.perf_counter_ns()
        self._perf.add_count("send_request", 1)
        if not self.enabled:
            self.ui_events.publish_gauge_err(ts=time.time(), err="gauge not enabled")
            self._perf.add_time_ns("send_request", time.perf_counter_ns() - t_send0_ns)
            self._flush_perf_if_due()
            return

        self._ensure_open()

        if not self._ser or not self.request_cmd:
            self._perf.add_time_ns("send_request", time.perf_counter_ns() - t_send0_ns)
            self._flush_perf_if_due()
            return

        try:
            cmd = self.request_cmd
            if not cmd.endswith("\r"):
                cmd = cmd + "\r"
            payload = cmd.encode("ascii", errors="ignore")

            try:
                t_rib0_ns = time.perf_counter_ns()
                self._ser.reset_input_buffer()
                self._perf.add_time_ns("reset_input_buffer", time.perf_counter_ns() - t_rib0_ns)
            except Exception:
                pass

            self._ser.write(payload)
            try:
                self._ser.flush()
            except Exception:
                pass

            try:
                logger.debug("GAUGE_TX cmd=%s", cmd.strip())
            except Exception:
                pass
            self.ui_events.publish_gauge_tx(ts=time.time(), cmd=cmd.strip())
        except Exception as e:
            try:
                logger.exception("GAUGE_TX_ERR")
            except Exception:
                pass
            self.ui_events.publish_gauge_err(ts=time.time(), err=f"gauge write failed: {e}")
        finally:
            self._perf.add_time_ns("send_request", time.perf_counter_ns() - t_send0_ns)
            self._flush_perf_if_due()

    def get_last(self) -> Optional[GaugeSample]:
        with self._lock:
            s = self.last
            if s is not None:
                self._last_consumed_seq = int(self._last_seq)
            return s

    def _flush_perf_if_due(self) -> None:
        snap = self._perf.drain_if_due(every_s=1.0)
        if snap is None:
            return
        c = snap.counts
        t = snap.times

        def _time_avg_max_ms(key: str) -> tuple[float, float]:
            st = t.get(key)
            if st is None or st.n <= 0:
                return 0.0, 0.0
            return (ns_to_ms(int(st.sum_ns)) / float(st.n), ns_to_ms(int(st.max_ns)))

        send_avg_ms, send_max_ms = _time_avg_max_ms("send_request")
        rib_avg_ms, rib_max_ms = _time_avg_max_ms("reset_input_buffer")
        read_avg_ms, read_max_ms = _time_avg_max_ms("read_until")

        ok = int(c.get("parsed_ok", 0))
        frames_s = (float(ok) / float(snap.elapsed_s)) if snap.elapsed_s > 1e-9 else 0.0
        try:
            perf_logger.info(
                "[GAUGE_PERF] send=%d send_avg_ms=%.3f send_max_ms=%.3f "
                "rib_avg_ms=%.3f rib_max_ms=%.3f read_calls=%d ok=%d bad=%d timeout=%d "
                "read_avg_ms=%.3f read_max_ms=%.3f frames_s=%.1f latest_overwrite=%d overwrite_unconsumed=%d",
                int(c.get("send_request", 0)),
                float(send_avg_ms),
                float(send_max_ms),
                float(rib_avg_ms),
                float(rib_max_ms),
                int(c.get("read_until", 0)),
                ok,
                int(c.get("parsed_bad", 0)),
                int(c.get("timeout_count", 0)),
                float(read_avg_ms),
                float(read_max_ms),
                float(frames_s),
                int(c.get("latest_overwrite", 0)),
                int(c.get("overwrite_unconsumed", 0)),
            )
        except Exception:
            pass

    def run(self):
        while not self.stop_event.is_set():
            try:
                self._flush_perf_if_due()
                if not self.enabled:
                    time.sleep(0.2)
                    continue

                self._ensure_open()

                t_read0_ns = time.perf_counter_ns()
                raw = self._ser.read_until(b"\r") if self._ser else b""
                self._perf.add_count("read_until", 1)
                self._perf.add_time_ns("read_until", time.perf_counter_ns() - t_read0_ns)
                if not raw:
                    self._perf.add_count("timeout_count", 1)
                    continue
                # If CR delimiter was not received before timeout, discard partial frame
                if not raw.endswith(b"\r"):
                    self._perf.add_count("parsed_bad", 1)
                    continue

                try:
                    line = raw.decode("ascii", errors="ignore").strip()
                except Exception:
                    line = ""

                parsed = self._parse_line(line)
                if parsed is None:
                    self._perf.add_count("parsed_bad", 1)
                    try:
                        logger.debug("GAUGE_RAW_UNPARSED raw=%s", line)
                    except Exception:
                        pass
                    self.ui_events.publish_gauge_raw(ts=time.time(), raw=line)
                    continue

                od1, j1, od2, j2 = parsed
                s = GaugeSample(
                    ts=time.time(),
                    od=float(od1),
                    judge=str(j1 or "UNK"),
                    od2=(float(od2) if od2 is not None else None),
                    judge2=str(j2 or "UNK"),
                    raw=line,
                )
                with self._lock:
                    if self.last is not None:
                        self._perf.add_count("latest_overwrite", 1)
                        if int(self._last_consumed_seq) < int(self._last_seq):
                            self._perf.add_count("overwrite_unconsumed", 1)
                    self._last_seq += 1
                    self.last = s

                self._perf.add_count("parsed_ok", 1)
                self.ui_events.publish_gauge_ok(
                    ts=s.ts,
                    od=float(s.od),
                    judge=s.judge,
                    od2=(float(s.od2) if s.od2 is not None else None),
                    judge2=str(s.judge2 or "UNK"),
                    raw=line,
                )

            except Exception as e:
                try:
                    logger.exception("GAUGE_LOOP_ERR")
                except Exception:
                    pass
                try:
                    with self._lock:
                        self.enabled = False
                except Exception:
                    self.enabled = False

                self._close()
                self.ui_events.publish_gauge_err(ts=time.time(), err=str(e))
                time.sleep(0.5)
            finally:
                self._flush_perf_if_due()


def list_serial_ports() -> list[str]:
    """Return list of available serial ports."""
    if serial is None:
        return []
    try:
        ports = [p.device for p in serial.tools.list_ports.comports()]
        return sorted(set(ports), key=lambda s: (len(s), s))
    except Exception:
        return []
