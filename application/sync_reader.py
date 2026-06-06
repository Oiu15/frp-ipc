from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from drivers.plc_client import CmdReadRegs


logger = logging.getLogger("frp.modbus")


class PlcSyncReader:
    """Synchronous PLC read boundary backed by PlcWorker one-shot reads."""

    def __init__(
        self,
        *,
        cmd_q: Any,
        sync_reads: dict[str, dict[str, Any]],
        sync_reads_lock: threading.Lock,
        perf_sync_read: Any,
        perf_ui_queue: Any,
        perf_group: Callable[[int, int], str],
        flush_perf: Callable[[], None],
    ) -> None:
        self.cmd_q = cmd_q
        self.sync_reads = sync_reads
        self.sync_reads_lock = sync_reads_lock
        self.perf_sync_read = perf_sync_read
        self.perf_ui_queue = perf_ui_queue
        self.perf_group = perf_group
        self.flush_perf = flush_perf

    def read_regs_sync(self, d_addr: int, count: int, timeout_s: float = 0.35) -> list[int] | None:
        t_total0_ns = time.perf_counter_ns()
        perf_cat = self.perf_group(int(d_addr), int(count))
        self.perf_sync_read.add_count(f"{perf_cat}.n", 1)
        tag = f"sync:{time.time_ns()}"
        evt = threading.Event()
        with self.sync_reads_lock:
            self.sync_reads[tag] = {
                "evt": evt,
                "regs": None,
                "perf_cat": perf_cat,
            }
        try:
            t_put0_ns = time.perf_counter_ns()
            self.cmd_q.put(CmdReadRegs(d_addr, int(count), tag))
            self.perf_sync_read.add_time_ns(f"{perf_cat}.put_cmd", time.perf_counter_ns() - t_put0_ns)
        except Exception:
            with self.sync_reads_lock:
                self.sync_reads.pop(tag, None)
            self._record_timeout(perf_cat, t_total0_ns)
            return None

        t_wait0_ns = time.perf_counter_ns()
        wait_ok = bool(evt.wait(float(timeout_s)))
        self.perf_sync_read.add_time_ns(f"{perf_cat}.wait_evt", time.perf_counter_ns() - t_wait0_ns)
        if not wait_ok:
            try:
                logger.debug(
                    "SYNC_READ_TIMEOUT d_addr=%s count=%s timeout_s=%.3f",
                    d_addr,
                    count,
                    float(timeout_s),
                )
            except Exception:
                pass
            with self.sync_reads_lock:
                self.sync_reads.pop(tag, None)
            self._record_timeout(perf_cat, t_total0_ns)
            return None

        with self.sync_reads_lock:
            slot = self.sync_reads.pop(tag, None)
        if not slot:
            self._record_timeout(perf_cat, t_total0_ns)
            return None
        regs = slot.get("regs", None)
        try:
            if regs is not None:
                logger.debug("SYNC_READ_OK d_addr=%s count=%s", d_addr, count)
        except Exception:
            pass
        if regs is None:
            self._record_timeout(perf_cat, t_total0_ns)
            return None
        try:
            return list(regs)
        except Exception:
            return None
        finally:
            self.perf_sync_read.add_time_ns(f"{perf_cat}.total", time.perf_counter_ns() - t_total0_ns)
            self.flush_perf()

    def handle_plc_read_payload(self, payload: dict[str, Any]) -> bool:
        tag = payload.get("tag", "")
        if not (isinstance(tag, str) and tag.startswith("sync:")):
            return False

        regs = payload.get("regs", [])
        now_ns = time.perf_counter_ns()
        try:
            t_uiq_put_ns = int(payload.get("t_uiq_put_ns", 0) or 0)
            if t_uiq_put_ns > 0:
                self.perf_ui_queue.add_time_ns("evt_delay", now_ns - t_uiq_put_ns)
        except Exception:
            pass
        try:
            with self.sync_reads_lock:
                slot = self.sync_reads.get(tag, None)
                if slot is not None:
                    slot["regs"] = list(regs)
                    try:
                        perf_cat = str(slot.get("perf_cat", "other") or "other")
                        t_uiq_put_ns = int(payload.get("t_uiq_put_ns", 0) or 0)
                        if t_uiq_put_ns > 0:
                            self.perf_sync_read.add_time_ns(
                                f"{perf_cat}.evt_delay",
                                now_ns - t_uiq_put_ns,
                            )
                    except Exception:
                        pass
                    try:
                        slot["evt"].set()
                    except Exception:
                        pass
        except Exception:
            pass
        return True

    def _record_timeout(self, perf_cat: str, t_total0_ns: int) -> None:
        self.perf_sync_read.add_count(f"{perf_cat}.timeout", 1)
        self.perf_sync_read.add_time_ns(f"{perf_cat}.total", time.perf_counter_ns() - t_total0_ns)
        self.flush_perf()


__all__ = ["PlcSyncReader"]
