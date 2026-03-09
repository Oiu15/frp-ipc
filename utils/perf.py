from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TimeStat:
    n: int = 0
    sum_ns: int = 0
    max_ns: int = 0


@dataclass
class ValueStat:
    n: int = 0
    sum_v: float = 0.0
    max_v: float = float("-inf")


@dataclass
class PerfSnapshot:
    elapsed_s: float
    counts: Dict[str, int]
    times: Dict[str, TimeStat]
    values: Dict[str, ValueStat]


def ns_to_ms(ns: int) -> float:
    return float(ns) / 1_000_000.0


class PerfAggregator:
    """Thread-safe, low-overhead metric aggregator for periodic logging."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = {}
        self._times: Dict[str, TimeStat] = {}
        self._values: Dict[str, ValueStat] = {}
        self._last_flush_ns: int = time.perf_counter_ns()

    def add_count(self, key: str, n: int = 1) -> None:
        k = str(key)
        v = int(n)
        with self._lock:
            self._counts[k] = int(self._counts.get(k, 0)) + v

    def add_time_ns(self, key: str, dt_ns: int) -> None:
        k = str(key)
        dt = max(0, int(dt_ns))
        with self._lock:
            st = self._times.get(k)
            if st is None:
                self._times[k] = TimeStat(n=1, sum_ns=dt, max_ns=dt)
            else:
                st.n += 1
                st.sum_ns += dt
                if dt > st.max_ns:
                    st.max_ns = dt

    def add_value(self, key: str, value: float) -> None:
        k = str(key)
        vv = float(value)
        with self._lock:
            st = self._values.get(k)
            if st is None:
                self._values[k] = ValueStat(n=1, sum_v=vv, max_v=vv)
            else:
                st.n += 1
                st.sum_v += vv
                if vv > st.max_v:
                    st.max_v = vv

    def _drain_locked(self, now_ns: int) -> PerfSnapshot:
        elapsed_s = max(0.0, float(now_ns - self._last_flush_ns) / 1_000_000_000.0)
        counts = self._counts
        times = self._times
        values = self._values
        self._counts = {}
        self._times = {}
        self._values = {}
        self._last_flush_ns = now_ns
        return PerfSnapshot(
            elapsed_s=elapsed_s,
            counts=counts,
            times=times,
            values=values,
        )

    def drain_if_due(self, every_s: float = 1.0, *, force: bool = False) -> Optional[PerfSnapshot]:
        every_ns = max(1, int(float(every_s) * 1_000_000_000.0))
        now_ns = time.perf_counter_ns()
        with self._lock:
            if (not force) and (now_ns - self._last_flush_ns) < every_ns:
                return None
            return self._drain_locked(now_ns)

    @staticmethod
    def _fmt(v: Any) -> str:
        try:
            if isinstance(v, float):
                return f"{float(v):.3f}"
            return str(v)
        except Exception:
            return "nan"

    def flush_if_due(
        self,
        logger,
        tag: str,
        *,
        every_s: float = 1.0,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> bool:
        snap = self.drain_if_due(every_s=every_s)
        if snap is None:
            return False
        parts = [f"dt_s={self._fmt(snap.elapsed_s)}"]
        for k in sorted(snap.counts):
            parts.append(f"{k}={snap.counts[k]}")
        for k in sorted(snap.times):
            st = snap.times[k]
            if st.n <= 0:
                continue
            avg_ms = ns_to_ms(int(st.sum_ns)) / float(st.n)
            parts.append(f"{k}_n={int(st.n)}")
            parts.append(f"{k}_avg_ms={self._fmt(avg_ms)}")
            parts.append(f"{k}_max_ms={self._fmt(ns_to_ms(int(st.max_ns)))}")
        for k in sorted(snap.values):
            st = snap.values[k]
            if st.n <= 0:
                continue
            avg_v = float(st.sum_v) / float(st.n)
            parts.append(f"{k}_n={int(st.n)}")
            parts.append(f"{k}_avg={self._fmt(avg_v)}")
            parts.append(f"{k}_max={self._fmt(st.max_v)}")
        if extra_fields:
            for k, v in extra_fields.items():
                parts.append(f"{k}={self._fmt(v)}")
        try:
            logger.info("%s %s", str(tag), " ".join(parts))
        except Exception:
            return False
        return True
