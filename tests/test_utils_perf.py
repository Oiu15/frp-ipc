"""Tests for utils/perf.py — PerfAggregator and helpers.

All logic is pure; no IO or threading mockery needed beyond what
the class itself provides (lock is internal).
"""

from __future__ import annotations

import time

from utils.perf import PerfAggregator, ns_to_ms


class TestNsToMs:
    def test_conversion(self) -> None:
        assert ns_to_ms(1_000_000) == 1.0
        assert ns_to_ms(500_000) == 0.5
        assert ns_to_ms(0) == 0.0


class TestPerfAggregator:
    """PerfAggregator: record timings → drain snapshot → flush log line."""

    @staticmethod
    def _new() -> PerfAggregator:
        return PerfAggregator()

    # ------------------------------------------------------------------
    # add_time_ns
    # ------------------------------------------------------------------

    def test_add_time_records_min_max_avg(self) -> None:
        agg = self._new()
        agg.add_time_ns("t", 100)
        agg.add_time_ns("t", 300)

        snap = agg.drain_if_due(force=True)
        assert snap is not None
        stat = snap.times["t"]
        assert stat.n == 2
        assert stat.sum_ns == 400
        assert stat.max_ns == 300

    def test_add_time_negative_is_clamped_to_zero(self) -> None:
        agg = self._new()
        agg.add_time_ns("t", -50)
        snap = agg.drain_if_due(force=True)
        assert snap is not None
        assert snap.times["t"].sum_ns == 0

    # ------------------------------------------------------------------
    # add_count
    # ------------------------------------------------------------------

    def test_add_count_increments(self) -> None:
        agg = self._new()
        agg.add_count("c", 3)
        agg.add_count("c", 2)
        snap = agg.drain_if_due(force=True)
        assert snap is not None
        assert snap.counts["c"] == 5

    # ------------------------------------------------------------------
    # add_value
    # ------------------------------------------------------------------

    def test_add_value_records_min_max_avg(self) -> None:
        agg = self._new()
        agg.add_value("v", 1.0)
        agg.add_value("v", 4.0)

        snap = agg.drain_if_due(force=True)
        assert snap is not None
        stat = snap.values["v"]
        assert stat.n == 2
        assert stat.sum_v == 5.0
        assert stat.max_v == 4.0

    # ------------------------------------------------------------------
    # drain_if_due
    # ------------------------------------------------------------------

    def test_drain_returns_none_before_interval(self) -> None:
        agg = self._new()
        agg.add_count("x")
        # force=False + default 1s interval → should not drain yet
        assert agg.drain_if_due() is None

    def test_force_drain_always_returns_snapshot(self) -> None:
        agg = self._new()
        agg.add_count("x")
        assert agg.drain_if_due(force=True) is not None

    def test_drain_clears_accumulated_data(self) -> None:
        agg = self._new()
        agg.add_count("x", 5)
        agg.drain_if_due(force=True)
        # after drain, data is reset
        snap = agg.drain_if_due(force=True)
        assert snap is not None
        assert "x" not in snap.counts

    # ------------------------------------------------------------------
    # drain returns a PerfSnapshot with elapsed time
    # ------------------------------------------------------------------

    def test_drain_sets_elapsed_s(self) -> None:
        agg = self._new()
        time.sleep(0.01)  # small delay so elapsed > 0
        snap = agg.drain_if_due(force=True)
        assert snap is not None
        assert snap.elapsed_s >= 0.0

    # ------------------------------------------------------------------
    # flush_if_due
    # ------------------------------------------------------------------

    def test_flush_returns_false_when_not_due(self) -> None:
        agg = self._new()
        assert agg.flush_if_due(logger=None, tag="test") is False  # type: ignore[arg-type]

    def test_flush_logs_timing_data(self) -> None:
        agg = self._new()
        agg.add_time_ns("t", 100)
        # force-drain first so the next drain_if_due call returns immediately
        agg.drain_if_due(force=True)
        # add more data, then set a very short interval
        agg.add_time_ns("t", 200)

        class _Log:
            def info(self, fmt: str, *args: object) -> None:
                pass

        assert agg.flush_if_due(logger=_Log(), tag="TAG", every_s=0.0) is True

    def test_flush_survives_logger_exception(self) -> None:
        agg = self._new()
        agg.add_time_ns("t", 100)
        agg.drain_if_due(force=True)
        agg.add_time_ns("t", 200)

        class _BrokenLog:
            def info(self, fmt: str, *args: object) -> None:
                raise RuntimeError("log failed")

        assert agg.flush_if_due(logger=_BrokenLog(), tag="TAG", every_s=0.0) is False
