from __future__ import annotations

import logging
import queue
import time
from typing import Any, cast

import pytest

from drivers.plc_client import CmdSetPollProfile, PlcWorker


class _ReadResult:
    def __init__(self, count: int = 0, *, bits: bool = False) -> None:
        self.registers = [0] * int(count)
        self.bits = [False] * int(count) if bits else []

    def isError(self) -> bool:
        return False


class _FakeModbusClient:
    def read_holding_registers(self, _address: int, *, count: int, device_id: int) -> _ReadResult:
        del device_id
        return _ReadResult(count)

    def read_coils(self, _address: int, *, count: int, device_id: int) -> _ReadResult:
        del device_id
        return _ReadResult(count, bits=True)

    def close(self) -> None:
        pass


def _connected_worker(*, poll_interval_s: float = 0.0) -> PlcWorker:
    worker = PlcWorker(queue.Queue(), queue.Queue(), poll_interval_s=poll_interval_s, connect_on_start=False)
    worker._client = cast(Any, _FakeModbusClient())
    worker._connected = True
    return worker


class TestPlcPollTiming:
    def test_normal_profile_records_full_poll_timing_without_sleep_in_active_total(self) -> None:
        worker = _connected_worker(poll_interval_s=0.02)
        worker.ui_events.publish_plc_ok = lambda **_payload: worker.stop()  # type: ignore[method-assign]

        started = time.perf_counter()
        worker.run()
        elapsed_s = time.perf_counter() - started
        snap = worker._perf.drain_if_due(force=True)
        assert snap is not None

        assert "poll.normal.axis_total" in snap.times
        assert "poll.normal.axis_ax0" in snap.times
        assert "poll.normal.axis_ax4" in snap.times
        assert "poll.normal.cl_measurements" in snap.times
        assert "poll.normal.cl_counters" in snap.times
        assert "poll.normal.keytest_x" in snap.times
        assert "poll.normal.keytest_y" in snap.times
        assert "poll.normal.poll_active_total" in snap.times
        assert "loop_total" in snap.times
        assert "poll.sampling.poll_active_total" not in snap.times
        assert elapsed_s >= 0.018
        assert (
            snap.times["loop_total"].max_ns - snap.times["poll.normal.poll_active_total"].max_ns
        ) > 10_000_000

    def test_sampling_profile_records_only_executed_background_reads(self) -> None:
        worker = _connected_worker()
        worker.cmd_q.put(CmdSetPollProfile("sampling"))
        worker.ui_events.publish_plc_ok = lambda **_payload: worker.stop()  # type: ignore[method-assign]

        worker.run()
        snap = worker._perf.drain_if_due(force=True)
        assert snap is not None

        assert "poll.sampling.axis_total" in snap.times
        assert "poll.sampling.axis_ax3" in snap.times
        assert "poll.sampling.keytest_x" in snap.times
        assert "poll.sampling.axis_ax0" not in snap.times
        assert "poll.sampling.cl_measurements" not in snap.times
        assert "poll.sampling.cl_counters" not in snap.times
        assert "poll.sampling.keytest_y" not in snap.times

    def test_flush_logs_separate_lines_for_profiles_seen_in_one_window(self, caplog: pytest.LogCaptureFixture) -> None:
        worker = _connected_worker()
        poll_count = 0

        def after_poll(**_payload) -> None:
            nonlocal poll_count
            poll_count += 1
            if poll_count == 1:
                worker.cmd_q.put(CmdSetPollProfile("sampling"))
            else:
                worker.stop()

        worker.ui_events.publish_plc_ok = after_poll  # type: ignore[method-assign]
        worker.run()
        worker._perf._last_flush_ns = 0

        with caplog.at_level(logging.INFO, logger="frp.plc.perf"):
            worker._flush_perf_if_due()

        logs = "\n".join(caplog.messages)
        normal_line = next(line for line in caplog.text.splitlines() if "[PLC_POLL_TIMING][NORMAL]" in line)
        sampling_line = next(line for line in caplog.text.splitlines() if "[PLC_POLL_TIMING][SAMPLING]" in line)
        assert "[PLC_WORKER_PERF]" in logs
        assert "cl_measurements_avg_ms=" in normal_line
        assert "keytest_y_avg_ms=" in normal_line
        assert "axis_ax3_avg_ms=" in sampling_line
        assert "keytest_x_avg_ms=" in sampling_line
        assert "cl_measurements_avg_ms=" not in sampling_line
        assert "keytest_y_avg_ms=" not in sampling_line
