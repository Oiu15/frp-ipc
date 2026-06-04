import queue
import sys
import time
import types
import unittest
from typing import Any, cast

_pymodbus = types.ModuleType("pymodbus")
_pymodbus_client = types.ModuleType("pymodbus.client")


class _ImportOnlyModbusTcpClient:
    pass


setattr(_pymodbus_client, "ModbusTcpClient", _ImportOnlyModbusTcpClient)
setattr(_pymodbus, "client", _pymodbus_client)
sys.modules.setdefault("pymodbus", _pymodbus)
sys.modules.setdefault("pymodbus.client", _pymodbus_client)

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


class PlcPollTimingTest(unittest.TestCase):
    def test_normal_profile_records_full_poll_timing_without_sleep_in_active_total(self) -> None:
        worker = _connected_worker(poll_interval_s=0.02)
        worker.ui_events.publish_plc_ok = lambda **_payload: worker.stop()  # type: ignore[method-assign]

        started = time.perf_counter()
        worker.run()
        elapsed_s = time.perf_counter() - started
        snap = worker._perf.drain_if_due(force=True)
        assert snap is not None

        self.assertIn("poll.normal.axis_total", snap.times)
        self.assertIn("poll.normal.axis_ax0", snap.times)
        self.assertIn("poll.normal.axis_ax4", snap.times)
        self.assertIn("poll.normal.cl_measurements", snap.times)
        self.assertIn("poll.normal.cl_counters", snap.times)
        self.assertIn("poll.normal.keytest_x", snap.times)
        self.assertIn("poll.normal.keytest_y", snap.times)
        self.assertIn("poll.normal.poll_active_total", snap.times)
        self.assertIn("loop_total", snap.times)
        self.assertNotIn("poll.sampling.poll_active_total", snap.times)
        self.assertGreaterEqual(elapsed_s, 0.018)
        self.assertGreater(
            snap.times["loop_total"].max_ns - snap.times["poll.normal.poll_active_total"].max_ns,
            10_000_000,
        )

    def test_sampling_profile_records_only_executed_background_reads(self) -> None:
        worker = _connected_worker()
        worker.cmd_q.put(CmdSetPollProfile("sampling"))
        worker.ui_events.publish_plc_ok = lambda **_payload: worker.stop()  # type: ignore[method-assign]

        worker.run()
        snap = worker._perf.drain_if_due(force=True)
        assert snap is not None

        self.assertIn("poll.sampling.axis_total", snap.times)
        self.assertIn("poll.sampling.axis_ax3", snap.times)
        self.assertIn("poll.sampling.keytest_x", snap.times)
        self.assertNotIn("poll.sampling.axis_ax0", snap.times)
        self.assertNotIn("poll.sampling.cl_measurements", snap.times)
        self.assertNotIn("poll.sampling.cl_counters", snap.times)
        self.assertNotIn("poll.sampling.keytest_y", snap.times)

    def test_flush_logs_separate_lines_for_profiles_seen_in_one_window(self) -> None:
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

        with self.assertLogs("frp.plc.perf", level="INFO") as captured:
            worker._flush_perf_if_due()

        logs = "\n".join(captured.output)
        normal_line = next(line for line in captured.output if "[PLC_POLL_TIMING][NORMAL]" in line)
        sampling_line = next(line for line in captured.output if "[PLC_POLL_TIMING][SAMPLING]" in line)
        self.assertIn("[PLC_WORKER_PERF]", logs)
        self.assertIn("cl_measurements_avg_ms=", normal_line)
        self.assertIn("keytest_y_avg_ms=", normal_line)
        self.assertIn("axis_ax3_avg_ms=", sampling_line)
        self.assertIn("keytest_x_avg_ms=", sampling_line)
        self.assertNotIn("cl_measurements_avg_ms=", sampling_line)
        self.assertNotIn("keytest_y_avg_ms=", sampling_line)


if __name__ == "__main__":
    unittest.main()
