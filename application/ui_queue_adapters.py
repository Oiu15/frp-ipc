from __future__ import annotations

import queue
from collections.abc import Mapping, Sequence
from typing import Any

from application.contracts import EventSink
from application.ui_events import UiEventBase
from core.models import AxisComm, MeasureRow


class UiQueueCompatAdapter:
    """Compatibility adapter for the existing ``ui_q`` tuple protocol.

    The queue payload shape is intentionally kept unchanged during migration:
    producers use explicit adapter methods, while consumers still receive
    ``(event_name, payload)`` tuples.
    """

    def __init__(self, ui_q: queue.Queue) -> None:
        self.ui_q = ui_q

    def publish_legacy(self, event_name: str, payload: Any) -> None:
        self.ui_q.put((str(event_name), payload))

    def publish_typed(self, event: UiEventBase) -> None:
        self.ui_q.put(event.to_ui_event())


class WorkerUiEventAdapter(UiQueueCompatAdapter):
    """Worker-side adapter that preserves the legacy UI queue payloads."""

    def publish_plc_manual(self, *, ip: str, port: int) -> None:
        self.publish_legacy('plc_manual', {'ip': str(ip), 'port': int(port)})

    def publish_plc_giveup(self, *, retry: int, max_tries: int) -> None:
        self.publish_legacy('plc_giveup', {'retry': int(retry), 'max': int(max_tries)})

    def publish_plc_err(
        self,
        *,
        err: str,
        retry: int | None = None,
        max_tries: int | None = None,
        backoff_s: float | None = None,
    ) -> None:
        payload: dict[str, Any] = {'err': str(err)}
        if retry is not None:
            payload['retry'] = int(retry)
        if max_tries is not None:
            payload['max'] = int(max_tries)
        if backoff_s is not None:
            payload['backoff_s'] = float(backoff_s)
        self.publish_legacy('plc_err', payload)

    def publish_plc_read(
        self,
        *,
        tag: str,
        d_addr: int,
        count: int,
        regs: Sequence[int],
        t_uiq_put_ns: int | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            'tag': str(tag),
            'd_addr': int(d_addr),
            'count': int(count),
            'regs': [int(v) for v in regs],
        }
        if t_uiq_put_ns is not None:
            payload['t_uiq_put_ns'] = int(t_uiq_put_ns)
        self.publish_legacy('plc_read', payload)

    def publish_plc_ok(
        self,
        *,
        axes: Sequence[AxisComm],
        cl_out1_raw: int | None = None,
        cl_out1_mm: float | None = None,
        cl_out1_cnt: int | None = None,
        cl_out2_raw: int | None = None,
        cl_out2_mm: float | None = None,
        cl_out2_cnt: int | None = None,
        cl_out3_raw: int | None = None,
        cl_out3_mm: float | None = None,
        cl_out3_cnt: int | None = None,
        cl_out4_raw: int | None = None,
        cl_out4_mm: float | None = None,
        cl_out4_cnt: int | None = None,
        cl_out5_raw: int | None = None,
        cl_out5_mm: float | None = None,
        cl_out5_cnt: int | None = None,
        keytest_x_bits: Sequence[int] | None = None,
        keytest_y_bits: Sequence[int] | None = None,
    ) -> None:
        self.publish_legacy(
            'plc_ok',
            {
                'axes': list(axes),
                'cl_out1_raw': cl_out1_raw,
                'cl_out1_mm': cl_out1_mm,
                'cl_out1_cnt': cl_out1_cnt,
                'cl_out2_raw': cl_out2_raw,
                'cl_out2_mm': cl_out2_mm,
                'cl_out2_cnt': cl_out2_cnt,
                'cl_out3_raw': cl_out3_raw,
                'cl_out3_mm': cl_out3_mm,
                'cl_out3_cnt': cl_out3_cnt,
                'cl_out4_raw': cl_out4_raw,
                'cl_out4_mm': cl_out4_mm,
                'cl_out4_cnt': cl_out4_cnt,
                'cl_out5_raw': cl_out5_raw,
                'cl_out5_mm': cl_out5_mm,
                'cl_out5_cnt': cl_out5_cnt,
                'keytest_x_bits': list(keytest_x_bits) if keytest_x_bits is not None else None,
                'keytest_y_bits': list(keytest_y_bits) if keytest_y_bits is not None else None,
            },
        )

    def publish_gauge_conn(
        self,
        *,
        ts: float,
        connected: bool,
        port: str | None = None,
        baud: int | None = None,
    ) -> None:
        payload: dict[str, Any] = {'ts': float(ts), 'connected': bool(connected)}
        if port is not None:
            payload['port'] = str(port)
        if baud is not None:
            payload['baud'] = int(baud)
        self.publish_legacy('gauge_conn', payload)

    def publish_gauge_tx(self, *, ts: float, cmd: str) -> None:
        self.publish_legacy('gauge_tx', {'ts': float(ts), 'cmd': str(cmd)})

    def publish_gauge_raw(self, *, ts: float, raw: str) -> None:
        self.publish_legacy('gauge_raw', {'ts': float(ts), 'raw': str(raw)})

    def publish_gauge_ok(
        self,
        *,
        ts: float,
        od: float,
        judge: str,
        od2: float | None = None,
        judge2: str = 'UNK',
        raw: str = '',
    ) -> None:
        self.publish_legacy(
            'gauge_ok',
            {
                'ts': float(ts),
                'od': float(od),
                'judge': str(judge),
                'od2': (float(od2) if od2 is not None else None),
                'judge2': str(judge2 or 'UNK'),
                'raw': str(raw),
            },
        )

    def publish_gauge_err(self, *, ts: float, err: str) -> None:
        self.publish_legacy('gauge_err', {'ts': float(ts), 'err': str(err)})


class WorkflowUiEventAdapter(UiQueueCompatAdapter, EventSink):
    """Workflow-side adapter that preserves the legacy UI queue payloads."""

    def publish_state(self, state: str, message: str) -> None:
        self.publish_legacy('auto_state', {'state': state, 'msg': message})

    def publish_progress(
        self,
        *,
        section_index: int,
        section_total: int,
        z_pos_mm: float,
        ax0_abs: float,
    ) -> None:
        self.publish_legacy(
            'auto_progress',
            {
                'idx': max(0, int(section_index) - 1),
                'total': int(section_total),
                'x_ui': float(z_pos_mm),
                'x_abs': float(ax0_abs),
            },
        )

    def publish_length(self, payload: Mapping[str, Any]) -> None:
        self.publish_legacy('auto_len', dict(payload))

    def publish_coverage(self, payload: Mapping[str, Any]) -> None:
        self.publish_legacy('auto_cov', dict(payload))

    def publish_raw_points(self, points: Sequence[Mapping[str, Any]]) -> None:
        self.publish_legacy('auto_raw_points', {'points': [dict(point) for point in points]})

    def publish_row(self, row: MeasureRow) -> None:
        self.publish_legacy('auto_row', {'row': row})

    def publish_straightness(self, payload: Mapping[str, Any]) -> None:
        self.publish_legacy('auto_straightness', dict(payload))

    def publish_postcalc(self, payload: Mapping[str, Any]) -> None:
        self.publish_legacy('auto_postcalc', dict(payload))


__all__ = ['UiQueueCompatAdapter', 'WorkerUiEventAdapter', 'WorkflowUiEventAdapter']
