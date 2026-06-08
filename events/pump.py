from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Any

from events.dispatcher import UiEventDispatcher
from utils.logger import log


@dataclass(frozen=True, slots=True)
class UiQueuePumpResult:
    batch_size: int = 0
    plc_read_n: int = 0


class UiQueuePump:
    """Drain legacy UI queue events and route them through typed dispatchers."""

    def __init__(
        self,
        *,
        ui_q: queue.Queue[Any],
        device_dispatcher: UiEventDispatcher,
        measurement_dispatcher: UiEventDispatcher,
        perf_ui_queue: Any,
        log_filter: set[str],
    ) -> None:
        self.ui_q = ui_q
        self.device_dispatcher = device_dispatcher
        self.measurement_dispatcher = measurement_dispatcher
        self.perf_ui_queue = perf_ui_queue
        self.log_filter = set(log_filter)

    def drain(self) -> UiQueuePumpResult:
        batch_size = 0
        plc_read_n = 0
        try:
            while True:
                k, payload = self.ui_q.get_nowait()
                batch_size += 1
                self._log_event(str(k), payload)
                if k == "plc_read":
                    plc_read_n += 1
                handled = self.device_dispatcher.dispatch(k, payload)
                if not handled:
                    self.measurement_dispatcher.dispatch(k, payload)
        except queue.Empty:
            pass
        return UiQueuePumpResult(batch_size=batch_size, plc_read_n=plc_read_n)

    def _log_event(self, k: str, payload: Any) -> None:
        t_evtlog0_ns = time.perf_counter_ns()
        try:
            data = payload if isinstance(payload, dict) else {}
            if k in self.log_filter:
                if k == "auto_row":
                    row = data.get("row", None)
                    if row is not None:
                        log(
                            "UI_AUTO_ROW",
                            idx=getattr(row, "idx", None),
                            od_dev=getattr(row, "od_dev", None),
                            od_runout=getattr(row, "od_runout", None),
                            od_round=getattr(row, "od_round", None),
                            id_dev=getattr(row, "id_dev", None),
                            id_runout=getattr(row, "id_runout", None),
                            id_round=getattr(row, "id_round", None),
                            concentricity=getattr(row, "concentricity", None),
                            ok=getattr(row, "ok", None),
                        )
                    else:
                        log("UI_EVT", k=k)
                elif k == "auto_state":
                    log("UI_AUTO_STATE", state=data.get("state", None), message=data.get("msg", None))
                elif k == "auto_progress":
                    log(
                        "UI_AUTO_PROGRESS",
                        idx=data.get("idx", None),
                        total=data.get("total", None),
                        x_ui=data.get("x_ui", None),
                        x_abs=data.get("x_abs", None),
                    )
                elif k == "auto_cov":
                    log(
                        "UI_AUTO_COV",
                        idx=data.get("idx", None),
                        cov=data.get("cov", None),
                        miss=data.get("miss", None),
                        reason=data.get("reason", None),
                        revs=data.get("revs", None),
                        elapsed=data.get("elapsed", None),
                    )
                elif k == "auto_postcalc":
                    log(
                        "UI_AUTO_POSTCALC",
                        ecc_od=data.get("ecc_od", None),
                        ecc_id=data.get("ecc_id", None),
                        straight_od=data.get("straight_od", None),
                        straight_id=data.get("straight_id", None),
                        axis_dist=data.get("axis_dist", None),
                    )
                elif k == "auto_straightness":
                    log(
                        "UI_AUTO_STRAIGHT",
                        straight_od=data.get("straight_od", None),
                        straight_id=data.get("straight_id", None),
                        axis_dist=data.get("axis_dist", None),
                    )
                elif k == "auto_clear":
                    log("UI_AUTO_CLEAR")
                elif k == "gauge_err":
                    log("UI_GAUGE_ERR", err=data.get("err", None))
                elif k == "gauge_conn":
                    log(
                        "UI_GAUGE_CONN",
                        connected=data.get("connected", None),
                        port=data.get("port", None),
                        baud=data.get("baud", None),
                    )
                elif k == "plc_err":
                    log(
                        "UI_PLC_ERR",
                        err=data.get("err", None),
                        retry=data.get("retry", None),
                        max=data.get("max", None),
                        backoff_s=data.get("backoff_s", None),
                    )
                elif k == "plc_giveup":
                    log("UI_PLC_GIVEUP", retry=data.get("retry", None), max=data.get("max", None))
                elif k == "plc_manual":
                    log("UI_PLC_MANUAL", ip=data.get("ip", None), port=data.get("port", None))
                else:
                    log("UI_EVT", k=k)
        except Exception:
            pass
        self.perf_ui_queue.add_time_ns("event_log", time.perf_counter_ns() - t_evtlog0_ns)


__all__ = ["UiQueuePump", "UiQueuePumpResult"]
