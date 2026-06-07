from __future__ import annotations

import queue
from typing import Any

import pytest

from events.adapters import WorkerUiEventAdapter
from application.adapters.ui_queue import WorkflowUiEventAdapter
from core.models import AxisComm, MeasureRow


class TestUiQueueAdapters:
    """Adapter → queue round-trips — every adapter method must produce
    the exact legacy (name, dict) tuple expected by downstream consumers."""

    # ------------------------------------------------------------------
    # shared test data
    # ------------------------------------------------------------------

    _SAMPLE_ROW = MeasureRow(
        idx=1,
        x_ui=100.0,
        x_abs=200.0,
        od_avg=187.31,
        od_dev=0.01,
        od_runout=0.02,
        od_round=0.03,
        id_avg=152.70,
        id_dev=0.01,
        id_runout=0.02,
        id_round=0.03,
        concentricity=0.04,
    )

    _SAMPLE_AXES = [AxisComm(act_pos=12.5)]

    # ------------------------------------------------------------------
    # single-publish cases — one method call → one queue entry
    # ------------------------------------------------------------------

    _SINGLE_PUBLISH_CASES: list[tuple[type, str, dict[str, Any], str, dict[str, Any]]] = [
        # (adapter_cls, method, kwargs, expected_name, expected_payload)
        (
            WorkerUiEventAdapter,
            "publish_plc_ok",
            {
                "axes": _SAMPLE_AXES,
                "cl_out4_mm": 152.7,
                "cl_out4_cnt": 3,
                "keytest_x_bits": [1, 0, 1],
                "keytest_y_bits": [0, 1],
            },
            "plc_ok",
            {
                "axes": _SAMPLE_AXES,
                "cl_out4_mm": 152.7,
                "cl_out4_cnt": 3,
                "keytest_x_bits": [1, 0, 1],
                "keytest_y_bits": [0, 1],
            },
        ),
        (
            WorkerUiEventAdapter,
            "publish_gauge_ok",
            {
                "ts": 123.4,
                "od": 187.3,
                "judge": "GO",
                "od2": 187.1,
                "judge2": "GO",
                "raw": "M0,1,+187.3,GO,+187.1,GO",
            },
            "gauge_ok",
            {
                "ts": 123.4,
                "od": 187.3,
                "judge": "GO",
                "od2": 187.1,
                "judge2": "GO",
                "raw": "M0,1,+187.3,GO,+187.1,GO",
            },
        ),
        (
            WorkflowUiEventAdapter,
            "publish_progress",
            {"section_index": 2, "section_total": 5, "z_pos_mm": 100.0, "ax0_abs": 200.0},
            "auto_progress",
            {"idx": 1, "total": 5, "x_ui": 100.0, "x_abs": 200.0},
        ),
    ]

    @pytest.mark.parametrize(
        ("adapter_cls", "method", "kwargs", "expected_name", "expected_payload"),
        _SINGLE_PUBLISH_CASES,
    )
    def test_single_publish_roundtrip(
        self,
        adapter_cls: type,
        method: str,
        kwargs: dict[str, Any],
        expected_name: str,
        expected_payload: dict[str, Any],
    ) -> None:
        ui_q: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        adapter = adapter_cls(ui_q)

        getattr(adapter, method)(**kwargs)

        name, payload = ui_q.get_nowait()
        assert name == expected_name
        for key, expected in expected_payload.items():
            assert payload[key] == expected, f"payload[{key!r}] mismatch"

    # ------------------------------------------------------------------
    # multi-publish case — two calls → two queue entries in order
    # ------------------------------------------------------------------

    def test_publish_row_and_raw_points_preserves_legacy_shape(self) -> None:
        ui_q: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        adapter = WorkflowUiEventAdapter(ui_q)

        adapter.publish_row(self._SAMPLE_ROW)
        adapter.publish_raw_points([{"section_idx": 1, "theta_deg": 0.0, "od_mm": 187.3}])

        assert ui_q.get_nowait() == ("auto_row", {"row": self._SAMPLE_ROW})
        assert ui_q.get_nowait() == (
            "auto_raw_points",
            {"points": [{"section_idx": 1, "theta_deg": 0.0, "od_mm": 187.3}]},
        )
