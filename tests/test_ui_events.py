from __future__ import annotations

import subprocess
import sys

import pytest

import events
from events.types import (
    AutoRowEvent,
    AutoStateEvent,
    GaugeOkEvent,
    PlcErrEvent,
    PlcOkEvent,
    parse_ui_event,
    parse_ui_event_tuple,
)
from core.models import AxisComm, MeasureRow


class TestUiEvents:
    """Strongly-typed UI event round-trips and parsing."""

    def test_package_facade_is_lazy_and_preserves_public_exports(self) -> None:
        assert events.PlcOkEvent is PlcOkEvent

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import events.types; "
                    "assert 'events.pump' not in sys.modules; "
                    "assert 'utils.logger' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    # ------------------------------------------------------------------
    # event round-trip (from_payload → to_ui_event → parse → compare)
    # ------------------------------------------------------------------

    _ROUNDTRIP_PAYLOADS = [
        (
            "plc_err",
            PlcErrEvent,
            {"err": "connect failed", "retry": 2, "max": 5, "backoff_s": 15.0},
            {"err": "connect failed", "retry": 2, "max": 5, "backoff_s": 15.0},
        ),
        (
            "gauge_ok",
            GaugeOkEvent,
            GaugeOkEvent(
                ts=123.4,
                od=187.3,
                judge="GO",
                od2=187.1,
                judge2="GO",
                raw="M0,1,+187.3,GO,+187.1,GO",
            ),
            {"od": 187.3, "judge": "GO", "od2": 187.1, "judge2": "GO"},
        ),
        (
            "auto_state",
            AutoStateEvent,
            AutoStateEvent(state="DONE", msg="completed"),
            {"state": "DONE", "msg": "completed"},
        ),
    ]

    @pytest.mark.parametrize(("evt_name", "cls", "source", "checks"), _ROUNDTRIP_PAYLOADS)
    def test_event_roundtrip(self, evt_name: str, cls: type, source, checks: dict) -> None:
        # Build event → tuple → parse back → verify value equality
        if isinstance(source, dict):
            event = cls.from_payload(source)
            parsed = parse_ui_event_tuple(event.to_ui_event())
            assert isinstance(parsed, cls)
            for key, expected in checks.items():
                assert getattr(parsed, key) == expected
        else:
            # source is already an event instance
            parsed = parse_ui_event_tuple(source.to_ui_event())
            assert isinstance(parsed, cls)
            for key, expected in checks.items():
                assert getattr(parsed, key) == pytest.approx(expected)

    def test_auto_row_roundtrip_preserves_row_identity(self) -> None:
        """AutoRowEvent serialises the row by reference — the exact same
        MeasureRow object must survive the to_ui_event / parse round-trip."""
        row = MeasureRow(
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
        event = AutoRowEvent(row=row)
        parsed = parse_ui_event_tuple(event.to_ui_event())
        assert isinstance(parsed, AutoRowEvent)
        # The row payload is passed by reference — identity check, not value.
        assert parsed.row is row

    # ------------------------------------------------------------------
    # plc_ok parse — structurally rich payload
    # ------------------------------------------------------------------

    def test_plc_ok_parse_preserves_axes_and_bits(self) -> None:
        payload = {
            "axes": [AxisComm(act_pos=12.5)],
            "cl_out4_mm": 152.7,
            "cl_out4_cnt": 3,
            "keytest_x_bits": [1, 0, 1],
            "keytest_y_bits": [0, 1],
        }
        event = parse_ui_event("plc_ok", payload)
        assert isinstance(event, PlcOkEvent)
        assert len(event.axes) == 1
        assert event.axes[0].act_pos == pytest.approx(12.5)
        assert event.cl_out4_mm == 152.7
        assert event.keytest_x_bits == [1, 0, 1]
        assert event.keytest_y_bits == [0, 1]
