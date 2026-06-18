from __future__ import annotations

from typing import Any

from frp_workflow.steps.record_row import RecordRowStep


class FakePort:
    def __init__(self) -> None:
        self.rows: list[Any] = []

    def _record_row_impl(self, row: Any) -> None:
        self.rows.append(row)


def test_record_row_step_delegates_to_impl() -> None:
    row = {"section_idx": 2}
    port = FakePort()
    step = RecordRowStep(port)

    step.execute(row)

    assert port.rows == [row]
    assert port.rows[0] is row
    assert step.name == "record_row"
