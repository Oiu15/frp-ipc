from __future__ import annotations

from typing import Any, cast

from frp_workflow.autoflow_orchestrator import AutoFlowOrchestrator


class FakeProductionWorkflow:
    def __init__(self) -> None:
        self.rows: list[Any] = []

    def record_row(self, row: Any) -> None:
        self.rows.append(row)


def test_record_row_impl_records_row_object() -> None:
    orchestrator = cast(Any, AutoFlowOrchestrator.__new__(AutoFlowOrchestrator))
    workflow = FakeProductionWorkflow()
    row = {"section_idx": 2}
    orchestrator.production_workflow = workflow

    AutoFlowOrchestrator._record_row_impl(orchestrator, row)

    assert workflow.rows == [row]
    assert workflow.rows[0] is row
