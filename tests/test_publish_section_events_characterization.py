from __future__ import annotations

from typing import Any, cast

from frp_workflow.autoflow_orchestrator import AutoFlowOrchestrator
from frp_workflow.steps.measure_section_context import MeasureSectionContext
from frp_workflow.steps.publish_events_context import PublishEventsContext


class _FakeEventSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def publish_raw_points(self, points: Any) -> None:
        self.events.append(("raw", points))

    def publish_coverage(self, payload: Any) -> None:
        self.events.append(("coverage", payload))

    def publish_row(self, row: Any) -> None:
        self.events.append(("row", row))


class _FakeWorkflow:
    def __init__(self) -> None:
        self.raw_points: list[Any] = []
        self.coverage: list[Any] = []
        self.rows: list[Any] = []

    def record_raw_points(self, points: Any) -> None:
        self.raw_points.append(points)

    def record_coverage(self, payload: Any) -> None:
        self.coverage.append(payload)

    def record_row(self, row: Any) -> None:
        self.rows.append(row)


def _minimal_orchestrator(*, event_sink: _FakeEventSink, workflow: _FakeWorkflow | None) -> Any:
    orchestrator = cast(Any, AutoFlowOrchestrator.__new__(AutoFlowOrchestrator))
    orchestrator.event_sink = event_sink
    orchestrator.production_workflow = workflow
    return orchestrator


def test_publish_section_events_preserves_payloads_and_order() -> None:
    event_sink = _FakeEventSink()
    workflow = _FakeWorkflow()
    orchestrator = _minimal_orchestrator(event_sink=event_sink, workflow=workflow)
    measure_context = MeasureSectionContext(
        section_index=3,
        z_pos_mm=12.5,
        x_abs=34.5,
        centers_xyz=[],
        centers_xyz_id=[],
        concentricity_list=[],
    )
    raw_points = [{"theta_deg": 90.0}]
    coverage_payload = {"section_idx": 3, "coverage": {"ok": True}}
    row = {"section_idx": 3, "od": 100.0}
    context = PublishEventsContext(
        measure_context=measure_context,
        raw_points=raw_points,
        coverage_payload=coverage_payload,
        row=row,
    )

    AutoFlowOrchestrator._publish_section_events_impl(orchestrator, context)

    assert event_sink.events == [
        ("raw", raw_points),
        ("coverage", coverage_payload),
        ("row", row),
    ]
    assert raw_points == [
        {
            "theta_deg": 90.0,
            "section_idx": 3,
            "z_pos_mm": 12.5,
            "sample_idx": 0,
        }
    ]
    assert workflow.raw_points == [raw_points]
    assert workflow.coverage == [coverage_payload]
    assert workflow.rows == []


def test_publish_section_events_does_not_require_workflow() -> None:
    event_sink = _FakeEventSink()
    orchestrator = _minimal_orchestrator(event_sink=event_sink, workflow=None)
    measure_context = MeasureSectionContext(
        section_index=1,
        z_pos_mm=2.5,
        x_abs=3.5,
        centers_xyz=[],
        centers_xyz_id=[],
        concentricity_list=[],
    )
    row = {"row": "only"}

    AutoFlowOrchestrator._publish_section_events_impl(
        orchestrator,
        PublishEventsContext(measure_context=measure_context, row=row),
    )

    assert event_sink.events == [("row", row)]
