from __future__ import annotations

from frp_workflow.steps.measure_section_context import MeasureSectionContext
from frp_workflow.steps.publish_events_context import PublishEventsContext


def test_publish_events_context_keeps_fields() -> None:
    measure_context = MeasureSectionContext(
        section_index=2,
        z_pos_mm=12.5,
        x_abs=34.5,
        centers_xyz=[],
        centers_xyz_id=[],
        concentricity_list=[],
    )
    raw_points = [{"theta_deg": 0.0}]
    coverage_payload = {"coverage": True}
    row = {"section_idx": 2}

    context = PublishEventsContext(
        measure_context=measure_context,
        raw_points=raw_points,
        coverage_payload=coverage_payload,
        row=row,
    )

    assert context.measure_context is measure_context
    assert context.raw_points is raw_points
    assert context.coverage_payload is coverage_payload
    assert context.row is row
