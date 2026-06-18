from __future__ import annotations

from frp_workflow.steps.measure_section_context import MeasureSectionContext
from frp_workflow.steps.publish_events import PublishEventsStep
from frp_workflow.steps.publish_events_context import PublishEventsContext


class FakePort:
    def __init__(self) -> None:
        self.context: PublishEventsContext | None = None

    def _publish_section_events_impl(self, context: PublishEventsContext) -> None:
        self.context = context


def test_publish_events_step_delegates_to_impl() -> None:
    measure_context = MeasureSectionContext(
        section_index=2,
        z_pos_mm=12.5,
        x_abs=34.5,
        centers_xyz=[],
        centers_xyz_id=[],
        concentricity_list=[],
    )
    context = PublishEventsContext(
        measure_context=measure_context,
        raw_points=[],
    )
    port = FakePort()
    step = PublishEventsStep(port)

    step.execute(context)

    assert port.context is context
    assert step.name == "publish_events"
