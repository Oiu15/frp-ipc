from __future__ import annotations

"""Row build step boundary for one measured section."""

from dataclasses import dataclass
from typing import Protocol

from frp_workflow.steps.measure_section_context import MeasureSectionContext
from frp_workflow.steps.row_build_result import RowBuildResult
from frp_workflow.steps.sampling_result import SamplingResult


class RowBuildPort(Protocol):
    """Narrow surface that the row build step needs."""

    def _build_row_impl(
        self,
        context: MeasureSectionContext,
        sampling_result: SamplingResult,
    ) -> RowBuildResult: ...


@dataclass(slots=True)
class RowBuildStep:
    """Build the existing measurement row via the orchestrator boundary."""

    port: RowBuildPort
    name: str = "row_build"

    def execute(
        self,
        context: MeasureSectionContext,
        sampling_result: SamplingResult,
    ) -> RowBuildResult:
        return self.port._build_row_impl(context, sampling_result)


__all__ = ["RowBuildPort", "RowBuildStep"]
