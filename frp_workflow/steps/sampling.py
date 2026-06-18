from __future__ import annotations

"""Sampling step boundary for one measured section."""

from dataclasses import dataclass
from typing import Protocol

from frp_workflow.steps.measure_section_context import MeasureSectionContext
from frp_workflow.steps.sampling_result import SamplingResult


class SamplingPort(Protocol):
    """Narrow surface that the sampling step needs."""

    def _sample_section_impl(self, context: MeasureSectionContext) -> SamplingResult: ...


@dataclass(slots=True)
class SamplingStep:
    """Run the existing section sampling phase."""

    port: SamplingPort
    name: str = "sampling"

    def execute(self, context: MeasureSectionContext) -> SamplingResult:
        """Run sampling via the legacy implementation boundary."""
        return self.port._sample_section_impl(context)


__all__ = ["SamplingPort", "SamplingStep"]
