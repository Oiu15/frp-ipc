from __future__ import annotations

"""Minimal step protocol — Phase 5 first round."""

from typing import Any, Protocol


class WorkflowStep(Protocol):
    """A single phase of the AutoFlow measurement pipeline."""

    name: str

    def execute(self) -> Any:
        """Run this step and return its result (if any)."""
        ...


__all__ = ["WorkflowStep"]
