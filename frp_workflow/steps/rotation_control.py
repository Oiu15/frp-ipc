from __future__ import annotations

"""Rotation control step boundary for split section sampling."""

from dataclasses import dataclass
from typing import Protocol


class RotationControlPort(Protocol):
    """Narrow surface needed to restart rotation for split sampling."""

    def _restart_rotation_for_split_impl(self) -> None: ...


@dataclass(slots=True)
class RotationControlStep:
    """Delegate split-sampling rotation restart to the orchestrator boundary."""

    port: RotationControlPort
    name: str = "rotation_control"

    def restart_for_split(self) -> None:
        self.port._restart_rotation_for_split_impl()


__all__ = ["RotationControlPort", "RotationControlStep"]
