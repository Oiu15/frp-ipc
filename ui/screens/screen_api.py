from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ScreenApi:
    """Small facade that lets legacy screen code read presenter/controller/ui separately."""

    presenter: Any
    controller: Any
    ui: Any

    def __getattr__(self, name: str) -> Any:
        for source in (self.ui, self.presenter, self.controller):
            try:
                return getattr(source, name)
            except AttributeError:
                continue
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"presenter", "controller", "ui"}:
            object.__setattr__(self, name, value)
            return
        raise AttributeError(name)

    def __delattr__(self, name: str) -> None:
        raise AttributeError(name)


__all__ = ["ScreenApi"]
