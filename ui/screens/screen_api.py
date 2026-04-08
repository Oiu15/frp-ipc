from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ScreenApi:
    """Small facade that lets legacy screen code read presenter/controller/ui separately.

    The constructor explicitly receives the three screen-facing roles, while
    ``__getattr__`` keeps migration diff small inside existing screen builders.
    Resolution order favors UI state first, then presenter helpers, then
    controller actions. Attribute writes are always redirected to the UI
    context so legacy widget/variable registration keeps working.
    """

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
        setattr(self.ui, name, value)

    def __delattr__(self, name: str) -> None:
        if name in {"presenter", "controller", "ui"}:
            raise AttributeError(name)
        delattr(self.ui, name)


__all__ = ["ScreenApi"]
