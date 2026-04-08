from __future__ import annotations

from typing import Any


class GaugeScreenPresenter:
    """Translate gauge-screen UI events into controller intents."""

    def __init__(self, host: Any, controller: Any) -> None:
        self.host = host
        self.controller = controller

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.host, name)
        if callable(attr) and not (name.startswith('_refresh') or name.startswith('_list')):
            raise AttributeError(name)
        return attr

    def handle_request_command_changed(self, cmd: str) -> Any:
        norm = str(cmd or 'M1,1').strip() or 'M1,1'
        fn = getattr(self.controller, 'set_gauge_request_command', None)
        if callable(fn):
            return fn(norm)
        return None


__all__ = ['GaugeScreenPresenter']
