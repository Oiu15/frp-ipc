from __future__ import annotations

import tkinter as tk
from typing import Any


class AxisScreenPresenter:
    """Own per-axis UI state and translate screen events into controller intents."""

    def __init__(self, host: Any, controller: Any) -> None:
        self.host = host
        self.controller = controller
        self._axis_widgets: dict[int, dict[str, Any]] = {}
        self._axis_power_vars: dict[int, tk.IntVar] = {}
        self._current_axis: int = 0

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.host, name)
        if callable(attr) and not (name.startswith('_refresh') or name.startswith('_list')):
            raise AttributeError(name)
        return attr

    def create_power_var(self, master: tk.Misc, axis: int) -> tk.IntVar:
        ax = int(axis)
        existing = self._axis_power_vars.get(ax)
        if existing is not None:
            return existing
        var = tk.IntVar(master=master, value=0)
        self._axis_power_vars[ax] = var
        return var

    def register_axis_widgets(self, axis: int, widgets: dict[str, Any], power_var: Any) -> None:
        ax = int(axis)
        self._axis_widgets[ax] = dict(widgets)
        self._axis_power_vars[ax] = power_var

    def activate_axis(self, axis: int) -> int:
        ax = max(0, min(len(getattr(self.host, '_axis_snapshot', [])) - 1, int(axis)))
        self._current_axis = ax
        try:
            self.host.axis_idx.set(ax)
        except Exception:
            pass
        return ax

    @property
    def current_axis(self) -> int:
        return int(self._current_axis)

    def widget_for(self, axis: int, name: str) -> Any:
        return self._axis_widgets.get(int(axis), {}).get(name)

    def current_widget(self, name: str) -> Any:
        return self.widget_for(self._current_axis, name)

    def power_var_for(self, axis: int | None = None) -> Any:
        ax = self._current_axis if axis is None else int(axis)
        return self._axis_power_vars.get(ax)

    def handle_axis_selected(self, axis: int) -> None:
        self.activate_axis(axis)
        fn = getattr(self.controller, '_refresh_axis_panel', None)
        if callable(fn):
            fn()

    def handle_action(self, axis: int, action_name: str) -> Any:
        self.activate_axis(axis)
        fn = getattr(self.controller, action_name, None)
        if callable(fn):
            return fn()
        return None

    def handle_jog(self, axis: int, direction: str, on: bool) -> Any:
        self.activate_axis(axis)
        fn = getattr(self.controller, '_jog_hold', None)
        if callable(fn):
            return fn(direction, on)
        return None


__all__ = ['AxisScreenPresenter']
