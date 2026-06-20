from __future__ import annotations

import tkinter as tk
from typing import Any, Protocol


class AxisScreenViewPort(Protocol):
    def axis_count(self) -> int: ...
    def get_axis_index_var(self) -> Any: ...
    def refresh_axis_panel(self) -> None: ...


class AxisCommandPort(Protocol):
    def refresh_axis_panel(self) -> Any: ...
    def dispatch_axis_action(self, action_name: str) -> Any: ...
    def jog_hold(self, direction: str, on: bool) -> Any: ...


class AxisScreenHostView:
    def __init__(self, app: Any) -> None:
        self._app = app

    def axis_count(self) -> int:
        axes = getattr(self._app, "_axis" + "_snapshot", ())
        try:
            return len(axes)
        except Exception:
            return 0

    def get_axis_index_var(self) -> Any:
        return getattr(self._app, "axis_idx")

    def refresh_axis_panel(self) -> None:
        fn = getattr(self._app, "_refresh_axis_panel", None)
        if callable(fn):
            fn()


class AxisScreenPresenter:
    """Own per-axis UI state and translate screen events into controller intents."""

    _HOST_ATTR_ALLOWLIST = {
        'axis_idx',
    }
    _HOST_CALL_PREFIX_ALLOWLIST = (
        '_list',
        '_refresh',
    )

    def __init__(self, view: Any, controller: AxisCommandPort) -> None:
        if all(hasattr(view, name) for name in ("axis_count", "get_axis_index_var", "refresh_axis_panel")):
            resolved_view = view
        else:
            resolved_view = AxisScreenHostView(view)
        self._view = resolved_view
        self.controller = controller
        self._axis_widgets: dict[int, dict[str, Any]] = {}
        self._axis_power_vars: dict[int, tk.IntVar] = {}
        self._current_axis: int = 0

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    @property
    def axis_idx(self) -> Any:
        return self._view.get_axis_index_var()

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
        count = max(1, int(self._view.axis_count() or 0))
        ax = max(0, min(count - 1, int(axis)))
        self._current_axis = ax
        try:
            self._view.get_axis_index_var().set(ax)
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
        self.controller.refresh_axis_panel()

    def handle_action(self, axis: int, action_name: str) -> Any:
        self.activate_axis(axis)
        return self.controller.dispatch_axis_action(action_name)

    def handle_jog(self, axis: int, direction: str, on: bool) -> Any:
        self.activate_axis(axis)
        return self.controller.jog_hold(direction, on)


__all__ = ['AxisCommandPort', 'AxisScreenHostView', 'AxisScreenPresenter', 'AxisScreenViewPort']
