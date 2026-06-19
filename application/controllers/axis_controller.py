from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class AxisHostPort(Protocol):
    def _refresh_axis_panel(self) -> Any: ...

    def _on_power_toggle(self) -> Any: ...

    def _write_common_params(self) -> Any: ...

    def _do_reset(self) -> Any: ...

    def _do_stop(self) -> Any: ...

    def _do_halt(self) -> Any: ...

    def _do_movea(self) -> Any: ...

    def _do_mover(self) -> Any: ...

    def _do_vel_start(self) -> Any: ...

    def _do_vel_stop(self) -> Any: ...

    def _jog_hold(self, direction: str, on: bool) -> Any: ...


class AxisController:
    def __init__(self, host: AxisHostPort) -> None:
        self._host = host

    def refresh_axis_panel(self) -> Any:
        return self._host._refresh_axis_panel()

    def on_power_toggle(self) -> Any:
        return self._host._on_power_toggle()

    def write_common_params(self) -> Any:
        return self._host._write_common_params()

    def reset(self) -> Any:
        return self._host._do_reset()

    def stop(self) -> Any:
        return self._host._do_stop()

    def halt(self) -> Any:
        return self._host._do_halt()

    def movea(self) -> Any:
        return self._host._do_movea()

    def mover(self) -> Any:
        return self._host._do_mover()

    def vel_start(self) -> Any:
        return self._host._do_vel_start()

    def vel_stop(self) -> Any:
        return self._host._do_vel_stop()

    def jog_hold(self, direction: str, on: bool) -> Any:
        return self._host._jog_hold(direction, on)

    def dispatch_axis_action(self, action_name: str) -> Any:
        actions: dict[str, Callable[[], Any]] = {
            "_on_power_toggle": self.on_power_toggle,
            "_write_common_params": self.write_common_params,
            "_do_reset": self.reset,
            "_do_stop": self.stop,
            "_do_halt": self.halt,
            "_do_movea": self.movea,
            "_do_mover": self.mover,
            "_do_vel_start": self.vel_start,
            "_do_vel_stop": self.vel_stop,
        }
        action = actions.get(str(action_name))
        if action is None:
            return None
        return action()


__all__ = ["AxisController", "AxisHostPort"]
