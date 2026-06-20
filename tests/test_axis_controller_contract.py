from __future__ import annotations

from typing import Any

from application.controllers.axis_controller import AxisController


class _FakeHost:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def _refresh_axis_panel(self) -> str:
        self.calls.append(("_refresh_axis_panel", ()))
        return "refresh"

    def _on_power_toggle(self) -> str:
        self.calls.append(("_on_power_toggle", ()))
        return "power"

    def _write_common_params(self) -> str:
        self.calls.append(("_write_common_params", ()))
        return "write"

    def _do_reset(self) -> str:
        self.calls.append(("_do_reset", ()))
        return "reset"

    def _do_stop(self) -> str:
        self.calls.append(("_do_stop", ()))
        return "stop"

    def _do_halt(self) -> str:
        self.calls.append(("_do_halt", ()))
        return "halt"

    def _do_movea(self) -> str:
        self.calls.append(("_do_movea", ()))
        return "movea"

    def _do_mover(self) -> str:
        self.calls.append(("_do_mover", ()))
        return "mover"

    def _do_vel_start(self) -> str:
        self.calls.append(("_do_vel_start", ()))
        return "vel_start"

    def _do_vel_stop(self) -> str:
        self.calls.append(("_do_vel_stop", ()))
        return "vel_stop"

    def _jog_hold(self, direction: str, on: bool) -> tuple[str, bool]:
        self.calls.append(("_jog_hold", (direction, on)))
        return (direction, on)


def test_axis_controller_delegates_explicit_methods_to_host() -> None:
    host = _FakeHost()
    controller = AxisController(host)

    assert controller.refresh_axis_panel() == "refresh"
    assert controller.on_power_toggle() == "power"
    assert controller.write_common_params() == "write"
    assert controller.reset() == "reset"
    assert controller.stop() == "stop"
    assert controller.halt() == "halt"
    assert controller.movea() == "movea"
    assert controller.mover() == "mover"
    assert controller.vel_start() == "vel_start"
    assert controller.vel_stop() == "vel_stop"
    assert controller.jog_hold("fwd", True) == ("fwd", True)

    assert host.calls == [
        ("_refresh_axis_panel", ()),
        ("_on_power_toggle", ()),
        ("_write_common_params", ()),
        ("_do_reset", ()),
        ("_do_stop", ()),
        ("_do_halt", ()),
        ("_do_movea", ()),
        ("_do_mover", ()),
        ("_do_vel_start", ()),
        ("_do_vel_stop", ()),
        ("_jog_hold", ("fwd", True)),
    ]


def test_axis_controller_dispatches_known_axis_actions_and_ignores_unknown() -> None:
    host = _FakeHost()
    controller = AxisController(host)

    assert controller.dispatch_axis_action("_do_movea") == "movea"
    assert controller.dispatch_axis_action("_do_vel_stop") == "vel_stop"
    assert controller.dispatch_axis_action("missing") is None

    assert host.calls == [
        ("_do_movea", ()),
        ("_do_vel_stop", ()),
    ]
