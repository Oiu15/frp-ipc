from __future__ import annotations

from application.controllers.key_test_controller import KeyTestController


class _FakeKeyTestHost:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def write_keytest_y(self, y_point: int, value: int) -> None:
        self.calls.append((y_point, value))


def test_key_test_controller_delegates_write_y() -> None:
    host = _FakeKeyTestHost()
    controller = KeyTestController(host)

    controller.write_keytest_y(10, 1)
    controller.write_keytest_y(15, 0)

    assert host.calls == [(10, 1), (15, 0)]
