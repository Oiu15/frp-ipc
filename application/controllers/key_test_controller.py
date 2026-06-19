from __future__ import annotations

from typing import Protocol


class KeyTestHostPort(Protocol):
    def write_keytest_y(self, y_point: int, value: int) -> None:
        ...


class KeyTestController:
    def __init__(self, host: KeyTestHostPort) -> None:
        self._host = host

    def write_keytest_y(self, y_point: int, value: int) -> None:
        self._host.write_keytest_y(y_point, value)


__all__ = ["KeyTestController", "KeyTestHostPort"]
