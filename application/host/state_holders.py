from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, overload


T = TypeVar("T")


class StateField(Generic[T]):
    """Descriptor mapping a legacy AppHost attribute to a state-holder field."""

    def __init__(self, holder_name: str, field_name: str) -> None:
        self.holder_name = holder_name
        self.field_name = field_name

    def _holder(self, obj: Any) -> Any:
        try:
            return object.__getattribute__(obj, self.holder_name)
        except AttributeError:
            factories = {
                "cl_snapshot_state": ClSnapshotState,
                "keytest_state": KeyTestState,
                "calibration_ui_display_state": CalibrationUiDisplayState,
            }
            factory = factories[self.holder_name]
            holder = factory()
            object.__setattr__(obj, self.holder_name, holder)
            return holder

    @overload
    def __get__(self, obj: None, objtype: type[Any] | None = None) -> StateField[T]: ...

    @overload
    def __get__(self, obj: Any, objtype: type[Any] | None = None) -> T: ...

    def __get__(self, obj: Any, objtype: type[Any] | None = None) -> T | StateField[T]:
        if obj is None:
            return self
        return getattr(self._holder(obj), self.field_name)

    def __set__(self, obj: Any, value: T) -> None:
        setattr(self._holder(obj), self.field_name, value)


@dataclass(slots=True)
class ClSnapshotState:
    """Latest CL background-poll snapshots owned by AppHost."""

    id_mm: float | None = None
    id_raw: int | None = None
    id_cnt: int | None = None
    id_ts: float = 0.0
    out1_mm: float | None = None
    out1_raw: int | None = None
    out1_cnt: int | None = None
    out2_mm: float | None = None
    out2_raw: int | None = None
    out2_cnt: int | None = None
    out4_mm: float | None = None
    out4_raw: int | None = None
    out4_cnt: int | None = None
    out5_mm: float | None = None
    out5_raw: int | None = None
    out5_cnt: int | None = None
    out_ts: float = 0.0
    last_cl_cnt: int | None = None


@dataclass(slots=True)
class KeyTestState:
    """Thread-safe keytest X/Y state owned by AppHost."""

    x_vars: list[Any] = field(default_factory=list)
    y_vars: list[Any] = field(default_factory=list)
    y_lastcmd_vars: list[Any] = field(default_factory=list)
    x_bits: Any = None
    y_bits: Any = None
    bits_lock: threading.Lock = field(default_factory=threading.Lock)
    x_points_state: list[int] = field(default_factory=list)
    y_points_state: list[int] = field(default_factory=list)
    y_points_has_read: bool = False
    y_last_command_state: list[int] = field(default_factory=list)


@dataclass(slots=True)
class CalibrationUiDisplayState:
    """Calibration UI display dictionaries owned by AppHost."""

    axis_cal_vars: dict[str, Any] = field(default_factory=dict)
    axis_cal_field_status_vars: dict[str, Any] = field(default_factory=dict)
    axis_cal_status_vars: dict[str, Any] = field(default_factory=dict)


__all__ = ["CalibrationUiDisplayState", "ClSnapshotState", "KeyTestState", "StateField"]
