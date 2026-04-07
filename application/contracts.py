from __future__ import annotations

"""Application-layer contracts for the measurement main flow.

This module intentionally defines only the boundaries that the current
"主界面开始测量 -> AutoFlow/Orchestrator -> 结果回传 -> 导出" main chain
really needs.

Scope notes:
- The workflow should receive a `Recipe` snapshot directly, so a recipe
  repository protocol is intentionally not included here yet.
- These protocols are transition-oriented: they keep the boundary small,
  but still map cleanly to the current App/worker capabilities.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from core.models import AxisCal, AxisComm, GaugeSample, MeasureRow, Recipe

EventPayload = Mapping[str, Any]
RawPoint = Mapping[str, Any]
PollProfile = Literal["normal", "sampling"]
RunStatus = Literal["DONE", "STOP", "ERR"]
OperatorConfirmResult = Literal["confirm", "stop", "timeout", "cancel"]


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Stable run identity allocated before a measurement starts."""

    serial: str
    run_id: str
    started_at_ts: float


@dataclass(frozen=True, slots=True)
class OdCalibrationSnapshot:
    """OD calibration data required by the measurement flow."""

    b_active_mm: float
    out1_map: str = "L"
    d_ref_mm: float | None = None
    request_cmd: str = ""


@dataclass(frozen=True, slots=True)
class IdCalibrationSnapshot:
    """ID calibration data required by the measurement flow."""

    delta_c_mm: float
    d_ref_mm: float | None = None


@dataclass(frozen=True, slots=True)
class ClOut145Sample:
    """CL OUT1/OUT2/OUT4/OUT5 sample plus raw/counter metadata."""

    out1_mm: float | None
    out2_mm: float | None
    out4_mm: float | None
    out5_mm: float | None
    raw: Mapping[str, int | None] = field(default_factory=dict)
    counters: Mapping[str, int | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClOut3Sample:
    """Backward-compatible single-channel CL sample."""

    value_mm: float | None
    raw_value: int | None
    counter: int | None


@dataclass(slots=True)
class RunSnapshot:
    """Run aggregate passed to the run repository/export layer."""

    identity: RunIdentity
    recipe: Recipe
    rows: Sequence[MeasureRow] = field(default_factory=tuple)
    raw_points: Sequence[RawPoint] = field(default_factory=tuple)
    section_coverage: Mapping[int, EventPayload] = field(default_factory=dict)
    length_result: EventPayload | None = None
    summary: EventPayload = field(default_factory=dict)
    finished_at_ts: float | None = None
    status: RunStatus = "DONE"


@runtime_checkable
class EventSink(Protocol):
    """Workflow -> outside world event publishing contract."""

    def publish_state(self, state: str, message: str) -> None: ...

    def publish_progress(
        self,
        *,
        section_index: int,
        section_total: int,
        z_pos_mm: float,
        ax0_abs: float,
    ) -> None: ...

    def publish_length(self, payload: EventPayload) -> None: ...

    def publish_coverage(self, payload: EventPayload) -> None: ...

    def publish_raw_points(self, points: Sequence[RawPoint]) -> None: ...

    def publish_row(self, row: MeasureRow) -> None: ...

    def publish_straightness(self, payload: EventPayload) -> None: ...

    def publish_postcalc(self, payload: EventPayload) -> None: ...


@runtime_checkable
class MachineGateway(Protocol):
    """Workflow-facing machine and sensor contract.

    This is intentionally a workflow contract, not a raw driver contract.
    It exposes the smallest set of machine/sensor operations that the
    current measurement main chain actually depends on.
    """

    def get_axis_calibration(self) -> AxisCal: ...

    def get_axis_snapshot(self, axis: int) -> AxisComm: ...

    def get_x_point(self, x_point: int) -> int: ...

    def write_y_point(self, y_point: int, value: int) -> None: ...

    def operator_confirm(
        self,
        title: str,
        message: str,
        *,
        allow_stop: bool = True,
        timeout_s: float | None = None,
    ) -> OperatorConfirmResult: ...

    def set_poll_profile(self, profile: PollProfile) -> None: ...

    def set_cmd_bits(self, axis: int, *, set_mask: int = 0, clr_mask: int = 0) -> None: ...

    def pulse_cmd_bits(self, axis: int, pulse_mask: int, pulse_ms: int = 120) -> None: ...

    def move_abs(self, axis: int, pos_abs: float, *, context: str = "MoveA") -> None: ...

    def velmove_start(
        self,
        axis: int,
        velocity: float,
        *,
        acc: float = 80.0,
        dec: float = 80.0,
        jerk: float = 300.0,
    ) -> None: ...

    def velmove_stop(self, axis: int) -> None: ...

    def abort_motion(self, axes: Sequence[int] | None = None) -> None: ...

    def apply_soft_limits_abs(
        self,
        axis: int,
        target_abs: float,
        *,
        strict: bool = False,
        context: str = "",
    ) -> float: ...

    def get_latest_axis_angle_deg(self) -> float | None: ...

    def read_axis_angle_deg_sync(self, axis: int = 3, timeout_s: float = 0.35) -> float | None: ...

    def request_gauge_sample(self) -> None: ...

    def get_latest_gauge_sample(self) -> GaugeSample | None: ...

    def is_sim_gauge_enabled(self) -> bool: ...

    def simulate_gauge_once(self, recipe: Recipe) -> tuple[float, str]: ...

    def is_sim_disp_enabled(self) -> bool: ...

    def simulate_disp_once(self, recipe: Recipe) -> tuple[float, str]: ...

    def get_cached_cl_out145(self) -> ClOut145Sample | None: ...

    def read_cl_out145_sync(self, timeout_s: float = 0.5) -> ClOut145Sample | None: ...

    def get_cached_cl_out3(self) -> ClOut3Sample | None: ...

    def read_cl_out3_sync(self, timeout_s: float = 0.35) -> ClOut3Sample | None: ...


@runtime_checkable
class RunRepositoryProtocol(Protocol):
    """Run identity allocation + export boundary for the measurement main flow."""

    def prepare_run(self, recipe_name: str) -> RunIdentity: ...

    def export_run(self, snapshot: RunSnapshot) -> str: ...

    def export_daily_summary(self, snapshot: RunSnapshot) -> None: ...


@runtime_checkable
class CalibrationRepositoryProtocol(Protocol):
    """Read-only calibration access required by the measurement main flow."""

    def load_od_calibration(self) -> OdCalibrationSnapshot | None: ...

    def load_id_calibration(self) -> IdCalibrationSnapshot | None: ...


__all__ = [
    "CalibrationRepositoryProtocol",
    "ClOut145Sample",
    "ClOut3Sample",
    "EventPayload",
    "EventSink",
    "IdCalibrationSnapshot",
    "MachineGateway",
    "OdCalibrationSnapshot",
    "OperatorConfirmResult",
    "PollProfile",
    "RawPoint",
    "RunIdentity",
    "RunRepositoryProtocol",
    "RunSnapshot",
    "RunStatus",
]
