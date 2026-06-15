from __future__ import annotations

"""Minimal runtime ports for the formal measurement workflow.

Each port is inferred from the actual calls made by AutoFlowOrchestrator,
ProductionWorkflow, and the legacy executor — not designed a priori.
"""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from core.models import AxisComm, MeasureRow, Recipe
from domain.state import CalibrationSnapshot
from machine.device_gateway import ClChannel, ClReadResult, PollProfile


# ---------------------------------------------------------------------------
# MotionPort — axis motion + PLC write + poll profile
# ---------------------------------------------------------------------------


@runtime_checkable
class MotionPort(Protocol):
    """Axis motion and PLC write surface used by AutoFlowOrchestrator and
    the legacy executor's motion/sampling mixins."""

    def get_axis_copy(self, axis: int) -> AxisComm: ...

    def movea_abs(self, axis: int, pos_abs: float, *, context: str = "MoveA") -> None: ...

    def velmove(
        self, axis: int, velocity: float,
        *, acc: float = 80.0, dec: float = 80.0, jerk: float = 300.0,
    ) -> None: ...

    def stop(self, axis: int) -> None: ...

    def halt(self, axis: int) -> None: ...

    def reset(self, axis: int) -> None: ...

    def enable(self, axis: int) -> None: ...

    def abort_motion(self, axes: Sequence[int] | None = None) -> None: ...

    def apply_soft_limits_abs(
        self, axis: int, target_abs: float, *,
        strict: bool = False, context: str = "",
    ) -> float: ...

    def pulse_cmd_mask(self, axis: int, pulse_mask: int, pulse_ms: int = 120) -> None: ...

    def set_plc_poll_profile(self, profile: PollProfile = "normal") -> None: ...

    def write_coil(self, coil_addr: int, value: int | bool) -> None: ...


# ---------------------------------------------------------------------------
# PlcCommandPort — low-level PLC register/bit operations
# ---------------------------------------------------------------------------
# This is a transitional port for the legacy executor's motion mixin.
# New code should use MotionPort for high-level motion commands.
# Once the legacy executor is fully retired, this port can be removed.


@runtime_checkable
class PlcCommandPort(Protocol):
    """Low-level PLC surface used by ExecutorMotionMixin."""

    def _base(self, axis: int) -> int: ...
    def _write_regs(self, addr: int, values: list[int]) -> None: ...
    def set_cmd_bits(self, axis: int, *, set_mask: int = 0, clr_mask: int = 0) -> None: ...
    def _pulse_cmd_bits(self, axis: int, mask: int) -> None: ...
    def _velmove_start_axis(
        self, axis: int, velocity: float, *, acc: float = 80.0, dec: float = 80.0, jerk: float = 300.0,
    ) -> None: ...
    def _get_ax0_z_disp_limits(self) -> tuple[float, float, float]: ...


# ---------------------------------------------------------------------------
# SensorPort — sync reads, caching, simulation, recipe/calibration
# ---------------------------------------------------------------------------


@runtime_checkable
class SensorPort(Protocol):
    """Sensor read surface (PLC, gauge, simulation) used by the executor's
    sampling and core mixins."""

    def read_axis_angle_deg_sync(
        self, axis: int = 3, timeout_s: float = 0.35,
    ) -> float | None: ...

    def read_cl_sync(
        self, channel: ClChannel, *, timeout_s: float = 0.5,
    ) -> ClReadResult | None: ...

    def get_recipe_copy(self) -> Recipe: ...

    def get_calibration_snapshot(self) -> CalibrationSnapshot | None: ...

    # cached fast-path reads (from the app's PLC polling thread)
    @property
    def latest_ax3_angle_deg(self) -> float | None: ...

    @property
    def latest_cl145(self) -> Any: ...

    @property
    def latest_cl3(self) -> Any: ...

    # simulation / dispatch mode
    @property
    def sim_gauge_enabled(self) -> bool: ...

    @property
    def sim_disp_enabled(self) -> bool: ...

    def simulate_gauge_once(self, recipe: Recipe) -> tuple[float, str]: ...

    def simulate_disp_once(self, recipe: Recipe) -> tuple[float, str]: ...

    # calibration data
    @property
    def axis_cal(self) -> Any: ...

    # ID single-probe computation
    def calc_id_single_from_out2(
        self, theta_deg: list[float], out2_mm: list[float], recipe: Recipe,
    ) -> Any: ...

    # ID diameter fitting (replaces AppHost._idcal_fit_diameter)
    def fit_id_diameter(
        self, theta_deg: Any, c_mm: Any, m_mm: Any, delta_c: float,
    ) -> dict[str, Any] | None: ...

    # gauge worker (for serial send/receive during sampling)
    @property
    def gauge_worker(self) -> Any: ...


# ---------------------------------------------------------------------------
# OperatorPort — buttons, confirmation dialogs, pneumatic clamp outputs
# ---------------------------------------------------------------------------


@runtime_checkable
class OperatorPort(Protocol):
    """Operator interaction surface — inputs, outputs, confirmations."""

    def get_x_point(self, x_point: int) -> int: ...

    def get_y_point(self, y_point: int) -> int: ...

    def plc_write_y_point(self, y_point: int, value: int) -> None: ...

    def operator_confirm(
        self, title: str, message: str, *,
        allow_stop: bool = True, timeout_s: float | None = None,
    ) -> str: ...


# ---------------------------------------------------------------------------
# RunSessionPort — run lifecycle
# ---------------------------------------------------------------------------


@runtime_checkable
class RunSessionPort(Protocol):
    """Run-session lifecycle port — not a general-purpose state bag."""

    def current_run(self) -> Any | None: ...

    def start_run(self, recipe_name: str) -> Any: ...

    def update_section_result(self, row: MeasureRow) -> None: ...

    def mark_done(self) -> None: ...

    def mark_aborted(self, reason: str) -> None: ...


# ---------------------------------------------------------------------------
# LegacyAutoFlowRuntimePort — the full surface the legacy executor expects
# ---------------------------------------------------------------------------
# This is NOT a new port to implement.  It documents what the legacy
# AutoFlow executor actually calls, so that any replacement for
# AppDeviceGateway can be checked against it.  New code should use the
# narrow ports above; this exists only as a migration checkpoint.


@runtime_checkable
class LegacyAutoFlowRuntimePort(MotionPort, SensorPort, OperatorPort, PlcCommandPort, Protocol):
    """Full legacy executor surface: Motion + Sensors + Operator + low-level PLC.

    This combines the three narrow ports and adds the low-level PLC surface
    required by ExecutorMotionMixin and ExecutorSamplingMixin.  New code
    should depend on MotionPort / SensorPort / OperatorPort individually.
    """

    # -- low-level PLC (required by _executor_motion.py) -------------------
    def _base(self, axis: int) -> int: ...
    def _write_regs(self, addr: int, values: list[int]) -> None: ...
    def set_cmd_bits(self, axis: int, *, set_mask: int = 0, clr_mask: int = 0) -> None: ...
    def _pulse_cmd_bits(self, axis: int, mask: int) -> None: ...


# ---------------------------------------------------------------------------
# RotationPort — axis rotation control (generic, used by calibration + workflow)
# ---------------------------------------------------------------------------


@runtime_checkable
class RotationPort(Protocol):
    """Generic axis rotation control — start/stop AX3 rotation at given speed."""

    def start_rotation(self, rpm: float) -> None: ...

    def stop_rotation(self) -> None: ...


__all__ = [
    "LegacyAutoFlowRuntimePort",
    "MotionPort",
    "OperatorPort",
    "PlcCommandPort",
    "RotationPort",
    "RunSessionPort",
    "SensorPort",
]
