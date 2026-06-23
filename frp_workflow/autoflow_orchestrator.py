from __future__ import annotations

"""Workflow-level orchestrator for the formal measurement flow.

Current scope is still intentionally staged:
- keep the constructor dependency boundary explicit
- own start/stop, outer state transitions, and section loop sequencing
- move measurement-chain logic out of App/AutoFlow incrementally
- reuse existing sampling/fit helpers instead of rewriting algorithms
"""

import math
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from domain.protocols import RunRepositoryProtocol
from events.protocols import EventSink
from machine.device_gateway import DeviceGateway
from machine.ports import MotionPort, OperatorPort, PlcCommandPort, SensorPort
from domain.state import CalibrationSnapshot, RunSession, RuntimeState
from core.models import MeasureRow, Recipe
from domain.planning import (
    build_recipe_section_plan,
    plan_section_positions,
    require_ax2_rotate_target_abs,
    resolve_ax2_position_plan,
    resolve_standby_plan,
    resolve_start_anchor_plan,
)
from domain.summaries import compute_postcalc_result
from domain.sampling import _split_slip_diag
from frp_workflow.production_workflow import ProductionWorkflow, RunResult, RunResultStatus
from frp_workflow.row_math import _compute_measure_row_result
from frp_workflow.autoflow_executor import (
    AutoFlow,
    log as legacy_log,
)
from frp_workflow.executor import SamplingResult
from frp_workflow.steps.build_section_plan import BuildSectionPlanStep
from frp_workflow.steps.finalize_run import FinalizeRunStep
from frp_workflow.steps.measure_section import MeasureSectionStep
from frp_workflow.steps.measure_section_context import MeasureSectionContext
from frp_workflow.steps.measure_row_build_inputs import MeasureRowBuildInputs
from frp_workflow.steps.prepare_run_context import PrepareRunContextStep
from frp_workflow.steps.postcalc_summary import PostcalcSummaryStep
from frp_workflow.steps.publish_events import PublishEventsStep
from frp_workflow.steps.publish_events_context import PublishEventsContext
from frp_workflow.steps.record_row import RecordRowStep
from frp_workflow.steps.rotation_control import RotationControlStep
from frp_workflow.steps.row_build import RowBuildStep
from frp_workflow.steps.row_build_result import RowBuildResult
from frp_workflow.steps.sampling import SamplingStep
from frp_workflow.steps.sampling_result import SamplingResult as SectionSamplingResult
from frp_workflow.steps.section_capture import SectionCaptureStep
from frp_workflow.steps.section_context import SectionExecutionContext
from frp_workflow.steps.section_execution import SectionExecutionStep
from frp_workflow.steps.section_geometry_accumulator import SectionGeometryAccumulator

if TYPE_CHECKING:  # pragma: no cover
    from core.models import AxisCal


class _OrchestratorRuntimeHost(Protocol):
    """Runtime host surface still used directly by AutoFlowOrchestrator."""

    axis_cal: "AxisCal"

    def operator_confirm(
        self,
        title: str,
        message: str,
        *,
        allow_stop: bool = True,
        timeout_s: float | None = None,
    ) -> str: ...

    def get_x_point(self, point: int) -> int: ...

    def get_y_point(self, point: int) -> int: ...

    def plc_write_y_point(self, point: int, value: int) -> None: ...

    def _apply_start_anchor_from_recipe(self) -> None: ...


class _StopRequested(RuntimeError):
    """Internal sentinel used to unwind the workflow loop cleanly."""

    def __init__(self, message: str = "User stopped") -> None:
        super().__init__(message)
        self.message = message


def _unwrap_theta_span_deg(theta_values_deg: list[float]) -> float:
    if len(theta_values_deg) < 2:
        return 0.0

    prev = float(theta_values_deg[0])
    cursor = prev
    min_cursor = cursor
    max_cursor = cursor
    for raw_theta in theta_values_deg[1:]:
        theta = float(raw_theta)
        delta = theta - prev
        while delta <= -180.0:
            delta += 360.0
        while delta > 180.0:
            delta -= 360.0
        cursor += delta
        min_cursor = min(min_cursor, cursor)
        max_cursor = max(max_cursor, cursor)
        prev = theta
    return float(max_cursor - min_cursor)


def _optional_finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return float(numeric)


def _point_float_values(raw_points: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for point in raw_points:
        raw_value = point.get(key)
        if raw_value is not None:
            values.append(float(cast(Any, raw_value)))
    return values


def _resolve_recipe_sampling_mode(recipe: Recipe) -> str:
    mode = str(
        getattr(
            recipe,
            "section_sampling_mode",
            getattr(recipe, "scan_mode", "sync"),
        )
        or "sync"
    ).strip().lower()
    if mode not in {"sync", "split"}:
        return "SYNC"
    return mode.upper()


def _annotate_validation_raw_points(
    *,
    raw_points: list[dict],
    section_index: int,
    z_pos_mm: float,
    window_index: int,
    window_role: str,
) -> list[dict]:
    annotated: list[dict] = []
    for point_index, point in enumerate(raw_points or []):
        copied = dict(point) if isinstance(point, dict) else {}
        copied["section_idx"] = int(section_index)
        copied["z_pos_mm"] = float(z_pos_mm)
        copied["window_index"] = int(window_index)
        copied["window_role"] = str(window_role)
        copied["point_index_in_window"] = int(point_index)
        annotated.append(copied)
    return annotated


def _build_validation_window_payload(
    *,
    window_index: int,
    window_role: str,
    raw_points: list[dict],
    sample_cov: tuple[int, int, int] | Any,
    sample_reason: tuple[str, float, float] | Any,
    n_od: int | None,
    n_id: int | None,
    max_gap_deg: float | None,
) -> dict[str, Any]:
    total_bins = filled_bins = miss_bins = None
    try:
        total_bins, filled_bins, miss_bins = sample_cov
    except Exception:
        pass

    reason = ""
    revs = None
    elapsed = None
    try:
        reason, revs, elapsed = sample_reason
    except Exception:
        pass

    ts_values = _point_float_values(raw_points or [], "ts")
    theta_values = _point_float_values(raw_points or [], "theta_deg")

    theta_start_deg = theta_values[0] if theta_values else None
    theta_end_deg = theta_values[-1] if theta_values else None
    ts_start = min(ts_values) if ts_values else None
    ts_end = max(ts_values) if ts_values else None
    return {
        "window_index": int(window_index),
        "window_role": str(window_role),
        "point_start_index": (0 if raw_points else None),
        "point_end_index": ((len(raw_points) - 1) if raw_points else None),
        "point_count": int(len(raw_points or [])),
        "ts_start": ts_start,
        "ts_end": ts_end,
        "theta_start_deg": theta_start_deg,
        "theta_end_deg": theta_end_deg,
        "theta_span_deg": _unwrap_theta_span_deg(theta_values),
        "filled_bins": (None if filled_bins is None else int(filled_bins)),
        "total_bins": (None if total_bins is None else int(total_bins)),
        "miss_bins": (None if miss_bins is None else int(miss_bins)),
        "n_od": (None if n_od is None else int(n_od)),
        "n_id": (None if n_id is None else int(n_id)),
        "reason": str(reason or ""),
        "revs": (None if revs is None else float(revs)),
        "elapsed_s": (None if elapsed is None else float(elapsed)),
        "max_gap_deg": (None if max_gap_deg is None else float(max_gap_deg)),
    }


def _build_validation_coverage_payload(
    *,
    primary_sample: SamplingResult,
    id_sample: SamplingResult | None,
    section_index: int,
    scan_mode: str,
    split_shift_deg: float | None,
    coax_unreliable: bool | None,
    keep_spinning: bool,
) -> dict[str, Any]:
    n_total, n_hit, n_miss = primary_sample.sample_cov
    cov = (float(n_hit) / float(n_total)) if n_total else None
    reason, revs, elapsed = primary_sample.sample_reason

    payload: dict[str, Any] = {
        "idx": int(section_index),
        "cov": cov,
        "cov_od": cov,
        "n_od": primary_sample.n_od,
        "n_id": primary_sample.n_id,
        "miss": n_miss,
        "max_gap_deg": primary_sample.max_gap_deg,
        "reason": reason,
        "revs": revs,
        "elapsed": elapsed,
    }
    if str(scan_mode or "").upper() == "SPLIT" and id_sample is not None:
        n_total_i, n_hit_i, n_miss_i = id_sample.sample_cov
        cov_i = (float(n_hit_i) / float(n_total_i)) if n_total_i else None
        reason_i, revs_i, elapsed_i = id_sample.sample_reason
        payload.update(
            {
                "cov_id": cov_i,
                "n_od": primary_sample.n_od,
                "n_id": id_sample.n_id,
                "miss_id": n_miss_i,
                "max_gap_deg_id": id_sample.max_gap_deg,
                "reason_id": reason_i,
                "revs_id": revs_i,
                "elapsed_id": elapsed_i,
                "split_shift_deg": split_shift_deg,
                "coax_unreliable": coax_unreliable,
                "keep_spinning": keep_spinning,
            }
        )
    return payload


def _build_measure_row_from_sampling(inputs: MeasureRowBuildInputs) -> MeasureRow:
    recipe = inputs.recipe
    section_index = inputs.section_index
    z_pos_mm = inputs.z_pos_mm
    x_abs = inputs.x_abs
    raw_od = inputs.raw_od
    raw_id = inputs.raw_id
    split_shift_deg = inputs.split_shift_deg
    coax_unreliable = inputs.coax_unreliable
    centers_xyz = inputs.centers_xyz
    centers_xyz_id = inputs.centers_xyz_id
    concentricity_list = inputs.concentricity_list
    geometry_accumulator = inputs.geometry_accumulator
    validation_fit_payload = inputs.validation_fit_payload
    if geometry_accumulator is None:
        geometry_accumulator = SectionGeometryAccumulator(
            centers_xyz=centers_xyz,
            centers_xyz_id=centers_xyz_id,
            concentricity_list=concentricity_list,
        )

    computation = _compute_measure_row_result(inputs)
    geometry_accumulator.append_od_center(computation.od_center)
    if computation.id_center is not None:
        geometry_accumulator.append_concentricity(float(computation.concentricity))
        geometry_accumulator.append_id_center(computation.id_center)

    od_use_edges = bool(getattr(recipe, "od_use_edges", False))
    if validation_fit_payload is not None:
        validation_fit_payload.clear()
        validation_fit_payload.update(
            {
                "od_center_x_mm": _optional_finite_float(computation.center_od_x),
                "od_center_y_mm": _optional_finite_float(computation.center_od_y),
                "od_radius_mm": _optional_finite_float(computation.od_radius_fit_mm),
                "od_diameter_fit_mm": _optional_finite_float(computation.od_diameter_fit_mm),
                "id_center_x_mm": _optional_finite_float(computation.center_id_x),
                "id_center_y_mm": _optional_finite_float(computation.center_id_y),
                "id_radius_mm": _optional_finite_float(computation.id_radius_fit_mm),
                "id_diameter_fit_mm": _optional_finite_float(computation.id_diameter_fit_mm),
                "od_ecc_mm": (
                    _optional_finite_float(computation.od_e)
                    if od_use_edges
                    else None
                ),
                "id_ecc_mm": _optional_finite_float(computation.id_e),
                "concentricity_mm": _optional_finite_float(computation.concentricity),
            }
        )

    try:
        od_tol_v = float(recipe.od_tol_mm)
    except Exception:
        od_tol_v = 0.0
    if computation.id_dev is None:
        ok_flag = abs(computation.od_dev) <= float(od_tol_v)
    else:
        ok_flag = (abs(computation.od_dev) <= float(od_tol_v)) and (abs(computation.id_dev) <= float(od_tol_v))

    return MeasureRow(
        idx=int(section_index),
        x_ui=float(z_pos_mm),
        x_abs=float(x_abs),
        od_avg=computation.od_avg,
        od_dev=computation.od_dev,
        od_runout=computation.od_runout,
        od_round=computation.od_round,
        od_round_fit_mm=computation.od_round_fit_mm,
        od_round_fit_rob_mm=computation.od_round_fit_rob_mm,
        od_pp_mm=(None if computation.od_pp_mm is None else float(computation.od_pp_mm)),
        od_pp_rob_mm=(None if computation.od_pp_rob_mm is None else float(computation.od_pp_rob_mm)),
        id_round_fit_mm=computation.id_round_fit_mm,
        id_round_fit_rob_mm=computation.id_round_fit_rob_mm,
        id_pp_mm=(None if computation.id_pp_mm is None else float(computation.id_pp_mm)),
        id_pp_rob_mm=(None if computation.id_pp_rob_mm is None else float(computation.id_pp_rob_mm)),
        od_e=(float(computation.od_e) if od_use_edges else None),
        od_phi_deg=(float(computation.od_phi_deg) if (od_use_edges and computation.od_phi_deg is not None) else None),
        id_e=computation.id_e,
        id_phi_deg=computation.id_phi_deg,
        id_mode=computation.id_mode,
        id_avg=cast(float, computation.id_avg),
        id_dev=cast(float, computation.id_dev),
        id_runout=cast(float, computation.id_runout),
        id_round=cast(float, computation.id_round),
        concentricity=cast(float, computation.concentricity),
        split_shift_deg=split_shift_deg,
        coax_unreliable=coax_unreliable,
        od_diam_v2=computation.od_diam_v2,
        od_round_v2=computation.od_round_v2,
        od_cx_v2=computation.od_cx_v2,
        od_cy_v2=computation.od_cy_v2,
        id_diam_v2=computation.id_diam_v2,
        id_round_v2=computation.id_round_v2,
        id_cx_v2=computation.id_cx_v2,
        id_cy_v2=computation.id_cy_v2,
        concentricity_v2=computation.concentricity_v2,
        ok=ok_flag,
        raw=f"OD:{raw_od}  ID:{raw_id}",
    )


def measure_current_position_section_capture(
    *,
    gateway: DeviceGateway,
    recipe: Recipe,
    calibration: CalibrationSnapshot,
    event_sink: EventSink | None = None,
    sensors: "SensorPort | None" = None,
) -> tuple[MeasureRow, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    sensor_port: Any = sensors if sensors is not None else gateway
    sink: EventSink = event_sink if event_sink is not None else _NoOpEventSink()
    host: Any = _LegacyAppAdapter(
        cast("MotionPort", gateway),
        sensor_port,
        cast("OperatorPort", gateway),
    )
    legacy = AutoFlow(
        device=gateway, event_sink=sink,
        motion=host._motion, sensors=host._sensors,
        operator=host._operator, plc=host,
    )
    legacy.set_runtime_context(recipe, calibration)

    section_index = 1
    try:
        x_abs = float(getattr(gateway.get_axis_copy(0), "act_pos", 0.0) or 0.0)
    except Exception:
        x_abs = 0.0
    try:
        axis_cal = getattr(host, "axis_cal", None)
        if axis_cal is not None and hasattr(axis_cal, "abs_to_z_disp"):
            z_pos_mm = float(axis_cal.abs_to_z_disp(0, x_abs))
        else:
            z_pos_mm = float(x_abs)
    except Exception:
        z_pos_mm = float(x_abs)

    scan_mode = _resolve_recipe_sampling_mode(recipe)
    split_shift_deg = None
    coax_unreliable = None
    keep_spinning = bool(getattr(recipe, "split_keep_spinning", True))
    slip_check = bool(getattr(recipe, "split_slip_check", True))
    slip_max_deg = float(getattr(recipe, "split_slip_max_deg", 5.0) or 5.0)
    omega_cv_max = float(getattr(recipe, "split_omega_cv_max", 0.25) or 0.25)

    windows: list[dict[str, Any]] = []
    centers_xyz: list[tuple[float, float, float]] = []
    centers_xyz_id: list[tuple[float, float, float]] = []
    concentricity_list: list[float] = []
    fit_payload: dict[str, Any] = {}
    primary_sample: SamplingResult
    id_sample: SamplingResult | None = None

    if scan_mode == "SPLIT":
        od_sample = legacy.sample_circle_points_result(
            recipe,
            section_idx=0,
            sample_od=True,
            sample_id=False,
            phase="OD",
        )

        id_sample = legacy.sample_circle_points_result(
            recipe,
            section_idx=0,
            sample_od=False,
            sample_id=True,
            phase="ID",
        )
        primary_sample = od_sample
        coords_od = od_sample.coords_od
        coords_id = id_sample.coords_id
        raw_od = od_sample.raw_od
        raw_id = id_sample.raw_id
        raw_points_od = od_sample.raw_points
        raw_points_id = id_sample.raw_points

        if slip_check:
            try:
                split_shift_deg, coax_unreliable = _split_slip_diag(
                    raw_points_od=raw_points_od,
                    raw_points_id=raw_points_id,
                    slip_max_deg=float(slip_max_deg),
                    omega_cv_max=float(omega_cv_max),
                )
            except Exception:
                split_shift_deg, coax_unreliable = None, None

        annotated_od = _annotate_validation_raw_points(
            raw_points=list(raw_points_od or []),
            section_index=section_index,
            z_pos_mm=z_pos_mm,
            window_index=1,
            window_role="OD",
        )
        annotated_id = _annotate_validation_raw_points(
            raw_points=list(raw_points_id or []),
            section_index=section_index,
            z_pos_mm=z_pos_mm,
            window_index=2,
            window_role="ID",
        )
        raw_points = list(annotated_od) + list(annotated_id)
        for sample_idx, point in enumerate(raw_points):
            if isinstance(point, dict):
                point["sample_idx"] = int(sample_idx)

        windows.append(
            _build_validation_window_payload(
                window_index=1,
                window_role="OD",
                raw_points=annotated_od,
                sample_cov=od_sample.sample_cov,
                sample_reason=od_sample.sample_reason,
                n_od=od_sample.n_od,
                n_id=None,
                max_gap_deg=od_sample.max_gap_deg,
            )
        )
        windows.append(
            _build_validation_window_payload(
                window_index=2,
                window_role="ID",
                raw_points=annotated_id,
                sample_cov=id_sample.sample_cov,
                sample_reason=id_sample.sample_reason,
                n_od=None,
                n_id=id_sample.n_id,
                max_gap_deg=id_sample.max_gap_deg,
            )
        )
    else:
        sync_sample = legacy.sample_circle_points_result(
            recipe,
            section_idx=0,
            sample_od=True,
            sample_id=True,
            phase="SYNC",
        )
        primary_sample = sync_sample
        coords_od = sync_sample.coords_od
        coords_id = sync_sample.coords_id
        raw_od = sync_sample.raw_od
        raw_id = sync_sample.raw_id
        raw_points = _annotate_validation_raw_points(
            raw_points=list(sync_sample.raw_points or []),
            section_index=section_index,
            z_pos_mm=z_pos_mm,
            window_index=1,
            window_role="SYNC",
        )
        for sample_idx, point in enumerate(raw_points):
            if isinstance(point, dict):
                point["sample_idx"] = int(sample_idx)
        windows.append(
            _build_validation_window_payload(
                window_index=1,
                window_role="SYNC",
                raw_points=raw_points,
                sample_cov=sync_sample.sample_cov,
                sample_reason=sync_sample.sample_reason,
                n_od=sync_sample.n_od,
                n_id=sync_sample.n_id,
                max_gap_deg=sync_sample.max_gap_deg,
            )
        )

    coverage_payload = _build_validation_coverage_payload(
        primary_sample=primary_sample,
        id_sample=id_sample,
        section_index=section_index,
        scan_mode=scan_mode,
        split_shift_deg=split_shift_deg,
        coax_unreliable=coax_unreliable,
        keep_spinning=keep_spinning,
    )
    row = _build_measure_row_from_sampling(
        MeasureRowBuildInputs(
            legacy=legacy,
            recipe=recipe,
            sensors=gateway,
            section_index=section_index,
            z_pos_mm=float(z_pos_mm),
            x_abs=float(x_abs),
            coords_od=coords_od,
            coords_id=coords_id,
            raw_od=str(raw_od),
            raw_id=str(raw_id),
            raw_points=raw_points,
            fit_weights_od=primary_sample.fit_weights_od,
            fit_weights_id=(id_sample.fit_weights_id if id_sample is not None else primary_sample.fit_weights_id),
            scan_mode=scan_mode,
            split_shift_deg=split_shift_deg,
            coax_unreliable=coax_unreliable,
            centers_xyz=centers_xyz,
            centers_xyz_id=centers_xyz_id,
            concentricity_list=concentricity_list,
            validation_fit_payload=fit_payload,
            calibration=calibration,
        )
    )
    return row, raw_points, windows, coverage_payload, dict(fit_payload)


def measure_current_position_od_avg(
    *,
    gateway: DeviceGateway,
    recipe: Recipe,
    calibration: CalibrationSnapshot,
    event_sink: EventSink | None = None,
    sensors: "SensorPort | None" = None,
) -> float:
    """Sample OD once at the current machine position and return od_avg."""
    capture = measure_current_position_section_capture(
        gateway=gateway, recipe=recipe, calibration=calibration,
        event_sink=event_sink, sensors=sensors,
    )
    row = capture[0]
    return float(row.od_avg)


class _LegacyAppAdapter:
    """Bridges MotionPort + SensorPort + OperatorPort to the legacy AutoFlow
    executor surface.  Satisfies LegacyAutoFlowRuntimePort.

    The legacy executor expects a single 'app' object with methods like
    _write_fp64, _pulse_cmd_bits, set_cmd_bits, etc.  This adapter provides
    only what the legacy flow actually calls, delegating to the typed ports.
    """

    def __init__(
        self,
        motion: "MotionPort",
        sensors: "SensorPort",
        operator: "OperatorPort",
    ) -> None:
        self.ui_q = _NoOpQueue()  # event_sink is injected separately
        self._motion = motion
        self._sensors = sensors
        self._operator = operator

    # -- motion delegate (the heavy surface) ------------------------------
    def get_axis_copy(self, axis: int) -> Any:
        return self._motion.get_axis_copy(axis)

    def apply_soft_limits_abs(self, axis: int, target: float, *, strict: bool = False, context: str = "") -> float:
        return self._motion.apply_soft_limits_abs(axis, target, strict=strict, context=context)

    def movea_abs(self, axis: int, pos_abs: float, *, context: str = "MoveA") -> None:
        self._motion.movea_abs(axis, pos_abs, context=context)

    def velmove(self, axis: int, velocity: float, *, acc: float = 80.0, dec: float = 80.0, jerk: float = 300.0) -> None:
        self._motion.velmove(axis, velocity, acc=acc, dec=dec, jerk=jerk)

    def stop(self, axis: int) -> None:
        self._motion.stop(axis)

    def halt(self, axis: int) -> None:
        self._motion.halt(axis)

    def reset(self, axis: int) -> None:
        self._motion.reset(axis)

    def enable(self, axis: int) -> None:
        self._motion.enable(axis)

    def abort_motion(self, axes: Any = None) -> None:
        self._motion.abort_motion(axes)

    def set_plc_poll_profile(self, profile: str = "normal") -> None:
        self._motion.set_plc_poll_profile(profile)  # type: ignore[arg-type]

    def pulse_cmd_mask(self, axis: int, pulse_mask: int, pulse_ms: int = 120) -> None:
        self._motion.pulse_cmd_mask(axis, pulse_mask, pulse_ms=pulse_ms)

    def write_coil(self, coil_addr: int, value: Any) -> None:
        self._motion.write_coil(coil_addr, value)

    # legacy executor PLC surface. The adapter now requires the explicit
    # PlcCommandPort names instead of probing private AppHost-style methods.
    def _plc(self) -> PlcCommandPort:
        if isinstance(self._motion, PlcCommandPort):
            return self._motion
        raise RuntimeError("MotionPort does not provide PlcCommandPort — use AppDeviceGateway or a compatible port")

    def base_for_axis(self, axis: int) -> int:
        return self._plc().base_for_axis(int(axis))

    def write_regs(self, addr: int, values: Any) -> None:
        self._plc().write_regs(addr, list(values))

    def set_cmd_bits(self, axis: int, *, set_mask: int = 0, clr_mask: int = 0) -> None:
        self._plc().set_cmd_bits(axis, set_mask=set_mask, clr_mask=clr_mask)

    def pulse_cmd_bits(self, axis: int, mask: int, pulse_ms: int = 120) -> None:
        self._plc().pulse_cmd_bits(axis, mask, pulse_ms=pulse_ms)

    def start_velocity_move(
        self, axis: int, vel_velmove: float, *, acc: float = 80.0, dec: float = 80.0, jerk: float = 300.0,
    ) -> None:
        self._plc().start_velocity_move(axis, vel_velmove, acc=acc, dec=dec, jerk=jerk)

    def get_ax0_z_disp_limits(self) -> tuple[float, float, float]:
        return self._plc().get_ax0_z_disp_limits()

    # -- sensor delegate --------------------------------------------------
    @property
    def latest_ax3_angle_deg(self) -> float | None:
        return self._sensors.latest_ax3_angle_deg

    @property
    def latest_cl145(self) -> Any:
        return self._sensors.latest_cl145

    @property
    def latest_cl3(self) -> Any:
        return self._sensors.latest_cl3

    def get_recipe_copy(self) -> Any:
        return self._sensors.get_recipe_copy()

    def get_calibration_snapshot(self) -> Any:
        return self._sensors.get_calibration_snapshot()

    @property
    def sim_gauge_enabled(self) -> bool:
        return self._sensors.sim_gauge_enabled

    @property
    def sim_disp_enabled(self) -> bool:
        return self._sensors.sim_disp_enabled

    def simulate_gauge_once(self, recipe: Any) -> tuple[float, str]:
        return self._sensors.simulate_gauge_once(recipe)

    def simulate_disp_once(self, recipe: Any) -> tuple[float, str]:
        return self._sensors.simulate_disp_once(recipe)

    def calc_id_single_from_out2(self, th: list, o2: list, recipe: Any) -> dict:
        return self._sensors.calc_id_single_from_out2(th, o2, recipe)

    @property
    def axis_cal(self) -> Any:
        return self._sensors.axis_cal

    @property
    def gauge_worker(self) -> Any:
        return self._sensors.gauge_worker

    # -- operator delegate ------------------------------------------------
    def get_x_point(self, point: int) -> int:
        return self._operator.get_x_point(point)

    def get_y_point(self, point: int) -> int:
        return self._operator.get_y_point(point)

    def plc_write_y_point(self, point: int, value: int) -> None:
        self._operator.plc_write_y_point(point, value)

    def operator_confirm(self, title: str, message: str, *, allow_stop: bool = True, timeout_s: float | None = None) -> str:
        return self._operator.operator_confirm(title, message, allow_stop=allow_stop, timeout_s=timeout_s)

    # -- misc -------------------------------------------------------------
    def _log_ax3_speed_trace(self, tag: str, recipe_obj: Any = None) -> None:
        pass  # debug-logging no-op

    def _apply_start_anchor_from_recipe(self) -> None:
        pass  # no-op in port mode


class _NoOpQueue:
    def put(self, *a: Any, **kw: Any) -> None:
        pass


class _NoOpEventSink:
    """EventSink that discards all events — used when no real sink is available."""

    def publish_state(self, state: str, message: str) -> None: pass
    def publish_progress(self, *, section_index: int, section_total: int, z_pos_mm: float, ax0_abs: float) -> None: pass
    def publish_length(self, payload: Any) -> None: pass
    def publish_coverage(self, payload: Any) -> None: pass
    def publish_raw_points(self, points: Any) -> None: pass
    def publish_row(self, row: Any) -> None: pass
    def publish_straightness(self, payload: Any) -> None: pass
    def publish_postcalc(self, payload: Any) -> None: pass
    def publish_auto_state(self, state: str, message: str) -> None: pass
    def publish_auto_done(self, message: str) -> None: pass
    def publish_auto_error(self, message: str) -> None: pass
    def publish_auto_row(self, row: Any) -> None: pass
    def publish_auto_len(self, payload: Any) -> None: pass
    def publish_auto_progress(self, *, section_index: int, section_total: int, z_pos_mm: float, ax0_abs: float) -> None: pass
    def publish_auto_cov(self, payload: Any) -> None: pass
    def publish_auto_raw_points(self, points: Any) -> None: pass
    def publish_auto_clear(self) -> None: pass
    def publish_auto_postcalc(self, payload: Any) -> None: pass


class AutoFlowOrchestrator:
    """Explicit dependency shell for the formal measurement workflow."""

    def __init__(
        self,
        gateway: DeviceGateway,
        recipe: Recipe,
        calibration: CalibrationSnapshot,
        run_session: RunSession,
        event_sink: EventSink,
        *,
        motion: MotionPort | None = None,
        sensors: SensorPort | None = None,
        operator: OperatorPort | None = None,
        runtime_state: RuntimeState | None = None,
        run_repository: RunRepositoryProtocol | None = None,
    ) -> None:
        self.gateway = gateway
        self.motion: MotionPort = motion if motion is not None else gateway
        self.sensors: SensorPort = sensors if sensors is not None else gateway  # type: ignore[assignment]
        self.operator: OperatorPort = operator if operator is not None else gateway  # type: ignore[assignment]
        self.recipe = recipe
        self.calibration = calibration
        self.run_session = run_session
        self.event_sink = event_sink
        self.runtime_state = runtime_state or RuntimeState.from_run_session(run_session)
        self.run_repository = run_repository
        self.production_workflow = (
            ProductionWorkflow(
                recipe=recipe,
                calibration=calibration,
                runtime_state=self.runtime_state,
                gateway=gateway,
                run_repository=run_repository,
            )
            if run_repository is not None
            else None
        )
        self.run_result: RunResult | None = None
        self.state = "IDLE"
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._legacy_flow: AutoFlow | None = None
        self._return_standby_after_stop = False
        if self.motion is not None and self.sensors is not None and self.operator is not None:
            adapter = _LegacyAppAdapter(self.motion, self.sensors, self.operator)
            self._legacy_flow = AutoFlow(
                device=self.motion, event_sink=self.event_sink,
                motion=self.motion, sensors=self.sensors,
                operator=self.operator, plc=adapter,
            )
            self._legacy_flow.stop_event = self._stop_event
            self._legacy_flow.set_runtime_context(recipe, calibration)

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    def is_alive(self) -> bool:
        """Compatibility helper so App can A/B old and new runners easily."""
        return self.is_running

    def start(self) -> None:
        """Start the orchestrator on a background thread."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._return_standby_after_stop = False
        self.run_session.end_ts = None
        self._set_internal_state("STARTING")
        self._thread = threading.Thread(
            target=self.run,
            name="AutoFlowOrchestrator",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Request the orchestrator to stop gracefully."""
        if not self.is_running:
            return
        self._return_standby_after_stop = True
        self._stop_event.set()
        self._set_internal_state("STOPPING")
        self._emit_state("STOPPING", "Stop request received")

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def run(self) -> None:
        """Workflow entrypoint for the staged measurement orchestrator."""
        self._prepare_run_context()

        status: RunResultStatus = "DONE"
        message = "Measurement completed"
        try:
            self._run_main_loop()
        except _StopRequested as exc:
            status = "STOP"
            message = str(exc) or "User stopped"
            self._set_internal_state("STOPPED")
        except Exception as exc:
            status = "ERR"
            message = str(exc) or f"{type(exc).__name__}: {exc!r}"
            self._set_internal_state("ERROR")
        finally:
            self._finalize_run(status, message)

    # -- Phase 5: prepare run context step extraction ----------------------

    def _prepare_run_context_impl(self) -> None:
        if self.run_session.start_ts is None:
            self.run_session.start_ts = time.time()
        self.run_session.end_ts = None
        self.runtime_state.started_at_ts = self.run_session.start_ts
        self.runtime_state.finished_at_ts = None
        if self.production_workflow is not None:
            try:
                self.production_workflow.ensure_identity()
            except Exception:
                pass
        self._set_internal_state("RUNNING")
        self._emit_state("RUN", "Auto measurement started")

    def _prepare_run_context(self) -> None:
        PrepareRunContextStep(self).execute()

    def _run_main_loop(self) -> None:
        centers_xyz: list[tuple[float, float, float]] = []
        centers_xyz_id: list[tuple[float, float, float]] = []
        concentricity_list: list[float] = []

        self._apply_start_anchor_if_available()
        self._prepare_linear_axes()
        self._prepare_ax2_and_clamps()
        self._run_optional_length_stage()
        self._move_ax2_to_rotate_position()
        section_plan = self._build_section_plan()
        if not section_plan.sections:
            raise ValueError("section_count must be > 0")
        self._confirm_rotate_clamp()
        self._prepare_ax3_rotation()
        self._run_section_loop(
            section_plan,
            centers_xyz=centers_xyz,
            centers_xyz_id=centers_xyz_id,
            concentricity_list=concentricity_list,
        )
        self._run_postcalc(
            centers_xyz=centers_xyz,
            centers_xyz_id=centers_xyz_id,
            concentricity_list=concentricity_list,
        )
        self._stop_ax3_rotation()
        self._return_to_standby()

    # -- Phase 5: finalize step extraction ---------------------------------

    def _finalize_run_impl(self, status: RunResultStatus, message: str) -> None:
        self.run_session.end_ts = time.time()
        try:
            self.motion.stop(3)
        except Exception:
            pass
        if self._stop_event.is_set():
            try:
                self.motion.abort_motion()
            except Exception:
                pass
            if self._return_standby_after_stop:
                self._return_to_standby_after_user_stop()

        if status == "DONE":
            self._set_internal_state("DONE")
        if self.production_workflow is not None:
            try:
                self.run_result = self.production_workflow.build_run_result(
                    status=status,
                    message=message,
                    finished_at_ts=self.run_session.end_ts,
                )
            except Exception:
                self.run_result = None
        self._emit_state(status, message)

    def _finalize_run(self, status: RunResultStatus, message: str) -> None:
        FinalizeRunStep(self, status, message).execute()

    def _prepare_linear_axes(self) -> None:
        for axis in (0, 1, 4):
            self._ensure_axis_ready(axis)

    def _prepare_ax2_and_clamps(self) -> None:
        self._ensure_axis_ready(2)
        if self._clamps_are_closed():
            self._emit_state("PREP", "Clamp already closed; skip clamp output")
            return

        self._emit_state("PREP", "Clamp prepare: close dual clamps")
        self._write_y_point(10, 1)
        self._write_y_point(11, 1)

        wait_s = float(getattr(self.recipe, "clamp_confirm_wait_s", 3.0) or 0.0)
        if wait_s < 0.0:
            self._operator_confirm_or_stop(
                "Clamp Confirm",
                "Confirm clamps are closed.\n\nX3: continue\nX4: cancel workflow",
                timeout_s=None,
            )
            return

        if wait_s > 0.0:
            self._emit_state("PREP", f"Clamp close wait {wait_s:.3f}s")
            self._wait_cancelable(wait_s)

    def _run_optional_length_stage(self) -> None:
        if not bool(getattr(self.recipe, "len_enable", False)):
            return

        current_ax2_abs = float(getattr(self.motion.get_axis_copy(2), "act_pos", 0.0) or 0.0)
        ax2_plan = resolve_ax2_position_plan(self.recipe, current_ax2_abs=current_ax2_abs)
        if ax2_plan.has_length_target:
            self._move_axis_abs(
                2,
                float(cast(Any, ax2_plan.length_target_abs)),
                strict=True,
                context="AUTO_AX2_LEN",
                state="PREP",
                message_template="AX2 -> length position: {target:.3f}",
            )
        else:
            self._emit_state("WARN", "Length enabled but AX2 length position is not saved")

        payload = self._measure_length_legacy()
        if self.production_workflow is not None:
            self.production_workflow.record_length(payload)
        self.event_sink.publish_length(payload)
        try:
            self.run_session.length_result = dict(payload) if isinstance(payload, dict) else None
        except Exception:
            pass

        standby_plan = resolve_standby_plan(self.recipe)
        if standby_plan.enabled and 0 in standby_plan.targets_abs:
            try:
                self._move_axis_abs(
                    0,
                    float(standby_plan.targets_abs[0]),
                    strict=True,
                    context="AUTO_AX0_STANDBY_AFTER_LEN",
                    state="PREP",
                    message_template="AX0 -> standby: {target:.3f}",
                )
            except Exception as exc:
                self._emit_state("WARN", f"AX0 standby move failed: {exc}")

        self._raise_if_stop_requested()

    def _move_ax2_to_rotate_position(self) -> None:
        if not bool(getattr(self.recipe, "len_enable", False)):
            self._verify_ax2_rotate_position_when_length_disabled()
            return

        self._move_axis_abs(
            2,
            require_ax2_rotate_target_abs(self.recipe),
            strict=True,
            context="AUTO_AX2_ROT",
            state="PREP",
            message_template="AX2 -> rotate position: {target:.3f}",
        )

    def _confirm_rotate_clamp(self) -> None:
        self._raise_if_stop_requested()

    def _operator_confirm_or_stop(
        self,
        title: str,
        message: str,
        *,
        timeout_s: float | None,
    ) -> None:
        result = "timeout"
        try:
            result = self.operator.operator_confirm(
                title,
                message,
                allow_stop=True,
                timeout_s=timeout_s,
            )
        except Exception:
            result = "timeout"
        if result != "confirm":
            self._return_standby_after_stop = True
            raise _StopRequested(f"Operator canceled: {result}")

    def _verify_ax2_rotate_position_when_length_disabled(self, tolerance_mm: float = 10.0) -> None:
        if not bool(getattr(self.recipe, "ax2_rot_valid", False)):
            self._operator_confirm_or_stop(
                "AX2 Position Confirm",
                "Length detection is disabled, but AX2 rotate position is not saved.\n\nX3: continue\nX4: cancel workflow",
                timeout_s=None,
            )
            return

        target = float(getattr(self.recipe, "ax2_rot_abs", 0.0) or 0.0)
        current = float(getattr(self.motion.get_axis_copy(2), "act_pos", 0.0) or 0.0)
        delta = current - target
        if abs(delta) <= float(tolerance_mm):
            self._emit_state("PREP", f"AX2 position verified: current {current:.3f}, target {target:.3f}")
            return

        self._operator_confirm_or_stop(
            "AX2 Position Deviation",
            (
                "Length detection is disabled; AX2 will not move automatically.\n\n"
                f"Current: {current:.3f} mm\n"
                f"Target: {target:.3f} mm\n"
                f"Delta: {delta:.3f} mm\n\n"
                "X3: continue\nX4: cancel workflow"
            ),
            timeout_s=None,
        )

    def _prepare_ax3_rotation(self) -> None:
        self._ensure_axis_ready(3)
        self._start_ax3_rotation(emit_state=True)

    def _run_section_loop(
        self,
        section_plan,
        *,
        centers_xyz: list[tuple[float, float, float]],
        centers_xyz_id: list[tuple[float, float, float]],
        concentricity_list: list[float],
    ) -> None:
        section_total = len(section_plan.sections)
        for row in section_plan.sections:
            self._execute_section(
                row,
                section_total=section_total,
                centers_xyz=centers_xyz,
                centers_xyz_id=centers_xyz_id,
                concentricity_list=concentricity_list,
            )

    def _execute_section(
        self,
        section,
        *,
        section_total: int,
        centers_xyz: list[tuple[float, float, float]],
        centers_xyz_id: list[tuple[float, float, float]],
        concentricity_list: list[float],
    ) -> None:
        context = SectionExecutionContext(
            section=section,
            section_index=int(section.section_index),
            total_sections=section_total,
            centers_xyz=centers_xyz,
            centers_xyz_id=centers_xyz_id,
            concentricity_list=concentricity_list,
        )
        SectionExecutionStep(self).execute(context)

    def _execute_section_impl(self, context: SectionExecutionContext) -> None:
        section = context.section
        section_index = context.section_index
        section_total = context.total_sections

        self._raise_if_stop_requested()
        z_pos_mm = float(section.z_od_disp)
        targets = section.linear_targets()
        self._emit_progress(
            section_index=section_index,
            section_total=section_total,
            z_pos_mm=float(z_pos_mm),
            ax0_abs=float(section.ax0_abs),
        )
        self._emit_state("RUN", f"Section {section_index}/{section_total} positioning")
        self._move_linear_axes_to_targets(
            targets,
            context=f"AUTO_SEC_{section_index}",
        )
        self._wait_before_section_capture(
            section_index=section_index,
            section_total=section_total,
            delay_s=float(getattr(self.recipe, "sample_delay_s", 0.0) or 0.0),
        )
        SectionCaptureStep(self).execute(context)

    def _capture_section_impl(self, context: SectionExecutionContext) -> None:
        section = context.section
        section_index = context.section_index
        centers_xyz = context.centers_xyz
        centers_xyz_id = context.centers_xyz_id
        concentricity_list = context.concentricity_list
        z_pos_mm = float(section.z_od_disp)

        self._measure_section(
            section_index=section_index,
            z_pos_mm=float(z_pos_mm),
            x_abs=float(section.ax0_abs),
            centers_xyz=centers_xyz,
            centers_xyz_id=centers_xyz_id,
            concentricity_list=concentricity_list,
        )

    def _wait_before_section_capture(
        self,
        *,
        section_index: int,
        section_total: int,
        delay_s: float,
    ) -> None:
        delay = float(delay_s or 0.0)
        if delay <= 0.0:
            return
        self._emit_state(
            "RUN",
            f"Section {int(section_index)}/{int(section_total)} wait sample delay: {delay:.3f}s",
        )
        deadline = time.monotonic() + delay
        while True:
            self._raise_if_stop_requested()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            time.sleep(min(0.05, remaining))

    def _wait_cancelable(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, float(seconds))
        while True:
            self._raise_if_stop_requested()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            time.sleep(min(0.05, remaining))

    def _measure_section(
        self,
        *,
        section_index: int,
        z_pos_mm: float,
        x_abs: float,
        centers_xyz: list[tuple[float, float, float]],
        centers_xyz_id: list[tuple[float, float, float]],
        concentricity_list: list[float],
    ) -> None:
        context = MeasureSectionContext(
            section_index=section_index,
            z_pos_mm=z_pos_mm,
            x_abs=x_abs,
            centers_xyz=centers_xyz,
            centers_xyz_id=centers_xyz_id,
            concentricity_list=concentricity_list,
        )
        MeasureSectionStep(self).execute(context)

    def _measure_section_impl(self, context: MeasureSectionContext) -> None:
        section_index = context.section_index

        sampling_result = SamplingStep(self).execute(context)
        scan_mode = sampling_result.scan_mode
        keep_spinning = sampling_result.keep_spinning
        primary_sample = sampling_result.primary_sample
        id_sample = sampling_result.id_sample
        raw_points = sampling_result.raw_points
        split_shift_deg = sampling_result.split_shift_deg
        coax_unreliable = sampling_result.coax_unreliable

        PublishEventsStep(self).execute(PublishEventsContext(
            measure_context=context,
            raw_points=raw_points,
        ))
        coverage_payload = _build_validation_coverage_payload(
            primary_sample=primary_sample,
            id_sample=id_sample,
            section_index=section_index,
            scan_mode=scan_mode,
            split_shift_deg=split_shift_deg,
            coax_unreliable=coax_unreliable,
            keep_spinning=keep_spinning,
        )
        PublishEventsStep(self).execute(PublishEventsContext(
            measure_context=context,
            coverage_payload=coverage_payload,
        ))

        row_build_result = RowBuildStep(self).execute(context, sampling_result)
        row = row_build_result.row
        if self.production_workflow is not None:
            RecordRowStep(self).execute(row)
        PublishEventsStep(self).execute(PublishEventsContext(
            measure_context=context,
            row=row,
        ))

    def _publish_section_events_impl(self, context: PublishEventsContext) -> None:
        measure_context = context.measure_context
        if context.raw_points is not None:
            self._publish_section_raw_points(
                raw_points=context.raw_points,
                section_index=measure_context.section_index,
                z_pos_mm=float(measure_context.z_pos_mm),
            )
        if context.coverage_payload is not None:
            self._publish_section_coverage(
                payload=context.coverage_payload,
            )
        if context.row is not None:
            self.event_sink.publish_row(context.row)

    def _record_row_impl(self, row: Any) -> None:
        cast(Any, self.production_workflow).record_row(row)

    def _build_row_impl(
        self,
        context: MeasureSectionContext,
        sampling_result: SectionSamplingResult,
    ) -> RowBuildResult:
        section_index = context.section_index
        z_pos_mm = context.z_pos_mm
        x_abs = context.x_abs
        centers_xyz = context.centers_xyz
        centers_xyz_id = context.centers_xyz_id
        concentricity_list = context.concentricity_list

        scan_mode = sampling_result.scan_mode
        primary_sample = sampling_result.primary_sample
        id_sample = sampling_result.id_sample
        coords_od = sampling_result.coords_od
        coords_id = sampling_result.coords_id
        raw_od = sampling_result.raw_od
        raw_id = sampling_result.raw_id
        raw_points = sampling_result.raw_points
        split_shift_deg = sampling_result.split_shift_deg
        coax_unreliable = sampling_result.coax_unreliable

        row = self._build_section_row(
            section_index=section_index,
            z_pos_mm=float(z_pos_mm),
            x_abs=float(x_abs),
            coords_od=coords_od,
            coords_id=coords_id,
            raw_od=str(raw_od),
            raw_id=str(raw_id),
            raw_points=raw_points,
            fit_weights_od=primary_sample.fit_weights_od,
            fit_weights_id=(id_sample.fit_weights_id if id_sample is not None else primary_sample.fit_weights_id),
            scan_mode=scan_mode,
            split_shift_deg=split_shift_deg,
            coax_unreliable=coax_unreliable,
            centers_xyz=centers_xyz,
            centers_xyz_id=centers_xyz_id,
            concentricity_list=concentricity_list,
        )
        return RowBuildResult(row=row)

    def _sample_section_impl(self, context: MeasureSectionContext) -> SectionSamplingResult:
        section_index = context.section_index
        z_pos_mm = context.z_pos_mm
        x_abs = context.x_abs

        legacy = self._require_legacy_flow()
        recipe = self.recipe
        i = int(section_index) - 1

        scan_mode = _resolve_recipe_sampling_mode(recipe)
        split_shift_deg = None
        coax_unreliable = None
        keep_spinning = bool(getattr(recipe, "split_keep_spinning", True))
        slip_check = bool(getattr(recipe, "split_slip_check", True))
        slip_max_deg = float(getattr(recipe, "split_slip_max_deg", 5.0) or 5.0)
        omega_cv_max = float(getattr(recipe, "split_omega_cv_max", 0.25) or 0.25)
        primary_sample: SamplingResult
        id_sample: SamplingResult | None = None

        try:
            legacy_log(
                "SECTION_START",
                section=section_index,
                z_disp=z_pos_mm,
                ax0_abs=x_abs,
            )
        except Exception:
            pass

        if scan_mode == "SPLIT":
            od_sample = legacy.sample_circle_points_result(
                recipe,
                section_idx=i,
                sample_od=True,
                sample_id=False,
                phase="OD",
            )

            if not keep_spinning:
                RotationControlStep(self).restart_for_split()

            id_sample = legacy.sample_circle_points_result(
                recipe,
                section_idx=i,
                sample_od=False,
                sample_id=True,
                phase="ID",
            )
            primary_sample = od_sample
            coords_od = od_sample.coords_od
            coords_id = id_sample.coords_id
            raw_od = od_sample.raw_od
            raw_id = id_sample.raw_id
            raw_points_od = od_sample.raw_points
            raw_points_id = id_sample.raw_points

            if slip_check:
                try:
                    split_shift_deg, coax_unreliable = _split_slip_diag(
                        raw_points_od=raw_points_od,
                        raw_points_id=raw_points_id,
                        slip_max_deg=float(slip_max_deg),
                        omega_cv_max=float(omega_cv_max),
                    )
                except Exception:
                    split_shift_deg, coax_unreliable = None, None

            raw_points = list(raw_points_od or []) + list(raw_points_id or [])
        else:
            sync_sample = legacy.sample_circle_points_result(
                recipe,
                section_idx=i,
                sample_od=True,
                sample_id=True,
                phase="SYNC",
            )
            primary_sample = sync_sample
            coords_od = sync_sample.coords_od
            coords_id = sync_sample.coords_id
            raw_od = sync_sample.raw_od
            raw_id = sync_sample.raw_id
            raw_points = sync_sample.raw_points

        return SectionSamplingResult(
            scan_mode=scan_mode,
            keep_spinning=keep_spinning,
            primary_sample=primary_sample,
            id_sample=id_sample,
            coords_od=coords_od,
            coords_id=coords_id,
            raw_od=raw_od,
            raw_id=raw_id,
            raw_points=raw_points,
            split_shift_deg=split_shift_deg,
            coax_unreliable=coax_unreliable,
        )

    def _restart_rotation_for_split_impl(self) -> None:
        try:
            self._stop_ax3_rotation()
        except Exception:
            pass
        try:
            self._start_ax3_rotation(emit_state=False)
        except Exception:
            pass

    def _publish_section_raw_points(
        self,
        *,
        raw_points: list[dict],
        section_index: int,
        z_pos_mm: float,
    ) -> None:
        try:
            for j, point in enumerate(raw_points):
                if isinstance(point, dict):
                    point["section_idx"] = int(section_index)
                    point["z_pos_mm"] = float(z_pos_mm)
                    point["sample_idx"] = int(j)
        except Exception:
            pass
        if self.production_workflow is not None:
            self.production_workflow.record_raw_points(raw_points)
        self.event_sink.publish_raw_points(raw_points)

    def _publish_section_coverage(
        self,
        *,
        payload: dict[str, Any],
    ) -> None:
        try:
            if self.production_workflow is not None:
                self.production_workflow.record_coverage(payload)
            self.event_sink.publish_coverage(payload)
        except Exception:
            pass

    def _build_section_row(
        self,
        *,
        section_index: int,
        z_pos_mm: float,
        x_abs: float,
        coords_od: np.ndarray,
        coords_id: np.ndarray,
        raw_od: str,
        raw_id: str,
        raw_points: list[dict],
        fit_weights_od: Any,
        fit_weights_id: Any,
        scan_mode: str,
        split_shift_deg: float | None,
        coax_unreliable: bool | None,
        centers_xyz: list[tuple[float, float, float]],
        centers_xyz_id: list[tuple[float, float, float]],
        concentricity_list: list[float],
    ) -> MeasureRow:
        return _build_measure_row_from_sampling(
            MeasureRowBuildInputs(
                legacy=self._require_legacy_flow(),
                recipe=self.recipe,
                sensors=self.sensors,
                section_index=section_index,
                z_pos_mm=z_pos_mm,
                x_abs=x_abs,
                coords_od=coords_od,
                coords_id=coords_id,
                raw_od=raw_od,
                raw_id=raw_id,
                raw_points=raw_points,
                fit_weights_od=fit_weights_od,
                fit_weights_id=fit_weights_id,
                scan_mode=scan_mode,
                split_shift_deg=split_shift_deg,
                coax_unreliable=coax_unreliable,
                centers_xyz=centers_xyz,
                centers_xyz_id=centers_xyz_id,
                concentricity_list=concentricity_list,
                calibration=getattr(self, "calibration", None),
            )
        )

    def _run_postcalc(
        self,
        *,
        centers_xyz: list[tuple[float, float, float]],
        centers_xyz_id: list[tuple[float, float, float]],
        concentricity_list: list[float],
    ) -> None:
        PostcalcSummaryStep(self).execute(
            centers_xyz=centers_xyz,
            centers_xyz_id=centers_xyz_id,
            concentricity_list=concentricity_list,
        )

    def _run_postcalc_impl(
        self,
        *,
        centers_xyz: list[tuple[float, float, float]],
        centers_xyz_id: list[tuple[float, float, float]],
        concentricity_list: list[float],
    ) -> None:
        recipe = self.recipe
        try:
            result = compute_postcalc_result(
                centers_xyz,
                centers_xyz_id,
                concentricity_list=concentricity_list,
                id_single_enable=bool(getattr(recipe, "id_single_enable", False)),
            )
            if self.production_workflow is not None:
                self.production_workflow.record_summary(result.straightness_payload, source="straightness")
            self.event_sink.publish_straightness(result.straightness_payload)
            if self.production_workflow is not None:
                self.production_workflow.record_summary(result.postcalc_payload, source="postcalc")
            self.event_sink.publish_postcalc(result.postcalc_payload)
        except Exception:
            if self.production_workflow is not None:
                self.production_workflow.record_summary(
                    {
                        "straight_od": None,
                        "straight_id": None,
                        "axis_dist": None,
                        "conc_max": None,
                        "axis_span_max": None,
                    },
                    source="straightness",
                )
            self.event_sink.publish_straightness(
                {
                    "straight_od": None,
                    "straight_id": None,
                    "axis_dist": None,
                    "conc_max": None,
                    "axis_span_max": None,
                }
            )

    def _return_to_standby(self) -> None:
        standby_plan = resolve_standby_plan(self.recipe)
        if not standby_plan.enabled:
            return
        try:
            self._move_linear_axes_to_targets(
                dict(standby_plan.targets_abs),
                context="AUTO_STANDBY",
                strict=False,
            )
        except Exception:
            pass

    def _return_to_standby_after_user_stop(self) -> None:
        standby_plan = resolve_standby_plan(self.recipe)
        if not standby_plan.enabled:
            self._emit_state("STOPPING", "Standby position is not saved; skip AX0/AX1/AX4 return")
            return
        try:
            self._emit_state("STOPPING", "Return AX0/AX1/AX4 to standby after stop")
            resolved: dict[int, float] = {}
            for axis, target in dict(standby_plan.targets_abs).items():
                resolved[int(axis)] = self.motion.apply_soft_limits_abs(
                    int(axis),
                    float(target),
                    strict=False,
                    context="AUTO_STOP_STANDBY",
                )
            for axis, target in resolved.items():
                self.motion.movea_abs(int(axis), float(target), context="AUTO_STOP_STANDBY")
            for axis, target in resolved.items():
                self._wait_in_position_ignoring_user_stop(
                    int(axis),
                    float(target),
                    pos_tol=0.05,
                    timeout_s=30.0,
                )
        except Exception as exc:
            self._emit_state("WARN", f"Return standby after stop failed: {exc}")

    def _measure_length_legacy(self) -> dict[str, Any]:
        if not bool(getattr(self.recipe, "len_enable", False)):
            return {
                "enabled": False,
                "skipped": True,
                "ok": False,
                "reason": "DISABLED",
                "z_low": None,
                "z_high": None,
                "length_mm": None,
                "t_s": 0.0,
            }
        if not bool(getattr(self.recipe, "ax2_len_valid", False)):
            return {
                "enabled": True,
                "skipped": True,
                "ok": False,
                "reason": "NO_AX2_LEN_POS",
                "z_low": None,
                "z_high": None,
                "length_mm": None,
                "t_s": 0.0,
            }
        if self._legacy_flow is None:
            return {
                "enabled": True,
                "skipped": True,
                "ok": False,
                "reason": "ORCHESTRATOR_STAGE_ONLY",
                "z_low": None,
                "z_high": None,
                "length_mm": None,
                "t_s": 0.0,
            }

        self._emit_state("LEN", "Auto length measurement")
        try:
            return dict(self._require_legacy_flow().measure_length_result(self.recipe))
        except Exception as exc:
            return {
                "enabled": True,
                "skipped": False,
                "ok": False,
                "reason": f"EXC({exc})",
                "z_low": None,
                "z_high": None,
                "length_mm": None,
                "t_s": 0.0,
            }

    def _resolve_section_positions(self) -> list[float]:
        return list(plan_section_positions(self.recipe).positions_z)

    def _build_section_plan_impl(self):
        axis_cal = self._require_axis_cal()
        soft_limits = {
            0: self._soft_limits_from_axis(0),
            1: self._soft_limits_from_axis(1),
            4: self._soft_limits_from_axis(4),
        }
        return build_recipe_section_plan(
            self.recipe,
            axis_cal,
            soft_limits_abs=soft_limits,
        )

    def _build_section_plan(self):
        return BuildSectionPlanStep(self).execute()

    def _resolve_section_targets(
        self,
        *,
        axis_cal: AxisCal,
        section_index: int,
        z_pos_mm: float,
    ) -> dict[int, float]:
        del axis_cal
        del z_pos_mm
        row = self._build_section_plan().section_at(section_index)
        targets = row.linear_targets()
        for axis, target in list(targets.items()):
            targets[axis] = self.motion.apply_soft_limits_abs(
                int(axis),
                float(target),
                strict=True,
                context=f"AUTO_SEC_{section_index}",
            )
        return targets

    def _move_linear_axes_to_targets(
        self,
        targets: dict[int, float],
        *,
        context: str,
        strict: bool = True,
    ) -> None:
        resolved: dict[int, float] = {}
        for axis, target in targets.items():
            resolved[int(axis)] = self.motion.apply_soft_limits_abs(
                int(axis),
                float(target),
                strict=bool(strict),
                context=context,
            )
        for axis, target in resolved.items():
            self.motion.movea_abs(int(axis), float(target), context=context)
        for axis, target in resolved.items():
            ok = self._wait_in_position(int(axis), float(target), pos_tol=0.05, timeout_s=30.0)
            if not ok:
                self._raise_if_stop_requested()
                raise TimeoutError(f"AX{axis} in-position timeout: {target:.3f}")

    def _move_axis_abs(
        self,
        axis: int,
        target: float,
        *,
        strict: bool,
        context: str,
        state: str,
        message_template: str,
    ) -> None:
        target_resolved = self.motion.apply_soft_limits_abs(
            int(axis),
            float(target),
            strict=bool(strict),
            context=context,
        )
        self._emit_state(state, message_template.format(target=float(target_resolved)))
        self.motion.movea_abs(int(axis), float(target_resolved), context=context)
        ok = self._wait_in_position(int(axis), float(target_resolved), pos_tol=0.05, timeout_s=25.0)
        if not ok:
            self._raise_if_stop_requested()
            raise TimeoutError(f"AX{axis} in-position timeout: {target_resolved:.3f}")

    def _start_ax3_rotation(self, *, emit_state: bool) -> None:
        velocity = self._get_ax3_velocity()
        if emit_state:
            self._emit_state("PREP", f"AX3 rotate start: {velocity:.3f}")
        self.motion.velmove(3, float(velocity))
        time.sleep(0.20)

    def _stop_ax3_rotation(self) -> None:
        try:
            self.motion.stop(3)
            t0 = time.time()
            while (time.time() - t0) < 10.0:
                self._raise_if_stop_requested()
                ac3 = self.motion.get_axis_copy(3)
                if not self._is_moving(int(getattr(ac3, "sts", 0))):
                    break
                time.sleep(0.08)
        except _StopRequested:
            raise
        except Exception:
            pass

    def _get_ax3_velocity(self) -> float:
        try:
            velocity = float(getattr(self.recipe, "rot_vel_velmove", 0.0) or 0.0)
        except Exception:
            velocity = 0.0
        if abs(velocity) <= 1e-9:
            velocity = 200.0
        return float(velocity)

    def _fit_line_and_dist(
        self, points_xyz: list[tuple[float, float, float]]
    ) -> tuple[float, list[float], np.ndarray, np.ndarray]:
        if len(points_xyz) < 2:
            return 0.0, [0.0 for _ in points_xyz], np.zeros(3, dtype=float), np.array([0.0, 0.0, 1.0], dtype=float)
        P = np.array(points_xyz, dtype=float)
        p0 = P.mean(axis=0)
        Q = P - p0
        C = (Q.T @ Q) / max(1, Q.shape[0])
        w, v = np.linalg.eigh(C)
        d = v[:, int(np.argmax(w))]
        d = d / (np.linalg.norm(d) + 1e-12)
        t = Q @ d
        proj = np.outer(t, d)
        R = Q - proj
        dist = np.linalg.norm(R, axis=1)
        straight = float(dist.max() - dist.min()) if dist.size else 0.0
        return straight, [float(x) for x in dist.tolist()], p0, d

    def _line_distance(self, p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> float:
        d1n = d1 / (np.linalg.norm(d1) + 1e-12)
        d2n = d2 / (np.linalg.norm(d2) + 1e-12)
        n = np.cross(d1n, d2n)
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            v = p2 - p1
            return float(np.linalg.norm(np.cross(v, d1n)))
        return float(abs(np.dot((p2 - p1), n)) / nn)

    def _tilt_and_end_offset(
        self,
        p0: np.ndarray,
        d: np.ndarray,
        pts_xyz: list[tuple[float, float, float]],
    ) -> tuple[float | None, float | None, float | None]:
        try:
            if not pts_xyz or len(pts_xyz) < 2:
                return None, None, None
            z_list = [float(p[2]) for p in pts_xyz]
            z_min = float(min(z_list))
            z_max = float(max(z_list))
            dz = float(d[2])
            if abs(dz) < 1e-12:
                return None, None, None
            sx = float(d[0] / dz)
            sy = float(d[1] / dz)
            slope = float(math.hypot(sx, sy))
            tilt_deg = float(math.degrees(math.atan(slope)))
            t_min = (z_min - float(p0[2])) / dz
            t_max = (z_max - float(p0[2])) / dz
            p_min = p0 + t_min * d
            p_max = p0 + t_max * d
            end_off = float(math.hypot(float(p_max[0] - p_min[0]), float(p_max[1] - p_min[1])))
            return tilt_deg, end_off, slope
        except Exception:
            return None, None, None

    def _apply_start_anchor_if_available(self) -> None:
        plan = resolve_start_anchor_plan(self.recipe)
        if not plan.enabled:
            return
        apply_start = getattr(self.motion, "_apply_start_anchor_from_recipe", None)
        if callable(apply_start):
            apply_start()

    def _ensure_axis_ready(self, axis: int) -> None:
        snapshot = self.motion.get_axis_copy(int(axis))
        sts = int(getattr(snapshot, "sts", 0) or 0)
        err = int(getattr(snapshot, "err", 0) or 0)
        if self._is_fault(sts, err):
            raise RuntimeError(f"AX{axis} fault, err={err}")
        if not self._is_enabled(sts):
            self.motion.enable(int(axis))
            time.sleep(0.15)

    def _wait_in_position(self, axis: int, target_abs: float, *, pos_tol: float, timeout_s: float) -> bool:
        if self._legacy_flow is not None:
            return bool(self._legacy_flow.wait_in_position_result(int(axis), float(target_abs), float(pos_tol), float(timeout_s)))

        t0 = time.time()
        while (time.time() - t0) < float(timeout_s):
            self._raise_if_stop_requested()
            snapshot = self.motion.get_axis_copy(int(axis))
            sts = int(getattr(snapshot, "sts", 0) or 0)
            err = int(getattr(snapshot, "err", 0) or 0)
            if self._is_fault(sts, err):
                raise RuntimeError(f"AX{axis} fault, err={err}")
            pos_err = abs(float(getattr(snapshot, "act_pos", 0.0) or 0.0) - float(target_abs))
            if pos_err <= float(pos_tol) and (not self._is_moving(sts)):
                return True
            time.sleep(0.08)
        return False

    def _wait_in_position_ignoring_user_stop(
        self,
        axis: int,
        target_abs: float,
        *,
        pos_tol: float,
        timeout_s: float,
    ) -> bool:
        t0 = time.time()
        while (time.time() - t0) < float(timeout_s):
            try:
                if int(self.operator.get_x_point(0)) == 0:
                    return False
            except Exception:
                pass
            snapshot = self.motion.get_axis_copy(int(axis))
            sts = int(getattr(snapshot, "sts", 0) or 0)
            err = int(getattr(snapshot, "err", 0) or 0)
            if self._is_fault(sts, err):
                raise RuntimeError(f"AX{axis} fault, err={err}")
            pos_err = abs(float(getattr(snapshot, "act_pos", 0.0) or 0.0) - float(target_abs))
            if pos_err <= float(pos_tol) and (not self._is_moving(sts)):
                return True
            time.sleep(0.08)
        return False

    def _is_fault(self, sts: int, err: int) -> bool:
        if self._legacy_flow is not None:
            return bool(self._legacy_flow.is_fault_status(int(sts), int(err)))
        return int(err) != 0

    def _is_enabled(self, sts: int) -> bool:
        if self._legacy_flow is not None:
            return bool(self._legacy_flow.is_enabled_status(int(sts)))
        return int(sts) != 0

    def _is_moving(self, sts: int) -> bool:
        if self._legacy_flow is not None:
            return bool(self._legacy_flow.is_moving_status(int(sts)))
        return False

    def _require_axis_cal(self) -> AxisCal:
        axis_cal = getattr(self.sensors, "axis_cal", None)
        if axis_cal is None:
            raise RuntimeError("AxisCal is not available")
        return axis_cal

    def _require_legacy_flow(self) -> AutoFlow:
        if self._legacy_flow is None:
            raise RuntimeError("Legacy AutoFlow helpers are not available")
        self._legacy_flow.set_runtime_context(self.recipe, self.calibration)
        return self._legacy_flow

    def _soft_limits_from_axis(self, axis: int) -> tuple[float, float]:
        snapshot = self.motion.get_axis_copy(int(axis))
        return (
            float(getattr(snapshot, "softlim_pos", 0.0) or 0.0),
            float(getattr(snapshot, "softlim_neg", 0.0) or 0.0),
        )

    def _write_y_point(self, point: int, value: int) -> None:
        self.operator.plc_write_y_point(int(point), int(value))

    def _read_y_point(self, point: int) -> int:
        try:
            return int(self.operator.get_y_point(int(point)))
        except Exception:
            return 0

    def _clamps_are_closed(self) -> bool:
        return bool(self._read_y_point(10) == 1 and self._read_y_point(11) == 1)

    def _emit_progress(
        self,
        *,
        section_index: int,
        section_total: int,
        z_pos_mm: float,
        ax0_abs: float,
    ) -> None:
        if self.production_workflow is not None:
            self.production_workflow.record_progress(
                section_index=section_index,
                section_total=section_total,
                z_pos_mm=z_pos_mm,
                ax0_abs=ax0_abs,
            )
        self.event_sink.publish_progress(
            section_index=section_index,
            section_total=section_total,
            z_pos_mm=z_pos_mm,
            ax0_abs=ax0_abs,
        )

    def _emit_state(self, state: str, message: str) -> None:
        if self.production_workflow is not None:
            self.production_workflow.record_state(state, message)
        self.event_sink.publish_state(state, message)

    def _raise_if_stop_requested(self) -> None:
        if self._stop_event.is_set():
            raise _StopRequested("User stopped")
        try:
            if int(self.operator.get_x_point(0)) == 0:
                self._stop_event.set()
                raise _StopRequested("E-stop triggered")
        except _StopRequested:
            raise
        except Exception:
            return

    def _set_internal_state(self, state: str) -> None:
        with self._state_lock:
            self.state = state


__all__ = [
    "AutoFlowOrchestrator",
    "measure_current_position_od_avg",
    "measure_current_position_section_capture",
]
