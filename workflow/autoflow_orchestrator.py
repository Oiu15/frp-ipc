from __future__ import annotations

"""Workflow-level orchestrator for the formal measurement flow.

Current scope is still intentionally staged:
- keep the constructor dependency boundary explicit
- own start/stop, outer state transitions, and section loop sequencing
- migrate clamp/AX2/AX3/standby orchestration first
- keep section sampling details out of this file for now
"""

import threading
import time
from typing import TYPE_CHECKING, Any

from application.contracts import EventSink, MachineGateway
from application.state import CalibrationSnapshot, RunSession
from core.models import Recipe
from services.autoflow_service import AutoFlow

if TYPE_CHECKING:  # pragma: no cover
    from app import App
    from core.models import AxisCal


class _StopRequested(RuntimeError):
    """Internal sentinel used to unwind the workflow loop cleanly."""

    def __init__(self, message: str = "User stopped") -> None:
        super().__init__(message)
        self.message = message


class AutoFlowOrchestrator:
    """Explicit dependency shell for the formal measurement workflow."""

    def __init__(
        self,
        gateway: MachineGateway,
        recipe: Recipe,
        calibration: CalibrationSnapshot,
        run_session: RunSession,
        event_sink: EventSink,
    ) -> None:
        self.gateway = gateway
        self.recipe = recipe
        self.calibration = calibration
        self.run_session = run_session
        self.event_sink = event_sink
        self.state = "IDLE"
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._runtime_app: App | None = getattr(gateway, "app", None)
        self._legacy_flow: AutoFlow | None = None
        if self._runtime_app is not None:
            self._legacy_flow = AutoFlow(self._runtime_app)
            self._legacy_flow.stop_event = self._stop_event
            self._legacy_flow._current_recipe = recipe
            self._legacy_flow._calibration_snapshot = calibration

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
        self._stop_event.set()
        self._set_internal_state("STOPPING")
        self._emit_state("STOPPING", "Stop request received")

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def run(self) -> None:
        """Workflow entrypoint for the staged measurement orchestrator."""
        if self.run_session.start_ts is None:
            self.run_session.start_ts = time.time()
        self.run_session.end_ts = None
        self._set_internal_state("RUNNING")
        self._emit_state("RUN", "Auto measurement started")

        status = "DONE"
        message = "Measurement completed"
        try:
            self._run_main_loop()
        except _StopRequested as exc:
            status = "STOP"
            message = str(exc) or "User stopped"
            self._set_internal_state("STOPPED")
        except Exception as exc:
            status = "ERR"
            message = str(exc)
            self._set_internal_state("ERROR")
        finally:
            self.run_session.end_ts = time.time()
            try:
                self.gateway.stop(3)
            except Exception:
                pass
            if self._stop_event.is_set():
                try:
                    self.gateway.abort_motion()
                except Exception:
                    pass

        if status == "DONE":
            self._set_internal_state("DONE")
        self._emit_state(status, message)

    def _run_main_loop(self) -> None:
        section_positions = self._resolve_section_positions()
        if not section_positions:
            raise ValueError("section_count must be > 0")

        self._apply_start_anchor_if_available()
        self._prepare_linear_axes()
        self._prepare_ax2_and_clamps()
        self._run_optional_length_stage()
        self._move_ax2_to_rotate_position()
        self._confirm_rotate_clamp()
        self._prepare_ax3_rotation()
        self._run_section_loop(section_positions)
        self._return_to_standby()

    def _prepare_linear_axes(self) -> None:
        for axis in (0, 1, 4):
            self._ensure_axis_ready(axis)

    def _prepare_ax2_and_clamps(self) -> None:
        self._ensure_axis_ready(2)
        self._emit_state("PREP", "Clamp prepare: main on, sub off")
        self._write_y_point(10, 1)
        self._write_y_point(11, 0)
        time.sleep(0.25)

    def _run_optional_length_stage(self) -> None:
        if not bool(getattr(self.recipe, "len_enable", False)):
            return

        if bool(getattr(self.recipe, "ax2_len_valid", False)):
            target = float(getattr(self.recipe, "ax2_len_abs", 0.0))
            self._move_axis_abs(
                2,
                target,
                strict=True,
                context="AUTO_AX2_LEN",
                state="PREP",
                message_template="AX2 -> length position: {target:.3f}",
            )
        else:
            self._emit_state("WARN", "Length enabled but AX2 length position is not saved")

        payload = self._measure_length_legacy()
        self.event_sink.publish_length(payload)
        app = self._runtime_app
        if app is not None:
            try:
                setattr(app, "_run_len_result", payload)
            except Exception:
                pass

        if bool(getattr(self.recipe, "standby_valid", False)):
            try:
                self._move_axis_abs(
                    0,
                    float(getattr(self.recipe, "standby_ax0_abs", 0.0)),
                    strict=True,
                    context="AUTO_AX0_STANDBY_AFTER_LEN",
                    state="PREP",
                    message_template="AX0 -> standby: {target:.3f}",
                )
            except Exception as exc:
                self._emit_state("WARN", f"AX0 standby move failed: {exc}")

        self._raise_if_stop_requested()

    def _move_ax2_to_rotate_position(self) -> None:
        if not bool(getattr(self.recipe, "ax2_rot_valid", False)):
            raise RuntimeError("AX2 rotate position is not saved")
        self._move_axis_abs(
            2,
            float(getattr(self.recipe, "ax2_rot_abs", 0.0)),
            strict=True,
            context="AUTO_AX2_ROT",
            state="PREP",
            message_template="AX2 -> rotate position: {target:.3f}",
        )

    def _confirm_rotate_clamp(self) -> None:
        self._write_y_point(11, 1)
        time.sleep(0.25)
        self._raise_if_stop_requested()

        app = self._runtime_app
        if app is None or not hasattr(app, "operator_confirm"):
            return
        result = "timeout"
        try:
            result = app.operator_confirm(
                "Clamp Confirm",
                "Confirm sub clamp is ready.\n\n- Press X3 or click confirm to continue\n- Stop to abort",
                allow_stop=True,
                timeout_s=60.0,
            )
        except Exception:
            result = "timeout"
        if result != "confirm":
            raise _StopRequested(f"Operator canceled: {result}")

    def _prepare_ax3_rotation(self) -> None:
        self._ensure_axis_ready(3)
        try:
            velocity = float(getattr(self.recipe, "rot_vel_velmove", 0.0) or 0.0)
        except Exception:
            velocity = 0.0
        if abs(velocity) <= 1e-9:
            velocity = 200.0
        self._emit_state("PREP", f"AX3 rotate start: {velocity:.3f}")
        self.gateway.velmove(3, float(velocity))
        time.sleep(0.20)

    def _run_section_loop(self, section_positions: list[float]) -> None:
        axis_cal = self._require_axis_cal()
        section_total = len(section_positions)
        for section_index, z_pos_mm in enumerate(section_positions, start=1):
            self._raise_if_stop_requested()
            targets = self._resolve_section_targets(
                axis_cal=axis_cal,
                section_index=section_index,
                z_pos_mm=float(z_pos_mm),
            )
            self._emit_progress(
                section_index=section_index,
                section_total=section_total,
                z_pos_mm=float(z_pos_mm),
                ax0_abs=float(targets[0]),
            )
            self._emit_state("RUN", f"Section {section_index}/{section_total} positioning")
            self._move_linear_axes_to_targets(
                targets,
                context=f"AUTO_SEC_{section_index}",
            )
            self._run_section_placeholder(
                section_index=section_index,
                section_total=section_total,
                z_pos_mm=float(z_pos_mm),
            )

    def _run_section_placeholder(
        self,
        *,
        section_index: int,
        section_total: int,
        z_pos_mm: float,
    ) -> None:
        """Section hook reserved for the future sampling migration."""
        _ = (section_index, section_total, z_pos_mm)

    def _return_to_standby(self) -> None:
        if not bool(getattr(self.recipe, "standby_valid", False)):
            return
        targets = {
            1: float(getattr(self.recipe, "standby_ax1_abs", 0.0)),
            4: float(getattr(self.recipe, "standby_ax4_abs", 0.0)),
            0: float(getattr(self.recipe, "standby_ax0_abs", 0.0)),
        }
        try:
            self._move_linear_axes_to_targets(targets, context="AUTO_STANDBY", strict=False)
        except Exception:
            pass

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
            return dict(self._legacy_flow._auto_measure_length(self.recipe))
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
        section_positions = list(getattr(self.recipe, "section_pos_z", []) or [])
        if len(section_positions) == int(getattr(self.recipe, "section_count", 0) or 0):
            return section_positions
        return list(self.recipe.compute_default_positions_z())

    def _resolve_section_targets(
        self,
        *,
        axis_cal: AxisCal,
        section_index: int,
        z_pos_mm: float,
    ) -> dict[int, float]:
        try:
            ax2_abs = float(self._get_ax2_keepout_ref_abs())
        except Exception:
            ax2_abs = float(getattr(self.gateway.get_axis_copy(2), "act_pos", 0.0) or 0.0)

        soft_limits = {
            0: self._soft_limits_from_axis(0),
            1: self._soft_limits_from_axis(1),
            4: self._soft_limits_from_axis(4),
        }
        resolved = axis_cal.od_z_disp_to_targets(
            float(z_pos_mm),
            ax2_abs=ax2_abs,
            softlims_abs=soft_limits,
        )
        targets = {
            1: float(resolved["ax1_abs"]),
            4: float(resolved["ax4_abs"]),
            0: float(resolved["ax0_abs"]),
        }
        for axis, target in list(targets.items()):
            targets[axis] = self.gateway.apply_soft_limits_abs(
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
            resolved[int(axis)] = self.gateway.apply_soft_limits_abs(
                int(axis),
                float(target),
                strict=bool(strict),
                context=context,
            )
        for axis, target in resolved.items():
            self.gateway.movea_abs(int(axis), float(target), context=context)
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
        target_resolved = self.gateway.apply_soft_limits_abs(
            int(axis),
            float(target),
            strict=bool(strict),
            context=context,
        )
        self._emit_state(state, message_template.format(target=float(target_resolved)))
        self.gateway.movea_abs(int(axis), float(target_resolved), context=context)
        ok = self._wait_in_position(int(axis), float(target_resolved), pos_tol=0.05, timeout_s=25.0)
        if not ok:
            self._raise_if_stop_requested()
            raise TimeoutError(f"AX{axis} in-position timeout: {target_resolved:.3f}")

    def _apply_start_anchor_if_available(self) -> None:
        app = self._runtime_app
        if app is None:
            return
        apply_start = getattr(app, "_apply_start_anchor_from_recipe", None)
        if callable(apply_start):
            apply_start()

    def _ensure_axis_ready(self, axis: int) -> None:
        snapshot = self.gateway.get_axis_copy(int(axis))
        sts = int(getattr(snapshot, "sts", 0) or 0)
        err = int(getattr(snapshot, "err", 0) or 0)
        if self._is_fault(sts, err):
            raise RuntimeError(f"AX{axis} fault, err={err}")
        if not self._is_enabled(sts):
            self.gateway.enable(int(axis))
            time.sleep(0.15)

    def _wait_in_position(self, axis: int, target_abs: float, *, pos_tol: float, timeout_s: float) -> bool:
        if self._legacy_flow is not None:
            return bool(self._legacy_flow._wait_in_position(int(axis), float(target_abs), float(pos_tol), float(timeout_s)))

        t0 = time.time()
        while (time.time() - t0) < float(timeout_s):
            self._raise_if_stop_requested()
            snapshot = self.gateway.get_axis_copy(int(axis))
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
            return bool(self._legacy_flow._is_fault(int(sts), int(err)))
        return int(err) != 0

    def _is_enabled(self, sts: int) -> bool:
        if self._legacy_flow is not None:
            return bool(self._legacy_flow._is_enabled(int(sts)))
        return int(sts) != 0

    def _is_moving(self, sts: int) -> bool:
        if self._legacy_flow is not None:
            return bool(self._legacy_flow._is_moving(int(sts)))
        return False

    def _require_axis_cal(self) -> AxisCal:
        app = self._runtime_app
        if app is None:
            raise RuntimeError("Legacy runtime host is required for axis calibration")
        axis_cal = getattr(app, "axis_cal", None)
        if axis_cal is None:
            raise RuntimeError("AxisCal is not available")
        return axis_cal

    def _get_ax2_keepout_ref_abs(self) -> float:
        app = self._runtime_app
        if app is None:
            return float(getattr(self.gateway.get_axis_copy(2), "act_pos", 0.0) or 0.0)
        getter = getattr(app, "_get_ax2_keepout_ref_abs", None)
        if callable(getter):
            return float(getter(prefer_rot=True))
        return float(getattr(self.gateway.get_axis_copy(2), "act_pos", 0.0) or 0.0)

    def _soft_limits_from_axis(self, axis: int) -> tuple[float, float]:
        snapshot = self.gateway.get_axis_copy(int(axis))
        return (
            float(getattr(snapshot, "softlim_pos", 0.0) or 0.0),
            float(getattr(snapshot, "softlim_neg", 0.0) or 0.0),
        )

    def _write_y_point(self, point: int, value: int) -> None:
        app = self._runtime_app
        if app is None:
            raise RuntimeError("Legacy runtime host is required for clamp outputs")
        app.plc_write_y_point(int(point), int(value))

    def _emit_progress(
        self,
        *,
        section_index: int,
        section_total: int,
        z_pos_mm: float,
        ax0_abs: float,
    ) -> None:
        self.event_sink.publish_progress(
            section_index=section_index,
            section_total=section_total,
            z_pos_mm=z_pos_mm,
            ax0_abs=ax0_abs,
        )

    def _emit_state(self, state: str, message: str) -> None:
        self.event_sink.publish_state(state, message)

    def _raise_if_stop_requested(self) -> None:
        if self._stop_event.is_set():
            raise _StopRequested("User stopped")
        app = self._runtime_app
        if app is None:
            return
        try:
            if int(app.get_x_point(0)) == 0:
                self._stop_event.set()
                raise _StopRequested("E-stop triggered")
        except _StopRequested:
            raise
        except Exception:
            return

    def _set_internal_state(self, state: str) -> None:
        with self._state_lock:
            self.state = state


__all__ = ["AutoFlowOrchestrator"]
