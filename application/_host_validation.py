from __future__ import annotations

"""Validation workflow mixin for AppHost."""

import math
import threading
import tkinter as tk
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np

from application.app_adapters import AppDeviceGateway
from core.models import Recipe
from domain.planning import format_recipe_section_name, plan_section_positions
from domain.state import (
    CalibrationSnapshot,
    FIXED_SECTION_PRIMARY_METRICS,
    RuntimeState,
    VALIDATION_MOVE_CHANNELS,
    VALIDATION_MOVE_SCENARIOS,
    ValidationSession,
)
from domain.validation_models import FixedSectionRepeatabilityRequest
from frp_workflow.validation_workflow import ValidationWorkflow
from machine.validation_gateway import ValidationActionCancelled
from repositories.run_repository import RunRepository
from repositories.validation_repository import ValidationRepository
from utils.logger import log


class HostValidationMixin:
    """Mixin providing Validation screen navigation and run lifecycle."""

    validation_status_var: tk.StringVar
    validation_phase_var: tk.StringVar
    validation_wait_phase_var: tk.StringVar
    validation_wait_remaining_s_var: tk.StringVar
    validation_current_repeat_var: tk.StringVar
    validation_result_var: tk.StringVar
    validation_error_var: tk.StringVar
    validation_export_path_var: tk.StringVar
    validation_move_target_pos_var: tk.StringVar
    validation_move_actual_pos_var: tk.StringVar
    validation_current_metric_value_var: tk.StringVar
    validation_current_section_var: tk.StringVar
    validation_current_z_pos_var: tk.StringVar
    validation_current_concentricity_var: tk.StringVar
    validation_summary_count_var: tk.StringVar
    validation_summary_mean_var: tk.StringVar
    validation_summary_std_var: tk.StringVar
    validation_summary_min_var: tk.StringVar
    validation_summary_max_var: tk.StringVar
    validation_summary_range_var: tk.StringVar

    mode_machine: Any
    validation_session: ValidationSession
    _plc_poll_profile_req: str
    _validation_thread: threading.Thread | None
    _validation_running: bool
    _validation_cancel_event: threading.Event
    _validation_cancel_requested: bool

    if TYPE_CHECKING:
        def _recipe_apply_from_ui(self) -> Recipe: ...
        def get_recipe_copy(self) -> Recipe: ...
        def _gauge_ui_widget(self, name: str) -> Any: ...
        def _current_mode_kind_name(self) -> str: ...
        def _is_auto_thread_alive(self) -> bool: ...
        def set_plc_poll_profile(self, profile: str = "normal", *, caller: str | None = None) -> None: ...
        def abort_motion(self, axes: Iterable[int] | None = None) -> Any: ...
        def get_calibration_snapshot(self) -> CalibrationSnapshot: ...
        def _make_run_repository(self) -> RunRepository: ...
        def _make_validation_repository(self) -> ValidationRepository: ...
        def after(self, *args: Any, **kwargs: Any) -> Any: ...
        def update_idletasks(self) -> None: ...

    def open_validation_screen(self) -> None:
        notebook = getattr(self, "_notebook", None)
        tab = getattr(self, "_tab_validation", None)
        if notebook is None or tab is None:
            return None
        try:
            notebook.select(tab)
        except Exception:
            return None
        return None

    def _set_validation_feedback(
        self,
        *,
        status: str = "",
        phase: str | None = None,
        wait_phase: str | None = None,
        wait_remaining_s: float | str | None = None,
        current_repeat: str | None = None,
        result: str = "",
        error: str = "",
        export_path: str = "",
        move_target_pos: object | None = None,
        move_actual_pos: object | None = None,
    ) -> None:
        try:
            self.validation_status_var.set(str(status or ""))
        except Exception:
            pass
        if phase is not None:
            self._set_validation_phase(phase)
        if wait_phase is not None:
            self._set_validation_wait_phase(wait_phase)
        if wait_remaining_s is not None:
            self._set_validation_wait_remaining(wait_remaining_s)
        if current_repeat is not None:
            self._set_validation_current_repeat(current_repeat)
        try:
            self.validation_result_var.set(str(result or ""))
        except Exception:
            pass
        try:
            self.validation_error_var.set(str(error or ""))
        except Exception:
            pass
        try:
            self.validation_export_path_var.set(str(export_path or ""))
        except Exception:
            pass
        if move_target_pos is not None or move_actual_pos is not None:
            self._set_validation_move_position(
                target_pos=move_target_pos,
                actual_pos=move_actual_pos,
            )

    def _set_validation_phase(self, phase: str) -> None:
        try:
            self.validation_phase_var.set(self._format_validation_phase(phase))
        except Exception:
            pass

    def _set_validation_move_position(
        self,
        *,
        target_pos: object | None = None,
        actual_pos: object | None = None,
    ) -> None:
        if target_pos is not None:
            try:
                self.validation_move_target_pos_var.set(
                    self._format_validation_position(target_pos)
                )
            except Exception:
                pass
        if actual_pos is not None:
            try:
                self.validation_move_actual_pos_var.set(
                    self._format_validation_position(actual_pos)
                )
            except Exception:
                pass

    def _set_validation_wait_phase(self, phase: str) -> None:
        try:
            self.validation_wait_phase_var.set(self._format_validation_phase(phase) if str(phase or "").strip() else "")
        except Exception:
            pass

    def _set_validation_wait_remaining(self, remaining_s: float | str) -> None:
        try:
            text = ""
            if remaining_s not in (None, ""):
                text = f"{float(remaining_s):.3f}s"
            self.validation_wait_remaining_s_var.set(text)
        except Exception:
            pass

    def _set_validation_current_repeat(self, current_repeat: str) -> None:
        try:
            self.validation_current_repeat_var.set(str(current_repeat or ""))
        except Exception:
            pass

    def _reset_validation_summary_panel(self) -> None:
        defaults = {
            "validation_current_metric_value_var": "",
            "validation_current_section_var": "",
            "validation_current_z_pos_var": "",
            "validation_current_concentricity_var": "",
            "validation_summary_count_var": "0",
            "validation_summary_mean_var": "",
            "validation_summary_std_var": "",
            "validation_summary_min_var": "",
            "validation_summary_max_var": "",
            "validation_summary_range_var": "",
        }
        for attr_name, value in defaults.items():
            try:
                getattr(self, attr_name).set(value)
            except Exception:
                pass

    def _set_validation_current_repeat_result(self, capture: object | None = None) -> None:
        if capture is None:
            return
        try:
            measured_value_mm = getattr(capture, "measured_value_mm", None)
            self.validation_current_metric_value_var.set(
                self._format_validation_numeric(measured_value_mm, digits=6)
            )
        except Exception:
            pass
        try:
            section_name = getattr(capture, "measure_section_name", "") or getattr(capture, "section_name", "")
            self.validation_current_section_var.set(str(section_name or ""))
        except Exception:
            pass
        try:
            measured_z_pos_mm = getattr(capture, "measured_z_pos_mm", None)
            self.validation_current_z_pos_var.set(
                self._format_validation_numeric(measured_z_pos_mm, digits=3)
            )
        except Exception:
            pass
        try:
            fit_result = getattr(capture, "fit_result", None)
            concentricity_mm = None if fit_result is None else getattr(fit_result, "concentricity_mm", None)
            self.validation_current_concentricity_var.set(
                self._format_validation_numeric(concentricity_mm, digits=6)
            )
        except Exception:
            pass

    def _set_validation_summary_values(self, summary: Mapping[str, Any] | None = None) -> None:
        payload = dict(summary or {})
        try:
            count_value = int(payload.get("count", 0) or 0)
        except Exception:
            count_value = 0
        try:
            self.validation_summary_count_var.set(str(count_value))
        except Exception:
            pass
        for attr_name, field_name in (
            ("validation_summary_mean_var", "mean"),
            ("validation_summary_std_var", "std"),
            ("validation_summary_min_var", "min"),
            ("validation_summary_max_var", "max"),
            ("validation_summary_range_var", "range"),
        ):
            try:
                getattr(self, attr_name).set(
                    self._format_validation_numeric(payload.get(field_name), digits=6)
                )
            except Exception:
                pass

    @staticmethod
    def _format_validation_phase(phase: str) -> str:
        raw = str(phase or "IDLE").strip()
        if not raw:
            raw = "IDLE"
        return raw.upper()

    @staticmethod
    def _format_validation_numeric(value: object, *, digits: int = 6) -> str:
        if value in (None, ""):
            return ""
        try:
            if not isinstance(value, (str, int, float, np.number)):
                return str(value)
            numeric = float(value)
        except Exception:
            return str(value)
        if not math.isfinite(numeric):
            return ""
        return f"{numeric:.{int(digits)}f}"

    @staticmethod
    def _format_validation_position(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, Mapping):
            parts: list[str] = []
            for key in sorted(value.keys(), key=lambda item: str(item)):
                try:
                    parts.append(f"{key}={float(value[key]):.3f}")
                except Exception:
                    parts.append(f"{key}={value[key]}")
            return " ".join(parts)
        text = str(value).strip()
        if not text:
            return ""
        try:
            return f"{float(text):.3f}"
        except Exception:
            return text

    def _validation_recipe_snapshot_from_ui(self) -> Recipe:
        try:
            return self._recipe_apply_from_ui()
        except Exception:
            return self.get_recipe_copy()

    def list_validation_section_choices(self) -> list[str]:
        try:
            recipe = self._validation_recipe_snapshot_from_ui()
            positions = list(plan_section_positions(recipe).positions_z)
        except Exception:
            positions = []
        if not positions:
            return ["1"]
        return [
            format_recipe_section_name(index, z_pos)
            for index, z_pos in enumerate(positions, start=1)
        ]

    def _set_validation_start_button_state(self, enabled: bool) -> None:
        for widget_name in ('validation_screen_start_btn',):
            start_btn = self._gauge_ui_widget(widget_name)
            if start_btn is None:
                continue
            try:
                start_btn.configure(state='normal' if enabled else 'disabled')
            except Exception:
                pass

    def _set_validation_stop_button_state(self, enabled: bool) -> None:
        for widget_name in ('validation_screen_stop_btn',):
            stop_btn = self._gauge_ui_widget(widget_name)
            if stop_btn is None:
                continue
            try:
                stop_btn.configure(state='normal' if enabled else 'disabled')
            except Exception:
                pass

    def _sync_validation_mode(self, workflow_state: str, message: str = "") -> None:
        try:
            sync_mode_state = getattr(self.mode_machine, "sync_validation_workflow_state", None)
            if callable(sync_mode_state):
                sync_mode_state(str(workflow_state or ""), str(message or ""))
        except Exception:
            pass

    def is_validation_cancel_requested(self) -> bool:
        cancel_event = getattr(self, "_validation_cancel_event", None)
        try:
            if cancel_event is not None and bool(cancel_event.is_set()):
                return True
        except Exception:
            pass
        try:
            return bool(getattr(self, "_validation_cancel_requested", False))
        except Exception:
            return False

    def _prepare_validation_run(self, *, move_scenario: str) -> None:
        cancel_event = getattr(self, "_validation_cancel_event", None)
        if cancel_event is None:
            cancel_event = threading.Event()
            self._validation_cancel_event = cancel_event
        try:
            cancel_event.clear()
        except Exception:
            pass
        self._validation_cancel_requested = False
        current_profile = str(getattr(self, "_plc_poll_profile_req", "normal") or "normal")
        log(
            "VALIDATION_ENTER",
            current_poll_profile=current_profile,
            move_scenario=str(move_scenario or ""),
            mode_kind=self._current_mode_kind_name(),
            validation_running=bool(getattr(self, "_validation_running", False)),
            auto_thread_alive=self._is_auto_thread_alive(),
        )
        self.set_plc_poll_profile("normal", caller="validation_enter")

    def _cleanup_validation_run(
        self,
        *,
        status: str,
        phase: str | None = None,
        error: str = "",
    ) -> None:
        self.set_plc_poll_profile("normal", caller="validation_exit")
        cancel_event = getattr(self, "_validation_cancel_event", None)
        try:
            if cancel_event is not None:
                cancel_event.clear()
        except Exception:
            pass
        self._validation_cancel_requested = False
        self._validation_running = False
        self._validation_thread = None
        workflow_state = "ERR"
        normalized_status = str(status or "").strip().upper()
        if normalized_status == "DONE":
            workflow_state = "DONE"
        elif normalized_status == "STOP":
            workflow_state = "STOP"
        elif normalized_status == "STOPPING":
            workflow_state = "STOPPING"
        self._sync_validation_mode(workflow_state, error)
        log(
            "VALIDATION_EXIT",
            status=str(status or ""),
            phase=str(phase or ""),
            error=str(error or ""),
            poll_profile_after_cleanup=str(getattr(self, "_plc_poll_profile_req", "")),
        )

    def _finish_validation_run_ui(
        self,
        *,
        status: str,
        phase: str | None = None,
        result: str = "",
        error: str = "",
        export_path: str = "",
    ) -> None:
        self._cleanup_validation_run(status=status, phase=phase, error=error)
        self._set_validation_feedback(
            status=status,
            phase=phase,
            wait_phase="",
            wait_remaining_s="",
            result=result,
            error=error,
            export_path=export_path,
        )
        try:
            sync_mode_state = getattr(self.mode_machine, 'sync_current_mode_state', None)
            if callable(sync_mode_state):
                sync_mode_state()
        except Exception:
            pass
        self._set_validation_start_button_state(True)
        self._set_validation_stop_button_state(False)
        try:
            self.update_idletasks()
        except Exception:
            pass

    def stop_validation_run(self) -> None:
        if not bool(getattr(self, "_validation_running", False)):
            return None
        cancel_event = getattr(self, "_validation_cancel_event", None)
        if cancel_event is None:
            cancel_event = threading.Event()
            self._validation_cancel_event = cancel_event
        self._validation_cancel_requested = True
        try:
            cancel_event.set()
        except Exception:
            pass
        log(
            "VALIDATION_STOP_REQUEST",
            current_poll_profile=str(getattr(self, "_plc_poll_profile_req", "") or ""),
            mode_kind=self._current_mode_kind_name(),
            auto_thread_alive=self._is_auto_thread_alive(),
        )
        self._sync_validation_mode("STOPPING")
        self._set_validation_feedback(
            status="STOPPING",
            result="",
            error="",
        )
        self._set_validation_stop_button_state(False)
        try:
            self.abort_motion()
        except Exception:
            pass
        try:
            self.update_idletasks()
        except Exception:
            pass
        return None

    def start_validation_run(
        self,
        *,
        section_name: str,
        metric_name: str,
        repeat_count: int,
        reclamp_between_repeats: bool = False,
        reclamp_enabled: bool = False,
        rotation_stop_before_measure: bool = False,
        release_settle_s: float = 0.0,
        clamp_settle_s: float = 0.0,
        position_settle_s: float = 0.0,
        sample_delay_s: float = 0.0,
        validation_ax3_speed_dps: float = 60.0,
        move_enabled: bool = False,
        move_channel: str = "od_channel",
        move_away_delta_mm: float = 0.0,
        move_scenario: str = "distance_round_trip",
        move_from_section_index: int = 1,
        move_target_section_index: int = 1,
        move_return_section_index: int = 1,
    ) -> Optional[str]:
        try:
            def _bool_param(value) -> bool:
                if isinstance(value, bool):
                    return value
                text = str(value or "").strip().lower()
                if text in {"1", "true", "yes", "y", "on"}:
                    return True
                if text in {"", "0", "false", "no", "n", "off"}:
                    return False
                return bool(value)

            def _settle_param(value, field_name: str) -> float:
                text = str(value or "").strip()
                if not text:
                    return 0.0
                try:
                    numeric = float(text)
                except Exception as exc:
                    raise ValueError(f"{field_name} must be a number") from exc
                if numeric < 0.0:
                    raise ValueError(f"{field_name} must be >= 0")
                return numeric

            def _positive_param(value, field_name: str) -> float:
                text = str(value or "").strip()
                if not text:
                    raise ValueError(f"{field_name} must be > 0")
                try:
                    numeric = float(text)
                except Exception as exc:
                    raise ValueError(f"{field_name} must be a number") from exc
                if numeric <= 0.0:
                    raise ValueError(f"{field_name} must be > 0")
                return numeric

            def _choice_param(value, field_name: str, choices) -> str:
                text = str(value or "").strip()
                if text not in choices:
                    raise ValueError(f"{field_name} must be one of: " + ", ".join(choices))
                return text

            def _section_index_param(value, field_name: str) -> int:
                text = str(value or "").strip()
                if ":" in text:
                    text = text.split(":", 1)[0].strip()
                if not text:
                    raise ValueError(f"{field_name} must be >= 1")
                try:
                    numeric = int(float(text))
                except Exception as exc:
                    raise ValueError(f"{field_name} must be an integer") from exc
                if numeric < 1:
                    raise ValueError(f"{field_name} must be >= 1")
                return numeric

            metric = str(metric_name or "").strip()
            if metric not in FIXED_SECTION_PRIMARY_METRICS:
                raise ValueError(
                    "metric_name must be one of: " + ", ".join(FIXED_SECTION_PRIMARY_METRICS)
                )
            repeat = int(repeat_count)
            if repeat < 1:
                raise ValueError("repeat_count must be >= 1")
            request = FixedSectionRepeatabilityRequest(
                section_name=str(section_name or "").strip(),
                metric_name=metric,
                repeat_count=repeat,
                reclamp_between_repeats=_bool_param(reclamp_between_repeats),
                reclamp_enabled=_bool_param(reclamp_enabled),
                rotation_stop_before_measure=_bool_param(rotation_stop_before_measure),
                release_settle_s=_settle_param(release_settle_s, "release_settle_s"),
                clamp_settle_s=_settle_param(clamp_settle_s, "clamp_settle_s"),
                position_settle_s=_settle_param(position_settle_s, "position_settle_s"),
                sample_delay_s=_settle_param(sample_delay_s, "sample_delay_s"),
                validation_ax3_speed_dps=_positive_param(validation_ax3_speed_dps, "validation_ax3_speed_dps"),
                move_enabled=_bool_param(move_enabled),
                move_channel=_choice_param(
                    move_channel,
                    "move_channel",
                    VALIDATION_MOVE_CHANNELS,
                ),
                move_away_delta_mm=_settle_param(move_away_delta_mm, "move_away_delta_mm"),
                move_scenario=_choice_param(
                    move_scenario,
                    "move_scenario",
                    VALIDATION_MOVE_SCENARIOS,
                ),
                move_from_section_index=_section_index_param(
                    move_from_section_index,
                    "move_from_section_index",
                ),
                move_target_section_index=_section_index_param(
                    move_target_section_index,
                    "move_target_section_index",
                ),
                move_return_section_index=_section_index_param(
                    move_return_section_index,
                    "move_return_section_index",
                ),
            )

            if bool(getattr(self, '_validation_running', False)):
                raise RuntimeError("固定截面重复性验证正在运行")

            try:
                enter_validation = getattr(self.mode_machine, 'enter_validation', None)
                if callable(enter_validation):
                    enter_validation()
            except Exception:
                pass
            self._prepare_validation_run(move_scenario=request.move_scenario)

            validation_session = ValidationSession()
            self.validation_session = validation_session
            validation_runtime_state = RuntimeState.from_validation_session(validation_session)
            recipe_snapshot = self._validation_recipe_snapshot_from_ui()
            workflow = ValidationWorkflow(
                recipe=recipe_snapshot,
                calibration=self.get_calibration_snapshot(),
                runtime_state=validation_runtime_state,
                gateway=AppDeviceGateway(cast(Any, self)),
                run_repository=self._make_run_repository(),
                validation_session=validation_session,
            )
            validation_repository = self._make_validation_repository()

            self._validation_running = True
            self._sync_validation_mode("RUN")
            self._set_validation_start_button_state(False)
            self._set_validation_stop_button_state(True)
            self._reset_validation_summary_panel()
            self._set_validation_feedback(
                status=f"RUNNING 0/{repeat}",
                phase="PREPARE",
                wait_phase="",
                wait_remaining_s="",
                current_repeat=f"0/{repeat}",
                result="",
                error="",
                export_path="",
                move_target_pos="",
                move_actual_pos="",
            )
            try:
                self.update_idletasks()
            except Exception:
                pass

            def _worker() -> None:
                try:
                    def _running_summary_snapshot() -> tuple[object | None, dict[str, Any]]:
                        latest_capture = None
                        summary_payload: dict[str, Any] = {}
                        try:
                            captures = tuple(getattr(workflow, "fixed_section_repeat_captures", ()) or ())
                            if captures:
                                latest_capture = captures[-1]
                        except Exception:
                            latest_capture = None
                        try:
                            build_summary = getattr(workflow, "build_fixed_section_repeatability_summary", None)
                            if callable(build_summary):
                                raw_summary = build_summary()
                                if isinstance(raw_summary, Mapping):
                                    summary_payload = dict(raw_summary)
                        except Exception:
                            summary_payload = {}
                        return latest_capture, summary_payload

                    def _progress_update(index: int, total_count: int) -> None:
                        latest_capture, summary_payload = _running_summary_snapshot()
                        def _apply_progress() -> None:
                            self._set_validation_feedback(
                                status=f"RUNNING {int(index)}/{int(total_count)}",
                                result="",
                                error="",
                                export_path="",
                            )
                            self._set_validation_current_repeat_result(latest_capture)
                            self._set_validation_summary_values(summary_payload)
                        self.after(0, _apply_progress)

                    def _status_update(status_text: str) -> None:
                        def _apply_status() -> None:
                            self._set_validation_feedback(
                                status=str(status_text),
                                result="",
                                error="",
                                export_path="",
                            )
                        self.after(0, _apply_status)

                    def _phase_update(phase_event) -> None:
                        phase_name = str(getattr(phase_event, 'phase', '') or '')
                        payload = getattr(phase_event, 'payload', {}) or {}
                        try:
                            target_pos = payload.get(
                                'target_positions_mm',
                                payload.get('target_position_mm'),
                            )
                        except Exception:
                            target_pos = None
                        try:
                            actual_pos = payload.get(
                                'actual_positions_mm',
                                payload.get('actual_position_mm'),
                            )
                        except Exception:
                            actual_pos = None
                        repeat_index = int(getattr(phase_event, 'repeat_index', 0) or 0)
                        total_count = int(getattr(phase_event, 'total', 0) or 0)
                        def _apply_phase() -> None:
                            wait_phase = phase_name if phase_name.startswith('wait_') else ""
                            self._set_validation_feedback(
                                phase=phase_name,
                                wait_phase=wait_phase,
                                wait_remaining_s=(0.0 if wait_phase else ""),
                                current_repeat=(f"{repeat_index}/{total_count}" if total_count > 0 else ""),
                            )
                            self._set_validation_move_position(
                                target_pos=target_pos,
                                actual_pos=actual_pos,
                            )
                        self.after(0, _apply_phase)

                    def _wait_update(phase_name: str, repeat_index: int, total_count: int, remaining_s: float) -> None:
                        def _apply_wait() -> None:
                            self._set_validation_feedback(
                                wait_phase=phase_name,
                                wait_remaining_s=float(remaining_s),
                                current_repeat=(f"{int(repeat_index)}/{int(total_count)}" if int(total_count) > 0 else ""),
                            )
                        self.after(0, _apply_wait)

                    rows, summary = workflow.run_fixed_section_repeatability(
                        request,
                        progress_callback=_progress_update,
                        status_callback=_status_update,
                        phase_callback=_phase_update,
                        wait_callback=_wait_update,
                    )
                    export_dir = validation_repository.export_fixed_section_repeatability(
                        context=workflow.build_export_context(),
                        request=request,
                        rows=rows,
                        summary=summary,
                        captures=workflow.fixed_section_repeat_captures,
                    )
                    result_text = (
                        f"{metric} "
                        f"count={int(summary.get('count', 0))} "
                        f"mean={float(summary.get('mean', 0.0)):.6f} "
                        f"std={float(summary.get('std', 0.0)):.6f} "
                        f"min={float(summary.get('min', 0.0)):.6f} "
                        f"max={float(summary.get('max', 0.0)):.6f} "
                        f"range={float(summary.get('range', 0.0)):.6f}"
                    )
                    latest_capture, summary_payload = _running_summary_snapshot()
                    def _on_success() -> None:
                        self.validation_session = workflow.validation_session or validation_session
                        self._set_validation_current_repeat_result(latest_capture)
                        self._set_validation_summary_values(summary if summary else summary_payload)
                        self._finish_validation_run_ui(
                            status="DONE",
                            result=result_text,
                            error="",
                            export_path=export_dir,
                        )
                    self.after(0, _on_success)
                except ValidationActionCancelled:
                    def _on_cancel() -> None:
                        self.validation_session = workflow.validation_session or validation_session
                        self._finish_validation_run_ui(
                            status="STOP",
                            phase="STOP",
                            result="",
                            error="",
                            export_path="",
                        )
                    try:
                        self.after(0, _on_cancel)
                    except Exception:
                        self.validation_session = workflow.validation_session or validation_session
                        self._cleanup_validation_run(
                            status="STOP",
                            phase="STOP",
                            error="",
                        )
                except Exception as exc:
                    error_text = str(exc)
                    def _on_error() -> None:
                        self.validation_session = workflow.validation_session or validation_session
                        self._finish_validation_run_ui(
                            status="ERR",
                            result="",
                            error=error_text,
                            export_path="",
                        )
                    try:
                        self.after(0, _on_error)
                    except Exception:
                        self.validation_session = workflow.validation_session or validation_session
                        self._cleanup_validation_run(
                            status="ERR",
                            error=error_text,
                        )

            log(
                "VALIDATION_THREAD_STARTING",
                current_poll_profile=str(getattr(self, "_plc_poll_profile_req", "") or ""),
                move_scenario=str(request.move_scenario or ""),
            )
            worker = threading.Thread(
                target=_worker,
                name="validation-fixed-section-repeatability",
                daemon=True,
            )
            self._validation_thread = worker
            worker.start()
            return None
        except Exception as exc:
            self._finish_validation_run_ui(
                status="ERR",
                phase="IDLE",
                result="",
                error=str(exc),
                export_path="",
            )
            return None

    _set_validation_debug_feedback = _set_validation_feedback
    _set_validation_debug_phase = _set_validation_phase
    _set_validation_debug_move_position = _set_validation_move_position
    _set_validation_debug_wait_phase = _set_validation_wait_phase
    _set_validation_debug_wait_remaining = _set_validation_wait_remaining
    _set_validation_debug_current_repeat = _set_validation_current_repeat
    _reset_validation_debug_summary_panel = _reset_validation_summary_panel
    _set_validation_debug_current_repeat_result = _set_validation_current_repeat_result
    _set_validation_debug_summary_values = _set_validation_summary_values
    _format_validation_debug_phase = _format_validation_phase
    _format_validation_debug_numeric = _format_validation_numeric
    _format_validation_debug_position = _format_validation_position
    _set_validation_debug_start_button_state = _set_validation_start_button_state
    _set_validation_debug_stop_button_state = _set_validation_stop_button_state
    _sync_validation_debug_mode = _sync_validation_mode
    _prepare_validation_debug_run = _prepare_validation_run
    _cleanup_validation_debug_run = _cleanup_validation_run
    _finish_validation_debug_run_ui = _finish_validation_run_ui
    start_fixed_section_repeatability_debug = start_validation_run
    stop_fixed_section_repeatability_debug = stop_validation_run
