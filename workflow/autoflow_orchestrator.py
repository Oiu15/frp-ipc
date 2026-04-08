from __future__ import annotations

"""Workflow-level orchestrator skeleton for the formal measurement flow.

Current scope is intentionally minimal:
- define the constructor dependency boundary
- support start/stop and state transitions
- keep a minimal main-loop/event skeleton for future AutoFlow migration

The legacy threaded AutoFlow implementation still owns the real runtime
logic today. This orchestrator exists as the next landing zone so we can
move behavior incrementally without changing dependency signatures again.
"""

import threading
import time

from application.contracts import EventSink, MachineGateway
from application.state import CalibrationSnapshot, RunSession
from core.models import Recipe


class _StopRequested(RuntimeError):
    """Internal sentinel used to unwind the workflow loop cleanly."""


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

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

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
        self._emit_state("STOPPING", "停止请求已接收")

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def run(self) -> None:
        """Workflow entrypoint placeholder for future migration."""
        if self.run_session.start_ts is None:
            self.run_session.start_ts = time.time()
        self.run_session.end_ts = None
        self._set_internal_state("RUNNING")
        self._emit_state("RUN", "自动测量开始")
        try:
            self._run_main_loop()
        except _StopRequested:
            self.run_session.end_ts = time.time()
            self._set_internal_state("STOPPED")
            self._emit_state("STOP", "用户停止")
            return
        except Exception as exc:
            self.run_session.end_ts = time.time()
            self._set_internal_state("ERROR")
            self._emit_state("ERR", str(exc))
            raise

        self.run_session.end_ts = time.time()
        self._set_internal_state("DONE")
        self._emit_state("DONE", "测量完成")

    def _run_main_loop(self) -> None:
        """Minimal section loop skeleton.

        Sampling and fitting details are intentionally *not* migrated yet.
        This loop only establishes the future section-oriented control flow
        and event emission boundary.
        """
        section_positions = self._resolve_section_positions()
        section_total = len(section_positions)
        if section_total <= 0:
            return

        for section_index, z_pos_mm in enumerate(section_positions, start=1):
            self._raise_if_stop_requested()
            self._emit_progress(
                section_index=section_index,
                section_total=section_total,
                z_pos_mm=float(z_pos_mm),
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
        """Section hook reserved for the future AutoFlow migration."""
        _ = (section_index, section_total, z_pos_mm)

    def _resolve_section_positions(self) -> list[float]:
        section_positions = list(getattr(self.recipe, "section_pos_z", []) or [])
        if section_positions:
            return section_positions
        return list(self.recipe.compute_default_positions_z())

    def _emit_progress(
        self,
        *,
        section_index: int,
        section_total: int,
        z_pos_mm: float,
    ) -> None:
        axis0 = self.gateway.get_axis_copy(0)
        ax0_abs = float(getattr(axis0, "act_pos", 0.0) or 0.0)
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
            raise _StopRequested()

    def _set_internal_state(self, state: str) -> None:
        with self._state_lock:
            self.state = state


__all__ = ["AutoFlowOrchestrator"]
