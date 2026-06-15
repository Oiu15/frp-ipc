from __future__ import annotations

"""ID diameter calibration service — port-based, host-free.

Extracted from ``calibration_service.py``.  Handles ID (inner-diameter)
calibration capture using CL-3000 OUT4/OUT5 channels.
"""

import math
import time
from typing import Any

import numpy as np

from domain.calibration import solve_id_delta_candidate
from machine.ports import RotationPort
from repositories.calibration_repository import CalibrationRepository
from services.calibration_ports import (
    CalibrationSensorPort,
    CalibrationStateSink,
    PollProfilePort,
    SchedulerPort,
)
from services.calibration_context import CalibrationProgress, ClSample


class IdCalibrationService:
    """Port-based ID diameter calibration capture and computation."""

    def __init__(
        self,
        *,
        rotation: RotationPort,
        sensors: CalibrationSensorPort,
        scheduler: SchedulerPort,
        state_sink: CalibrationStateSink,
        poll_profile: PollProfilePort,
        repository: CalibrationRepository,
    ) -> None:
        self._rotation = rotation
        self._sensors = sensors
        self._scheduler = scheduler
        self._state_sink = state_sink
        self._poll_profile = poll_profile
        self._repository = repository

        self._capturing: bool = False
        self._samples: list[dict[str, Any]] = []
        self._start_ts: float | None = None
        self._stop_at_ts: float | None = None
        self._theta_start: float | None = None
        self._theta_last: float | None = None
        self._theta_unwrap: float = 0.0
        self._rev_progress_deg: float = 0.0
        self._schedule_handle: object | None = None
        self._one_rev: bool = False
        self._one_rev_timeout_ts: float | None = None
        self._force_one_rev: bool = False
        self._delta_candidate: float | None = None
        self._prev_poll_profile: str = "normal"

    # -- public API ---------------------------------------------------------

    def start_capture(
        self,
        *,
        rotation_speed_dps: float,
        sampling_hz: float,
        capture_duration_s: float,
        mode: str = "timed",
        force_one_rev: bool = False,
    ) -> None:
        if self._capturing:
            return
        self._one_rev = mode == "one_rev" or bool(force_one_rev)
        self._force_one_rev = bool(force_one_rev)
        self._samples = []
        self._theta_start = None
        self._theta_last = None
        self._theta_unwrap = 0.0
        self._rev_progress_deg = 0.0
        self._start_ts = time.time()
        self._stop_at_ts = self._start_ts + max(0.5, capture_duration_s) if not self._one_rev else None
        self._one_rev_timeout_ts = self._start_ts + 60.0
        self._prev_poll_profile = "normal"
        self._poll_profile.use_poll_profile("sampling")
        self._rotation.start_rotation(rotation_speed_dps)
        self._capturing = True
        self._state_sink.begin_capture()
        self._schedule_tick(sampling_hz)

    def stop_capture(self, reason: str = "") -> None:
        self._capturing = False
        self._cancel_tick()
        try:
            self._rotation.stop_rotation()
        except Exception:
            pass
        try:
            self._poll_profile.use_poll_profile(self._prev_poll_profile)
        except Exception:
            pass
        self._state_sink.end_capture()

    # -- internal -----------------------------------------------------------

    def _schedule_tick(self, hz: float) -> None:
        period_ms = int(max(5, round(1000.0 / max(1.0, min(100.0, hz)))))
        self._schedule_handle = self._scheduler.schedule_once(period_ms, self._tick)

    def _cancel_tick(self) -> None:
        if self._schedule_handle is not None:
            self._scheduler.cancel(self._schedule_handle)
            self._schedule_handle = None

    def _tick(self) -> None:
        if not self._capturing:
            return
        now = time.time()
        if self._one_rev_timeout_ts is not None and now >= self._one_rev_timeout_ts:
            self.stop_capture("一圈超时")
            return
        if (not self._one_rev) and self._stop_at_ts is not None and now >= self._stop_at_ts:
            self.stop_capture("定时结束")
            return
        # read angle
        try:
            theta = self._sensors.read_axis_angle_deg()
        except Exception:
            theta = float("nan")
        if math.isfinite(theta):
            self._update_rev_progress(float(theta))
            if self._one_rev and self._rev_done():
                self.stop_capture("已采满一圈")
                return
        # read CL
        try:
            cl = self._sensors.read_cl_out145_cached()
        except Exception:
            cl = ClSample(ok=False)
        if cl.ok and cl.out4 is not None and math.isfinite(float(cl.out4)) and cl.out5 is not None and math.isfinite(float(cl.out5)):
            self._samples.append(
                {
                    "ts": now,
                    "theta_deg": theta,
                    "c_mm": float(cl.out4),
                    "m_mm": float(cl.out5),
                }
            )
        # progress
        elapsed = now - (self._start_ts or now)
        self._state_sink.publish_progress(
            CalibrationProgress(
                angle_deg=self._rev_progress_deg,
                elapsed_s=elapsed,
                sample_count=len(self._samples),
            )
        )
        self._schedule_tick(20.0)

    def _update_rev_progress(self, theta: float) -> None:
        if self._theta_start is None:
            self._theta_start = theta
        if self._theta_last is not None:
            delta = theta - self._theta_last
            while delta <= -180.0:
                delta += 360.0
            while delta > 180.0:
                delta -= 360.0
            self._theta_unwrap += delta
        self._theta_last = theta
        self._rev_progress_deg = float(self._theta_unwrap)

    def _rev_done(self) -> bool:
        return bool(self._theta_start is not None and self._rev_progress_deg >= 360.0)

    def compute_candidate(
        self,
        reference_diameter_mm: float = 150.0,
    ) -> dict[str, Any]:
        """Compute delta_candidate from captured ID samples."""
        pts = [
            p for p in self._samples
            if (p.get("theta_deg") is not None and math.isfinite(float(p["theta_deg"]))
                and p.get("c_mm") is not None and p.get("m_mm") is not None)
        ]
        if len(pts) < 20:
            # Fallback: use max(c_mm)
            cs = [p["c_mm"] for p in self._samples if p.get("c_mm") is not None]
            if not cs:
                return {"ok": False, "reason": "无有效OUT4"}
            delta = float(reference_diameter_mm) - float(max(cs))
            self._delta_candidate = delta
            return {"ok": True, "delta_c_mm": delta, "fallback": True, "n": len(self._samples)}
        theta_arr = np.array([p["theta_deg"] for p in pts], dtype=float)
        c_arr = np.array([p["c_mm"] for p in pts], dtype=float)
        m_arr = np.array([p["m_mm"] for p in pts], dtype=float)
        result = solve_id_delta_candidate(theta_arr, c_arr, m_arr, float(reference_diameter_mm))
        if not result.ok or result.delta_candidate is None:
            return {"ok": False, "reason": result.reason or "拟合失败"}
        self._delta_candidate = float(result.delta_candidate)
        return {
            "ok": True,
            "delta_c_mm": float(result.delta_candidate),
            "fallback": result.fallback_used,
            "n": len(pts),
        }

    def apply_result(self, reference_diameter_mm: float = 150.0) -> dict[str, Any]:
        """Persist the computed delta candidate."""
        if self._delta_candidate is None:
            return {"ok": False, "reason": "请先计算"}
        data = {"delta_c_mm": float(self._delta_candidate), "D_ref": float(reference_diameter_mm), "ts": time.time()}
        self._repository.save_id_active(data)
        return {"ok": True, "delta_c_mm": float(self._delta_candidate)}


__all__ = ["IdCalibrationService"]
