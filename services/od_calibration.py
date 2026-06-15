from __future__ import annotations

"""OD calibration service — port-based, host-free.

Extracted from ``calibration_service.py``.  Handles timed or one-revolution
OD (outer-diameter) capture using the external gauge (Keyence CL-3000 OUT1).
"""

import math
import time
from typing import Any

from domain.calibration import compute_od_b_candidate
from machine.ports import RotationPort
from services.calibration_ports import CalibrationRepositoryProtocol
from services.calibration_ports import (
    CalibrationSensorPort,
    CalibrationStateSink,
    PollProfilePort,
    SchedulerPort,
)
from services.calibration_context import CalibrationProgress


class OdCalibrationService:
    """Port-based OD calibration capture and computation."""

    def __init__(
        self,
        *,
        rotation: RotationPort,
        sensors: CalibrationSensorPort,
        scheduler: SchedulerPort,
        state_sink: CalibrationStateSink,
        poll_profile: PollProfilePort,
        repository: CalibrationRepositoryProtocol,
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
        self._angle_enabled: bool = True
        self._filter_mode: str = ""
        self._outlier_sigma: float = 3.0
        self._one_rev: bool = False
        self._drop_count: int = 0
        self._sampling_hz: float = 20.0
        self._prev_poll_profile: str = "normal"

    # -- public API ---------------------------------------------------------

    def start_capture(
        self,
        *,
        rotation_speed_dps: float,
        sampling_hz: float,
        capture_duration_s: float,
        mode: str = "timed",
        angle_enabled: bool = True,
        filter_mode: str = "",
        outlier_sigma: float = 3.0,
        gauge_cmd: str = "M0,1",
    ) -> None:
        if self._capturing:
            return
        if mode == "one_rev" and not angle_enabled:
            mode = "timed"
            self._state_sink.capture_failed("角度来源=无角度：已自动切换为定时采样")
        self._one_rev = mode == "one_rev"
        self._angle_enabled = angle_enabled
        self._filter_mode = filter_mode
        self._outlier_sigma = outlier_sigma
        self._samples = []
        self._drop_count = 0
        self._theta_start = None
        self._theta_last = None
        self._theta_unwrap = 0.0
        self._rev_progress_deg = 0.0
        self._start_ts = time.time()
        self._stop_at_ts = self._start_ts + max(0.5, capture_duration_s) if not self._one_rev else None
        self._prev_poll_profile = "normal"
        self._poll_profile.use_poll_profile("sampling")  # type: ignore[arg-type]
        self._sensors.set_gauge_command(gauge_cmd)
        self._rotation.start_rotation(rotation_speed_dps)
        self._capturing = True
        self._state_sink.begin_capture()
        self._schedule_tick(sampling_hz)

    def stop_capture(self, reason: str = "") -> tuple[list[dict], str]:
        self._capturing = False
        self._cancel_tick()
        try:
            self._rotation.stop_rotation()
        except Exception:
            pass
        try:
            self._poll_profile.use_poll_profile(self._prev_poll_profile)  # type: ignore[arg-type]
        except Exception:
            pass
        self._state_sink.end_capture()
        return list(self._samples), reason

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
        if (not self._one_rev) and self._stop_at_ts is not None and now >= self._stop_at_ts:
            self.stop_capture("定时结束")
            return
        # read angle
        if self._angle_enabled:
            try:
                theta = self._sensors.read_axis_angle_deg()
            except Exception:
                theta = float("nan")
            if math.isfinite(theta):
                self._update_rev_progress(float(theta))
                if self._one_rev and self._rev_done():
                    self.stop_capture("已采满一圈")
                    return
        # read gauge
        sample = self._sensors.request_gauge_sample()
        if sample.ok:
            self._samples.append(
                {"ts": now, "od_mm": sample.value_mm, "raw": sample.raw}
            )
        else:
            self._drop_count += 1
        # progress
        elapsed = now - (self._start_ts or now)
        self._state_sink.publish_od_progress(
            CalibrationProgress(
                angle_deg=self._rev_progress_deg,
                elapsed_s=elapsed,
                sample_count=len(self._samples),
            )
        )
        # schedule next tick
        self._schedule_tick(self._sampling_hz)

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
        reference_diameter_mm: float = 180.0,
        outlier_sigma: float = 3.0,
    ) -> dict[str, Any]:
        """Compute B candidate from captured OD samples."""
        values = [
            p["od_mm"] for p in self._samples
            if isinstance(p.get("od_mm"), (int, float)) and math.isfinite(float(p["od_mm"]))
        ]
        if len(values) < 10:
            return {"ok": False, "reason": f"样本不足 (需>=10, got {len(values)})"}
        result = compute_od_b_candidate(values, float(reference_diameter_mm))
        return {"ok": True, "b_mm": result.b_candidate, "mean_mm": result.mean_sum, "n": len(values)}


__all__ = ["OdCalibrationService"]
