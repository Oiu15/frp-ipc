from __future__ import annotations

"""OD calibration service — port-based, host-free.

Extracted from ``calibration_service.py``.  Handles timed or one-revolution
OD (outer-diameter) capture using the external gauge (Keyence CL-3000 OUT1).
"""

import math
import time
from typing import Any, Mapping

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
        self._b_candidate: float | None = None
        self._last_reference_diameter_mm: float = 180.0

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
        self._sampling_hz = float(sampling_hz)
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

    def clear_capture(self) -> None:
        self._capturing = False
        self._cancel_tick()
        self._samples = []
        self._drop_count = 0
        self._start_ts = None
        self._stop_at_ts = None
        self._theta_start = None
        self._theta_last = None
        self._theta_unwrap = 0.0
        self._rev_progress_deg = 0.0
        self._b_candidate = None
        self._state_sink.end_capture()

    def handle_gauge_sample(self, payload: Mapping[str, Any]) -> None:
        """Accept one async OD gauge sample without depending on AppHost."""
        if not self._capturing:
            return

        now = time.time()
        theta: float | None = None
        theta_rel: float | None = None
        if self._angle_enabled:
            try:
                angle = float(self._sensors.read_axis_angle_deg())
            except Exception:
                angle = float("nan")
            if math.isfinite(angle):
                theta = angle
                if self._one_rev:
                    self._update_rev_progress(angle)
                    theta_rel = float(self._rev_progress_deg)

        j1 = str(payload.get("judge", "") or "").strip().upper()
        j2 = str(payload.get("judge2", "") or "").strip().upper()
        if j1 and j1 != "GO":
            self._drop_count += 1
        if j2 and j2 != "GO":
            self._drop_count += 1

        try:
            ts = float(payload.get("ts", now))
        except Exception:
            ts = now

        point: dict[str, Any] = {
            "ts": ts,
            "raw": str(payload.get("raw", "") or "").strip(),
            "v1": payload.get("od"),
            "j1": j1,
            "v2": payload.get("od2"),
            "j2": j2,
            "theta": theta,
            "theta_rel": theta_rel,
        }

        self._samples.append(point)
        self._state_sink.publish_od_sample(point, len(self._samples), self._drop_count)
        elapsed = now - (self._start_ts or now)
        self._state_sink.publish_od_progress(
            CalibrationProgress(
                angle_deg=self._rev_progress_deg,
                elapsed_s=elapsed,
                sample_count=len(self._samples),
            )
        )
        if self._one_rev and self._rev_done():
            self.stop_capture("已采满一圈")

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
        values: list[float] = []
        for p in self._samples:
            try:
                if p.get("v1") is not None and p.get("v2") is not None:
                    value = float(p["v1"]) + float(p["v2"])
                else:
                    value = float(p["od_mm"])
                if math.isfinite(value):
                    values.append(value)
            except Exception:
                continue
        if len(values) < 10:
            return {"ok": False, "reason": f"样本不足 (需>=10, got {len(values)})"}
        result = compute_od_b_candidate(values, float(reference_diameter_mm))
        if not result.ok or result.b_candidate is None:
            return {"ok": False, "reason": result.reason or "OD candidate failed"}
        self._b_candidate = float(result.b_candidate)
        self._last_reference_diameter_mm = float(reference_diameter_mm)
        return {"ok": True, "b_mm": self._b_candidate, "mean_mm": result.mean_sum, "n": len(values)}

    def apply_result(
        self,
        reference_diameter_mm: float | None = None,
        *,
        gauge_cmd: str = "M0,1",
        out1_map: str = "L",
    ) -> dict[str, Any]:
        if self._b_candidate is None:
            return {"ok": False, "reason": "请先计算"}
        d_ref = self._last_reference_diameter_mm if reference_diameter_mm is None else float(reference_diameter_mm)
        data = {
            "B_active": float(self._b_candidate),
            "D_ref": float(d_ref),
            "cmd_used": str(gauge_cmd or ""),
            "out_map": {
                "OUT1": str(out1_map or "L"),
                "OUT2": ("R" if str(out1_map or "L").upper() == "L" else "L"),
            },
            "created_at_ts": time.time(),
            "stats": {"n": len(self._samples)},
        }
        self._repository.save_od_active(data)
        return {"ok": True, "b_mm": float(self._b_candidate)}

    def export_raw(self) -> dict[str, Any]:
        """Export captured raw OD samples through the repository."""
        points = list(self._samples)
        if not points:
            return {"ok": False, "reason": "无数据", "n": 0}
        try:
            path = self._repository.export_od_raw(points)
            return {"ok": True, "path": path, "n": len(points)}
        except Exception as exc:
            return {"ok": False, "reason": f"导出失败: {exc}", "n": len(points)}


__all__ = ["OdCalibrationService"]
