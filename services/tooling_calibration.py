from __future__ import annotations

"""Tooling (geometry_v2) calibration service — port-based, host-free.

Drives the 「几何标定 V2」page:
  - ID probe pose (Phase 3): multi-reclamp joint LM fit of s / axis / q.
  - OD axis azimuth psi (Phase 2): single-edge support reconstruction +
    cross-correlation against the reference profile.
  - Synthetic self-test (Box5).

Capture mechanics mirror IdCalibrationService (rotation + scheduler tick +
state-sink progress). The compute/apply methods are independent of the tick
loop so they can be unit-tested with pre-built datasets.
"""

import math
import time
from typing import Any

import numpy as np

from domain.geometry_calibration import (
    IdCalDataset,
    ToolingCalibration,
    calibrate_id_tooling,
    estimate_od_axis_psi,
    run_synthetic_selftest,
)
from machine.ports import RotationPort
from services.calibration_context import CalibrationProgress, ClSample
from services.calibration_ports import (
    CalibrationRepositoryProtocol,
    CalibrationSensorPort,
    CalibrationStateSink,
    PollProfilePort,
    SchedulerPort,
)


class ToolingCalibrationService:
    """Port-based geometry_v2 tooling calibration (ID pose + OD psi + selftest)."""

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

        self._capturing = False
        self._mode = "id"  # "id" | "od"
        self._samples: list[dict[str, Any]] = []
        self._datasets: list[IdCalDataset] = []
        self._start_ts: float | None = None
        self._theta_start: float | None = None
        self._theta_last: float | None = None
        self._theta_unwrap = 0.0
        self._rev_progress_deg = 0.0
        self._schedule_handle: object | None = None
        self._sampling_hz = 20.0
        self._id_pose_candidate: dict[str, Any] | None = None
        self._od_psi_candidate: float | None = None

    # -- capture lifecycle --------------------------------------------------

    def start_capture(
        self,
        *,
        mode: str = "id",
        rotation_speed_dps: float = 10.0,
        sampling_hz: float = 20.0,
    ) -> None:
        if self._capturing:
            return
        self._mode = "od" if str(mode).lower() == "od" else "id"
        self._samples = []
        self._theta_start = None
        self._theta_last = None
        self._theta_unwrap = 0.0
        self._rev_progress_deg = 0.0
        self._sampling_hz = float(sampling_hz)
        self._start_ts = time.time()
        self._poll_profile.use_poll_profile("sampling")  # type: ignore[arg-type]
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
            self._poll_profile.use_poll_profile("normal")  # type: ignore[arg-type]
        except Exception:
            pass
        self._state_sink.end_capture()

    def clear_capture(self) -> None:
        self._capturing = False
        self._cancel_tick()
        self._samples = []
        self._theta_start = None
        self._theta_last = None
        self._theta_unwrap = 0.0
        self._rev_progress_deg = 0.0
        self._state_sink.end_capture()

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
        try:
            theta = self._sensors.read_axis_angle_deg()
        except Exception:
            theta = float("nan")
        if math.isfinite(theta):
            self._update_rev_progress(float(theta))
            if self._rev_progress_deg >= 360.0:
                self.stop_capture("已采满一圈")
                return
        try:
            cl = self._sensors.read_cl_out145_cached()
        except Exception:
            cl = ClSample(ok=False)
        if math.isfinite(theta) and cl.ok:
            if self._mode == "id":
                if cl.out1 is not None and cl.out2 is not None and math.isfinite(float(cl.out1)) and math.isfinite(float(cl.out2)):
                    self._samples.append({"theta_deg": theta, "x1": float(cl.out1), "x2": float(cl.out2)})
            else:  # od: single-edge support proxy via OUT1
                if cl.out1 is not None and math.isfinite(float(cl.out1)):
                    self._samples.append({"theta_deg": theta, "h": float(cl.out1)})
        self._state_sink.publish_id_progress(
            CalibrationProgress(
                angle_deg=self._rev_progress_deg,
                elapsed_s=now - (self._start_ts or now),
                sample_count=len(self._samples),
            )
        )
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

    # -- ID pose (Phase 3) --------------------------------------------------

    def add_dataset(self) -> dict[str, Any]:
        """Snapshot the current ID capture into a reclamp dataset."""
        th, x1, x2 = self._id_arrays(self._samples)
        return self._append_dataset(th, x1, x2)

    def add_dataset_arrays(self, theta_deg: Any, x1: Any, x2: Any) -> dict[str, Any]:
        """Add a dataset directly (used by tests / external capture)."""
        return self._append_dataset(
            np.asarray(theta_deg, dtype=float),
            np.asarray(x1, dtype=float),
            np.asarray(x2, dtype=float),
        )

    def _append_dataset(self, theta_deg: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> dict[str, Any]:
        if theta_deg.size < 8:
            return {"ok": False, "reason": "样本不足(<8)", "n_sets": len(self._datasets)}
        self._datasets.append(IdCalDataset(np.deg2rad(theta_deg), x1, x2))
        return {"ok": True, "n_sets": len(self._datasets), "n_points": int(theta_deg.size)}

    def clear_datasets(self) -> dict[str, Any]:
        self._datasets = []
        self._id_pose_candidate = None
        return {"ok": True, "n_sets": 0}

    def dataset_count(self) -> int:
        return len(self._datasets)

    @staticmethod
    def _id_arrays(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        th: list[float] = []
        x1: list[float] = []
        x2: list[float] = []
        for p in samples:
            t, a, b = p.get("theta_deg"), p.get("x1"), p.get("x2")
            if t is None or a is None or b is None:
                continue
            th.append(float(t))
            x1.append(float(a))
            x2.append(float(b))
        return np.asarray(th, float), np.asarray(x1, float), np.asarray(x2, float)

    def compute_id_pose(self, *, r_known: float, d_init: float) -> dict[str, Any]:
        if not self._datasets:
            return {"ok": False, "reason": "无数据集,请先采集并加入"}
        try:
            res = calibrate_id_tooling(self._datasets, r_known=float(r_known), D_init=float(d_init))
        except Exception as exc:
            return {"ok": False, "reason": f"拟合失败: {exc}"}
        n = res.tooling.probe_b.n
        perp = np.array([-n[1], n[0]])
        s_rec = float((res.tooling.probe_b.f - res.tooling.probe_a.f) @ perp)
        q = (res.tooling.probe_a.f + res.tooling.probe_b.f) / 2.0
        axis_deg = float(np.rad2deg(math.atan2(float(n[1]), float(n[0]))))
        candidate = {
            "ok": bool(res.success),
            "s_lateral": s_rec,
            "axis_deg": axis_deg,
            "qx": float(q[0]),
            "qy": float(q[1]),
            "D_eff": float(d_init),
            "cost": float(res.cost),
            "n_sets": len(self._datasets),
        }
        self._id_pose_candidate = candidate
        return candidate

    def apply_id_pose(self) -> dict[str, Any]:
        if self._id_pose_candidate is None:
            return {"ok": False, "reason": "请先计算"}
        c = self._id_pose_candidate
        tc = self.load_tooling()
        tc.id_D_eff = float(c["D_eff"])
        tc.id_s_lateral = float(c["s_lateral"])
        tc.id_axis_deg = float(c["axis_deg"])
        tc.id_qx = float(c["qx"])
        tc.id_qy = float(c["qy"])
        tc.meta = {**dict(tc.meta), "id_pose_ts": time.time(), "id_pose_cost": float(c["cost"])}
        self._repository.save_tooling_active(tc.to_dict())
        return {"ok": True, **c}

    # -- OD psi (Phase 2) ---------------------------------------------------

    def compute_od_psi(self, *, ref_phi: Any = None, ref_dr: Any = None) -> dict[str, Any]:
        th: list[float] = []
        h: list[float] = []
        for p in self._samples:
            t, hv = p.get("theta_deg"), p.get("h")
            if t is None or hv is None:
                continue
            th.append(float(t))
            h.append(float(hv))
        if len(th) < 8:
            return {"ok": False, "reason": "OD 支撑样本不足(<8)"}
        rphi = None if ref_phi is None else np.asarray(ref_phi, float)
        rdr = None if ref_dr is None else np.asarray(ref_dr, float)
        psi = estimate_od_axis_psi(np.asarray(th, float), np.asarray(h, float), rphi, rdr)
        self._od_psi_candidate = float(psi)
        return {"ok": True, "psi_deg": float(psi), "has_reference": rphi is not None and rdr is not None, "n": len(th)}

    def apply_od_psi(self) -> dict[str, Any]:
        if self._od_psi_candidate is None:
            return {"ok": False, "reason": "请先计算"}
        tc = self.load_tooling()
        tc.od_psi_deg = float(self._od_psi_candidate)
        tc.meta = {**dict(tc.meta), "od_psi_ts": time.time()}
        self._repository.save_tooling_active(tc.to_dict())
        return {"ok": True, "psi_deg": float(self._od_psi_candidate)}

    # -- shared -------------------------------------------------------------

    def run_selftest(self) -> dict[str, Any]:
        return run_synthetic_selftest()

    def load_tooling(self) -> ToolingCalibration:
        try:
            return ToolingCalibration.from_dict(self._repository.load_tooling_active())
        except Exception:
            return ToolingCalibration()

    def load_active(self) -> dict[str, Any]:
        return dict(self._repository.load_tooling_active() or {})

    def clear_all(self) -> dict[str, Any]:
        self._datasets = []
        self._id_pose_candidate = None
        self._od_psi_candidate = None
        self._repository.save_tooling_active({})
        return {"ok": True}


__all__ = ["ToolingCalibrationService"]
