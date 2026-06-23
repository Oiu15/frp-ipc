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
from typing import Any, Optional

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
        self._od_zero_candidate: dict[str, float] | None = None
        self._axis_stations: list[tuple[float, float, float]] = []
        self._axis_candidate: tuple[float, float] | None = None
        self._chuck_candidate: float | None = None
        self._delta_reg_candidate: tuple[float, float] | None = None

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
        if math.isfinite(theta):
            sample: dict[str, Any] = {"theta_deg": theta}
            # ID probes (Keyence CL OUT1/OUT2 = x1/x2): id & delta modes
            if self._mode in ("id", "delta"):
                try:
                    cl = self._sensors.read_cl_out145_cached()
                except Exception:
                    cl = ClSample(ok=False)
                if cl.ok and cl.out1 is not None and cl.out2 is not None \
                        and math.isfinite(float(cl.out1)) and math.isfinite(float(cl.out2)):
                    sample["x1"] = float(cl.out1)
                    sample["x2"] = float(cl.out2)
            # OD 测径仪 single-edge support (external serial gauge OUT1): od & delta modes
            if self._mode in ("od", "delta"):
                try:
                    g = self._sensors.request_gauge_sample()
                    od1 = float(g.value_mm) if (g.ok and g.value_mm is not None) else None
                except Exception:
                    od1 = None
                if od1 is not None and math.isfinite(od1):
                    sample["od_out1"] = od1
            # accept the sample only if it carries the data its mode needs
            need = {"id": ("x1", "x2"), "od": ("od_out1",), "delta": ("x1", "od_out1")}[self._mode]
            if all(k in sample for k in need):
                self._samples.append(sample)
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

    # -- OD support helpers -------------------------------------------------

    def _od_theta_h(self) -> tuple[np.ndarray, np.ndarray]:
        th: list[float] = []
        h: list[float] = []
        for p in self._samples:
            t, hv = p.get("theta_deg"), p.get("od_out1")
            if t is None or hv is None:
                continue
            th.append(float(t))
            h.append(float(hv))
        return np.deg2rad(np.asarray(th, float)), np.asarray(h, float)

    def _od_center(self) -> Optional[np.ndarray]:
        theta, h = self._od_theta_h()
        if theta.size < 8:
            return None
        from domain.geometry_calibration import reconstruct_od_circle

        cf, _ = reconstruct_od_circle(theta, h, self.load_tooling().od_cal())
        return np.array([cf.cx, cf.cy])

    def _id_center(self) -> Optional[np.ndarray]:
        th: list[float] = []
        x1: list[float] = []
        x2: list[float] = []
        for p in self._samples:
            t, a, b = p.get("theta_deg"), p.get("x1"), p.get("x2")
            if t is None or a is None or b is None:
                continue
            th.append(float(t))
            x1.append(float(a))
            x2.append(float(b))
        if len(th) < 8:
            return None
        from domain.geometry_calibration import id_points_from_readings
        from domain.geometry_fit import fit_circle_geometric

        tc = self.load_tooling()
        if not tc.id_calibrated():
            return None
        pts = id_points_from_readings(
            tc.id_tooling(), np.deg2rad(np.asarray(th, float)),
            np.asarray(x1, float), np.asarray(x2, float),
        )
        cf = fit_circle_geometric(pts[:, 0], pts[:, 1])
        return np.array([cf.cx, cf.cy])

    # -- OD zero (Phase 1, v2) ----------------------------------------------

    def compute_od_zero(self, *, known_od: float) -> dict[str, Any]:
        theta, h = self._od_theta_h()
        if theta.size < 8:
            return {"ok": False, "reason": "OD 支撑样本不足(<8)"}
        from domain.geometry_calibration import solve_od_zero

        cal = solve_od_zero(theta, h, float(known_od))
        self._od_zero_candidate = cal
        return {"ok": True, "od_b": float(cal["b"]), "known_od": float(known_od)}

    def apply_od_zero(self) -> dict[str, Any]:
        if self._od_zero_candidate is None:
            return {"ok": False, "reason": "请先计算"}
        tc = self.load_tooling()
        tc.od_k0 = float(self._od_zero_candidate["k0"])
        tc.od_b = float(self._od_zero_candidate["b"])
        tc.meta = {**dict(tc.meta), "od_zero_ts": time.time()}
        self._repository.save_tooling_active(tc.to_dict())
        return {"ok": True, "od_b": float(tc.od_b)}

    # -- OD psi (Phase 2) ---------------------------------------------------

    def capture_od_reference(self) -> dict[str, Any]:
        """Store the master's radial-deviation profile as the psi angular reference."""
        theta, h = self._od_theta_h()
        if theta.size < 8:
            return {"ok": False, "reason": "OD 支撑样本不足(<8)"}
        from domain.geometry_calibration import od_reference_profile

        phi, dr = od_reference_profile(theta, h, self.load_tooling().od_cal())
        tc = self.load_tooling()
        tc.meta = {**dict(tc.meta), "od_ref_phi": [float(x) for x in phi.tolist()],
                   "od_ref_dr": [float(x) for x in dr.tolist()], "od_ref_ts": time.time()}
        self._repository.save_tooling_active(tc.to_dict())
        return {"ok": True, "n": int(phi.size)}

    def compute_od_psi(self) -> dict[str, Any]:
        theta, h = self._od_theta_h()
        if theta.size < 8:
            return {"ok": False, "reason": "OD 支撑样本不足(<8)"}
        meta = self.load_tooling().meta
        rphi = meta.get("od_ref_phi")
        rdr = meta.get("od_ref_dr")
        ref_phi = np.asarray(rphi, float) if rphi else None
        ref_dr = np.asarray(rdr, float) if rdr else None
        psi = estimate_od_axis_psi(theta, h, ref_phi, ref_dr)
        self._od_psi_candidate = float(psi)
        return {"ok": True, "psi_deg": float(psi), "has_reference": ref_phi is not None, "n": int(theta.size)}

    def apply_od_psi(self) -> dict[str, Any]:
        if self._od_psi_candidate is None:
            return {"ok": False, "reason": "请先计算"}
        tc = self.load_tooling()
        tc.od_psi_deg = float(self._od_psi_candidate)
        tc.meta = {**dict(tc.meta), "od_psi_ts": time.time()}
        self._repository.save_tooling_active(tc.to_dict())
        return {"ok": True, "psi_deg": float(self._od_psi_candidate)}

    # -- spindle axis straightness + chuck error (Phase 0) ------------------

    def record_axis_station(self, *, z: float) -> dict[str, Any]:
        """Snapshot the current OD circle center at axial height z (mm)."""
        c = self._od_center()
        if c is None:
            return {"ok": False, "reason": "OD 圆心不可得(支撑样本不足或未采)"}
        self._axis_stations.append((float(z), float(c[0]), float(c[1])))
        return {"ok": True, "n_stations": len(self._axis_stations), "z": float(z),
                "cx": float(c[0]), "cy": float(c[1])}

    def clear_axis_stations(self) -> dict[str, Any]:
        self._axis_stations = []
        return {"ok": True, "n_stations": 0}

    def compute_axis(self) -> dict[str, Any]:
        if len(self._axis_stations) < 2:
            return {"ok": False, "reason": "至少需要两个高度站点"}
        from domain.geometry_fit import centerline_tilt

        zs = np.array([s[0] for s in self._axis_stations], float)
        centers = np.array([[s[1], s[2]] for s in self._axis_stations], float)
        tau = centerline_tilt(centers, zs)
        self._axis_candidate = (float(tau[0]), float(tau[1]))
        return {"ok": True, "axis_slope_x": float(tau[0]), "axis_slope_y": float(tau[1]),
                "n_stations": len(self._axis_stations)}

    def apply_axis(self) -> dict[str, Any]:
        if self._axis_candidate is None:
            return {"ok": False, "reason": "请先计算"}
        tc = self.load_tooling()
        tc.axis_slope_x, tc.axis_slope_y = float(self._axis_candidate[0]), float(self._axis_candidate[1])
        tc.meta = {**dict(tc.meta), "axis_stations": list(self._axis_stations), "axis_ts": time.time()}
        self._repository.save_tooling_active(tc.to_dict())
        return {"ok": True, "axis_slope_x": tc.axis_slope_x, "axis_slope_y": tc.axis_slope_y}

    def compute_chuck_bound(self, *, cert_roundness: float) -> dict[str, Any]:
        theta, h = self._od_theta_h()
        if theta.size < 8:
            return {"ok": False, "reason": "OD 支撑样本不足(<8)"}
        from domain.geometry_calibration import reconstruct_od_circle

        _, rnd = reconstruct_od_circle(theta, h, self.load_tooling().od_cal())
        e_bound = max(0.0, float(rnd.roundness_lsc) - float(cert_roundness))
        self._chuck_candidate = e_bound
        return {"ok": True, "chuck_error_bound": e_bound, "roundness_meas": float(rnd.roundness_lsc),
                "over_budget": bool(e_bound > 0.035)}

    def apply_chuck_bound(self) -> dict[str, Any]:
        if self._chuck_candidate is None:
            return {"ok": False, "reason": "请先计算"}
        tc = self.load_tooling()
        tc.chuck_error_bound = float(self._chuck_candidate)
        tc.meta = {**dict(tc.meta), "chuck_ts": time.time()}
        self._repository.save_tooling_active(tc.to_dict())
        return {"ok": True, "chuck_error_bound": float(tc.chuck_error_bound)}

    # -- cross-registration delta_reg (Phase 4) -----------------------------

    def compute_delta_reg(self) -> dict[str, Any]:
        c_o = self._od_center()
        c_i = self._id_center()
        if c_o is None or c_i is None:
            return {"ok": False, "reason": "需同测内外圆心(OD 已标零位 + ID 已标位姿)"}
        d = c_o - c_i
        self._delta_reg_candidate = (float(d[0]), float(d[1]))
        return {"ok": True, "delta_reg": [float(d[0]), float(d[1])]}

    def apply_delta_reg(self) -> dict[str, Any]:
        if self._delta_reg_candidate is None:
            return {"ok": False, "reason": "请先计算"}
        tc = self.load_tooling()
        tc.delta_reg = (float(self._delta_reg_candidate[0]), float(self._delta_reg_candidate[1]))
        tc.meta = {**dict(tc.meta), "delta_reg_ts": time.time()}
        self._repository.save_tooling_active(tc.to_dict())
        return {"ok": True, "delta_reg": list(tc.delta_reg)}

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
        self._od_zero_candidate = None
        self._axis_stations = []
        self._axis_candidate = None
        self._chuck_candidate = None
        self._delta_reg_candidate = None
        self._repository.save_tooling_active({})
        return {"ok": True}


__all__ = ["ToolingCalibrationService"]
