"""Tests for IdCalibrationService — port-based ID calibration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from machine.device_gateway import PollProfile
from services.calibration_context import CalibrationProgress, ClSample
from services.id_calibration import IdCalibrationService


class _FakeRotationPort:
    def __init__(self) -> None:
        self.started: list[float] = []
        self.stopped = 0
    def start_rotation(self, rpm: float) -> None:
        self.started.append(rpm)
    def stop_rotation(self) -> None:
        self.stopped += 1


class _FakeSensorPort:
    def __init__(self) -> None:
        self.angle_deg = 45.0
        self.angles: list[float] = []
    def read_axis_angle_deg(self) -> float:
        if self.angles:
            return self.angles.pop(0)
        return self.angle_deg
    def read_cl_out145_cached(self) -> ClSample:
        return ClSample(out1=100.0, out2=49.0, out4=50.0, out5=50.0, ok=True)
    def request_gauge_sample(self) -> Any:
        pass
    def set_gauge_command(self, cmd: str) -> None:
        pass


class _FakeSchedulerPort:
    def __init__(self) -> None:
        self.scheduled: list[tuple[int, object]] = []
        self.cancelled: list[object] = []
    def schedule_once(self, delay_ms: int, callback: object) -> object:
        self.scheduled.append((delay_ms, callback))
        return ("handle", delay_ms)
    def cancel(self, handle: object) -> None:
        self.cancelled.append(handle)


class _FakeStateSink:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.verify_results: list[dict[str, Any]] = []
    def begin_capture(self) -> None:
        self.events.append("begin_capture")
    def end_capture(self) -> None:
        self.events.append("end_capture")
    def capture_failed(self, msg: str) -> None:
        self.events.append(f"failed:{msg}")
    def publish_progress(self, progress: CalibrationProgress) -> None:
        pass
    def publish_od_progress(self, progress: CalibrationProgress) -> None:
        pass
    def publish_id_progress(self, progress: CalibrationProgress) -> None:
        pass
    def publish_id_single_progress(self, progress: CalibrationProgress) -> None:
        pass
    def publish_id_verify_result(self, result: dict[str, Any]) -> None:
        self.verify_results.append(dict(result))


class _FakePollProfilePort:
    def __init__(self) -> None:
        self.profiles: list[str] = []
    def use_poll_profile(self, profile: PollProfile) -> None:
        self.profiles.append(profile)


class _FakeCalibrationRepo:
    def __init__(self) -> None:
        self.saved: list[dict] = []
        self.exported_id: list[list[dict[str, Any]]] = []
        self.export_error: Exception | None = None
        self.active: dict[str, Any] = {}
    def save_id_active(self, data: dict) -> None:
        self.saved.append(data)
    def load_id_active(self) -> dict[str, Any]:
        return dict(self.active)
    def export_id_raw(self, points: list[dict[str, Any]]) -> Path:
        if self.export_error is not None:
            raise self.export_error
        self.exported_id.append(points)
        return Path("id_raw.csv")


def _make_service(**overrides: Any) -> IdCalibrationService:
    return IdCalibrationService(
        rotation=overrides.get("rotation", _FakeRotationPort()),
        sensors=overrides.get("sensors", _FakeSensorPort()),
        scheduler=overrides.get("scheduler", _FakeSchedulerPort()),
        state_sink=overrides.get("state_sink", _FakeStateSink()),
        poll_profile=overrides.get("poll_profile", _FakePollProfilePort()),
        repository=overrides.get("repository", _FakeCalibrationRepo()),
    )


class TestIdLifecycle:
    def test_start_capture_starts_rotation(self) -> None:
        rot = _FakeRotationPort()
        svc = _make_service(rotation=rot)
        svc.start_capture(rotation_speed_dps=10.0, sampling_hz=20.0, capture_duration_s=10.0)
        assert rot.started == [10.0]

    def test_stop_capture_stops_rotation(self) -> None:
        rot = _FakeRotationPort()
        svc = _make_service(rotation=rot)
        svc.start_capture(rotation_speed_dps=10.0, sampling_hz=20.0, capture_duration_s=1.0)
        svc.stop_capture()
        assert rot.stopped == 1

    def test_start_capture_clears_old_samples(self) -> None:
        svc = _make_service()
        svc._samples = [{"old": True}]
        svc.start_capture(rotation_speed_dps=10.0, sampling_hz=20.0, capture_duration_s=10.0)
        assert svc._samples == []

    def test_start_capture_uses_requested_sampling_hz(self) -> None:
        sched = _FakeSchedulerPort()
        svc = _make_service(scheduler=sched)
        svc.start_capture(rotation_speed_dps=10.0, sampling_hz=10.0, capture_duration_s=10.0)
        assert sched.scheduled[-1][0] == 100

    def test_start_verify_capture_forces_one_rev_and_starts_rotation(self) -> None:
        rot = _FakeRotationPort()
        svc = _make_service(rotation=rot)

        svc.start_verify_capture(
            rotation_speed_dps=12.0,
            sampling_hz=20.0,
            capture_duration_s=10.0,
            delta_c_mm=0.0,
            reference_diameter_mm=150.0,
        )

        assert svc._one_rev is True
        assert svc._verify_pending is True
        assert rot.started == [12.0]


class TestIdComputation:
    def test_compute_candidate_fails_with_insufficient_samples(self) -> None:
        svc = _make_service()
        result = svc.compute_candidate(150.0)
        assert result["ok"] is False

    def test_compute_candidate_succeeds_with_valid_samples(self) -> None:
        svc = _make_service()
        svc._samples = [
            {"theta_deg": float(i), "c_mm": 50.0 + i * 0.01, "m_mm": 50.0 - i * 0.01}
            for i in range(0, 360, 5)
        ]
        result = svc.compute_candidate(150.0)
        assert result["ok"] is True

    def test_apply_result_persists(self) -> None:
        repo = _FakeCalibrationRepo()
        svc = _make_service(repository=repo)
        svc._samples = [{"theta_deg": float(i), "c_mm": 50.0 + i * 0.01, "m_mm": 50.0 - i * 0.01} for i in range(0, 360, 5)]
        svc.compute_candidate(150.0)
        result = svc.apply_result(150.0)
        assert result["ok"] is True
        assert len(repo.saved) == 1


class TestIdRawExport:
    def test_export_raw_uses_repository_samples(self) -> None:
        repo = _FakeCalibrationRepo()
        svc = _make_service(repository=repo)
        svc._samples = [{"theta_deg": 1.0, "c_mm": 2.0, "m_mm": 3.0}]

        result = svc.export_raw()

        assert result["ok"] is True
        assert result["path"].name == "id_raw.csv"
        assert result["n"] == 1
        assert repo.exported_id == [[{"theta_deg": 1.0, "c_mm": 2.0, "m_mm": 3.0}]]

    def test_export_raw_returns_no_data_without_samples(self) -> None:
        svc = _make_service()

        result = svc.export_raw()

        assert result == {"ok": False, "reason": "无数据", "n": 0}

    def test_export_raw_reports_repository_error(self) -> None:
        repo = _FakeCalibrationRepo()
        repo.export_error = RuntimeError("disk full")
        svc = _make_service(repository=repo)
        svc._samples = [{"theta_deg": 1.0, "c_mm": 2.0, "m_mm": 3.0}]

        result = svc.export_raw()

        assert result["ok"] is False
        assert result["reason"] == "导出失败: disk full"


class TestIdVerify:
    def test_verify_stop_publishes_success_result(self) -> None:
        sink = _FakeStateSink()
        svc = _make_service(state_sink=sink)
        svc.start_verify_capture(
            rotation_speed_dps=10.0,
            sampling_hz=20.0,
            capture_duration_s=10.0,
            delta_c_mm=100.0,
            reference_diameter_mm=150.0,
        )
        svc._samples = [
            {"theta_deg": float(i), "c_mm": 50.0, "m_mm": 50.0}
            for i in range(0, 361, 5)
        ]

        svc.stop_capture("已采满一圈")

        assert sink.verify_results
        result = sink.verify_results[-1]
        assert result["ok"] is True
        assert result["n"] == 73
        assert abs(result["err_mm"]) < 1e-6
        assert sink.events[-1] == "end_capture"

    def test_verify_stop_publishes_sample_shortage_failure(self) -> None:
        sink = _FakeStateSink()
        svc = _make_service(state_sink=sink)
        svc.start_verify_capture(
            rotation_speed_dps=10.0,
            sampling_hz=20.0,
            capture_duration_s=10.0,
            delta_c_mm=0.0,
            reference_diameter_mm=150.0,
        )
        svc._samples = [{"theta_deg": 1.0, "c_mm": 50.0, "m_mm": 50.0}]

        svc.stop_capture("manual")

        assert sink.verify_results[-1] == {"ok": False, "reason": "复核样本不足: N=1", "n": 1}
        assert sink.events[-1] == "failed:复核样本不足: N=1"

    def test_one_rev_tick_auto_stop_publishes_verify_result(self) -> None:
        sensors = _FakeSensorPort()
        sensors.angles = [0.0, 120.0, 240.0, 360.0]
        sink = _FakeStateSink()
        sched = _FakeSchedulerPort()
        svc = _make_service(sensors=sensors, state_sink=sink, scheduler=sched)
        svc.start_verify_capture(
            rotation_speed_dps=10.0,
            sampling_hz=20.0,
            capture_duration_s=10.0,
            delta_c_mm=0.0,
            reference_diameter_mm=150.0,
        )

        for _ in range(4):
            callback = sched.scheduled[-1][1]
            callback()

        assert svc._capturing is False
        assert sink.verify_results
