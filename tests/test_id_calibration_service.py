"""Tests for IdCalibrationService — port-based ID calibration."""
from __future__ import annotations

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
    def read_axis_angle_deg(self) -> float:
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
    def begin_capture(self) -> None:
        self.events.append("begin_capture")
    def end_capture(self) -> None:
        self.events.append("end_capture")
    def capture_failed(self, msg: str) -> None:
        self.events.append(f"failed:{msg}")
    def publish_progress(self, progress: CalibrationProgress) -> None:
        pass


class _FakePollProfilePort:
    def __init__(self) -> None:
        self.profiles: list[str] = []
    def use_poll_profile(self, profile: PollProfile) -> None:
        self.profiles.append(profile)


class _FakeCalibrationRepo:
    def __init__(self) -> None:
        self.saved: list[dict] = []
    def save_id_active(self, data: dict) -> None:
        self.saved.append(data)


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
