"""Tests for OdCalibrationService — port-based OD gauge calibration."""
from __future__ import annotations

from typing import Any

from machine.device_gateway import PollProfile
from services.calibration_context import CalibrationProgress, GaugeSample
from services.od_calibration import OdCalibrationService


# -- Fake ports (same pattern as test_id_single_calibration_service) --------

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
        self._gauge_samples: list[GaugeSample] = []
        self._gauge_cmd: str = ""
    def read_axis_angle_deg(self) -> float:
        return self.angle_deg
    def read_cl_out145_cached(self) -> Any:
        return GaugeSample(value_mm=0.0, ok=True)
    def request_gauge_sample(self) -> GaugeSample:
        if self._gauge_samples:
            return self._gauge_samples.pop(0)
        return GaugeSample(value_mm=100.0, ok=True)
    def set_gauge_command(self, cmd: str) -> None:
        self._gauge_cmd = cmd


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
        self.progress: list[CalibrationProgress] = []
    def begin_capture(self) -> None:
        self.events.append("begin_capture")
    def end_capture(self) -> None:
        self.events.append("end_capture")
    def capture_failed(self, msg: str) -> None:
        self.events.append(f"failed:{msg}")
    def publish_progress(self, progress: CalibrationProgress) -> None:
        self.progress.append(progress)

    def publish_od_progress(self, progress: CalibrationProgress) -> None:
        self.progress.append(progress)

    def publish_id_progress(self, progress: CalibrationProgress) -> None:
        self.progress.append(progress)

    def publish_id_single_progress(self, progress: CalibrationProgress) -> None:
        self.progress.append(progress)


class _FakePollProfilePort:
    def __init__(self) -> None:
        self.profiles: list[str] = []
    def use_poll_profile(self, profile: PollProfile) -> None:
        self.profiles.append(profile)


class _FakeCalibrationRepo:
    def __init__(self) -> None:
        self.saved: list[dict] = []
    def save_od_active(self, data: dict) -> None:
        self.saved.append(data)


def _make_service(**overrides: Any) -> OdCalibrationService:
    return OdCalibrationService(
        rotation=overrides.get("rotation", _FakeRotationPort()),
        sensors=overrides.get("sensors", _FakeSensorPort()),
        scheduler=overrides.get("scheduler", _FakeSchedulerPort()),
        state_sink=overrides.get("state_sink", _FakeStateSink()),
        poll_profile=overrides.get("poll_profile", _FakePollProfilePort()),
        repository=overrides.get("repository", _FakeCalibrationRepo()),
    )


class TestOdLifecycle:
    def test_start_capture_sets_poll_profile_and_starts_rotation(self) -> None:
        poll = _FakePollProfilePort()
        rot = _FakeRotationPort()
        svc = _make_service(poll_profile=poll, rotation=rot)
        svc.start_capture(rotation_speed_dps=10.0, sampling_hz=20.0, capture_duration_s=10.0)
        assert poll.profiles == ["sampling"]
        assert rot.started == [10.0]

    def test_start_capture_sets_gauge_command(self) -> None:
        sensors = _FakeSensorPort()
        svc = _make_service(sensors=sensors)
        svc.start_capture(rotation_speed_dps=10.0, sampling_hz=20.0, capture_duration_s=10.0, gauge_cmd="M0,1")
        assert sensors._gauge_cmd == "M0,1"

    def test_stop_capture_stops_rotation_and_restores_profile(self) -> None:
        poll = _FakePollProfilePort()
        rot = _FakeRotationPort()
        svc = _make_service(poll_profile=poll, rotation=rot)
        svc.start_capture(rotation_speed_dps=10.0, sampling_hz=20.0, capture_duration_s=10.0)
        svc.stop_capture()
        assert rot.stopped == 1
        assert poll.profiles == ["sampling", "normal"]

    def test_one_rev_without_angle_falls_back_to_timed(self) -> None:
        sensors = _FakeSensorPort()
        sink = _FakeStateSink()
        svc = _make_service(sensors=sensors, state_sink=sink)
        svc.start_capture(rotation_speed_dps=10.0, sampling_hz=20.0, capture_duration_s=10.0, mode="one_rev", angle_enabled=False)
        assert "failed" in str(sink.events)


class TestOdComputation:
    def test_compute_candidate_fails_with_insufficient_samples(self) -> None:
        svc = _make_service()
        result = svc.compute_candidate(180.0, 3.0)
        assert result["ok"] is False

    def test_compute_candidate_succeeds_with_valid_samples(self) -> None:
        svc = _make_service()
        svc._samples = [{"od_mm": 50.0 + i * 0.01} for i in range(100)]
        result = svc.compute_candidate(100.0, 3.0)
        assert result["ok"] is True
        assert "b_mm" in result
