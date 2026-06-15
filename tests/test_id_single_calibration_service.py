"""Tests for IdSingleCalibrationService — test-first with FakePorts.

Covers lifecycle (start/stop/clear), sampling (tick), computation,
and critical error recovery (rotation stop, poll profile restore).
"""
from __future__ import annotations

from typing import Any

from machine.device_gateway import PollProfile
from services.calibration_context import CalibrationProgress, ClSample, GaugeSample
from services.id_single_calibration import IdSingleCalibrationService


# ---------------------------------------------------------------------------
# Fake ports
# ---------------------------------------------------------------------------

class _FakeRotationPort:
    def __init__(self) -> None:
        self.started: list[float] = []
        self.stopped = 0
        self._raise_on: str | None = None
        self._angle_feed: list[float] = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0]
        self._angle_idx = 0

    def start_rotation(self, rpm: float) -> None:
        if self._raise_on == "start":
            raise RuntimeError("rotation start failed")
        self.started.append(rpm)

    def stop_rotation(self) -> None:
        if self._raise_on == "stop":
            raise RuntimeError("rotation stop failed")
        self.stopped += 1

    def next_angle(self) -> float:
        val = self._angle_feed[self._angle_idx % len(self._angle_feed)]
        self._angle_idx += 1
        return val


class _FakeSensorPort:
    def __init__(self) -> None:
        self.angle_deg = 45.0
        self._raise_on_read: str | None = None
        self._gauge_samples: list[GaugeSample] = [GaugeSample(value_mm=100.0, ok=True)]

    def read_axis_angle_deg(self) -> float:
        if self._raise_on_read == "angle":
            raise RuntimeError("angle read failed")
        return self.angle_deg

    def read_cl_out145_cached(self) -> ClSample:
        if self._raise_on_read == "cl":
            raise RuntimeError("CL read failed")
        return ClSample(out1=100.0, out4=50.0, out5=50.0, timestamp=0.0, ok=True)

    def request_gauge_sample(self) -> GaugeSample:
        if self._raise_on_read == "gauge":
            return GaugeSample(value_mm=0.0, ok=False, error="gauge error")
        return self._gauge_samples.pop(0) if self._gauge_samples else GaugeSample(value_mm=100.0, ok=True)


class _FakeSchedulerPort:
    def __init__(self) -> None:
        self.scheduled: list[tuple[int, object]] = []
        self.cancelled: list[object] = []
        self._tick_callback: object | None = None

    def schedule_once(self, delay_ms: int, callback: object) -> object:
        self.scheduled.append((delay_ms, callback))
        self._tick_callback = callback
        return ("handle", delay_ms)

    def cancel(self, handle: object) -> None:
        self.cancelled.append(handle)

    def run_pending(self) -> None:
        """Fire the most recently scheduled callback synchronously."""
        if self._tick_callback is not None:
            cb = self._tick_callback
            self._tick_callback = None
            cb()  # type: ignore[operator]


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


class _FakePollProfilePort:
    def __init__(self) -> None:
        self.profiles: list[str] = []

    def use_poll_profile(self, profile: PollProfile) -> None:
        self.profiles.append(profile)


class _FakeCalibrationRepo:
    def __init__(self) -> None:
        self.saved: list[dict] = []

    def save_id_single_active(self, data: dict) -> None:
        self.saved.append(data)


def _make_service(**overrides: Any) -> IdSingleCalibrationService:
    return IdSingleCalibrationService(
        rotation=overrides.get("rotation", _FakeRotationPort()),
        sensors=overrides.get("sensors", _FakeSensorPort()),
        scheduler=overrides.get("scheduler", _FakeSchedulerPort()),
        state_sink=overrides.get("state_sink", _FakeStateSink()),
        poll_profile=overrides.get("poll_profile", _FakePollProfilePort()),
        repository=overrides.get("repository", _FakeCalibrationRepo()),
    )


# ===================================================================
# Lifecycle tests
# ===================================================================

class TestLifecycle:
    """start / stop / clear behaviour."""

    def test_start_capture_clears_old_samples(self) -> None:
        svc = _make_service()
        svc._samples = [{"old": True}]
        svc.start_capture(
            rotation_speed_dps=10.0,
            sampling_hz=20.0,
            capture_duration_s=10.0,
            reference_diameter_mm=150.0,
        )
        assert svc._samples == []

    def test_start_capture_sets_poll_profile(self) -> None:
        poll = _FakePollProfilePort()
        svc = _make_service(poll_profile=poll)
        svc.start_capture(
            rotation_speed_dps=10.0, sampling_hz=20.0,
            capture_duration_s=10.0, reference_diameter_mm=150.0,
        )
        assert poll.profiles == ["sampling"]
        assert svc._capturing is True

    def test_start_capture_starts_rotation(self) -> None:
        rot = _FakeRotationPort()
        svc = _make_service(rotation=rot)
        svc.start_capture(
            rotation_speed_dps=10.0, sampling_hz=20.0,
            capture_duration_s=10.0, reference_diameter_mm=150.0,
        )
        assert rot.started == [10.0]
        assert svc._schedule_handle is not None

    def test_start_capture_records_state_sink_begin(self) -> None:
        sink = _FakeStateSink()
        svc = _make_service(state_sink=sink)
        svc.start_capture(
            rotation_speed_dps=10.0, sampling_hz=20.0,
            capture_duration_s=10.0, reference_diameter_mm=150.0,
        )
        assert "begin_capture" in sink.events

    def test_stop_capture_stops_rotation(self) -> None:
        rot = _FakeRotationPort()
        svc = _make_service(rotation=rot)
        svc.start_capture(
            rotation_speed_dps=10.0, sampling_hz=20.0,
            capture_duration_s=10.0, reference_diameter_mm=150.0,
        )
        svc.stop_capture()
        assert rot.stopped == 1
        assert svc._capturing is False

    def test_stop_capture_cancels_scheduled_tick(self) -> None:
        sched = _FakeSchedulerPort()
        svc = _make_service(scheduler=sched)
        svc.start_capture(
            rotation_speed_dps=10.0, sampling_hz=20.0,
            capture_duration_s=10.0, reference_diameter_mm=150.0,
        )
        svc.stop_capture()
        assert len(sched.cancelled) == 1

    def test_stop_capture_restores_poll_profile(self) -> None:
        poll = _FakePollProfilePort()
        svc = _make_service(poll_profile=poll)
        svc.start_capture(
            rotation_speed_dps=10.0, sampling_hz=20.0,
            capture_duration_s=10.0, reference_diameter_mm=150.0,
        )
        svc.stop_capture()
        assert poll.profiles == ["sampling", "normal"]


# ===================================================================
# Sampling tests
# ===================================================================

class TestSampling:
    """tick behaviour."""

    def test_tick_does_nothing_when_not_capturing(self) -> None:
        svc = _make_service()
        svc._tick()  # should not raise, should not schedule
        assert len(svc._samples) == 0

    def test_tick_records_sample_when_capturing(self) -> None:
        svc = _make_service()
        svc.start_capture(
            rotation_speed_dps=10.0, sampling_hz=20.0,
            capture_duration_s=10.0, reference_diameter_mm=150.0,
        )
        svc._tick()
        assert len(svc._samples) >= 1

    def test_tick_shows_progress_updates(self) -> None:
        sink = _FakeStateSink()
        svc = _make_service(state_sink=sink, rotation=_FakeRotationPort())
        svc.start_capture(
            rotation_speed_dps=10.0, sampling_hz=20.0,
            capture_duration_s=10.0, reference_diameter_mm=150.0,
        )
        for _ in range(5):
            svc._tick()
        assert len(sink.progress) >= 1

    def test_tick_enters_failed_on_sensor_error(self) -> None:
        sensors = _FakeSensorPort()
        sensors._raise_on_read = "cl"
        sink = _FakeStateSink()
        rot = _FakeRotationPort()
        svc = _make_service(sensors=sensors, state_sink=sink, rotation=rot)
        svc.start_capture(
            rotation_speed_dps=10.0, sampling_hz=20.0,
            capture_duration_s=10.0, reference_diameter_mm=150.0,
        )
        svc._tick()
        assert any("failed" in e for e in sink.events)
        assert svc._capturing is False

    def test_tick_stops_rotation_on_error(self) -> None:
        sensors = _FakeSensorPort()
        sensors._raise_on_read = "cl"
        rot = _FakeRotationPort()
        svc = _make_service(sensors=sensors, rotation=rot)
        svc.start_capture(
            rotation_speed_dps=10.0, sampling_hz=20.0,
            capture_duration_s=10.0, reference_diameter_mm=150.0,
        )
        svc._tick()
        assert rot.stopped >= 1

    def test_tick_restores_poll_profile_on_error(self) -> None:
        sensors = _FakeSensorPort()
        sensors._raise_on_read = "cl"
        poll = _FakePollProfilePort()
        svc = _make_service(sensors=sensors, poll_profile=poll)
        svc.start_capture(
            rotation_speed_dps=10.0, sampling_hz=20.0,
            capture_duration_s=10.0, reference_diameter_mm=150.0,
        )
        svc._tick()
        assert poll.profiles[-1] == "normal"


# ===================================================================
# Computation tests
# ===================================================================

class TestComputation:
    """compute_apply behaviour."""

    def test_compute_apply_fails_when_samples_insufficient(self) -> None:
        svc = _make_service()
        svc._samples = [{"theta_deg": 0.0, "out2_mm": 50.0}]
        result = svc._compute_apply(150.0, 10.0, 20.0)
        assert result["ok"] is False

    def test_compute_apply_writes_to_repository_on_success(self) -> None:
        repo = _FakeCalibrationRepo()
        rot = _FakeRotationPort()
        svc = _make_service(repository=repo, rotation=rot)
        svc._samples = [
            {"theta_deg": 0.0, "out2_mm": 50.0},
            {"theta_deg": 120.0, "out2_mm": 52.0},
            {"theta_deg": 240.0, "out2_mm": 48.0},
        ]
        result = svc._compute_apply(150.0, 10.0, 20.0)
        assert result["ok"] is True
        assert len(repo.saved) == 1
