from __future__ import annotations

"""批次5b:ToolingCalibrationService — ID 位姿恢复 + 落盘 + OD ψ + 自检。"""

from typing import Any, cast

import numpy as np
import pytest

from domain.geometry_calibration import (
    ToolingCalibration,
    id_predict_L,
    id_tooling_from_simple,
)
from repositories.calibration_repository import CalibrationRepository
from services.tooling_calibration import ToolingCalibrationService


class _NoOpPorts:
    """Satisfies rotation/sensors/scheduler/state_sink/poll_profile no-op-wise.

    Compute/apply paths don't run the tick loop, so these are inert here.
    """

    def start_rotation(self, *a: Any, **k: Any) -> None: ...
    def stop_rotation(self, *a: Any, **k: Any) -> None: ...
    def read_axis_angle_deg(self) -> float: return 0.0
    def read_cl_out145_cached(self) -> Any: return None
    def request_gauge_sample(self) -> Any: return None
    def set_gauge_command(self, cmd: str) -> None: ...
    def schedule_once(self, delay_ms: int, cb: Any) -> object: return object()
    def cancel(self, handle: object) -> None: ...
    def begin_capture(self) -> None: ...
    def end_capture(self) -> None: ...
    def capture_failed(self, msg: str) -> None: ...
    def publish_progress(self, p: Any) -> None: ...
    def publish_od_progress(self, p: Any) -> None: ...
    def publish_od_sample(self, *a: Any, **k: Any) -> None: ...
    def publish_id_progress(self, p: Any) -> None: ...
    def publish_id_single_progress(self, p: Any) -> None: ...
    def publish_id_verify_result(self, r: Any) -> None: ...
    def use_poll_profile(self, profile: Any) -> None: ...


def _service(tmp_path):
    repo = CalibrationRepository(app_root_dir=tmp_path)
    fake = cast(Any, _NoOpPorts())
    svc = ToolingCalibrationService(
        rotation=fake, sensors=fake, scheduler=fake, state_sink=fake,
        poll_profile=fake, repository=repo,
    )
    return svc, repo


def _dataset_arrays(truth, e, n=200, r=76.35):
    th = np.linspace(0, 360, n, endpoint=False)
    x1 = np.array([id_predict_L(truth.probe_a, np.array(e), np.deg2rad(t), r) for t in th])
    x2 = np.array([id_predict_L(truth.probe_b, np.array(e), np.deg2rad(t), r) for t in th])
    return th, x1, x2


def test_id_pose_compute_and_apply(tmp_path):
    pytest.importorskip("scipy")
    svc, repo = _service(tmp_path)
    truth = id_tooling_from_simple(D=140.0, s=0.35, axis_deg=8.0, q=(1.5, 0.7))
    for e in ([1.0, 0.5], [-1.3, 0.9], [0.4, -1.6]):
        th, x1, x2 = _dataset_arrays(truth, e)
        assert svc.add_dataset_arrays(th, x1, x2)["ok"]
    assert svc.dataset_count() == 3

    cand = svc.compute_id_pose(r_known=76.35, d_init=140.0)
    assert cand["ok"]
    assert abs(cand["s_lateral"] - 0.35) < 0.02

    applied = svc.apply_id_pose()
    assert applied["ok"]
    tc = ToolingCalibration.from_dict(repo.load_tooling_active())
    assert tc.id_calibrated()
    assert tc.id_D_eff == pytest.approx(140.0)
    assert abs(tc.id_s_lateral - 0.35) < 0.02


def test_id_pose_compute_without_data_fails(tmp_path):
    svc, _ = _service(tmp_path)
    assert svc.compute_id_pose(r_known=76.35, d_init=140.0)["ok"] is False
    assert svc.apply_id_pose()["ok"] is False


def test_od_psi_compute_without_reference_returns_zero(tmp_path):
    svc, repo = _service(tmp_path)
    # inject OD support samples directly
    svc._samples = [{"theta_deg": float(t), "h": 95.0} for t in np.linspace(0, 360, 60, endpoint=False)]
    out = svc.compute_od_psi()
    assert out["ok"] and out["psi_deg"] == 0.0 and out["has_reference"] is False
    assert svc.apply_od_psi()["ok"]
    assert ToolingCalibration.from_dict(repo.load_tooling_active()).od_psi_deg == 0.0


def test_run_selftest_and_clear_all(tmp_path):
    pytest.importorskip("scipy")
    svc, repo = _service(tmp_path)
    assert svc.run_selftest()["ok"] is True
    repo.save_tooling_active(ToolingCalibration(id_D_eff=140.0).to_dict())
    assert svc.clear_all()["ok"]
    assert ToolingCalibration.from_dict(repo.load_tooling_active()).id_calibrated() is False
