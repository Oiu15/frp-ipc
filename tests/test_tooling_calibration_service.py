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
    # OD support samples (single-edge OUT1); no stored reference profile
    _seed_od_samples(svc, R=95.0)
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


def _seed_od_samples(svc, R=95.0, center=(0.5, -0.3), offset=0.0, n=360):
    from tests.test_geometry_v2 import _make_od_support

    th, h = _make_od_support(R, center=center, lobes={3: 0.01}, n=n)
    svc._samples = [{"theta_deg": float(np.rad2deg(t)), "od_out1": float(hv - offset)}
                    for t, hv in zip(th, h)]
    return th, h


def test_compute_and_apply_od_zero(tmp_path):
    svc, repo = _service(tmp_path)
    _seed_od_samples(svc, R=95.0, offset=0.6)  # uncalibrated zero offset
    out = svc.compute_od_zero(known_od=190.0)
    assert out["ok"] and abs(out["od_b"] - 0.6) < 0.05
    assert svc.apply_od_zero()["ok"]
    tc = ToolingCalibration.from_dict(repo.load_tooling_active())
    assert tc.od_calibrated() and abs(tc.od_b - 0.6) < 0.05


def test_capture_reference_then_psi(tmp_path):
    svc, repo = _service(tmp_path)
    # store reference profile from master at zero orientation
    from tests.test_geometry_v2 import _od_support_with_phase

    th0, h0 = _od_support_with_phase(95.0, 0.05, 3, 0.0)
    svc._samples = [{"theta_deg": float(np.rad2deg(t)), "od_out1": float(hv)} for t, hv in zip(th0, h0)]
    assert svc.capture_od_reference()["ok"]
    assert "od_ref_phi" in repo.load_tooling_active().get("meta", {})
    # remounted, rotated
    thp, hp = _od_support_with_phase(95.0, 0.05, 3, 18.0)
    svc._samples = [{"theta_deg": float(np.rad2deg(t)), "od_out1": float(hv)} for t, hv in zip(thp, hp)]
    out = svc.compute_od_psi()
    assert out["ok"] and out["has_reference"]
    resid = ((out["psi_deg"] - 18.0 + 60.0) % 120.0) - 60.0
    assert abs(resid) < 8.0


def test_axis_and_chuck(tmp_path):
    svc, repo = _service(tmp_path)
    # low station center ~ (0.0, 0.0) at z=0
    _seed_od_samples(svc, center=(0.0, 0.0))
    assert svc.record_axis_station(z=0.0)["ok"]
    # high station center shifted at z=1700 -> known slope
    _seed_od_samples(svc, center=(0.34, -0.17))
    assert svc.record_axis_station(z=1700.0)["ok"]
    ax = svc.compute_axis()
    assert ax["ok"] and abs(ax["axis_slope_x"] - 0.34 / 1700.0) < 1e-4
    assert svc.apply_axis()["ok"]
    assert ToolingCalibration.from_dict(repo.load_tooling_active()).axis_calibrated()

    # chuck bound: measured roundness vs a tiny cert -> positive bound
    _seed_od_samples(svc, center=(0.0, 0.0))
    ch = svc.compute_chuck_bound(cert_roundness=0.0)
    assert ch["ok"] and ch["chuck_error_bound"] >= 0.0


def test_delta_reg_needs_both_centers(tmp_path):
    pytest.importorskip("scipy")
    svc, repo = _service(tmp_path)
    # OD-only -> fails (no ID center)
    _seed_od_samples(svc)
    assert svc.compute_delta_reg()["ok"] is False


class _FakeView:
    def __init__(self) -> None:
        self.vals: dict[str, Any] = {}

    def get_value(self, name: str, default: Any = None) -> Any:
        return self.vals.get(name, default)

    def set_value(self, name: str, value: Any) -> None:
        self.vals[name] = value

    def get_float(self, name: str, default: float) -> float:
        try:
            return float(self.vals.get(name, default))
        except Exception:
            return default


def test_calibration_controller_tcal_methods_route_to_service(tmp_path):
    pytest.importorskip("scipy")
    from modes.mode_machine import ModeMachine
    from services.calibration_controller import CalibrationController

    svc, repo = _service(tmp_path)
    view = _FakeView()
    view.vals["tcal_r_known_var"] = "76.35"
    view.vals["tcal_d_init_var"] = "140.0"
    ctrl = CalibrationController(
        mode_machine=cast(ModeMachine, cast(Any, object())),
        view=cast(Any, view),
        tooling_service=svc,
    )

    truth = id_tooling_from_simple(D=140.0, s=0.35, axis_deg=8.0, q=(1.5, 0.7))
    for e in ([1.0, 0.5], [-1.3, 0.9], [0.4, -1.6]):
        th, x1, x2 = _dataset_arrays(truth, e)
        svc.add_dataset_arrays(th, x1, x2)

    ctrl.fit_tcal_id_pose()
    assert view.vals.get("tcal_id_s_var", "").startswith(("+0.3", "+0.34", "+0.35", "+0.36"))
    ctrl.apply_tcal_id_pose()
    assert ToolingCalibration.from_dict(repo.load_tooling_active()).id_calibrated()
    ctrl.run_tcal_selftest()
    assert "通过" in view.vals.get("tcal_selftest_var", "")


def test_calibration_controller_od_zero_and_axis_route(tmp_path):
    from modes.mode_machine import ModeMachine
    from services.calibration_controller import CalibrationController

    svc, repo = _service(tmp_path)
    view = _FakeView()
    view.vals.update({"tcal_known_od_var": "190.0", "tcal_axis_z_low_var": "0.0",
                      "tcal_axis_z_high_var": "1700.0", "tcal_cert_round_var": "0.0"})
    ctrl = CalibrationController(
        mode_machine=cast(ModeMachine, cast(Any, object())),
        view=cast(Any, view),
        tooling_service=svc,
    )

    _seed_od_samples(svc, R=95.0, offset=0.6)
    ctrl.compute_tcal_od_zero()
    assert view.vals.get("tcal_od_b_var", "--") != "--"
    ctrl.apply_tcal_od_zero()
    assert ToolingCalibration.from_dict(repo.load_tooling_active()).od_calibrated()

    _seed_od_samples(svc, center=(0.0, 0.0))
    ctrl.record_tcal_axis_low()
    _seed_od_samples(svc, center=(0.34, -0.17))
    ctrl.record_tcal_axis_high()
    ctrl.compute_tcal_axis()
    ctrl.apply_tcal_axis()
    assert ToolingCalibration.from_dict(repo.load_tooling_active()).axis_calibrated()
