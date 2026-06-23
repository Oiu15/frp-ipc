from __future__ import annotations

"""批次2:tooling 持久化 + algo_version round-trip(纯追加,不动旧 schema)。"""

from domain.geometry_calibration import ToolingCalibration
from repositories.calibration_repository import CalibrationRepository
from repositories.run_repository import RunRepository
from core.models import Recipe


def test_recipe_algo_version_default_is_legacy():
    assert Recipe().algo_version == "legacy"


def test_tooling_active_round_trip(tmp_path):
    repo = CalibrationRepository(app_root_dir=tmp_path)
    tc = ToolingCalibration(
        od_k0=1.001, od_b=0.5, od_psi_deg=1.2,
        id_D_eff=140.0, id_s_lateral=0.35, id_axis_deg=8.0, id_qx=1.5, id_qy=0.7,
        delta_reg=(0.05, -0.02), ref_coaxiality_observed=(-0.01, 0.005),
        ref_od=190.0, ref_id=152.7, chuck_error_bound=0.02,
    )
    repo.save_tooling_active(tc.to_dict())

    assert repo.tooling_calibration_file().exists()
    assert repo.tooling_history_file().exists()  # history appended
    assert "tooling" in repo.active_paths()
    assert "tooling" in repo.history_paths()

    restored = ToolingCalibration.from_dict(repo.load_tooling_active())
    assert restored.id_D_eff == 140.0
    assert restored.id_s_lateral == 0.35
    assert restored.delta_reg == (0.05, -0.02)
    assert restored.ref_coaxiality_observed == (-0.01, 0.005)


def test_load_snapshot_tooling_absent_is_none(tmp_path):
    repo = CalibrationRepository(app_root_dir=tmp_path)
    snap = repo.load_snapshot()
    assert snap.tooling is None
    # 旧字段不受影响,仍返回默认
    assert snap.od_b_active_mm == 0.0
    assert snap.id_delta_c_mm == 0.0


def test_load_snapshot_tooling_present(tmp_path):
    repo = CalibrationRepository(app_root_dir=tmp_path)
    repo.save_tooling_active(
        ToolingCalibration(id_D_eff=152.7, id_s_lateral=0.3).to_dict()
    )
    snap = repo.load_snapshot()
    assert snap.tooling is not None
    assert snap.tooling.id_D_eff == 152.7
    assert snap.tooling.id_s_lateral == 0.3


def test_save_tooling_does_not_touch_od_id_files(tmp_path):
    repo = CalibrationRepository(app_root_dir=tmp_path)
    repo.save_od_active({"B_active": 1.23, "D_ref": 190.0})
    repo.save_id_active({"delta_c_mm": 0.5, "D_ref": 152.7})
    repo.save_tooling_active(ToolingCalibration(id_D_eff=140.0).to_dict())
    # 旧文件未被 tooling 写入污染
    assert repo.load_od_active()["B_active"] == 1.23
    assert repo.load_id_active()["delta_c_mm"] == 0.5


def test_recipe_dump_dict_includes_algo_version(tmp_path):
    repo = RunRepository(app_root_dir=tmp_path)
    dump = repo._recipe_dump_dict(Recipe(algo_version="geometry_v2"))
    assert dump["algo_version"] == "geometry_v2"
    # 默认 legacy 也能 dump
    assert repo._recipe_dump_dict(Recipe())["algo_version"] == "legacy"
