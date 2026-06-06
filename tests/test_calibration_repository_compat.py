import json
import shutil
from pathlib import Path

from repositories.calibration_repository import CalibrationRepository
import pytest


class TestCalibrationRepositoryCompat:
    def _number(self, value: float | None) -> float:
        assert value is not None
        assert value is not None
        return float(value)

    def _case_root(self, name: str) -> Path:
        root = Path('.test-artifacts') / name
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        return root / 'FRP_IPC'

    def _fixture_dir(self) -> Path:
        return Path(__file__).resolve().parent / 'fixtures' / 'calibration_compat'

    def _install_legacy_samples(self, app_root: Path) -> None:
        calib_dir = app_root / 'calibration'
        calib_dir.mkdir(parents=True, exist_ok=True)
        fixture_dir = self._fixture_dir()
        shutil.copyfile(fixture_dir / 'od_calibration.json', calib_dir / 'od_calibration.json')
        shutil.copyfile(fixture_dir / 'id_calibration.json', calib_dir / 'id_calibration.json')

    def test_load_prefill_from_legacy_od_and_id_samples(self) -> None:
        app_root = self._case_root('calibration_compat_prefill')
        self._install_legacy_samples(app_root)
        repo = CalibrationRepository(app_root_dir=app_root)

        od_prefill = repo.load_od_prefill()
        id_prefill = repo.load_id_prefill()

        assert float(od_prefill['B_active']) == pytest.approx(188.76543)
        assert float(od_prefill['D_ref']) == pytest.approx(180.0)
        assert od_prefill['cmd_used'] == 'M0,1'
        assert od_prefill['out1_map'] == 'R'
        assert od_prefill['angle_src_ui'] == 'AX3'
        assert od_prefill['filter'] == '\u65e0'
        assert od_prefill['outlier_sigma'] == '2.5'
        assert len(od_prefill['defect_template_mask']) == 360
        assert od_prefill['defect_template_mask'][358] == 1
        assert od_prefill['defect_template_mask'][0] == 1
        assert od_prefill['defect_template_mask'][2] == 1
        assert od_prefill['defect_template_mask'][90] == 1
        assert od_prefill['defect_template_mask'][180] == 0

        assert float(id_prefill['delta_c_mm']) == pytest.approx(-0.3456)
        assert float(id_prefill['D_ref']) == pytest.approx(150.0)

    def test_load_snapshot_from_legacy_od_and_id_samples(self) -> None:
        app_root = self._case_root('calibration_compat_snapshot')
        self._install_legacy_samples(app_root)
        repo = CalibrationRepository(app_root_dir=app_root)

        snapshot = repo.load_snapshot()

        assert snapshot.od_b_active_mm == pytest.approx(188.76543)
        assert snapshot.od_out1_map == 'R'
        assert self._number(snapshot.od_d_ref_mm) == pytest.approx(180.0)
        assert snapshot.od_request_cmd == 'M0,1'
        assert snapshot.id_delta_c_mm == pytest.approx(-0.3456)
        assert self._number(snapshot.id_d_ref_mm) == pytest.approx(150.0)
        assert not snapshot.id_single_enabled
        assert snapshot.id_single_k == pytest.approx(1.0)
        assert snapshot.id_single_b_mm == pytest.approx(0.0)
        assert snapshot.id_single_d_ref_mm is None

    def test_fixture_files_keep_legacy_schema_shape(self) -> None:
        fixture_dir = self._fixture_dir()
        od_data = json.loads((fixture_dir / 'od_calibration.json').read_text(encoding='utf-8-sig'))
        id_data = json.loads((fixture_dir / 'id_calibration.json').read_text(encoding='utf-8-sig'))

        assert 'B_active' in od_data
        assert 'D_ref' in od_data
        assert 'cmd_used' in od_data
        assert 'out_map' in od_data
        assert 'params' in od_data
        assert 'delta_c_mm' in id_data
        assert 'D_ref' in id_data


if __name__ == '__main__':
    unittest.main()
