import json
import shutil
import unittest
from pathlib import Path

from repositories.calibration_repository import CalibrationRepository


class CalibrationRepositoryCompatTest(unittest.TestCase):
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

        self.assertAlmostEqual(float(od_prefill['B_active']), 188.76543, places=6)
        self.assertAlmostEqual(float(od_prefill['D_ref']), 180.0, places=6)
        self.assertEqual(od_prefill['cmd_used'], 'M0,1')
        self.assertEqual(od_prefill['out1_map'], 'R')
        self.assertEqual(od_prefill['angle_src_ui'], 'AX3')
        self.assertEqual(od_prefill['filter'], '\u65e0')
        self.assertEqual(od_prefill['outlier_sigma'], '2.5')
        self.assertEqual(len(od_prefill['defect_template_mask']), 360)
        self.assertEqual(od_prefill['defect_template_mask'][358], 1)
        self.assertEqual(od_prefill['defect_template_mask'][0], 1)
        self.assertEqual(od_prefill['defect_template_mask'][2], 1)
        self.assertEqual(od_prefill['defect_template_mask'][90], 1)
        self.assertEqual(od_prefill['defect_template_mask'][180], 0)

        self.assertAlmostEqual(float(id_prefill['delta_c_mm']), -0.3456, places=6)
        self.assertAlmostEqual(float(id_prefill['D_ref']), 150.0, places=6)

    def test_load_snapshot_from_legacy_od_and_id_samples(self) -> None:
        app_root = self._case_root('calibration_compat_snapshot')
        self._install_legacy_samples(app_root)
        repo = CalibrationRepository(app_root_dir=app_root)

        snapshot = repo.load_snapshot()

        self.assertAlmostEqual(snapshot.od_b_active_mm, 188.76543, places=6)
        self.assertEqual(snapshot.od_out1_map, 'R')
        self.assertAlmostEqual(float(snapshot.od_d_ref_mm), 180.0, places=6)
        self.assertEqual(snapshot.od_request_cmd, 'M0,1')
        self.assertAlmostEqual(snapshot.id_delta_c_mm, -0.3456, places=6)
        self.assertAlmostEqual(float(snapshot.id_d_ref_mm), 150.0, places=6)
        self.assertFalse(snapshot.id_single_enabled)
        self.assertAlmostEqual(snapshot.id_single_k, 1.0, places=6)
        self.assertAlmostEqual(snapshot.id_single_b_mm, 0.0, places=6)
        self.assertIsNone(snapshot.id_single_d_ref_mm)

    def test_fixture_files_keep_legacy_schema_shape(self) -> None:
        fixture_dir = self._fixture_dir()
        od_data = json.loads((fixture_dir / 'od_calibration.json').read_text(encoding='utf-8-sig'))
        id_data = json.loads((fixture_dir / 'id_calibration.json').read_text(encoding='utf-8-sig'))

        self.assertIn('B_active', od_data)
        self.assertIn('D_ref', od_data)
        self.assertIn('cmd_used', od_data)
        self.assertIn('out_map', od_data)
        self.assertIn('params', od_data)
        self.assertIn('delta_c_mm', id_data)
        self.assertIn('D_ref', id_data)


if __name__ == '__main__':
    unittest.main()
