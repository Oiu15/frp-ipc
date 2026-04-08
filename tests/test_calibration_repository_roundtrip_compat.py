import json
import shutil
import unittest
from pathlib import Path

from repositories.calibration_repository import CalibrationRepository


class CalibrationRepositoryRoundtripCompatTest(unittest.TestCase):
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

    def test_save_active_roundtrip_preserves_core_legacy_fields(self) -> None:
        app_root = self._case_root('calibration_repository_roundtrip')
        self._install_legacy_samples(app_root)
        repo = CalibrationRepository(app_root_dir=app_root)

        od_active = repo.load_od_active()
        id_active = repo.load_id_active()
        repo.save_od_active(od_active)
        repo.save_id_active(id_active)

        od_saved = json.loads((app_root / 'calibration' / 'od_calibration.json').read_text(encoding='utf-8'))
        id_saved = json.loads((app_root / 'calibration' / 'id_calibration.json').read_text(encoding='utf-8'))

        for key in ('B_active', 'D_ref', 'cmd_used', 'out_map', 'params'):
            self.assertIn(key, od_saved)
        for key in ('delta_c_mm', 'D_ref'):
            self.assertIn(key, id_saved)

        self.assertTrue((app_root / 'calibration' / 'od_calibration_history.jsonl').exists())
        self.assertTrue((app_root / 'calibration' / 'id_calibration_history.jsonl').exists())
        self.assertAlmostEqual(float(od_saved['B_active']), 188.76543, places=6)
        self.assertAlmostEqual(float(id_saved['delta_c_mm']), -0.3456, places=6)


if __name__ == '__main__':
    unittest.main()