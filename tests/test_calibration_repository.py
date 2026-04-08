import shutil
import unittest
from pathlib import Path

from repositories.calibration_repository import CalibrationRepository


class CalibrationRepositoryTest(unittest.TestCase):
    def _case_root(self, name: str) -> Path:
        root = Path('tests/.tmp') / name
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        return root / 'FRP_IPC'

    def test_export_od_raw_uses_legacy_export_path(self) -> None:
        repo = CalibrationRepository(app_root_dir=self._case_root('calibration_repo_od'))

        path = repo.export_od_raw([
            {
                'ts': 1.0,
                'theta': 12.0,
                'theta_rel': 3.0,
                'raw': 'M0,1',
                'v1': 10.0,
                'j1': 'GO',
                'v2': 11.0,
                'j2': 'GO',
            }
        ])

        self.assertTrue(path.exists())
        self.assertEqual(path.parent, repo.od_raw_export_dir())
        self.assertEqual(path.parent.name, 'od_calib')
        header = path.read_text(encoding='utf-8').splitlines()[0]
        self.assertEqual(header, 'ts,theta,theta_rel,raw,v1,j1,v2,j2')

    def test_export_id_raw_uses_legacy_calibration_path(self) -> None:
        repo = CalibrationRepository(app_root_dir=self._case_root('calibration_repo_id'))

        path = repo.export_id_raw([
            {
                'ts': 2.0,
                'theta_deg': 45.0,
                'x1_mm': 1.0,
                'x2_mm': 2.0,
                'c_mm': 3.0,
                'm_mm': 4.0,
                'cnt': {'out4': 5, 'out5': 6},
            }
        ])

        self.assertTrue(path.exists())
        self.assertEqual(path.parent, repo.id_raw_export_dir())
        self.assertEqual(path.parent.name, 'calibration')
        header = path.read_text(encoding='utf-8').splitlines()[0]
        self.assertEqual(header, 'ts,theta_deg,x1_mm,x2_mm,c_mm,m_mm,cnt_out4,cnt_out5')


if __name__ == '__main__':
    unittest.main()
