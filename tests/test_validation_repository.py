import json
import shutil
import time
import unittest
from pathlib import Path

from application.state import CalibrationSnapshot, ValidationExportContext, RunIdentity
from core.models import Recipe
from repositories.validation_repository import ValidationRepository


class ValidationRepositoryTest(unittest.TestCase):
    def _case_root(self, name: str) -> Path:
        root = Path('.test-artifacts') / name
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        return root / 'FRP_IPC'

    def test_export_uses_separate_validation_root_and_schema(self) -> None:
        app_root = self._case_root('validation_repository_export')
        repo = ValidationRepository(app_root_dir=app_root, software_version='test-build')
        identity = RunIdentity(serial='20260408-validation-001', run_id='run-123', started_at_ts=time.time())
        context = ValidationExportContext(
            identity=identity,
            recipe=Recipe(name='validation-fixture'),
            calibration=CalibrationSnapshot(),
            standard_piece_id='STD-RING-001',
            validation_batch_id='VAL-BATCH-001',
            repeat_measurement_count=5,
            summary={'baseline_ok': True, 'delta_mm': 0.01},
            events=[{'type': 'state', 'state': 'DONE'}],
            started_at_ts=identity.started_at_ts,
            finished_at_ts=identity.started_at_ts + 12.5,
            status='DONE',
            message='completed',
        )

        run_dir = Path(repo.export_run(context))

        self.assertEqual(run_dir.parent.parent.name, 'validation_exports')
        self.assertTrue((run_dir / 'validation_result.json').exists())
        self.assertTrue((run_dir / 'validation_events.json').exists())
        self.assertTrue((run_dir.parent / 'summary.csv').exists())
        self.assertFalse((app_root / 'exports').exists())

        payload = json.loads((run_dir / 'validation_result.json').read_text(encoding='utf-8'))
        self.assertEqual(payload['standard_piece_id'], 'STD-RING-001')
        self.assertEqual(payload['validation_batch_id'], 'VAL-BATCH-001')
        self.assertEqual(payload['repeat_measurement_count'], 5)
        self.assertEqual(payload['status'], 'DONE')
        self.assertIn('summary', payload)
        self.assertIn('calibration', payload)
        self.assertNotIn('section_results_csv', json.dumps(payload, ensure_ascii=False))


if __name__ == '__main__':
    unittest.main()
