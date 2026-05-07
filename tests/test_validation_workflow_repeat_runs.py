import unittest
from pathlib import Path

from application.state import CalibrationSnapshot, RuntimeState, ValidationSession
from core.models import Recipe
from tests.fakes import RecordingValidationRepository, SequentialRunRepository, StrictDeviceGateway
from frp_workflow.validation_workflow import ValidationWorkflow


class ValidationWorkflowRepeatRunsTest(unittest.TestCase):
    def test_repeat_validation_runs_with_fake_gateway_and_repositories(self) -> None:
        run_repo = SequentialRunRepository()
        export_repo = RecordingValidationRepository(Path('/virtual/app_root'))
        exported_serials: list[str] = []

        for repeat_idx in range(1, 4):
            session = ValidationSession(
                standard_piece_id='STD-RING-001',
                validation_batch_id='VAL-BATCH-042',
                repeat_measurement_count=repeat_idx,
            )
            workflow = ValidationWorkflow(
                recipe=Recipe(name='validation-repeat'),
                calibration=CalibrationSnapshot(),
                runtime_state=RuntimeState.from_validation_session(session),
                gateway=StrictDeviceGateway(),
                run_repository=run_repo,
                validation_session=session,
            )

            identity = workflow.ensure_identity()
            workflow.record_state('PREP', f'prepare #{repeat_idx}')
            workflow.record_progress(step='baseline', index=repeat_idx, total=3, message='collecting')
            workflow.record_summary({'baseline_ok': True, 'repeat_idx': repeat_idx}, source='baseline')
            workflow.record_state('DONE', f'completed #{repeat_idx}')
            result = workflow.build_result(status='DONE', message=f'completed #{repeat_idx}', finished_at_ts=identity.started_at_ts + 5.0)
            export_ctx = workflow.build_export_context()
            run_dir = Path(export_repo.export_run(export_ctx))

            exported_serials.append(identity.serial)
            self.assertEqual(workflow.runtime_state.status, 'completed')
            self.assertEqual(result.status, 'DONE')
            self.assertEqual(result.identity, identity)
            self.assertEqual(result.standard_piece_id, 'STD-RING-001')
            self.assertEqual(result.validation_batch_id, 'VAL-BATCH-042')
            self.assertEqual(result.repeat_measurement_count, repeat_idx)
            self.assertEqual(session.summary_cache['repeat_idx'], repeat_idx)
            self.assertEqual(run_dir.parts[-3], 'validation_exports')
            self.assertEqual(run_dir.parts[-1], identity.serial)
            self.assertNotIn('exports', run_dir.parts[:-3])

        self.assertEqual(len(run_repo.prepared), 3)
        self.assertEqual(len(export_repo.exported_run_paths), 3)
        self.assertEqual(len(export_repo.exported_summary_paths), 3)
        self.assertEqual(export_repo.exported_statuses, ['DONE', 'DONE', 'DONE'])
        self.assertEqual(len(set(exported_serials)), 3)
        self.assertEqual(exported_serials, [
            '20260408-validation-001',
            '20260408-validation-002',
            '20260408-validation-003',
        ])
        for path in export_repo.exported_run_paths:
            self.assertIn('validation_exports', path.parts)
            self.assertNotIn('exports', path.parts[:-3])
        for path in export_repo.exported_summary_paths:
            self.assertEqual(path.name, 'summary.csv')
            self.assertIn('validation_exports', path.parts)


if __name__ == '__main__':
    unittest.main()
