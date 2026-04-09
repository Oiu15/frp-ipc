import datetime as dt
import unittest
from pathlib import Path

from application.state import CalibrationSnapshot, RuntimeState, ValidationSession
from core.models import Recipe
from frp_workflow.validation_workflow import ValidationWorkflow


class FakeGateway:
    def __getattr__(self, name: str):
        raise AssertionError(f'unexpected gateway call: {name}')


class FakeRunRepository:
    def __init__(self) -> None:
        self._seq = 0
        self.prepared: list[tuple[str, str]] = []

    def prepare_run(self, recipe_name: str):
        from application.state import RunIdentity

        self._seq += 1
        serial = f'20260408-validation-{self._seq:03d}'
        run_id = f'run-{self._seq:03d}'
        started_at_ts = 1775606400.0 + float(self._seq)
        self.prepared.append((recipe_name, serial))
        return RunIdentity(serial=serial, run_id=run_id, started_at_ts=started_at_ts)


class FakeValidationRepository:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.exported_run_paths: list[Path] = []
        self.exported_summary_paths: list[Path] = []
        self.exported_statuses: list[str] = []

    def export_run(self, context):
        start_ts = float(context.started_at_ts if context.started_at_ts is not None else context.identity.started_at_ts)
        day_tag = dt.date.fromtimestamp(start_ts).strftime('%Y-%m-%d')
        run_dir = self.root / 'validation_exports' / day_tag / str(context.identity.serial)
        self.exported_run_paths.append(run_dir)
        self.exported_statuses.append(str(context.status))
        self.export_daily_summary(context)
        return str(run_dir)

    def export_daily_summary(self, context):
        start_ts = float(context.started_at_ts if context.started_at_ts is not None else context.identity.started_at_ts)
        day_tag = dt.date.fromtimestamp(start_ts).strftime('%Y-%m-%d')
        summary_path = self.root / 'validation_exports' / day_tag / 'summary.csv'
        self.exported_summary_paths.append(summary_path)


class ValidationWorkflowRepeatRunsTest(unittest.TestCase):
    def test_repeat_validation_runs_with_fake_gateway_and_repositories(self) -> None:
        run_repo = FakeRunRepository()
        export_repo = FakeValidationRepository(Path('/virtual/app_root'))
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
                gateway=FakeGateway(),
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
