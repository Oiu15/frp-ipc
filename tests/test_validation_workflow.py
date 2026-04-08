import shutil
import time
import unittest
from pathlib import Path

from application.state import CalibrationSnapshot, RuntimeState, ValidationSession
from core.models import Recipe
from repositories.run_repository import RunRepository
from workflow.validation_workflow import ValidationWorkflow, ValidationWorkflowEventType


class FakeGateway:
    def __getattr__(self, name: str):
        raise AssertionError(f'unexpected gateway call: {name}')


class ValidationWorkflowSmokeTest(unittest.TestCase):
    def test_smoke_events_and_result(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tmp_root = repo_root / '.compile_check' / 'validation_workflow_smoke'
        tmp_root.mkdir(parents=True, exist_ok=True)
        case_root = tmp_root / f'case_{int(time.time() * 1000)}'
        self.addCleanup(shutil.rmtree, case_root, True)
        app_root = case_root / 'FRP_IPC'
        app_root.mkdir(parents=True, exist_ok=True)

        session = ValidationSession(
            standard_piece_id='STD-RING-001',
            validation_batch_id='VAL-20260408-A',
            repeat_measurement_count=3,
        )
        workflow = ValidationWorkflow(
            recipe=Recipe(name='validation-smoke'),
            calibration=CalibrationSnapshot(),
            runtime_state=RuntimeState.from_validation_session(session),
            gateway=FakeGateway(),
            run_repository=RunRepository(app_root_dir=app_root),
            validation_session=session,
        )

        identity = workflow.ensure_identity()
        workflow.record_state('PREP', 'prepare validation')
        workflow.record_progress(step='acquire_baseline', index=1, total=2, message='collecting')
        workflow.record_summary({'baseline_ok': True, 'delta_mm': 0.012}, source='baseline')
        workflow.record_state('DONE', 'completed')
        result = workflow.build_result(status='DONE', message='completed', finished_at_ts=time.time())

        self.assertEqual(identity.serial, workflow.runtime_state.serial)
        self.assertEqual(identity.serial, session.serial)
        self.assertEqual(identity.run_id, session.run_id)
        self.assertEqual(workflow.runtime_state.status, 'completed')
        self.assertEqual(result.status, 'DONE')
        self.assertEqual(result.identity, identity)
        self.assertTrue(result.finished_at_ts is not None)
        self.assertEqual(result.standard_piece_id, 'STD-RING-001')
        self.assertEqual(result.validation_batch_id, 'VAL-20260408-A')
        self.assertEqual(result.repeat_measurement_count, 3)
        self.assertEqual(result.summary['baseline_ok'], True)
        self.assertEqual(session.summary_cache['baseline_ok'], True)
        self.assertEqual(session.summary_cache['delta_mm'], 0.012)
        self.assertEqual(
            [event.type for event in workflow.events],
            [
                ValidationWorkflowEventType.STATE,
                ValidationWorkflowEventType.PROGRESS,
                ValidationWorkflowEventType.SUMMARY,
                ValidationWorkflowEventType.STATE,
            ],
        )


if __name__ == '__main__':
    unittest.main()
