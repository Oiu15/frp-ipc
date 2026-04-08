import shutil
import time
import unittest
from pathlib import Path

from application.state import CalibrationSnapshot, RuntimeState
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

        workflow = ValidationWorkflow(
            recipe=Recipe(name='validation-smoke'),
            calibration=CalibrationSnapshot(),
            runtime_state=RuntimeState(),
            gateway=FakeGateway(),
            run_repository=RunRepository(app_root_dir=app_root),
        )

        identity = workflow.ensure_identity()
        workflow.record_state('PREP', 'prepare validation')
        workflow.record_progress(step='acquire_baseline', index=1, total=2, message='collecting')
        workflow.record_summary({'baseline_ok': True, 'delta_mm': 0.012}, source='baseline')
        workflow.record_state('DONE', 'completed')
        result = workflow.build_result(status='DONE', message='completed', finished_at_ts=time.time())

        self.assertEqual(identity.serial, workflow.runtime_state.serial)
        self.assertEqual(workflow.runtime_state.status, 'completed')
        self.assertEqual(result.status, 'DONE')
        self.assertEqual(result.identity, identity)
        self.assertTrue(result.finished_at_ts is not None)
        self.assertEqual(result.summary['baseline_ok'], True)
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
