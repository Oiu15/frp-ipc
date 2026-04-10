import json
import shutil
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from application.state import (
    CalibrationSnapshot,
    FixedSectionRepeatabilitySession,
    RuntimeState,
    ValidationSession,
)
from core.models import MeasureRow, Recipe
from repositories.run_repository import RunRepository
from repositories.validation_repository import ValidationRepository
from frp_workflow.validation_workflow import (
    FixedSectionRepeatabilityRequest,
    ValidationPhase,
    ValidationWorkflow,
    ValidationWorkflowEventType,
)


class FakeGateway:
    def __getattr__(self, name: str):
        raise AssertionError(f'unexpected gateway call: {name}')


class RecordingValidationActionGateway(FakeGateway):
    def __init__(self) -> None:
        self.actions: list[str | tuple[str, float]] = []

    def stop_rotation(self) -> None:
        self.actions.append('stop_rotation')

    def clamp_release(self) -> None:
        self.actions.append('clamp_release')

    def clamp_close(self) -> None:
        self.actions.append('clamp_close')

    def wait_cancelable(self, duration_s: float, **_kwargs) -> None:
        self.actions.append(('wait_cancelable', float(duration_s)))


class ValidationWorkflowSmokeTest(unittest.TestCase):
    def test_fixed_section_reclamp_request_and_state_fields(self) -> None:
        default_request = FixedSectionRepeatabilityRequest()
        self.assertFalse(default_request.reclamp_enabled)
        self.assertFalse(default_request.rotation_stop_before_measure)
        self.assertEqual(default_request.release_settle_s, 0.0)
        self.assertEqual(default_request.clamp_settle_s, 0.0)

        request = FixedSectionRepeatabilityRequest(
            reclamp_enabled=True,
            rotation_stop_before_measure=True,
            release_settle_s=0.25,
            clamp_settle_s=0.5,
        )
        self.assertTrue(request.reclamp_enabled)
        self.assertTrue(request.rotation_stop_before_measure)
        self.assertEqual(request.release_settle_s, 0.25)
        self.assertEqual(request.clamp_settle_s, 0.5)

        session = FixedSectionRepeatabilitySession(
            reclamp_enabled=True,
            rotation_stop_before_measure=True,
            release_settle_s=0.25,
            clamp_settle_s=0.5,
        )
        self.assertTrue(session.reclamp_enabled)
        self.assertTrue(session.rotation_stop_before_measure)
        self.assertEqual(session.release_settle_s, 0.25)
        self.assertEqual(session.clamp_settle_s, 0.5)

    def test_fixed_section_before_capture_runs_configured_reclamp_actions(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tmp_root = repo_root / '.compile_check' / 'validation_before_capture_reclamp'
        tmp_root.mkdir(parents=True, exist_ok=True)
        case_root = tmp_root / f'case_{int(time.time() * 1000)}'
        self.addCleanup(shutil.rmtree, case_root, True)
        app_root = case_root / 'FRP_IPC'
        app_root.mkdir(parents=True, exist_ok=True)

        gateway = RecordingValidationActionGateway()
        session = ValidationSession()
        workflow = ValidationWorkflow(
            recipe=Recipe(name='validation-before-capture'),
            calibration=CalibrationSnapshot(),
            runtime_state=RuntimeState.from_validation_session(session),
            gateway=gateway,
            run_repository=RunRepository(app_root_dir=app_root),
            validation_session=session,
        )
        request = FixedSectionRepeatabilityRequest(
            section_name='S1',
            metric_name='od_avg',
            repeat_count=1,
            reclamp_enabled=True,
            rotation_stop_before_measure=True,
            release_settle_s=0.25,
            clamp_settle_s=0.5,
        )
        section_result = MeasureRow(
            idx=1,
            x_ui=0.0,
            x_abs=0.0,
            od_avg=100.0,
            od_dev=0.0,
            od_runout=0.0,
            od_round=0.0,
            id_avg=80.0,
            id_dev=0.0,
            id_runout=0.0,
            id_round=0.0,
            concentricity=0.0,
        )
        raw_points = [
            {
                'phase': 'SYNC',
                'ts': float(i),
                'theta_deg': float(i * 10),
                'bin': i,
                'od_mm': 100.0,
            }
            for i in range(6)
        ]
        windows = [
            {
                'window_index': 1,
                'window_role': 'SYNC',
                'point_count': len(raw_points),
                'theta_span_deg': 50.0,
            }
        ]

        def _capture_side_effect(**_kwargs):
            gateway.actions.append('capture')
            return section_result, raw_points, windows, {'cov': 1.0}

        with patch(
            'frp_workflow.validation_workflow.measure_current_position_section_capture',
            side_effect=_capture_side_effect,
        ) as capture_mock:
            rows, summary = workflow.run_fixed_section_repeatability(request)

        capture_mock.assert_called_once()
        self.assertEqual(len(rows), 1)
        self.assertEqual(summary['count'], 1)
        self.assertEqual(
            gateway.actions,
            [
                'stop_rotation',
                'clamp_release',
                ('wait_cancelable', 0.25),
                'clamp_close',
                ('wait_cancelable', 0.5),
                'capture',
            ],
        )

    def test_smoke_events_result_and_export(self) -> None:
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

        export_repo = ValidationRepository(app_root_dir=app_root)
        export_ctx = workflow.build_export_context()
        run_dir = Path(export_repo.export_run(export_ctx))

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

        self.assertEqual(run_dir.parent.parent.name, 'validation_exports')
        self.assertTrue((run_dir / 'validation_result.json').exists())
        self.assertTrue((run_dir / 'validation_events.json').exists())
        self.assertTrue((run_dir.parent / 'summary.csv').exists())
        self.assertFalse((app_root / 'exports').exists())

        payload = json.loads((run_dir / 'validation_result.json').read_text(encoding='utf-8'))
        self.assertEqual(payload['serial'], identity.serial)
        self.assertEqual(payload['standard_piece_id'], 'STD-RING-001')
        self.assertEqual(payload['validation_batch_id'], 'VAL-20260408-A')
        self.assertEqual(payload['repeat_measurement_count'], 3)

    def test_fixed_section_repeatability_smoke_runs_sampling_chain(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tmp_root = repo_root / '.compile_check' / 'validation_fixed_section_smoke'
        tmp_root.mkdir(parents=True, exist_ok=True)
        case_root = tmp_root / f'case_{int(time.time() * 1000)}'
        self.addCleanup(shutil.rmtree, case_root, True)
        app_root = case_root / 'FRP_IPC'
        app_root.mkdir(parents=True, exist_ok=True)

        session = ValidationSession()
        workflow = ValidationWorkflow(
            recipe=Recipe(name='validation-fixed-section-smoke'),
            calibration=CalibrationSnapshot(),
            runtime_state=RuntimeState.from_validation_session(session),
            gateway=FakeGateway(),
            run_repository=RunRepository(app_root_dir=app_root),
            validation_session=session,
        )
        request = FixedSectionRepeatabilityRequest(
            section_name='S1',
            metric_name='od_avg',
            repeat_count=1,
        )
        section_result = MeasureRow(
            idx=1,
            x_ui=12.0,
            x_abs=34.0,
            od_avg=123.456,
            od_dev=0.156,
            od_runout=0.01,
            od_round=0.02,
            id_avg=80.0,
            id_dev=0.0,
            id_runout=0.0,
            id_round=0.0,
            concentricity=0.0,
        )
        raw_points = [
            {
                'phase': 'SYNC',
                'ts': float(i),
                'theta_deg': float(i * 10),
                'bin': i,
                'od_mm': 123.456,
            }
            for i in range(6)
        ]
        windows = [
            {
                'window_index': 1,
                'window_role': 'SYNC',
                'point_start_index': 0,
                'point_end_index': 5,
                'point_count': len(raw_points),
                'ts_start': 0.0,
                'ts_end': 5.0,
                'theta_start_deg': 0.0,
                'theta_end_deg': 50.0,
                'theta_span_deg': 50.0,
                'filled_bins': 6,
                'total_bins': 6,
                'miss_bins': 0,
                'n_od': 6,
                'n_id': 0,
                'reason': 'COV',
                'revs': 0.2,
                'elapsed_s': 0.5,
                'max_gap_deg': 10.0,
            }
        ]

        with patch(
            'frp_workflow.validation_workflow.measure_current_position_section_capture',
            return_value=(section_result, raw_points, windows, {'cov': 1.0}),
        ) as capture_mock:
            rows, summary = workflow.run_fixed_section_repeatability(request)

        capture_mock.assert_called_once()
        capture_kwargs = capture_mock.call_args.kwargs
        self.assertIs(capture_kwargs['gateway'], workflow.gateway)
        self.assertIs(capture_kwargs['recipe'], workflow.recipe)
        self.assertIs(capture_kwargs['calibration'], workflow.calibration)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].repeat_index, 1)
        self.assertEqual(rows[0].section_name, 'S1')
        self.assertEqual(rows[0].metric_name, 'od_avg')
        self.assertEqual(rows[0].measured_value_mm, 123.456)
        self.assertEqual(summary['count'], 1)
        self.assertEqual(summary['primary_metric']['od_avg']['count'], 1)
        self.assertEqual(summary['primary_metric']['od_avg']['mean'], 123.456)
        self.assertEqual(workflow.runtime_state.status, 'completed')
        self.assertEqual(workflow.result.status, 'DONE')
        self.assertEqual(session.repeat_measurement_count, 1)
        self.assertEqual(len(workflow.runtime_state.rows), 1)
        self.assertEqual(len(workflow.runtime_state.raw_points), 6)
        self.assertEqual(len(workflow.fixed_section_repeat_captures), 1)
        capture = workflow.fixed_section_repeat_captures[0]
        self.assertEqual(capture.section_result, section_result)
        self.assertEqual(len(capture.raw_points), 6)
        self.assertEqual(len(capture.windows), 1)
        self.assertEqual(capture.windows[0].point_count, 6)
        self.assertEqual(capture.coverage['cov'], 1.0)

        export_context = workflow.build_export_context()
        self.assertEqual(export_context.status, 'DONE')
        self.assertEqual(export_context.repeat_measurement_count, 1)
        self.assertEqual(export_context.summary['count'], 1)

    def test_fixed_section_repeatability_records_phase_sequence(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tmp_root = repo_root / '.compile_check' / 'validation_workflow_phase'
        tmp_root.mkdir(parents=True, exist_ok=True)
        case_root = tmp_root / f'case_{int(time.time() * 1000)}'
        self.addCleanup(shutil.rmtree, case_root, True)
        app_root = case_root / 'FRP_IPC'
        app_root.mkdir(parents=True, exist_ok=True)

        session = ValidationSession()
        workflow = ValidationWorkflow(
            recipe=Recipe(name='validation-phase'),
            calibration=CalibrationSnapshot(),
            runtime_state=RuntimeState.from_validation_session(session),
            gateway=FakeGateway(),
            run_repository=RunRepository(app_root_dir=app_root),
            validation_session=session,
        )
        request = FixedSectionRepeatabilityRequest(
            section_name='S1',
            metric_name='od_avg',
            repeat_count=2,
        )
        section_result = MeasureRow(
            idx=1,
            x_ui=0.0,
            x_abs=0.0,
            od_avg=100.0,
            od_dev=0.0,
            od_runout=0.0,
            od_round=0.0,
            id_avg=80.0,
            id_dev=0.0,
            id_runout=0.0,
            id_round=0.0,
            concentricity=0.0,
        )
        raw_points = [
            {
                'phase': 'SYNC',
                'ts': float(i),
                'theta_deg': float(i * 10),
                'bin': i,
                'od_mm': 100.0,
            }
            for i in range(6)
        ]
        windows = [
            {
                'window_index': 1,
                'window_role': 'SYNC',
                'point_count': len(raw_points),
                'theta_span_deg': 50.0,
            }
        ]

        progress_seen: list[tuple[int, int]] = []
        phase_seen: list[tuple[str, int, int]] = []
        with patch(
            'frp_workflow.validation_workflow.measure_current_position_section_capture',
            return_value=(section_result, raw_points, windows, {'cov': 1.0}),
        ) as capture_mock:
            rows, summary = workflow.run_fixed_section_repeatability(
                request,
                progress_callback=lambda index, total: progress_seen.append((index, total)),
                phase_callback=lambda event: phase_seen.append((event.phase, event.repeat_index, event.total)),
            )

        self.assertEqual(capture_mock.call_count, 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary['count'], 2)
        self.assertEqual(progress_seen, [(1, 2), (2, 2)])
        self.assertEqual(workflow.current_phase, ValidationPhase.SAVE_RESULT)

        phase_events = [
            event
            for event in workflow.events
            if event.type == ValidationWorkflowEventType.PHASE
        ]
        self.assertEqual(
            [event.phase for event in phase_events],
            [
                ValidationPhase.PREPARE.value,
                ValidationPhase.BEFORE_CAPTURE.value,
                ValidationPhase.CAPTURE.value,
                ValidationPhase.FIT_CALC.value,
                ValidationPhase.SAVE_RESULT.value,
                ValidationPhase.PREPARE.value,
                ValidationPhase.BEFORE_CAPTURE.value,
                ValidationPhase.CAPTURE.value,
                ValidationPhase.FIT_CALC.value,
                ValidationPhase.SAVE_RESULT.value,
            ],
        )
        self.assertEqual(
            [event.repeat_index for event in phase_events],
            [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        )
        self.assertEqual(
            phase_seen,
            [(event.phase, event.repeat_index, event.total) for event in phase_events],
        )


if __name__ == '__main__':
    unittest.main()
