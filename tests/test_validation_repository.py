import json
import shutil
import time
import unittest
from csv import DictReader
from pathlib import Path

from application.state import CalibrationSnapshot, ValidationExportContext, RunIdentity
from core.models import MeasureRow, Recipe
from repositories.validation_repository import ValidationRepository
from frp_workflow.validation_workflow import (
    FixedSectionRepeatabilityRequest,
    FixedSectionRepeatCapture,
    FixedSectionRepeatRow,
    FixedSectionWindow,
)


class ValidationRepositoryTest(unittest.TestCase):
    def _case_root(self, name: str) -> Path:
        root = Path(__file__).resolve().parents[1] / '.test-artifacts' / name
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

    def test_fixed_section_export_includes_motion_request_and_events(self) -> None:
        app_root = self._case_root('validation_repository_fixed_section_export')
        repo = ValidationRepository(app_root_dir=app_root, software_version='test-build')
        identity = RunIdentity(serial='20260408-validation-fixed-001', run_id='run-fixed-123', started_at_ts=time.time())
        context = ValidationExportContext(
            identity=identity,
            recipe=Recipe(name='validation-fixture', section_count=3, section_pos_z=[0.0, 20.0, 40.0]),
            calibration=CalibrationSnapshot(),
            standard_piece_id=None,
            validation_batch_id='VAL-BATCH-SECTION',
            repeat_measurement_count=0,
            summary={},
            events=[{
                'type': 'phase',
                'phase': 'move_to_target_section',
                'payload': {
                    'from_section_index': 1,
                    'target_section_index': 2,
                    'return_section_index': 3,
                    'z_pos_mm': 20.0,
                    'planned_targets_mm': {'AX0': -20.0, 'AX1': -10.0, 'AX4': -10.0},
                    'actual_positions_after_wait_mm': {'AX0': -20.0, 'AX1': -10.0, 'AX4': -10.0},
                },
            }],
            started_at_ts=identity.started_at_ts,
            finished_at_ts=identity.started_at_ts + 1.0,
            status='DONE',
            message='completed',
        )
        request = FixedSectionRepeatabilityRequest(
            move_enabled=True,
            move_channel='od_id_sync',
            move_scenario='switch_and_return',
            move_from_section_index=1,
            move_target_section_index=2,
            move_return_section_index=3,
        )

        run_dir = Path(repo.export_fixed_section_repeatability(
            context=context,
            request=request,
            rows=[],
            summary={},
            captures=[],
        ))

        meta = json.loads((run_dir / 'validation_meta.json').read_text(encoding='utf-8'))
        events = json.loads((run_dir / 'validation_events.json').read_text(encoding='utf-8'))
        self.assertEqual(meta['move_channel'], 'od_id_sync')
        self.assertEqual(meta['move_scenario'], 'switch_and_return')
        self.assertEqual(meta['move_from_section_index'], 1)
        self.assertEqual(meta['move_target_section_index'], 2)
        self.assertEqual(meta['move_return_section_index'], 3)
        self.assertEqual(events[0]['payload']['planned_targets_mm']['AX0'], -20.0)
        self.assertEqual(events[0]['payload']['actual_positions_after_wait_mm']['AX4'], -10.0)

    def test_fixed_section_export_includes_per_repeat_window_timing_fields(self) -> None:
        app_root = self._case_root('validation_repository_fixed_section_fields')
        repo = ValidationRepository(app_root_dir=app_root, software_version='test-build')
        identity = RunIdentity(serial='20260411-validation-001', run_id='run-456', started_at_ts=time.time())
        context = ValidationExportContext(
            identity=identity,
            recipe=Recipe(name='validation-fixture'),
            calibration=CalibrationSnapshot(),
            standard_piece_id='STD-RING-002',
            validation_batch_id='VAL-BATCH-002',
            repeat_measurement_count=1,
            summary={'count': 1},
            events=[],
            started_at_ts=identity.started_at_ts,
            finished_at_ts=identity.started_at_ts + 1.0,
            status='DONE',
            message='completed',
        )
        request = FixedSectionRepeatabilityRequest(
            section_name='S1',
            metric_name='od_avg',
            position_settle_s=0.2,
            sample_delay_s=0.1,
        )
        row = FixedSectionRepeatRow(
            repeat_index=1,
            section_name='S1',
            metric_name='od_avg',
            measured_value_mm=100.123,
            settle_s_used=0.2,
            sample_delay_s_used=0.1,
            capture_start_ts=10.0,
            capture_end_ts=12.5,
            measured_at_ts=13.0,
        )
        capture = FixedSectionRepeatCapture(
            repeat_index=1,
            section_name='S1',
            metric_name='od_avg',
            measured_at_ts=13.0,
            measured_value_mm=100.123,
            settle_s_used=0.2,
            sample_delay_s_used=0.1,
            capture_start_ts=10.0,
            capture_end_ts=12.5,
            section_result=MeasureRow(
                idx=1,
                x_ui=0.0,
                x_abs=0.0,
                od_avg=100.123,
                od_dev=0.0,
                od_runout=0.0,
                od_round=0.0,
                id_avg=80.0,
                id_dev=0.0,
                id_runout=0.0,
                id_round=0.0,
                concentricity=0.0,
            ),
            windows=(
                FixedSectionWindow(
                    repeat_index=1,
                    window_index=1,
                    window_role='SYNC',
                    point_start_index=0,
                    point_end_index=5,
                    point_count=6,
                    ts_start=10.0,
                    ts_end=12.5,
                    theta_start_deg=0.0,
                    theta_end_deg=50.0,
                    theta_span_deg=50.0,
                    filled_bins=6,
                    total_bins=6,
                    miss_bins=0,
                    n_od=6,
                    n_id=0,
                    reason='COV',
                    revs=0.2,
                    elapsed_s=2.5,
                    max_gap_deg=10.0,
                ),
            ),
            raw_points=(),
            coverage={'cov': 1.0},
        )

        run_dir = Path(
            repo.export_fixed_section_repeatability(
                context=context,
                request=request,
                rows=[row],
                summary={'count': 1},
                captures=[capture],
            )
        )

        with open(run_dir / 'repeat_rows.csv', 'r', encoding='utf-8-sig', newline='') as f:
            rows_reader = list(DictReader(f))
        self.assertEqual(rows_reader[0]['settle_s_used'], '0.200')
        self.assertEqual(rows_reader[0]['sample_delay_s_used'], '0.100')
        self.assertEqual(rows_reader[0]['capture_start_ts'], '10.000000')
        self.assertEqual(rows_reader[0]['capture_end_ts'], '12.500000')

        with open(run_dir / 'repeat_section_results.csv', 'r', encoding='utf-8-sig', newline='') as f:
            results_reader = list(DictReader(f))
        self.assertEqual(results_reader[0]['settle_s_used'], '0.200')
        self.assertEqual(results_reader[0]['sample_delay_s_used'], '0.100')
        self.assertEqual(results_reader[0]['capture_start_ts'], '10.000000')
        self.assertEqual(results_reader[0]['capture_end_ts'], '12.500000')


if __name__ == '__main__':
    unittest.main()
