import json
import shutil
import sys
import time
import types
import unittest
from csv import DictReader
from pathlib import Path

_pymodbus = types.ModuleType("pymodbus")
_pymodbus_client = types.ModuleType("pymodbus.client")


class _FakeModbusTcpClient:
    pass


_pymodbus_client.ModbusTcpClient = _FakeModbusTcpClient
_pymodbus.client = _pymodbus_client
sys.modules.setdefault("pymodbus", _pymodbus)
sys.modules.setdefault("pymodbus.client", _pymodbus_client)

from application.state import CalibrationSnapshot, ValidationExportContext, RunIdentity, ValidationFitResult
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
            recipe=Recipe(
                name='validation-fixture',
                section_sampling_mode='split',
                sampling_window_mode='separate_channels',
                scan_mode='split',
            ),
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
        self.assertEqual(payload['recipe']['section_sampling_mode'], 'split')
        self.assertEqual(payload['recipe']['sampling_window_mode'], 'separate_channels')

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
            section_name='1: 10.000',
            metric_name='od_avg',
            measured_value_mm=100.123,
            settle_s_used=0.2,
            sample_delay_s_used=0.1,
            capture_start_ts=10.0,
            capture_end_ts=12.5,
            measured_at_ts=13.0,
            measure_section_index=1,
            measure_section_name='1: 10.000',
            measured_z_pos_mm=10.0,
        )
        capture = FixedSectionRepeatCapture(
            repeat_index=1,
            section_name='1: 10.000',
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
            raw_points=(
                {
                    'theta_deg': 0.0,
                    'od_mm': 100.123,
                },
            ),
            coverage={'cov': 1.0},
            measure_section_index=1,
            measure_section_name='1: 10.000',
            measured_z_pos_mm=10.0,
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

        self.assertTrue((run_dir / 'validation_result.json').exists())
        self.assertTrue((run_dir / 'repeat_results.csv').exists())
        self.assertTrue((run_dir / 'repeat_rows.csv').exists())
        self.assertTrue((run_dir / 'repeat_section_results.csv').exists())
        self.assertTrue((run_dir / 'repeat_windows.csv').exists())
        self.assertTrue((run_dir / 'repeat_summary.json').exists())

        with open(run_dir / 'repeat_results.csv', 'r', encoding='utf-8-sig', newline='') as f:
            canonical_rows_reader = list(DictReader(f))
        with open(run_dir / 'repeat_rows.csv', 'r', encoding='utf-8-sig', newline='') as f:
            rows_reader = list(DictReader(f))
        meta = json.loads((run_dir / 'validation_meta.json').read_text(encoding='utf-8'))
        result = json.loads((run_dir / 'validation_result.json').read_text(encoding='utf-8'))
        self.assertEqual(meta['requested_section_name'], 'S1')
        self.assertEqual(meta['section_name'], '1: 10.000')
        self.assertEqual(meta['measure_section_index'], 1)
        self.assertEqual(meta['measure_section_name'], '1: 10.000')
        self.assertEqual(meta['measured_z_pos_mm'], 10.0)
        self.assertEqual(meta['schema_version'], 'validation_fixed_section_v1')
        self.assertEqual(meta['request']['metric_name'], 'od_avg')
        self.assertIn('repeat_results.csv', meta['canonical_outputs'])
        self.assertIn('repeat_rows.csv', meta['legacy_outputs'])
        self.assertIn('repeat_rows_csv', meta['exports'])
        self.assertIn('repeat_results_csv', meta['exports'])
        self.assertEqual(canonical_rows_reader, rows_reader)
        self.assertEqual(result['schema_version'], 'validation_fixed_section_v1')
        self.assertEqual(result['validation_kind'], 'fixed_section_repeatability')
        self.assertEqual(result['config']['request']['metric_name'], 'od_avg')
        self.assertIn('validation_result.json', result['canonical_outputs'])
        self.assertIn('repeat_results.csv', result['canonical_outputs'])
        self.assertIn('repeat_rows.csv', result['legacy_outputs'])
        self.assertIn('repeat_section_results.csv', result['legacy_outputs'])
        self.assertIn('repeat_windows.csv', result['legacy_outputs'])
        self.assertIn('repeat_summary.json', result['legacy_outputs'])
        self.assertEqual(result['final_summary']['measure_section_name'], '1: 10.000')
        self.assertEqual(rows_reader[0]['section_name'], '1: 10.000')
        self.assertEqual(rows_reader[0]['measure_section_index'], '1')
        self.assertEqual(rows_reader[0]['measure_section_name'], '1: 10.000')
        self.assertEqual(rows_reader[0]['measured_z_pos_mm'], '10.000')
        self.assertEqual(rows_reader[0]['settle_s_used'], '0.200')
        self.assertEqual(rows_reader[0]['sample_delay_s_used'], '0.100')
        self.assertEqual(rows_reader[0]['capture_start_ts'], '10.000000')
        self.assertEqual(rows_reader[0]['capture_end_ts'], '12.500000')

        with open(run_dir / 'repeat_section_results.csv', 'r', encoding='utf-8-sig', newline='') as f:
            results_reader = list(DictReader(f))
        self.assertEqual(results_reader[0]['section_name'], '1: 10.000')
        self.assertEqual(results_reader[0]['measure_section_index'], '1')
        self.assertEqual(results_reader[0]['measure_section_name'], '1: 10.000')
        self.assertEqual(results_reader[0]['measured_z_pos_mm'], '10.000')
        self.assertEqual(results_reader[0]['settle_s_used'], '0.200')
        self.assertEqual(results_reader[0]['sample_delay_s_used'], '0.100')
        self.assertEqual(results_reader[0]['capture_start_ts'], '10.000000')
        self.assertEqual(results_reader[0]['capture_end_ts'], '12.500000')

        with open(run_dir / 'repeat_raw_points.csv', 'r', encoding='utf-8-sig', newline='') as f:
            raw_reader = list(DictReader(f))
        self.assertEqual(raw_reader[0]['section_name'], '1: 10.000')
        self.assertEqual(raw_reader[0]['measure_section_index'], '1')
        self.assertEqual(raw_reader[0]['measure_section_name'], '1: 10.000')
        self.assertEqual(raw_reader[0]['measured_z_pos_mm'], '10.0')

    def test_fixed_section_export_writes_repeat_fit_results_with_stable_schema(self) -> None:
        app_root = self._case_root('validation_repository_fit_results')
        repo = ValidationRepository(app_root_dir=app_root, software_version='test-build')
        identity = RunIdentity(serial='20260415-validation-fit-001', run_id='run-fit-001', started_at_ts=time.time())
        context = ValidationExportContext(
            identity=identity,
            recipe=Recipe(name='validation-fit-fixture'),
            calibration=CalibrationSnapshot(),
            validation_batch_id='VAL-BATCH-FIT',
            summary={'count': 2},
            events=[],
            started_at_ts=identity.started_at_ts,
            finished_at_ts=identity.started_at_ts + 2.0,
            status='DONE',
            message='completed',
        )
        request = FixedSectionRepeatabilityRequest(
            section_name='S2',
            metric_name='concentricity',
            repeat_count=2,
        )
        rows = [
            FixedSectionRepeatRow(
                repeat_index=1,
                section_name='2: 20.000',
                metric_name='concentricity',
                measured_value_mm=0.321,
                settle_s_used=0.2,
                sample_delay_s_used=0.1,
                capture_start_ts=10.0,
                capture_end_ts=12.0,
                measured_at_ts=12.5,
                measure_section_index=2,
                measure_section_name='2: 20.000',
                measured_z_pos_mm=20.0,
            ),
            FixedSectionRepeatRow(
                repeat_index=2,
                section_name='current: 12.500',
                metric_name='concentricity',
                measured_value_mm=0.0,
                settle_s_used=0.2,
                sample_delay_s_used=0.1,
                capture_start_ts=20.0,
                capture_end_ts=21.0,
                measured_at_ts=21.5,
                measure_section_index=None,
                measure_section_name='current: 12.500',
                measured_z_pos_mm=12.5,
            ),
        ]
        base_section_result = MeasureRow(
            idx=1,
            x_ui=20.0,
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
        captures = [
            FixedSectionRepeatCapture(
                repeat_index=1,
                section_name='2: 20.000',
                metric_name='concentricity',
                measured_at_ts=12.5,
                measured_value_mm=0.321,
                settle_s_used=0.2,
                sample_delay_s_used=0.1,
                capture_start_ts=10.0,
                capture_end_ts=12.0,
                section_result=base_section_result,
                windows=(),
                raw_points=(),
                coverage={},
                measure_section_index=2,
                measure_section_name='2: 20.000',
                measured_z_pos_mm=20.0,
                fit_result=ValidationFitResult(
                    measure_section_index=2,
                    measure_section_name='2: 20.000',
                    measured_z_pos_mm=20.0,
                    od_center_x_mm=0.12,
                    od_center_y_mm=-0.34,
                    od_radius_mm=61.728,
                    od_diameter_fit_mm=123.456,
                    id_center_x_mm=0.02,
                    id_center_y_mm=-0.03,
                    id_radius_mm=None,
                    id_diameter_fit_mm=80.0,
                    od_ecc_mm=None,
                    id_ecc_mm=0.111,
                    concentricity_mm=0.321,
                ),
            ),
            FixedSectionRepeatCapture(
                repeat_index=2,
                section_name='current: 12.500',
                metric_name='concentricity',
                measured_at_ts=21.5,
                measured_value_mm=0.0,
                settle_s_used=0.2,
                sample_delay_s_used=0.1,
                capture_start_ts=20.0,
                capture_end_ts=21.0,
                section_result=base_section_result,
                windows=(),
                raw_points=(),
                coverage={},
                measure_section_index=None,
                measure_section_name='current: 12.500',
                measured_z_pos_mm=12.5,
                fit_result=None,
            ),
        ]

        run_dir = Path(
            repo.export_fixed_section_repeatability(
                context=context,
                request=request,
                rows=rows,
                summary={'count': 2},
                captures=captures,
            )
        )

        fit_path = run_dir / 'repeat_fit_results.csv'
        self.assertTrue(fit_path.exists())
        with open(fit_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = DictReader(f)
            fit_field_names = list(reader.fieldnames or [])
            fit_reader = list(reader)
        self.assertEqual(
            fit_field_names,
            [
                'repeat_index',
                'measure_section_index',
                'measure_section_name',
                'measured_z_pos_mm',
                'od_center_x_mm',
                'od_center_y_mm',
                'od_radius_mm',
                'od_diameter_fit_mm',
                'id_center_x_mm',
                'id_center_y_mm',
                'id_radius_mm',
                'id_diameter_fit_mm',
                'od_ecc_mm',
                'id_ecc_mm',
                'concentricity_mm',
            ],
        )
        self.assertEqual(len(fit_reader), 2)
        self.assertEqual(fit_reader[0]['repeat_index'], '1')
        self.assertEqual(fit_reader[0]['measure_section_index'], '2')
        self.assertEqual(fit_reader[0]['measure_section_name'], '2: 20.000')
        self.assertEqual(fit_reader[0]['measured_z_pos_mm'], '20.000')
        self.assertEqual(fit_reader[0]['od_center_x_mm'], '0.120000')
        self.assertEqual(fit_reader[0]['od_center_y_mm'], '-0.340000')
        self.assertEqual(fit_reader[0]['od_radius_mm'], '61.728000')
        self.assertEqual(fit_reader[0]['od_diameter_fit_mm'], '123.456000')
        self.assertEqual(fit_reader[0]['id_center_x_mm'], '0.020000')
        self.assertEqual(fit_reader[0]['id_center_y_mm'], '-0.030000')
        self.assertEqual(fit_reader[0]['id_radius_mm'], '')
        self.assertEqual(fit_reader[0]['id_diameter_fit_mm'], '80.000000')
        self.assertEqual(fit_reader[0]['od_ecc_mm'], '')
        self.assertEqual(fit_reader[0]['id_ecc_mm'], '0.111000')
        self.assertEqual(fit_reader[0]['concentricity_mm'], '0.321000')
        self.assertEqual(fit_reader[1]['repeat_index'], '2')
        self.assertEqual(fit_reader[1]['measure_section_index'], '')
        self.assertEqual(fit_reader[1]['measure_section_name'], 'current: 12.500')
        self.assertEqual(fit_reader[1]['measured_z_pos_mm'], '12.500')
        self.assertEqual(fit_reader[1]['od_center_x_mm'], '')
        self.assertEqual(fit_reader[1]['od_center_y_mm'], '')
        self.assertEqual(fit_reader[1]['od_radius_mm'], '')
        self.assertEqual(fit_reader[1]['od_diameter_fit_mm'], '')
        self.assertEqual(fit_reader[1]['id_center_x_mm'], '')
        self.assertEqual(fit_reader[1]['id_center_y_mm'], '')
        self.assertEqual(fit_reader[1]['id_radius_mm'], '')
        self.assertEqual(fit_reader[1]['id_diameter_fit_mm'], '')
        self.assertEqual(fit_reader[1]['od_ecc_mm'], '')
        self.assertEqual(fit_reader[1]['id_ecc_mm'], '')
        self.assertEqual(fit_reader[1]['concentricity_mm'], '')

        meta = json.loads((run_dir / 'validation_meta.json').read_text(encoding='utf-8'))
        self.assertEqual(meta['exports']['repeat_fit_results_csv'], str(fit_path))


if __name__ == '__main__':
    unittest.main()
