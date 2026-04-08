import csv
import json
import shutil
import unittest
from datetime import datetime
from pathlib import Path

from application.state import CalibrationSnapshot, RunContext, RunIdentity
from core.models import MeasureRow, Recipe
from repositories.run_repository import RunRepository


SECTION_RESULTS_HEADER = [
    'serial', 'run_id',
    'start_time', 'end_time', 'duration_s',
    'section_idx', 'z_pos_mm',
    'od_avg_mm', 'od_dev_mm', 'od_runout_mm', 'od_round_mm', 'od_e_mm', 'od_phi_deg',
    'id_avg_mm', 'id_dev_mm', 'id_runout_mm', 'id_round_mm', 'id_e_mm', 'id_phi_deg',
    'concentricity_mm', 'split_shift_deg', 'coax_unreliable', 'od_ecc_mm', 'id_ecc_mm',
    'cov_pct', 'miss_bin', 'max_gap_deg', 'revs', 'cov_elapsed_s', 'cov_reason',
    'od_round_fit_mm', 'od_round_fit_rob_mm',
    'id_round_fit_mm', 'id_round_fit_rob_mm',
    'od_pp_mm', 'od_pp_rob_mm', 'id_pp_mm', 'id_pp_rob_mm',
    'raw',
]

RAW_POINTS_HEADER = [
    'serial', 'run_id',
    'section_idx', 'z_pos_mm', 'sample_idx',
    'ts', 'theta_deg', 'bin',
    'phase',
    'od_mm', 'id_mm', 'cl_cnt',
    'raw_od', 'raw_id',
]

SUMMARY_HEADER = [
    'date', 'start_time', 'end_time', 'duration_s', 'serial', 'run_id', 'recipe_name', 'device_code',
    'od_std_mm', 'id_std_mm', 'od_tol_mm', 'len_enabled', 'len_skipped', 'len_ok', 'len_mm', 'len_z_low',
    'len_z_high', 'len_reason', 'len_t_s', 'straight_od_mm', 'straight_id_mm', 'axis_dist_mm', 'conc_max_mm',
    'axis_span_max_mm', 'od_tilt_deg', 'od_end_off_mm', 'od_slope_mm_per_mm', 'id_tilt_deg', 'id_end_off_mm',
    'id_slope_mm_per_mm', 'max_od_dev_abs_mm', 'max_id_dev_abs_mm', 'max_od_round_mm', 'max_id_round_mm',
    'max_od_pp_mm', 'max_od_pp_rob_mm', 'max_od_fit_res_mm', 'od_range_mm', 'id_range_mm', 'od_mean_mm',
    'od_d_pp_mm', 'od_e_mm', 'id_mean_mm', 'id_d_pp_mm', 'id_mode', 'id_est_mm', 'id_ecc_amp_mm',
    'id_ecc_ang_deg', 'id_pp_rob_mm', 'split_shift_deg', 'coax_unreliable', 'summary_ok', 'summary_reason',
    'status', 'software_version',
]


class RunRepositoryCompatTest(unittest.TestCase):
    def _case_root(self, name: str) -> Path:
        root = Path('.test-artifacts') / name
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        return root / 'FRP_IPC'

    def _build_context(self) -> RunContext:
        start_ts = datetime(2025, 1, 2, 3, 4, 5).timestamp()
        identity = RunIdentity(
            serial='20250102-compat-001',
            run_id='run-compat-001',
            started_at_ts=start_ts,
        )
        recipe = Recipe(
            name='compat_recipe',
            section_count=1,
            section_pos_z=[100.0],
            od_std_mm=187.3,
            id_std_mm=152.7,
            od_tol_mm=0.15,
            len_enable=True,
            id_single_enable=False,
        )
        row = MeasureRow(
            idx=1,
            x_ui=100.0,
            x_abs=200.0,
            od_avg=187.31,
            od_dev=0.01,
            od_runout=0.02,
            od_round=0.03,
            id_avg=152.70,
            id_dev=0.01,
            id_runout=0.02,
            id_round=0.03,
            concentricity=0.04,
            raw='raw-json-compat',
        )
        summary = {
            'straight_od': 0.12,
            'straight_id': 0.08,
            'axis_dist': 1.20,
            'conc_max': 0.04,
            'axis_span_max': 0.50,
            'max_od_dev_abs': 0.11,
            'max_id_dev_abs': 0.09,
            'max_od_round': 0.03,
            'max_id_round': 0.02,
            'max_od_pp': 0.04,
            'max_od_pp_rob': 0.035,
            'max_od_fit_res': 0.001,
            'od_range': 0.22,
            'id_range': 0.18,
            'od_mean': 187.31,
            'od_d_pp': 0.02,
            'od_e': 0.01,
            'id_mean': 152.70,
            'id_d_pp': 0.03,
            'split_shift_deg': 5.5,
            'coax_unreliable': False,
            'ok': True,
            'reason': 'compat',
        }
        return RunContext(
            identity=identity,
            recipe=recipe,
            calibration=CalibrationSnapshot(),
            rows=[row],
            raw_points=[{
                'section_idx': 1,
                'z_pos_mm': 100.0,
                'sample_idx': 0,
                'ts': 123.456,
                'theta_deg': 0.0,
                'bin': 0,
                'phase': 'od',
                'od_mm': 187.3,
                'id_mm': 152.7,
                'cl_cnt': 1,
                'raw_od': 187.3,
                'raw_id': 152.7,
            }],
            section_coverage={1: {'cov': 0.98, 'miss': 2, 'max_gap_deg': 5.0, 'revs': 1.1, 'elapsed': 1.25, 'reason': 'COV'}},
            length_result={'enabled': True, 'skipped': False, 'ok': True, 'length_mm': 1680.0, 'z_low': 10.0, 'z_high': 1690.0, 'reason': 'OK', 't_s': 1.234},
            summary=summary,
            finished_at_ts=start_ts + 12.345,
            status='DONE',
        )

    def test_export_run_keeps_legacy_csv_and_meta_schema(self) -> None:
        app_root = self._case_root('run_repository_compat_export')
        repo = RunRepository(
            app_root_dir=app_root,
            software_version='compat-test-sw',
            device_code='device-compat-001',
            plc_info={'ip': '192.168.0.10', 'port': 502, 'unit': 1},
            gauge_info={'enabled': True, 'port': 'COM3'},
        )
        context = self._build_context()

        run_dir = Path(repo.export_run(context))
        self.assertEqual(run_dir, app_root / 'exports' / '2025-01-02' / context.identity.serial)

        with open(run_dir / 'section_results.csv', 'r', encoding='utf-8', newline='') as f:
            section_rows = list(csv.reader(f))
        with open(run_dir / 'raw_points.csv', 'r', encoding='utf-8', newline='') as f:
            raw_rows = list(csv.reader(f))
        with open(run_dir.parent / 'summary.csv', 'r', encoding='utf-8-sig', newline='') as f:
            summary_rows = list(csv.reader(f))
        meta = json.loads((run_dir / 'meta.json').read_text(encoding='utf-8'))

        self.assertEqual(section_rows[0], SECTION_RESULTS_HEADER)
        self.assertEqual(raw_rows[0], RAW_POINTS_HEADER)
        self.assertEqual(summary_rows[0], SUMMARY_HEADER)

        self.assertEqual(section_rows[1][0], context.identity.serial)
        self.assertEqual(section_rows[1][1], context.identity.run_id)
        self.assertEqual(raw_rows[1][0], context.identity.serial)
        self.assertEqual(raw_rows[1][1], context.identity.run_id)
        self.assertEqual(summary_rows[1][SUMMARY_HEADER.index('serial')], context.identity.serial)
        self.assertEqual(summary_rows[1][SUMMARY_HEADER.index('run_id')], context.identity.run_id)
        self.assertEqual(summary_rows[1][SUMMARY_HEADER.index('recipe_name')], 'compat_recipe')
        self.assertEqual(summary_rows[1][SUMMARY_HEADER.index('status')], 'DONE')
        self.assertEqual(summary_rows[1][SUMMARY_HEADER.index('software_version')], 'compat-test-sw')

        self.assertEqual(meta['serial'], context.identity.serial)
        self.assertEqual(meta['run_id'], context.identity.run_id)
        self.assertEqual(meta['recipe']['name'], 'compat_recipe')
        self.assertEqual(meta['software_version'], 'compat-test-sw')
        self.assertEqual(meta['plc']['ip'], '192.168.0.10')
        self.assertEqual(meta['gauge']['port'], 'COM3')
        self.assertEqual(Path(meta['exports']['section_results_csv']).name, 'section_results.csv')
        self.assertEqual(Path(meta['exports']['raw_points_csv']).name, 'raw_points.csv')
        self.assertEqual(Path(meta['exports']['meta_json']).name, 'meta.json')

    def test_export_daily_summary_upserts_without_changing_legacy_header(self) -> None:
        app_root = self._case_root('run_repository_compat_summary')
        repo = RunRepository(app_root_dir=app_root, software_version='compat-test-sw', device_code='device-compat-001')
        context = self._build_context()

        repo.export_run(context)
        context.summary['reason'] = 'compat-updated'
        repo.export_daily_summary(context)

        summary_path = app_root / 'exports' / '2025-01-02' / 'summary.csv'
        with open(summary_path, 'r', encoding='utf-8-sig', newline='') as f:
            rows = list(csv.reader(f))

        self.assertEqual(rows[0], SUMMARY_HEADER)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][SUMMARY_HEADER.index('run_id')], context.identity.run_id)
        self.assertEqual(rows[1][SUMMARY_HEADER.index('summary_reason')], 'compat-updated')


if __name__ == '__main__':
    unittest.main()