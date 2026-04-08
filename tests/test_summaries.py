import unittest

from core.models import MeasureRow, Recipe
from domain.summaries import (
    EccentricityUpdate,
    apply_eccentricity_updates,
    compute_postcalc_result,
    compute_run_summary,
    merge_summary_snapshot,
    summary_snapshot_from_payload,
)


class SummariesTest(unittest.TestCase):
    def test_summary_snapshot_and_merge_are_pure(self) -> None:
        seed = {'conc_max': 0.7}
        snapshot = summary_snapshot_from_payload(
            {
                'straightness': 1.2,
                'axis_dist': 0.3,
                'od_tilt_deg': 0.4,
            }
        )

        merged = merge_summary_snapshot(seed, snapshot)

        self.assertEqual(snapshot.straight_od, 1.2)
        self.assertEqual(snapshot.axis_dist, 0.3)
        self.assertEqual(snapshot.od_tilt_deg, 0.4)
        self.assertEqual(seed, {'conc_max': 0.7})
        self.assertEqual(merged['conc_max'], 0.7)
        self.assertEqual(merged['straight_od'], 1.2)
        self.assertEqual(merged['axis_dist'], 0.3)
        self.assertEqual(merged['od_tilt_deg'], 0.4)

    def test_apply_eccentricity_updates_returns_new_rows(self) -> None:
        rows = [
            MeasureRow(
                idx=1,
                x_ui=10.0,
                x_abs=20.0,
                od_avg=100.0,
                od_dev=0.1,
                od_runout=0.2,
                od_round=0.3,
                id_avg=80.0,
                id_dev=0.1,
                id_runout=0.2,
                id_round=0.3,
                concentricity=0.4,
            )
        ]

        updated = apply_eccentricity_updates(
            rows,
            [EccentricityUpdate(row_index=0, od_ecc=0.55, id_ecc=0.66)],
        )

        self.assertIsNone(rows[0].od_ecc)
        self.assertIsNone(rows[0].id_ecc)
        self.assertEqual(updated[0].od_ecc, 0.55)
        self.assertEqual(updated[0].id_ecc, 0.66)
        self.assertIsNot(updated[0], rows[0])

    def test_compute_postcalc_result_returns_straightness_and_postcalc_payloads(self) -> None:
        result = compute_postcalc_result(
            [(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)],
            [(1.0, 0.0, 0.0), (1.0, 0.0, 10.0)],
            concentricity_list=[0.8, 1.2],
            id_single_enable=False,
        )

        self.assertAlmostEqual(result.straightness_payload['straight_od'], 0.0)
        self.assertAlmostEqual(result.straightness_payload['straight_id'], 0.0)
        self.assertAlmostEqual(result.straightness_payload['axis_dist'], 1.0)
        self.assertAlmostEqual(result.straightness_payload['conc_max'], 1.2)
        self.assertAlmostEqual(result.straightness_payload['axis_span_max'], 1.0)
        for value in result.postcalc_payload['ecc_od']:
            self.assertAlmostEqual(value, 0.0, places=9)
        for value in result.postcalc_payload['ecc_id']:
            self.assertAlmostEqual(value, 0.0, places=9)

    def test_compute_run_summary_is_input_only(self) -> None:
        recipe = Recipe(name='summary-test')
        rows = [
            MeasureRow(
                idx=1,
                x_ui=10.0,
                x_abs=20.0,
                od_avg=100.0,
                od_dev=0.2,
                od_runout=0.4,
                od_round=0.5,
                id_avg=80.0,
                id_dev=-0.1,
                id_runout=0.3,
                id_round=0.4,
                concentricity=0.6,
                ok=True,
            )
        ]
        raw_points = [
            {'od_mm': 100.0, 'id_mm': 80.0},
            {'od_mm': 101.0, 'id_mm': 80.5},
        ]
        summary = compute_run_summary(
            recipe=recipe,
            rows=rows,
            raw_points=raw_points,
            summary_cache={'straight_od': 1.1, 'axis_dist': 0.7},
        )

        self.assertTrue(summary['ok'])
        self.assertEqual(summary['max_od_dev_abs'], 0.2)
        self.assertEqual(summary['max_id_dev_abs'], 0.1)
        self.assertEqual(summary['conc_max'], 0.6)
        self.assertEqual(summary['straight_od'], 1.1)
        self.assertEqual(summary['axis_dist'], 0.7)
        self.assertEqual(summary['od_range'], 1.0)
        self.assertEqual(summary['id_range'], 0.5)
        self.assertEqual(summary['judge_ok_cnt'], 1)
        self.assertEqual(summary['judge_total'], 1)


if __name__ == '__main__':
    unittest.main()
