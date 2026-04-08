import unittest

import numpy as np

from core.models import Recipe
from domain.calibration import (
    compute_od_b_candidate,
    fit_id_diameter,
    fit_id_single_from_out2,
    solve_id_delta_candidate,
    verify_id_calibration,
)


class CalibrationDomainTest(unittest.TestCase):
    def test_compute_od_b_candidate_from_sums(self) -> None:
        result = compute_od_b_candidate([10.0, 12.0, 14.0], 180.0)

        self.assertTrue(result.ok)
        self.assertAlmostEqual(float(result.mean_sum), 12.0, places=6)
        self.assertAlmostEqual(float(result.b_candidate), 192.0, places=6)
        self.assertEqual(result.n_used, 3)

    def test_fit_id_single_from_out2_constant_signal(self) -> None:
        recipe = Recipe(id_single_k=1.0, id_single_b=0.0, bin_count=12, bin_method='median', pp_mode='p99_p1')
        theta = list(np.linspace(0.0, 330.0, 12))
        out2 = [75.0] * 12

        result = fit_id_single_from_out2(theta, out2, recipe)

        self.assertTrue(result.ok)
        self.assertAlmostEqual(float(result.mean_L2_decenter), 75.0, places=6)
        self.assertAlmostEqual(float(result.id_est_mm), 75.0, places=6)
        self.assertGreaterEqual(float(result.cov), 1.0)

    def test_fit_id_diameter_constant_chord(self) -> None:
        theta = np.linspace(0.0, 330.0, 12)
        c_mm = np.full(12, 100.0)
        m_mm = np.zeros(12)

        result = fit_id_diameter(theta, c_mm, m_mm, 0.0)

        self.assertAlmostEqual(result.diam, 100.0, places=6)
        self.assertAlmostEqual(result.e, 0.0, places=6)

    def test_solve_id_delta_candidate_matches_reference_diameter(self) -> None:
        theta = np.linspace(0.0, 330.0, 36)
        c_mm = np.full(36, 100.0)
        m_mm = np.zeros(36)

        result = solve_id_delta_candidate(theta, c_mm, m_mm, 120.0)

        self.assertTrue(result.ok)
        self.assertFalse(result.fallback_used)
        self.assertAlmostEqual(float(result.delta_candidate), 20.0, places=4)
        self.assertIsNotNone(result.fit)
        self.assertAlmostEqual(float(result.fit.diam), 120.0, places=4)

    def test_verify_id_calibration_reports_ok_for_full_coverage(self) -> None:
        theta = np.linspace(0.0, 359.0, 36)
        c_mm = np.full(36, 100.0)
        m_mm = np.zeros(36)

        result = verify_id_calibration(theta, c_mm, m_mm, delta_c=0.0, d_ref=100.0)

        self.assertTrue(result.ok)
        self.assertAlmostEqual(float(result.err_mm), 0.0, places=6)
        self.assertGreaterEqual(float(result.cov_pct), 95.0)
        self.assertEqual(result.sample_count, 36)


if __name__ == '__main__':
    unittest.main()
