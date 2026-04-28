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
    def _number(self, value: float | None) -> float:
        self.assertIsNotNone(value)
        assert value is not None
        return float(value)

    def test_compute_od_b_candidate_from_sums(self) -> None:
        result = compute_od_b_candidate([10.0, 12.0, 14.0], 180.0)

        self.assertTrue(result.ok)
        self.assertAlmostEqual(self._number(result.mean_sum), 12.0, places=6)
        self.assertAlmostEqual(self._number(result.b_candidate), 192.0, places=6)
        self.assertEqual(result.n_used, 3)

    def test_compute_od_b_candidate_rejects_empty_sums(self) -> None:
        result = compute_od_b_candidate([], 180.0)

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, 'empty_sums')
        self.assertIsNone(result.b_candidate)
        self.assertEqual(result.n_used, 0)

    def test_fit_id_single_from_out2_reports_too_few_bins(self) -> None:
        recipe = Recipe(bin_count=12, bin_method='median', pp_mode='p99_p1')

        result = fit_id_single_from_out2([0.0, 1.0, 2.0], [10.0, 10.0, 10.0], recipe)

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, 'too_few_bins')
        self.assertLess(float(result.cov or 0.0), 0.5)

    def test_fit_id_single_from_out2_constant_signal(self) -> None:
        recipe = Recipe(id_single_k=1.0, id_single_b=0.0, bin_count=12, bin_method='median', pp_mode='p99_p1')
        theta = list(np.linspace(0.0, 330.0, 12))
        out2 = [75.0] * 12

        result = fit_id_single_from_out2(theta, out2, recipe)

        self.assertTrue(result.ok)
        self.assertAlmostEqual(self._number(result.mean_L2_decenter), 75.0, places=6)
        self.assertAlmostEqual(self._number(result.id_est_mm), 75.0, places=6)
        self.assertGreaterEqual(self._number(result.cov), 1.0)

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
        self.assertAlmostEqual(self._number(result.delta_candidate), 20.0, places=4)
        fit = result.fit
        self.assertIsNotNone(fit)
        assert fit is not None
        self.assertAlmostEqual(float(fit.diam), 120.0, places=4)

    def test_solve_id_delta_candidate_falls_back_to_cmax_when_samples_are_few(self) -> None:
        theta = np.linspace(0.0, 90.0, 10)
        c_mm = np.full(10, 100.0)
        m_mm = np.zeros(10)

        result = solve_id_delta_candidate(theta, c_mm, m_mm, 120.0)

        self.assertTrue(result.ok)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.reason, 'fallback_cmax')
        self.assertAlmostEqual(self._number(result.delta_candidate), 20.0, places=6)
        self.assertIsNone(result.fit)

    def test_verify_id_calibration_reports_ok_for_full_coverage(self) -> None:
        theta = np.linspace(0.0, 359.0, 36)
        c_mm = np.full(36, 100.0)
        m_mm = np.zeros(36)

        result = verify_id_calibration(theta, c_mm, m_mm, delta_c=0.0, d_ref=100.0)

        self.assertTrue(result.ok)
        self.assertAlmostEqual(self._number(result.err_mm), 0.0, places=6)
        self.assertGreaterEqual(self._number(result.cov_pct), 95.0)
        self.assertEqual(result.sample_count, 36)

    def test_verify_id_calibration_raises_for_too_few_samples(self) -> None:
        theta = np.linspace(0.0, 100.0, 10)
        c_mm = np.full(10, 100.0)
        m_mm = np.zeros(10)

        with self.assertRaises(ValueError):
            verify_id_calibration(theta, c_mm, m_mm, delta_c=0.0, d_ref=100.0)

    def test_verify_id_calibration_reports_ng_when_error_exceeds_limit(self) -> None:
        theta = np.linspace(0.0, 359.0, 36)
        c_mm = np.full(36, 100.0)
        m_mm = np.zeros(36)

        result = verify_id_calibration(theta, c_mm, m_mm, delta_c=0.0, d_ref=110.0)

        self.assertFalse(result.ok)
        self.assertAlmostEqual(self._number(result.err_mm), -10.0, places=6)
        self.assertGreaterEqual(self._number(result.cov_pct), 95.0)


if __name__ == '__main__':
    unittest.main()
