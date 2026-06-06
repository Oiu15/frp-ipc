from __future__ import annotations

import numpy as np
import pytest

from services.calibration_service import CalibrationService
from core.models import Recipe


class TestCalibrationService:
    def test_fit_id_diameter_constant_chord(self) -> None:
        service = CalibrationService()
        theta = np.linspace(0.0, 330.0, 12)
        c_mm = np.full(12, 100.0)
        m_mm = np.zeros(12)

        result = service.fit_id_diameter(theta, c_mm, m_mm, 0.0)

        assert result["diam"] == pytest.approx(100.0)
        assert result["e"] == pytest.approx(0.0)

    def test_calc_id_single_from_out2_constant_signal(self) -> None:
        service = CalibrationService()
        recipe = Recipe(id_single_k=1.0, id_single_b=0.0, bin_count=12, bin_method="median", pp_mode="p99_p1")
        theta = list(np.linspace(0.0, 330.0, 12))
        out2 = [75.0] * 12

        result = service.calc_id_single_from_out2(theta, out2, recipe)

        assert result["ok"] is True
        assert float(result["mean_L2_decenter"]) == pytest.approx(75.0)
        assert float(result["id_est_mm"]) == pytest.approx(75.0)
        assert float(result["cov"]) >= 1.0
