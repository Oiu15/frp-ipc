from __future__ import annotations

from typing import Any

import numpy as np

from tests.fakes import FakeVar

from application.app_host import AppHost


class _FakeCalibrationRepository:
    def __init__(self, prefill: dict[str, Any] | None = None) -> None:
        self.prefill = dict(prefill or {})
        self.saved: list[dict[str, Any]] = []

    def load_od_prefill(self) -> dict[str, Any]:
        return dict(self.prefill)

    def save_od_active(self, data: dict[str, Any]) -> None:
        self.saved.append(dict(data))


class _FakeOdCalHost:
    _odcal_update_rev_progress = AppHost._odcal_update_rev_progress
    _odcal_rev_done = AppHost._odcal_rev_done
    _odcal_deg_from_point = AppHost._odcal_deg_from_point
    _odcal_bins_median = AppHost._odcal_bins_median
    _odcal_fit_harmonics = AppHost._odcal_fit_harmonics
    _odcal_residual_bins = AppHost._odcal_residual_bins
    _odcal_mask_to_ranges = AppHost._odcal_mask_to_ranges
    _odcal_ranges_to_mask = AppHost._odcal_ranges_to_mask
    _odcal_ranges_str = AppHost._odcal_ranges_str
    _odcal_shift_mask = AppHost._odcal_shift_mask
    _odcal_detect_defect_mask = AppHost._odcal_detect_defect_mask
    _odcal_best_shift_template = AppHost._odcal_best_shift_template
    _odcal_best_shift_by_overlap = AppHost._odcal_best_shift_by_overlap
    _odcal_prepare_sums = AppHost._odcal_prepare_sums
    _odcal_update_stats = AppHost._odcal_update_stats
    _odcal_save_active = AppHost._odcal_save_active
    _odcal_load_active = AppHost._odcal_load_active

    def __init__(self) -> None:
        self._odcal_theta_start = None
        self._odcal_theta_last = None
        self._odcal_theta_unwrap = 0.0
        self._odcal_rev_progress_deg = 0.0
        self._odcal_rev_target_deg = 360.0
        self._odcal_points: list[dict[str, Any]] = []
        self._odcal_drop_cnt = 0
        self._odcal_defect_template_mask = [0] * 360
        self._odcal_defect_learn_A_data = None
        self.odcal_map_out1_var = FakeVar("L")
        self.odcal_filter_var = FakeVar("无")
        self.odcal_outlier_sigma_var = FakeVar("0")
        self.odcal_defect_dyn_enable_var = FakeVar(0)
        self.odcal_defect_mode_var = FakeVar("OFF")
        self.odcal_defect_shift_var = FakeVar("--")
        self.odcal_defects_var = FakeVar("--")
        self.odcal_sum_mean_var = FakeVar("--")
        self.odcal_sum_std_var = FakeVar("--")
        self.odcal_sum_min_var = FakeVar("--")
        self.odcal_sum_max_var = FakeVar("--")
        self.odcal_drop_rate_var = FakeVar("--")
        self.odcal_B_active_var = FakeVar("--")
        self.odcal_dref_var = FakeVar("180.000")
        self.odcal_cmd_var = FakeVar("M0,1")
        self.odcal_angle_src_var = FakeVar("AX3")
        self.calibration_repository = _FakeCalibrationRepository()


class TestAppHostOdCalibration:
    def test_one_rev_progress_unwraps_across_zero_and_detects_done(self) -> None:
        host = _FakeOdCalHost()

        assert host._odcal_update_rev_progress(350.0) == 0.0
        assert host._odcal_update_rev_progress(10.0) == 20.0
        assert host._odcal_update_rev_progress(100.0) == 110.0
        assert not host._odcal_rev_done()

        host._odcal_rev_progress_deg = 359.1
        assert host._odcal_rev_done()

    def test_degree_source_prefers_relative_angle_then_absolute_angle(self) -> None:
        host = _FakeOdCalHost()

        assert host._odcal_deg_from_point({"theta_rel": 361.9, "theta": 10}) == 1
        assert host._odcal_deg_from_point({"theta": -1.0}) == 359
        assert host._odcal_deg_from_point({}) is None

    def test_mask_ranges_round_trip_wraparound_segments(self) -> None:
        host = _FakeOdCalHost()

        mask = host._odcal_ranges_to_mask([(358, 2), (90, 91)])

        assert [idx for idx, value in enumerate(mask) if value] == [0, 1, 2, 90, 91, 358, 359]
        assert host._odcal_mask_to_ranges(mask) == [(358, 2), (90, 91)]
        assert "90~91" in host._odcal_ranges_str([(90, 91)])

    def test_detect_defect_mask_closes_gap_and_applies_padding(self) -> None:
        host = _FakeOdCalHost()
        r_bin = np.zeros((360,), dtype=float)
        has = np.ones((360,), dtype=bool)
        r_bin[10] = -0.050
        r_bin[12] = -0.050

        mask, debug = host._odcal_detect_defect_mask(
            r_bin,
            has,
            abs_thr=0.010,
            k_sigma=0.0,
            gap1_close=True,
            min_len=2,
            pad=1,
        )

        assert [idx for idx, value in enumerate(mask) if value] == [9, 10, 11, 12, 13]
        assert debug["segments"]

    def test_best_shift_helpers_align_template_to_current_defect(self) -> None:
        host = _FakeOdCalHost()
        template = host._odcal_ranges_to_mask([(10, 11)])
        r_bin = np.zeros((360,), dtype=float)
        has = np.ones((360,), dtype=bool)
        r_bin[25] = -0.100
        r_bin[26] = -0.080

        shift, debug = host._odcal_best_shift_template(template, r_bin, has)

        assert shift == 15
        assert shift is not None
        assert debug["n"] == 2
        shifted = host._odcal_shift_mask(template, int(shift))
        assert [idx for idx, value in enumerate(shifted) if value] == [25, 26]
        assert host._odcal_best_shift_by_overlap(template, shifted)[0] == 15

    def test_prepare_sums_and_update_stats_apply_mapping_and_drop_rate(self) -> None:
        host = _FakeOdCalHost()
        host.odcal_map_out1_var.set("R")
        host._odcal_points = [
            {"v1": 1.0, "v2": 2.0, "theta": 0},
            {"v1": 2.0, "v2": 3.0, "theta": 1},
            {"v1": 3.0, "v2": 4.0, "theta": 2},
        ]
        host._odcal_drop_cnt = 1

        sums, meta = host._odcal_prepare_sums()
        host._odcal_update_stats()

        assert sums == [3.0, 5.0, 7.0]
        assert meta["defect_mode"] == "OFF"
        assert host.odcal_sum_mean_var.get() == "5.00000"
        assert host.odcal_sum_min_var.get() == "3.00000"
        assert host.odcal_sum_max_var.get() == "7.00000"
        assert host.odcal_drop_rate_var.get() == "33.3%"

    def test_load_and_save_active_calibration_sync_ui_fields_and_template(self) -> None:
        host = _FakeOdCalHost()
        mask = host._odcal_ranges_to_mask([(358, 2), (90, 90)])
        repo = _FakeCalibrationRepository(
            {
                "B_active": 1.234567,
                "D_ref": 180.25,
                "cmd_used": "M1,1",
                "out1_map": "R",
                "angle_src_ui": "无角度",
                "filter": "中值(3)",
                "outlier_sigma": "2.5",
                "defect_template_mask": mask,
            }
        )
        host.calibration_repository = repo

        host._odcal_load_active()
        host._odcal_save_active({"B_active": 2.0})

        assert host.odcal_B_active_var.get() == "1.23457"
        assert host.odcal_dref_var.get() == "180.250"
        assert host.odcal_cmd_var.get() == "M1,1"
        assert host.odcal_map_out1_var.get() == "R"
        assert host.odcal_angle_src_var.get() == "无角度"
        assert host.odcal_filter_var.get() == "中值(3)"
        assert host.odcal_outlier_sigma_var.get() == "2.5"
        assert host.odcal_defect_mode_var.get() == "TEMPLATE"
        assert host._odcal_defect_template_mask == mask
        assert repo.saved == [{"B_active": 2.0}]
