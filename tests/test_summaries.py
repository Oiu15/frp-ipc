from __future__ import annotations

import pytest

from core.models import MeasureRow, Recipe
from domain.summaries import (
    EccentricityUpdate,
    apply_eccentricity_updates,
    compute_postcalc_result,
    compute_run_summary,
    merge_summary_snapshot,
    summary_snapshot_from_payload,
)


class TestSummaries:
    def test_summary_snapshot_and_merge_are_pure(self) -> None:
        seed = {"conc_max": 0.7}
        snapshot = summary_snapshot_from_payload(
            {
                "straightness": 1.2,
                "axis_dist": 0.3,
                "od_tilt_deg": 0.4,
            }
        )

        merged = merge_summary_snapshot(seed, snapshot)

        assert snapshot.straight_od == 1.2
        assert snapshot.axis_dist == 0.3
        assert snapshot.od_tilt_deg == 0.4
        assert seed == {"conc_max": 0.7}
        assert merged["conc_max"] == 0.7
        assert merged["straight_od"] == 1.2
        assert merged["axis_dist"] == 0.3
        assert merged["od_tilt_deg"] == 0.4

    def test_apply_eccentricity_updates_returns_new_rows(self) -> None:
        rows = [
            MeasureRow(
                idx=1, x_ui=10.0, x_abs=20.0,
                od_avg=100.0, od_dev=0.1, od_runout=0.2, od_round=0.3,
                id_avg=80.0, id_dev=0.1, id_runout=0.2, id_round=0.3,
                concentricity=0.4,
            )
        ]

        updated = apply_eccentricity_updates(
            rows,
            [EccentricityUpdate(row_index=0, od_ecc=0.55, id_ecc=0.66)],
        )

        assert rows[0].od_ecc is None
        assert rows[0].id_ecc is None
        assert updated[0].od_ecc == 0.55
        assert updated[0].id_ecc == 0.66
        assert updated[0] is not rows[0]

    def test_compute_postcalc_result_returns_straightness_and_postcalc_payloads(self) -> None:
        result = compute_postcalc_result(
            [(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)],
            [(1.0, 0.0, 0.0), (1.0, 0.0, 10.0)],
            concentricity_list=[0.8, 1.2],
            id_single_enable=False,
        )

        assert result.straightness_payload["straight_od"] == 0.0
        assert result.straightness_payload["straight_id"] == 0.0
        assert result.straightness_payload["axis_dist"] == pytest.approx(1.0)
        assert result.straightness_payload["conc_max"] == 1.2
        assert result.straightness_payload["axis_span_max"] == pytest.approx(1.0)
        for value in result.postcalc_payload["ecc_od"]:
            assert value == pytest.approx(0.0, abs=1e-10)
        for value in result.postcalc_payload["ecc_id"]:
            assert value == pytest.approx(0.0, abs=1e-10)

    def test_compute_run_summary_is_input_only(self) -> None:
        recipe = Recipe(name="summary-test")
        rows = [
            MeasureRow(
                idx=1, x_ui=10.0, x_abs=20.0,
                od_avg=100.0, od_dev=0.2, od_runout=0.4, od_round=0.5,
                id_avg=80.0, id_dev=-0.1, id_runout=0.3, id_round=0.4,
                concentricity=0.6, ok=True,
            )
        ]
        raw_points = [
            {"od_mm": 100.0, "id_mm": 80.0},
            {"od_mm": 101.0, "id_mm": 80.5},
        ]
        summary = compute_run_summary(
            recipe=recipe,
            rows=rows,
            raw_points=raw_points,
            summary_cache={"straight_od": 1.1, "axis_dist": 0.7},
        )

        assert summary["ok"] is True
        assert summary["max_od_dev_abs"] == 0.2
        assert summary["max_id_dev_abs"] == 0.1
        assert summary["conc_max"] == 0.6
        assert summary["straight_od"] == 1.1
        assert summary["axis_dist"] == 0.7
        assert summary["od_range"] == 1.0
        assert summary["id_range"] == 0.5
        assert summary["judge_ok_cnt"] == 1
        assert summary["judge_total"] == 1
