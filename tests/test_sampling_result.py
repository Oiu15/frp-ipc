from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from frp_workflow.steps.sampling_result import SamplingResult


def test_sampling_result_keeps_fields() -> None:
    primary_sample = object()
    id_sample = object()
    coords_od = np.array([[1.0, 2.0]])
    coords_id = np.array([[3.0, 4.0]])
    raw_points = [{"theta_deg": 0.0}]

    result = SamplingResult(
        primary_sample=primary_sample,
        id_sample=id_sample,
        coords_od=coords_od,
        coords_id=coords_id,
        raw_od="od",
        raw_id="id",
        raw_points=raw_points,
        split_shift_deg=1.25,
        coax_unreliable=False,
    )

    assert result.primary_sample is primary_sample
    assert result.id_sample is id_sample
    assert result.coords_od is coords_od
    assert result.coords_id is coords_id
    assert result.raw_od == "od"
    assert result.raw_id == "id"
    assert result.raw_points is raw_points
    assert result.split_shift_deg == 1.25
    assert result.coax_unreliable is False


def test_sampling_result_is_frozen() -> None:
    result = SamplingResult(
        primary_sample=object(),
        id_sample=None,
        coords_od=np.array([]),
        coords_id=np.array([]),
        raw_od="",
        raw_id="",
        raw_points=[],
        split_shift_deg=None,
        coax_unreliable=None,
    )

    with pytest.raises(FrozenInstanceError):
        setattr(result, "raw_points", [])
