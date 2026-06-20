from __future__ import annotations

"""Unit tests for LengthService."""

import pytest

from services.length_service import LengthCalcRequest, LengthService


class TestLengthServiceCalculateLength:
    def test_calculate_normal_length(self) -> None:
        svc = LengthService()
        result = svc.calculate_length(
            LengthCalcRequest(edge_low=500.0, edge_high=100.0)
        )
        assert result.length == pytest.approx(400.0)

    def test_calculate_reversed_edges_returns_none(self) -> None:
        svc = LengthService()
        result = svc.calculate_length(
            LengthCalcRequest(edge_low=100.0, edge_high=500.0)
        )
        assert result.length is None

    def test_calculate_zero_length_returns_none(self) -> None:
        svc = LengthService()
        result = svc.calculate_length(
            LengthCalcRequest(edge_low=100.0, edge_high=100.0)
        )
        assert result.length is None

    def test_dto_is_frozen(self) -> None:
        req = LengthCalcRequest(edge_low=1.0, edge_high=2.0)
        assert req.edge_low == 1.0
        assert req.edge_high == 2.0
