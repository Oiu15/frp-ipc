from __future__ import annotations

"""Tests for domain/length_math.py pure functions."""

import pytest

from domain.length_math import length_from_edges, average_edge_pair, clamp_z
from domain.length_math import LengthRange


class TestLengthFromEdges:
    def test_normal_length(self) -> None:
        result = length_from_edges(500.0, 100.0)
        assert result == pytest.approx(400.0)

    def test_reversed_edges_returns_none(self) -> None:
        result = length_from_edges(100.0, 500.0)
        assert result is None

    def test_zero_length_returns_none(self) -> None:
        result = length_from_edges(100.0, 100.0)
        assert result is None

    def test_negative_edge_returns_none(self) -> None:
        result = length_from_edges(-10.0, 100.0)
        assert result is None

    def test_non_finite_returns_none(self) -> None:
        result = length_from_edges(float("nan"), 100.0)
        assert result is None


class TestAverageEdgePair:
    def test_average_of_two_values(self) -> None:
        assert average_edge_pair(10.0, 20.0) == pytest.approx(15.0)

    def test_same_values(self) -> None:
        assert average_edge_pair(3.0, 3.0) == pytest.approx(3.0)


class TestClampZ:
    def test_within_range(self) -> None:
        r = LengthRange(z_min=0.0, z_max=100.0, travel=100.0)
        assert clamp_z(50.0, r) == pytest.approx(50.0)

    def test_below_range(self) -> None:
        r = LengthRange(z_min=0.0, z_max=100.0, travel=100.0)
        assert clamp_z(-10.0, r) == pytest.approx(0.0)

    def test_above_range(self) -> None:
        r = LengthRange(z_min=0.0, z_max=100.0, travel=100.0)
        assert clamp_z(150.0, r) == pytest.approx(100.0)
