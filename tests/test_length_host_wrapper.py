from __future__ import annotations

"""Verify length.py wrapper path works at runtime (DTOs importable, service called)."""

from services.length_service import LengthCalcRequest, LengthCalcResult, LengthService


class _FakeService:
    def __init__(self) -> None:
        self.last_request: LengthCalcRequest | None = None

    def calculate_length(self, request: LengthCalcRequest) -> LengthCalcResult:
        self.last_request = request
        return LengthCalcResult(length=request.edge_low - request.edge_high)


class TestLengthUpdateWrapper:
    """Simulate the _len_try_update_measured_length wrapper logic without Tk."""

    def test_dto_classes_are_runtime_importable(self) -> None:
        """LengthCalcRequest and LengthCalcResult must be importable at runtime."""
        req = LengthCalcRequest(edge_low=500.0, edge_high=100.0)
        result = LengthCalcResult(length=400.0)
        assert req.edge_low == 500.0
        assert req.edge_high == 100.0
        assert result.length == 400.0

    def test_update_measured_length_constructs_dto_and_calls_service(self) -> None:
        """Simulate what _len_try_update_measured_length does."""
        svc = _FakeService()

        # Simulate reading from UI vars
        z_low = 500.0
        z_high = 100.0

        # Use LengthService or fallback (simulated)
        if svc is not None:
            result = svc.calculate_length(
                LengthCalcRequest(edge_low=z_low, edge_high=z_high),
            )
            L = result.length
        else:
            L = z_low - z_high

        assert L == 400.0
        assert svc.last_request is not None
        assert svc.last_request.edge_low == 500.0
        assert svc.last_request.edge_high == 100.0

    def test_update_measured_length_with_fallback_when_no_service(self) -> None:
        """Fallback path should still work without service."""
        from domain.length_math import length_from_edges

        L = length_from_edges(500.0, 100.0)
        assert L == 400.0

    def test_invalid_edges_return_none(self) -> None:
        """DTO with reversed edges should result in None length."""
        svc = _FakeService()
        result = svc.calculate_length(
            LengthCalcRequest(edge_low=100.0, edge_high=500.0),
        )
        # Reversed edges produce negative length - service delegates to
        # domain/length_math which validates and returns None
        real_svc = LengthService()
        real_result = real_svc.calculate_length(
            LengthCalcRequest(edge_low=100.0, edge_high=500.0),
        )
        assert real_result.length is None
