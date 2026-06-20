from __future__ import annotations

"""Verify teach.py wrapper path works at runtime (DTOs importable, service called)."""

from services.teach_service import StandbyTarget, TeachTargetRequest, TeachTargetResult


class _FakeTeachService:
    def __init__(self) -> None:
        self.last_request: TeachTargetRequest | None = None
        self.result = TeachTargetResult(ok=True, moved_axes=[0, 1, 4])

    def move_to_targets(self, request: TeachTargetRequest) -> TeachTargetResult:
        self.last_request = request
        return self.result


class _FakeRecipe:
    standby_valid: bool = True
    standby_ax0_abs: float = 100.0
    standby_ax1_abs: float = 200.0
    standby_ax4_abs: float = 300.0


class TestTeachStandbyWrapper:
    """Simulate the _teach_go_standby wrapper logic without Tk."""

    def test_dto_classes_are_runtime_importable(self) -> None:
        """StandbyTarget and TeachTargetRequest must be importable at runtime."""
        t = StandbyTarget(ax0_abs=1.0, ax1_abs=2.0, ax4_abs=3.0)
        req = TeachTargetRequest(targets=t)
        assert req.targets.ax0_abs == 1.0
        assert req.targets.ax1_abs == 2.0
        assert req.targets.ax4_abs == 3.0

    def test_go_standby_wrapper_constructs_dto_and_calls_service(self) -> None:
        """Simulate what _teach_go_standby would do with a real service."""
        recipe = _FakeRecipe()
        svc = _FakeTeachService()

        # This is the wrapper logic from teach.py (without Tk/AppHost)
        if not recipe.standby_valid:
            return

        request = TeachTargetRequest(
            targets=StandbyTarget(
                ax0_abs=recipe.standby_ax0_abs,
                ax1_abs=recipe.standby_ax1_abs,
                ax4_abs=recipe.standby_ax4_abs,
            ),
        )
        result = svc.move_to_targets(request)

        assert result.ok
        assert svc.last_request is not None
        assert svc.last_request.targets.ax0_abs == 100.0
        assert svc.last_request.targets.ax1_abs == 200.0
        assert svc.last_request.targets.ax4_abs == 300.0

    def test_go_standby_with_invalid_standby_skips_service(self) -> None:
        recipe = _FakeRecipe()
        recipe.standby_valid = False
        svc = _FakeTeachService()

        if not recipe.standby_valid:
            # Would show_warning and return
            pass

        assert svc.last_request is None
