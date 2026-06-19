from __future__ import annotations

"""Unit tests for TeachService with fake ports."""

from services.teach_service import (
    StandbyTarget,
    TeachMotionPort,
    TeachOperatorPort,
    TeachRecipePort,
    TeachService,
    TeachTargetRequest,
    TeachTargetResult,
)


class _FakeMotion(TeachMotionPort):
    def __init__(self, fail_on_axis: int | None = None) -> None:
        self.calls: list[tuple[int, float, dict]] = []
        self._fail_axis = fail_on_axis

    def movea_abs(self, axis: int, pos_abs: float, *, context: str = "MoveA") -> None:
        self.calls.append((axis, pos_abs, {"context": context}))
        if self._fail_axis == axis:
            raise RuntimeError(f"axis {axis} move failed")


class _FakeOperator(TeachOperatorPort):
    def __init__(self) -> None:
        self.warnings: list[tuple[str, str]] = []
        self.errors: list[tuple[str, str]] = []
        self.confirm_return: bool = True

    def show_warning(self, title: str, message: str) -> None:
        self.warnings.append((title, message))

    def show_error(self, title: str, message: str) -> None:
        self.errors.append((title, message))


class _FakeRecipes(TeachRecipePort):
    pass


# -- helpers --------------------------------------------------------------


def _make_request(ax0: float = 100.0, ax1: float = 200.0, ax4: float = 300.0) -> TeachTargetRequest:
    return TeachTargetRequest(
        targets=StandbyTarget(ax0_abs=ax0, ax1_abs=ax1, ax4_abs=ax4),
    )


# -- tests ----------------------------------------------------------------


class TestTeachServiceMoveToTargets:
    def test_move_to_targets_calls_movea_abs_for_each_axis(self) -> None:
        motion = _FakeMotion()
        operator = _FakeOperator()
        svc = TeachService(motion=motion, operator=operator, recipes=_FakeRecipes())

        result = svc.move_to_targets(_make_request(1.0, 2.0, 3.0))

        assert result.ok
        assert result.moved_axes == [0, 1, 4]
        assert len(motion.calls) == 3
        assert motion.calls[0] == (0, 1.0, {"context": "TeachStandby"})
        assert motion.calls[1] == (1, 2.0, {"context": "TeachStandby"})
        assert motion.calls[2] == (4, 3.0, {"context": "TeachStandby"})

    def test_move_to_targets_failure_on_first_axis_stops_and_reports(self) -> None:
        motion = _FakeMotion(fail_on_axis=0)
        svc = TeachService(motion=motion, operator=_FakeOperator(), recipes=_FakeRecipes())

        result = svc.move_to_targets(_make_request(1.0, 2.0, 3.0))

        assert not result.ok
        assert result.moved_axes == []
        assert "axis 0 move failed" in result.reason

    def test_move_to_targets_failure_on_third_axis_reports_partial(self) -> None:
        motion = _FakeMotion(fail_on_axis=4)
        svc = TeachService(motion=motion, operator=_FakeOperator(), recipes=_FakeRecipes())

        result = svc.move_to_targets(_make_request(1.0, 2.0, 3.0))

        assert not result.ok
        assert result.moved_axes == [0, 1]
        assert "axis 4" in result.reason

    def test_operator_not_called_during_normal_move(self) -> None:
        """TeachService should NOT call operator during move — operator is
        for the caller (teach.py) to use for confirm/warn/error."""
        operator = _FakeOperator()
        svc = TeachService(motion=_FakeMotion(), operator=operator, recipes=_FakeRecipes())

        svc.move_to_targets(_make_request())

        assert operator.warnings == []
        assert operator.errors == []


class TestTeachServiceMoveToStandby:
    def test_move_to_standby_delegates_to_move_to_targets(self) -> None:
        motion = _FakeMotion()
        svc = TeachService(motion=motion, operator=_FakeOperator(), recipes=_FakeRecipes())

        result = svc.move_to_standby(_make_request(10.0, 20.0, 30.0))

        assert result.ok
        assert len(motion.calls) == 3
        assert motion.calls[0] == (0, 10.0, {"context": "TeachStandby"})
        assert motion.calls[1] == (1, 20.0, {"context": "TeachStandby"})
        assert motion.calls[2] == (4, 30.0, {"context": "TeachStandby"})


class TestStandbyTargetDto:
    def test_standby_target_is_dataclass(self) -> None:
        t = StandbyTarget(ax0_abs=1.0, ax1_abs=2.0, ax4_abs=3.0)
        assert t.ax0_abs == 1.0
        assert t.ax1_abs == 2.0
        assert t.ax4_abs == 3.0


class TestTeachTargetResult:
    def test_ok_result(self) -> None:
        r = TeachTargetResult(ok=True, moved_axes=[0, 1, 4])
        assert r.ok
        assert r.moved_axes == [0, 1, 4]
        assert r.reason == ""

    def test_error_result(self) -> None:
        r = TeachTargetResult(ok=False, moved_axes=[], reason="timeout")
        assert not r.ok
        assert r.moved_axes == []
        assert r.reason == "timeout"
