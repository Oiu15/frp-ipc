from __future__ import annotations

from typing import Any

from tests.fakes import FakeVar

from application.host.teach import HostTeachMixin
from config.addresses import CMD_JOG_B_REQ, CMD_JOG_F_REQ
from core.models import AxisCal, AxisComm, Recipe
from services.teach_service import TeachService, TeachTargetRequest, TeachTargetResult


class _FailingTeachService:
    def __init__(self) -> None:
        self.last_request: TeachTargetRequest | None = None

    def move_to_targets(self, request: TeachTargetRequest) -> TeachTargetResult:
        self.last_request = request
        return TeachTargetResult(ok=False, moved_axes=[], reason="boom")


class _FakeTeachHost(HostTeachMixin):
    def __init__(self) -> None:
        self.axis_cal = AxisCal(sign=1)
        self.recipe = Recipe(teach_axes_mode=3)
        self.teach_axes_mode_var = FakeVar(3)
        self.teach_rel_dist_var = FakeVar("5")
        self.center_pos_var = FakeVar("")
        self.axis_cal_vars: dict[str, Any] = {}
        self.axis_cal_field_status_vars: dict[str, Any] = {}
        self.axes = {
            0: AxisComm(act_pos=100.0, softlim_pos=1000.0, softlim_neg=-1000.0),
            1: AxisComm(act_pos=200.0, softlim_pos=1000.0, softlim_neg=-1000.0),
            2: AxisComm(act_pos=10.0, softlim_pos=1000.0, softlim_neg=-1000.0),
            4: AxisComm(act_pos=50.0, softlim_pos=1000.0, softlim_neg=-1000.0),
        }
        self.moves: list[tuple[int, float, str]] = []
        self.cmd_bits: list[tuple[int, int, int]] = []
        self.warnings: list[tuple[str, str]] = []
        self.errors: list[tuple[str, str]] = []
        self.refresh_count = 0
        self.teach_service: Any = None

    def get_axis_copy(self, axis: int) -> AxisComm:
        return self.axes[int(axis)]

    def movea_abs(self, axis: int, pos_abs: float, *, context: str = "MoveA") -> None:
        self.moves.append((int(axis), float(pos_abs), str(context)))

    def set_cmd_bits(self, axis: int, set_mask: int = 0, clr_mask: int = 0) -> None:
        self.cmd_bits.append((int(axis), int(set_mask), int(clr_mask)))

    def _write_axis_params(self, axis: int) -> None:
        return None

    def after(self, ms: Any, func: Any | None = None, *args: Any) -> Any:
        return None

    def _refresh_teach_pos(self) -> None:
        self.refresh_count += 1

    def show_warning(self, title: str, message: str) -> None:
        self.warnings.append((title, message))

    def show_error(self, title: str, message: str) -> None:
        self.errors.append((title, message))


class TestHostTeachMixin:
    def test_teach_move_relative_moves_center_axis_in_z_disp(self) -> None:
        host = _FakeTeachHost()

        host._teach_move_relative()

        assert host.moves == [(2, 15.0, "TeachRel")]
        assert host.refresh_count == 1

    def test_teach_jog_release_clears_selected_axes(self) -> None:
        host = _FakeTeachHost()
        host.recipe.teach_axes_mode = 2

        host._teach_jog_hold("fwd", False)

        clear_mask = CMD_JOG_F_REQ | CMD_JOG_B_REQ
        assert host.cmd_bits == [
                (0, 0, clear_mask),
                (1, 0, clear_mask),
                (4, 0, clear_mask),
            ]

    def test_teach_go_standby_calls_real_service_with_recipe_targets(self) -> None:
        host = _FakeTeachHost()
        host.recipe.standby_valid = True
        host.recipe.standby_ax0_abs = 1.0
        host.recipe.standby_ax1_abs = 2.0
        host.recipe.standby_ax4_abs = 3.0
        host.teach_service = TeachService(motion=host, operator=host, recipes=host)

        host._teach_go_standby()

        assert host.moves == [
            (0, 1.0, "TeachStandby"),
            (1, 2.0, "TeachStandby"),
            (4, 3.0, "TeachStandby"),
        ]
        assert host.warnings == []
        assert host.errors == []

    def test_teach_go_standby_warns_and_skips_service_when_not_configured(self) -> None:
        host = _FakeTeachHost()
        host.recipe.standby_valid = False
        host.teach_service = TeachService(motion=host, operator=host, recipes=host)

        host._teach_go_standby()

        assert host.moves == []
        assert host.warnings
        assert "待定点尚未设置" in host.warnings[-1][1]
        assert host.errors == []

    def test_teach_go_standby_reports_service_failure_reason(self) -> None:
        host = _FakeTeachHost()
        service = _FailingTeachService()
        host.recipe.standby_valid = True
        host.recipe.standby_ax0_abs = 1.0
        host.recipe.standby_ax1_abs = 2.0
        host.recipe.standby_ax4_abs = 3.0
        host.teach_service = service

        host._teach_go_standby()

        assert service.last_request is not None
        assert service.last_request.targets.ax0_abs == 1.0
        assert service.last_request.targets.ax1_abs == 2.0
        assert service.last_request.targets.ax4_abs == 3.0
        assert host.moves == []
        assert host.errors
        assert "boom" in host.errors[-1][1]
