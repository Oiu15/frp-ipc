from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from domain.state import CalibrationSnapshot


class ExecutorClampsMixin:
    """Mixin providing clamp management and calibration snapshot helpers.

    Expects the following attributes/methods on ``self``:
        app: Any
        device: Any
        stop_event: threading.Event

        # Methods called from other mixins:
        # self._sleep_cancelable(seconds, poll_s=0.05) -> bool
        # self._should_stop() -> bool
        # self._emit_auto_state(state, msg) -> None
    """

    app: Any
    device: Any
    stop_event: threading.Event

    def _clamps_are_closed(self) -> bool:
        try:
            return bool(int(self.app.get_y_point(10)) == 1 and int(self.app.get_y_point(11)) == 1)
        except Exception:
            return False

    def _prepare_clamps_for_auto(self, recipe) -> bool:
        if self._clamps_are_closed():
            self._emit_auto_state("PREP", "夹爪已夹紧，跳过夹紧输出")
            return True

        self._emit_auto_state("PREP", "夹爪准备：执行夹紧")
        try:
            self.app.plc_write_y_point(10, 1)
            self.app.plc_write_y_point(11, 1)
        except Exception:
            pass

        wait_s = float(getattr(recipe, "clamp_confirm_wait_s", 3.0) or 0.0)
        if wait_s < 0.0:
            try:
                res = self.app.operator_confirm(
                    "夹爪确认",
                    "请确认夹爪已经夹紧。\n\nX3：确认继续\nX4：取消流程",
                    allow_stop=True,
                    timeout_s=None,
                )
            except Exception:
                res = "timeout"
            if res != "confirm" or self._should_stop():
                self._emit_auto_state("STOP", f"夹爪确认取消/超时：{res}")
                return False
            return True

        if wait_s > 0.0:
            self._emit_auto_state("PREP", f"夹爪夹紧后等待 {wait_s:.1f}s 自动确认")
            if not self._sleep_cancelable(wait_s):
                self._emit_auto_state("STOP", "夹爪等待被中止")
                return False
        return not self._should_stop()

    def _verify_ax2_when_length_disabled(self, recipe, ax_clamp: int = 2, tolerance_mm: float = 10.0) -> bool:
        if bool(getattr(recipe, "len_enable", False)):
            return True

        if not bool(getattr(recipe, "ax2_rot_valid", False)):
            try:
                res = self.app.operator_confirm(
                    "AX2位置确认",
                    "长度检测未启用，但配方未保存 AX2 旋转测量位。\n\nX3：确认继续\nX4：取消流程",
                    allow_stop=True,
                    timeout_s=None,
                )
            except Exception:
                res = "timeout"
            if res != "confirm" or self._should_stop():
                self._emit_auto_state("STOP", f"AX2位置确认取消/超时：{res}")
                return False
            return True

        target = float(getattr(recipe, "ax2_rot_abs", 0.0) or 0.0)
        try:
            current = float(getattr(self.device.get_axis_copy(int(ax_clamp)), "act_pos", 0.0) or 0.0)
        except Exception:
            current = 0.0
        delta = current - target
        if abs(delta) <= float(tolerance_mm):
            self._emit_auto_state("PREP", f"AX2当前位置确认通过：当前 {current:.3f}，目标 {target:.3f}")
            return True

        try:
            res = self.app.operator_confirm(
                "AX2位置偏差确认",
                (
                    "长度检测未启用，AX2不会自动定位。\n\n"
                    f"当前值：{current:.3f} mm\n"
                    f"目标值：{target:.3f} mm\n"
                    f"偏差：{delta:.3f} mm\n\n"
                    "X3：确认继续\nX4：取消流程"
                ),
                allow_stop=True,
                timeout_s=None,
            )
        except Exception:
            res = "timeout"
        if res != "confirm" or self._should_stop():
            self._emit_auto_state("STOP", f"AX2位置偏差确认取消/超时：{res}")
            return False
        return True

    def _get_calibration_snapshot(self, refresh: bool = False) -> CalibrationSnapshot:
        """Get a workflow-facing calibration snapshot from the runtime host."""
        if refresh or self._calibration_snapshot is None:
            snapshot = None
            try:
                getter = getattr(self.app, "get_calibration_snapshot", None)
                if callable(getter):
                    snapshot = getter()
            except Exception:
                snapshot = None

            if snapshot is None:
                try:
                    repo = getattr(self.app, "calibration_repository", None)
                    if repo is not None and hasattr(repo, "load_snapshot"):
                        snapshot = repo.load_snapshot()
                except Exception:
                    snapshot = None

            self._calibration_snapshot = snapshot if isinstance(snapshot, CalibrationSnapshot) else CalibrationSnapshot()
        return self._calibration_snapshot


__all__ = ["ExecutorClampsMixin"]
