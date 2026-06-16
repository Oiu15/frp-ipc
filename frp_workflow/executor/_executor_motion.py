from __future__ import annotations

import logging
import time
from typing import Any

from config.addresses import (
    OFF_VEL_MOVEA,
    OFF_VEL_VELMOVE,
    OFF_ACC,
    OFF_DEC,
    OFF_JERK,
    STS_RAW_NOT_ENABLED,
    STS_RAW_MOVING,
    STS_RAW_VELRUN,
    STS_RAW_SYNC,
    STS_RAW_HOMING,
    STS_RAW_STOPPING,
    STS_RAW_FAULT,
    STS_RAW_GROUP,
)
from core.modbus_codec import encode_fp64_le as encode_float64_to_4regs


class ExecutorMotionMixin:
    """Mixin providing axis motion helpers.

    Expects the following attributes/methods on ``self``:
        device: Any
        stop_event: Any
        _current_recipe: Any
    """

    # Typed port accessors — set by ExecutorCoreMixin.__init__, shared via MRO
    _typed_motion: Any = None  # type: ignore[assignment]
    _typed_sensors: Any = None  # type: ignore[assignment]
    _typed_operator: Any = None  # type: ignore[assignment]
    _typed_plc: Any = None  # type: ignore[assignment]
    device: Any
    stop_event: Any
    _current_recipe: Any

    # Methods called from other mixins (cooperative MRO)
    _should_stop: Any

    # =========================
    # Helpers (Axis_Ctrl raw state)
    # =========================
    def _is_fault(self, sts: int, err: int) -> bool:
        return (int(err) != 0) or (int(sts) == STS_RAW_FAULT)

    def is_fault_status(self, sts: int, err: int) -> bool:
        return self._is_fault(sts, err)

    def _is_enabled(self, sts: int) -> bool:
        return int(sts) != STS_RAW_NOT_ENABLED

    def is_enabled_status(self, sts: int) -> bool:
        return self._is_enabled(sts)

    def _is_moving(self, sts: int) -> bool:
        s = int(sts)
        return s in {
            STS_RAW_MOVING,
            STS_RAW_VELRUN,
            STS_RAW_SYNC,
            STS_RAW_HOMING,
            STS_RAW_STOPPING,
            STS_RAW_GROUP,
        }

    def is_moving_status(self, sts: int) -> bool:
        return self._is_moving(sts)

    def _write_fp64(self, axis: int, off: int, value: float) -> None:
        base = self._typed_plc._base(int(axis))
        self._typed_plc._write_regs(base + int(off), encode_float64_to_4regs(float(value)))

    def _ensure_movea_setpoints(
        self,
        axis: int,
        default_vel: float = 100.0,
        default_acc: float = 200.0,
        default_dec: float = 200.0,
        default_jerk: float = 500.0,
    ) -> None:
        """Ensure MoveA-related setpoints exist (non-zero) in PLC."""
        ac = self.device.get_axis_copy(int(axis))
        # vel is legacy mirror of Vel_MoveA
        if float(getattr(ac, "vel", 0.0) or 0.0) <= 0.0:
            self._write_fp64(axis, OFF_VEL_MOVEA, default_vel)
        if float(getattr(ac, "acc", 0.0) or 0.0) <= 0.0:
            self._write_fp64(axis, OFF_ACC, default_acc)
        if float(getattr(ac, "dec", 0.0) or 0.0) <= 0.0:
            self._write_fp64(axis, OFF_DEC, default_dec)
        if float(getattr(ac, "jerk", 0.0) or 0.0) <= 0.0:
            self._write_fp64(axis, OFF_JERK, default_jerk)

    def _ensure_velmove_setpoints(
        self,
        axis: int,
        default_vel: float = 200.0,
        default_acc: float = 200.0,
        default_dec: float = 200.0,
        default_jerk: float = 500.0,
    ) -> None:
        """Ensure VelMove-related setpoints exist (non-zero) in PLC."""
        ac = self.device.get_axis_copy(int(axis))
        if int(axis) == 3:
            try:
                # Use the planned recipe speed for this run, not runtime snapshot.
                current_recipe = self._current_recipe
                target_vel = float(current_recipe.rot_vel_velmove) if current_recipe is not None else 0.0
            except Exception:
                target_vel = 0.0

            if target_vel <= 0.0:
                default_vel = 200.0
                logging.info(
                    f"[AX3_SETPOINT] ensure_default_applied "
                    f"reason=target_vel_invalid({target_vel}) "
                    f"default={default_vel}"
                )
                self._write_fp64(axis, OFF_VEL_VELMOVE, default_vel)
            else:
                logging.info(
                    f"[AX3_SETPOINT] ensure_skip "
                    f"reason=valid_target_vel({target_vel})"
                )
        else:
            v = float(getattr(ac, "vel_velmove", 0.0) or 0.0)
            if v <= 0.0:
                self._write_fp64(axis, OFF_VEL_VELMOVE, default_vel)

        # For simplicity, reuse common acc/dec/jerk
        if float(getattr(ac, "acc", 0.0) or 0.0) <= 0.0:
            self._write_fp64(axis, OFF_ACC, default_acc)
        if float(getattr(ac, "dec", 0.0) or 0.0) <= 0.0:
            self._write_fp64(axis, OFF_DEC, default_dec)
        if float(getattr(ac, "jerk", 0.0) or 0.0) <= 0.0:
            self._write_fp64(axis, OFF_JERK, default_jerk)

    def _wait_in_position(
        self, axis: int, tgt_abs: float, pos_tol: float, timeout_s: float
    ) -> bool:
        t0 = time.time()
        while (time.time() - t0) < float(timeout_s):
            if self._should_stop():
                return False
            ac = self.device.get_axis_copy(axis)

            sts = int(getattr(ac, "sts", 0))
            err_code = int(getattr(ac, "err", 0))
            if self._is_fault(sts, err_code):
                raise RuntimeError(f"AX{axis} 故障中，Err={err_code}")

            pos_err = abs(float(getattr(ac, "act_pos", 0.0)) - float(tgt_abs))

            # acceptance: position error small AND axis not in a moving state
            if (pos_err <= float(pos_tol)) and (not self._is_moving(sts)):
                return True

            time.sleep(0.08)

        return False

    def wait_in_position_result(
        self, axis: int, tgt_abs: float, pos_tol: float, timeout_s: float,
    ) -> bool:
        return self._wait_in_position(axis, tgt_abs, pos_tol, timeout_s)


__all__ = ["ExecutorMotionMixin"]
