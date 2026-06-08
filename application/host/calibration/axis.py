from __future__ import annotations

"""Axis calibration mixin for AppHost."""

import queue
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from application.host.calibration.state import AxisCalibrationState
from config.addresses import AXISCAL_MB_BASE, AXISCAL_WORDS, LINEAR_AXES
from core.models import AxisCal, AxisComm
from drivers.plc_client import CmdReadRegs, CmdWriteRegs


class HostAxisCalibrationMixin:
    """Mixin providing AxisCal UI state, PLC read/write, and calibration helpers."""

    cmd_q: queue.Queue[Any]
    axis_cal: AxisCal
    axis_cal_vars: dict[str, Any]
    axis_cal_field_status_vars: dict[str, Any]
    axis_cal_status_vars: dict[str, Any]
    _axis_cal_state: AxisCalibrationState
    _axis_cal_write_expect_regs: list[int] | None

    if TYPE_CHECKING:
        def get_axis_copy(self, axis: int) -> AxisComm: ...

    def _get_axis_calibration_state(self) -> AxisCalibrationState:
        state = self.__dict__.get("_axis_cal_state", None)
        if not isinstance(state, AxisCalibrationState):
            state = AxisCalibrationState(getattr(self, "axis_cal", AxisCal()))
            existing = getattr(self, "_axis_cal_write_expect_regs", None)
            if existing is not None:
                state.set_expected_regs(existing)
            self._axis_cal_state = state
        return state

    def _set_axis_cal(self, cal: AxisCal) -> AxisCal:
        self.axis_cal = self._get_axis_calibration_state().set_current(cal)
        return cal

    def _set_axis_cal_write_expect_regs(self, regs: Iterable[int] | None) -> None:
        if regs is None:
            self._axis_cal_write_expect_regs = None
            self._get_axis_calibration_state().clear_expected_regs()
            return
        expected = self._get_axis_calibration_state().set_expected_regs(regs)
        self._axis_cal_write_expect_regs = expected

    def _axis_cal_set_field_status(self, keys: Iterable[str], text: str) -> None:
        """Update per-field status label(s) on the AxisCal page."""
        sv = getattr(self, "axis_cal_field_status_vars", None)
        if not isinstance(sv, dict):
            return
        self._get_axis_calibration_state().set_field_status(sv, keys, text)

    def _axis_cal_from_ui(self) -> AxisCal:
        """Build an AxisCal instance from UI entry variables.

        Note: z_pos is IPC-only (will not be written to PLC), but we keep it in memory.
        """
        return self._get_axis_calibration_state().read_from_vars(self.axis_cal_vars)

    def _axis_cal_to_ui(self, cal: AxisCal) -> None:
        """Push an AxisCal instance into UI entry variables."""
        try:
            self._get_axis_calibration_state().write_to_vars(self.axis_cal_vars, cal)
        except Exception:
            pass

    def axis_cal_read(self) -> None:
        """Read the axis calibration block from PLC (HD1000..)."""
        try:
            self._axis_cal_set_field_status(
                ["sign", "off_ax0", "off_ax1", "off_ax2", "off_ax4", "b14", "b2", "keepout_w"],
                "读取中",
            )
            self.cmd_q.put(CmdReadRegs(AXISCAL_MB_BASE, AXISCAL_WORDS, "axis_cal"))
            print(f"[axis_cal] request read: addr={AXISCAL_MB_BASE} count={AXISCAL_WORDS}")
        except Exception as e:
            print(f"[axis_cal] enqueue read failed: {e}")

    def axis_cal_write(self) -> None:
        """Write the axis calibration block to PLC (HD1000..).

        Note: z_pos will NOT be written to PLC.
        """
        try:
            cal = self._axis_cal_from_ui()
            # Keep IPC copy
            self._set_axis_cal(cal)
            regs = cal.to_regs()
            # Enqueue write then read back to verify
            self._set_axis_cal_write_expect_regs(regs)
            self._axis_cal_set_field_status(
                ["sign", "off_ax0", "off_ax1", "off_ax2", "off_ax4", "b14", "b2", "keepout_w"],
                "写入中",
            )
            self.cmd_q.put(CmdWriteRegs(AXISCAL_MB_BASE, regs))
            self.cmd_q.put(CmdReadRegs(AXISCAL_MB_BASE, AXISCAL_WORDS, "axis_cal_verify"))
            print(
                f"[axis_cal] write+verify: addr={AXISCAL_MB_BASE} words={len(regs)} "
                f"(will read back {AXISCAL_WORDS} words)"
            )
        except Exception as e:
            self._axis_cal_set_field_status(
                ["sign", "off_ax0", "off_ax1", "off_ax2", "off_ax4", "b14", "b2", "keepout_w"],
                "写入失败",
            )
            print(f"[axis_cal] write failed: {e}")

    def axis_cal_capture_offsets(self) -> None:
        """Capture Off_AX0/1/2/4 from current axis feedback (Act_Pos).

        Semantics:
        - Off_AXn is defined as the servo feedback position (abs) at Z_raw == 0.
        - Capturing Off_AXn at the current position makes current Z_raw become 0.

        This function only updates IPC UI/in-memory values. Use "Write" to persist to PLC.
        """
        try:
            cal = self._axis_cal_from_ui()
            act0 = float(self.get_axis_copy(0).act_pos)
            act1 = float(self.get_axis_copy(1).act_pos)
            act2 = float(self.get_axis_copy(2).act_pos)
            act4 = float(self.get_axis_copy(4).act_pos)

            cal.off_ax0 = act0
            cal.off_ax1 = act1
            cal.off_ax2 = act2
            cal.off_ax4 = act4

            self._axis_cal_set_field_status(
                ["off_ax0", "off_ax1", "off_ax2", "off_ax4"],
                "已采集/未写入",
            )

            self._set_axis_cal(cal)
            self._axis_cal_to_ui(cal)
            self.axis_cal_refresh_status()
            print(
                "[axis_cal] capture offsets: "
                f"off_ax0={act0:.6f} off_ax1={act1:.6f} off_ax2={act2:.6f} off_ax4={act4:.6f}"
            )
        except Exception as e:
            print(f"[axis_cal] capture offsets failed: {e}")

    def axis_cal_calibrate_b14(self) -> None:
        """Calibrate B14 based on current OD/ID plane alignment.

        Uses current feedback positions:
            z_od_raw = Z0_raw (from AX0)
            z_id_raw = Z1_raw + Z4_raw (AX1 + AX4)
            B14 = z_id_raw - z_od_raw

        Only updates IPC UI/in-memory values. Use "Write" to persist to PLC.
        """
        try:
            cal = self._axis_cal_from_ui()
            act0 = float(self.get_axis_copy(0).act_pos)
            act1 = float(self.get_axis_copy(1).act_pos)
            act4 = float(self.get_axis_copy(4).act_pos)

            z0_raw = cal.abs_to_z_raw(0, act0)
            z1_raw = cal.abs_to_z_raw(1, act1)
            z4_raw = cal.abs_to_z_raw(4, act4)
            zid_raw = z1_raw + z4_raw

            cal.b14 = float(zid_raw - z0_raw)
            self._axis_cal_set_field_status(["b14"], "已标定/未写入")
            self._set_axis_cal(cal)
            self._axis_cal_to_ui(cal)
            self.axis_cal_refresh_status()
            print(
                "[axis_cal] calibrate B14: "
                f"z0_raw={z0_raw:.6f} zid_raw={zid_raw:.6f} -> b14={cal.b14:.6f}"
            )
        except Exception as e:
            print(f"[axis_cal] calibrate B14 failed: {e}")

    def axis_cal_calibrate_keepout(self) -> None:
        """Calibrate AX2 keepout parameters (b2, keepout_w) based on current AX0/AX1/AX2 positions.

        Intended workflow:
        - Move AX2 (center clamp) to the working position you want to bind.
        - Move AX0 to one keepout boundary and AX1 (or AX1+AX4 combined) to the other boundary.
        - Click this button to compute:
            keepout_w = (z_high - z_low) / 2
            b2        = z_center - z2_raw

        Notes:
        - Uses Z_raw coordinates.
        - Only updates IPC UI/in-memory values. Use "Write" to persist to PLC.
        """
        try:
            cal = self._axis_cal_from_ui()

            act0 = float(self.get_axis_copy(0).act_pos)
            act1 = float(self.get_axis_copy(1).act_pos)
            act2 = float(self.get_axis_copy(2).act_pos)

            z0_raw = cal.abs_to_z_raw(0, act0)
            z1_raw = cal.abs_to_z_raw(1, act1)
            z2_raw = cal.abs_to_z_raw(2, act2)

            z_low = min(z0_raw, z1_raw)
            z_high = max(z0_raw, z1_raw)

            zc = 0.5 * (z_low + z_high)
            w = 0.5 * (z_high - z_low)

            cal.keepout_w = float(abs(w))
            cal.b2 = float(zc - z2_raw)

            self._axis_cal_set_field_status(["b2", "keepout_w"], "已标定/未写入")
            self._set_axis_cal(cal)
            self._axis_cal_to_ui(cal)
            self.axis_cal_refresh_status()

            print(
                "[axis_cal] calibrate keepout: "
                f"z0_raw={z0_raw:.6f} z1_raw={z1_raw:.6f} z2_raw={z2_raw:.6f} "
                f"-> z_low={z_low:.6f} z_high={z_high:.6f} "
                f"b2={cal.b2:.6f} keepout_w={cal.keepout_w:.6f}"
            )
        except Exception as e:
            print(f"[axis_cal] calibrate keepout failed: {e}")

    def axis_cal_set_zpos_zero(self) -> None:
        """Set IPC-only z_pos so that current OD plane shows Z_disp == 0.

        z_pos is defined as a UI shift:
            z_disp = z_raw - z_pos
        Thus setting z_pos = current z_od_raw makes current z_od_disp == 0.
        """
        try:
            cal = self._axis_cal_from_ui()
            act0 = float(self.get_axis_copy(0).act_pos)
            z0_raw = cal.abs_to_z_raw(0, act0)
            cal.z_pos = float(z0_raw)
            self._axis_cal_set_field_status(["z_pos"], "已设置/未写入")
            self._set_axis_cal(cal)
            self._axis_cal_to_ui(cal)
            self.axis_cal_refresh_status()
            print(f"[axis_cal] set z_pos: z_pos={cal.z_pos:.6f} (OD disp -> 0)")
        except Exception as e:
            print(f"[axis_cal] set z_pos failed: {e}")

    def axis_cal_refresh_status(self) -> None:
        """Refresh read-only display block on the AxisCal screen.

        This is a pure UI helper that computes current Z_raw/Z_disp from snapshots.
        It is safe to call frequently.
        """
        v = getattr(self, "axis_cal_status_vars", None)
        if not isinstance(v, dict):
            return

        try:
            cal = self._axis_cal_from_ui()
        except Exception:
            cal = getattr(self, "axis_cal", AxisCal())

        try:
            act0 = float(self.get_axis_copy(0).act_pos)
            act1 = float(self.get_axis_copy(1).act_pos)
            act2 = float(self.get_axis_copy(2).act_pos)
            act4 = float(self.get_axis_copy(4).act_pos)
        except Exception:
            return

        # Current Z_raw
        z0_raw = cal.abs_to_z_raw(0, act0)
        z1_raw = cal.abs_to_z_raw(1, act1)
        z2_raw = cal.abs_to_z_raw(2, act2)
        z4_raw = cal.abs_to_z_raw(4, act4)
        zid_raw = z1_raw + z4_raw

        # Current Z_disp
        z0_disp = cal.z_raw_to_z_disp(z0_raw)
        zid_disp = cal.z_raw_to_z_disp(zid_raw)

        # Alignment check (OD/ID planes)
        # When aligned: (Z_id_raw - Z_od_raw) ~= B14
        delta = (zid_raw - z0_raw) - float(cal.b14)
        tol = 0.50  # mm, pragmatic default
        aligned = abs(delta) <= tol

        # PLC keepout parameter display (not used for IPC motion clamping)
        z_center = z2_raw + float(getattr(cal, 'b2', 0.0))
        w = float(getattr(cal, 'keepout_w', 0.0))
        z_low_k = z_center - w
        z_high_k = z_center + w
        z_low_disp = cal.z_raw_to_z_disp(z_low_k)
        z_high_disp = cal.z_raw_to_z_disp(z_high_k)

        try:
            v["off_abs"].set(
                "已标定 Off(abs@Z_raw=0): "
                f"AX0={cal.off_ax0:.3f}  AX1={cal.off_ax1:.3f}  AX2={cal.off_ax2:.3f}  AX4={cal.off_ax4:.3f}"
            )
            v["act_abs"].set(
                "当前 Act_Pos(abs): "
                f"AX0={act0:.3f}  AX1={act1:.3f}  AX2={act2:.3f}  AX4={act4:.3f}"
            )
            # Soft limits (absolute position) are now polled inside AXIS_Ctrl block:
            #   Softlim_pos @ OFF 52 (D152 for AX0)
            #   Softlim_neg @ OFF 56 (D156 for AX0)
            try:
                pos_parts = []
                neg_parts = []
                for ax in LINEAR_AXES:
                    ac = self.get_axis_copy(ax)
                    try:
                        p = float(getattr(ac, "softlim_pos", float("nan")))
                        n = float(getattr(ac, "softlim_neg", float("nan")))
                        if p == p and n == n:  # not NaN
                            pos_parts.append(f"AX{ax}={p:.3f}")
                            neg_parts.append(f"AX{ax}={n:.3f}")
                        else:
                            pos_parts.append(f"AX{ax}=--")
                            neg_parts.append(f"AX{ax}=--")
                    except Exception:
                        pos_parts.append(f"AX{ax}=--")
                        neg_parts.append(f"AX{ax}=--")
                v["softlim_pos"].set("软限位+(abs): " + "  ".join(pos_parts) if pos_parts else "软限位+(abs): -")
                v["softlim_neg"].set("软限位-(abs): " + "  ".join(neg_parts) if neg_parts else "软限位-(abs): -")
            except Exception:
                pass

            v["z_raw"].set(
                "当前 Z_raw(mm): "
                f"Z0={z0_raw:.3f}  Z1={z1_raw:.3f}  Z2={z2_raw:.3f}  Z4={z4_raw:.3f}  Zid={zid_raw:.3f}"
            )
            v["keepout_raw"].set(
                "避让区 Z_raw(mm): "
                f"Zc={z_center:.3f}  W={w:.3f}  z_low={z_low_k:.3f}  z_high={z_high_k:.3f}"
            )
            v["keepout_disp"].set(
                "避让区 Z_disp(mm): "
                f"z_low={z_low_disp:.3f}  z_high={z_high_disp:.3f}"
            )


            if aligned:
                v["z_disp"].set(
                    "当前 Z_disp(mm): "
                    f"Zod={z0_disp:.3f}  Zid={zid_disp:.3f}  (Δ={(delta):+.3f}mm)"
                )
            else:
                v["z_disp"].set(
                    "OD与ID测量截面未对齐  "
                    f"(Δ={(delta):+.3f}mm)  "
                    f"Zod_disp={z0_disp:.3f}  Zid_disp={zid_disp:.3f}"
                )
        except Exception:
            # Never crash UI for status updates
            pass
