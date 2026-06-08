from __future__ import annotations

import threading
import time
from typing import Any, Optional, Tuple


from config.addresses import (
    CMD_STOP_REQ,
    CMD_VELMOVE_REQ,
    OFF_VEL_VELMOVE,
)
from core.models import Recipe
from utils.logger import log

__all__ = ["ExecutorLengthMixin"]


def _len_wait_new_gauge(self, gw, ts0: float, tmax: float = 1.5) -> Optional[Tuple[float, object]]:
    """Wait for a new gauge sample (valid numeric)."""
    t_wait0 = time.time()
    last_ts = float(ts0)
    while (not self._should_stop()) and ((time.time() - t_wait0) < float(tmax)):
        try:
            gw.send_request()
        except Exception:
            pass
        time.sleep(0.06)
        s = None
        try:
            s = gw.get_last()
        except Exception:
            s = None
        if s is None:
            continue
        ts = float(getattr(s, "ts", 0.0) or 0.0)
        if ts > last_ts:
            return ts, s
    return None


def _len_wait_axis_settled(self, tmax: float = 1.5) -> bool:
    """Wait until AX0 is not in stopping transient (warn=1003) and position is stable."""
    t_s0 = time.time()
    last_p = None
    stable = 0
    while (not self._should_stop()) and ((time.time() - t_s0) < float(tmax)):
        ac = self.device.get_axis_copy(0)
        try:
            warn = int(getattr(ac, 'warn', 0) or 0)
        except Exception:
            warn = 0
        try:
            p = float(getattr(ac, 'act_pos', 0.0) or 0.0)
        except Exception:
            p = 0.0
        if warn == 1003:
            stable = 0
            last_p = p
            time.sleep(0.05)
            continue
        if last_p is None:
            last_p = p
            stable = 0
            time.sleep(0.05)
            continue
        if abs(p - last_p) <= 0.02:
            stable += 1
        else:
            stable = 0
        last_p = p
        if stable >= 8:
            return True
        time.sleep(0.05)
    return False


class ExecutorLengthMixin:
    app: Any
    device: Any
    stop_event: threading.Event

    # Methods called from other mixins (cooperative MRO)
    _is_enabled: Any
    _is_fault: Any
    _should_stop: Any
    _write_fp64: Any
    _wait_in_position: Any

    def _auto_measure_length(self, recipe: Recipe) -> dict:
        """Run length measurement inside auto flow.

        Returns a dict payload for UI/telemetry:
            {
              enabled: bool,
              skipped: bool,
              ok: bool,
              reason: str,
              z_low: float|None,
              z_high: float|None,
              length_mm: float|None,
              t_s: float,
            }

        Notes:
        - Failure must NOT break the overall auto flow.
        - Uses gauge "judge" (GO -> HI) for bottom edge.
        - For top edge we also use judge GO->HI, and lock the last GO position as the edge.
        """

        t0 = time.time()
        payload: dict = {
            "enabled": bool(getattr(recipe, "len_enable", False)),
            "skipped": False,
            "ok": False,
            "reason": "",
            "z_low": None,
            "z_high": None,
            "length_mm": None,
            "t_s": 0.0,
        }

        if not bool(getattr(recipe, "len_enable", False)):
            payload["skipped"] = True
            payload["reason"] = "DISABLED"
            payload["t_s"] = time.time() - t0
            return payload

        # prerequisites
        cal = getattr(self.app, "axis_cal", None)
        gw = getattr(self.app, "gauge_worker", None)
        if cal is None:
            payload["reason"] = "NO_AXIS_CAL"
            payload["t_s"] = time.time() - t0
            return payload
        if gw is None:
            payload["reason"] = "NO_GAUGE"
            payload["t_s"] = time.time() - t0
            return payload

        try:
            z_min, z_max, _travel = self.app._get_ax0_z_disp_limits()
        except Exception:
            # safe fallback
            z_min, z_max = -1e9, 1e9

        # parameters
        abs_low_approach = float(getattr(recipe, "len_low_approach_abs", 0.0) or 0.0)
        # Convert to Z_disp for travel checks / reporting
        z_low_approach = float(cal.abs_to_z_disp(0, abs_low_approach))
        d_low = float(getattr(recipe, "len_low_search_dist", 60.0) or 0.0)
        d_low = max(0.0, d_low)
        d_high = float(getattr(recipe, "len_high_search_dist", 60.0) or 0.0)
        d_high = max(0.0, d_high)
        # NOTE: Recipe uses `pipe_len_mm` (not `pipe_len`). Using the wrong field
        # will silently make auto length measurement run with 0.0mm and always fail.
        pipe_len = float(getattr(recipe, "pipe_len_mm", 0.0) or 0.0)
        hi_margin = float(getattr(recipe, "len_high_margin", 0.0) or 0.0)
        v_z = abs(float(getattr(recipe, "len_search_vel", 10.0) or 10.0))
        timeout_s = max(1.0, float(getattr(recipe, "len_search_timeout_s", 8.0) or 8.0))
        backoff_mm = max(0.0, float(getattr(recipe, "len_backoff_mm", 0.0) or 0.0))
        deb_k = max(1, int(float(getattr(recipe, "len_debounce_k", 2) or 2)))

        # feasibility check: rough Lmax based on travel + max search windows
        try:
            z_low_edge_max = min(float(z_max), float(z_low_approach) + float(d_low))
            lmax = float(z_low_edge_max) + float(hi_margin) - float(d_high) - float(z_min)
            if pipe_len > (lmax + 1.0):
                payload["skipped"] = True
                payload["reason"] = f"TOO_LONG(max≈{lmax:.1f}mm)"
                payload["t_s"] = time.time() - t0
                return payload
        except Exception:
            pass

        # axis state check
        ac0 = self.device.get_axis_copy(0)
        if (not self._is_enabled(int(getattr(ac0, "sts", 0)))) or self._is_fault(
            int(getattr(ac0, "sts", 0)), int(getattr(ac0, "err", 0))
        ):
            payload["reason"] = f"AX0_NOT_READY(err={int(getattr(ac0,'err',0) or 0)})"
            payload["t_s"] = time.time() - t0
            return payload

        # ---------------- bottom edge: GO -> HI -> GO (bidirectional average) ----------------
        def _scan_edge_bidirectional(
            z_start: float,
            dir_sign: int,
            max_dist: float,
            last_ts0: float,
            label: str,
        ) -> Tuple[Optional[float], float, str]:
            """Scan in Z_disp direction to find GO->HI, then reverse to find HI->GO; return averaged edge.

            dir_sign: +1 means +Z_disp, -1 means -Z_disp
            Returns: (edge_avg, last_ts, reason)
            """
            last_ts = float(last_ts0)
            # ---------- Pass 1: GO -> HI ----------
            edge1: Optional[float] = None
            t_search0 = time.time()
            unk_cnt = 0
            hi_cnt = 0
            last_go_z = float(z_start)

            # Motion guard (avoid waiting to timeout if axis is stuck)
            last_move_z: Optional[float] = None
            last_move_ts: float = time.time()

            def _axis_not_moving(z_cur: float) -> bool:
                nonlocal last_move_z, last_move_ts
                if last_move_z is None:
                    last_move_z = float(z_cur)
                    last_move_ts = time.time()
                    return False
                if abs(float(z_cur) - float(last_move_z)) >= 0.15:
                    last_move_z = float(z_cur)
                    last_move_ts = time.time()
                    return False
                return (time.time() - float(last_move_ts)) >= 1.0

            vel_abs = float(v_z) * float(dir_sign) * float(cal.sign_eff(0))
            try:
                self.app._velmove_start_axis(0, vel_abs, acc=80.0, dec=80.0, jerk=300.0)
            except Exception:
                self._write_fp64(0, OFF_VEL_VELMOVE, vel_abs)
                self.app.set_cmd_bits(0, set_mask=CMD_VELMOVE_REQ, clr_mask=0)

            while not self._should_stop():
                ac0 = self.device.get_axis_copy(0)
                z_cur = float(cal.abs_to_z_disp(0, ac0.act_pos))

                dist = (z_cur - float(z_start)) if int(dir_sign) > 0 else (float(z_start) - z_cur)
                if float(max_dist) > 0.0 and dist >= (float(max_dist) - 1e-6):
                    return None, last_ts, f"{label}_NOT_FOUND_MAXDIST_P1"
                if (time.time() - t_search0) >= float(timeout_s):
                    return None, last_ts, f"{label}_NOT_FOUND_TIMEOUT_P1"
                if self._is_fault(int(getattr(ac0, 'sts', 0)), int(getattr(ac0, 'err', 0))):
                    return None, last_ts, f"{label}_AX0_FAULT(err={int(getattr(ac0,'err',0) or 0)})"
                if _axis_not_moving(z_cur):
                    return None, last_ts, f"{label}_AX0_NOT_MOVING"

                r = _len_wait_new_gauge(self, gw, last_ts, 0.35)
                if r is None:
                    continue
                last_ts, s = r
                j = str(getattr(s, "judge", "UNK") or "UNK").strip().upper()
                if j == "UNK":
                    unk_cnt += 1
                    if unk_cnt >= 8:
                        return None, last_ts, f"{label}_JUDGE_UNK"
                    continue
                unk_cnt = 0

                if j == "GO":
                    last_go_z = float(z_cur)
                    hi_cnt = 0
                    continue

                if j in ("HI", "HH"):
                    hi_cnt += 1
                    if hi_cnt >= int(deb_k):
                        edge1 = float(last_go_z)
                        break
                else:
                    hi_cnt = 0

            # stop always
            try:
                self.device.stop(0)
            except Exception:
                try:
                    self.app.set_cmd_bits(0, set_mask=0, clr_mask=CMD_VELMOVE_REQ)
                    self.app._pulse_cmd_bits(0, CMD_STOP_REQ)
                except Exception:
                    pass

            if self._should_stop():
                return None, last_ts, "ABORT"

            if edge1 is None:
                return None, last_ts, f"{label}_NOT_FOUND_P1"

            # ---------- Pass 2: HI -> GO ----------
            _len_wait_axis_settled(self, 1.5)
            time.sleep(0.05)
            z_start2 = float(cal.abs_to_z_disp(0, self.device.get_axis_copy(0).act_pos))
            vel_abs2 = -float(v_z) * float(dir_sign) * float(cal.sign_eff(0))
            try:
                self.app._velmove_start_axis(0, vel_abs2, acc=80.0, dec=80.0, jerk=300.0)
            except Exception:
                self._write_fp64(0, OFF_VEL_VELMOVE, vel_abs2)
                self.app.set_cmd_bits(0, set_mask=CMD_VELMOVE_REQ, clr_mask=0)

            t_search1 = time.time()
            unk_cnt = 0
            go_cnt = 0
            seen_hi = False
            last_hi_z = float(z_start2)
            edge2: Optional[float] = None
            last_move_z = None
            last_move_ts = time.time()

            while not self._should_stop():
                ac0 = self.device.get_axis_copy(0)
                z_cur = float(cal.abs_to_z_disp(0, ac0.act_pos))

                dist = (float(z_start2) - z_cur) if int(dir_sign) > 0 else (z_cur - float(z_start2))
                if float(max_dist) > 0.0 and dist >= (float(max_dist) - 1e-6):
                    return None, last_ts, f"{label}_NOT_FOUND_MAXDIST_P2"
                if (time.time() - t_search1) >= float(timeout_s):
                    return None, last_ts, f"{label}_NOT_FOUND_TIMEOUT_P2"
                if self._is_fault(int(getattr(ac0, 'sts', 0)), int(getattr(ac0, 'err', 0))):
                    return None, last_ts, f"{label}_AX0_FAULT(err={int(getattr(ac0,'err',0) or 0)})"
                if _axis_not_moving(z_cur):
                    return None, last_ts, f"{label}_AX0_NOT_MOVING"

                r = _len_wait_new_gauge(self, gw, last_ts, 0.35)
                if r is None:
                    continue
                last_ts, s = r
                j = str(getattr(s, "judge", "UNK") or "UNK").strip().upper()
                if j == "UNK":
                    unk_cnt += 1
                    if unk_cnt >= 8:
                        return None, last_ts, f"{label}_JUDGE_UNK"
                    continue
                unk_cnt = 0

                if not seen_hi:
                    if j in ("HI", "HH"):
                        seen_hi = True
                        last_hi_z = float(z_cur)
                    continue

                if j in ("HI", "HH"):
                    last_hi_z = float(z_cur)
                    go_cnt = 0
                    continue

                if j == "GO":
                    go_cnt += 1
                    if go_cnt >= int(deb_k):
                        edge2 = float(last_hi_z)
                        break
                else:
                    go_cnt = 0

            # stop always
            try:
                self.device.stop(0)
            except Exception:
                try:
                    self.app.set_cmd_bits(0, set_mask=0, clr_mask=CMD_VELMOVE_REQ)
                    self.app._pulse_cmd_bits(0, CMD_STOP_REQ)
                except Exception:
                    pass

            if self._should_stop():
                return None, last_ts, "ABORT"

            if edge2 is None:
                return None, last_ts, f"{label}_NOT_FOUND_P2"

            edge_avg = 0.5 * (float(edge1) + float(edge2))
            return edge_avg, last_ts, "OK"

        z_low_edge: Optional[float] = None
        try:
            z_appr = max(float(z_min), min(float(z_max), float(z_low_approach)))
            abs_tgt = float(cal.z_disp_to_abs(0, z_appr))
            self.device.movea_abs(0, abs_tgt, context="AutoLenLowAppr")
            if not self._wait_in_position(0, abs_tgt, pos_tol=1.0, timeout_s=15.0):
                payload["reason"] = "LOW_APPR_TIMEOUT"
                payload["t_s"] = time.time() - t0
                return payload

            # make sure gauge is updating
            r = _len_wait_new_gauge(self, gw, 0.0, 1.5)
            if r is None:
                payload["reason"] = "LOW_NO_GAUGE"
                payload["t_s"] = time.time() - t0
                return payload
            last_ts, last_s = r
            judge0 = str(getattr(last_s, "judge", "UNK") or "UNK").strip().upper()
            if judge0 == "UNK":
                payload["reason"] = "LOW_JUDGE_UNK"
                payload["t_s"] = time.time() - t0
                return payload
            if judge0 != "GO":
                payload["reason"] = f"LOW_START_NOT_GO({judge0})"
                payload["t_s"] = time.time() - t0
                return payload

            z_start = float(cal.abs_to_z_disp(0, self.device.get_axis_copy(0).act_pos))
            edge_avg, last_ts, reason_scan = _scan_edge_bidirectional(
                z_start=z_start,
                dir_sign=+1,
                max_dist=float(d_low),
                last_ts0=float(last_ts),
                label="LOW",
            )
            if reason_scan != "OK" or edge_avg is None:
                payload["reason"] = str(reason_scan)
                payload["t_s"] = time.time() - t0
                return payload

            z_low_edge = float(edge_avg)
            payload["z_low"] = z_low_edge

            # optional backoff towards -Z_disp (inside tube)
            if backoff_mm > 1e-6:
                try:
                    z_back = max(float(z_min), min(float(z_max), float(z_low_edge) - float(backoff_mm)))
                    self.device.movea_abs(0, float(cal.z_disp_to_abs(0, z_back)), context="AutoLenLowBackoff")
                    self._wait_in_position(0, float(cal.z_disp_to_abs(0, z_back)), pos_tol=1.2, timeout_s=10.0)
                except Exception:
                    pass

            if self._should_stop():
                payload["reason"] = "ABORT"
                payload["t_s"] = time.time() - t0
                return payload

        except Exception as e:
            payload["reason"] = f"LOW_EXC({e})"
            try:
                self.device.stop(0)
            except Exception:
                pass
            payload["t_s"] = time.time() - t0
            return payload


        # ---------------- top edge: GO -> HI -> GO (bidirectional average) ----------------
        if pipe_len <= 1e-6:
            payload["reason"] = "PIPE_LEN_ZERO"
            payload["t_s"] = time.time() - t0
            return payload

        z_high_edge: Optional[float] = None
        try:
            z_appr = float(z_low_edge - pipe_len + hi_margin)
            z_appr = max(float(z_min), min(float(z_max), float(z_appr)))
            abs_tgt = float(cal.z_disp_to_abs(0, z_appr))
            self.device.movea_abs(0, abs_tgt, context="AutoLenHighAppr")
            if not self._wait_in_position(0, abs_tgt, pos_tol=1.0, timeout_s=15.0):
                payload["reason"] = "HIGH_APPR_TIMEOUT"
                payload["t_s"] = time.time() - t0
                return payload

            # pre-check: gauge must be updating and should be GO at approach (inside tube)
            r = _len_wait_new_gauge(self, gw, 0.0, 1.5)
            if r is None:
                payload["reason"] = "HIGH_NO_GAUGE"
                payload["t_s"] = time.time() - t0
                return payload
            last_ts, _s = r
            j0 = str(getattr(_s, "judge", "UNK") or "UNK").strip().upper()
            if j0 == "UNK":
                payload["reason"] = "HIGH_JUDGE_UNK"
                payload["t_s"] = time.time() - t0
                return payload
            if j0 != "GO":
                payload["reason"] = f"HIGH_START_NOT_GO({j0})"
                payload["t_s"] = time.time() - t0
                return payload

            z_start = float(cal.abs_to_z_disp(0, self.device.get_axis_copy(0).act_pos))
            edge_avg, last_ts, reason_scan = _scan_edge_bidirectional(
                z_start=z_start,
                dir_sign=-1,
                max_dist=float(d_high),
                last_ts0=float(last_ts),
                label="HIGH",
            )
            if reason_scan != "OK" or edge_avg is None:
                payload["reason"] = str(reason_scan)
                payload["t_s"] = time.time() - t0
                return payload

            z_high_edge = float(edge_avg)
            payload["z_high"] = z_high_edge

            # optional backoff towards +Z_disp (inside tube)
            if backoff_mm > 1e-6:
                try:
                    z_back = max(float(z_min), min(float(z_max), float(z_high_edge) + float(backoff_mm)))
                    self.device.movea_abs(0, float(cal.z_disp_to_abs(0, z_back)), context="AutoLenHighBackoff")
                    self._wait_in_position(0, float(cal.z_disp_to_abs(0, z_back)), pos_tol=1.2, timeout_s=10.0)
                except Exception:
                    pass

            if self._should_stop():
                payload["reason"] = "ABORT"
                payload["t_s"] = time.time() - t0
                return payload

        except Exception as e:
            payload["reason"] = f"HIGH_EXC({e})"
            try:
                self.device.stop(0)
            except Exception:
                pass
            payload["t_s"] = time.time() - t0
            return payload

# compute length
        try:
            length_mm = float(z_low_edge - z_high_edge)
            payload["length_mm"] = length_mm
            payload["ok"] = True
            payload["reason"] = "OK"
        except Exception:
            payload["reason"] = "LEN_CALC_FAIL"

        payload["t_s"] = time.time() - t0

        # lightweight telemetry for debugging
        try:
            log(
                "AUTO_LEN_RESULT",
                ok=bool(payload.get("ok")),
                reason=str(payload.get("reason", "")),
                z_low=payload.get("z_low"),
                z_high=payload.get("z_high"),
                length_mm=payload.get("length_mm"),
                t_s=payload.get("t_s"),
            )
        except Exception:
            pass
        return payload
