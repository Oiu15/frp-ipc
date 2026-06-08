from __future__ import annotations

import math
import time
from typing import Any, List, Mapping, Tuple

import numpy as np
from utils.perf import PerfAggregator, ns_to_ms

from core.models import Recipe
from domain.sampling import (
    _max_gap_deg_from_bins,
)
from frp_workflow.executor import _executor_helpers
from frp_workflow.executor._executor_helpers import log, perf_logger, logger


def _t_avg_max_ms(snap, key: str) -> tuple[float, float]:
    """Extract average and max ms from a PerfAggregator snapshot time entry."""
    st = snap.times.get(key)
    if st is None or st.n <= 0:
        return 0.0, 0.0
    return (ns_to_ms(int(st.sum_ns)) / float(st.n), ns_to_ms(int(st.max_ns)))


def _flush_sample_perf_if_due(raw_total: int, cov_bins: int, perf, *, force: bool = False) -> None:
    """Drain and log sampling performance metrics when due."""
    snap = perf.drain_if_due(every_s=1.0, force=bool(force))
    if snap is None:
        return
    c = snap.counts
    loop_avg_ms, loop_max_ms = _t_avg_max_ms(snap, "loop")
    theta_avg_ms, theta_max_ms = _t_avg_max_ms(snap, "theta")
    od_send_avg_ms, od_send_max_ms = _t_avg_max_ms(snap, "od_send")
    od_wait_avg_ms, od_wait_max_ms = _t_avg_max_ms(snap, "od_wait")
    id145_avg_ms, id145_max_ms = _t_avg_max_ms(snap, "id145")
    id3_avg_ms, id3_max_ms = _t_avg_max_ms(snap, "id3")
    append_avg_ms, append_max_ms = _t_avg_max_ms(snap, "append")
    theta_t = snap.times.get("theta")
    od_send_t = snap.times.get("od_send")
    od_wait_t = snap.times.get("od_wait")
    id145_t = snap.times.get("id145")
    id3_t = snap.times.get("id3")
    append_t = snap.times.get("append")
    theta_n = int(theta_t.n) if theta_t is not None else 0
    od_send_n = int(od_send_t.n) if od_send_t is not None else 0
    od_wait_n = int(od_wait_t.n) if od_wait_t is not None else 0
    id145_n = int(id145_t.n) if id145_t is not None else 0
    id3_n = int(id3_t.n) if id3_t is not None else 0
    append_n = int(append_t.n) if append_t is not None else 0
    try:
        perf_logger.info(
            "[AUTOFLOW_PERF] loops=%d loop_avg_ms=%.3f loop_max_ms=%.3f "
            "theta_n=%d theta_avg_ms=%.3f theta_max_ms=%.3f "
            "od_send_n=%d od_send_avg_ms=%.3f od_send_max_ms=%.3f "
            "od_wait_n=%d od_wait_avg_ms=%.3f od_wait_max_ms=%.3f "
            "id145_n=%d id145_avg_ms=%.3f id145_max_ms=%.3f "
            "id3_n=%d id3_avg_ms=%.3f id3_max_ms=%.3f "
            "append_n=%d append_avg_ms=%.3f append_max_ms=%.3f "
            "od_req=%d od_new=%d id_ok=%d append=%d raw_total=%d cov=%d "
            "skip_no_new_od=%d skip_od_outlier=%d skip_id_none=%d skip_id_outlier=%d dedup=%d skip_sync_mismatch=%d",
            int(c.get("loops", 0)),
            float(loop_avg_ms),
            float(loop_max_ms),
            int(theta_n),
            float(theta_avg_ms),
            float(theta_max_ms),
            int(od_send_n),
            float(od_send_avg_ms),
            float(od_send_max_ms),
            int(od_wait_n),
            float(od_wait_avg_ms),
            float(od_wait_max_ms),
            int(id145_n),
            float(id145_avg_ms),
            float(id145_max_ms),
            int(id3_n),
            float(id3_avg_ms),
            float(id3_max_ms),
            int(append_n),
            float(append_avg_ms),
            float(append_max_ms),
            int(c.get("od_req", 0)),
            int(c.get("od_new", 0)),
            int(c.get("id_ok", 0)),
            int(c.get("append", 0)),
            int(raw_total),
            int(cov_bins),
            int(c.get("skip_no_new_od", 0)),
            int(c.get("skip_od_outlier", 0)),
            int(c.get("skip_id_none", 0)),
            int(c.get("skip_id_outlier", 0)),
            int(c.get("dedup", 0)),
            int(c.get("skip_sync_mismatch", 0)),
        )
    except Exception:
        pass


def _loop_done(t_loop0_ns: int, raw_total: int, cov_bins: int, perf) -> None:
    """Record loop timing and flush perf snapshot if due."""
    perf.add_time_ns("loop", time.perf_counter_ns() - int(t_loop0_ns))
    _flush_sample_perf_if_due(raw_total, cov_bins, perf)


def _pp_trim_list(lst, trim_ratio: float = 0.01) -> float:
    """Trimmed peak-to-peak range (remove top/bottom trim_ratio fraction)."""
    if not lst or len(lst) < 2:
        return 0.0
    b0 = sorted([float(x) for x in lst])
    m0 = len(b0)
    k0 = int(max(0, math.floor(float(trim_ratio) * m0)))
    if (2 * k0) >= (m0 - 1):
        k0 = 0
    return float(b0[m0 - 1 - k0] - b0[k0])


class ExecutorSamplingMixin:
    """Mixin providing equal-angle sampling (OD/ID circle-point acquisition).

    Expects the following attributes/methods on ``self``:
        app: Any
        device: Any
        stop_event: Any
        _current_recipe: Any
        _calibration_snapshot: Any

        # Methods called from other mixins:
        # self._should_stop() -> bool           (core)
        # self._sleep_cancelable(s) -> bool     (core)
        # self._emit_auto_state(state, msg)     (core)
        # self._is_fault(sts, err) -> bool      (motion)
        # self._is_enabled(sts) -> bool         (motion)
        # self._is_moving(sts) -> bool          (motion)
        # self._write_fp64(axis, off, val)      (motion)
        # self._ensure_velmove_setpoints(...)   (motion)
        # self._wait_in_position(...)           (motion)
        # self._id_fit_from_raw_points(...)     (fitting)
        # self._od_round_fit_from_raw_points()  (fitting)
        # self._id_round_fit_from_raw_points()  (fitting)
    """

    app: Any
    device: Any
    stop_event: Any
    _current_recipe: Any
    _calibration_snapshot: Any

    # Dunder attrs set internally:
    _last_sample_cov: Any
    _last_sample_reason: Any
    _last_sample_max_gap_deg: Any
    _last_sample_n_od: Any
    _last_sample_n_id: Any
    _last_fit_weights_od: Any
    _last_fit_weights_id: Any
    _last_sample_debug: Any

    def _sample_circle_points_dual(
        self,
        recipe: Recipe,
        section_idx: int = 0,
        *,
        sample_od: bool = True,
        sample_id: bool = True,
        phase: str = "SYNC",
    ) -> Tuple[np.ndarray, np.ndarray, str, str, list]:
        """Equal-angle sampling (OD/ID can be sampled independently).

        - Angle source: AX3 act_pos (deg)
        - OD source: gauge (real or simulated)
        - ID source: CL-3000 OUT3 (mapped via PLC) or simulated displacement meter

        Args:
            sample_od: sample OD in this pass
            sample_id: sample ID in this pass
            phase: tag written into raw_points for export/diagnostics (e.g. 'OD','ID','SYNC')

        Returns:
            (coords_od, coords_id, raw_last_od, raw_last_id, raw_points)
        """
        if (not sample_od) and (not sample_id):
            raise ValueError("sample_od and sample_id cannot both be False")

        n = max(3, int(getattr(recipe, "points_per_rev", 120)))
        min_cov = float(getattr(recipe, "min_bin_coverage", 0.95))
        min_cov = max(0.0, min(1.0, min_cov))
        timeout_s = float(getattr(recipe, "sample_timeout_s", 5.0))
        timeout_s = max(0.5, timeout_s)
        max_revs = float(getattr(recipe, "max_revolutions", 2.0))
        max_revs = max(0.25, max_revs)

        # speedtest: optionally skip ID reads to improve sampling throughput
        disable_id_modbus = (
            (
                bool(getattr(recipe, 'disable_id_modbus', False))
                or bool(_executor_helpers.SPEEDTEST_DISABLE_ID_MODBUS)
            )
            and (not bool(getattr(self.app, "sim_disp_enabled", False)))
            and bool(sample_id)
        )
        # In SPLIT mode, ID pass must not be disabled by OD-only switch.
        try:
            sm2 = str(getattr(recipe, "scan_mode", "sync") or "sync").strip().lower()
            if bool(sample_id) and sm2.startswith("split"):
                disable_id_modbus = False
        except Exception:
            pass
        # Single-probe ID rescue requires OUT2, so never disable ID reads in that mode.
        try:
            id_single_enable = bool(getattr(recipe, "id_single_enable", False))
        except Exception:
            id_single_enable = False
        if bool(sample_id) and id_single_enable:
            disable_id_modbus = False

        # Fit strategy affects sampling stop criteria.
        # - a: raw-point fit, keep all accepted raw points, bins only for coverage statistics.
        #      Sampling should run until max_revs (unless timeout/stop).
        # - b: raw-point fit with per-bin balancing weights. Can stop when coverage reached.
        # - c: per-bin radius averaging. Can stop when coverage reached.
        fs = str(getattr(recipe, "fit_strategy", "b 原始点按bin权重均衡") or "").strip().lower()
        mode = "b"
        if fs.startswith("a"):
            mode = "a"
        elif fs.startswith("b"):
            mode = "b"
        elif fs.startswith("c"):
            mode = "c"

        try:
            logger.info("[AX3_MODE] using latest-angle cache for OD+ID")
        except Exception:
            pass
        try:
            logger.info("[ID_MODE] using latest-cache for CL145/CL3")
        except Exception:
            pass

        # Reduce background polling during sampling to improve sync-read latency.
        self.app.set_plc_poll_profile("sampling")
        try:
            perf = PerfAggregator()
            od_req_total = 0
            od_new_total = 0
            id_ok_total = 0
            raw_append_total = 0
            skip_sync_mismatch = 0
            dedup_count = 0
            skip_gate_od_none = 0
            skip_gate_id_none = 0

            # 等角bin：将 0~360° 划分为 n 个bin
            sum_x_od = [0.0] * n
            sum_y_od = [0.0] * n
            sum_x_id = [0.0] * n
            sum_y_id = [0.0] * n
            cnt = [0] * n
            sum_r_od = [0.0] * n
            sum_r_id = [0.0] * n

            # debug counters
            iters = 0
            skip_no_new_od = 0
            skip_od_outlier = 0
            skip_id_none = 0
            skip_id_outlier = 0
            bin_fill_logs = 0

            od_min = None
            od_max = None
            id_min = None
            id_max = None
            filled = 0
            raw_last_od = ""
            raw_last_id = ""
            last_id_cnt4 = None  # gate duplicate CL OUT4 samples when using id_use_fit
            last_id_cnt2 = None  # gate duplicate CL OUT2 samples in single-probe mode

            # Raw sample points for export/diagnostics
            raw_points: list[dict] = []

            t_start = time.time()
            need = max(3, int(math.ceil(min_cov * n)))
            reason = "COV"  # COV / TIMEOUT / REV
            prev_theta = None
            unwrapped_deg = 0.0
            revs = 0.0

            while True:
                t_loop0_ns = time.perf_counter_ns()
                perf.add_count("loops", 1)
                if self._should_stop():
                    raise RuntimeError("测量被用户停止")

                iters += 1

                if mode != "a" and filled >= need:
                    reason = "COV"
                    _loop_done(t_loop0_ns, len(raw_points), filled, perf)
                    break
                if revs >= max_revs:
                    reason = "REV"
                    _loop_done(t_loop0_ns, len(raw_points), filled, perf)
                    break
                if (time.time() - t_start) >= timeout_s:
                    reason = "TIMEOUT"
                    _loop_done(t_loop0_ns, len(raw_points), filled, perf)
                    break

                # Angle snapshot first (deg)
                theta_deg = None
                t_theta0_ns = time.perf_counter_ns()
                try:
                    theta_deg = self.app._get_latest_ax3_angle_deg()
                except Exception:
                    theta_deg = None
                if theta_deg is None:
                    try:
                        theta_deg = self.device.read_axis_angle_deg_sync(axis=3, timeout_s=0.5)
                    except Exception:
                        theta_deg = None
                if theta_deg is None:
                    try:
                        a3 = self.device.get_axis_copy(3)
                        theta_deg = float(a3.act_pos) % 360.0
                    except Exception:
                        theta_deg = None
                perf.add_time_ns("theta", time.perf_counter_ns() - t_theta0_ns)
                if theta_deg is None:
                    _loop_done(t_loop0_ns, len(raw_points), filled, perf)
                    continue

                # unwrap to estimate revolutions (robust to wrap-around)
                if prev_theta is None:
                    prev_theta = float(theta_deg)
                else:
                    d = float(theta_deg) - float(prev_theta)
                    if d < -180.0:
                        d += 360.0
                    elif d > 180.0:
                        d -= 360.0
                    unwrapped_deg += abs(d)
                    prev_theta = float(theta_deg)
                    revs = unwrapped_deg / 360.0
                theta = math.radians(float(theta_deg))

                # map to bin
                b = int((theta_deg / 360.0) * n)
                if b >= n:
                    b = 0

                # ---------------- OD ----------------
                od = None
                od_out1 = None
                od_out2 = None
                od_B = None
                od_map_out1 = 'L'
                od_L = None
                od_R = None
                od_delta = None

                if sample_od:
                    if self.app.sim_gauge_enabled:
                        od_val, raw = self.app.simulate_gauge_once(recipe)
                        od = float(od_val)
                        od_out1 = float(od)
                        raw_last_od = raw
                    else:
                        gw = self.app.gauge_worker
                        if gw is None:
                            raise RuntimeError("测径仪未启用：请勾选“模拟测径仪”或连接真实串口。")

                        # Ensure OUT1/OUT2 are available for edge-based OD algorithm
                        if bool(getattr(recipe, 'od_use_edges', False)):
                            try:
                                req_cmd = str(getattr(gw, 'request_cmd', '') or '').strip().upper()
                                if not req_cmd.startswith('M0'):
                                    gw.configure(
                                        enabled=gw.enabled,
                                        port=gw.port,
                                        baud=gw.baud,
                                        timeout_s=gw.timeout_s,
                                        eol=gw.eol,
                                        request_cmd='M0,1',
                                        bytesize=gw.bytesize,
                                        parity=gw.parity,
                                        stopbits=gw.stopbits,
                                    )
                            except Exception:
                                pass

                        t_req = time.time()
                        t_od_send0_ns = time.perf_counter_ns()
                        gw.send_request()
                        perf.add_time_ns("od_send", time.perf_counter_ns() - t_od_send0_ns)
                        perf.add_count("od_req", 1)
                        od_req_total += 1
                        t0 = time.time()
                        t_od_wait0_ns = time.perf_counter_ns()
                        while (time.time() - t0) < 1.2:
                            if self._should_stop():
                                raise RuntimeError("测量被用户停止")
                            s = gw.get_last()
                            if s and s.ts >= t_req:
                                perf.add_count("od_new", 1)
                                od_new_total += 1
                                od_use_edges = bool(getattr(recipe, 'od_use_edges', False))
                                od_std = float(getattr(recipe, 'od_std_mm', 0.0) or 0.0)
                                od_tol = float(getattr(recipe, 'od_tol_mm', 0.0) or 0.0)
                                out1 = float(s.od)
                                out2 = None
                                try:
                                    out2 = None if getattr(s, 'od2', None) is None else float(getattr(s, 'od2'))
                                except Exception:
                                    out2 = None

                                B_active = None
                                map_out1 = 'L'
                                L_val = None
                                R_val = None
                                delta_val = None

                                if od_use_edges:
                                    calibration = self._get_calibration_snapshot()
                                    try:
                                        b_val = float(calibration.od_b_active_mm)
                                        if math.isfinite(b_val) and abs(b_val) > 1e-12:
                                            B_active = b_val
                                    except Exception:
                                        B_active = None
                                    try:
                                        map_out1 = str(calibration.od_out1_map or 'L').strip().upper() or 'L'
                                        if map_out1 not in ('L', 'R'):
                                            map_out1 = 'L'
                                    except Exception:
                                        map_out1 = 'L'

                                    if B_active is None:
                                        raise RuntimeError('新外径算法(边缘距离)需要先标定B值')
                                    if out2 is None:
                                        raise RuntimeError('新外径算法(边缘距离)需要同时读取OUT1/OUT2 (建议选择 M0,1)')

                                    if map_out1 == 'L':
                                        L_val, R_val = out1, out2
                                    else:
                                        L_val, R_val = out2, out1
                                    od_calc = float(B_active) - (float(L_val) + float(R_val))
                                    delta_val = 0.5 * (float(L_val) - float(R_val))
                                else:
                                    od_calc = out1

                                # plausibility filter: drop extreme outliers
                                if od_std > 0.0:
                                    margin = max(5.0, 10.0 * max(od_tol, 0.1), 0.05 * od_std)
                                    if (not math.isfinite(od_calc)) or abs(float(od_calc) - od_std) > margin:
                                        skip_od_outlier += 1
                                        perf.add_count("skip_od_outlier", 1)
                                        od = None
                                        break

                                od = float(od_calc)
                                od_out1 = out1
                                od_out2 = out2
                                od_B = B_active
                                od_map_out1 = map_out1
                                od_L = L_val
                                od_R = R_val
                                od_delta = delta_val
                                raw_last_od = s.raw
                                break
                            time.sleep(0.02)
                        else:
                            skip_no_new_od += 1
                            perf.add_count("skip_no_new_od", 1)
                            od = None
                        perf.add_time_ns("od_wait", time.perf_counter_ns() - t_od_wait0_ns)

                else:
                    raw_last_od = "OD_SKIPPED"

                if sample_od and (od is None):
                    skip_gate_od_none += 1
                    perf.add_count("skip_sync_mismatch", 1)
                    skip_sync_mismatch += 1
                    _loop_done(t_loop0_ns, len(raw_points), filled, perf)
                    continue

                # ---------------- ID ----------------
                id_mm = None
                cnt_i = None
                id_x1_mm = None
                id_x2_mm = None
                id_c_mm = None
                id_m_mm = None
                id_cnt_out4 = None
                id_cnt_out5 = None
                id_out2_mm = None
                id_cnt_out2 = None
                ecc_x = 0.0
                ecc_y = 0.0

                if sample_id:
                    if id_single_enable:
                        if getattr(self.app, "sim_disp_enabled", False):
                            id_val, raw_id = self.app.simulate_disp_once(recipe)
                            id_out2_mm = float(id_val) if id_val is not None else None
                            raw_last_id = raw_id
                            cnt_i = None
                        else:
                            t_id145_ns = time.perf_counter_ns()
                            latest_id145 = None
                            try:
                                latest_id145 = self.app._get_latest_cl145()
                            except Exception:
                                latest_id145 = None
                            if latest_id145 is None:
                                try:
                                    latest_id145 = self.device.read_cl_sync("out145", timeout_s=0.5)
                                except Exception:
                                    latest_id145 = None
                            if latest_id145 is not None and len(latest_id145) == 6:
                                _, id_out2_mm, _, _, raw_dict, cnt_dict = latest_id145
                            else:
                                id_out2_mm, raw_dict, cnt_dict = None, {}, {}
                            raw_map: Mapping[str, int | None] = raw_dict if isinstance(raw_dict, Mapping) else {}
                            cnt_map: Mapping[str, int | None] = cnt_dict if isinstance(cnt_dict, Mapping) else {}
                            perf.add_time_ns("id145", time.perf_counter_ns() - t_id145_ns)
                            try:
                                out2_count = cnt_map.get("out2")
                                id_cnt_out2 = int(out2_count) if out2_count is not None else None
                            except Exception:
                                id_cnt_out2 = None
                            # gate duplicates by OUT2 update counter if available
                            if id_cnt_out2 is not None:
                                if last_id_cnt2 is not None and int(id_cnt_out2) == int(last_id_cnt2):
                                    skip_id_none += 1
                                    perf.add_count("skip_id_none", 1)
                                    perf.add_count("dedup", 1)
                                    dedup_count += 1
                                    _loop_done(t_loop0_ns, len(raw_points), filled, perf)
                                    continue
                                last_id_cnt2 = int(id_cnt_out2)
                            cnt_i = id_cnt_out2
                            raw_last_id = f"OUT2={raw_map.get('out2', None)} cnt2={id_cnt_out2}"
                        if id_out2_mm is None:
                            skip_id_none += 1
                            perf.add_count("skip_id_none", 1)
                            _loop_done(t_loop0_ns, len(raw_points), filled, perf)
                            continue
                    elif disable_id_modbus:
                        # placeholder ID so downstream code stays intact
                        id_mm = float(getattr(recipe, "id_std_mm", 0.0) or 0.0)
                        if (not math.isfinite(id_mm)) or id_mm <= 0.0:
                            id_mm = 80.0
                        raw_last_id = "ID_DISABLED"
                    elif getattr(self.app, "sim_disp_enabled", False):
                        id_val, raw_id = self.app.simulate_disp_once(recipe)
                        id_mm = float(id_val) if id_val is not None else None
                        raw_last_id = raw_id
                        ecc_x = 0.05 * math.sin(0.9 * float(section_idx))
                        ecc_y = 0.05 * math.cos(0.7 * float(section_idx))
                    else:
                        if bool(getattr(recipe, "id_use_fit", False)):
                            t_id145_ns = time.perf_counter_ns()
                            latest_id145 = None
                            try:
                                latest_id145 = self.app._get_latest_cl145()
                            except Exception:
                                latest_id145 = None
                            if latest_id145 is None:
                                try:
                                    latest_id145 = self.device.read_cl_sync("out145", timeout_s=0.5)
                                except Exception:
                                    latest_id145 = None
                            if latest_id145 is not None and len(latest_id145) == 6:
                                id_x1_mm, id_x2_mm, id_c_mm, id_m_mm, raw_dict, cnt_dict = latest_id145
                            else:
                                id_x1_mm, id_x2_mm, id_c_mm, id_m_mm, raw_dict, cnt_dict = (None, None, None, None, {}, {})
                            raw_map: Mapping[str, int | None] = raw_dict if isinstance(raw_dict, Mapping) else {}
                            cnt_map: Mapping[str, int | None] = cnt_dict if isinstance(cnt_dict, Mapping) else {}
                            perf.add_time_ns("id145", time.perf_counter_ns() - t_id145_ns)
                            try:
                                out4_count = cnt_map.get("out4")
                                out5_count = cnt_map.get("out5")
                                id_cnt_out4 = int(out4_count) if out4_count is not None else None
                                id_cnt_out5 = int(out5_count) if out5_count is not None else None
                            except Exception:
                                id_cnt_out4 = None
                                id_cnt_out5 = None

                            # gate duplicates by OUT4 update counter if available
                            if id_cnt_out4 is not None:
                                if last_id_cnt4 is not None and int(id_cnt_out4) == int(last_id_cnt4):
                                    skip_id_none += 1
                                    perf.add_count("skip_id_none", 1)
                                    perf.add_count("dedup", 1)
                                    dedup_count += 1
                                    _loop_done(t_loop0_ns, len(raw_points), filled, perf)
                                    continue
                                last_id_cnt4 = int(id_cnt_out4)

                            cnt_i = id_cnt_out4
                            raw_last_id = f"OUT4={raw_map.get('out4', None)} OUT5={raw_map.get('out5', None)} cnt4={id_cnt_out4} cnt5={id_cnt_out5}"
                            if id_c_mm is None:
                                skip_id_none += 1
                                perf.add_count("skip_id_none", 1)
                                _loop_done(t_loop0_ns, len(raw_points), filled, perf)
                                continue

                            # Use OUT5 as m if valid, else derive from x1/x2.
                            m_used = id_m_mm
                            try:
                                if m_used is None or (not math.isfinite(float(m_used))):
                                    if id_x1_mm is not None and id_x2_mm is not None and math.isfinite(float(id_x1_mm)) and math.isfinite(float(id_x2_mm)):
                                        m_used = 0.5 * (float(id_x1_mm) - float(id_x2_mm))
                            except Exception:
                                pass
                            if m_used is None or (not math.isfinite(float(m_used))):
                                skip_id_none += 1
                                perf.add_count("skip_id_none", 1)
                                _loop_done(t_loop0_ns, len(raw_points), filled, perf)
                                continue
                            id_m_mm = float(m_used)

                            # For legacy coords_id synthesis, treat chord length as "diameter-like" scalar.
                            id_mm = float(id_c_mm)

                            # plausibility filter on chord (must be positive and not wildly larger than nominal D)
                            id_std = float(getattr(recipe, "id_std_mm", 0.0) or 0.0)
                            od_tol = float(getattr(recipe, "od_tol_mm", 0.0) or 0.0)
                            if id_std > 0.0:
                                margin = max(10.0, 20.0 * max(od_tol, 0.1), 0.10 * id_std)
                                if (not math.isfinite(float(id_mm))) or (float(id_mm) <= 0.0) or (float(id_mm) > (id_std + margin)):
                                    skip_id_outlier += 1
                                    perf.add_count("skip_id_outlier", 1)
                                    _loop_done(t_loop0_ns, len(raw_points), filled, perf)
                                    continue

                        else:
                            t_id3_ns = time.perf_counter_ns()
                            latest_id3 = None
                            try:
                                latest_id3 = self.app._get_latest_cl3()
                            except Exception:
                                latest_id3 = None
                            if latest_id3 is None:
                                try:
                                    latest_id3 = self.device.read_cl_sync("out3", timeout_s=0.5)
                                except Exception:
                                    latest_id3 = None
                            if latest_id3 is not None and len(latest_id3) == 3:
                                id_val, raw_i, cnt_i = latest_id3
                            else:
                                id_val, raw_i, cnt_i = (None, None, None)
                            perf.add_time_ns("id3", time.perf_counter_ns() - t_id3_ns)
                            raw_last_id = f"OUT3={raw_i} cnt={cnt_i}"
                            if id_val is None:
                                skip_id_none += 1
                                perf.add_count("skip_id_none", 1)
                                _loop_done(t_loop0_ns, len(raw_points), filled, perf)
                                continue
                            id_mm = float(id_val)

                            id_std = float(getattr(recipe, "id_std_mm", 0.0) or 0.0)
                            od_tol = float(getattr(recipe, "od_tol_mm", 0.0) or 0.0)
                            if id_std > 0.0:
                                margin = max(5.0, 10.0 * max(od_tol, 0.1), 0.05 * id_std)
                                if (not math.isfinite(float(id_mm))) or abs(float(id_mm) - id_std) > margin:
                                    skip_id_outlier += 1
                                    perf.add_count("skip_id_outlier", 1)
                                    _loop_done(t_loop0_ns, len(raw_points), filled, perf)
                                    continue

                else:
                    raw_last_id = "ID_SKIPPED"

                if sample_id and (not id_single_enable) and (id_mm is None):
                    skip_gate_id_none += 1
                    perf.add_count("skip_sync_mismatch", 1)
                    skip_sync_mismatch += 1
                    _loop_done(t_loop0_ns, len(raw_points), filled, perf)
                    continue
                if sample_id and id_single_enable and (id_out2_mm is None):
                    skip_gate_id_none += 1
                    perf.add_count("skip_sync_mismatch", 1)
                    skip_sync_mismatch += 1
                    _loop_done(t_loop0_ns, len(raw_points), filled, perf)
                    continue

                if sample_id:
                    if id_single_enable:
                        if id_out2_mm is not None:
                            perf.add_count("id_ok", 1)
                            id_ok_total += 1
                    else:
                        if id_mm is not None:
                            perf.add_count("id_ok", 1)
                            id_ok_total += 1

                # ---------------- accepted sample ----------------
                # track extremes on accepted samples
                try:
                    if sample_od and od is not None:
                        if od_min is None or float(od) < float(od_min):
                            od_min = float(od)
                            log("SAMPLE_OD_MIN", section=section_idx+1, theta_deg=float(theta_deg), od=float(od), raw=raw_last_od)
                        if od_max is None or float(od) > float(od_max):
                            od_max = float(od)
                            log("SAMPLE_OD_MAX", section=section_idx+1, theta_deg=float(theta_deg), od=float(od), raw=raw_last_od)
                    if sample_id and id_mm is not None:
                        if id_min is None or float(id_mm) < float(id_min):
                            id_min = float(id_mm)
                            log("SAMPLE_ID_MIN", section=section_idx+1, theta_deg=float(theta_deg), id=float(id_mm), raw=raw_last_id)
                        if id_max is None or float(id_mm) > float(id_max):
                            id_max = float(id_mm)
                            log("SAMPLE_ID_MAX", section=section_idx+1, theta_deg=float(theta_deg), id=float(id_mm), raw=raw_last_id)
                except Exception:
                    pass

                # record raw point
                t_append0_ns = time.perf_counter_ns()
                try:
                    raw_points.append({
                        "phase": str(phase),
                        "ts": float(time.time()),
                        "theta_deg": float(theta_deg),
                        "od_mm": None if (od is None) else float(od),
                        "id_mm": None if (id_mm is None) else float(id_mm),
                        "id_out2_mm": None if (id_out2_mm is None) else float(id_out2_mm),
                        "id_c_mm": None if id_c_mm is None else float(id_c_mm),
                        "id_m_mm": None if id_m_mm is None else float(id_m_mm),
                        "id_x1_mm": None if id_x1_mm is None else float(id_x1_mm),
                        "id_x2_mm": None if id_x2_mm is None else float(id_x2_mm),
                        "id_cnt_out4": None if id_cnt_out4 is None else int(id_cnt_out4),
                        "id_cnt_out5": None if id_cnt_out5 is None else int(id_cnt_out5),
                        "id_cnt_out2": None if id_cnt_out2 is None else int(id_cnt_out2),
                        "od_out1": None if od_out1 is None else float(od_out1),
                        "od_out2": None if od_out2 is None else float(od_out2),
                        "od_B": None if od_B is None else float(od_B),
                        "od_map_out1": str(od_map_out1),
                        "od_L": None if od_L is None else float(od_L),
                        "od_R": None if od_R is None else float(od_R),
                        "od_delta": None if od_delta is None else float(od_delta),
                        "cl_cnt": None if cnt_i is None else int(cnt_i),
                        "bin": int(b),
                        "raw_od": str(raw_last_od),
                        "raw_id": str(raw_last_id),
                    })
                    perf.add_count("append", 1)
                    raw_append_total += 1
                except Exception:
                    pass
                perf.add_time_ns("append", time.perf_counter_ns() - t_append0_ns)

                if cnt[b] == 0:
                    filled += 1
                    try:
                        bin_fill_logs += 1
                        log("SAMPLE_BIN_FILL", section=section_idx+1, bin=b, theta_deg=float(theta_deg),
                            od=None if od is None else float(od),
                            id=None if id_mm is None else float(id_mm),
                            raw_od=raw_last_od, raw_id=raw_last_id)
                    except Exception:
                        pass
                cnt[b] += 1

                if sample_od and od is not None:
                    r_od = 0.5 * float(od)
                    x_od = r_od * math.cos(theta)
                    y_od = r_od * math.sin(theta)
                    sum_x_od[b] += x_od
                    sum_y_od[b] += y_od
                    sum_r_od[b] += float(r_od)

                if sample_id and (not id_single_enable) and id_mm is not None:
                    r_id = 0.5 * float(id_mm)
                    x_id = r_id * math.cos(theta) + float(ecc_x)
                    y_id = r_id * math.sin(theta) + float(ecc_y)
                    sum_x_id[b] += x_id
                    sum_y_id[b] += y_id
                    sum_r_id[b] += float(r_id)

                time.sleep(0.005)
                _loop_done(t_loop0_ns, len(raw_points), filled, perf)

            _flush_sample_perf_if_due(len(raw_points), filled, perf, force=True)
            # build coords according to fit strategy (mode already determined above)

            max_gap_deg = _max_gap_deg_from_bins(cnt, n)
            self._last_sample_max_gap_deg = float(max_gap_deg)

            coords_od: List[Tuple[float, float]] = []
            coords_id: List[Tuple[float, float]] = []
            self._last_fit_weights_od = None
            self._last_fit_weights_id = None

            if mode == "c":
                miss = 0
                for i in range(n):
                    if cnt[i] > 0:
                        th = math.radians((float(i) + 0.5) * (360.0 / float(n)))
                        if sample_od:
                            r_od_bin = float(sum_r_od[i]) / float(cnt[i])
                            coords_od.append((r_od_bin * math.cos(th), r_od_bin * math.sin(th)))
                        if sample_id and (not id_single_enable):
                            r_id_bin = float(sum_r_id[i]) / float(cnt[i])
                            coords_id.append((r_id_bin * math.cos(th), r_id_bin * math.sin(th)))
                    else:
                        miss += 1
            else:
                # Raw-point based strategies
                for p in raw_points:
                    if not isinstance(p, dict):
                        continue
                    try:
                        theta_value = p.get("theta_deg")
                        if theta_value is None:
                            continue
                        th_deg = float(theta_value)
                        th = math.radians(th_deg)
                    except Exception:
                        continue
                    if sample_od:
                        od_v = p.get("od_mm", None)
                        if od_v is not None:
                            try:
                                od_f = float(od_v)
                                r_od = 0.5 * od_f
                                coords_od.append((r_od * math.cos(th), r_od * math.sin(th)))
                            except Exception:
                                pass
                    if sample_id and (not id_single_enable):
                        id_v = p.get("id_mm", None)
                        if id_v is not None:
                            try:
                                id_f = float(id_v)
                                r_id = 0.5 * id_f
                                coords_id.append((r_id * math.cos(th), r_id * math.sin(th)))
                            except Exception:
                                pass

                if mode == "b":
                    try:
                        if sample_od:
                            w_od = []
                            for p in raw_points:
                                if p.get("od_mm", None) is None:
                                    continue
                                bin_value = p.get("bin")
                                if bin_value is None:
                                    continue
                                bidx = int(bin_value)
                                c = int(cnt[bidx]) if 0 <= bidx < n else 0
                                w_od.append(1.0 / float(c) if c > 0 else 0.0)
                            w_od = np.asarray(w_od, dtype=float)
                            if w_od.size == len(coords_od) and w_od.size > 0:
                                self._last_fit_weights_od = w_od
                        if sample_id and (not id_single_enable):
                            w_id = []
                            for p in raw_points:
                                if p.get("id_mm", None) is None:
                                    continue
                                bin_value = p.get("bin")
                                if bin_value is None:
                                    continue
                                bidx = int(bin_value)
                                c = int(cnt[bidx]) if 0 <= bidx < n else 0
                                w_id.append(1.0 / float(c) if c > 0 else 0.0)
                            w_id = np.asarray(w_id, dtype=float)
                            if w_id.size == len(coords_id) and w_id.size > 0:
                                self._last_fit_weights_id = w_id
                    except Exception:
                        self._last_fit_weights_od = None
                        self._last_fit_weights_id = None

                miss = int(n - sum(1 for i in range(n) if cnt[i] > 0))

            # store last coverage/reason for UI/debug
            elapsed = float(time.time() - t_start)
            self._last_sample_cov = (n, n - miss, miss)
            self._last_sample_reason = (reason, revs, elapsed)

            # debug summary (radius-based, per-bin averages)
            try:
                rbin_od = [sum_r_od[i] / cnt[i] for i in range(n) if cnt[i] > 0] if sample_od else []
                rbin_id = [sum_r_id[i] / cnt[i] for i in range(n) if cnt[i] > 0] if sample_id else []

                od_r_pp = float(max(rbin_od) - min(rbin_od)) if len(rbin_od) >= 2 else 0.0
                id_r_pp = float(max(rbin_id) - min(rbin_id)) if len(rbin_id) >= 2 else 0.0
                od_r_pp_t = _pp_trim_list(rbin_od, 0.01)
                id_r_pp_t = _pp_trim_list(rbin_id, 0.01)

                self._last_sample_debug = {
                    "phase": str(phase),
                    "iters": int(iters),
                    "skip_no_new_od": int(skip_no_new_od),
                    "skip_od_outlier": int(skip_od_outlier),
                    "skip_id_none": int(skip_id_none),
                    "skip_id_outlier": int(skip_id_outlier),
                    "od_min": od_min, "od_max": od_max,
                    "id_min": id_min, "id_max": id_max,
                    "od_r_pp": od_r_pp, "id_r_pp": id_r_pp,
                    "od_r_pp_trim": od_r_pp_t, "id_r_pp_trim": id_r_pp_t,
                    "filled": int(n - miss), "miss": int(miss),
                    "need": int(need), "n": int(n),
                    "reason": str(reason), "revs": float(revs), "elapsed": float(elapsed),
                }

                log(
                    "SAMPLE_DONE",
                    section=section_idx + 1,
                    phase=str(phase),
                    n=n,
                    need=need,
                    filled=(n - miss),
                    miss=miss,
                    reason=reason,
                    revs=revs,
                    elapsed=elapsed,
                    iters=iters,
                    skip_no_new_od=skip_no_new_od,
                    skip_od_outlier=skip_od_outlier,
                    skip_id_none=skip_id_none,
                    skip_id_outlier=skip_id_outlier,
                    od_min=od_min,
                    od_max=od_max,
                    id_min=id_min,
                    id_max=id_max,
                    od_r_pp=od_r_pp,
                    od_r_pp_trim=od_r_pp_t,
                    id_r_pp=id_r_pp,
                    id_r_pp_trim=id_r_pp_t,
                )
                try:
                    loops_per_s = (float(iters) / float(elapsed)) if float(elapsed) > 1e-9 else 0.0
                    n_od_acc = int(
                        sum(1 for _p in (raw_points or []) if isinstance(_p, dict) and _p.get("od_mm", None) is not None)
                    )
                    if id_single_enable:
                        n_id_acc = int(
                            sum(
                                1
                                for _p in (raw_points or [])
                                if isinstance(_p, dict) and _p.get("id_out2_mm", None) is not None
                            )
                        )
                    else:
                        n_id_acc = int(
                            sum(1 for _p in (raw_points or []) if isinstance(_p, dict) and _p.get("id_mm", None) is not None)
                        )
                    perf_logger.info(
                        "[AUTOFLOW_PERF] section=%d phase=%s elapsed_s=%.3f loops_total=%d loops_per_s=%.2f "
                        "od_req_total=%d od_new_total=%d id_ok_total=%d raw_append_total=%d "
                        "skip_no_new_od=%d skip_od_outlier=%d skip_id_none=%d skip_id_outlier=%d dedup=%d skip_sync_mismatch=%d "
                        "end=%s n_od=%d n_id=%d miss_bin=%d max_gap_deg=%.3f",
                        int(section_idx + 1),
                        str(phase),
                        float(elapsed),
                        int(iters),
                        float(loops_per_s),
                        int(od_req_total),
                        int(od_new_total),
                        int(id_ok_total),
                        int(raw_append_total),
                        int(skip_no_new_od),
                        int(skip_od_outlier),
                        int(skip_id_none),
                        int(skip_id_outlier),
                        int(dedup_count),
                        int(skip_sync_mismatch),
                        str(reason),
                        int(n_od_acc),
                        int(n_id_acc),
                        int(miss),
                        float(max_gap_deg),
                    )
                except Exception:
                    pass
            except Exception as e:
                try:
                    log("SAMPLE_DONE_ERR", section=section_idx + 1, err=str(e))
                except Exception:
                    pass

            if sample_od and len(coords_od) < 3:
                raise RuntimeError("等角采样覆盖不足：外径有效点数 < 3，无法拟合圆。")
            if sample_id and (not id_single_enable) and len(coords_id) < 3:
                raise RuntimeError("等角采样覆盖不足：内径有效点数 < 3，无法拟合圆。")

            # expose per-channel sample counts for UI diagnostics

            try:

                self._last_sample_n_od = int(sum(1 for _p in (raw_points or []) if isinstance(_p, dict) and _p.get("od_mm", None) is not None))

                if id_single_enable:
                    self._last_sample_n_id = int(
                        sum(1 for _p in (raw_points or []) if isinstance(_p, dict) and _p.get("id_out2_mm", None) is not None)
                    )
                else:
                    self._last_sample_n_id = int(
                        sum(1 for _p in (raw_points or []) if isinstance(_p, dict) and _p.get("id_mm", None) is not None)
                    )

            except Exception:

                self._last_sample_n_od = None

                self._last_sample_n_id = None


            return (
                np.asarray(coords_od, dtype=float),
                np.asarray(coords_id, dtype=float),
                raw_last_od,
                raw_last_id,
                raw_points,
            )

        finally:
            self.app.set_plc_poll_profile("normal")

    def _sample_circle_points(self, recipe: Recipe) -> Tuple[np.ndarray, str]:
        """Backward-compatible OD-only sampling wrapper."""
        coords_od, _coords_id, raw_od, _raw_id, _raw_pts = self._sample_circle_points_dual(
            recipe,
            section_idx=0,
            sample_od=True,
            sample_id=False,
            phase="OD",
        )
        return coords_od, raw_od


__all__ = ["ExecutorSamplingMixin"]
