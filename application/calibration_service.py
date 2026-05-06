from __future__ import annotations

import datetime
import math
import re
import time
from typing import Any, Iterable, Mapping, cast

import numpy as np

from core.models import Recipe
from domain.calibration import (
    compute_od_b_candidate,
    fit_id_diameter as fit_id_diameter_pure,
    fit_id_single_from_out2 as fit_id_single_from_out2_pure,
    lsq_fit_cos_sin as lsq_fit_cos_sin_pure,
    robust_span as robust_span_pure,
    solve_id_delta_candidate,
    verify_id_calibration,
)


class CalibrationService:
    """Calibration logic extracted from legacy UI callbacks.

    This service is intentionally transitional: it moves acquisition, fitting,
    persistence, and raw export logic out of screen-bound callbacks while still
    reusing the existing legacy host helpers and Tk variables where needed.
    """

    @staticmethod
    def _host_var_value(host: Any, name: str, default: Any) -> Any:
        var = getattr(host, name, None)
        if var is None:
            return default
        return cast(Any, var).get()

    def _mode_acquiring(self, host: Any) -> None:
        try:
            host.calibration_mode.begin_acquiring()
        except Exception:
            pass

    def _mode_fitting(self, host: Any) -> None:
        try:
            host.calibration_mode.begin_fitting()
        except Exception:
            pass

    def _mode_saving(self, host: Any) -> None:
        try:
            host.calibration_mode.begin_saving()
        except Exception:
            pass

    def _mode_complete(self, host: Any) -> None:
        try:
            host.calibration_mode.complete()
        except Exception:
            pass

    def _mode_fail(self, host: Any, message: str) -> None:
        try:
            host.calibration_mode.fail(message)
        except Exception:
            pass

    def start_od_capture(self, host: Any) -> None:
        try:
            if getattr(host, '_odcal_capturing', False):
                return
            mode = (host.odcal_mode_var.get() or 'timed').strip()
            angle_src = str(host.odcal_angle_src_var.get() or 'AX3').strip()
            no_angle = ('无' in angle_src) or (angle_src.upper() == 'NONE')
            host._odcal_angle_enabled = not no_angle
            host._odcal_filter_mode = str(host.odcal_filter_var.get() or '无').strip()
            try:
                host._odcal_outlier_sigma = float(host._parse_float(host.odcal_outlier_sigma_var.get(), 3.0))
            except Exception:
                host._odcal_outlier_sigma = 3.0
            if mode == 'one_rev' and no_angle:
                mode = 'timed'
                try:
                    host.odcal_mode_var.set('timed')
                except Exception:
                    pass
                host.odcal_msg_var.set('角度来源=无角度：已自动切换为定时采样。')
            host._odcal_one_rev = mode == 'one_rev'
            host._odcal_stop_reason = ''
            cmd = (host.odcal_cmd_var.get() or 'M0,1').strip()
            if getattr(host, 'gauge_worker', None) is not None:
                host.gauge_worker.request_cmd = cmd
            if not cmd.upper().startswith('M0'):
                host.odcal_msg_var.set('提示：标定 B 建议使用 M0,*（同时输出 OUT1+OUT2）。')
            hz = max(1.0, float(host._parse_float(host.odcal_hz_var.get(), 20.0)))
            dur = max(0.5, float(host._parse_float(host.odcal_duration_var.get(), 10.0)))
            host._odcal_points = []
            host._odcal_drop_cnt = 0
            host._odcal_theta_start = None
            host._odcal_theta_last = None
            host._odcal_theta_unwrap = 0.0
            host._odcal_rev_progress_deg = 0.0
            host._odcal_ax3_rotating = False
            try:
                host._odcal_ax3_speed_degps = float(host._parse_float(host.odcal_rot_degps_var.get(), 10.0))
            except Exception:
                host._odcal_ax3_speed_degps = 10.0
            host._odcal_capturing = True
            host._odcal_start_ts = time.time()
            host._odcal_stop_at_ts = host._odcal_start_ts + dur
            self._mode_acquiring(host)
            host.odcal_state_var.set('CAPTURING')
            if host._odcal_one_rev:
                spd = float(host._odcal_ax3_speed_degps)
                host._odcal_start_ax3_rotation(spd)
                host.odcal_msg_var.set(f'一圈采样... cmd={cmd}  {hz:.1f}Hz  spd={spd:.2f}deg/s  timeout={dur:.1f}s')
            else:
                host.odcal_msg_var.set(f'采集中... cmd={cmd}  {hz:.1f}Hz x {dur:.1f}s')
            host.odcal_n_var.set('0')
            host.odcal_elapsed_var.set('0.0s')
            host.odcal_B_candidate_var.set('--')
            self.od_tick(host, hz=hz)
        except Exception as exc:
            try:
                host._odcal_stop_ax3_rotation()
            except Exception:
                pass
            host.odcal_state_var.set('ERROR')
            host.odcal_msg_var.set(f'启动采集失败: {exc}')
            host._odcal_capturing = False
            self._mode_fail(host, str(exc))

    def stop_od_capture(self, host: Any, reason: str = '') -> None:
        try:
            if not getattr(host, '_odcal_capturing', False):
                return
            host._odcal_capturing = False
            host._odcal_stop_reason = str(reason or host._odcal_stop_reason or '')
            try:
                host._odcal_stop_ax3_rotation()
            except Exception:
                pass
            if getattr(host, '_odcal_after_id', None):
                try:
                    host.after_cancel(host._odcal_after_id)
                except Exception:
                    pass
                host._odcal_after_id = None
            host._odcal_stop_at_ts = None
            host.odcal_state_var.set('DONE')
            msg = '采集完成，可计算 B'
            if host._odcal_stop_reason == 'timeout':
                msg = '采集结束：超时停止，可计算 B'
            elif host._odcal_stop_reason == 'one_rev':
                msg = '采集结束：完成一圈，可计算 B'
            elif host._odcal_stop_reason == 'manual':
                msg = '采集结束：手动停止，可计算 B'
            host.odcal_msg_var.set(msg)
            host._odcal_update_stats()
            self._mode_complete(host)
        except Exception:
            pass

    def clear_od_capture(self, host: Any) -> None:
        try:
            try:
                host._odcal_stop_ax3_rotation()
            except Exception:
                pass
            host._odcal_points = []
            host._odcal_drop_cnt = 0
            host._odcal_capturing = False
            host._odcal_start_ts = None
            host._odcal_stop_at_ts = None
            host._odcal_one_rev = False
            host._odcal_stop_reason = ''
            host._odcal_theta_start = None
            host._odcal_theta_last = None
            host._odcal_theta_unwrap = 0.0
            host._odcal_rev_progress_deg = 0.0
            host.odcal_state_var.set('IDLE')
            host.odcal_msg_var.set('-')
            host.odcal_B_candidate_var.set('--')
            host.odcal_n_var.set('0')
            host.odcal_elapsed_var.set('--')
            host.odcal_sum_mean_var.set('--')
            host.odcal_sum_std_var.set('--')
            host.odcal_sum_min_var.set('--')
            host.odcal_sum_max_var.set('--')
            host.odcal_drop_rate_var.set('--')
            try:
                tpl = getattr(host, '_odcal_defect_template_mask', [0] * 360)
                if tpl and sum(int(x) for x in tpl) > 0:
                    host.odcal_defect_mode_var.set('TEMPLATE')
                    host.odcal_defect_shift_var.set('--')
                    host.odcal_defects_var.set('模板: ' + host._odcal_ranges_str(host._odcal_mask_to_ranges(tpl)))
                else:
                    host.odcal_defect_mode_var.set('OFF')
                    host.odcal_defect_shift_var.set('--')
                    host.odcal_defects_var.set('--')
            except Exception:
                pass
            self._mode_complete(host)
        except Exception:
            pass

    def od_tick(self, host: Any, hz: float = 20.0) -> None:
        if not getattr(host, '_odcal_capturing', False):
            return
        now = time.time()
        if getattr(host, '_odcal_stop_at_ts', None) is not None and now >= host._odcal_stop_at_ts:
            self.stop_od_capture(host, 'timeout' if getattr(host, '_odcal_one_rev', False) else '')
            return
        if getattr(host, '_odcal_start_ts', None) is not None:
            host.odcal_elapsed_var.set(f"{(now - host._odcal_start_ts):.1f}s")
        if getattr(host, '_odcal_one_rev', False):
            try:
                th = host._odcal_get_ax3_pos()
                host._odcal_update_rev_progress(th)
                if host._odcal_rev_done():
                    self.stop_od_capture(host, 'one_rev')
                    return
            except Exception:
                pass
        try:
            if getattr(host, 'gauge_worker', None) is not None:
                host.gauge_worker.send_request()
        except Exception:
            pass
        dt_ms = int(max(20.0, 1000.0 / float(hz)))
        try:
            host._odcal_after_id = host.after(dt_ms, lambda: self.od_tick(host, hz=hz))
        except Exception:
            host._odcal_after_id = None

    def on_od_gauge_sample(self, host: Any, payload: Mapping[str, Any]) -> None:
        if not getattr(host, '_odcal_capturing', False):
            return
        try:
            v1 = payload.get('od', None)
            v2 = payload.get('od2', None)
            j1 = str(payload.get('judge', '') or '').strip().upper()
            j2 = str(payload.get('judge2', '') or '').strip().upper()
            raw = str(payload.get('raw', '') or '').strip()
            ts = float(payload.get('ts', time.time()))
            theta = None
            theta_rel = None
            if bool(getattr(host, '_odcal_angle_enabled', True)):
                try:
                    theta = host._odcal_get_ax3_pos()
                    if getattr(host, '_odcal_one_rev', False):
                        theta_rel = host._odcal_update_rev_progress(theta)
                except Exception:
                    theta = None
                    theta_rel = None
            if j1 and j1 != 'GO':
                host._odcal_drop_cnt += 1
            if j2 and j2 != 'GO':
                host._odcal_drop_cnt += 1
            host._odcal_points.append({'ts': ts, 'raw': raw, 'v1': v1, 'j1': j1, 'v2': v2, 'j2': j2, 'theta': theta, 'theta_rel': theta_rel})
            host.odcal_n_var.set(str(len(host._odcal_points)))
            if getattr(host, '_odcal_one_rev', False) and host._odcal_rev_done():
                self.stop_od_capture(host, 'one_rev')
        except Exception:
            pass

    def compute_od_candidate(self, host: Any) -> None:
        self._mode_fitting(host)
        try:
            dref = float(host._parse_float(host.odcal_dref_var.get(), 180.0))
            sums, _meta = host._odcal_prepare_sums()
            if not sums:
                host.odcal_msg_var.set('无法计算：当前采集数据没有两路（OUT1+OUT2）值。请使用 M0,* 采集。')
                host.odcal_state_var.set('ERROR')
                self._mode_fail(host, 'No OD calibration points')
                return
            result = compute_od_b_candidate(sums, dref)
            if not result.ok or result.b_candidate is None:
                host.odcal_msg_var.set('无法计算：标定和差为空')
                host.odcal_state_var.set('ERROR')
                self._mode_fail(host, result.reason or 'OD candidate failed')
                return
            b_val = float(result.b_candidate)
            host.odcal_B_candidate_var.set(f'{b_val:.5f}')
            host.odcal_state_var.set('DONE')
            host.odcal_msg_var.set('已计算 B_candidate，可应用')
            host._odcal_update_stats()
            self._mode_complete(host)
        except Exception as exc:
            host.odcal_state_var.set('ERROR')
            host.odcal_msg_var.set(f'\u8ba1\u7b97\u5931\u8d25: {exc}')
            self._mode_fail(host, str(exc))

    def build_od_record(self, host: Any, *, b_active: float, d_ref: float, cmd_used: str, out1_map: str) -> dict[str, Any]:
        try:
            stats = {
                'n': int(len(host._odcal_points) if hasattr(host, '_odcal_points') else 0),
                'mean_sum': host.odcal_sum_mean_var.get(),
                'std_sum': host.odcal_sum_std_var.get(),
                'min_sum': host.odcal_sum_min_var.get(),
                'max_sum': host.odcal_sum_max_var.get(),
                'drop_rate': host.odcal_drop_rate_var.get(),
            }
        except Exception:
            stats = {}
        return {
            'B_active': float(b_active),
            'D_ref': float(d_ref),
            'cmd_used': str(cmd_used or ''),
            'out_map': {'OUT1': str(out1_map or 'L'), 'OUT2': ('R' if str(out1_map or 'L').upper() == 'L' else 'L')},
            'params': {
                'angle_src': str(self._host_var_value(host, 'odcal_angle_src_var', 'AX3')),
                'filter': str(self._host_var_value(host, 'odcal_filter_var', '无')),
                'outlier_sigma': str(self._host_var_value(host, 'odcal_outlier_sigma_var', '3.0')),
            },
            'defects': {
                'template_mask': (list(getattr(host, '_odcal_defect_template_mask', []) or []) if (hasattr(host, '_odcal_defect_template_mask') and sum(int(x) for x in (getattr(host, '_odcal_defect_template_mask', []) or [])) > 0) else []),
                'template_ranges': ([[a, b] for a, b in host._odcal_mask_to_ranges(getattr(host, '_odcal_defect_template_mask', [0] * 360))] if (hasattr(host, '_odcal_defect_template_mask') and sum(int(x) for x in (getattr(host, '_odcal_defect_template_mask', []) or [])) > 0) else []),
            },
            'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
            'stats': stats,
        }

    def apply_od_candidate(self, host: Any) -> None:
        self._mode_saving(host)
        try:
            b_val = float((host.odcal_B_candidate_var.get() or '').strip())
        except Exception:
            host.odcal_msg_var.set('无有效 B_candidate')
            self._mode_fail(host, 'Invalid OD candidate')
            return
        try:
            dref = float(host._parse_float(host.odcal_dref_var.get(), 180.0))
            cmd = (host.odcal_cmd_var.get() or 'M0,1').strip()
            out1_map = (host.odcal_map_out1_var.get() or 'L').strip().upper()
            data = self.build_od_record(host, b_active=b_val, d_ref=dref, cmd_used=cmd, out1_map=out1_map)
            host.calibration_repository.save_od_active(data)
            host.odcal_B_active_var.set(f'{b_val:.5f}')
            host.odcal_state_var.set('APPLIED')
            host.odcal_msg_var.set('已应用并保存')
            self._mode_complete(host)
        except Exception as exc:
            host.odcal_state_var.set('ERROR')
            host.odcal_msg_var.set(f'应用失败: {exc}')
            self._mode_fail(host, str(exc))

    def export_od_raw(self, host: Any) -> None:
        try:
            points = list(getattr(host, '_odcal_points', None) or [])
            if not points:
                host.odcal_msg_var.set('\u65e0\u6570\u636e\u53ef\u5bfc\u51fa')
                return
            path = host.calibration_repository.export_od_raw(points)
            host.odcal_msg_var.set(f'\u5df2\u5bfc\u51fa: {path}')
        except Exception as exc:
            host.odcal_msg_var.set(f'\u5bfc\u51fa\u5931\u8d25: {exc}')

    def clear_id_single_capture(self, host: Any) -> None:
        host._id_single_cal_points = []
        host._id_single_cal_start_ts = None
        host._id_single_cal_theta_start = None
        host._id_single_cal_theta_last = None
        host._id_single_cal_theta_unwrap = 0.0
        host._id_single_cal_rev_progress_deg = 0.0
        host._id_single_cal_last_out2_cnt = None
        host.id_single_cal_mean_var.set('--')
        host.id_single_cal_B_var.set('--')
        host.id_single_cal_ecc_amp_var.set('--')
        host.id_single_cal_ecc_ang_var.set('--')
        host.id_single_cal_cov_var.set('--')
        host.id_single_cal_warn_var.set('')
        host.id_single_cal_state_var.set('IDLE')
        host.id_single_cal_msg_var.set('已清空')
        self._mode_complete(host)

    def stop_id_single_capture(self, host: Any, reason: str = '') -> None:
        host._id_single_cal_capturing = False
        try:
            if host._id_single_cal_after_id is not None:
                host.after_cancel(host._id_single_cal_after_id)
        except Exception:
            pass
        host._id_single_cal_after_id = None
        try:
            host._id_single_cal_stop_ax3_rotation()
        except Exception:
            pass
        try:
            prev = getattr(host, '_id_single_cal_prev_poll_profile', None)
            if prev:
                host.set_plc_poll_profile(prev)
        except Exception:
            pass
        host._id_single_cal_prev_poll_profile = None
        host.id_single_cal_state_var.set('STOP')
        host.id_single_cal_msg_var.set(reason or '已停止')
        self._mode_complete(host)

    def start_id_single_capture(self, host: Any) -> None:
        if getattr(host, '_id_single_cal_capturing', False):
            return
        if getattr(host, '_idcal_capturing', False):
            host.id_single_cal_state_var.set('ERR')
            host.id_single_cal_msg_var.set('ID 标定进行中')
            self._mode_fail(host, 'ID calibration busy')
            return
        try:
            host._id_single_cal_prev_poll_profile = getattr(host, '_plc_poll_profile_req', 'normal')
            host.set_plc_poll_profile('normal')
        except Exception:
            host._id_single_cal_prev_poll_profile = None
        self.clear_id_single_capture(host)
        host._id_single_cal_start_ts = time.time()
        host._id_single_cal_one_rev_timeout_ts = (host._id_single_cal_start_ts or time.time()) + 60.0
        try:
            spd = float(host._parse_float(host.idcal_rot_degps_var.get(), 10.0))
        except Exception:
            spd = 10.0
        try:
            host._id_single_cal_start_ax3_rotation(spd)
        except Exception as exc:
            host.id_single_cal_state_var.set('ERR')
            host.id_single_cal_msg_var.set(f'启动AX3失败: {exc}')
            self._mode_fail(host, str(exc))
            return
        host._id_single_cal_capturing = True
        host.id_single_cal_state_var.set('CAPTURING')
        host.id_single_cal_msg_var.set('采集中...')
        self._mode_acquiring(host)
        self.id_single_tick(host)

    def id_single_tick(self, host: Any) -> None:
        if not getattr(host, '_id_single_cal_capturing', False):
            return
        now = time.time()
        if getattr(host, '_id_single_cal_one_rev_timeout_ts', None) is not None:
            try:
                if now >= float(host._id_single_cal_one_rev_timeout_ts):
                    self.stop_id_single_capture(host, '一圈超时')
                    return
            except Exception:
                pass
        theta_deg = float('nan')
        try:
            with host._snapshot_lock:
                theta_deg = float(host._axis_snapshot[3].act_pos)
        except Exception:
            pass
        if math.isfinite(theta_deg):
            host._id_single_cal_update_rev_progress(float(theta_deg))
            if host._id_single_cal_rev_done():
                self.stop_id_single_capture(host, '已采满一圈')
                return
        _x1_mm, x2_mm, _c_mm, _m_mm, raw, cnt = host.get_cl_out145_cached()
        out2_cnt = None
        try:
            out2_cnt = cnt.get('out2', None) if isinstance(cnt, dict) else None
        except Exception:
            out2_cnt = None
        accept = False
        if x2_mm is not None and math.isfinite(float(x2_mm)):
            if out2_cnt is None:
                accept = True
            else:
                last = getattr(host, '_id_single_cal_last_out2_cnt', None)
                accept = (last is None) or (int(out2_cnt) != int(last))
            if accept and out2_cnt is not None:
                host._id_single_cal_last_out2_cnt = int(out2_cnt)
        if accept:
            host._id_single_cal_points.append({'ts': now, 'theta_deg': float(cast(Any, theta_deg)), 'out2_mm': float(cast(Any, x2_mm)), 'raw': raw, 'cnt': cnt})
        try:
            hz = float(host._parse_float(host.idcal_hz_var.get(), 20.0))
            hz = max(1.0, min(100.0, hz))
        except Exception:
            hz = 20.0
        period_ms = int(max(5, round(1000.0 / hz)))
        host._id_single_cal_after_id = host.after(period_ms, lambda: self.id_single_tick(host))

    @staticmethod
    def lsq_fit_cos_sin(theta_rad: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        return lsq_fit_cos_sin_pure(theta_rad, y)

    @staticmethod
    def _robust_span(values: np.ndarray, mode: str = 'p99_p1') -> float:
        return robust_span_pure(values, mode)


    def calc_id_single_from_out2(self, theta_deg: Iterable[float], out2_mm: Iterable[float], recipe: Recipe) -> dict[str, Any]:
        return fit_id_single_from_out2_pure(theta_deg, out2_mm, recipe).to_legacy_dict()


    def compute_apply_id_single(self, host: Any) -> None:
        self._mode_fitting(host)
        if getattr(host, '_id_single_cal_capturing', False):
            host.id_single_cal_state_var.set('ERR')
            host.id_single_cal_msg_var.set('采集中，请先停止')
            self._mode_fail(host, 'Single-probe capture still running')
            return
        if not getattr(host, '_id_single_cal_points', None):
            host.id_single_cal_state_var.set('ERR')
            host.id_single_cal_msg_var.set('无数据')
            self._mode_fail(host, 'No single-probe calibration data')
            return
        try:
            dref = float(host._parse_float(host.id_single_cal_dref_var.get(), 150.0))
        except Exception:
            dref = 150.0
        theta = [p.get('theta_deg') for p in host._id_single_cal_points if p.get('theta_deg') is not None]
        out2 = [p.get('out2_mm') for p in host._id_single_cal_points if p.get('out2_mm') is not None]
        recipe = getattr(host, 'recipe', Recipe())
        res = self.calc_id_single_from_out2(theta, out2, recipe)
        if not res or not bool(res.get('ok', False)):
            host.id_single_cal_state_var.set('ERR')
            host.id_single_cal_msg_var.set('计算失败')
            self._mode_fail(host, 'Single-probe fit failed')
            return
        mean_l2 = res.get('mean_L2_decenter', None)
        if mean_l2 is None:
            host.id_single_cal_state_var.set('ERR')
            host.id_single_cal_msg_var.set('均值无效')
            self._mode_fail(host, 'Invalid single-probe mean')
            return
        b_val = float(dref) - float(mean_l2)
        host.id_single_cal_mean_var.set(f'{float(mean_l2):.5f}')
        host.id_single_cal_B_var.set(f'{float(b_val):.5f}')
        try:
            id_ecc_amp = res.get('id_ecc_amp_mm', None)
            id_ecc_ang = res.get('id_ecc_ang_deg', None)
            host.id_single_cal_ecc_amp_var.set('--' if id_ecc_amp is None else f"{float(cast(Any, id_ecc_amp)):.5f}")
            host.id_single_cal_ecc_ang_var.set('--' if id_ecc_ang is None else f"{float(cast(Any, id_ecc_ang)):.1f}")
        except Exception:
            host.id_single_cal_ecc_amp_var.set('--')
            host.id_single_cal_ecc_ang_var.set('--')
        try:
            cov = float(res.get('cov', 0.0) or 0.0)
            host.id_single_cal_cov_var.set(f'{cov * 100:.1f}%')
        except Exception:
            cov = 0.0
            host.id_single_cal_cov_var.set('--')
        try:
            min_cov = float(getattr(host.recipe, 'min_bin_coverage', 0.95) or 0.95)
        except Exception:
            min_cov = 0.95
        host.id_single_cal_warn_var.set('覆盖率偏低' if cov < min_cov else '')
        host._cal_id_single_last = {
            'theta_deg': [float(t) for t in theta],
            'out2_mm': [float(v) for v in out2],
            'cov': float(cov),
            'n_used': int(res.get('n_used', 0) or 0),
            'n_bins': int(res.get('n_bins', 0) or 0),
        }
        self._mode_saving(host)
        try:
            host.recipe.id_single_enable = True
            host.recipe.id_single_k = 1.0
            host.recipe.id_single_b = float(b_val)
            host.recipe.disable_id_modbus = False
        except Exception:
            pass
        try:
            if hasattr(host, 'id_single_enable_var'):
                host.id_single_enable_var.set(True)
            if hasattr(host, 'id_single_k_var'):
                host.id_single_k_var.set('1.0')
            if hasattr(host, 'id_single_b_var'):
                host.id_single_b_var.set(f'{float(b_val):.5f}')
            if hasattr(host, 'disable_id_modbus_var'):
                host.disable_id_modbus_var.set(False)
        except Exception:
            pass
        host.calibration_repository.save_id_single_active({'id_single_enable': True, 'id_single_k': 1.0, 'id_single_b': float(b_val), 'D_ref': float(dref), 'cov': float(cov), 'n_used': int(res.get('n_used', 0) or 0), 'n_bins': int(res.get('n_bins', 0) or 0), 'ts': time.time()})
        try:
            data = host._recipe_dump_dict(host.recipe)
            safe = host.recipe_store.save(host.recipe.name, data)
            try:
                host.recipe_store.save_index({'last_recipe': safe})
            except Exception:
                pass
        except Exception as exc:
            host.id_single_cal_state_var.set('ERR')
            host.id_single_cal_msg_var.set(f'保存失败: {exc}')
            self._mode_fail(host, str(exc))
            return
        host.id_single_cal_state_var.set('APPLIED')
        host.id_single_cal_msg_var.set('已写入仓储/配方')
        self._mode_complete(host)

    def clear_id_capture(self, host: Any) -> None:
        host._idcal_points = []
        host._idcal_start_ts = None
        host._idcal_stop_at_ts = None
        host._idcal_theta_start = None
        host._idcal_theta_last = None
        host._idcal_theta_unwrap = 0.0
        host._idcal_rev_progress_deg = 0.0
        host.idcal_delta_candidate_var.set('--')
        host.idcal_cmax_var.set('--')
        host.idcal_mmean_var.set('--')
        host.idcal_mpp_var.set('--')
        host.idcal_fit_diam_var.set('--')
        host.idcal_fit_e_var.set('--')
        host.idcal_fit_y0_var.set('--')
        host.idcal_fit_rmse_var.set('--')
        host.idcal_chk_err_var.set('--')
        host.idcal_chk_cov_var.set('--')
        host.idcal_chk_n_var.set('--')
        host.idcal_chk_dtheta_var.set('--')
        host.idcal_state_var.set('IDLE')
        host.idcal_msg_var.set('已清空')
        self._mode_complete(host)

    def stop_id_capture(self, host: Any) -> None:
        host._idcal_capturing = False
        try:
            if host._idcal_after_id is not None:
                host.after_cancel(host._idcal_after_id)
        except Exception:
            pass
        host._idcal_after_id = None
        try:
            if host._idcal_one_rev:
                host._idcal_stop_ax3_rotation()
        except Exception:
            pass
        host._idcal_one_rev = False
        try:
            prev = getattr(host, '_idcal_prev_poll_profile', None)
            if prev:
                host.set_plc_poll_profile(prev)
        except Exception:
            pass
        host._idcal_prev_poll_profile = None
        if getattr(host, '_idcal_verify_pending', False):
            try:
                host._idcal_verify_pending = False
                self.compute_id_verify(host)
                return
            except Exception as exc:
                host.idcal_state_var.set('ERR')
                host.idcal_msg_var.set(f'复核失败: {exc}')
                self._mode_fail(host, str(exc))
                return
        host.idcal_state_var.set('STOP')
        host.idcal_msg_var.set(getattr(host, '_idcal_stop_reason', '') or '已停止')
        self._mode_complete(host)

    def start_id_capture(self, host: Any) -> None:
        if getattr(host, '_idcal_capturing', False):
            return
        try:
            host._idcal_prev_poll_profile = getattr(host, '_plc_poll_profile_req', 'normal')
            host.set_plc_poll_profile('normal')
        except Exception:
            host._idcal_prev_poll_profile = None
        host._idcal_last_out4_cnt = None
        host._idcal_one_rev_timeout_ts = None
        mode = str(host.idcal_mode_var.get() or 'one_rev').strip()
        force_one_rev = bool(getattr(host, '_idcal_force_one_rev', False))
        host._idcal_force_one_rev = False
        host._idcal_one_rev = force_one_rev or (mode == 'one_rev')
        host._idcal_points = []
        host._idcal_start_ts = time.time()
        host._idcal_stop_reason = ''
        if host._idcal_one_rev:
            try:
                spd_tmp = float(host._parse_float(host.idcal_rot_degps_var.get(), 10.0))
                spd_abs = abs(spd_tmp) if abs(spd_tmp) > 1e-6 else 10.0
                req_s = 360.0 / spd_abs
                timeout_s = max(8.0, 2.5 * req_s)
                try:
                    user_t = float(host._parse_float(host.idcal_duration_var.get(), timeout_s))
                    if math.isfinite(user_t) and (user_t >= req_s):
                        timeout_s = max(8.0, user_t)
                except Exception:
                    pass
                host._idcal_one_rev_timeout_ts = (host._idcal_start_ts or time.time()) + float(timeout_s)
            except Exception:
                host._idcal_one_rev_timeout_ts = (host._idcal_start_ts or time.time()) + 60.0
            try:
                spd = float(host._parse_float(host.idcal_rot_degps_var.get(), 10.0))
                host._idcal_ax3_speed_degps = spd
                host._idcal_start_ax3_rotation(spd)
            except Exception as exc:
                host.idcal_state_var.set('ERR')
                host.idcal_msg_var.set(f'启动AX3失败: {exc}')
                self._mode_fail(host, str(exc))
                return
        else:
            try:
                dur = float(host._parse_float(host.idcal_duration_var.get(), 10.0))
                host._idcal_stop_at_ts = (host._idcal_start_ts or time.time()) + max(0.5, dur)
            except Exception:
                host._idcal_stop_at_ts = None
        host._idcal_theta_start = None
        host._idcal_theta_last = None
        host._idcal_theta_unwrap = 0.0
        host._idcal_rev_progress_deg = 0.0
        host._idcal_capturing = True
        host.idcal_state_var.set('CAPTURING')
        host.idcal_msg_var.set('采集中...')
        self._mode_acquiring(host)
        self.id_tick(host)

    def id_tick(self, host: Any) -> None:
        if not getattr(host, '_idcal_capturing', False):
            return
        now = time.time()
        if (not getattr(host, '_idcal_one_rev', False)) and (getattr(host, '_idcal_stop_at_ts', None) is not None):
            if now >= float(host._idcal_stop_at_ts):
                host._idcal_stop_reason = '定时结束'
                self.stop_id_capture(host)
                return
        if getattr(host, '_idcal_one_rev', False) and (getattr(host, '_idcal_one_rev_timeout_ts', None) is not None):
            try:
                if now >= float(host._idcal_one_rev_timeout_ts):
                    host._idcal_stop_reason = '一圈超时(θ无效/刷新慢)'
                    self.stop_id_capture(host)
                    return
            except Exception:
                pass
        theta_deg = float('nan')
        try:
            with host._snapshot_lock:
                theta_deg = float(host._axis_snapshot[3].act_pos)
        except Exception:
            pass
        if getattr(host, '_idcal_one_rev', False) and math.isfinite(theta_deg):
            host._idcal_update_rev_progress(float(theta_deg))
            if host._idcal_rev_done():
                host._idcal_stop_reason = '已采满一圈'
                self.stop_id_capture(host)
                return
        x1_mm, x2_mm, c_mm, m_mm, raw, cnt = host.get_cl_out145_cached()
        out4_cnt = None
        try:
            out4_cnt = cnt.get('out4', None) if isinstance(cnt, dict) else None
        except Exception:
            out4_cnt = None
        accept = False
        if (c_mm is not None) and (m_mm is not None):
            if out4_cnt is None:
                accept = True
            else:
                last = getattr(host, '_idcal_last_out4_cnt', None)
                accept = (last is None) or (int(out4_cnt) != int(last))
            if accept and out4_cnt is not None:
                host._idcal_last_out4_cnt = int(out4_cnt)
        if accept:
            host._idcal_points.append({'ts': now, 'theta_deg': float(cast(Any, theta_deg)), 'x1_mm': x1_mm, 'x2_mm': x2_mm, 'c_mm': float(cast(Any, c_mm)), 'm_mm': float(cast(Any, m_mm)), 'raw': raw, 'cnt': cnt})
            try:
                cs = [p['c_mm'] for p in host._idcal_points if p.get('c_mm') is not None]
                ms = [p['m_mm'] for p in host._idcal_points if p.get('m_mm') is not None]
                if cs:
                    host.idcal_cmax_var.set(f'{max(cs):.3f}')
                if ms:
                    host.idcal_mmean_var.set(f'{(sum(ms) / len(ms)):.4f}')
                    host.idcal_mpp_var.set(f'{(max(ms) - min(ms)):.4f}')
            except Exception:
                pass
        try:
            hz = float(host._parse_float(host.idcal_hz_var.get(), 20.0))
            hz = max(1.0, min(100.0, hz))
        except Exception:
            hz = 20.0
        period_ms = int(max(5, round(1000.0 / hz)))
        host._idcal_after_id = host.after(period_ms, lambda: self.id_tick(host))

    def fit_id_diameter(self, theta_deg: np.ndarray, c_mm: np.ndarray, m_mm: np.ndarray, delta_c: float) -> dict[str, float]:
        return fit_id_diameter_pure(theta_deg, c_mm, m_mm, delta_c).to_legacy_dict()


    def compute_id_candidate(self, host: Any) -> None:
        self._mode_fitting(host)
        if not getattr(host, '_idcal_points', None):
            host.idcal_state_var.set('ERR')
            host.idcal_msg_var.set('无数据')
            self._mode_fail(host, 'No ID calibration data')
            return
        try:
            dref = float(host._parse_float(host.idcal_dref_var.get(), 150.0))
        except Exception:
            dref = 150.0
        pts = [p for p in host._idcal_points if (p.get('theta_deg') is not None and math.isfinite(float(p.get('theta_deg'))) and p.get('c_mm') is not None and p.get('m_mm') is not None)]
        if len(pts) < 20:
            cs = [p['c_mm'] for p in host._idcal_points if p.get('c_mm') is not None]
            if not cs:
                host.idcal_state_var.set('ERR')
                host.idcal_msg_var.set('无有效OUT4')
                self._mode_fail(host, 'No valid OUT4 values')
                return
            cmax = float(max(cs))
            delta = float(dref - cmax)
            host._idcal_delta_candidate = delta
            host.idcal_delta_candidate_var.set(f'{delta:.4f}')
            host.idcal_state_var.set('READY')
            host.idcal_msg_var.set('样本不足，采用 c_max 标定')
            self._mode_complete(host)
            return
        theta = np.array([p['theta_deg'] for p in pts], dtype=float)
        c = np.array([p['c_mm'] for p in pts], dtype=float)
        m = np.array([p['m_mm'] for p in pts], dtype=float)
        result = solve_id_delta_candidate(theta, c, m, dref)
        if result.cmax is not None:
            host.idcal_cmax_var.set(f'{float(result.cmax):.3f}')
        host.idcal_mmean_var.set(f'{float(np.mean(m)):.4f}')
        host.idcal_mpp_var.set(f'{float(np.max(m) - np.min(m)):.4f}')
        if not result.ok or result.delta_candidate is None:
            host.idcal_state_var.set('ERR')
            host.idcal_msg_var.set('拟合失败')
            self._mode_fail(host, result.reason or 'ID candidate failed')
            return
        if result.fallback_used:
            host.idcal_msg_var.set('样本不足，采用 c_max 标定')
        host._idcal_delta_candidate = float(result.delta_candidate)
        host.idcal_delta_candidate_var.set(f'{float(result.delta_candidate):.4f}')
        if result.fit is not None:
            host.idcal_fit_diam_var.set(f"{float(result.fit.diam):.3f}")
            host.idcal_fit_e_var.set(f"{float(result.fit.e):.4f}")
            host.idcal_fit_y0_var.set(f"{float(result.fit.y0):.4f}")
            host.idcal_fit_rmse_var.set(f"{float(result.fit.rmse_r2):.6f}")
            host.idcal_msg_var.set('计算完成（拟合+δc）')
        host.idcal_state_var.set('READY')
        self._mode_complete(host)

    def apply_id_candidate(self, host: Any) -> None:
        self._mode_saving(host)
        if getattr(host, '_idcal_delta_candidate', None) is None:
            host.idcal_state_var.set('ERR')
            host.idcal_msg_var.set('请先计算')
            self._mode_fail(host, 'No ID delta candidate')
            return
        try:
            dref = float(host._parse_float(host.idcal_dref_var.get(), 150.0))
        except Exception:
            dref = 150.0
        data = {'delta_c_mm': float(host._idcal_delta_candidate), 'D_ref': float(dref), 'ts': time.time()}
        host.calibration_repository.save_id_active(data)
        host.idcal_delta_active_var.set(f'{float(host._idcal_delta_candidate):.4f}')
        host.idcal_state_var.set('APPLIED')
        host.idcal_msg_var.set('已应用并保存')
        self._mode_complete(host)

    def export_id_raw(self, host: Any) -> None:
        points = list(getattr(host, '_idcal_points', None) or [])
        if not points:
            host.idcal_state_var.set('ERR')
            host.idcal_msg_var.set('\u65e0\u6570\u636e')
            return
        try:
            path = host.calibration_repository.export_id_raw(points)
            host.idcal_msg_var.set(f'\u5df2\u5bfc\u51fa: {path.name}')
        except Exception as exc:
            host.idcal_state_var.set('ERR')
            host.idcal_msg_var.set(f'\u5bfc\u51fa\u5931\u8d25: {exc}')

    def verify_id(self, host: Any) -> None:
        if getattr(host, '_idcal_capturing', False):
            return
        delta = None
        dref = None
        try:
            data = host.calibration_repository.load_id_active()
            if data.get('delta_c_mm', None) is not None:
                delta = float(data['delta_c_mm'])
            if data.get('D_ref', None) is not None:
                dref = float(data['D_ref'])
        except Exception:
            pass
        if delta is None:
            try:
                delta = float(host.idcal_delta_active_var.get())
            except Exception:
                delta = None
        if dref is None:
            try:
                dref = float(host._parse_float(host.idcal_dref_var.get(), 150.0))
            except Exception:
                dref = 150.0
        if delta is None or (not math.isfinite(float(delta))):
            host.idcal_state_var.set('ERR')
            host.idcal_msg_var.set('复核失败：未找到 δc_active（请先“应用”）')
            self._mode_fail(host, 'Missing ID active delta')
            return
        host._idcal_verify_pending = True
        host._idcal_verify_delta = float(delta)
        host._idcal_verify_dref = float(dref)
        host.idcal_chk_err_var.set('--')
        host.idcal_chk_cov_var.set('--')
        host.idcal_chk_n_var.set('--')
        host.idcal_chk_dtheta_var.set('--')
        try:
            host._idcal_force_one_rev = True
        except Exception:
            pass
        host.idcal_state_var.set('CHK')
        host.idcal_msg_var.set('复核采集中...')
        self.start_id_capture(host)

    def compute_id_verify(self, host: Any) -> None:
        delta = getattr(host, '_idcal_verify_delta', None)
        dref = getattr(host, '_idcal_verify_dref', None)
        if delta is None or dref is None:
            raise RuntimeError('verify参数缺失')
        pts = [p for p in host._idcal_points if (p.get('theta_deg') is not None and math.isfinite(float(p.get('theta_deg'))) and p.get('c_mm') is not None and p.get('m_mm') is not None)]
        if len(pts) < 30:
            host.idcal_state_var.set('ERR')
            host.idcal_msg_var.set(f'\u590d\u6838\u6837\u672c\u4e0d\u8db3: N={len(pts)}')
            self._mode_fail(host, 'ID verify sample too small')
            return
        theta = np.array([float(p['theta_deg']) for p in pts], dtype=float)
        c = np.array([float(p['c_mm']) for p in pts], dtype=float)
        m = np.array([float(p['m_mm']) for p in pts], dtype=float)
        result = verify_id_calibration(theta, c, m, delta_c=float(delta), d_ref=float(dref))
        host.idcal_chk_err_var.set(f'{float(result.err_mm):+.4f}')
        host.idcal_chk_cov_var.set(f'{float(result.cov_pct):.2f}%')
        host.idcal_chk_n_var.set(str(int(result.sample_count)))
        host.idcal_chk_dtheta_var.set(f'{float(result.dtheta_max_deg):.3f}')
        ok = bool(result.ok)
        err = float(result.err_mm)
        host.idcal_state_var.set('CHK_OK' if ok else 'CHK_NG')
        host.idcal_msg_var.set(f"\u590d\u6838{'OK' if ok else 'NG'}: \u0394D={err:+.4f}mm  N={int(result.sample_count)}  cover={float(result.cov_pct):.2f}%")
        if ok:
            self._mode_complete(host)
        else:
            self._mode_fail(host, f'ID verify NG: {err:+.4f}mm')


__all__ = ['CalibrationService']



