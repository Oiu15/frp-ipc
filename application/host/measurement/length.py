from __future__ import annotations

"""Length measurement and manual edge-search mixin for AppHost."""

import math
import threading
import time
import tkinter as tk
from tkinter import messagebox
from typing import TYPE_CHECKING, Any, Tuple

from core.models import AxisCal, AxisComm, Recipe
# GaugeWorker removed from drivers import — attribute is now typed as Any
# to eliminate the application/host -> drivers dependency.


# AX0 soft limits (absolute position, mm). Used for Z_disp travel estimation when PLC is offline.
# If PLC provides non-zero soft limits, those values will take precedence.
AX0_SOFTLIM_NEG_ABS = -350.0
AX0_SOFTLIM_POS_ABS = 1200.0


class HostLengthMeasurementMixin:
    """Mixin providing length travel estimates and manual edge-search controls."""

    axis_cal: AxisCal
    recipe: Recipe
    gauge_worker: Any | None  # concrete type was GaugeWorker — now accessed via attribute
    sim_gauge_var: tk.IntVar
    sim_gauge_enabled: bool

    len_enable_var: tk.BooleanVar
    len_z_low_approach_var: tk.StringVar
    len_low_search_dist_var: tk.StringVar
    len_high_search_dist_var: tk.StringVar
    len_search_vel_var: tk.StringVar
    len_search_timeout_var: tk.StringVar
    len_tol_var: tk.StringVar
    len_high_margin_var: tk.StringVar
    len_debounce_k_var: tk.StringVar
    len_backoff_var: tk.StringVar
    pipe_len_var: tk.StringVar
    len_info_var: tk.StringVar
    len_status_var: tk.StringVar
    len_edge_state_var: tk.StringVar
    len_edge_low_var: tk.StringVar
    len_edge_high_var: tk.StringVar
    len_edge_len_var: tk.StringVar
    _len_edge_search_stop_evt: threading.Event
    _len_edge_search_high_stop_evt: threading.Event
    _len_edge_search_thread: threading.Thread
    _len_edge_search_high_thread: threading.Thread

    if TYPE_CHECKING:
        def get_axis_copy(self, axis: int) -> AxisComm: ...
        def movea_abs(self, axis: int, pos_abs: float, *, context: str = "MoveA") -> None: ...
        def _recipe_ui_widget(self, name: str) -> Any: ...
        def _ui_set(self, var: tk.Variable, value: str) -> None: ...
        def _ui_btn_text(self, btn: Any, text: str) -> None: ...
        def _velmove_start_axis(
            self,
            axis: int,
            vel_velmove: float,
            *,
            acc: float = 80.0,
            dec: float = 80.0,
            jerk: float = 300.0,
        ) -> None: ...
        def _velmove_stop_axis(self, axis: int) -> None: ...
        def _wait_axis_stop_settled(
            self,
            axis: int,
            *,
            timeout_s: float = 1.5,
            stable_cycles: int = 8,
            eps_abs: float = 0.02,
            stop_evt: threading.Event | None = None,
        ) -> bool: ...
        def after(self, ms: Any, func: Any | None = None, *args: Any) -> Any: ...

    # =========================
    # Length measurement helpers
    # =========================
    def _len_pick_low_approach(self) -> None:
        """Pick current AX0 position as bottom-approach (AX0 abs) for length measurement."""
        try:
            if not hasattr(self, "len_z_low_approach_var"):
                return
            a0 = float(self.get_axis_copy(0).act_pos)
            # store absolute act_pos to decouple from Start(Z_Pos)
            self.len_z_low_approach_var.set(f"{a0:.3f}")
            self._refresh_length_info()
        except Exception as e:
            messagebox.showerror("长度测量", f"取当前位置失败: {e}")

    def _get_ax0_softlims_abs(self) -> Tuple[float, float]:
        """Return (abs_min, abs_max) soft limits for AX0."""
        try:
            ac0 = self.get_axis_copy(0)
            p = float(getattr(ac0, "softlim_pos", 0.0))
            n = float(getattr(ac0, "softlim_neg", 0.0))
            # When PLC is disconnected, some values may be 0.
            if abs(p) < 1e-6 and abs(n) < 1e-6:
                raise ValueError
            if abs(p - n) < 1e-6:
                raise ValueError
            return (min(p, n), max(p, n))
        except Exception:
            return (AX0_SOFTLIM_NEG_ABS, AX0_SOFTLIM_POS_ABS)

    def _get_ax0_z_disp_limits(self) -> Tuple[float, float, float]:
        """Return (z_min, z_max, travel) in Z_disp(mm) for AX0."""
        lo_abs, hi_abs = self._get_ax0_softlims_abs()
        z1 = float(self.axis_cal.abs_to_z_disp(0, lo_abs))
        z2 = float(self.axis_cal.abs_to_z_disp(0, hi_abs))
        z_min = min(z1, z2)
        z_max = max(z1, z2)
        return (z_min, z_max, max(0.0, z_max - z_min))

    def _refresh_length_info(self) -> None:
        """Refresh length measurement read-only info (Lmax/status) on recipe screen."""
        if not hasattr(self, "len_info_var") or not hasattr(self, "len_status_var"):
            return
        try:
            enabled = False
            try:
                enabled = bool(self.len_enable_var.get())
            except Exception:
                enabled = bool(getattr(self.recipe, "len_enable", False))

            z_min, z_max, travel = self._get_ax0_z_disp_limits()

            # Parse operator inputs
            def _f(v, d=0.0):
                try:
                    return float(v)
                except Exception:
                    return float(d)

            abs_low_appr = _f(getattr(self, "len_z_low_approach_var", tk.StringVar(value="0")).get(), 0.0)
            z_low_appr = float(self.axis_cal.abs_to_z_disp(0, abs_low_appr))
            d_low = _f(getattr(self, "len_low_search_dist_var", tk.StringVar(value="0")).get(), 0.0)
            d_high = _f(getattr(self, "len_high_search_dist_var", tk.StringVar(value="0")).get(), 0.0)
            hi_margin = _f(getattr(self, "len_high_margin_var", tk.StringVar(value="0")).get(), 0.0)
            pipe_len = _f(getattr(self, "pipe_len_var", tk.StringVar(value="0")).get(), 0.0)

            # Conservative Lmax estimation based on current approach/search settings
            z_low_edge_max = min(z_max, z_low_appr + d_low)
            lmax = z_low_edge_max + hi_margin - d_high - z_min
            if lmax < 0:
                lmax = 0.0

            self.len_info_var.set(f"{lmax:.0f}")

            if not enabled:
                self.len_status_var.set("未启用")
                return

            # Basic sanity checks
            if not (z_min <= z_low_appr <= z_max):
                self.len_status_var.set("底边接近位超出行程")
                return
            if z_low_appr + d_low > z_max + 1e-6:
                self.len_status_var.set("底边慢搜超出行程")
                return
            if lmax <= 1.0:
                self.len_status_var.set("行程不足")
                return
            if pipe_len > lmax + 1e-6:
                self.len_status_var.set(f"将跳过(管长>{lmax:.0f})")
                return

            # OK
            self.len_status_var.set("OK")
        except Exception:
            # Keep UI robust: do not raise from refresh
            try:
                self.len_info_var.set("--")
                self.len_status_var.set("--")
            except Exception:
                pass

    # =========================
    # Teach: Length edge search (manual debug)
    # =========================
    def _len_try_update_measured_length(self) -> None:
        """If both edges are known, compute pipe length and update UI vars."""
        try:
            if not hasattr(self, 'len_edge_low_var') or not hasattr(self, 'len_edge_high_var'):
                return
            try:
                z_low = float(str(self.len_edge_low_var.get()).strip())
                z_high = float(str(self.len_edge_high_var.get()).strip())
            except Exception:
                return
            L = float(z_low - z_high)
            if L <= 0 or (not math.isfinite(L)):
                return
            if hasattr(self, 'len_edge_len_var'):
                self.len_edge_len_var.set(f"{L:.3f}")
            if hasattr(self, 'len_edge_state_var'):
                self.len_edge_state_var.set(f"边沿已锁定：L={L:.1f} mm")
        except Exception:
            pass

    def _teach_len_search_low_toggle(self) -> None:
        """Toggle bottom-edge search thread (GO -> HI)."""
        try:
            th = getattr(self, '_len_edge_search_thread', None)
            if th is not None and getattr(th, 'is_alive', lambda: False)():
                # request stop
                evt = getattr(self, '_len_edge_search_stop_evt', None)
                if evt is not None:
                    evt.set()
                try:
                    if hasattr(self, 'len_edge_state_var'):
                        self.len_edge_state_var.set('底边搜索：停止中...')
                    btn = self._recipe_ui_widget('btn_len_search_low')
                    if btn is not None:
                        btn.configure(text='尝试搜索底边(GO→HI)')
                except Exception:
                    pass
                return

            # start new
            stop_evt = threading.Event()
            self._len_edge_search_stop_evt = stop_evt
            th = threading.Thread(target=self._teach_len_search_low_worker, args=(stop_evt,), daemon=True)
            self._len_edge_search_thread = th

            try:
                btn = self._recipe_ui_widget('btn_len_search_low')
                if btn is not None:
                    btn.configure(text='停止搜索底边')
                if hasattr(self, 'len_edge_state_var'):
                    self.len_edge_state_var.set('底边搜索：准备...')
            except Exception:
                pass

            th.start()
        except Exception as e:
            messagebox.showerror('底边搜索', str(e))

    def _teach_len_search_high_toggle(self) -> None:
        """Toggle top-edge search thread (GO -> HI)."""
        try:
            th = getattr(self, '_len_edge_search_high_thread', None)
            if th is not None and getattr(th, 'is_alive', lambda: False)():
                evt = getattr(self, '_len_edge_search_high_stop_evt', None)
                if evt is not None:
                    evt.set()
                try:
                    if hasattr(self, 'len_edge_state_var'):
                        self.len_edge_state_var.set('顶边搜索：停止中...')
                    btn = self._recipe_ui_widget('btn_len_search_high')
                    if btn is not None:
                        btn.configure(text='尝试搜索顶边(GO→HI)')
                except Exception:
                    pass
                return

            # Do not run concurrently with bottom search
            try:
                th_low = getattr(self, '_len_edge_search_thread', None)
                if th_low is not None and getattr(th_low, 'is_alive', lambda: False)():
                    evt_low = getattr(self, '_len_edge_search_stop_evt', None)
                    if evt_low is not None:
                        evt_low.set()
            except Exception:
                pass

            stop_evt = threading.Event()
            self._len_edge_search_high_stop_evt = stop_evt
            th = threading.Thread(target=self._teach_len_search_high_worker, args=(stop_evt,), daemon=True)
            self._len_edge_search_high_thread = th

            try:
                btn = self._recipe_ui_widget('btn_len_search_high')
                if btn is not None:
                    btn.configure(text='停止搜索顶边')
                if hasattr(self, 'len_edge_state_var'):
                    self.len_edge_state_var.set('顶边搜索：准备...')
            except Exception:
                pass

            th.start()
        except Exception as e:
            messagebox.showerror('顶边搜索', str(e))


    def _teach_len_search_low_worker(self, stop_evt: threading.Event) -> None:
        """Worker thread: bottom-edge bidirectional search (GO->HI then HI->GO) and lock AX0 Z_disp.

        机制说明：
        - 需要测径仪返回比较器判定字段(judge)，例如 GO/HI/LO。
        - 第1段：从接近位开始，沿 +Z_disp 方向慢速运动，检测 GO→HI，并锁定“最后一次GO”的 Z_disp 作为 edge1。
        - 第2段：反向沿 -Z_disp 方向慢速运动，检测 HI→GO，并锁定“最后一次HI”的 Z_disp 作为 edge2。
        - 最终边沿 = (edge1 + edge2) / 2。
        """

        # local UI helpers
        def ui_msg(msg: str) -> None:
            try:
                if hasattr(self, 'len_edge_state_var'):
                    self._ui_set(self.len_edge_state_var, msg)
            except Exception:
                pass

        def ui_done_btn() -> None:
            try:
                btn = self._recipe_ui_widget('btn_len_search_low')
                if btn is not None:
                    self._ui_btn_text(btn, '尝试搜索底边(GO→HI)')
            except Exception:
                pass

        def _wait_new_judge(ts0: float, tmax: float = 0.8):
            """Return (ts, judge) for a new gauge sample; None on timeout."""
            t0 = time.time()
            last_ts = float(ts0)
            while (not stop_evt.is_set()) and ((time.time() - t0) < float(tmax)):
                worker = gw
                if worker is None:
                    return None
                try:
                    worker.send_request()
                except Exception:
                    pass
                time.sleep(0.06)
                s = None
                try:
                    s = worker.get_last()
                except Exception:
                    s = None
                if s is None:
                    continue
                ts = float(getattr(s, 'ts', 0.0) or 0.0)
                if ts <= last_ts:
                    continue
                j = str(getattr(s, 'judge', 'UNK') or 'UNK').strip().upper()
                return ts, j
            return None

        def _axis_not_moving_guard(z_cur: float):
            """Detect 'not moving' and bail early to avoid waiting until timeout."""
            nonlocal last_move_z, last_move_ts
            if last_move_z is None:
                last_move_z = float(z_cur)
                last_move_ts = time.time()
                return False
            if abs(float(z_cur) - float(last_move_z)) >= 0.15:
                last_move_z = float(z_cur)
                last_move_ts = time.time()
                return False
            if (time.time() - float(last_move_ts)) >= 1.0:
                return True
            return False

        edge_avg = None

        try:
            # --- validations ---
            if bool(getattr(self, 'sim_gauge_enabled', False)) or (hasattr(self, 'sim_gauge_var') and int(self.sim_gauge_var.get() or 0) == 1):
                ui_msg('底边搜索：模拟测径仪不支持比较器(GO)')
                return

            gw = getattr(self, 'gauge_worker', None)
            if gw is None or (not getattr(gw, 'enabled', False)):
                ui_msg('底边搜索：请先连接测径仪(串口)')
                return

            # Require comparator mode (M1,1 / M0,1) to get judge
            try:
                req_cmd = str(getattr(gw, 'request_cmd', '') or '').upper().replace(' ', '')
            except Exception:
                req_cmd = ''
            if (',1' not in req_cmd) and ('M0,1' not in req_cmd) and ('M1,1' not in req_cmd):
                ui_msg('底边搜索：请将测径仪请求设为 M1,1 或 M0,1 (需包含比较器字段)')
                return

            # AX0 must be enabled
            ac0 = self.get_axis_copy(0)
            if int(getattr(ac0, 'sts', 0) or 0) == 0:
                ui_msg('底边搜索：请先使能 AX0')
                return

            # Read parameters from UI/recipe
            def _f(var, d=0.0):
                try:
                    return float(var.get())
                except Exception:
                    try:
                        return float(var)
                    except Exception:
                        return float(d)

            abs_appr = _f(getattr(self, 'len_z_low_approach_var', 0.0), 0.0)  # AX0 abs
            d_max = max(0.0, _f(getattr(self, 'len_low_search_dist_var', 0.0), 0.0))
            v_z = abs(_f(getattr(self, 'len_search_vel_var', 10.0), 10.0))
            timeout_s = max(1.0, _f(getattr(self, 'len_search_timeout_var', 8.0), 8.0))
            tol_z = max(0.1, _f(getattr(self, 'len_tol_var', 0.5), 0.5))

            deb_k = 2
            try:
                deb_k = int(float(getattr(self, 'len_debounce_k_var').get()))
            except Exception:
                deb_k = 2

            # Move to approach (absolute)
            ui_msg('底边搜索：移动到接近位...')
            abs_tgt = float(abs_appr)
            self.movea_abs(0, abs_tgt, context='LenEdgeLowAppr')

            # Wait until close to approach
            t0 = time.time()
            while (not stop_evt.is_set()) and (time.time() - t0 < 15.0):
                ac0 = self.get_axis_copy(0)
                if int(getattr(ac0, 'err', 0) or 0) != 0:
                    ui_msg(f"底边搜索：AX0错误({int(getattr(ac0,'err',0) or 0)})")
                    return
                if abs(float(ac0.act_pos) - abs_tgt) <= max(0.5, tol_z):
                    break
                time.sleep(0.05)

            if stop_evt.is_set():
                ui_msg('底边搜索：已停止')
                return

            if abs(float(self.get_axis_copy(0).act_pos) - abs_tgt) > max(0.8, tol_z * 2.0):
                ui_msg('底边搜索：到达接近位超时')
                return

            # Pre-check: ensure judge exists and is GO at approach
            ui_msg('底边搜索：确认比较器(GO/HI)...')
            last_ts = 0.0
            r = _wait_new_judge(last_ts, 1.5)
            if r is None:
                ui_msg('底边搜索：未收到测径仪数据(请检查串口/请求指令)')
                return
            last_ts, j0 = r
            if j0 == 'UNK':
                ui_msg('底边搜索：未收到比较器(judge)字段，请确认请求为 M1,1')
                return
            if j0 != 'GO':
                ui_msg(f'底边搜索：起点不是GO({j0})，请调整接近位/比较器阈值')
                return

            # ---------- Pass 1: GO -> HI (move +Z_disp) ----------
            ui_msg('底边搜索：第1段(GO→HI)慢速搜索中...')
            z_start = float(self.axis_cal.abs_to_z_disp(0, self.get_axis_copy(0).act_pos))
            vel_abs = float(v_z) * float(self.axis_cal.sign_eff(0))
            self._velmove_start_axis(0, vel_abs, acc=80.0, dec=80.0, jerk=300.0)

            t_search0 = time.time()
            hi_cnt = 0
            unk_cnt = 0
            edge1 = None
            last_go_z = float(z_start)
            last_move_z = None
            last_move_ts = time.time()

            while not stop_evt.is_set():
                ac0 = self.get_axis_copy(0)
                z_cur = float(self.axis_cal.abs_to_z_disp(0, ac0.act_pos))

                if d_max > 0.0 and (z_cur - z_start) >= (d_max - 1e-6):
                    ui_msg('底边搜索：第1段未找到(到达最大距离)')
                    break
                if (time.time() - t_search0) >= timeout_s:
                    ui_msg('底边搜索：第1段未找到(超时)')
                    break
                if int(getattr(ac0, 'err', 0) or 0) != 0:
                    ui_msg(f"底边搜索：AX0错误({int(getattr(ac0,'err',0) or 0)})")
                    break
                if _axis_not_moving_guard(z_cur):
                    ui_msg('底边搜索：AX0未运动(可能到达软限位/未进入速度模式)')
                    break

                r = _wait_new_judge(last_ts, 0.5)
                if r is None:
                    continue
                last_ts, j = r
                if j == 'UNK':
                    unk_cnt += 1
                    if unk_cnt >= 8:
                        ui_msg('底边搜索：未收到比较器(judge)字段，请确认请求为 M1,1')
                        break
                    continue
                unk_cnt = 0

                if j == 'GO':
                    last_go_z = float(z_cur)
                    hi_cnt = 0
                    continue

                if j in ('HI', 'HH'):
                    hi_cnt += 1
                    if hi_cnt >= max(1, int(deb_k)):
                        edge1 = float(last_go_z)
                        ui_msg(f"底边搜索：第1段锁定 {edge1:.3f} (GO→HI)")
                        break
                else:
                    hi_cnt = 0

            # stop motion always
            self._velmove_stop_axis(0)
            # wait axis to settle to avoid 1003 (axis stopping) on next velmove
            try:
                self._wait_axis_stop_settled(0, timeout_s=1.5, stable_cycles=8, eps_abs=0.02, stop_evt=stop_evt)
            except Exception:
                pass
            time.sleep(0.05)

            if stop_evt.is_set():
                ui_msg('底边搜索：已停止')
                return

            if edge1 is None:
                # message already set
                return

            # ---------- Pass 2: HI -> GO (move -Z_disp) ----------
            ui_msg('底边搜索：第2段(HI→GO)回扫中...')
            z_start2 = float(self.axis_cal.abs_to_z_disp(0, self.get_axis_copy(0).act_pos))
            vel_abs2 = -float(v_z) * float(self.axis_cal.sign_eff(0))
            self._velmove_start_axis(0, vel_abs2, acc=80.0, dec=80.0, jerk=300.0)

            t_search1 = time.time()
            go_cnt = 0
            unk_cnt = 0
            edge2 = None
            seen_hi = False
            last_hi_z = float(z_start2)
            last_move_z = None
            last_move_ts = time.time()

            while not stop_evt.is_set():
                ac0 = self.get_axis_copy(0)
                z_cur = float(self.axis_cal.abs_to_z_disp(0, ac0.act_pos))

                if d_max > 0.0 and (z_start2 - z_cur) >= (d_max - 1e-6):
                    ui_msg('底边搜索：第2段未找到(到达最大距离)')
                    break
                if (time.time() - t_search1) >= timeout_s:
                    ui_msg('底边搜索：第2段未找到(超时)')
                    break
                if int(getattr(ac0, 'err', 0) or 0) != 0:
                    ui_msg(f"底边搜索：AX0错误({int(getattr(ac0,'err',0) or 0)})")
                    break
                if _axis_not_moving_guard(z_cur):
                    ui_msg('底边搜索：AX0未运动(可能到达软限位/未进入速度模式)')
                    break

                r = _wait_new_judge(last_ts, 0.5)
                if r is None:
                    continue
                last_ts, j = r
                if j == 'UNK':
                    unk_cnt += 1
                    if unk_cnt >= 8:
                        ui_msg('底边搜索：未收到比较器(judge)字段，请确认请求为 M1,1')
                        break
                    continue
                unk_cnt = 0

                if not seen_hi:
                    if j in ('HI', 'HH'):
                        seen_hi = True
                        last_hi_z = float(z_cur)
                    continue

                if j in ('HI', 'HH'):
                    last_hi_z = float(z_cur)
                    go_cnt = 0
                    continue

                if j == 'GO':
                    go_cnt += 1
                    if go_cnt >= max(1, int(deb_k)):
                        edge2 = float(last_hi_z)
                        ui_msg(f"底边搜索：第2段锁定 {edge2:.3f} (HI→GO)")
                        break
                else:
                    go_cnt = 0

            # stop motion always
            self._velmove_stop_axis(0)
            # wait axis to settle to avoid 1003 (axis stopping) on next velmove
            try:
                self._wait_axis_stop_settled(0, timeout_s=1.5, stable_cycles=8, eps_abs=0.02, stop_evt=stop_evt)
            except Exception:
                pass
            time.sleep(0.05)

            if stop_evt.is_set():
                ui_msg('底边搜索：已停止')
                return

            if edge2 is None:
                return

            # Average
            edge_avg = 0.5 * (float(edge1) + float(edge2))
            ui_msg(f"底边搜索：锁定 {edge_avg:.3f} (双向均值)")

            try:
                if hasattr(self, 'len_edge_low_var'):
                    self._ui_set(self.len_edge_low_var, f"{float(edge_avg):.3f}")
                try:
                    self.after(0, self._len_try_update_measured_length)
                except Exception:
                    pass
            except Exception:
                pass

        except Exception as e:
            try:
                ui_msg(f"底边搜索：异常 {e}")
            except Exception:
                pass
            try:
                self._velmove_stop_axis(0)
            except Exception:
                pass
        finally:
            ui_done_btn()

    def _teach_len_search_high_worker(self, stop_evt: threading.Event) -> None:
        """Worker thread: top-edge bidirectional search (GO->HI then HI->GO) and lock AX0 Z_disp.

        机制说明：
        - 需要测径仪返回比较器判定字段(judge)，例如 GO/HI/LO。
        - 顶边判定采用双向扫描：
          1) 沿 -Z_disp 方向检测 GO→HI，锁定“最后一次GO”的 Z_disp 为 edge1；
          2) 反向沿 +Z_disp 方向检测 HI→GO，锁定“最后一次HI”的 Z_disp 为 edge2；
          3) 顶边 = (edge1 + edge2)/2。
        """

        def ui_msg(msg: str) -> None:
            try:
                if hasattr(self, 'len_edge_state_var'):
                    self._ui_set(self.len_edge_state_var, msg)
            except Exception:
                pass

        def ui_done_btn() -> None:
            try:
                btn = self._recipe_ui_widget('btn_len_search_high')
                if btn is not None:
                    self._ui_btn_text(btn, '尝试搜索顶边(GO→HI)')
            except Exception:
                pass

        def _wait_new_judge(ts0: float, tmax: float = 0.8):
            t0 = time.time()
            last_ts = float(ts0)
            while (not stop_evt.is_set()) and ((time.time() - t0) < float(tmax)):
                worker = gw
                if worker is None:
                    return None
                try:
                    worker.send_request()
                except Exception:
                    pass
                time.sleep(0.06)
                s = None
                try:
                    s = worker.get_last()
                except Exception:
                    s = None
                if s is None:
                    continue
                ts = float(getattr(s, 'ts', 0.0) or 0.0)
                if ts <= last_ts:
                    continue
                j = str(getattr(s, 'judge', 'UNK') or 'UNK').strip().upper()
                return ts, j
            return None

        def _axis_not_moving_guard(z_cur: float):
            nonlocal last_move_z, last_move_ts
            if last_move_z is None:
                last_move_z = float(z_cur)
                last_move_ts = time.time()
                return False
            if abs(float(z_cur) - float(last_move_z)) >= 0.15:
                last_move_z = float(z_cur)
                last_move_ts = time.time()
                return False
            if (time.time() - float(last_move_ts)) >= 1.0:
                return True
            return False

        edge_avg = None

        try:
            # --- validations ---
            if bool(getattr(self, 'sim_gauge_enabled', False)) or (hasattr(self, 'sim_gauge_var') and int(self.sim_gauge_var.get() or 0) == 1):
                ui_msg('顶边搜索：模拟测径仪不支持')
                return

            gw = getattr(self, 'gauge_worker', None)
            if gw is None or (not getattr(gw, 'enabled', False)):
                ui_msg('顶边搜索：请先连接测径仪(串口)')
                return

            # Require comparator mode (M1,1 / M0,1) so we can use judge(GO/HI) for edge detection
            try:
                req_cmd = str(getattr(gw, 'request_cmd', '') or '').strip().upper()
            except Exception:
                req_cmd = ''
            if (',1' not in req_cmd) and ('M1,1' not in req_cmd) and ('M0,1' not in req_cmd):
                ui_msg('顶边搜索：请将测径仪请求设为 M1,1 或 M0,1 (需包含比较器字段)')
                return

            # AX0 must be enabled
            ac0 = self.get_axis_copy(0)
            if int(getattr(ac0, 'sts', 0) or 0) == 0:
                ui_msg('顶边搜索：请先使能 AX0')
                return

            # Require bottom edge known
            if (not hasattr(self, 'len_edge_low_var')) or (str(self.len_edge_low_var.get()).strip() in ('', '--')):
                ui_msg('顶边搜索：请先搜索底边')
                return
            try:
                z_low_edge = float(str(self.len_edge_low_var.get()).strip())
            except Exception:
                ui_msg('顶边搜索：底边数据无效')
                return

            # Read parameters from UI/recipe
            def _f(var, d=0.0):
                try:
                    return float(var.get())
                except Exception:
                    try:
                        return float(var)
                    except Exception:
                        return float(d)

            pipe_len = max(0.0, _f(getattr(self, 'pipe_len_var', 0.0), 0.0))
            hi_margin = _f(getattr(self, 'len_high_margin_var', 0.0), 0.0)
            d_max = max(0.0, _f(getattr(self, 'len_high_search_dist_var', 0.0), 0.0))
            v_z = abs(_f(getattr(self, 'len_search_vel_var', 10.0), 10.0))
            timeout_s = max(1.0, _f(getattr(self, 'len_search_timeout_var', 8.0), 8.0))
            tol_z = max(0.1, _f(getattr(self, 'len_tol_var', 0.5), 0.5))
            backoff_mm = max(0.0, _f(getattr(self, 'len_backoff_var', 0.0), 0.0))

            deb_k = 2
            try:
                deb_k = int(float(getattr(self, 'len_debounce_k_var').get()))
            except Exception:
                deb_k = 2

            # Compute approach point for top edge (in Z_disp)
            if pipe_len <= 1e-6:
                ui_msg('顶边搜索：管长(配方)为0')
                return
            z_appr = float(z_low_edge - pipe_len + hi_margin)

            # Clamp to travel limits
            z_min, z_max, _travel = self._get_ax0_z_disp_limits()
            z_appr_clamped = max(float(z_min), min(float(z_max), float(z_appr)))
            if abs(float(z_appr_clamped) - float(z_appr)) > 1e-6:
                # if clamped to limit, we might not have space to scan further
                ui_msg('顶边搜索：接近位被行程限制裁剪，可能导致搜索失败(到限位后超时)')
            z_appr = float(z_appr_clamped)

            # Move to approach
            ui_msg('顶边搜索：移动到接近位...')
            # NOTE: top-edge approach is computed in Z_disp; convert to AX0 absolute position for MoveA.
            abs_tgt = float(self.axis_cal.z_disp_to_abs(0, float(z_appr)))
            self.movea_abs(0, abs_tgt, context='LenEdgeHighAppr')

            t0 = time.time()
            while (not stop_evt.is_set()) and (time.time() - t0 < 15.0):
                ac0 = self.get_axis_copy(0)
                if int(getattr(ac0, 'err', 0) or 0) != 0:
                    ui_msg(f"顶边搜索：AX0错误({int(getattr(ac0,'err',0) or 0)})")
                    return
                if abs(float(ac0.act_pos) - abs_tgt) <= max(0.5, tol_z):
                    break
                time.sleep(0.05)

            if stop_evt.is_set():
                ui_msg('顶边搜索：已停止')
                return

            if abs(float(self.get_axis_copy(0).act_pos) - abs_tgt) > max(0.8, tol_z * 2.0):
                ui_msg('顶边搜索：到达接近位超时')
                return

            # Pre-check: ensure we can get judge at approach and start in GO
            ui_msg('顶边搜索：确认比较器(GO/HI)...')
            last_ts = 0.0
            r = _wait_new_judge(last_ts, 1.5)
            if r is None:
                ui_msg('顶边搜索：未收到测径仪数据(请检查串口/请求指令)')
                return
            last_ts, j0 = r
            if j0 == 'UNK':
                ui_msg('顶边搜索：未收到比较器(judge)字段，请确认请求为 M1,1')
                return
            if j0 != 'GO':
                ui_msg(f'顶边搜索：起点不是GO({j0})，请调整接近位/比较器阈值')
                return

            # ---------- Pass 1: GO -> HI (move -Z_disp) ----------
            ui_msg('顶边搜索：第1段(GO→HI)慢速搜索中...')
            z_start = float(self.axis_cal.abs_to_z_disp(0, self.get_axis_copy(0).act_pos))
            vel_abs = -float(v_z) * float(self.axis_cal.sign_eff(0))
            self._velmove_start_axis(0, vel_abs, acc=80.0, dec=80.0, jerk=300.0)

            t_search0 = time.time()
            unk_cnt = 0
            hi_cnt = 0
            edge1 = None
            last_go_z = float(z_start)
            last_move_z = None
            last_move_ts = time.time()

            while not stop_evt.is_set():
                ac0 = self.get_axis_copy(0)
                z_cur = float(self.axis_cal.abs_to_z_disp(0, ac0.act_pos))

                if d_max > 0.0 and (z_start - z_cur) >= (d_max - 1e-6):
                    ui_msg('顶边搜索：第1段未找到(到达最大距离)')
                    break
                if (time.time() - t_search0) >= timeout_s:
                    ui_msg('顶边搜索：第1段未找到(超时)')
                    break
                if int(getattr(ac0, 'err', 0) or 0) != 0:
                    ui_msg(f"顶边搜索：AX0错误({int(getattr(ac0,'err',0) or 0)})")
                    break
                if _axis_not_moving_guard(z_cur):
                    ui_msg('顶边搜索：AX0未运动(可能到达软限位/未进入速度模式)')
                    break

                r = _wait_new_judge(last_ts, 0.5)
                if r is None:
                    continue
                last_ts, j = r
                if j == 'UNK':
                    unk_cnt += 1
                    if unk_cnt >= 8:
                        ui_msg('顶边搜索：未收到比较器(judge)字段，请确认请求为 M1,1')
                        break
                    continue
                unk_cnt = 0

                if j == 'GO':
                    last_go_z = float(z_cur)
                    hi_cnt = 0
                    continue

                if j in ('HI', 'HH'):
                    hi_cnt += 1
                    if hi_cnt >= max(1, int(deb_k)):
                        edge1 = float(last_go_z)
                        ui_msg(f"顶边搜索：第1段锁定 {edge1:.3f} (GO→HI)")
                        break
                else:
                    hi_cnt = 0

            self._velmove_stop_axis(0)
            # wait axis to settle to avoid 1003 (axis stopping) on next velmove
            try:
                self._wait_axis_stop_settled(0, timeout_s=1.5, stable_cycles=8, eps_abs=0.02, stop_evt=stop_evt)
            except Exception:
                pass
            time.sleep(0.05)

            if stop_evt.is_set():
                ui_msg('顶边搜索：已停止')
                return

            if edge1 is None:
                return

            # ---------- Pass 2: HI -> GO (move +Z_disp) ----------
            ui_msg('顶边搜索：第2段(HI→GO)回扫中...')
            z_start2 = float(self.axis_cal.abs_to_z_disp(0, self.get_axis_copy(0).act_pos))
            vel_abs2 = float(v_z) * float(self.axis_cal.sign_eff(0))
            self._velmove_start_axis(0, vel_abs2, acc=80.0, dec=80.0, jerk=300.0)

            t_search1 = time.time()
            unk_cnt = 0
            go_cnt = 0
            edge2 = None
            seen_hi = False
            last_hi_z = float(z_start2)
            last_move_z = None
            last_move_ts = time.time()

            while not stop_evt.is_set():
                ac0 = self.get_axis_copy(0)
                z_cur = float(self.axis_cal.abs_to_z_disp(0, ac0.act_pos))

                if d_max > 0.0 and (z_cur - z_start2) >= (d_max - 1e-6):
                    ui_msg('顶边搜索：第2段未找到(到达最大距离)')
                    break
                if (time.time() - t_search1) >= timeout_s:
                    ui_msg('顶边搜索：第2段未找到(超时)')
                    break
                if int(getattr(ac0, 'err', 0) or 0) != 0:
                    ui_msg(f"顶边搜索：AX0错误({int(getattr(ac0,'err',0) or 0)})")
                    break
                if _axis_not_moving_guard(z_cur):
                    ui_msg('顶边搜索：AX0未运动(可能到达软限位/未进入速度模式)')
                    break

                r = _wait_new_judge(last_ts, 0.5)
                if r is None:
                    continue
                last_ts, j = r
                if j == 'UNK':
                    unk_cnt += 1
                    if unk_cnt >= 8:
                        ui_msg('顶边搜索：未收到比较器(judge)字段，请确认请求为 M1,1')
                        break
                    continue
                unk_cnt = 0

                if not seen_hi:
                    if j in ('HI', 'HH'):
                        seen_hi = True
                        last_hi_z = float(z_cur)
                    continue

                if j in ('HI', 'HH'):
                    last_hi_z = float(z_cur)
                    go_cnt = 0
                    continue

                if j == 'GO':
                    go_cnt += 1
                    if go_cnt >= max(1, int(deb_k)):
                        edge2 = float(last_hi_z)
                        ui_msg(f"顶边搜索：第2段锁定 {edge2:.3f} (HI→GO)")
                        break
                else:
                    go_cnt = 0

            self._velmove_stop_axis(0)
            # wait axis to settle to avoid 1003 (axis stopping) on next velmove
            try:
                self._wait_axis_stop_settled(0, timeout_s=1.5, stable_cycles=8, eps_abs=0.02, stop_evt=stop_evt)
            except Exception:
                pass
            time.sleep(0.05)

            if stop_evt.is_set():
                ui_msg('顶边搜索：已停止')
                return

            if edge2 is None:
                return

            edge_avg = 0.5 * (float(edge1) + float(edge2))
            ui_msg(f"顶边搜索：锁定 {edge_avg:.3f} (双向均值)")

            try:
                if hasattr(self, 'len_edge_high_var'):
                    self._ui_set(self.len_edge_high_var, f"{float(edge_avg):.3f}")
            except Exception:
                pass

            # Optional backoff to stay inside the tube (towards +Z_disp)
            if backoff_mm > 1e-6:
                try:
                    z_back = max(float(z_min), min(float(z_max), float(edge_avg) + float(backoff_mm)))
                    self.movea_abs(0, float(self.axis_cal.z_disp_to_abs(0, z_back)), context='LenEdgeHighBackoff')
                except Exception:
                    pass

            # Update measured length if possible
            try:
                self.after(0, self._len_try_update_measured_length)
            except Exception:
                pass

        except Exception as e:
            try:
                ui_msg(f"顶边搜索：异常 {e}")
            except Exception:
                pass
            try:
                self._velmove_stop_axis(0)
            except Exception:
                pass
        finally:
            ui_done_btn()
