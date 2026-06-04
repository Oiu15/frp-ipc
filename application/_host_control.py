from __future__ import annotations

"""Control mixin for AppHost — operator confirm dialogs, keytest, stack light.

Extracted from ``app_host.py``.  Groups B + C are extracted together
because they are mutually dependent.
"""

import queue
import threading
import time
import uuid
import tkinter as tk
from tkinter import ttk
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, cast

from config.addresses import KEYTEST_X_POINTS, KEYTEST_Y_POINTS, KEYTEST_Y_BASE_COIL
from controllers.measurement_controller import MeasurementController
from drivers.plc_client import CmdWriteCoil
from events.types import OpConfirmShowEvent, OpConfirmCloseEvent
from utils.logger import log_exc


class HostControlMixin:
    """Mixin providing operator confirm popups, keytest IO, and stack light control.

    Requires the AppHost to have:
      - self.ui_q, self.cmd_q (queue.Queue)
      - self.measurement_controller
      - self.plc_status_var, self.auto_state_var (tk.StringVar)
      - self.keytest_x_vars, keytest_y_vars, keytest_y_lastcmd_vars (lists of tk.Variable)
      - self._notebook, self._tab_main
      - self._is_auto_thread_alive()
    """

    ui_q: queue.Queue[Any]
    cmd_q: queue.Queue[Any]
    measurement_controller: MeasurementController

    plc_status_var: tk.StringVar
    auto_state_var: tk.StringVar
    keytest_x_vars: list[tk.IntVar]
    keytest_y_vars: list[tk.IntVar]
    keytest_y_lastcmd_vars: list[tk.StringVar]
    _notebook: ttk.Notebook
    _tab_main: ttk.Frame

    _keytest_bits_lock: threading.Lock
    _keytest_x_points_state: list[int]
    _keytest_y_points_state: list[int]
    _keytest_y_points_has_read: bool
    _keytest_y_last_command_state: list[int]

    _flow_confirm_lock: threading.Lock
    _flow_confirm_token: str | None
    _flow_confirm_evt: threading.Event | None
    _flow_confirm_result: str | None
    _flow_confirm_popup: tk.Toplevel | None
    _flow_confirm_confirm_cb: Callable[[], Any] | None
    _flow_confirm_cancel_cb: Callable[[], Any] | None

    _op_confirm_lock: threading.Lock
    _op_confirm_token: str | None
    _op_confirm_evt: threading.Event | None
    _op_confirm_result: str | None
    _op_confirm_popup: tk.Toplevel | None

    _stack_light_state: str | None
    _stack_light_buzzer_after_id: str | None

    after: Callable[..., str]
    after_cancel: Callable[[str], None]

    if TYPE_CHECKING:
        def _is_auto_thread_alive(self) -> bool: ...

    def write_keytest_y(self, y_point: int, value: int) -> None:
        self._keytest_write_y(y_point, value)

    def plc_write_y_point(self, y_point: int, value: int) -> None:
        """Thread-safe one-shot write to a Y point (Modbus coil). Safe to call from any thread."""
        try:
            y_point = int(y_point)
            value = 1 if int(value) != 0 else 0
            # X/Y points are octal labels: no 8/9 in coil address space.
            if y_point < 8:
                idx = y_point
            else:
                idx = y_point - 2  # skip 8/9
            coil = int(KEYTEST_Y_BASE_COIL) + int(idx)
            self.cmd_q.put(CmdWriteCoil(coil_addr=coil, value=value))
            try:
                i = KEYTEST_Y_POINTS.index(y_point)
                with self._keytest_bits_lock:
                    self._keytest_y_last_command_state[i] = value
            except Exception:
                pass
        except Exception:
            pass

    def write_coil(self, coil_addr: int, value: int | bool) -> None:
        """Thread-safe one-shot write to a raw Modbus coil address."""
        try:
            coil = int(coil_addr)
            vv = 1 if bool(value) else 0
            self.cmd_q.put(CmdWriteCoil(coil_addr=coil, value=vv))
        except Exception:
            pass

    def get_x_point(self, x_point: int) -> int:
        """Get cached X point value (0/1). Safe to call from any thread."""
        try:
            x_point = int(x_point)
            i = KEYTEST_X_POINTS.index(x_point)
        except Exception:
            return 0
        try:
            with self._keytest_bits_lock:
                arr = self._keytest_x_points_state
                return int(arr[i]) if 0 <= i < len(arr) else 0
        except Exception:
            return 0

    def get_y_point(self, y_point: int) -> int:
        """Get cached Y point value (0/1), falling back to the last IPC command."""
        try:
            y_point = int(y_point)
            i = KEYTEST_Y_POINTS.index(y_point)
        except Exception:
            return 0
        try:
            with self._keytest_bits_lock:
                arr = self._keytest_y_points_state
                if self._keytest_y_points_has_read and 0 <= i < len(arr):
                    v = int(arr[i])
                    if v in (0, 1):
                        return v
                cmd = self._keytest_y_last_command_state
                return int(cmd[i]) if 0 <= i < len(cmd) else 0
        except Exception:
            return 0

    def _clamps_are_closed(self) -> bool:
        return bool(self.get_y_point(10) == 1 and self.get_y_point(11) == 1)

    def _clamps_are_open(self) -> bool:
        return bool(self.get_y_point(10) == 0 and self.get_y_point(11) == 0)

    def _release_clamps_from_key(self) -> None:
        try:
            if self._clamps_are_open():
                self.plc_status_var.set("夹爪已松开")
                return
            self.plc_write_y_point(10, 0)
            self.plc_write_y_point(11, 0)
            self.plc_status_var.set("夹爪松开")
        except Exception:
            pass

    def set_stack_light(self, state: str) -> None:
        target = str(state or "IDLE_OR_READY").strip().upper()
        if target not in {"RUNNING", "ERROR_OR_ESTOP", "IDLE_OR_READY"}:
            target = "IDLE_OR_READY"
        try:
            if self._stack_light_state == target:
                return
            self._stack_light_state = target

            # Red/Y4, yellow/Y5, green/Y6 are mutually exclusive.
            for y in (4, 5, 6):
                self.plc_write_y_point(y, 0)
            if target == "ERROR_OR_ESTOP":
                self.plc_write_y_point(4, 1)
                self.plc_write_y_point(7, 1)
                try:
                    after_id = self._stack_light_buzzer_after_id
                    if after_id is not None:
                        self.after_cancel(after_id)
                except Exception:
                    pass
                try:
                    self._stack_light_buzzer_after_id = self.after(1000, lambda: self.plc_write_y_point(7, 0))
                except Exception:
                    self.plc_write_y_point(7, 0)
            elif target == "RUNNING":
                self.plc_write_y_point(6, 1)
            else:
                self.plc_write_y_point(5, 1)
        except Exception:
            pass

    def _refresh_stack_light_for_state(self, auto_state: str | None = None) -> None:
        try:
            if int(self.get_x_point(0)) == 0:
                self.set_stack_light("ERROR_OR_ESTOP")
                return
        except Exception:
            pass
        st = str(auto_state if auto_state is not None else self.auto_state_var.get()).strip().upper()
        if st in {"RUN", "PREP", "LEN"}:
            self.set_stack_light("RUNNING")
        elif st == "ERR":
            self.set_stack_light("ERROR_OR_ESTOP")
        else:
            self.set_stack_light("IDLE_OR_READY")

    def _keytest_write_y(self, y_point: int, value: int) -> None:
        """One-shot write to Y coil.

        - 写入与状态显示分离：写入后是否生效，以读回状态为准。
        - 不做持续写入，避免与 PLC 内部逻辑冲突。
        """
        try:
            y_point = int(y_point)
            value = 1 if int(value) != 0 else 0

            self.plc_write_y_point(y_point, value)
            # record last cmd
            try:
                idx = KEYTEST_Y_POINTS.index(y_point)
                ts = time.strftime("%H:%M:%S")
                self.keytest_y_lastcmd_vars[idx].set(f"写{value} @{ts}")
            except Exception:
                pass
        except Exception:
            pass

    def _keytest_apply_bits(self, x_bits, y_bits) -> None:
        """Update UI and cached X/Y states from polled coil bits.

        Also handles edge shortcuts:
        - X2 rising: start AutoFlow (same as clicking 'Start')
        - X3 rising: confirm the active operator dialog (if any)
        """
        try:
            self._keytest_x_bits = x_bits
            self._keytest_y_bits = y_bits

            cur_x = [0 for _ in range(len(KEYTEST_X_POINTS))]
            cur_y = [0 for _ in range(len(KEYTEST_Y_POINTS))]

            if isinstance(x_bits, (list, tuple)):
                for i, p in enumerate(KEYTEST_X_POINTS):
                    try:
                        pp = int(p)
                        idx = pp if pp < 8 else pp - 2
                        v = 1 if bool(x_bits[int(idx)]) else 0
                        cur_x[i] = v
                        self.keytest_x_vars[i].set(v)
                    except Exception:
                        pass

            if isinstance(y_bits, (list, tuple)):
                for i, p in enumerate(KEYTEST_Y_POINTS):
                    try:
                        pp = int(p)
                        idx = pp if pp < 8 else pp - 2
                        v = 1 if bool(y_bits[int(idx)]) else 0
                        cur_y[i] = v
                        self.keytest_y_vars[i].set(v)
                    except Exception:
                        pass

            start_edge = False
            confirm_edge = False
            cancel_edge = False
            with self._keytest_bits_lock:
                prev_x = list(self._keytest_x_points_state)
                self._keytest_x_points_state = list(cur_x)
                self._keytest_y_points_state = list(cur_y)
                if isinstance(y_bits, (list, tuple)):
                    self._keytest_y_points_has_read = True

            try:
                i2 = KEYTEST_X_POINTS.index(2)
                if i2 < len(prev_x):
                    start_edge = (prev_x[i2] == 0) and (cur_x[i2] == 1)
            except Exception:
                pass
            try:
                i3 = KEYTEST_X_POINTS.index(3)
                if i3 < len(prev_x):
                    confirm_edge = (prev_x[i3] == 0) and (cur_x[i3] == 1)
            except Exception:
                pass
            try:
                i4 = KEYTEST_X_POINTS.index(4)
                if i4 < len(prev_x):
                    cancel_edge = (prev_x[i4] == 0) and (cur_x[i4] == 1)
            except Exception:
                pass

            if start_edge:
                self._handle_x2_edge()

            if confirm_edge:
                self._handle_x3_edge()

            if cancel_edge:
                self._handle_x4_edge()
        except Exception:
            pass

    def _is_main_tab_selected(self) -> bool:
        try:
            nb = getattr(self, "_notebook", None)
            tab_main = getattr(self, "_tab_main", None)
            return bool(nb is not None and tab_main is not None and nb.select() == str(tab_main))
        except Exception:
            return True

    def _is_flow_confirm_active(self) -> bool:
        try:
            with self._flow_confirm_lock:
                return bool(self._flow_confirm_token)
        except Exception:
            return False

    def _show_flow_confirm_popup(
        self,
        *,
        token: str | None = None,
        title: str,
        message: str,
        confirm_text: str = "确认",
        cancel_text: str = "取消",
        on_confirm=None,
        on_cancel=None,
    ) -> None:
        try:
            token = str(token or uuid.uuid4())
            with self._flow_confirm_lock:
                self._flow_confirm_token = token
                self._flow_confirm_result = None
                self._flow_confirm_confirm_cb = on_confirm
                self._flow_confirm_cancel_cb = on_cancel

            try:
                if self._flow_confirm_popup is not None and self._flow_confirm_popup.winfo_exists():
                    self._flow_confirm_popup.destroy()
            except Exception:
                pass

            host = cast(tk.Tk, self)
            top = tk.Toplevel(host)
            self._flow_confirm_popup = top
            top.title(title or "确认")
            top.transient(host)
            try:
                top.grab_set()
            except Exception:
                pass

            frm = ttk.Frame(top, padding=12)
            frm.pack(fill="both", expand=True)
            ttk.Label(frm, text=message or "", wraplength=520, justify="left").pack(fill="x", pady=(0, 8))
            ttk.Label(frm, text="X3 = 确认，X4 = 取消", foreground="#666").pack(fill="x", pady=(0, 10))

            row = ttk.Frame(frm)
            row.pack(fill="x")
            ttk.Button(row, text=f"{confirm_text} (X3)", command=lambda: self._flow_confirm_set("confirm", token=token)).pack(side="left", padx=(0, 8))
            ttk.Button(row, text=f"{cancel_text} (X4)", command=lambda: self._flow_confirm_set("cancel", token=token)).pack(side="left")

            top.protocol("WM_DELETE_WINDOW", lambda: self._flow_confirm_set("cancel", token=token))
            top.bind("<Return>", lambda _e: self._flow_confirm_set("confirm", token=token))
            top.bind("<Escape>", lambda _e: self._flow_confirm_set("cancel", token=token))
        except Exception:
            try:
                if callable(on_cancel):
                    on_cancel()
            except Exception:
                pass

    def _flow_confirm_set(self, result: str, token: str | None = None) -> bool:
        cb = None
        try:
            with self._flow_confirm_lock:
                cur = self._flow_confirm_token
                if token is not None and cur is not None and str(token) != str(cur):
                    return False
                if not cur:
                    return False
                res = "confirm" if result == "confirm" else "cancel"
                self._flow_confirm_result = res
                evt = self._flow_confirm_evt
                cb = self._flow_confirm_confirm_cb if res == "confirm" else self._flow_confirm_cancel_cb
                self._flow_confirm_token = None
                self._flow_confirm_evt = None
                self._flow_confirm_confirm_cb = None
                self._flow_confirm_cancel_cb = None
            try:
                pop = self._flow_confirm_popup
                if pop is not None and pop.winfo_exists():
                    pop.destroy()
            except Exception:
                pass
            self._flow_confirm_popup = None
            if evt is not None:
                try:
                    evt.set()
                except Exception:
                    pass
            if callable(cb):
                try:
                    cb()
                except Exception:
                    pass
            return True
        except Exception:
            return False

    def flow_confirm(
        self,
        title: str,
        message: str,
        *,
        confirm_text: str = "确认",
        cancel_text: str = "取消",
        timeout_s: float | None = None,
    ) -> str:
        """Thread-safe flow confirmation dialog for hardware-key decisions.

        Returns: 'confirm' | 'cancel' | 'timeout'.
        """
        try:
            if threading.current_thread() is threading.main_thread():
                self._show_flow_confirm_popup(
                    title=title,
                    message=message,
                    confirm_text=confirm_text,
                    cancel_text=cancel_text,
                )
                return "timeout"

            token = str(uuid.uuid4())
            evt = threading.Event()
            with self._flow_confirm_lock:
                self._flow_confirm_token = token
                self._flow_confirm_evt = evt
                self._flow_confirm_result = None
                self._flow_confirm_confirm_cb = None
                self._flow_confirm_cancel_cb = None

            self.ui_q.put((
                "flow_confirm_show",
                {
                    "token": token,
                    "title": title,
                    "message": message,
                    "confirm_text": confirm_text,
                    "cancel_text": cancel_text,
                },
            ))

            if timeout_s is None:
                evt.wait()
            elif not evt.wait(float(timeout_s)):
                with self._flow_confirm_lock:
                    if self._flow_confirm_token == token and self._flow_confirm_result is None:
                        self._flow_confirm_result = "timeout"
                        self._flow_confirm_token = None
                        self._flow_confirm_evt = None
                        try:
                            evt.set()
                        except Exception:
                            pass
                self.ui_q.put(("flow_confirm_close", {"token": token}))

            with self._flow_confirm_lock:
                res = str(self._flow_confirm_result or "timeout")
                if self._flow_confirm_token in (None, token):
                    self._flow_confirm_token = None
                    self._flow_confirm_evt = None
                    self._flow_confirm_result = None
                    self._flow_confirm_confirm_cb = None
                    self._flow_confirm_cancel_cb = None
            return res if res in ("confirm", "cancel", "timeout") else "timeout"
        except Exception as exc:
            log_exc(
                f"FLOW_CONFIRM_ERROR title={str(title)[:80]} message={str(message)[:120]}",
                exc,
            )
            return "timeout"

    def _handle_x2_edge(self) -> None:
        try:
            if self._is_auto_thread_alive():
                self.measurement_controller.start_measurement()
                return
            if self._is_main_tab_selected():
                self.measurement_controller.start_measurement()
                return
            self._show_flow_confirm_popup(
                title="启动自动测量",
                message="当前不在主测量页面，是否仍要启动自动测量？",
                confirm_text="启动",
                cancel_text="取消",
                on_confirm=lambda: self.measurement_controller.start_measurement(),
            )
        except Exception:
            pass

    def _handle_x3_edge(self) -> None:
        try:
            if self._flow_confirm_set("confirm"):
                return
            self._op_confirm_set("confirm")
        except Exception:
            pass

    def _handle_x4_edge(self) -> None:
        try:
            if self._flow_confirm_set("cancel"):
                return
            if self._op_confirm_set("stop"):
                return
            if self._is_auto_thread_alive():
                self.measurement_controller.stop_measurement()
                return
            self._release_clamps_from_key()
        except Exception:
            pass

    # =========================
    # Operator confirm (no clamp feedback)
    # =========================
    def operator_confirm(self, title: str, message: str, *, allow_stop: bool = True, timeout_s: float | None = None) -> str:
        """Block current (non-UI) thread until operator confirms or stops.

        Returns: 'confirm' | 'stop' | 'timeout'.
        """
        try:
            res = self.flow_confirm(
                title or "确认",
                message,
                confirm_text="确认",
                cancel_text="停止" if allow_stop else "取消",
                timeout_s=timeout_s,
            )
            if res == "cancel":
                return "stop"
            return str(res)
        except Exception:
            return 'timeout'

    def _show_op_confirm_popup(self, token: str, title: str, message: str, allow_stop: bool) -> None:
        try:
            # close previous if any
            try:
                if self._op_confirm_popup is not None and self._op_confirm_popup.winfo_exists():
                    self._op_confirm_popup.destroy()
            except Exception:
                pass

            host = cast(tk.Tk, self)
            top = tk.Toplevel(host)
            self._op_confirm_popup = top
            top.title(title or '操作员确认')
            top.transient(host)
            try:
                top.grab_set()
            except Exception:
                pass

            frm = ttk.Frame(top, padding=12)
            frm.pack(fill='both', expand=True)

            lbl = ttk.Label(frm, text=message or '', wraplength=520, justify='left')
            lbl.pack(fill='x', pady=(0, 8))

            hint = ttk.Label(frm, text='提示：可按 X3 进行确认。', foreground='#666')
            hint.pack(fill='x', pady=(0, 10))

            btn_row = ttk.Frame(frm)
            btn_row.pack(fill='x')

            def _on_confirm():
                self._op_confirm_set('confirm', token=token)

            def _on_stop():
                try:
                    self.measurement_controller.stop_measurement()
                except Exception:
                    pass
                self._op_confirm_set('stop', token=token)

            b1 = ttk.Button(btn_row, text='确认夹紧 (X3)', command=_on_confirm)
            b1.pack(side='left', padx=(0, 8))

            if allow_stop:
                b2 = ttk.Button(btn_row, text='停止流程', command=_on_stop)
                b2.pack(side='left')

            def _on_close():
                # treat closing as stop
                _on_stop()

            try:
                top.protocol('WM_DELETE_WINDOW', _on_close)
                top.bind('<Return>', lambda _e: _on_confirm())
                top.bind('<Escape>', lambda _e: _on_stop())
            except Exception:
                pass

            try:
                b1.focus_set()
            except Exception:
                pass

        except Exception:
            pass

    def _close_op_confirm_popup(self, token: str) -> None:
        try:
            with self._op_confirm_lock:
                cur = self._op_confirm_token
            if token and cur and token != cur:
                return
            pop = self._op_confirm_popup
            if pop is not None and pop.winfo_exists():
                try:
                    pop.grab_release()
                except Exception:
                    pass
                try:
                    pop.destroy()
                except Exception:
                    pass
            self._op_confirm_popup = None
        except Exception:
            pass

    def _op_confirm_set(self, result: str, token: str | None = None) -> bool:
        try:
            with self._op_confirm_lock:
                cur = self._op_confirm_token
                evt = self._op_confirm_evt
                if cur is None:
                    return False
                if token is not None and token != cur:
                    return False
                self._op_confirm_result = str(result)
            try:
                if evt is not None:
                    evt.set()
            except Exception:
                pass
            # Close popup on UI thread
            try:
                pop = self._op_confirm_popup
                if pop is not None and pop.winfo_exists():
                    try:
                        pop.grab_release()
                    except Exception:
                        pass
                    pop.destroy()
            except Exception:
                pass
            self._op_confirm_popup = None
            return True
        except Exception:
            return False


    def _handle_op_confirm_show_event(self, event: OpConfirmShowEvent) -> None:
        payload = event.to_payload()
        try:
            self._show_op_confirm_popup(
                token=str(payload.get('token', '')),
                title=str(payload.get('title', '操作员确认')),
                message=str(payload.get('message', '')),
                allow_stop=bool(payload.get('allow_stop', True)),
            )
        except Exception:
            pass

    def _handle_op_confirm_close_event(self, event: OpConfirmCloseEvent) -> None:
        payload = event.to_payload()
        try:
            self._close_op_confirm_popup(str(payload.get('token', '')))
        except Exception:
            pass

    def _handle_flow_confirm_show_event(self, payload: Any) -> None:
        try:
            data = dict(payload or {}) if isinstance(payload, Mapping) else {}
            self._show_flow_confirm_popup(
                token=str(data.get("token", "")),
                title=str(data.get("title", "确认")),
                message=str(data.get("message", "")),
                confirm_text=str(data.get("confirm_text", "确认")),
                cancel_text=str(data.get("cancel_text", "取消")),
            )
        except Exception:
            pass

    def _handle_flow_confirm_close_event(self, payload: Any) -> None:
        try:
            data = dict(payload or {}) if isinstance(payload, Mapping) else {}
            token = str(data.get("token", ""))
            with self._flow_confirm_lock:
                cur = self._flow_confirm_token
            if token and cur and token != cur:
                return
            pop = self._flow_confirm_popup
            if pop is not None and pop.winfo_exists():
                try:
                    pop.grab_release()
                except Exception:
                    pass
                pop.destroy()
            self._flow_confirm_popup = None
        except Exception:
            pass

