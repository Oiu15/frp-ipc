from __future__ import annotations

"""Operator and flow confirmation mixin for AppHost."""

import queue
import threading
import uuid
import tkinter as tk
from tkinter import ttk
from collections.abc import Mapping
from typing import Any, Callable, cast

from controllers.measurement_controller import MeasurementController
from events.types import OpConfirmShowEvent, OpConfirmCloseEvent
from utils.logger import log_exc


class HostConfirmMixin:
    """Mixin providing operator/flow confirm popups and UI event handlers."""

    ui_q: queue.Queue[Any]
    measurement_controller: MeasurementController
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

