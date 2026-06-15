from __future__ import annotations

"""Keytest, hardware-key edge dispatch, and stack-light mixin for AppHost."""

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Any, Callable

from config.addresses import KEYTEST_X_POINTS, KEYTEST_Y_POINTS, KEYTEST_Y_BASE_COIL
from core.plc_commands import CmdWriteCoil
if TYPE_CHECKING:
    from services.measurement_service import MeasurementController


class HostKeytestMixin:
    """Mixin providing keytest X/Y IO, hardware-key shortcuts, and stack light control."""

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
    _stack_light_state: str | None
    _stack_light_buzzer_after_id: str | None
    after: Callable[..., str]
    after_cancel: Callable[[str], None]

    if TYPE_CHECKING:
        def _is_auto_thread_alive(self) -> bool: ...
        def _show_flow_confirm_popup(
            self,
            *,
            token: str | None = None,
            title: str,
            message: str,
            confirm_text: str = "确认",
            cancel_text: str = "取消",
            on_confirm: Any = None,
            on_cancel: Any = None,
        ) -> None: ...
        def _flow_confirm_set(self, result: str, token: str | None = None) -> bool: ...
        def _op_confirm_set(self, result: str, token: str | None = None) -> bool: ...

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
        state: str | None = auto_state
        if state is None:
            session = getattr(self, "_run_session", None)
            state = session.status if session is not None else "IDLE"
        st = str(state).strip().upper()
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
