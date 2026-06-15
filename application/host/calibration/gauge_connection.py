from __future__ import annotations

"""Gauge connection mixin for AppHost."""

import tkinter as tk
from tkinter import messagebox
from typing import TYPE_CHECKING, Any, List

from config.addresses import DEFAULT_GAUGE_PORT
# GaugeWorker and list_serial_ports imported lazily inside _list_serial_ports()
# to eliminate the application/host -> drivers dependency.


class HostGaugeConnectionMixin:
    """Mixin providing gauge port discovery, connection, and request controls."""

    gauge_worker: Any | None  # concrete GaugeWorker — now accessed lazily
    baud_var: tk.StringVar
    req_cmd_var: tk.StringVar
    gauge_conn_var: tk.StringVar
    gauge_err_var: tk.StringVar
    sim_gauge_var: tk.IntVar
    sim_disp_var: tk.IntVar
    sim_gauge_enabled: bool
    sim_disp_enabled: bool

    if TYPE_CHECKING:
        def _gauge_ui_widget(self, name: str) -> Any: ...

    def _auto_connect_gauge(self):
        """Startup auto-connect gauge once (COM2). Fail -> no retry."""
        gauge_worker = self.gauge_worker
        if gauge_worker is None:
            return
        port = "COM2"
        try:
            baud = int(self.baud_var.get().strip() or "115200")
        except Exception:
            baud = 115200

        # 给UI一个立即反馈（不依赖线程回报）
        self.gauge_conn_var.set(f"串口: 连接中... ({port}@{baud})")
        self.gauge_err_var.set("")

        try:
            gauge_worker.configure(
                enabled=True,
                port=port,
                baud=baud,
                timeout_s=0.5,
                eol="\r",
                request_cmd=self.req_cmd_var.get().strip() or "M1,1",
                bytesize=8,
                parity="N",
                stopbits=1,
            )
        except Exception as e:
            # 失败：禁用worker
            try:
                gauge_worker.configure(
                    enabled=False,
                    port="",
                    baud=115200,
                    timeout_s=0.5,
                    eol="\r",
                    request_cmd="",
                )
            except Exception:
                pass
            self.gauge_conn_var.set("串口: 未连接")
            self.gauge_err_var.set(f"启动自动连接失败: {e}")

    def refresh_gauge_ports(self):
        return self._refresh_ports()

    def toggle_sim_gauge(self):
        return self._on_sim_gauge_toggle()

    def connect_gauge(self):
        return self._gauge_connect()

    def disconnect_gauge(self):
        return self._gauge_disconnect()

    def request_gauge_once(self):
        return self._gauge_request_once()

    def _on_sim_gauge_toggle(self):
        self.sim_gauge_enabled = bool(self.sim_gauge_var.get())

    def _on_sim_disp_toggle(self):
        """Toggle simulated displacement meter (ID).

        Current phase: simulation only.
        """
        try:
            self.sim_disp_enabled = bool(self.sim_disp_var.get())
        except Exception:
            self.sim_disp_enabled = False

    def _list_serial_ports(self) -> List[str]:
        """Return list of available serial ports."""
        from drivers.gauge_driver import list_serial_ports  # lazy import
        return list_serial_ports()

    def _refresh_ports(self):
        ports = self._list_serial_ports()
        combo = self._gauge_ui_widget('port_combo')
        if combo is None:
            return
        combo.configure(values=ports)

        cur = (combo.get() or "").strip()

        if not ports:
            combo.set(DEFAULT_GAUGE_PORT)
            return

        if cur and (cur in ports):
            combo.set(cur)
            return

        if DEFAULT_GAUGE_PORT in ports:
            combo.set(DEFAULT_GAUGE_PORT)
            return

        combo.set(ports[0])

    def _gauge_connect(self):
        """连接测径仪（只在需要时打开串口）。
        说明：
        - 重复点击“连接”不会重复 open 串口，只会更新参数（避免 Windows 下 PermissionError(13)）。
        - 会自动关闭“模拟测径仪”开关。
        """
        try:
            gauge_worker = self.gauge_worker
            if gauge_worker is None:
                self.gauge_conn_var.set("Serial: unavailable")
                self.gauge_err_var.set("Gauge worker not available")
                return
            # if serial is None:
            #    raise RuntimeError("pyserial 未安装。")

            combo = self._gauge_ui_widget('port_combo')
            port = (combo.get().strip() if combo is not None else '') or DEFAULT_GAUGE_PORT
            baud = int(self.baud_var.get().strip() or "115200")
            cmd = (self.req_cmd_var.get() or "M1,1").strip()

            # 选择真实测径仪时，自动关闭模拟
            self.sim_gauge_var.set(0)
            self.sim_gauge_enabled = False

            # UI 立即给一个“连接中”的可见反馈；真正成功/失败由 gauge_conn/gauge_err 更新
            self.gauge_conn_var.set(f"串口: 连接中... ({port}@{baud})")
            self.gauge_err_var.set("")

            gauge_worker.configure(
                enabled=True,
                port=port,
                baud=baud,
                timeout_s=0.5,
                eol="\r",
                request_cmd=cmd,
                bytesize=8,
                parity="N",
                stopbits=1,
            )
        except Exception as e:
            self.gauge_conn_var.set("串口: 未连接")
            messagebox.showerror("连接测径仪失败", str(e))

    def _gauge_disconnect(self):
        """断开测径仪串口。"""
        try:
            gauge_worker = self.gauge_worker
            if gauge_worker is None:
                self.gauge_conn_var.set("Serial: unavailable")
                return
            gauge_worker.configure(
                enabled=False,
                port="",
                baud=115200,
                timeout_s=0.5,
                eol="\r",
                request_cmd="",
            )
            self.gauge_conn_var.set("串口: 未连接")
            self.gauge_err_var.set("已断开")
        except Exception:
            self.gauge_conn_var.set("串口: 未连接")

    def set_gauge_request_command(self, cmd: str) -> str:
        norm = str(cmd or 'M1,1').strip() or 'M1,1'
        try:
            gauge_worker = self.gauge_worker
            if gauge_worker is not None:
                gauge_worker.request_cmd = norm
        except Exception:
            pass
        return norm

    def _gauge_request_once(self):
        """发送一次测径仪请求命令（默认 M1,1\\r：包含鉴别结果）。
        - 需要先“连接”，否则会提示 not enabled。
        - 返回数据由后台线程解析后，自动更新 Gauge: OD。
        """
        try:
            # NOTE:
            # 请求指令在 UI 下拉中可随时更改（例如从 M1,1 -> M0,1），
            # 但 GaugeWorker.request_cmd 只会在 configure() 时更新。
            # 因此这里在“请求一次”前强制同步最新指令，避免出现：
            #   UI 显示 M0,1 但实际仍发送 M1,1，导致返回帧只有 OUT1。
            try:
                cmd = (self.req_cmd_var.get() if hasattr(self, "req_cmd_var") else "")
                cmd = (cmd or "M1,1").strip()
                self.set_gauge_request_command(cmd)
            except Exception:
                pass

            gauge_worker = self.gauge_worker
            if gauge_worker is None:
                self.gauge_err_var.set("Gauge ERROR: worker not available")
                return
            gauge_worker.send_request()
        except Exception as e:
            self.gauge_err_var.set(f"Gauge ERROR: {e}")
