# ./ui/screens/gauge_screen.py
from __future__ import annotations

"""测径仪通信页（UI 构建）。"""

from typing import TYPE_CHECKING
import tkinter as tk
from tkinter import ttk

from config.addresses import DEFAULT_GAUGE_PORT

if TYPE_CHECKING:  # pragma: no cover
    from app import App

def build_gauge_screen(app: "App", parent: ttk.Frame) -> None:
    """测径仪通信（外径 OD）页面。"""
    gbox = ttk.LabelFrame(parent, text="测径仪（外径）")
    gbox.pack(fill=tk.X, pady=(4, 8))

    app.sim_gauge_var = tk.IntVar(value=0)
    ttk.Checkbutton(
        gbox,
        text="模拟测径仪",
        variable=app.sim_gauge_var,
        command=app._on_sim_gauge_toggle,
    ).grid(row=0, column=0, padx=10, pady=6, sticky="w")

    ttk.Label(gbox, text="串口").grid(
        row=0, column=1, padx=(10, 2), pady=6, sticky="e"
    )
    app.port_combo = ttk.Combobox(
        gbox, width=12, state="readonly", values=app._list_serial_ports()
    )
    app.port_combo.grid(row=0, column=2, padx=6, pady=6, sticky="w")
    app.port_combo.set(DEFAULT_GAUGE_PORT)
    try:
        ports = list(app.port_combo.cget("values"))
        if DEFAULT_GAUGE_PORT in ports:
            app.port_combo.set(DEFAULT_GAUGE_PORT)
        else:
            app.port_combo.set(DEFAULT_GAUGE_PORT)
    except Exception:
        pass
    ttk.Button(gbox, text="刷新", command=app._refresh_ports).grid(
        row=0, column=3, padx=6, pady=6
    )

    ttk.Label(gbox, text="波特率").grid(
        row=0, column=4, padx=(10, 2), pady=6, sticky="e"
    )
    app.baud_var = tk.StringVar(value="9600")
    ttk.Entry(gbox, width=8, textvariable=app.baud_var).grid(
        row=0, column=5, padx=6, pady=6, sticky="w"
    )

    ttk.Label(gbox, text="请求指令(可空)").grid(
        row=0, column=6, padx=(10, 2), pady=6, sticky="e"
    )
    app.req_cmd_var = tk.StringVar(value="M1,0")
    ttk.Entry(gbox, width=18, textvariable=app.req_cmd_var).grid(
        row=0, column=7, padx=6, pady=6, sticky="w"
    )

    ttk.Button(gbox, text="连接", command=app._gauge_connect).grid(
        row=0, column=8, padx=6, pady=6
    )
    ttk.Button(gbox, text="断开", command=app._gauge_disconnect).grid(
        row=0, column=9, padx=6, pady=6
    )
    ttk.Button(gbox, text="请求一次", command=app._gauge_request_once).grid(
        row=0, column=10, padx=6, pady=6
    )

    ttk.Label(gbox, textvariable=app.gauge_conn_var).grid(
        row=1, column=8, columnspan=3, padx=6, pady=(2, 6), sticky="e"
    )

    ttk.Label(gbox, textvariable=app.gauge_last_var).grid(
        row=1, column=0, columnspan=8, padx=10, pady=(2, 6), sticky="w"
    )
    ttk.Label(gbox, textvariable=app.gauge_err_var, foreground="red").grid(
        row=2, column=0, columnspan=8, padx=10, pady=(0, 6), sticky="w"
    )
