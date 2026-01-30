# ./ui/screens/gauge_screen.py
from __future__ import annotations

"""外设通信页（UI 构建）。"""

from typing import TYPE_CHECKING
import tkinter as tk
from tkinter import ttk

from config.addresses import DEFAULT_GAUGE_PORT

if TYPE_CHECKING:  # pragma: no cover
    from app import App

def build_gauge_screen(app: "App", parent: ttk.Frame) -> None:
    """外设通信：PLC(Modbus) + 测径仪(外径) + CL(内径 OUT3)。"""
    # ------------------------------
    # PLC connection (Modbus TCP)
    # ------------------------------
    pbox = ttk.LabelFrame(parent, text="PLC 通信（Modbus TCP）")
    pbox.pack(fill=tk.X, pady=(4, 8))

    ttk.Label(pbox, text="IP").grid(row=0, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(pbox, width=14, textvariable=app.ip_var).grid(row=0, column=1, padx=6, pady=6, sticky="w")

    ttk.Label(pbox, text="Port").grid(row=0, column=2, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(pbox, width=6, textvariable=app.port_var).grid(row=0, column=3, padx=6, pady=6, sticky="w")

    ttk.Button(pbox, text="连接/重连", command=app._apply_conn).grid(row=0, column=4, padx=8, pady=6)

    ttk.Label(pbox, textvariable=app.plc_status_var).grid(row=0, column=5, padx=10, pady=6, sticky="w")

    # ------------------------------
    # Gauge + OD Calibration notebook
    # ------------------------------
    nb = ttk.Notebook(parent)
    nb.pack(fill=tk.X, pady=(4, 8))

    tab_gauge = ttk.Frame(nb)
    tab_calib = ttk.Frame(nb)
    nb.add(tab_gauge, text="测径仪实时")
    nb.add(tab_calib, text="外径标定(B)")

    gbox = ttk.LabelFrame(tab_gauge, text="测径仪（外径 OD, 串口）")
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

    ttk.Label(gbox, text="请求指令").grid(
        row=0, column=6, padx=(10, 2), pady=6, sticky="e"
    )
    # 下拉预设：便于不同算法选择不同读数指令。
    # 说明：
    # - M1,* : 仅 OUT1
    # - M0,* : OUT1+OUT2
    # - r=1  : 带比较器/鉴别结果(HI/GO/LO...)，用于边缘/有效性判断
    cmd_presets = [
        "M0,1",
        "M1,1",
        "M0,0",
        "M1,0",
        "M2,1",
        "M2,0",
    ]
    app.req_cmd_var = tk.StringVar(value="M1,1")
    app.req_cmd_combo = ttk.Combobox(
        gbox,
        width=10,
        textvariable=app.req_cmd_var,
        values=cmd_presets,
        state="normal",  # 允许现场临时输入自定义命令，同时保留下拉选择
    )
    app.req_cmd_combo.grid(row=0, column=7, padx=6, pady=6, sticky="w")

    # UI 下拉修改后立即同步到 worker，避免“UI 显示已选 M0,1 但 worker 仍用旧命令”。
    def _on_req_cmd_changed(*_args):  # pragma: no cover (UI callback)
        try:
            cmd = (app.req_cmd_var.get() or "M1,1").strip()
            if getattr(app, "gauge_worker", None) is not None:
                app.gauge_worker.request_cmd = cmd
        except Exception:
            pass

    try:
        app.req_cmd_var.trace_add("write", _on_req_cmd_changed)
    except Exception:
        # Tk older versions fallback
        try:
            app.req_cmd_var.trace("w", _on_req_cmd_changed)  # type: ignore
        except Exception:
            pass

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

    # ------------------------------
    # OD Calibration (B) tab
    # ------------------------------
    # A) Calibration parameters
    cbox = ttk.LabelFrame(tab_calib, text="标定参数")
    cbox.pack(fill=tk.X, pady=(4, 8))

    ttk.Label(cbox, text="请求指令").grid(row=0, column=0, padx=(10, 2), pady=6, sticky="e")
    calib_cmd_presets = ["M0,1", "M0,0", "M1,1", "M1,0", "M2,1", "M2,0"]
    app.odcal_cmd_combo = ttk.Combobox(
        cbox,
        width=10,
        textvariable=app.odcal_cmd_var,
        values=calib_cmd_presets,
        state="normal",
    )
    app.odcal_cmd_combo.grid(row=0, column=1, padx=6, pady=6, sticky="w")

    ttk.Label(cbox, text="环规直径 D_ref(mm)").grid(row=0, column=2, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(cbox, width=10, textvariable=app.odcal_dref_var).grid(row=0, column=3, padx=6, pady=6, sticky="w")

    ttk.Label(cbox, text="通道映射(OUT1→)").grid(row=0, column=4, padx=(10, 2), pady=6, sticky="e")
    app.odcal_out1_map_combo = ttk.Combobox(
        cbox,
        width=4,
        textvariable=app.odcal_map_out1_var,
        values=["L", "R"],
        state="readonly",
    )
    app.odcal_out1_map_combo.grid(row=0, column=5, padx=6, pady=6, sticky="w")

    out2_hint_var = tk.StringVar(value="OUT2→R")
    ttk.Label(cbox, textvariable=out2_hint_var).grid(row=0, column=6, padx=6, pady=6, sticky="w")

    def _refresh_out2_hint(*_args):  # pragma: no cover
        try:
            out1 = (app.odcal_map_out1_var.get() or "L").strip().upper()
            out2 = "R" if out1 == "L" else "L"
            out2_hint_var.set(f"OUT2→{out2}")
        except Exception:
            pass

    try:
        app.odcal_map_out1_var.trace_add("write", _refresh_out2_hint)
    except Exception:
        try:
            app.odcal_map_out1_var.trace("w", _refresh_out2_hint)  # type: ignore
        except Exception:
            pass
    _refresh_out2_hint()

    ttk.Label(
        cbox,
        text="建议：B 标定使用 M0,*（OUT1+OUT2）。一圈采样会自动控制 AX3 旋转（按 deg/s）。",
    ).grid(row=1, column=0, columnspan=8, padx=10, pady=(2, 6), sticky="w")

    # B) Capture controls
    abox = ttk.LabelFrame(tab_calib, text="采集控制")
    abox.pack(fill=tk.X, pady=(4, 8))

    ttk.Label(abox, text="采集模式").grid(row=0, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Radiobutton(abox, text="定时采样", variable=app.odcal_mode_var, value="timed").grid(
        row=0, column=1, padx=6, pady=6, sticky="w"
    )
    ttk.Radiobutton(abox, text="一圈采样", variable=app.odcal_mode_var, value="one_rev").grid(
        row=0, column=2, padx=6, pady=6, sticky="w"
    )

    ttk.Label(abox, text="采样频率(Hz)").grid(row=0, column=3, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(abox, width=8, textvariable=app.odcal_hz_var).grid(row=0, column=4, padx=6, pady=6, sticky="w")

    # duration / timeout label (switch with mode)
    dur_label_var = tk.StringVar(value="时长(s)")
    ttk.Label(abox, textvariable=dur_label_var).grid(row=0, column=5, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(abox, width=8, textvariable=app.odcal_duration_var).grid(row=0, column=6, padx=6, pady=6, sticky="w")

    ttk.Label(abox, text="旋转速度(deg/s)").grid(row=0, column=7, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(abox, width=8, textvariable=app.odcal_rot_degps_var).grid(row=0, column=8, padx=6, pady=6, sticky="w")

    def _refresh_dur_label(*_args):  # pragma: no cover
        try:
            mode = (app.odcal_mode_var.get() or "timed").strip()
            dur_label_var.set("超时(s)" if mode == "one_rev" else "时长(s)")
        except Exception:
            pass
    try:
        app.odcal_mode_var.trace_add("write", _refresh_dur_label)
    except Exception:
        try:
            app.odcal_mode_var.trace("w", _refresh_dur_label)  # type: ignore
        except Exception:
            pass
    _refresh_dur_label()

    # 注意：按钮不要使用相同的 grid 坐标，否则会互相覆盖导致“开始/停止按钮消失”。
    ttk.Button(abox, text="开始采集", command=app._odcal_start_capture).grid(row=1, column=0, padx=10, pady=6, sticky="w")
    ttk.Button(abox, text="停止", command=lambda: app._odcal_stop_capture("manual")).grid(
        row=1, column=1, padx=6, pady=6, sticky="w"
    )

    ttk.Button(abox, text="计算 B", command=app._odcal_compute).grid(row=1, column=2, padx=(16, 6), pady=6, sticky="w")
    ttk.Button(abox, text="应用", command=app._odcal_apply).grid(row=1, column=3, padx=6, pady=6, sticky="w")

    ttk.Button(abox, text="导出RAW", command=app._odcal_export_raw).grid(row=1, column=4, padx=(16, 6), pady=6, sticky="w")
    ttk.Button(abox, text="清空", command=app._odcal_clear).grid(row=1, column=5, padx=6, pady=6, sticky="w")


    # Advanced sampling params (collapsible)
    adv_open_var = tk.BooleanVar(value=False)

    adv_frame = ttk.Frame(abox)
    adv_frame.grid(row=3, column=0, columnspan=12, padx=10, pady=(0, 6), sticky="ew")

    ttk.Label(adv_frame, text="角度来源").grid(row=0, column=0, padx=(0, 2), pady=4, sticky="e")
    app.odcal_angle_src_combo = ttk.Combobox(
        adv_frame,
        width=10,
        textvariable=app.odcal_angle_src_var,
        values=["AX3", "无角度"],
        state="readonly",
    )
    app.odcal_angle_src_combo.grid(row=0, column=1, padx=6, pady=4, sticky="w")

    ttk.Label(adv_frame, text="去抖/滤波").grid(row=0, column=2, padx=(10, 2), pady=4, sticky="e")
    app.odcal_filter_combo = ttk.Combobox(
        adv_frame,
        width=10,
        textvariable=app.odcal_filter_var,
        values=["无", "中值(3)", "中值(5)"],
        state="readonly",
    )
    app.odcal_filter_combo.grid(row=0, column=3, padx=6, pady=4, sticky="w")

    ttk.Label(adv_frame, text="异常剔除阈值(σ)").grid(row=0, column=4, padx=(10, 2), pady=4, sticky="e")
    ttk.Entry(adv_frame, width=8, textvariable=app.odcal_outlier_sigma_var).grid(
        row=0, column=5, padx=6, pady=4, sticky="w"
    )

    ttk.Label(
        adv_frame,
        text="说明：角度=无角度 时，一圈采样会自动切到定时采样；σ=0 表示不做离群剔除。",
    ).grid(row=1, column=0, columnspan=8, padx=0, pady=(0, 4), sticky="w")

    def _on_angle_src_change(*_args):  # pragma: no cover
        try:
            ang = str(app.odcal_angle_src_var.get() or "AX3")
            if ("无" in ang) and (str(app.odcal_mode_var.get() or "timed") == "one_rev"):
                app.odcal_mode_var.set("timed")
        except Exception:
            pass

    try:
        app.odcal_angle_src_var.trace_add("write", _on_angle_src_change)
    except Exception:
        try:
            app.odcal_angle_src_var.trace("w", _on_angle_src_change)  # type: ignore
        except Exception:
            pass

    def _toggle_adv():  # pragma: no cover
        is_open = bool(adv_open_var.get())
        adv_open_var.set(not is_open)
        if adv_open_var.get():
            adv_btn.configure(text="高级参数 ▼")
            adv_frame.grid()
        else:
            adv_btn.configure(text="高级参数 ▸")
            adv_frame.grid_remove()

    adv_btn = ttk.Button(abox, text="高级参数 ▸", command=_toggle_adv)
    adv_btn.grid(row=1, column=6, padx=(16, 6), pady=6, sticky="w")

    # default collapsed
    adv_frame.grid_remove()

    ttk.Label(abox, textvariable=app.odcal_state_var, width=10).grid(row=2, column=0, padx=10, pady=6, sticky="w")
    ttk.Label(abox, textvariable=app.odcal_msg_var).grid(row=2, column=1, columnspan=6, padx=6, pady=6, sticky="w")

    # C) Result & stats
    rbox = ttk.LabelFrame(tab_calib, text="结果与质量")
    rbox.pack(fill=tk.X, pady=(4, 8))

    ttk.Label(rbox, text="B_candidate").grid(row=0, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(rbox, textvariable=app.odcal_B_candidate_var, width=12).grid(row=0, column=1, padx=6, pady=6, sticky="w")

    ttk.Label(rbox, text="B_active").grid(row=0, column=2, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(rbox, textvariable=app.odcal_B_active_var, width=12).grid(row=0, column=3, padx=6, pady=6, sticky="w")

    ttk.Label(rbox, text="样本数 N").grid(row=0, column=4, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(rbox, textvariable=app.odcal_n_var, width=8).grid(row=0, column=5, padx=6, pady=6, sticky="w")

    ttk.Label(rbox, text="已用时").grid(row=0, column=6, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(rbox, textvariable=app.odcal_elapsed_var, width=8).grid(row=0, column=7, padx=6, pady=6, sticky="w")

    ttk.Label(rbox, text="mean(lL+lR)").grid(row=1, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(rbox, textvariable=app.odcal_sum_mean_var, width=12).grid(row=1, column=1, padx=6, pady=6, sticky="w")

    ttk.Label(rbox, text="std(lL+lR)").grid(row=1, column=2, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(rbox, textvariable=app.odcal_sum_std_var, width=12).grid(row=1, column=3, padx=6, pady=6, sticky="w")

    ttk.Label(rbox, text="min/max").grid(row=1, column=4, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(rbox, textvariable=app.odcal_sum_min_var, width=12).grid(row=1, column=5, padx=6, pady=6, sticky="w")
    ttk.Label(rbox, textvariable=app.odcal_sum_max_var, width=12).grid(row=1, column=6, padx=6, pady=6, sticky="w")

    ttk.Label(rbox, text="drop(非GO)").grid(row=1, column=7, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(rbox, textvariable=app.odcal_drop_rate_var, width=8).grid(row=1, column=8, padx=6, pady=6, sticky="w")

    # ------------------------------
    # CL-3000 (ID) via PLC mapped registers (OUT3)
    # ------------------------------
    dbox = ttk.LabelFrame(parent, text="CL（内径 ID, OUT3）")
    dbox.pack(fill=tk.X, pady=(4, 8))

    ttk.Label(dbox, text="实时内径(ID)").grid(row=0, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(dbox, textvariable=app.cl_id_var, width=16).grid(row=0, column=1, padx=6, pady=6, sticky="w")

    ttk.Label(dbox, text="更新计数").grid(row=0, column=2, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(dbox, textvariable=app.cl_cnt_var, width=12).grid(row=0, column=3, padx=6, pady=6, sticky="w")

    ttk.Label(dbox, text="内径测量结果按截面显示在“主测量”结果表格中。").grid(
        row=1, column=0, columnspan=6, padx=10, pady=(2, 6), sticky="w"
    )

