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
    # OD / ID notebook
    # ------------------------------
    nb = ttk.Notebook(parent)
    nb.pack(fill=tk.X, pady=(4, 8))

    tab_od = ttk.Frame(nb)   # 测径仪实时 + 外径标定(B)
    tab_id = ttk.Frame(nb)   # 位移计实时(OUT) + 内径标定
    nb.add(tab_od, text="外径（实时+标定）")
    nb.add(tab_id, text="内径（实时+标定）")

    gbox = ttk.LabelFrame(tab_od, text="测径仪（外径 OD, 串口）")
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
    cbox = ttk.LabelFrame(tab_od, text="标定参数")
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
    abox = ttk.LabelFrame(tab_od, text="采集控制")
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


    ttk.Checkbutton(
        adv_frame,
        text="未学习模板时：动态屏蔽最深凹陷段",
        variable=app.odcal_defect_dyn_enable_var,
    ).grid(row=2, column=0, columnspan=6, padx=0, pady=(0, 4), sticky="w")

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

    # 凹陷表学习/清除（路线B）
    ttk.Button(abox, text="学习A", command=app._odcal_defect_learn_A).grid(row=1, column=7, padx=6, pady=6, sticky="w")
    ttk.Button(abox, text="学习B(生成表)", command=app._odcal_defect_learn_B).grid(row=1, column=8, padx=6, pady=6, sticky="w")
    ttk.Button(abox, text="清除凹陷表", command=app._odcal_defect_clear_template).grid(row=1, column=9, padx=6, pady=6, sticky="w")

    # default collapsed
    adv_frame.grid_remove()

    ttk.Label(abox, textvariable=app.odcal_state_var, width=10).grid(row=2, column=0, padx=10, pady=6, sticky="w")
    ttk.Label(abox, textvariable=app.odcal_msg_var).grid(row=2, column=1, columnspan=6, padx=6, pady=6, sticky="w")

    # C) Result & stats
    rbox = ttk.LabelFrame(tab_od, text="结果与质量")
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

    ttk.Label(rbox, text="凹陷屏蔽").grid(row=2, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(rbox, textvariable=app.odcal_defect_mode_var, width=10).grid(row=2, column=1, padx=6, pady=6, sticky="w")

    ttk.Label(rbox, text="shift").grid(row=2, column=2, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(rbox, textvariable=app.odcal_defect_shift_var, width=8).grid(row=2, column=3, padx=6, pady=6, sticky="w")

    ttk.Label(rbox, text="段").grid(row=2, column=4, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(rbox, textvariable=app.odcal_defects_var).grid(row=2, column=5, columnspan=4, padx=6, pady=6, sticky="w")


    # ------------------------------
    # ------------------------------
    # CL-3000 (Keyence) comm confirm via PLC mapped registers (OUT1..OUT5)
    # ------------------------------
    dbox = ttk.LabelFrame(tab_id, text="位移计实时（CL OUT1~OUT5）")
    dbox.pack(fill=tk.X, pady=(4, 8))
    # ------------------------------
    # ID Calibration (Chord OUT4 + m OUT5)
    # ------------------------------
    ibox = ttk.LabelFrame(tab_id, text="内径标定（OUT4弦长 + OUT5偏移m）")
    ibox.pack(fill=tk.X, pady=(0, 8))

    ttk.Label(ibox, text="ID_ref /mm").grid(row=0, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(ibox, width=10, textvariable=app.idcal_dref_var).grid(row=0, column=1, padx=6, pady=6, sticky="w")

    ttk.Label(ibox, text="模式").grid(row=0, column=2, padx=(10, 2), pady=6, sticky="e")
    ttk.Combobox(ibox, width=10, textvariable=app.idcal_mode_var, values=["one_rev", "timed"], state="readonly").grid(
        row=0, column=3, padx=6, pady=6, sticky="w"
    )

    ttk.Label(ibox, text="Hz").grid(row=0, column=4, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(ibox, width=6, textvariable=app.idcal_hz_var).grid(row=0, column=5, padx=6, pady=6, sticky="w")

    ttk.Label(ibox, text="T /s").grid(row=0, column=6, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(ibox, width=6, textvariable=app.idcal_duration_var).grid(row=0, column=7, padx=6, pady=6, sticky="w")

    ttk.Label(ibox, text="AX3 角速度 /deg/s").grid(row=0, column=8, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(ibox, width=8, textvariable=app.idcal_rot_degps_var).grid(row=0, column=9, padx=6, pady=6, sticky="w")

    ttk.Button(ibox, text="开始采集", command=app._idcal_start_capture).grid(row=1, column=0, padx=10, pady=6, sticky="w")
    ttk.Button(ibox, text="停止", command=app._idcal_stop_capture).grid(row=1, column=1, padx=6, pady=6, sticky="w")
    ttk.Button(ibox, text="清空", command=app._idcal_clear).grid(row=1, column=2, padx=6, pady=6, sticky="w")
    ttk.Button(ibox, text="计算", command=app._idcal_compute).grid(row=1, column=3, padx=6, pady=6, sticky="w")
    ttk.Button(ibox, text="应用", command=app._idcal_apply).grid(row=1, column=4, padx=6, pady=6, sticky="w")
    ttk.Button(ibox, text="导出raw", command=app._idcal_export_raw).grid(row=1, column=5, padx=6, pady=6, sticky="w")
    ttk.Button(ibox, text="复核", command=app._idcal_verify).grid(row=1, column=6, padx=6, pady=6, sticky="w")

    ttk.Label(ibox, textvariable=app.idcal_state_var, width=10).grid(row=1, column=7, padx=(16, 6), pady=6, sticky="w")
    ttk.Label(ibox, textvariable=app.idcal_msg_var).grid(row=1, column=8, columnspan=2, padx=6, pady=6, sticky="w")

    ttk.Label(ibox, text="δc_candidate /mm").grid(row=2, column=0, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=app.idcal_delta_candidate_var, width=12).grid(row=2, column=1, padx=6, pady=4, sticky="w")
    ttk.Label(ibox, text="δc_active /mm").grid(row=2, column=2, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=app.idcal_delta_active_var, width=12).grid(row=2, column=3, padx=6, pady=4, sticky="w")

    ttk.Label(ibox, text="c_max /mm").grid(row=2, column=4, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=app.idcal_cmax_var, width=12).grid(row=2, column=5, padx=6, pady=4, sticky="w")
    ttk.Label(ibox, text="m_mean /mm").grid(row=2, column=6, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=app.idcal_mmean_var, width=12).grid(row=2, column=7, padx=6, pady=4, sticky="w")
    ttk.Label(ibox, text="m_pp /mm").grid(row=2, column=8, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=app.idcal_mpp_var, width=12).grid(row=2, column=9, padx=6, pady=4, sticky="w")

    ttk.Label(ibox, text="拟合直径 2R /mm").grid(row=3, column=0, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=app.idcal_fit_diam_var, width=12).grid(row=3, column=1, padx=6, pady=4, sticky="w")
    ttk.Label(ibox, text="拟合 e /mm").grid(row=3, column=2, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=app.idcal_fit_e_var, width=12).grid(row=3, column=3, padx=6, pady=4, sticky="w")
    ttk.Label(ibox, text="拟合 y0 /mm").grid(row=3, column=4, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=app.idcal_fit_y0_var, width=12).grid(row=3, column=5, padx=6, pady=4, sticky="w")
    ttk.Label(ibox, text="rmse(R²) /mm²").grid(row=3, column=6, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=app.idcal_fit_rmse_var, width=12).grid(row=3, column=7, padx=6, pady=4, sticky="w")

    ttk.Label(ibox, text="复核ΔD /mm").grid(row=4, column=0, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=app.idcal_chk_err_var, width=12).grid(row=4, column=1, padx=6, pady=4, sticky="w")
    ttk.Label(ibox, text="覆盖率").grid(row=4, column=2, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=app.idcal_chk_cov_var, width=12).grid(row=4, column=3, padx=6, pady=4, sticky="w")
    ttk.Label(ibox, text="N").grid(row=4, column=4, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=app.idcal_chk_n_var, width=12).grid(row=4, column=5, padx=6, pady=4, sticky="w")
    ttk.Label(ibox, text="dθ_max").grid(row=4, column=6, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=app.idcal_chk_dtheta_var, width=12).grid(row=4, column=7, padx=6, pady=4, sticky="w")

    ttk.Label(
        ibox,
        text="说明：OUT4 为弦长。δc 是对 OUT4 的加法修正，使拟合直径≈ID_ref（或退化为 c_max 对齐）。",
    ).grid(row=5, column=0, columnspan=10, padx=10, pady=(0, 6), sticky="w")




    # ------------------------------
    # ID Single-Probe Calibration (OUT2/L2)
    # ------------------------------
    sbox = ttk.LabelFrame(tab_id, text="ID Single-Probe Calibration (OUT2/L2)")
    sbox.pack(fill=tk.X, pady=(0, 8))

    ttk.Label(sbox, text="ID_ref /mm").grid(row=0, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(sbox, width=10, textvariable=app.id_single_cal_dref_var).grid(row=0, column=1, padx=6, pady=6, sticky="w")

    ttk.Button(sbox, text="Capture 1 rev", command=app._id_single_cal_start_capture).grid(row=0, column=2, padx=(16, 6), pady=6, sticky="w")
    ttk.Button(sbox, text="Stop", command=lambda: app._id_single_cal_stop_capture("manual")).grid(row=0, column=3, padx=6, pady=6, sticky="w")
    ttk.Button(sbox, text="Compute & Write", command=app._id_single_cal_compute_apply).grid(row=0, column=4, padx=(16, 6), pady=6, sticky="w")

    ttk.Label(sbox, textvariable=app.id_single_cal_state_var, width=10).grid(row=1, column=0, padx=(10, 2), pady=4, sticky="w")
    ttk.Label(sbox, textvariable=app.id_single_cal_msg_var).grid(row=1, column=1, columnspan=3, padx=6, pady=4, sticky="w")
    ttk.Label(sbox, textvariable=app.id_single_cal_warn_var, foreground="red").grid(row=1, column=4, padx=6, pady=4, sticky="w")

    ttk.Label(sbox, text="mean(L2_decenter)").grid(row=2, column=0, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(sbox, textvariable=app.id_single_cal_mean_var, width=12).grid(row=2, column=1, padx=6, pady=4, sticky="w")
    ttk.Label(sbox, text="B").grid(row=2, column=2, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(sbox, textvariable=app.id_single_cal_B_var, width=12).grid(row=2, column=3, padx=6, pady=4, sticky="w")
    ttk.Label(sbox, text="cov").grid(row=2, column=4, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(sbox, textvariable=app.id_single_cal_cov_var, width=8).grid(row=2, column=5, padx=6, pady=4, sticky="w")

    ttk.Label(sbox, text="ecc_amp").grid(row=3, column=0, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(sbox, textvariable=app.id_single_cal_ecc_amp_var, width=12).grid(row=3, column=1, padx=6, pady=4, sticky="w")
    ttk.Label(sbox, text="ecc_ang(deg)").grid(row=3, column=2, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(sbox, textvariable=app.id_single_cal_ecc_ang_var, width=12).grid(row=3, column=3, padx=6, pady=4, sticky="w")
    # Row 0: OUT1 / OUT2
    ttk.Label(dbox, text="OUT1 x1(右) /mm").grid(row=0, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(dbox, textvariable=app.cl_out1_var, width=12).grid(row=0, column=1, padx=6, pady=6, sticky="w")
    ttk.Label(dbox, text="cnt").grid(row=0, column=2, padx=(6, 2), pady=6, sticky="e")
    ttk.Label(dbox, textvariable=app.cl_out1_cnt_var, width=10).grid(row=0, column=3, padx=6, pady=6, sticky="w")

    ttk.Label(dbox, text="OUT2 x2(左) /mm").grid(row=0, column=4, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(dbox, textvariable=app.cl_out2_var, width=12).grid(row=0, column=5, padx=6, pady=6, sticky="w")
    ttk.Label(dbox, text="cnt").grid(row=0, column=6, padx=(6, 2), pady=6, sticky="e")
    ttk.Label(dbox, textvariable=app.cl_out2_cnt_var, width=10).grid(row=0, column=7, padx=6, pady=6, sticky="w")

    # Row 1: OUT4(ID) / OUT5(m)
    ttk.Label(dbox, text="OUT4 内径ID /mm").grid(row=1, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(dbox, textvariable=app.cl_out4_var, width=12).grid(row=1, column=1, padx=6, pady=6, sticky="w")
    ttk.Label(dbox, text="cnt").grid(row=1, column=2, padx=(6, 2), pady=6, sticky="e")
    ttk.Label(dbox, textvariable=app.cl_out4_cnt_var, width=10).grid(row=1, column=3, padx=6, pady=6, sticky="w")

    ttk.Label(dbox, text="OUT5 m(投影) /mm").grid(row=1, column=4, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(dbox, textvariable=app.cl_out5_var, width=12).grid(row=1, column=5, padx=6, pady=6, sticky="w")
    ttk.Label(dbox, text="cnt").grid(row=1, column=6, padx=(6, 2), pady=6, sticky="e")
    ttk.Label(dbox, textvariable=app.cl_out5_cnt_var, width=10).grid(row=1, column=7, padx=6, pady=6, sticky="w")

    # Row 2: m-hat and diff
    ttk.Label(dbox, text="m̂=(x1-x2)/2").grid(row=2, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(dbox, textvariable=app.cl_m_calc_var, width=12).grid(row=2, column=1, padx=6, pady=6, sticky="w")

    ttk.Label(dbox, text="Δ=m̂-OUT5").grid(row=2, column=4, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(dbox, textvariable=app.cl_m_diff_var, width=12).grid(row=2, column=5, padx=6, pady=6, sticky="w")

    # Row 3: OUT3 reserved/thickness (optional)
    ttk.Label(dbox, text="OUT3(保留/厚度)").grid(row=3, column=0, padx=(10, 2), pady=(2, 6), sticky="e")
    ttk.Label(dbox, textvariable=app.cl_out3_var, width=12).grid(row=3, column=1, padx=6, pady=(2, 6), sticky="w")
    ttk.Label(dbox, text="cnt").grid(row=3, column=2, padx=(6, 2), pady=(2, 6), sticky="e")
    ttk.Label(dbox, textvariable=app.cl_out3_cnt_var, width=10).grid(row=3, column=3, padx=6, pady=(2, 6), sticky="w")

    ttk.Label(
        dbox,
        text="提示：主流程内径ID默认取 OUT4。m̂用于校验OUT5公式/符号是否一致。",
    ).grid(row=4, column=0, columnspan=8, padx=10, pady=(2, 6), sticky="w")