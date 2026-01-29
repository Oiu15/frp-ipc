# ./ui/screens/main_screen.py
from __future__ import annotations

"""主操作页（UI 构建）。"""

from typing import TYPE_CHECKING
import tkinter as tk
from tkinter import ttk

if TYPE_CHECKING:  # pragma: no cover
    from app import App

def build_main_screen(app: "App", parent: ttk.Frame) -> None:
    """主操作界面：自动测量启动/停止 + 状态/汇总结果显示 + 表格。"""

    top = ttk.Frame(parent)
    top.pack(fill=tk.X, pady=6)
    # Use grid so "测量状态" and "测量结果" can be truly equal-width.
    top.grid_columnconfigure(0, weight=1, uniform="top_panels")
    top.grid_columnconfigure(1, weight=1, uniform="top_panels")
    top.grid_columnconfigure(2, weight=0)
    top.grid_rowconfigure(0, weight=1)

    # ------------------------------
    # Status panel (runtime)
    # ------------------------------
    st = ttk.LabelFrame(top, text="测量状态")
    st.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
    st.columnconfigure(1, weight=1)

    ttk.Label(st, text="流水号").grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")
    ttk.Label(st, textvariable=app.pipe_sn_var, font=("Segoe UI", 10, "bold")).grid(
        row=0, column=1, padx=10, pady=(10, 2), sticky="w"
    )

    ttk.Label(st, text="测量计数").grid(row=1, column=0, padx=10, pady=2, sticky="w")
    ttk.Label(st, textvariable=app.meas_seq_var).grid(
        row=1, column=1, padx=10, pady=2, sticky="w"
    )

    ttk.Label(st, text="开始时间").grid(row=2, column=0, padx=10, pady=2, sticky="w")
    ttk.Label(st, textvariable=app.meas_start_var).grid(
        row=2, column=1, padx=10, pady=2, sticky="w"
    )

    ttk.Label(st, text="耗时").grid(row=3, column=0, padx=10, pady=2, sticky="w")
    ttk.Label(st, textvariable=app.meas_elapsed_var).grid(
        row=3, column=1, padx=10, pady=2, sticky="w"
    )

    ttk.Label(st, textvariable=app.auto_progress_var, font=("Segoe UI", 11, "bold")).grid(
        row=4, column=0, columnspan=2, padx=10, pady=(6, 2), sticky="w"
    )
    ttk.Label(st, textvariable=app.auto_done_var).grid(
        row=5, column=0, columnspan=2, padx=10, pady=2, sticky="w"
    )

    ttk.Label(st, text="自动状态").grid(row=6, column=0, padx=10, pady=(2, 2), sticky="w")
    ttk.Label(st, textvariable=app.auto_state_var).grid(
        row=6, column=1, padx=10, pady=(2, 2), sticky="w"
    )

    ttk.Label(st, text="信息").grid(row=7, column=0, padx=10, pady=(2, 10), sticky="w")
    ttk.Label(st, textvariable=app.auto_msg_var, wraplength=260, justify="left").grid(
        row=7, column=1, padx=10, pady=(2, 10), sticky="w"
    )

    # ------------------------------
    # Summary panel (results)
    # ------------------------------
    res = ttk.LabelFrame(top, text="测量结果")
    res.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
    res.columnconfigure(1, weight=1)

    ttk.Label(res, text="外径标准值").grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")
    app.lbl_od_std = ttk.Label(res, text="--")
    app.lbl_od_std.grid(row=0, column=1, padx=10, pady=(10, 2), sticky="w")

    ttk.Label(res, text="内径标准值").grid(row=1, column=0, padx=10, pady=2, sticky="w")
    app.lbl_id_std = ttk.Label(res, text="--")
    app.lbl_id_std.grid(row=1, column=1, padx=10, pady=2, sticky="w")

    ttk.Label(res, textvariable=app.straight_var, wraplength=520, justify="left").grid(
        row=2, column=0, columnspan=2, padx=10, pady=(6, 2), sticky="w"
    )
    ttk.Label(res, textvariable=app.conc_var, wraplength=520, justify="left").grid(
        row=3, column=0, columnspan=2, padx=10, pady=(0, 4), sticky="w"
    )

    # Optional length measurement result (shown when enabled)
    ttk.Label(res, text="长度测量值").grid(row=4, column=0, padx=10, pady=2, sticky="w")
    ttk.Label(res, textvariable=getattr(app, "len_meas_var", tk.StringVar(value="--")), wraplength=520, justify="left").grid(
        row=4, column=1, padx=10, pady=2, sticky="w"
    )

    ttk.Label(res, text="最大外径偏差").grid(row=5, column=0, padx=10, pady=2, sticky="w")
    ttk.Label(res, textvariable=app.max_od_dev_var).grid(row=5, column=1, padx=10, pady=2, sticky="w")

    ttk.Label(res, text="最大内径偏差").grid(row=6, column=0, padx=10, pady=2, sticky="w")
    ttk.Label(res, textvariable=app.max_id_dev_var).grid(row=6, column=1, padx=10, pady=2, sticky="w")

    ttk.Label(res, text="最大外圆真圆度").grid(row=7, column=0, padx=10, pady=2, sticky="w")
    ttk.Label(res, textvariable=app.max_od_round_var).grid(row=7, column=1, padx=10, pady=2, sticky="w")

    ttk.Label(res, text="最大内圆真圆度").grid(row=8, column=0, padx=10, pady=(2, 10), sticky="w")
    ttk.Label(res, textvariable=app.max_id_round_var).grid(row=8, column=1, padx=10, pady=(2, 10), sticky="w")

    # ------------------------------
    # Controls
    # ------------------------------
    ctrl = ttk.LabelFrame(top, text="控制")
    ctrl.grid(row=0, column=2, sticky="ns")

    ttk.Button(ctrl, text="开始测量", width=16, command=app._auto_start).pack(
        padx=10, pady=(10, 6)
    )
    ttk.Button(ctrl, text="停止", width=16, command=app._auto_stop).pack(
        padx=10, pady=6
    )
    ttk.Button(ctrl, text="清空结果", width=16, command=app._auto_clear_ui).pack(
        padx=10, pady=(6, 10)
    )

    # ------------------------------
    # Coverage line
    # ------------------------------
    info_line = ttk.Frame(parent)
    info_line.pack(fill=tk.X, pady=(0, 4))
    info_line.columnconfigure(0, weight=1)

    ttk.Label(
        info_line,
        textvariable=app.cov_var,
        anchor="w",
        justify="left",
        wraplength=900,
    ).grid(row=0, column=0, sticky="we", padx=(10, 10))

    # ------------------------------
    # Results table
    # ------------------------------
    mid = ttk.Frame(parent)
    mid.pack(fill=tk.BOTH, expand=True)

    # Keep full column set for internal storage/export.
    # UI can selectively show a subset via displaycolumns.
    cols = (
        "idx",
        "x_ui",
        "od_dev",
        "od_runout",
        "od_round",
        "id_dev",
        "id_runout",
        "id_round",
        "concentricity",
        "od_ecc",
        "id_ecc",
        # sampling stats (cached per section)
        "cov_pct",
        "miss_bin",
        "max_gap_deg",
        "revs",
        "cov_elapsed_s",
        "cov_reason",
    )

    # Hide sampling-stat columns in UI, but keep them in the underlying row values
    # so exports (section_results.csv) can still include them.
    visible_cols = (
        "idx",
        "x_ui",
        "od_dev",
        "od_runout",
        "od_round",
        "id_dev",
        "id_runout",
        "id_round",
        "concentricity",
        "od_ecc",
        "id_ecc",
    )

    tree_wrap = ttk.Frame(mid)
    tree_wrap.pack(fill=tk.BOTH, expand=True)

    app.result_tree = ttk.Treeview(
        tree_wrap,
        columns=cols,
        displaycolumns=visible_cols,
        show="headings",
    )
    # Clicking a row should refresh per-section coverage/info (cached during sampling)
    app.result_tree.bind("<<TreeviewSelect>>", app._on_result_select)
    app.result_tree.heading("idx", text="截面")
    app.result_tree.heading("x_ui", text="OD位置(Z,mm)")
    app.result_tree.heading("od_dev", text="外径偏差(mm)")
    app.result_tree.heading("od_runout", text="外径径向跳动(mm)")
    app.result_tree.heading("od_round", text="外径真圆度(mm)")
    app.result_tree.heading("id_dev", text="内径偏差(mm)")
    app.result_tree.heading("id_runout", text="内径径向跳动(mm)")
    app.result_tree.heading("id_round", text="内径真圆度(mm)")
    app.result_tree.heading("concentricity", text="同心度(mm)")
    app.result_tree.heading("od_ecc", text="外圆偏心度(mm)")
    app.result_tree.heading("id_ecc", text="内圆偏心度(mm)")
    app.result_tree.heading("cov_pct", text="覆盖率(%)")
    app.result_tree.heading("miss_bin", text="缺失bin")
    app.result_tree.heading("max_gap_deg", text="最大空窗角(°)")
    app.result_tree.heading("revs", text="圈数")
    app.result_tree.heading("cov_elapsed_s", text="采样用时(s)")
    app.result_tree.heading("cov_reason", text="覆盖判据")
    app.result_tree.column("idx", width=60, anchor="center")
    app.result_tree.column("x_ui", width=110, anchor="e")
    app.result_tree.column("od_dev", width=110, anchor="e")
    app.result_tree.column("od_runout", width=125, anchor="e")
    app.result_tree.column("od_round", width=115, anchor="e")
    app.result_tree.column("id_dev", width=110, anchor="e")
    app.result_tree.column("id_runout", width=125, anchor="e")
    app.result_tree.column("id_round", width=115, anchor="e")
    app.result_tree.column("concentricity", width=95, anchor="e")
    app.result_tree.column("od_ecc", width=105, anchor="e")
    app.result_tree.column("id_ecc", width=105, anchor="e")

    app.result_tree.column("cov_pct", width=90, anchor="e")
    app.result_tree.column("miss_bin", width=80, anchor="e")
    app.result_tree.column("max_gap_deg", width=110, anchor="e")
    app.result_tree.column("revs", width=70, anchor="e")
    app.result_tree.column("cov_elapsed_s", width=95, anchor="e")
    app.result_tree.column("cov_reason", width=110, anchor="w")

    app.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    ysb = ttk.Scrollbar(tree_wrap, orient="vertical", command=app.result_tree.yview)
    app.result_tree.configure(yscroll=ysb.set)
    ysb.pack(side=tk.RIGHT, fill=tk.Y)

    xsb = ttk.Scrollbar(mid, orient="horizontal", command=app.result_tree.xview)
    app.result_tree.configure(xscroll=xsb.set)
    xsb.pack(side=tk.BOTTOM, fill=tk.X)

    app._refresh_auto_std_panel()
