# ./ui/screens/main_screen.py
from __future__ import annotations

"""主操作页（UI 构建）。"""

from typing import TYPE_CHECKING
import tkinter as tk
from tkinter import ttk

if TYPE_CHECKING:  # pragma: no cover
    from app import App

def build_main_screen(app: "App", parent: ttk.Frame) -> None:
    """主操作界面：自动测量启动/停止 + 结果显示 + 直线度/覆盖率。"""
    top = ttk.Frame(parent)
    top.pack(fill=tk.X, pady=6)

    left = ttk.LabelFrame(top, text="测量状态")
    left.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

    ttk.Label(
        left, textvariable=app.auto_progress_var, font=("Segoe UI", 11, "bold")
    ).grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")
    ttk.Label(left, textvariable=app.auto_done_var).grid(
        row=1, column=0, padx=10, pady=2, sticky="w"
    )
    ttk.Label(left, text="外径标准值").grid(
        row=2, column=0, padx=10, pady=2, sticky="w"
    )
    app.lbl_od_std = ttk.Label(left, text="--")
    app.lbl_od_std.grid(row=2, column=1, padx=10, pady=2, sticky="w")
    ttk.Label(left, text="内径标准值").grid(
        row=3, column=0, padx=10, pady=2, sticky="w"
    )
    app.lbl_id_std = ttk.Label(left, text="--")
    app.lbl_id_std.grid(row=3, column=1, padx=10, pady=2, sticky="w")

    ttk.Label(left, text="自动状态").grid(
        row=4, column=0, padx=10, pady=(2, 10), sticky="w"
    )
    ttk.Label(left, textvariable=app.auto_state_var).grid(
        row=4, column=1, padx=10, pady=(2, 10), sticky="w"
    )
    ttk.Label(left, text="信息").grid(
        row=5, column=0, padx=10, pady=(2, 10), sticky="w"
    )
    ttk.Label(left, textvariable=app.auto_msg_var).grid(
        row=5, column=1, padx=10, pady=(2, 10), sticky="w"
    )

    ctrl = ttk.LabelFrame(top, text="控制")
    ctrl.pack(side=tk.LEFT, fill=tk.Y)

    ttk.Button(ctrl, text="开始测量", width=16, command=app._auto_start).pack(
        padx=10, pady=(10, 6)
    )
    ttk.Button(ctrl, text="停止", width=16, command=app._auto_stop).pack(
        padx=10, pady=6
    )
    ttk.Button(ctrl, text="清空结果", width=16, command=app._auto_clear_ui).pack(
        padx=10, pady=(6, 10)
    )

    # Straightness (overall)
    sline = ttk.Frame(parent)
    sline.pack(fill=tk.X, pady=(0, 4))
    ttk.Label(sline, textvariable=app.straight_var).pack(side=tk.LEFT, padx=10)
    ttk.Label(sline, textvariable=app.cov_var).pack(side=tk.LEFT, padx=10)

    # Results table
    mid = ttk.Frame(parent)
    mid.pack(fill=tk.BOTH, expand=True)

    cols = ("idx", "x_ui", "od_avg", "od_max", "od_min", "dev", "od_round", "ok")
    app.result_tree = ttk.Treeview(mid, columns=cols, show="headings")
    app.result_tree.heading("idx", text="截面")
    app.result_tree.heading("x_ui", text="位置(UI,mm)")
    app.result_tree.heading("od_avg", text="平均外径(mm)")
    app.result_tree.heading("od_max", text="最大外径(mm)")
    app.result_tree.heading("od_min", text="最小外径(mm)")
    app.result_tree.heading("dev", text="偏差(mm)")
    app.result_tree.heading("od_round", text="真圆度(mm)")
    app.result_tree.heading("ok", text="判定")
    app.result_tree.column("idx", width=60, anchor="center")
    app.result_tree.column("x_ui", width=120, anchor="e")
    app.result_tree.column("od_avg", width=110, anchor="e")
    app.result_tree.column("od_max", width=110, anchor="e")
    app.result_tree.column("od_min", width=110, anchor="e")
    app.result_tree.column("dev", width=90, anchor="e")
    app.result_tree.column("od_round", width=90, anchor="e")
    app.result_tree.column("ok", width=80, anchor="center")

    app.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    ysb = ttk.Scrollbar(mid, orient="vertical", command=app.result_tree.yview)
    app.result_tree.configure(yscroll=ysb.set)
    ysb.pack(side=tk.RIGHT, fill=tk.Y)

    app._refresh_auto_std_panel()

