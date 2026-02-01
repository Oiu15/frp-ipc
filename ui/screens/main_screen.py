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
    # Use grid so top panels can have weighted widths (status narrower, results wider).
    top.grid_columnconfigure(0, weight=1)
    top.grid_columnconfigure(1, weight=3)
    top.grid_columnconfigure(2, weight=0)
    top.grid_rowconfigure(0, weight=1)

    # ------------------------------
    # Status panel (runtime)
    # ------------------------------
    st = ttk.LabelFrame(top, text="测量状态")
    st.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
    st.columnconfigure(1, weight=1)
    # Let the "信息" row take remaining height so multi-line text can be fully visible.
    st.rowconfigure(7, weight=1)

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
    # Use an auto-wrapping label (wraplength tracks widget width) so long messages
    # are shown in multiple lines without truncation.
    app.lbl_auto_msg = ttk.Label(st, textvariable=app.auto_msg_var, justify="left")
    app.lbl_auto_msg.grid(row=7, column=1, padx=10, pady=(2, 10), sticky="we")

    def _sync_msg_wrap(_e=None) -> None:
        try:
            w = int(app.lbl_auto_msg.winfo_width() or 0)
            if w > 20:
                app.lbl_auto_msg.configure(wraplength=w)
        except Exception:
            pass

    try:
        app.lbl_auto_msg.bind("<Configure>", _sync_msg_wrap)
    except Exception:
        pass
    # ------------------------------
    # Summary panel (results)
    # ------------------------------
    res = ttk.LabelFrame(top, text="测量结果")
    res.grid(row=0, column=1, sticky="nsew", padx=(0, 10))

    # Three-column summary: Overall / OD / ID
    res.grid_columnconfigure(0, weight=1, uniform="sum_cols")
    res.grid_columnconfigure(1, weight=1, uniform="sum_cols")
    res.grid_columnconfigure(2, weight=1, uniform="sum_cols")

    all_box = ttk.LabelFrame(res, text="总体")
    all_box.grid(row=0, column=0, sticky="nsew", padx=(10, 6), pady=(10, 10))
    all_box.columnconfigure(1, weight=1)

    od_box = ttk.LabelFrame(res, text="外圆")
    od_box.grid(row=0, column=1, sticky="nsew", padx=(6, 6), pady=(10, 10))
    od_box.columnconfigure(1, weight=1)

    id_box = ttk.LabelFrame(res, text="内圆")
    id_box.grid(row=0, column=2, sticky="nsew", padx=(6, 10), pady=(10, 10))
    id_box.columnconfigure(1, weight=1)

    def _kv(box: ttk.LabelFrame, r: int, key: str, value_widget: tk.Widget, *, pady=(2, 2)) -> None:
        ttk.Label(box, text=key).grid(row=r, column=0, padx=10, pady=pady, sticky="w")
        value_widget.grid(row=r, column=1, padx=10, pady=pady, sticky="w")

    # ---- OD (外圆) ----
    app.lbl_od_std = ttk.Label(od_box, text="--")
    _kv(od_box, 0, "外径标准值", app.lbl_od_std, pady=(8, 2))

    # Re-ordered + simplified labels
    _kv(od_box, 1, "平均外径", ttk.Label(od_box, textvariable=getattr(app, "od_mean_var", tk.StringVar(value="--"))))
    _kv(od_box, 2, "外径极差", ttk.Label(od_box, textvariable=getattr(app, "od_range_var", tk.StringVar(value="--"))))
    _kv(od_box, 3, "轴线偏差峰峰值", ttk.Label(od_box, textvariable=getattr(app, "straight_od_var", tk.StringVar(value="--"))))
    _kv(od_box, 4, "外圆轴线倾斜", ttk.Label(od_box, textvariable=getattr(app, "od_tilt_var", tk.StringVar(value="--"))))
    _kv(od_box, 5, "外圆轴线斜率", ttk.Label(od_box, textvariable=getattr(app, "od_slope_var", tk.StringVar(value="--"))))
    _kv(od_box, 6, "端点偏移(代直线度)", ttk.Label(od_box, textvariable=getattr(app, "od_endoff_var", tk.StringVar(value="--"))))
    _kv(od_box, 7, "最大外圆真圆度", ttk.Label(od_box, textvariable=app.max_od_round_var), pady=(2, 8))

    # ---- ID (内圆) ----
    app.lbl_id_std = ttk.Label(id_box, text="--")
    _kv(id_box, 0, "内径标准值", app.lbl_id_std, pady=(8, 2))

    _kv(id_box, 1, "平均内径", ttk.Label(id_box, textvariable=getattr(app, "id_mean_var", tk.StringVar(value="--"))))
    _kv(id_box, 2, "内径极差", ttk.Label(id_box, textvariable=getattr(app, "id_range_var", tk.StringVar(value="--"))))
    _kv(id_box, 3, "轴线偏差峰峰值", ttk.Label(id_box, textvariable=getattr(app, "straight_id_var", tk.StringVar(value="--"))))
    _kv(id_box, 4, "内圆轴线倾斜", ttk.Label(id_box, textvariable=getattr(app, "id_tilt_var", tk.StringVar(value="--"))))
    _kv(id_box, 5, "内圆轴线斜率", ttk.Label(id_box, textvariable=getattr(app, "id_slope_var", tk.StringVar(value="--"))))
    _kv(id_box, 6, "端点偏移(代直线度)", ttk.Label(id_box, textvariable=getattr(app, "id_endoff_var", tk.StringVar(value="--"))))
    _kv(id_box, 7, "最大内圆真圆度", ttk.Label(id_box, textvariable=app.max_id_round_var), pady=(2, 8))

    # ---- Overall ----
    _kv(all_box, 0, "整体同心度", ttk.Label(all_box, textvariable=getattr(app, "axis_dist_var", tk.StringVar(value="--"))), pady=(8, 2))
    _kv(all_box, 1, "截面同心度max", ttk.Label(all_box, textvariable=getattr(app, "conc_max_var", tk.StringVar(value="--"))))
    _kv(all_box, 2, "轴线最大间距", ttk.Label(all_box, textvariable=getattr(app, "axis_span_max_var", tk.StringVar(value="--"))))
    _kv(all_box, 3, "长度测量值", ttk.Label(all_box, textvariable=getattr(app, "len_meas_var", tk.StringVar(value="--"))), pady=(2, 8))

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

        # OD group
        "od_dev",
        "od_runout",
        "od_round",
        "od_e",
        "od_phi_deg",
        "od_ecc",

        # ID group
        "id_dev",
        "id_runout",
        "id_round",
        "id_e",
        "id_phi_deg",
        "id_ecc",

        # cross
        "concentricity",

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

        # OD
        "od_dev",
        "od_round",
        "od_e",
        "od_phi_deg",
        "od_ecc",

        # ID
        "id_dev",
        "id_round",
        "id_e",
        "id_phi_deg",
        "id_ecc",

        # cross
        "concentricity",
    )

    tree_wrap = ttk.Frame(mid)
    tree_wrap.pack(fill=tk.BOTH, expand=True)

    # Two-level header: top row is a custom canvas (group headers),
    # second row is the Treeview built-in headings.
    header_canvas = tk.Canvas(tree_wrap, height=24, highlightthickness=0)
    header_canvas.pack(side=tk.TOP, fill=tk.X)

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
    # Runout definition (when new edge/chord algorithms are enabled): diameter runout ~= 2*eccentricity amplitude.
    app.result_tree.heading("od_runout", text="外径径向跳动(2e,mm)")
    app.result_tree.heading("od_round", text="外径真圆度(mm)")
    app.result_tree.heading("od_e", text="外圆偏心幅值(mm)")
    app.result_tree.heading("od_phi_deg", text="外圆偏心角(°)")
    app.result_tree.heading("od_ecc", text="外圆轴线偏差(mm)")
    app.result_tree.heading("id_dev", text="内径偏差(mm)")
    app.result_tree.heading("id_runout", text="内径径向跳动(2e,mm)")
    app.result_tree.heading("id_round", text="内径真圆度(mm)")
    app.result_tree.heading("id_e", text="内圆偏心幅值(mm)")
    app.result_tree.heading("id_phi_deg", text="内圆偏心角(°)")
    app.result_tree.heading("id_ecc", text="内圆轴线偏差(mm)")
    app.result_tree.heading("concentricity", text="同心度(mm)")
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
    app.result_tree.column("od_e", width=115, anchor="e")
    app.result_tree.column("od_phi_deg", width=110, anchor="e")
    app.result_tree.column("od_ecc", width=115, anchor="e")
    app.result_tree.column("id_dev", width=110, anchor="e")
    app.result_tree.column("id_runout", width=125, anchor="e")
    app.result_tree.column("id_round", width=115, anchor="e")
    app.result_tree.column("id_e", width=115, anchor="e")
    app.result_tree.column("id_phi_deg", width=110, anchor="e")
    app.result_tree.column("id_ecc", width=115, anchor="e")
    app.result_tree.column("concentricity", width=95, anchor="e")

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

    def _on_xscroll(first, last):
        # Keep scrollbar and custom group header in sync with Treeview's xview.
        try:
            xsb.set(first, last)
        except Exception:
            pass
        try:
            header_canvas.xview_moveto(first)
        except Exception:
            pass

    app.result_tree.configure(xscrollcommand=_on_xscroll)
    xsb.pack(side=tk.BOTTOM, fill=tk.X)

    def _draw_group_header() -> None:
        try:
            f0 = float((header_canvas.xview() or (0.0, 1.0))[0])
        except Exception:
            f0 = 0.0
        try:
            header_canvas.delete("all")
        except Exception:
            return

        # Derive style for a heading-like appearance
        try:
            st = ttk.Style()
            bg = st.lookup("Treeview.Heading", "background") or header_canvas.cget("bg")
            fg = st.lookup("Treeview.Heading", "foreground") or "black"
            font = st.lookup("Treeview.Heading", "font") or None
            header_canvas.configure(bg=bg)
        except Exception:
            fg = "black"
            font = None
        groups = [
            ("位置", ("idx", "x_ui")),
            ("外圆", ("od_dev", "od_round", "od_e", "od_phi_deg", "od_ecc")),
            ("内圆", ("id_dev", "id_round", "id_e", "id_phi_deg", "id_ecc")),
            ("综合", ("concentricity",)),
        ]

        # Compute per-column x positions based on the actual displaycolumns order.
        # Add a small separator width so group headers align with Treeview headings.
        order = list(visible_cols)
        sep_px = 1
        pos = {}
        widths = {}
        x = 0
        for i, c in enumerate(order):
            pos[c] = x
            try:
                w = int(app.result_tree.column(c, "width") or 0)
            except Exception:
                w = 0
            widths[c] = max(0, w)
            x += widths[c]
            if i != (len(order) - 1):
                x += sep_px

        h = 24
        for name, gcols in groups:
            g = [c for c in gcols if c in pos]
            if not g:
                continue
            x0 = pos[g[0]]
            last = g[-1]
            x1 = pos[last] + widths.get(last, 0)
            try:
                header_canvas.create_rectangle(x0, 0, x1, h, outline="")
                header_canvas.create_text((x0 + x1) / 2.0, h / 2.0, text=str(name), fill=fg, font=font)
            except Exception:
                pass

        # scrollregion based on total width of visible columns
        try:
            total = max(int(x), 1)
            header_canvas.configure(scrollregion=(0, 0, total, h))
            try:
                header_canvas.xview_moveto(f0)
            except Exception:
                pass
        except Exception:
            pass

    # Initial draw and redraw on resize
    _draw_group_header()
    try:
        app.result_tree.bind("<Configure>", lambda _e: _draw_group_header())
    except Exception:
        pass

    app._refresh_auto_std_panel()
