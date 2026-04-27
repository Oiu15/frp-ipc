from __future__ import annotations

"""Main measurement screen UI only."""

import tkinter as tk
from tkinter import ttk


def build_main_screen(parent: ttk.Frame, *, presenter, controller, ui) -> None:
    top = ttk.Frame(parent)
    top.pack(fill=tk.X, pady=6)
    top.grid_columnconfigure(0, weight=1)
    top.grid_columnconfigure(1, weight=3)
    top.grid_columnconfigure(2, weight=0)
    top.grid_rowconfigure(0, weight=1)

    st = ttk.LabelFrame(top, text="测量状态")
    st.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
    st.columnconfigure(1, weight=1)
    st.rowconfigure(8, weight=1)

    ttk.Label(st, text="流水号").grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")
    ttk.Label(st, textvariable=presenter.pipe_sn_var, font=('Segoe UI', 10, 'bold')).grid(row=0, column=1, padx=10, pady=(10, 2), sticky='w')
    ttk.Label(st, text="测量计数").grid(row=1, column=0, padx=10, pady=2, sticky="w")
    ttk.Label(st, textvariable=presenter.meas_seq_var).grid(row=1, column=1, padx=10, pady=2, sticky='w')
    ttk.Label(st, text="开始时间").grid(row=2, column=0, padx=10, pady=2, sticky="w")
    ttk.Label(st, textvariable=presenter.meas_start_var).grid(row=2, column=1, padx=10, pady=2, sticky='w')
    ttk.Label(st, text="耗时").grid(row=3, column=0, padx=10, pady=2, sticky="w")
    ttk.Label(st, textvariable=presenter.meas_elapsed_var).grid(row=3, column=1, padx=10, pady=2, sticky='w')
    ttk.Label(st, textvariable=presenter.auto_progress_var, font=('Segoe UI', 11, 'bold')).grid(row=4, column=0, columnspan=2, padx=10, pady=(6, 2), sticky='w')
    ttk.Label(st, textvariable=presenter.auto_done_var).grid(row=5, column=0, columnspan=2, padx=10, pady=2, sticky='w')
    ttk.Label(st, text="自动状态").grid(row=6, column=0, padx=10, pady=(2, 2), sticky="w")
    ttk.Label(st, textvariable=presenter.auto_state_var).grid(row=6, column=1, padx=10, pady=(2, 2), sticky='w')
    ttk.Label(st, text="检测模式").grid(row=7, column=0, padx=10, pady=(2, 2), sticky="w")
    ttk.Label(st, textvariable=presenter.ui_meas_mode_var).grid(row=7, column=1, padx=10, pady=(2, 2), sticky='w')
    ttk.Label(st, text="信息").grid(row=8, column=0, padx=10, pady=(2, 10), sticky="w")

    lbl_auto_msg = ttk.Label(st, textvariable=presenter.auto_msg_var, justify='left')
    lbl_auto_msg.grid(row=8, column=1, padx=10, pady=(2, 10), sticky='we')

    def _sync_msg_wrap(_e=None) -> None:
        try:
            width = int(lbl_auto_msg.winfo_width() or 0)
            if width > 20:
                lbl_auto_msg.configure(wraplength=width)
        except Exception:
            pass

    try:
        lbl_auto_msg.bind('<Configure>', _sync_msg_wrap)
    except Exception:
        pass

    res = ttk.LabelFrame(top, text="测量结果")
    res.grid(row=0, column=1, sticky='nsew', padx=(0, 10))
    res.grid_columnconfigure(0, weight=1, uniform='sum_cols')
    res.grid_columnconfigure(1, weight=1, uniform='sum_cols')
    res.grid_columnconfigure(2, weight=1, uniform='sum_cols')

    all_box = ttk.LabelFrame(res, text="总体")
    all_box.grid(row=0, column=0, sticky='nsew', padx=(10, 6), pady=(10, 10))
    all_box.columnconfigure(1, weight=1)

    od_box = ttk.LabelFrame(res, text="外圆")
    od_box.grid(row=0, column=1, sticky='nsew', padx=(6, 6), pady=(10, 10))
    od_box.columnconfigure(1, weight=1)

    id_box = ttk.LabelFrame(res, text="内圆")
    id_box.grid(row=0, column=2, sticky='nsew', padx=(6, 10), pady=(10, 10))
    id_box.columnconfigure(1, weight=1)

    def _kv(box: ttk.LabelFrame, row: int, label: str, widget: tk.Widget, *, pady=(2, 2)) -> None:
        ttk.Label(box, text=label).grid(row=row, column=0, padx=10, pady=pady, sticky='w')
        widget.grid(row=row, column=1, padx=10, pady=pady, sticky='w')

    lbl_od_std = presenter.remember_widget('lbl_od_std', ttk.Label(od_box, text='--'))
    _kv(od_box, 0, '外径标准值', lbl_od_std, pady=(8, 2))
    _kv(od_box, 1, '平均外径', ttk.Label(od_box, textvariable=presenter.od_mean_var))
    _kv(od_box, 2, '外径极差', ttk.Label(od_box, textvariable=presenter.od_range_var))
    _kv(od_box, 3, '最大外径峰峰', ttk.Label(od_box, textvariable=presenter.max_od_pp_var))
    _kv(od_box, 4, '最大外径真圆度', ttk.Label(od_box, textvariable=presenter.max_od_pp_rob_var))
    _kv(od_box, 5, '最大外径拟合残差', ttk.Label(od_box, textvariable=presenter.max_od_fit_res_var))
    _kv(od_box, 6, '外圆轴线倾斜', ttk.Label(od_box, textvariable=presenter.od_tilt_var))
    _kv(od_box, 7, '外圆轴线斜率', ttk.Label(od_box, textvariable=presenter.od_slope_var))
    _kv(od_box, 8, '端点偏移(代直线度)', ttk.Label(od_box, textvariable=presenter.od_endoff_var), pady=(2, 8))

    lbl_id_std = presenter.remember_widget('lbl_id_std', ttk.Label(id_box, text='--'))
    _kv(id_box, 0, '内径标准值', lbl_id_std, pady=(8, 2))
    _kv(id_box, 1, '平均内径', ttk.Label(id_box, textvariable=presenter.id_mean_var))
    _kv(id_box, 2, '内径极差', ttk.Label(id_box, textvariable=presenter.id_range_var))
    _kv(id_box, 3, '内圆轴线倾斜', ttk.Label(id_box, textvariable=presenter.id_tilt_var))
    _kv(id_box, 4, '内圆轴线斜率', ttk.Label(id_box, textvariable=presenter.id_slope_var))
    _kv(id_box, 5, '端点偏移(代直线度)', ttk.Label(id_box, textvariable=presenter.id_endoff_var))
    _kv(id_box, 6, "最大内圆真圆度", ttk.Label(id_box, textvariable=presenter.max_id_round_var), pady=(2, 8))

    _kv(all_box, 0, '整体同心度', ttk.Label(all_box, textvariable=presenter.axis_dist_var), pady=(8, 2))
    _kv(all_box, 1, '截面同心度max', ttk.Label(all_box, textvariable=presenter.conc_max_var))
    _kv(all_box, 2, '轴线最大间距', ttk.Label(all_box, textvariable=presenter.axis_span_max_var))
    _kv(all_box, 3, '长度测量值', ttk.Label(all_box, textvariable=presenter.len_meas_var), pady=(2, 8))

    ctrl = ttk.LabelFrame(top, text="控制")
    ctrl.grid(row=0, column=2, sticky='ns')
    ttk.Button(ctrl, text='开始测量', width=16, command=controller.start_measurement).pack(padx=10, pady=(10, 6))
    ttk.Button(ctrl, text='停止', width=16, command=controller.stop_measurement).pack(padx=10, pady=6)
    ttk.Button(ctrl, text='清空结果', width=16, command=controller.clear_measurement_results).pack(padx=10, pady=(6, 10))

    info_line = ttk.Frame(parent)
    info_line.pack(fill=tk.X, pady=(0, 4))
    info_line.columnconfigure(0, weight=1)
    ttk.Label(info_line, textvariable=presenter.cov_var, anchor='w', justify='left', wraplength=900).grid(row=0, column=0, sticky='we', padx=(10, 10))

    mid = ttk.Frame(parent)
    mid.pack(fill=tk.BOTH, expand=True)

    cols = (
        'idx', 'x_ui',
        'od_dev', 'od_runout', 'od_round', 'od_pp_rob', 'od_fit_res', 'od_e', 'od_phi_deg', 'od_ecc',
        'id_dev', 'id_runout', 'id_round', 'id_e', 'id_phi_deg', 'id_ecc',
        'concentricity',
        'cov_pct', 'miss_bin', 'max_gap_deg', 'revs', 'cov_elapsed_s', 'cov_reason',
    )
    visible_cols = (
        'idx', 'x_ui',
        'od_dev', 'od_pp_rob', 'od_fit_res', 'od_e',
        'id_dev', 'id_round', 'id_e',
        'concentricity',
    )

    tree_wrap = ttk.Frame(mid)
    tree_wrap.pack(fill=tk.BOTH, expand=True)

    enable_group_header = False
    header_canvas = None
    if enable_group_header:
        header_canvas = tk.Canvas(tree_wrap, height=24, highlightthickness=0)
        header_canvas.pack(side=tk.TOP, fill=tk.X)

    result_tree = presenter.remember_widget('result_tree', ttk.Treeview(tree_wrap, columns=cols, displaycolumns=visible_cols, show='headings'))
    result_tree.bind('<<TreeviewSelect>>', controller.handle_main_result_selection)

    headings = {
        'idx': '截面',
        'x_ui': 'OD位置(Z,mm)',
        'od_dev': '外径偏差(mm)',
        'od_runout': '外径径向跳动(2e,mm)',
        'od_round': '外径峰峰(mm)',
        'od_pp_rob': '外径真圆度(mm)',
        'od_fit_res': '外径拟合残差(mm)',
        'od_e': '外圆偏心幅值(mm)',
        'od_phi_deg': '外圆偏心角(°)',
        'od_ecc': '外圆轴线偏差(mm)',
        'id_dev': '内径偏差(mm)',
        'id_runout': '内径径向跳动(2e,mm)',
        'id_round': '内径真圆度(mm)',
        'id_e': '内圆偏心幅值(mm)',
        'id_phi_deg': '内圆偏心角(°)',
        'id_ecc': '内圆轴线偏差(mm)',
        'concentricity': '同心度(mm)',
        'cov_pct': '覆盖率(%)',
        'miss_bin': '缺失bin',
        'max_gap_deg': '最大空窗角(°)',
        'revs': '圈数',
        'cov_elapsed_s': '采样用时(s)',
        'cov_reason': '覆盖判据',
    }
    widths = {
        'idx': 60, 'x_ui': 110, 'od_dev': 110, 'od_runout': 125, 'od_round': 115, 'od_pp_rob': 130,
        'od_fit_res': 130, 'od_e': 115, 'od_phi_deg': 110, 'od_ecc': 115, 'id_dev': 110, 'id_runout': 125,
        'id_round': 115, 'id_e': 115, 'id_phi_deg': 110, 'id_ecc': 115, 'concentricity': 95, 'cov_pct': 90,
        'miss_bin': 80, 'max_gap_deg': 110, 'revs': 70, 'cov_elapsed_s': 95, 'cov_reason': 110,
    }
    min_widths = {
        'idx': 48, 'x_ui': 86, 'od_dev': 88, 'od_runout': 96, 'od_round': 90, 'od_pp_rob': 102,
        'od_fit_res': 102, 'od_e': 92, 'od_phi_deg': 88, 'od_ecc': 92, 'id_dev': 88, 'id_runout': 96,
        'id_round': 90, 'id_e': 92, 'id_phi_deg': 88, 'id_ecc': 92, 'concentricity': 84, 'cov_pct': 78,
        'miss_bin': 70, 'max_gap_deg': 92, 'revs': 62, 'cov_elapsed_s': 82, 'cov_reason': 92,
    }
    for col in cols:
        result_tree.heading(col, text=headings[col])
        result_tree.column(
            col,
            width=widths[col],
            minwidth=min_widths.get(col, 60),
            stretch=False,
            anchor='e' if col not in {'idx', 'cov_reason'} else ('center' if col == 'idx' else 'w'),
        )

    result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    ysb = ttk.Scrollbar(tree_wrap, orient='vertical', command=result_tree.yview)
    result_tree.configure(yscrollcommand=ysb.set)
    ysb.pack(side=tk.RIGHT, fill=tk.Y)

    xsb = ttk.Scrollbar(mid, orient='horizontal', command=result_tree.xview)

    def _on_xscroll(first, last):
        try:
            xsb.set(first, last)
        except Exception:
            pass
        if header_canvas is not None:
            try:
                header_canvas.xview_moveto(first)
            except Exception:
                pass

    result_tree.configure(xscrollcommand=_on_xscroll)
    xsb.pack(side=tk.BOTTOM, fill=tk.X)

    def _draw_group_header() -> None:
        if header_canvas is None:
            return
        try:
            first = float((header_canvas.xview() or (0.0, 1.0))[0])
        except Exception:
            first = 0.0
        try:
            header_canvas.delete('all')
        except Exception:
            return
        try:
            style = ttk.Style()
            bg = style.lookup('Treeview.Heading', 'background') or header_canvas.cget('bg')
            fg = style.lookup('Treeview.Heading', 'foreground') or 'black'
            font = style.lookup('Treeview.Heading', 'font') or None
            header_canvas.configure(bg=bg)
        except Exception:
            fg = 'black'
            font = None

        groups = [
            ("位置", ("idx", "x_ui")),
            ('外圆', ('od_dev', 'od_round', 'od_e', 'od_phi_deg', 'od_ecc')),
            ('内圆', ('id_dev', 'id_round', 'id_e', 'id_phi_deg', 'id_ecc')),
            ("综合", ("concentricity",)),
        ]
        order = list(visible_cols)
        pos: dict[str, int] = {}
        col_widths: dict[str, int] = {}
        x = 0
        for col in order:
            pos[col] = x
            try:
                width = int(result_tree.column(col, 'width') or 0)
            except Exception:
                width = 0
            col_widths[col] = max(0, width)
            x += col_widths[col]

        height = 24
        for name, group_cols in groups:
            group = [col for col in group_cols if col in pos]
            if not group:
                continue
            x0 = pos[group[0]]
            last_col = group[-1]
            x1 = pos[last_col] + col_widths.get(last_col, 0)
            try:
                header_canvas.create_rectangle(x0, 0, x1, height, outline='')
                header_canvas.create_text((x0 + x1) / 2.0, height / 2.0, text=name, fill=fg, font=font)
            except Exception:
                pass

        try:
            total = max(int(x), 1)
            header_canvas.configure(scrollregion=(0, 0, total, height))
            header_canvas.xview_moveto(first)
        except Exception:
            pass

    if header_canvas is not None:
        _draw_group_header()
        try:
            result_tree.bind('<Configure>', lambda _e: _draw_group_header())
        except Exception:
            pass

    presenter.remember_view_state('tree_displaycols_sync', visible_cols)
    presenter.remember_view_state('tree_displaycols_split', visible_cols)
    presenter.remember_view_state('tree_displaycols_od_only', ('idx', 'x_ui', 'od_dev', 'od_pp_rob', 'od_fit_res', 'od_e'))
    presenter.remember_view_state('tree_column_widths', widths)
    presenter.remember_view_state('tree_column_min_widths', min_widths)
    controller.refresh_main_summary_panel()
