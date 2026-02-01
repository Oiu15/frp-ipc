from __future__ import annotations

"""配方/示教页（UI 构建）。"""

from typing import TYPE_CHECKING, List, Tuple
import tkinter as tk
from tkinter import ttk

from config.addresses import AXIS_NAMES, AXIS_COUNT  # AXIS_COUNT 可能暂未用到，保留

if TYPE_CHECKING:  # pragma: no cover
    from app import App


def build_recipe_screen(app: "App", parent: ttk.Frame) -> None:
    """测量配方与示教页面（上下布局：参数/示教/截面结果）。"""

    # ====== Vars（保持你现有命名与业务逻辑）======
    app.recipe_name_var = tk.StringVar(value=app.recipe.name)
    app.pipe_len_var = tk.StringVar(value=str(app.recipe.pipe_len_mm))
    app.clamp_var = tk.StringVar(value=str(app.recipe.clamp_occupy_mm))
    app.margin_h_var = tk.StringVar(value=str(app.recipe.margin_head_mm))
    app.margin_t_var = tk.StringVar(value=str(app.recipe.margin_tail_mm))
    app.meas_total_len_var = tk.StringVar(value=str(getattr(app.recipe, "meas_total_len_mm", 0.0)))
    app.section_n_var = tk.StringVar(value=str(app.recipe.section_count))
    # Teach axes selection (0=OD(AX0), 1=ID(AX1+AX4), 2=OD+ID)
    app.teach_axes_mode_var = tk.IntVar(value=int(getattr(app.recipe, "teach_axes_mode", 2)))

    app.od_std_var = tk.StringVar(value=str(app.recipe.od_std_mm))
    app.id_std_var = tk.StringVar(value=str(app.recipe.id_std_mm))
    app.od_tol_var = tk.StringVar(value=str(app.recipe.od_tol_mm))
    # OD algorithm switch (persisted in recipe)
    app.od_use_edges_var = tk.BooleanVar(value=bool(getattr(app.recipe, "od_use_edges", False)))

    # ID algorithm switch (persisted in recipe)
    app.id_use_fit_var = tk.BooleanVar(value=bool(getattr(app.recipe, "id_use_fit", False)))

    app.points_per_rev_var = tk.StringVar(value=str(app.recipe.points_per_rev))
    app.min_cov_var = tk.StringVar(value=str(getattr(app.recipe, "min_bin_coverage", 0.95)))
    app.sample_timeout_var = tk.StringVar(value=str(getattr(app.recipe, "sample_timeout_s", 5.0)))
    app.max_revs_var = tk.StringVar(value=str(getattr(app.recipe, "max_revolutions", 2.0)))

    # Rotate measurement speed (AX3 VelMove speed)
    app.rot_vel_velmove_var = tk.StringVar(value=str(getattr(app.recipe, "rot_vel_velmove", 200.0)))

    # Fit strategy (persisted in recipe)
    FIT_STRATEGY_CHOICES = [
        "a 原始点拟合",
        "b 原始点按bin权重均衡",
        "c bin中心角+r_bin标量平均",
    ]
    app.fit_strategy_var = tk.StringVar(value=str(getattr(app.recipe, "fit_strategy", "b 原始点按bin权重均衡")))

    # ====== Length measurement vars (persisted in recipe) ======
    app.len_enable_var = tk.BooleanVar(value=bool(getattr(app.recipe, "len_enable", False)))
    # Bottom-approach is stored as AX0 absolute act_pos (mm) to decouple from Start(Z_Pos)
    legacy_z = float(getattr(app.recipe, "len_z_low_approach", 1300.0))
    abs_appr = float(getattr(app.recipe, "len_low_approach_abs", 0.0) or 0.0)
    if abs_appr == 0.0:
        try:
            abs_appr = float(app.axis_cal.z_disp_to_abs(0, legacy_z))
        except Exception:
            abs_appr = 0.0
    app.len_z_low_approach_var = tk.StringVar(value=str(abs_appr))
    app.len_low_search_dist_var = tk.StringVar(value=str(getattr(app.recipe, "len_low_search_dist", 220.0)))
    app.len_high_search_dist_var = tk.StringVar(value=str(getattr(app.recipe, "len_high_search_dist", 220.0)))
    app.len_search_vel_var = tk.StringVar(value=str(getattr(app.recipe, "len_search_vel", 5.0)))
    app.len_search_timeout_var = tk.StringVar(value=str(getattr(app.recipe, "len_search_timeout_s", 12.0)))
    app.len_tol_var = tk.StringVar(value=str(getattr(app.recipe, "len_tol_mm", 20.0)))

    # Advanced (folded by default)
    app.len_high_margin_var = tk.StringVar(value=str(getattr(app.recipe, "len_high_margin", 20.0)))
    app.len_debounce_k_var = tk.StringVar(value=str(getattr(app.recipe, "len_debounce_k", 6)))
    app.len_max_stale_ms_var = tk.StringVar(value=str(getattr(app.recipe, "len_max_stale_ms", 300)))
    app.len_backoff_var = tk.StringVar(value=str(getattr(app.recipe, "len_backoff_mm", 2.0)))


    # ====== Page layout ======
    # 上部：参数/示教/边沿搜索（可滚动）
    # 下部：截面位置表（尽量保持较大高度，便于查看）
    #
    # NOTE: Notebook 页通常使用 pack，但页内可自由使用 grid。
    parent.grid_rowconfigure(0, weight=1)
    parent.grid_columnconfigure(0, weight=1)

    paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
    paned.grid(row=0, column=0, sticky="nsew")

    upper = ttk.Frame(paned)
    lower = ttk.Frame(paned)

    # Weight hints: prefer giving more height to the upper controls.
    try:
        # 60% / 40% feel on resize
        paned.add(upper, weight=3)
        paned.add(lower, weight=2)
    except Exception:
        paned.add(upper)
        paned.add(lower)

    # 初始分割位置：上部约 60%，下部约 40%
    # NOTE:
    # - Notebook 中的 Tab 在“未选中”时高度通常为 1，直接 winfo_height 会拿到错误值。
    # - 因此采用：Map 事件触发 + after 重试，直到拿到可靠高度。
    def _init_sash(_retry: int = 0):
        try:
            if getattr(app, "_recipe_sash_inited", False):
                return
        except Exception:
            pass

        try:
            paned.update_idletasks()
            h = int(paned.winfo_height() or 0)
        except Exception:
            h = 0

        # 还没布局完成 / Tab 未显示：继续重试
        if h < 200 and _retry < 40:
            try:
                parent.after(80, lambda: _init_sash(_retry + 1))
            except Exception:
                pass
            return

        try:
            if h >= 200:
                paned.sashpos(0, int(h * 0.60))
                setattr(app, "_recipe_sash_inited", True)
        except Exception:
            pass

    # 当 Tab 真正显示出来时再执行一次（更可靠）
    try:
        paned.bind("<Map>", lambda _e: _init_sash(0))
    except Exception:
        pass

    # 再兜底：页面创建后延时尝试
    try:
        parent.after(120, lambda: _init_sash(0))
    except Exception:
        pass

    # --- build a scrollable area in upper ---
    upper.grid_rowconfigure(0, weight=1)
    upper.grid_columnconfigure(0, weight=1)

    canvas = tk.Canvas(upper, highlightthickness=0)
    vscroll = ttk.Scrollbar(upper, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vscroll.set)

    canvas.grid(row=0, column=0, sticky="nsew")
    vscroll.grid(row=0, column=1, sticky="ns")

    ui = ttk.Frame(canvas)
    _win = canvas.create_window((0, 0), window=ui, anchor="nw")

    def _on_ui_config(_evt=None):
        try:
            canvas.configure(scrollregion=canvas.bbox("all"))
        except Exception:
            pass

    def _on_canvas_config(evt):
        try:
            canvas.itemconfigure(_win, width=evt.width)
        except Exception:
            pass

    ui.bind("<Configure>", _on_ui_config)
    canvas.bind("<Configure>", _on_canvas_config)

    # Mouse wheel scrolling (only when cursor is over the canvas)
    def _on_mousewheel(evt):
        try:
            # Windows/macOS: evt.delta; Linux: handled by Button-4/5
            if getattr(evt, 'delta', 0):
                canvas.yview_scroll(int(-1 * (evt.delta / 120)), 'units')
        except Exception:
            pass

    def _on_button4(_evt):
        try:
            canvas.yview_scroll(-1, 'units')
        except Exception:
            pass

    def _on_button5(_evt):
        try:
            canvas.yview_scroll(1, 'units')
        except Exception:
            pass

    def _on_enter(_evt=None):
        try:
            canvas.focus_set()
        except Exception:
            pass

    canvas.bind('<Enter>', _on_enter)
    canvas.bind('<MouseWheel>', _on_mousewheel)
    canvas.bind('<Button-4>', _on_button4)
    canvas.bind('<Button-5>', _on_button5)

    # Layout for scrollable content (2 columns)
    ui.grid_rowconfigure(0, weight=0)
    ui.grid_rowconfigure(1, weight=0)
    ui.grid_columnconfigure(0, weight=4)
    ui.grid_columnconfigure(1, weight=2, minsize=360)

    # ---------------- Top: Recipe Params ----------------
    top = ttk.Frame(ui)
    top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 6))
    top.grid_columnconfigure(0, weight=1)

    # 让参数区内部更容易扩展：用 3 个分组框（语义分组）
    grp = ttk.Frame(top)
    grp.grid(row=0, column=0, sticky="ew")
    grp.grid_columnconfigure(0, weight=1)
    grp.grid_columnconfigure(1, weight=1)
    grp.grid_columnconfigure(2, weight=1)
    grp.grid_columnconfigure(3, weight=1)
    grp.grid_columnconfigure(4, weight=1)

    box_geom = ttk.LabelFrame(grp, text="工件 / 几何参数")
    box_plan = ttk.LabelFrame(grp, text="扫描 / 截面规划")
    box_center = ttk.LabelFrame(grp, text="中心架位置")
    box_len = ttk.LabelFrame(grp, text="长度测量")
    box_meas = ttk.LabelFrame(grp, text="测量 / 判定参数")

    box_geom.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    box_plan.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
    box_center.grid(row=0, column=2, sticky="nsew", padx=(0, 8))
    box_len.grid(row=0, column=3, sticky="nsew", padx=(0, 8))
    box_meas.grid(row=0, column=4, sticky="nsew")

    for b in (box_geom, box_plan, box_center, box_len, box_meas):
        b.grid_columnconfigure(0, weight=0)
        b.grid_columnconfigure(1, weight=1)

    # --- 中心架位置：AX2 的两个工位（长度测量位 / 旋转测量位）---
    box_center.grid_columnconfigure(0, weight=1)
    ttk.Button(box_center, text="保存为长度测量位", command=app._save_ax2_len_pos).grid(
        row=0, column=0, sticky="ew", padx=6, pady=(6, 4)
    )
    ttk.Button(box_center, text="保存为旋转测量位", command=app._save_ax2_rot_pos).grid(
        row=1, column=0, sticky="ew", padx=6, pady=(0, 6)
    )
    # 原先复用在“示教”里的两个按钮拆开，常驻此处
    ttk.Button(box_center, text="移动到长度测量位", command=app._teach_move_ax2_to_len_pos).grid(
        row=2, column=0, sticky="ew", padx=6, pady=(0, 4)
    )
    ttk.Button(box_center, text="移动到旋转测量位", command=app._teach_move_ax2_to_rot_pos).grid(
        row=3, column=0, sticky="ew", padx=6, pady=(0, 6)
    )
    ttk.Label(box_center, text="已保存位置").grid(row=4, column=0, sticky="w", padx=6)
    app.center_pos_var = tk.StringVar(value="--")
    ttk.Label(box_center, textvariable=app.center_pos_var, justify="left").grid(
        row=5, column=0, sticky="w", padx=6, pady=(2, 6)
    )

    # --- 表驱动字段定义：后续新增参数，往这里加一行即可 ---
    # (label, var)
    GEOM_FIELDS: List[Tuple[str, tk.Variable]] = [
        ("管长(mm)", app.pipe_len_var),
        ("测量总长(mm)", app.meas_total_len_var),
        ("夹爪占用(mm)", app.clamp_var),
        ("头部留边(mm)", app.margin_h_var),
        ("尾部留边(mm)", app.margin_t_var),
    ]
    PLAN_FIELDS: List[Tuple[str, tk.Variable]] = [
        ("截面数量(N)", app.section_n_var),
        # 示教轴是 Combobox，单独渲染
    ]
    MEAS_FIELDS: List[Tuple[str, tk.Variable]] = [
        ("OD标准(mm)", app.od_std_var),
        ("ID标准(mm)", app.id_std_var),
        ("OD公差(±mm)", app.od_tol_var),
        ("每圈采样点数", app.points_per_rev_var),
        ("采样覆盖率(0~1)", app.min_cov_var),
        ("单截面超时(s)", app.sample_timeout_var),
        ("最大采样圈数(转)", app.max_revs_var),
        ("旋转测量速度(AX3 VelMove)", app.rot_vel_velmove_var),
    ]

    # ---------------- Length measurement panel ----------------
    # Basic (operator-facing) fields
    LEN_FIELDS: List[Tuple[str, tk.Variable]] = [
        ("底边接近位abs(mm)", app.len_z_low_approach_var),
        ("底边慢搜距离(mm)", app.len_low_search_dist_var),
        ("顶边慢搜距离(mm)", app.len_high_search_dist_var),
        ("慢搜速度(mm/s)", app.len_search_vel_var),
    ]

    LEN_ADV_FIELDS: List[Tuple[str, tk.Variable]] = [
        ("搜寻超时(s)", app.len_search_timeout_var),
        ("长度容差(mm)", app.len_tol_var),
        ("顶边接近余量(mm)", app.len_high_margin_var),
        ("去抖次数(k)", app.len_debounce_k_var),
        ("数据停更阈值(ms)", app.len_max_stale_ms_var),
        ("触发后回退(mm)", app.len_backoff_var),
    ]

    # Header row: enable checkbox
    box_len.grid_columnconfigure(0, weight=0)
    box_len.grid_columnconfigure(1, weight=1)
    hdr = ttk.Frame(box_len)
    hdr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 2))
    hdr.grid_columnconfigure(1, weight=1)
    ttk.Checkbutton(hdr, text="启用长度测量", variable=app.len_enable_var).grid(
        row=0, column=0, sticky="w"
    )
    ttk.Button(hdr, text="取当前OD位置", command=getattr(app, "_len_pick_low_approach", None) or (lambda: None)).grid(
        row=0, column=1, sticky="e"
    )

    r_len = 1
    for label, var in LEN_FIELDS:
        app._kv_row(box_len, label, var, r_len)
        r_len += 1

    # Advanced toggle
    adv_state = tk.BooleanVar(value=False)
    adv_btn = ttk.Button(box_len, text="高级参数 ▾")
    adv_btn.grid(row=r_len, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 2))
    r_len += 1

    adv_frm = ttk.Frame(box_len)
    adv_frm.grid(row=r_len, column=0, columnspan=2, sticky="ew", padx=0, pady=0)
    adv_frm.grid_columnconfigure(0, weight=0)
    adv_frm.grid_columnconfigure(1, weight=1)
    # start hidden
    adv_frm.grid_remove()

    def _toggle_adv() -> None:
        try:
            v = bool(adv_state.get())
            if v:
                adv_state.set(False)
                adv_frm.grid_remove()
                adv_btn.config(text="高级参数 ▾")
            else:
                adv_state.set(True)
                adv_frm.grid()
                adv_btn.config(text="高级参数 ▴")
        except Exception:
            pass

    adv_btn.config(command=_toggle_adv)

    rr = 0
    for label, var in LEN_ADV_FIELDS:
        ttk.Label(adv_frm, text=label).grid(row=rr, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(adv_frm, width=18, textvariable=var).grid(row=rr, column=1, sticky="w", padx=6, pady=4)
        rr += 1

    # Read-only info (Lmax / status)
    app.len_info_var = tk.StringVar(value="--")
    app.len_status_var = tk.StringVar(value="--")
    info = ttk.Frame(box_len)
    info.grid(row=r_len + 1, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 6))
    info.grid_columnconfigure(1, weight=1)
    ttk.Label(info, text="理论最大可测长度Lmax(mm)").grid(row=0, column=0, sticky="w")
    ttk.Label(info, textvariable=app.len_info_var).grid(row=0, column=1, sticky="e")
    ttk.Label(info, text="长度测量状态").grid(row=1, column=0, sticky="w")
    ttk.Label(info, textvariable=app.len_status_var).grid(row=1, column=1, sticky="e")

    # 渲染：几何参数
    # 配方名：Combobox（可选择历史配方，也可手动输入新名称）
    ttk.Label(box_geom, text="配方名").grid(row=0, column=0, sticky="e", padx=6, pady=4)
    app.recipe_name_combo = ttk.Combobox(
        box_geom,
        textvariable=app.recipe_name_var,
        state="normal",  # allow typing
        width=18,
        values=[],
    )
    app.recipe_name_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
    app.recipe_name_combo.bind("<<ComboboxSelected>>", app._on_recipe_selected)
    app.recipe_name_combo.bind("<Return>", app._on_recipe_enter)

    r = 1
    for label, var in GEOM_FIELDS:
        app._kv_row(box_geom, label, var, r)
        r += 1

    # 渲染：截面规划
    r = 0
    for label, var in PLAN_FIELDS:
        app._kv_row(box_plan, label, var, r)
        r += 1

    ttk.Label(box_plan, text="示教轴").grid(row=r, column=0, sticky="e", padx=6, pady=4)
    app.teach_axes_combo = ttk.Combobox(
        box_plan,
        state="readonly",
        width=18,
        values=[
            "外径AX0",
            "内径AX1+4",
            "内径+外径AX0+1+4",
            "中心架AX2",
        ],
    )
    app.teach_axes_combo.current(int(app.teach_axes_mode_var.get()))
    app.teach_axes_combo.grid(row=r, column=1, sticky="ew", padx=6, pady=4)
    app.teach_axes_combo.bind("<<ComboboxSelected>>", app._on_teach_axes_selected)
    r += 1

    # 点动（按住运行）——示教界面的点动逻辑已移除（f1_13 起禁用）
    # 说明：点动请在“轴参数/调试”页使用。
    jog = ttk.Frame(box_plan)
    jog.grid(row=r, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 2))
    jog.grid_columnconfigure(0, weight=1)
    jog.grid_columnconfigure(1, weight=1)

    ttk.Label(jog, text="点动（按住运行）(已禁用)").grid(row=0, column=0, columnspan=2, sticky="w")
    btn_tjneg = ttk.Button(jog, text="Jog -", state="disabled")
    btn_tjpos = ttk.Button(jog, text="Jog +", state="disabled")
    btn_tjneg.grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=(4, 0))
    btn_tjpos.grid(row=1, column=1, sticky="ew", pady=(4, 0))
    r += 1

    # 相对运动
    rel = ttk.Frame(box_plan)
    rel.grid(row=r, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 0))
    rel.grid_columnconfigure(1, weight=1)
    app.teach_rel_dist_var = tk.StringVar(value="10")
    ttk.Label(rel, text="相对移动(mm)").grid(row=0, column=0, sticky="e", padx=(0, 6))
    ttk.Entry(rel, textvariable=app.teach_rel_dist_var, width=10).grid(
        row=0, column=1, sticky="w"
    )
    ttk.Button(rel, text="执行", command=app._teach_move_relative).grid(
        row=0, column=2, sticky="w", padx=(8, 0)
    )
    r += 1

    # 对齐按钮：放到“扫描/截面规划”区，紧跟在相对移动下方（避免占用示教区高度）
    align = ttk.Frame(box_plan)
    align.grid(row=r, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 0))
    align.grid_columnconfigure(0, weight=1)
    align.grid_columnconfigure(1, weight=1)
    ttk.Button(align, text="以OD截面为准对齐", command=app._teach_align_by_od).grid(
        row=0, column=0, sticky="ew", padx=(0, 6)
    )
    ttk.Button(align, text="以ID截面为准对齐", command=app._teach_align_by_id).grid(
        row=0, column=1, sticky="ew"
    )
    r += 1

    # (对齐按钮已在上方 align Frame 中渲染)
    # 渲染：测量/判定参数
    r = 0
    for label, var in MEAS_FIELDS:
        app._kv_row(box_meas, label, var, r)
        r += 1

    # ---------------- 算法参数（折叠） ----------------
    algo_open = tk.BooleanVar(value=False)

    algo_header = ttk.Frame(box_meas)
    algo_header.grid(row=r, column=0, columnspan=2, sticky="ew", padx=6, pady=(8, 2))
    algo_header.grid_columnconfigure(0, weight=1)

    algo_btn_text = tk.StringVar(value="算法参数 ▸")
    def _toggle_algo():
        if algo_open.get():
            algo_open.set(False)
            algo_body.grid_remove()
            algo_btn_text.set("算法参数 ▸")
        else:
            algo_open.set(True)
            algo_body.grid()
            algo_btn_text.set("算法参数 ▾")

    ttk.Button(algo_header, textvariable=algo_btn_text, command=_toggle_algo).grid(
        row=0, column=0, sticky="w"
    )

    algo_body = ttk.Frame(box_meas)
    algo_body.grid(row=r + 1, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 4))
    algo_body.grid_columnconfigure(1, weight=1)
    algo_body.grid_remove()

    ttk.Checkbutton(
        algo_body,
        text="外径使用新算法（OUT1+OUT2+B）",
        variable=app.od_use_edges_var,
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

    ttk.Checkbutton(
        algo_body,
        text="内径使用新算法（OUT4弦长 + m拟合直径）[预留]",
        variable=app.id_use_fit_var,
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))

    ttk.Label(algo_body, text="拟合算法").grid(row=2, column=0, sticky="e", padx=(0, 6), pady=4)
    app.fit_strategy_combo = ttk.Combobox(
        algo_body,
        textvariable=app.fit_strategy_var,
        values=FIT_STRATEGY_CHOICES,
        width=22,
        state="readonly",
    )
    # ensure current value is valid
    try:
        cur = str(app.fit_strategy_var.get() or "")
        if cur not in FIT_STRATEGY_CHOICES:
            cur = "b 原始点按bin权重均衡"
            app.fit_strategy_var.set(cur)
        app.fit_strategy_combo.current(FIT_STRATEGY_CHOICES.index(cur))
    except Exception:
        pass
    app.fit_strategy_combo.grid(row=2, column=1, sticky="w", pady=4)

    ttk.Label(
        algo_body,
        text="提示：新算法要求测径仪请求返回 OUT1/OUT2（如 M0）。未配置 OUT2 或未标定 B 时将自动回退旧算法。",
        foreground="#555",
    ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

    r = r + 2

    # --- Top buttons (统一放在参数区下方，避免散落) ---
    btns = ttk.Frame(top)
    btns.grid(row=2, column=0, sticky="ew", pady=(8, 0))
    btns.grid_columnconfigure(0, weight=1)

    btn_left = ttk.Frame(btns)
    btn_left.grid(row=0, column=0, sticky="w")

    ttk.Button(btn_left, text="计算截面位置", command=app._recipe_compute).pack(
        side=tk.LEFT, padx=(0, 6)
    )
    ttk.Button(btn_left, text="保存配方", command=app._recipe_save_backend).pack(
        side=tk.LEFT, padx=(0, 6)
    )
    ttk.Button(btn_left, text="删除配方", command=app._recipe_delete_backend).pack(
        side=tk.LEFT
    )

    # 小提示放右侧，减少纵向占用
    hint = ttk.Label(
        btns,
        text=(
            "说明：截面位置使用 Z_Pos（Z坐标，向下为正）。\n"
            "可选择示教轴（OD/ID/OD+ID），示教动作会联动所选轴。"
        ),
        justify="left",
    )
    hint.grid(row=0, column=1, sticky="e", padx=(12, 0))

    # ---------------- Mid: Teach ----------------
    mid = ttk.LabelFrame(ui, text="示教")
    mid.grid(row=1, column=0, sticky="nsew", padx=(8, 6), pady=(0, 6))
    mid.grid_columnconfigure(0, weight=1)
    mid.grid_columnconfigure(1, weight=1)
    mid.grid_columnconfigure(2, weight=1)

    # 左：动作按钮
    teach_actions = ttk.Frame(mid)
    teach_actions.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
    teach_actions.grid_columnconfigure(0, weight=1)

    app.teach_btn_move = ttk.Button(
        teach_actions, text="移动示教轴到选中截面", command=app._teach_move_to_selected
    )
    app.teach_btn_move.grid(row=0, column=0, sticky="ew", pady=(0, 6))

    app.teach_btn_update = ttk.Button(
        teach_actions,
        text="保存截面位置",
        command=app._teach_save_current_to_selected,
    )
    app.teach_btn_update.grid(row=1, column=0, sticky="ew", pady=(0, 6))

    ttk.Button(
        teach_actions, text="保存为测量区间起始位(Start)", command=app._teach_save_start
    ).grid(row=2, column=0, sticky="ew", pady=(0, 6))

    # Start/End 快捷移动（示教轴为AX2时置灰，由 _refresh_teach_action_buttons 控制）
    app.teach_btn_goto_start = ttk.Button(
        teach_actions, text="移动示教轴到Start位", command=app._teach_goto_start
    )
    app.teach_btn_goto_start.grid(row=3, column=0, sticky="ew", pady=(0, 6))

    app.teach_btn_goto_end = ttk.Button(
        teach_actions, text="移动示教轴到End位", command=app._teach_goto_end
    )
    app.teach_btn_goto_end.grid(row=4, column=0, sticky="ew")

    # 中：当前位置显示
    teach_status = ttk.Frame(mid)
    teach_status.grid(row=0, column=1, sticky="ew", padx=8, pady=8)
    teach_status.grid_columnconfigure(0, weight=1)

    app.teach_abs_var = tk.StringVar(value="--")
    app.teach_z_var = tk.StringVar(value="--")
    app.teach_align_var = tk.StringVar(value="--")
    app.teach_mode_var = tk.StringVar(value="--")
    app.teach_axes_var = tk.StringVar(value="--")
    app.start_info_var = tk.StringVar(value="Start: 未设置")

    ttk.Label(teach_status, text="示教区").grid(row=0, column=0, sticky="w")
    ttk.Label(teach_status, textvariable=app.teach_mode_var).grid(
        row=2, column=0, sticky="w", pady=(4, 0)
    )
    ttk.Label(teach_status, textvariable=app.teach_align_var).grid(
        row=3, column=0, sticky="w", pady=(2, 0)
    )
    ttk.Label(teach_status, textvariable=app.teach_abs_var).grid(
        row=4, column=0, sticky="w", pady=(2, 0)
    )
    ttk.Label(teach_status, textvariable=app.teach_z_var).grid(
        row=5, column=0, sticky="w", pady=(2, 0)
    )
    ttk.Label(teach_status, textvariable=app.teach_axes_var).grid(
        row=6, column=0, sticky="w", pady=(2, 0)
    )

    ttk.Label(teach_status, textvariable=app.start_info_var).grid(
        row=7, column=0, sticky="w", pady=(2, 0)
    )

    # 右：待定点（AX0/AX1/AX4）
    standby = ttk.LabelFrame(mid, text="待定点")
    standby.grid(row=0, column=2, sticky="nsew", padx=8, pady=8)
    standby.grid_columnconfigure(0, weight=1)

    app.standby_info_var = tk.StringVar(value="未设置")
    app.standby_state_var = tk.StringVar(value="-")

    ttk.Button(
        standby, text="将当下位置保存为待定位", command=app._teach_save_standby
    ).grid(row=0, column=0, sticky="ew", pady=(0, 6))
    ttk.Button(
        standby, text="回到待定位", command=app._teach_go_standby
    ).grid(row=2, column=0, sticky="ew", pady=(0, 6))

    ttk.Button(
        standby, text="待机位→设置Start", command=app._teach_start_from_standby
    ).grid(row=3, column=0, sticky="ew", pady=(0, 6))

    ttk.Label(standby, textvariable=app.standby_state_var).grid(
        row=4, column=0, sticky="w", pady=(2, 0)
    )
    ttk.Label(standby, textvariable=app.standby_info_var, justify="left").grid(
        row=5, column=0, sticky="w", pady=(2, 0)
    )

    # ---------------- Right: Length edge search ----------------
    # Button-1: search bottom edge (GO -> HI) and lock AX0 Z_disp.
    # Button-2: search top edge (GO -> HI) and lock AX0 Z_disp.
    # NOTE: 放到“示教”右侧，减少纵向占用，避免“截面计算结果”被挤出屏幕。
    len_dbg = ttk.LabelFrame(ui, text="长度边沿搜索（AX0 + 测径仪比较器）")
    len_dbg.grid(row=1, column=1, sticky="nsew", padx=(0, 8), pady=(0, 6))
    len_dbg.grid_columnconfigure(0, weight=1)

    # 两个按钮 + 3 行状态（压缩高度，给“截面计算结果”留出空间）
    app.btn_len_search_low = ttk.Button(
        len_dbg, text="尝试搜索底边(GO→HI)", command=getattr(app, "_teach_len_search_low_toggle", None)
    )
    app.btn_len_search_low.grid(row=0, column=0, sticky="ew", padx=8, pady=(10, 6))

    app.btn_len_search_high = ttk.Button(
        len_dbg, text="尝试搜索顶边(GO→HI)", command=getattr(app, "_teach_len_search_high_toggle", None)
    )
    app.btn_len_search_high.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))

    app.len_edge_state_var = tk.StringVar(value="--")
    app.len_edge_low_var = tk.StringVar(value="--")
    app.len_edge_high_var = tk.StringVar(value="--")

    ttk.Label(len_dbg, textvariable=app.len_edge_state_var, justify="left").grid(
        row=3, column=0, sticky="w", padx=10, pady=(0, 6)
    )

    row_low = ttk.Frame(len_dbg)
    row_low.grid(row=4, column=0, sticky="ew", padx=10)
    row_low.grid_columnconfigure(1, weight=1)
    ttk.Label(row_low, text="底边Z_disp:").grid(row=0, column=0, sticky="w")
    ttk.Label(row_low, textvariable=app.len_edge_low_var).grid(row=0, column=1, sticky="e")

    row_high = ttk.Frame(len_dbg)
    row_high.grid(row=5, column=0, sticky="ew", padx=10, pady=(6, 10))
    row_high.grid_columnconfigure(1, weight=1)
    ttk.Label(row_high, text="顶边Z_disp:").grid(row=0, column=0, sticky="w")
    ttk.Label(row_high, textvariable=app.len_edge_high_var).grid(row=0, column=1, sticky="e")

    app.len_edge_len_var = tk.StringVar(value="--")
    row_len = ttk.Frame(len_dbg)
    row_len.grid(row=6, column=0, sticky="ew", padx=10, pady=(0, 10))
    row_len.grid_columnconfigure(1, weight=1)
    ttk.Label(row_len, text="实测管长(mm):").grid(row=0, column=0, sticky="w")
    ttk.Label(row_len, textvariable=app.len_edge_len_var).grid(row=0, column=1, sticky="e")    # ---------------- Bottom: Result Table ----------------
    # 固定在下部 pane，保证表格高度。
    lower.grid_rowconfigure(0, weight=1)
    lower.grid_columnconfigure(0, weight=1)

    bottom = ttk.LabelFrame(lower, text="测量截面位置计算结果")
    bottom.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
    bottom.grid_rowconfigure(0, weight=1)
    bottom.grid_columnconfigure(0, weight=1)

    table_wrap = ttk.Frame(bottom)
    table_wrap.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
    table_wrap.grid_rowconfigure(0, weight=1)
    table_wrap.grid_columnconfigure(0, weight=1)

    cols = ("idx", "z_od", "z_id", "ax0_abs", "ax1_abs", "ax4_abs", "mode")
    app.recipe_tree = ttk.Treeview(table_wrap, columns=cols, show="headings", height=16)

    app.recipe_tree.heading("idx", text="截面")
    app.recipe_tree.heading("z_od", text="OD位置(mm)")
    app.recipe_tree.heading("z_id", text="ID位置(mm)")
    app.recipe_tree.heading("ax0_abs", text="AX0 abs(mm)")
    app.recipe_tree.heading("ax1_abs", text="AX1 abs(mm)")
    app.recipe_tree.heading("ax4_abs", text="AX4 abs(mm)")
    app.recipe_tree.heading("mode", text="来源")

    app.recipe_tree.column("idx", width=60, anchor="center")
    app.recipe_tree.column("z_od", width=120, anchor="e")
    app.recipe_tree.column("z_id", width=120, anchor="e")
    app.recipe_tree.column("ax0_abs", width=120, anchor="e")
    app.recipe_tree.column("ax1_abs", width=120, anchor="e")
    app.recipe_tree.column("ax4_abs", width=120, anchor="e")
    app.recipe_tree.column("mode", width=90, anchor="center")

    app.recipe_tree.grid(row=0, column=0, sticky="nsew")

    ysb = ttk.Scrollbar(table_wrap, orient="vertical", command=app.recipe_tree.yview)
    ysb.grid(row=0, column=1, sticky="ns")
    app.recipe_tree.configure(yscroll=ysb.set)

    # 如后续列变多，可打开水平滚动
    xsb = ttk.Scrollbar(table_wrap, orient="horizontal", command=app.recipe_tree.xview)
    xsb.grid(row=2, column=0, sticky="ew")
    app.recipe_tree.configure(xscroll=xsb.set)

    try:
        app._refresh_teach_action_buttons()
    except Exception:
        pass

    app._refresh_recipe_table()
    try:
        app._refresh_center_positions()
    except Exception:
        pass

    # Refresh / live-update length measurement info (Lmax/status)
    def _len_tr(_a=None, _b=None, _c=None):
        try:
            if hasattr(app, "_refresh_length_info"):
                app._refresh_length_info()
        except Exception:
            pass

    for _v in (
        getattr(app, "len_enable_var", None),
        getattr(app, "len_z_low_approach_var", None),
        getattr(app, "len_low_search_dist_var", None),
        getattr(app, "len_high_search_dist_var", None),
        getattr(app, "len_search_vel_var", None),
        getattr(app, "len_search_timeout_var", None),
        getattr(app, "len_tol_var", None),
        getattr(app, "len_high_margin_var", None),
        getattr(app, "pipe_len_var", None),
    ):
        try:
            if _v is not None:
                _v.trace_add("write", _len_tr)
        except Exception:
            pass

    _len_tr()
