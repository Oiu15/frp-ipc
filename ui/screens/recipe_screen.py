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
    app.section_n_var = tk.StringVar(value=str(app.recipe.section_count))
    app.scan_axis_var = tk.IntVar(value=app.recipe.scan_axis)

    app.od_std_var = tk.StringVar(value=str(app.recipe.od_std_mm))
    app.id_std_var = tk.StringVar(value=str(app.recipe.id_std_mm))
    app.od_tol_var = tk.StringVar(value=str(app.recipe.od_tol_mm))

    app.points_per_rev_var = tk.StringVar(value=str(app.recipe.points_per_rev))
    app.min_cov_var = tk.StringVar(value=str(getattr(app.recipe, "min_bin_coverage", 0.95)))
    app.sample_timeout_var = tk.StringVar(value=str(getattr(app.recipe, "sample_timeout_s", 5.0)))
    app.max_revs_var = tk.StringVar(value=str(getattr(app.recipe, "max_revolutions", 2.0)))

    # ====== Page layout: 3 rows (top=recipe, mid=teach, bottom=table) ======
    parent.grid_rowconfigure(0, weight=0)  # recipe params
    parent.grid_rowconfigure(1, weight=0)  # teach
    parent.grid_rowconfigure(2, weight=1)  # results table expands
    parent.grid_columnconfigure(0, weight=1)

    # ---------------- Top: Recipe Params ----------------
    top = ttk.Frame(parent)
    top.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
    top.grid_columnconfigure(0, weight=1)

    # 让参数区内部更容易扩展：用 3 个分组框（语义分组）
    grp = ttk.Frame(top)
    grp.grid(row=0, column=0, sticky="ew")
    grp.grid_columnconfigure(0, weight=1)
    grp.grid_columnconfigure(1, weight=1)
    grp.grid_columnconfigure(2, weight=1)

    box_geom = ttk.LabelFrame(grp, text="工件 / 几何参数")
    box_plan = ttk.LabelFrame(grp, text="扫描 / 截面规划")
    box_meas = ttk.LabelFrame(grp, text="测量 / 判定参数")

    box_geom.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    box_plan.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
    box_meas.grid(row=0, column=2, sticky="nsew")

    for b in (box_geom, box_plan, box_meas):
        b.grid_columnconfigure(0, weight=0)
        b.grid_columnconfigure(1, weight=1)

    # --- 表驱动字段定义：后续新增参数，往这里加一行即可 ---
    # (label, var)
    GEOM_FIELDS: List[Tuple[str, tk.Variable]] = [
        ("配方名", app.recipe_name_var),
        ("管长(mm)", app.pipe_len_var),
        ("夹爪占用(mm)", app.clamp_var),
        ("头部留边(mm)", app.margin_h_var),
        ("尾部留边(mm)", app.margin_t_var),
    ]
    PLAN_FIELDS: List[Tuple[str, tk.Variable]] = [
        ("截面数量(N)", app.section_n_var),
        # 扫描轴是 Combobox，单独渲染
    ]
    MEAS_FIELDS: List[Tuple[str, tk.Variable]] = [
        ("OD标准(mm)", app.od_std_var),
        ("ID标准(mm)", app.id_std_var),
        ("OD公差(±mm)", app.od_tol_var),
        ("每圈采样点数", app.points_per_rev_var),
        ("采样覆盖率(0~1)", app.min_cov_var),
        ("单截面超时(s)", app.sample_timeout_var),
        ("最大采样圈数(转)", app.max_revs_var),
    ]

    # 渲染：几何参数
    r = 0
    for label, var in GEOM_FIELDS:
        app._kv_row(box_geom, label, var, r)
        r += 1

    # 渲染：截面规划
    r = 0
    for label, var in PLAN_FIELDS:
        app._kv_row(box_plan, label, var, r)
        r += 1

    ttk.Label(box_plan, text="扫描轴").grid(row=r, column=0, sticky="e", padx=6, pady=4)
    app.scan_axis_combo = ttk.Combobox(
        box_plan,
        state="readonly",
        width=18,
        values=[f"{i}: {n}" for i, n in enumerate(AXIS_NAMES)],
    )
    app.scan_axis_combo.current(int(app.scan_axis_var.get()))
    app.scan_axis_combo.grid(row=r, column=1, sticky="ew", padx=6, pady=4)
    app.scan_axis_combo.bind("<<ComboboxSelected>>", app._on_scan_axis_selected)
    r += 1

    # 渲染：测量/判定参数
    r = 0
    for label, var in MEAS_FIELDS:
        app._kv_row(box_meas, label, var, r)
        r += 1

    # --- Top buttons (统一放在参数区下方，避免散落) ---
    btns = ttk.Frame(top)
    btns.grid(row=1, column=0, sticky="ew", pady=(8, 0))
    btns.grid_columnconfigure(0, weight=1)

    btn_left = ttk.Frame(btns)
    btn_left.grid(row=0, column=0, sticky="w")

    ttk.Button(btn_left, text="计算截面位置", command=app._recipe_compute).pack(
        side=tk.LEFT, padx=(0, 6)
    )
    ttk.Button(btn_left, text="保存配方(JSON)", command=app._recipe_save).pack(
        side=tk.LEFT, padx=(0, 6)
    )
    ttk.Button(btn_left, text="加载配方(JSON)", command=app._recipe_load).pack(
        side=tk.LEFT
    )

    # 小提示放右侧，减少纵向占用
    hint = ttk.Label(
        btns,
        text="说明：截面位置使用 UI_Pos（相对坐标）。\n“设当前为零”后，再示教位置更直观。",
        justify="left",
    )
    hint.grid(row=0, column=1, sticky="e", padx=(12, 0))

    # ---------------- Mid: Teach ----------------
    mid = ttk.LabelFrame(parent, text="示教")
    mid.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
    mid.grid_columnconfigure(0, weight=1)
    mid.grid_columnconfigure(1, weight=1)
    mid.grid_columnconfigure(2, weight=1)

    # 左：动作按钮
    teach_actions = ttk.Frame(mid)
    teach_actions.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
    teach_actions.grid_columnconfigure(0, weight=1)

    ttk.Button(
        teach_actions, text="移动到选中截面", command=app._teach_move_to_selected
    ).grid(row=0, column=0, sticky="ew", pady=(0, 6))

    ttk.Button(
        teach_actions,
        text="将当前位保存为该截面",
        command=app._teach_save_current_to_selected,
    ).grid(row=1, column=0, sticky="ew", pady=(0, 6))

    ttk.Button(
        teach_actions, text="将扫描轴当前位置设为零", command=app._set_scan_axis_zero
    ).grid(row=2, column=0, sticky="ew")

    # 中：当前位置显示
    teach_status = ttk.Frame(mid)
    teach_status.grid(row=0, column=1, sticky="ew", padx=8, pady=8)
    teach_status.grid_columnconfigure(0, weight=1)

    app.teach_abs_var = tk.StringVar(value="abs: --")
    app.teach_ui_var = tk.StringVar(value="ui: --")

    ttk.Label(teach_status, text="当前位置").grid(row=0, column=0, sticky="w")
    ttk.Label(teach_status, textvariable=app.teach_abs_var).grid(
        row=1, column=0, sticky="w", pady=(4, 0)
    )
    ttk.Label(teach_status, textvariable=app.teach_ui_var).grid(
        row=2, column=0, sticky="w", pady=(2, 0)
    )

    # 右：Jog（按住运行）
    teach_jog = ttk.Frame(mid)
    teach_jog.grid(row=0, column=2, sticky="ew", padx=8, pady=8)
    teach_jog.grid_columnconfigure(0, weight=1)
    teach_jog.grid_columnconfigure(1, weight=1)

    ttk.Label(teach_jog, text="点动（按住运行）").grid(row=0, column=0, columnspan=2, sticky="w")

    btn_tjneg = ttk.Button(teach_jog, text="Jog -")
    btn_tjpos = ttk.Button(teach_jog, text="Jog +")
    btn_tjneg.grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=(6, 0))
    btn_tjpos.grid(row=1, column=1, sticky="ew", pady=(6, 0))

    btn_tjneg.bind("<ButtonPress-1>", lambda _e: app._teach_jog_hold("rev", True))
    btn_tjneg.bind("<ButtonRelease-1>", lambda _e: app._teach_jog_hold("rev", False))
    btn_tjpos.bind("<ButtonPress-1>", lambda _e: app._teach_jog_hold("fwd", True))
    btn_tjpos.bind("<ButtonRelease-1>", lambda _e: app._teach_jog_hold("fwd", False))

    # ---------------- Bottom: Result Table ----------------
    bottom = ttk.LabelFrame(parent, text="测量截面位置计算结果")
    bottom.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
    bottom.grid_rowconfigure(0, weight=1)
    bottom.grid_columnconfigure(0, weight=1)

    table_wrap = ttk.Frame(bottom)
    table_wrap.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
    table_wrap.grid_rowconfigure(0, weight=1)
    table_wrap.grid_columnconfigure(0, weight=1)

    cols = ("idx", "x_ui", "x_abs", "mode")
    app.recipe_tree = ttk.Treeview(table_wrap, columns=cols, show="headings", height=12)

    app.recipe_tree.heading("idx", text="截面")
    app.recipe_tree.heading("x_ui", text="位置(UI,mm)")
    app.recipe_tree.heading("x_abs", text="目标(abs,mm)")
    app.recipe_tree.heading("mode", text="来源")

    app.recipe_tree.column("idx", width=70, anchor="center")
    app.recipe_tree.column("x_ui", width=150, anchor="e")
    app.recipe_tree.column("x_abs", width=170, anchor="e")
    app.recipe_tree.column("mode", width=110, anchor="center")

    app.recipe_tree.grid(row=0, column=0, sticky="nsew")

    ysb = ttk.Scrollbar(table_wrap, orient="vertical", command=app.recipe_tree.yview)
    ysb.grid(row=0, column=1, sticky="ns")
    app.recipe_tree.configure(yscroll=ysb.set)

    # 如后续列变多，可打开水平滚动
    xsb = ttk.Scrollbar(table_wrap, orient="horizontal", command=app.recipe_tree.xview)
    xsb.grid(row=1, column=0, sticky="ew")
    app.recipe_tree.configure(xscroll=xsb.set)

    app._refresh_recipe_table()
