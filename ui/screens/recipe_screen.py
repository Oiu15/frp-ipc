# ./ui/screens/recipe_screen.py
from __future__ import annotations

"""配方/示教页（UI 构建）。"""

from typing import TYPE_CHECKING
import tkinter as tk
from tkinter import ttk

from config.addresses import AXIS_NAMES, AXIS_COUNT

if TYPE_CHECKING:  # pragma: no cover
    from app import App

def build_recipe_screen(app: "App", parent: ttk.Frame) -> None:
    """测量配方与示教页面。"""
    left = ttk.Frame(parent)
    left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12), pady=8)

    box = ttk.LabelFrame(left, text="配方参数")
    box.pack(fill=tk.X)

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
    app.min_cov_var = tk.StringVar(value=str(getattr(app.recipe, 'min_bin_coverage', 0.95)))
    app.sample_timeout_var = tk.StringVar(value=str(getattr(app.recipe, 'sample_timeout_s', 5.0)))
    app.max_revs_var = tk.StringVar(value=str(getattr(app.recipe, 'max_revolutions', 2.0)))

    r = 0
    app._kv_row(box, "配方名", app.recipe_name_var, r)
    r += 1
    app._kv_row(box, "管长(mm)", app.pipe_len_var, r)
    r += 1
    app._kv_row(box, "夹爪占用(mm)", app.clamp_var, r)
    r += 1
    app._kv_row(box, "头部留边(mm)", app.margin_h_var, r)
    r += 1
    app._kv_row(box, "尾部留边(mm)", app.margin_t_var, r)
    r += 1
    app._kv_row(box, "截面数量(N)", app.section_n_var, r)
    r += 1

    ttk.Label(box, text="扫描轴").grid(row=r, column=0, sticky="e", padx=6, pady=4)
    app.scan_axis_combo = ttk.Combobox(
        box,
        state="readonly",
        width=18,
        values=[f"{i}: {n}" for i, n in enumerate(AXIS_NAMES)],
    )
    app.scan_axis_combo.current(int(app.scan_axis_var.get()))
    app.scan_axis_combo.grid(row=r, column=1, sticky="w", padx=6, pady=4)
    app.scan_axis_combo.bind("<<ComboboxSelected>>", app._on_scan_axis_selected)
    r += 1

    ttk.Separator(box, orient="horizontal").grid(
        row=r, column=0, columnspan=2, sticky="ew", padx=6, pady=6
    )
    r += 1

    app._kv_row(box, "OD标准(mm)", app.od_std_var, r)
    r += 1
    app._kv_row(box, "ID标准(mm)", app.id_std_var, r)
    r += 1
    app._kv_row(box, "OD公差(±mm)", app.od_tol_var, r)
    r += 1
    app._kv_row(box, "每圈采样点数", app.points_per_rev_var, r)
    r += 1
    app._kv_row(box, "采样覆盖率(0~1)", app.min_cov_var, r)
    r += 1
    app._kv_row(box, "单截面超时(s)", app.sample_timeout_var, r)
    r += 1
    app._kv_row(box, "最大采样圈数(转)", app.max_revs_var, r)
    r += 1

    btns = ttk.Frame(left)
    btns.pack(fill=tk.X, pady=(8, 0))
    ttk.Button(btns, text="计算截面位置", command=app._recipe_compute).pack(
        side=tk.LEFT, padx=(0, 6)
    )
    ttk.Button(btns, text="保存配方(JSON)", command=app._recipe_save).pack(
        side=tk.LEFT, padx=(0, 6)
    )
    ttk.Button(btns, text="加载配方(JSON)", command=app._recipe_load).pack(
        side=tk.LEFT
    )

    teach = ttk.LabelFrame(left, text="示教")
    teach.pack(fill=tk.X, pady=(12, 0))
    ttk.Button(
        teach, text="移动到选中截面", command=app._teach_move_to_selected
    ).pack(fill=tk.X, padx=6, pady=(6, 3))
    # --- teach: current position display ---
    app.teach_abs_var = tk.StringVar(value="abs: --")
    app.teach_ui_var = tk.StringVar(value="ui: --")
    ttk.Label(teach, textvariable=app.teach_abs_var).pack(
        fill=tk.X, padx=6, pady=(6, 0)
    )
    ttk.Label(teach, textvariable=app.teach_ui_var).pack(
        fill=tk.X, padx=6, pady=(0, 6)
    )
    # --- teach: jog buttons (hold-to-run) ---
    jrow = ttk.Frame(teach)
    jrow.pack(fill=tk.X, padx=6, pady=(0, 6))

    btn_tjneg = ttk.Button(jrow, text="Jog -", width=10)
    btn_tjpos = ttk.Button(jrow, text="Jog +", width=10)
    btn_tjneg.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))
    btn_tjpos.pack(side=tk.LEFT, expand=True, fill=tk.X)

    btn_tjneg.bind("<ButtonPress-1>", lambda _e: app._teach_jog_hold("rev", True))
    btn_tjneg.bind(
        "<ButtonRelease-1>", lambda _e: app._teach_jog_hold("rev", False)
    )
    btn_tjpos.bind("<ButtonPress-1>", lambda _e: app._teach_jog_hold("fwd", True))
    btn_tjpos.bind(
        "<ButtonRelease-1>", lambda _e: app._teach_jog_hold("fwd", False)
    )
    # --- ------------------------------- ---
    ttk.Button(
        teach,
        text="将当前位保存为该截面",
        command=app._teach_save_current_to_selected,
    ).pack(fill=tk.X, padx=6, pady=(3, 6))
    ttk.Button(
        teach, text="将扫描轴当前位置设为零", command=app._set_scan_axis_zero
    ).pack(fill=tk.X, padx=6, pady=(6, 3))

    hint = ttk.Label(
        left,
        text="说明：截面位置使用 UI_Pos（相对坐标）。\n“设当前为零”后，再示教位置更直观。",
        justify="left",
    )
    hint.pack(anchor="w", pady=(10, 0))

    # Right: table
    right = ttk.Frame(parent)
    right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=8)

    cols = ("idx", "x_ui", "x_abs", "mode")
    app.recipe_tree = ttk.Treeview(right, columns=cols, show="headings", height=18)
    app.recipe_tree.heading("idx", text="截面")
    app.recipe_tree.heading("x_ui", text="位置(UI,mm)")
    app.recipe_tree.heading("x_abs", text="目标(abs,mm)")
    app.recipe_tree.heading("mode", text="来源")
    app.recipe_tree.column("idx", width=60, anchor="center")
    app.recipe_tree.column("x_ui", width=140, anchor="e")
    app.recipe_tree.column("x_abs", width=160, anchor="e")
    app.recipe_tree.column("mode", width=90, anchor="center")

    app.recipe_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    ysb = ttk.Scrollbar(right, orient="vertical", command=app.recipe_tree.yview)
    app.recipe_tree.configure(yscroll=ysb.set)
    ysb.pack(side=tk.RIGHT, fill=tk.Y)

    app._refresh_recipe_table()

