# ./ui/screens/axis_screen.py
from __future__ import annotations

"""轴参数与轴调试页（UI 构建）。

该模块只负责“构建界面与控件绑定”，不包含业务逻辑。
业务逻辑/事件处理函数由 App 提供（如 _do_movea/_do_reset/_jog_hold 等）。
"""

from typing import TYPE_CHECKING
import tkinter as tk
from tkinter import ttk

from config.addresses import AXIS_NAMES

if TYPE_CHECKING:  # pragma: no cover
    from app import App

def build_axis_screen(app: "App", parent: ttk.Frame) -> None:
    """轴参数与轴调试页面。"""
    left = ttk.Frame(parent)
    left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12), pady=8)

    ttk.Label(left, text="选择轴").pack(anchor="w")
    app.axis_combo = ttk.Combobox(
        left,
        values=[f"{i}: {n}" for i, n in enumerate(AXIS_NAMES)],
        state="readonly",
        width=26,
    )
    app.axis_combo.current(0)
    app.axis_combo.pack(anchor="w", pady=(2, 10))
    app.axis_combo.bind("<<ComboboxSelected>>", app._on_axis_selected)

    app.lbl_actpos = ttk.Label(left, text="Act_Pos(abs): --")
    app.lbl_actpos.pack(anchor="w")
    app.lbl_uipos = ttk.Label(left, text="UI_Pos(相对): --")
    app.lbl_uipos.pack(anchor="w", pady=(0, 6))

    zero_box = ttk.LabelFrame(left, text="界面零点坐标（不影响PLC）")
    zero_box.pack(fill=tk.X, pady=(4, 8))
    app.zero_abs_var = tk.StringVar(value="0.0")
    app.sign_var = tk.IntVar(value=+1)
    ttk.Label(zero_box, text="ZeroAbs").grid(
        row=0, column=0, padx=6, pady=4, sticky="e"
    )
    ttk.Entry(zero_box, width=12, textvariable=app.zero_abs_var).grid(
        row=0, column=1, padx=6, pady=4, sticky="w"
    )
    ttk.Button(zero_box, text="设当前为零", command=app._set_current_zero).grid(
        row=0, column=2, padx=6, pady=4
    )
    ttk.Radiobutton(
        zero_box,
        text="正向=+abs",
        variable=app.sign_var,
        value=+1,
        command=app._on_sign_change,
    ).grid(row=1, column=0, columnspan=2, padx=6, sticky="w")
    ttk.Radiobutton(
        zero_box,
        text="正向=-abs",
        variable=app.sign_var,
        value=-1,
        command=app._on_sign_change,
    ).grid(row=2, column=0, columnspan=2, padx=6, sticky="w")

    app.lbl_err = ttk.Label(left, text="ErrCode: --  Warn: --")
    app.lbl_err.pack(anchor="w")
    app.lbl_sts = ttk.Label(left, text="Sts: --")
    app.lbl_sts.pack(anchor="w")
    app.lbl_flags = ttk.Label(left, text="READY/EN/BSY/DONE/FLT/JOG/VEL/ILK: --")
    app.lbl_flags.pack(anchor="w", pady=(2, 0))
    app.lbl_stid = ttk.Label(left, text="St_ID: --  Seq/Ack: --/--")
    app.lbl_stid.pack(anchor="w")
    app.lbl_cmd = ttk.Label(left, text="Cmd: 0x----  Mode: 0x----")
    app.lbl_cmd.pack(anchor="w")

    ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=10)

    # Power enable
    app.power_var = tk.IntVar(value=0)
    ttk.Checkbutton(
        left,
        text="Enable (CMD_EN_REQ)",
        variable=app.power_var,
        command=app._on_power_toggle,
    ).pack(anchor="w")

    # Vel direction (MODE_DIR_REV)
    app.rev_var = tk.IntVar(value=0)
    ttk.Checkbutton(
        left,
        text="Vel Reverse (MODE_DIR_REV)",
        variable=app.rev_var,
        command=app._on_rev_toggle,
    ).pack(anchor="w", pady=(6, 0))

    ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=10)

    btn_row = ttk.Frame(left)
    btn_row.pack(anchor="w", pady=(0, 6))
    ttk.Button(btn_row, text="Reset", width=10, command=app._do_reset).pack(
        side=tk.LEFT, padx=(0, 6)
    )
    ttk.Button(btn_row, text="Stop", width=10, command=app._do_stop).pack(
        side=tk.LEFT, padx=(0, 6)
    )
    ttk.Button(btn_row, text="Halt", width=10, command=app._do_halt).pack(
        side=tk.LEFT
    )

    right = ttk.Frame(parent)
    right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=8)

    # Common params
    params = ttk.LabelFrame(right, text="通用参数（写入 Vel/Acc/Dec/Jerk）")
    params.pack(fill=tk.X, pady=(0, 10))
    app.ent_vel = app._labeled_entry(params, "Vel(UINT)", "100", 0)
    app.ent_acc = app._labeled_entry(params, "Acc(UINT)", "200", 1)
    app.ent_dec = app._labeled_entry(params, "Dec(UINT)", "200", 2)
    app.ent_jerk = app._labeled_entry(params, "Jerk(UINT)", "1000", 3)
    ttk.Button(
        params, text="Write Params", width=14, command=app._write_common_params
    ).grid(row=0, column=10, padx=10, pady=6)

    # MoveA
    card_move = ttk.LabelFrame(
        right, text="绝对定位 MoveA（写 Tgt_Pos，然后脉冲 CMD_MOVEA_REQ）"
    )
    card_move.pack(fill=tk.X, pady=(0, 10))
    app.ent_pos = app._labeled_entry(card_move, "Tgt_Pos(PLC abs,FP64)", "0.0", 0)
    ttk.Label(
        card_move, text="说明：此处输入值直接写PLC绝对坐标，不经过UI零点/方向换算。"
    ).grid(row=1, column=0, columnspan=12, padx=10, pady=(0, 6), sticky="w")
    ttk.Button(card_move, text="MoveA", width=14, command=app._do_movea).grid(
        row=0, column=10, padx=10, pady=6
    )

    # VelMove
    card_vel = ttk.LabelFrame(
        right,
        text="速度模式 VelMove（仅建议 AX3；电平 CMD_VELMOVE_REQ + MODE_DIR_REV）",
    )
    card_vel.pack(fill=tk.X, pady=(0, 10))
    ttk.Button(
        card_vel, text="Start VelMove", width=14, command=app._do_vel_start
    ).grid(row=0, column=10, padx=10, pady=6)
    ttk.Button(
        card_vel, text="Stop VelMove", width=14, command=app._do_vel_stop
    ).grid(row=0, column=11, padx=6, pady=6)

    # Jog
    card_jog = ttk.LabelFrame(
        right,
        text="点动 Jog（按住有效：CMD_JOG_F_REQ / CMD_JOG_B_REQ；要求 MODE_INCH=0）",
    )
    card_jog.pack(fill=tk.X, pady=(0, 10))

    btn_jneg = ttk.Button(card_jog, text="Jog -", width=14)
    btn_jpos = ttk.Button(card_jog, text="Jog +", width=14)
    btn_jneg.grid(row=0, column=10, padx=10, pady=6)
    btn_jpos.grid(row=0, column=11, padx=6, pady=6)

    btn_jneg.bind(
        "<ButtonPress-1>", lambda _e: app._jog_hold(direction="rev", on=True)
    )
    btn_jneg.bind(
        "<ButtonRelease-1>", lambda _e: app._jog_hold(direction="rev", on=False)
    )
    btn_jpos.bind(
        "<ButtonPress-1>", lambda _e: app._jog_hold(direction="fwd", on=True)
    )
    btn_jpos.bind(
        "<ButtonRelease-1>", lambda _e: app._jog_hold(direction="fwd", on=False)
    )

    # Inch
    card_inch = ttk.LabelFrame(
        right,
        text="寸动 Inch（写 Tgt_Pos2=Dis，然后临时置 MODE_INCH，脉冲 CMD_JOG_F/B 形成上升沿）",
    )
    card_inch.pack(fill=tk.X, pady=(0, 10))
    app.ent_step = app._labeled_entry(card_inch, "Tgt_Pos2/Dis(FP64)", "1.0", 0)
    ttk.Button(
        card_inch, text="Inch -", width=14, command=lambda: app._do_inch("rev")
    ).grid(row=0, column=10, padx=10, pady=6)
    ttk.Button(
        card_inch, text="Inch +", width=14, command=lambda: app._do_inch("fwd")
    ).grid(row=0, column=11, padx=6, pady=6)

    tips = ttk.LabelFrame(right, text="说明（v0.1 关键点）")
    tips.pack(fill=tk.BOTH, expand=True)
    msg = (
        "1) 本程序新增“界面零点坐标”：以伺服反馈 Act_Pos 捕获一个 ZeroAbs，显示相对坐标 UI_Pos；不写入PLC，不影响伺服参数。\n"
        "2) 配方中的截面位置采用 UI_Pos（相对坐标），实际运动时再换算为绝对坐标 abs 发送 MoveA。\n"
        "3) 自动测量 v0.1：只测外径 OD；测径仪返回值即 OD(mm)。按角度+OD 采样并做圆拟合。可用模拟测径仪或真实串口（pyserial）。\n"
        "4) 若 FP64 显示不对：改右上角“FP64顺序”并 Apply。"
    )
    ttk.Label(tips, text=msg, justify="left").pack(anchor="w", padx=10, pady=8)

