# ./ui/screens/axis_screen.py
from __future__ import annotations

"""轴参数与轴调试页（UI 构建）。

本文件只负责构建 UI，不包含业务逻辑。

新版需求：
- 五个轴分成 5 个 panel（Notebook Tab），不再使用 axis_combo。
- 修复参数区出现两个 Vel_MoveR 的问题：应为 Vel_MoveA / Vel_MoveR。

与 App 的最小耦合方式：
- App 仍保留原有的业务方法（如 _write_common_params/_do_movea/_do_mover/_do_vel_start/_do_vel_stop/_jog_hold 等）。
- AxisScreen 在 Tab 切换时，将 app.axis_idx 设置为当前轴，并把 app.ent_* / app.lbl_* / app.power_var 等指针切换为该轴 panel 内的控件。
  这样 App 侧无需为“每轴多套控件”做大改动。

注意：App 若仍引用 axis_combo，将不再可用；你已确认不再需要 axis_combo。
"""

from typing import TYPE_CHECKING, Dict, Any
import tkinter as tk
from tkinter import ttk

from config.addresses import AXIS_NAMES

if TYPE_CHECKING:
    from app import App


def build_axis_screen(app: "App", parent: tk.Widget) -> ttk.Frame:
    root = ttk.Frame(parent)
    root.pack(fill=tk.BOTH, expand=True)

    nb = ttk.Notebook(root)
    nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # 每轴控件映射：axis -> {attr_name: widget}
    axis_widgets: Dict[int, Dict[str, Any]] = {}
    axis_power_vars: Dict[int, tk.IntVar] = {}

    def _activate_axis(axis: int):
        axis = max(0, min(len(AXIS_NAMES) - 1, int(axis)))
        try:
            app.axis_idx.set(axis)
        except Exception:
            pass

        w = axis_widgets.get(axis, {})
        for k, v in w.items():
            setattr(app, k, v)

        # power_var 也需要切到当前轴
        if axis in axis_power_vars:
            app.power_var = axis_power_vars[axis]

        # 刷新当前 panel 显示
        if hasattr(app, "_refresh_axis_panel"):
            try:
                app._refresh_axis_panel()
            except Exception:
                pass

    def _wrap(axis: int, fn_name: str):
        """返回一个回调：先激活轴，再调用 app.fn_name()"""

        def _cb(*_a, **_k):
            _activate_axis(axis)
            fn = getattr(app, fn_name, None)
            if callable(fn):
                return fn()
            return None

        return _cb

    def _wrap_jog(axis: int, direction: str, on: bool):
        def _cb(_evt=None):
            _activate_axis(axis)
            fn = getattr(app, "_jog_hold", None)
            if callable(fn):
                return fn(direction, on)
            return None

        return _cb

    # -------------------------
    # Build per-axis panels
    # -------------------------
    for axis in range(len(AXIS_NAMES)):
        tab = ttk.Frame(nb)
        nb.add(tab, text=f"{AXIS_NAMES[axis]}")

        # 顶部：使能
        top = ttk.Frame(tab)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 6))

        pvar = tk.IntVar(value=0)
        axis_power_vars[axis] = pvar
        cb = ttk.Checkbutton(
            top,
            text="使能请求(EN_REQ)",
            variable=pvar,
            command=_wrap(axis, "_on_power_toggle"),
        )
        cb.pack(side=tk.LEFT, padx=4)

        # 主体左右分栏：左状态，右手动
        body = ttk.Frame(tab)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=6)

        left = ttk.Labelframe(body, text="反馈/状态 (PLC->IPC)")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=6)

        right = ttk.Labelframe(body, text="手动调试 (IPC->PLC)")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=6)

        # -------- left: status labels --------
        lbl_actpos = ttk.Label(left, text="Act_Pos(abs): -")
        lbl_actpos.pack(anchor="w", padx=10, pady=(10, 4))

        lbl_uipos = ttk.Label(left, text="UI_Pos: -")
        lbl_uipos.pack(anchor="w", padx=10, pady=4)

        lbl_sts = ttk.Label(left, text="Sts(raw_axis_state): -")
        lbl_sts.pack(anchor="w", padx=10, pady=4)

        lbl_err = ttk.Label(left, text="ErrCode(raw_axis_err): -    Warn(non-axis): -")
        lbl_err.pack(anchor="w", padx=10, pady=4)

        lbl_stid = ttk.Label(left, text="St_ID: -    Seq/Ack: -/-")
        lbl_stid.pack(anchor="w", padx=10, pady=4)

        lbl_cmd = ttk.Label(left, text="Cmd(request): -")
        lbl_cmd.pack(anchor="w", padx=10, pady=4)

        # -------- right: common params --------
        params = ttk.Labelframe(right, text="运动参数 (写入 Axis_Ctrl)")
        params.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 8))

        def _mk_entry(row: int, col: int, label: str, default: str):
            ttk.Label(params, text=label).grid(
                row=row, column=col, sticky="e", padx=6, pady=4
            )
            e = ttk.Entry(params, width=10)
            e.grid(row=row, column=col + 1, sticky="w", padx=6, pady=4)
            e.insert(0, default)
            return e

        # Row0: Vel_MoveA / Vel_MoveR
        ent_vel_movea = _mk_entry(0, 0, "Vel_MoveA（绝对定位速度）", "100")
        ent_vel_mover = _mk_entry(0, 2, "Vel_MoveR（相对运动速度）", "100")

        # Row1: Vel_Jog / Vel_VelMove
        ent_vel_jog = _mk_entry(1, 0, "Vel_Jog（点动速度）", "80")
        ent_vel_velmove = _mk_entry(1, 2, "Vel_VelMove（速度模式速度）", "200")
        # Row2: Acc / Dec
        ent_acc = _mk_entry(2, 0, "Acc（加速度）", "200")
        ent_dec = _mk_entry(2, 2, "Dec（减速度）", "200")

        # Row3: Jerk / (button)
        ent_jerk = _mk_entry(3, 0, "Jerk（加加速度）", "500")
        btn_write = ttk.Button(
            params, text="写入参数", command=_wrap(axis, "_write_common_params")
        )
        btn_write.grid(row=3, column=2, columnspan=2, sticky="e", padx=6, pady=6)

        # -------- right: basic control buttons --------
        ctrl = ttk.Frame(right)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)

        ttk.Button(ctrl, text="RESET", command=_wrap(axis, "_do_reset")).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(ctrl, text="STOP", command=_wrap(axis, "_do_stop")).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(ctrl, text="HALT", command=_wrap(axis, "_do_halt")).pack(
            side=tk.LEFT, padx=4
        )

        # -------- MoveA --------
        movea = ttk.Labelframe(right, text="MoveA (绝对定位)")
        movea.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Label(movea, text="目标位置(Abs)").grid(
            row=0, column=0, sticky="e", padx=6, pady=6
        )
        ent_pos = ttk.Entry(movea, width=16)
        ent_pos.grid(row=0, column=1, sticky="w", padx=6, pady=6)
        ent_pos.insert(0, "0")

        ttk.Button(movea, text="执行 MoveA", command=_wrap(axis, "_do_movea")).grid(
            row=0, column=2, padx=6, pady=6
        )

        # -------- MoveR --------
        mover = ttk.Labelframe(right, text="MoveR (相对运动)")
        mover.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Label(mover, text="相对位移").grid(
            row=0, column=0, sticky="e", padx=6, pady=6
        )
        ent_pos_r = ttk.Entry(mover, width=16)
        ent_pos_r.grid(row=0, column=1, sticky="w", padx=6, pady=6)
        ent_pos_r.insert(0, "10")

        ttk.Label(mover, text="方向(Dir_MoveR)").grid(
            row=1, column=0, sticky="e", padx=6, pady=6
        )
        cmb_dir_mover = ttk.Combobox(
            mover,
            width=14,
            state="readonly",
            values=[
                "0: 无方向",
                "1: 正向",
                "2: 负向",
                "3: 最短路径",
                "4: 当前方向(模轴)",
            ],
        )
        cmb_dir_mover.grid(row=1, column=1, sticky="w", padx=6, pady=6)
        cmb_dir_mover.set("0: 无方向")

        ttk.Button(mover, text="执行 MoveR", command=_wrap(axis, "_do_mover")).grid(
            row=0, column=2, rowspan=2, padx=6, pady=6
        )

        # -------- VelMove --------
        vel = ttk.Labelframe(right, text="速度模式 (VelMove)")
        vel.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Button(
            vel, text="Start VelMove", command=_wrap(axis, "_do_vel_start")
        ).pack(side=tk.LEFT, padx=6, pady=6)
        ttk.Button(vel, text="Stop VelMove", command=_wrap(axis, "_do_vel_stop")).pack(
            side=tk.LEFT, padx=6, pady=6
        )

        # -------- Jog --------
        jog = ttk.Labelframe(right, text="点动 (Jog) — 按住有效")
        jog.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(8, 12))

        btn_f = ttk.Button(jog, text="JOG +")
        btn_b = ttk.Button(jog, text="JOG -")
        btn_f.pack(side=tk.LEFT, padx=6, pady=8)
        btn_b.pack(side=tk.LEFT, padx=6, pady=8)

        btn_f.bind("<ButtonPress-1>", _wrap_jog(axis, "fwd", True))
        btn_f.bind("<ButtonRelease-1>", _wrap_jog(axis, "fwd", False))
        btn_b.bind("<ButtonPress-1>", _wrap_jog(axis, "rev", True))
        btn_b.bind("<ButtonRelease-1>", _wrap_jog(axis, "rev", False))

        # 保存控件映射，用于 Tab 切换时挂载到 app
        axis_widgets[axis] = {
            # status labels
            "lbl_actpos": lbl_actpos,
            "lbl_uipos": lbl_uipos,
            "lbl_sts": lbl_sts,
            "lbl_err": lbl_err,
            "lbl_stid": lbl_stid,
            "lbl_cmd": lbl_cmd,
            # entries / controls
            "ent_vel_movea": ent_vel_movea,
            "ent_vel_mover": ent_vel_mover,
            "ent_vel_jog": ent_vel_jog,
            "ent_vel_velmove": ent_vel_velmove,
            "ent_acc": ent_acc,
            "ent_dec": ent_dec,
            "ent_jerk": ent_jerk,
            "ent_pos": ent_pos,
            "ent_pos_r": ent_pos_r,
            "ent_pos2": ent_pos_r,  # compatibility alias
            "cmb_dir_mover": cmb_dir_mover,
        }

    # Tab change binding
    def _on_tab_changed(_evt=None):
        try:
            idx = nb.index("current")
        except Exception:
            idx = 0
        _activate_axis(idx)

    nb.bind("<<NotebookTabChanged>>", _on_tab_changed)

    # 默认激活 0 轴
    nb.select(0)
    _activate_axis(0)

    # 暴露给 app（可选）
    app.axis_notebook = nb
    app._axis_widgets = axis_widgets

    return root
