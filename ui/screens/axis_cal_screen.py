# ./ui/screens/axis_cal_screen.py
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def _make_row(parent: ttk.Frame, row: int, label: str, var: tk.StringVar) -> None:
    ttk.Label(parent, text=label, width=18).grid(
        row=row, column=0, sticky="e", padx=6, pady=4
    )
    ttk.Entry(parent, textvariable=var, width=20).grid(
        row=row, column=1, sticky="w", padx=6, pady=4
    )


def build_axis_cal_screen(app, parent: ttk.Frame) -> None:
    """Build AxisCal read/write UI."""

    root = ttk.Frame(parent)
    root.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

    ttk.Label(
        root,
        text=(
            "PLC HD 轴位标定参数（仅读/写）。"
            "Z 正方向: 向下为正；Sign 默认 -1；z_pos 仅 IPC 临时。"
        ),
        justify="left",
    ).pack(anchor="w", pady=(0, 10))

    grid = ttk.Frame(root)
    grid.pack(anchor="nw")

    v = getattr(app, "axis_cal_vars", {})

    rows = [
        ("sign", "Sign (-1/+1)"),
        ("off_ax0", "Off_ax0"),
        ("off_ax1", "Off_ax1"),
        ("off_ax2", "Off_ax2"),
        ("off_ax4", "Off_ax4"),
        ("b14", "B14"),
        ("handoff_z", "Handoff_z"),
        ("z_pos", "z_pos (IPC only)"),
    ]

    for i, (key, label) in enumerate(rows):
        if key not in v:
            v[key] = tk.StringVar(value="")
        _make_row(grid, i, label, v[key])

    btns = ttk.Frame(root)
    btns.pack(anchor="w", pady=(12, 0))

    ttk.Button(btns, text="Read", command=getattr(app, "axis_cal_read", lambda: None)).pack(
        side=tk.LEFT, padx=6
    )
    ttk.Button(btns, text="Write", command=getattr(app, "axis_cal_write", lambda: None)).pack(
        side=tk.LEFT, padx=6
    )

    # Keep a handle for future expansion
    app._axis_cal_screen_root = root
