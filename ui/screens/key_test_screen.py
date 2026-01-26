# ./ui/screens/key_test_screen.py
from __future__ import annotations

"""按键/点位测试页（UI 构建）。

直接读写 PLC 的 X/Y 点（Modbus coils），避免 holding register 写入冲突。

重要：X/Y 点使用“八进制标签”，线圈地址空间中没有 8/9。
因此：
  X0..X7   -> BASE+0..7
  X10..X17 -> BASE+8..15
  Y0..Y7   -> BASE+0..7
  Y10..Y15 -> BASE+8..13

展示规则：
- 不显示 X8/X9
- 不显示 Y8/Y9
"""

from typing import TYPE_CHECKING
import tkinter as tk
from tkinter import ttk

from config.addresses import (
    KEYTEST_X_BASE_COIL,
    KEYTEST_Y_BASE_COIL,
    KEYTEST_X_POINTS,
    KEYTEST_Y_POINTS,
)

if TYPE_CHECKING:  # pragma: no cover
    from app import App


def _mk_group(parent: ttk.Frame, title: str) -> ttk.Labelframe:
    lf = ttk.Labelframe(parent, text=title)
    lf.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
    return lf


def _pt_to_idx(pt: int) -> int:
    """Convert X/Y label (octal-like) to coil index offset.

    Example: 0..7 -> 0..7; 10 -> 8; 17 -> 15
    """
    pt = int(pt)
    return pt if pt < 8 else (pt - 2)  # skip 8/9


def build_key_test_screen(app: "App", parent: ttk.Frame) -> None:
    top = ttk.Frame(parent)
    top.pack(fill=tk.X, pady=(8, 6))

    ttk.Label(
        top,
        text=(
            "说明：\n"
            "1) 本页用于测试 PLC 的 X/Y 点（Modbus coils）读写；\n"
            "2) X 点仅显示（物理输入），Y 点可单次写入 0/1；\n"
            "3) 写入是否成功以‘读回状态’为准（不做持续写入，避免与 PLC 内部逻辑冲突）。"
        ),
        justify="left",
    ).pack(side=tk.LEFT, padx=10)

    body = ttk.Frame(parent)
    body.pack(fill=tk.BOTH, expand=True)

    # -----------------
    # X points (read only)
    # -----------------
    gx = _mk_group(body, "X 点状态（只读）")
    ttk.Label(
        gx,
        text=(
            f"地址范围：X0..X7 -> coil {KEYTEST_X_BASE_COIL}..{KEYTEST_X_BASE_COIL+7}；"
            f"X10..X17 -> coil {KEYTEST_X_BASE_COIL+8}..{KEYTEST_X_BASE_COIL+15}（不显示 X8/X9）"
        ),
    ).pack(
        anchor="w", padx=10, pady=(6, 10)
    )

    grid_x = ttk.Frame(gx)
    grid_x.pack(fill=tk.X, padx=10, pady=(0, 10))

    # 2 rows x 8 columns
    cols = 8
    for ci in range(cols):
        grid_x.columnconfigure(ci, weight=1)

    for i, p in enumerate(KEYTEST_X_POINTS):
        r = i // cols
        c = i % cols
        try:
            v = app.keytest_x_vars[i]
        except Exception:
            v = tk.IntVar(value=0)
        cb = ttk.Checkbutton(
            grid_x,
            text=f"X{p}",
            variable=v,
        )
        cb.state(["disabled"])  # read only
        cb.grid(row=r, column=c, sticky="w", padx=(0, 12), pady=4)

    # -----------------
    # Y points (read + one-shot write)
    # -----------------
    gy = _mk_group(body, "Y 点状态与写入")
    ttk.Label(
        gy,
        text=(
            f"地址范围：Y0..Y7 -> coil {KEYTEST_Y_BASE_COIL}..{KEYTEST_Y_BASE_COIL+7}；"
            f"Y10..Y15 -> coil {KEYTEST_Y_BASE_COIL+8}..{KEYTEST_Y_BASE_COIL+13}（不显示 Y8/Y9）"
        ),
    ).pack(
        anchor="w", padx=10, pady=(6, 10)
    )

    # 使用“卡片网格”布局，避免表格过长导致按钮落到界面外。
    cards = ttk.Frame(gy)
    cards.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    # 用户现场显示空间有限：采用 5 列卡片，减少纵向高度占用。
    ncol = 5
    for c in range(ncol):
        cards.columnconfigure(c, weight=1)

    for i, p in enumerate(KEYTEST_Y_POINTS):
        pt = int(p)
        idx = _pt_to_idx(pt)
        addr = int(KEYTEST_Y_BASE_COIL) + int(idx)

        r = i // ncol
        c = i % ncol

        lf = ttk.Labelframe(cards, text=f"Y{pt}  (coil {addr})")
        # 每列之间留少量间距；最后一列不额外加右侧空隙
        px = (0, 8) if c < (ncol - 1) else (0, 0)
        lf.grid(row=r, column=c, sticky="nsew", padx=px, pady=6)

        # 让卡片内容在缩放时更自然
        lf.columnconfigure(0, weight=1)

        # status
        try:
            v = app.keytest_y_vars[i]
        except Exception:
            v = tk.IntVar(value=0)
        cb = ttk.Checkbutton(lf, text="状态(读回)", variable=v)
        cb.state(["disabled"])  # read only
        cb.grid(row=0, column=0, sticky="w", padx=8, pady=(6, 2))

        def _write1(_p=int(pt)):
            app._keytest_write_y(_p, 1)

        def _write0(_p=int(pt)):
            app._keytest_write_y(_p, 0)

        btns = ttk.Frame(lf)
        btns.grid(row=1, column=0, sticky="w", padx=8, pady=(0, 2))
        ttk.Button(btns, text="写 1", width=8, command=_write1).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btns, text="写 0", width=8, command=_write0).pack(side=tk.LEFT)

        try:
            sv = app.keytest_y_lastcmd_vars[i]
        except Exception:
            sv = tk.StringVar(value="--")
        ttk.Label(lf, textvariable=sv).grid(row=2, column=0, sticky="w", padx=8, pady=(0, 6))
