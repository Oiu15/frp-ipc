from __future__ import annotations

"""Key-test screen UI only."""

import tkinter as tk
from tkinter import ttk

from config.addresses import KEYTEST_X_BASE_COIL, KEYTEST_Y_BASE_COIL, KEYTEST_X_POINTS, KEYTEST_Y_POINTS


def _mk_group(parent: ttk.Frame, title: str) -> ttk.Labelframe:
    frame = ttk.Labelframe(parent, text=title)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
    return frame


def _pt_to_idx(pt: int) -> int:
    pt = int(pt)
    return pt if pt < 8 else (pt - 2)


def build_key_test_screen(parent: ttk.Frame, *, presenter, controller, ui) -> None:
    top = ttk.Frame(parent)
    top.pack(fill=tk.X, pady=(8, 6))

    ttk.Label(
        top,
        text=(
            '???\n'
            '1) ?????? PLC ? X/Y ??Modbus coils????\n'
            '2) X ???????????Y ?????? 0/1?\n'
            '3) ?????????????????????'
        ),
        justify='left',
    ).pack(side=tk.LEFT, padx=10)

    body = ttk.Frame(parent)
    body.pack(fill=tk.BOTH, expand=True)

    gx = _mk_group(body, 'X ???????')
    ttk.Label(
        gx,
        text=(
            f'?????X0..X7 -> coil {KEYTEST_X_BASE_COIL}..{KEYTEST_X_BASE_COIL + 7}?'
            f'X10..X17 -> coil {KEYTEST_X_BASE_COIL + 8}..{KEYTEST_X_BASE_COIL + 15}???? X8/X9?'
        ),
    ).pack(anchor='w', padx=10, pady=(6, 10))

    grid_x = ttk.Frame(gx)
    grid_x.pack(fill=tk.X, padx=10, pady=(0, 10))
    for col in range(8):
        grid_x.columnconfigure(col, weight=1)
    for i, point in enumerate(KEYTEST_X_POINTS):
        row = i // 8
        col = i % 8
        widget = ttk.Checkbutton(grid_x, text=f'X{point}', variable=presenter.keytest_x_vars[i])
        widget.state(['disabled'])
        widget.grid(row=row, column=col, sticky='w', padx=(0, 12), pady=4)

    gy = _mk_group(body, 'Y ??????')
    ttk.Label(
        gy,
        text=(
            f'?????Y0..Y7 -> coil {KEYTEST_Y_BASE_COIL}..{KEYTEST_Y_BASE_COIL + 7}?'
            f'Y10..Y15 -> coil {KEYTEST_Y_BASE_COIL + 8}..{KEYTEST_Y_BASE_COIL + 13}???? Y8/Y9?'
        ),
    ).pack(anchor='w', padx=10, pady=(6, 10))

    cards = ttk.Frame(gy)
    cards.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    for col in range(5):
        cards.columnconfigure(col, weight=1)

    for i, point in enumerate(KEYTEST_Y_POINTS):
        idx = _pt_to_idx(int(point))
        addr = int(KEYTEST_Y_BASE_COIL) + idx
        row = i // 5
        col = i % 5

        frame = ttk.Labelframe(cards, text=f'Y{int(point)}  (coil {addr})')
        frame.grid(row=row, column=col, sticky='nsew', padx=((0, 8) if col < 4 else (0, 0)), pady=6)
        frame.columnconfigure(0, weight=1)

        state_cb = ttk.Checkbutton(frame, text='??(??)', variable=presenter.keytest_y_vars[i])
        state_cb.state(['disabled'])
        state_cb.grid(row=0, column=0, sticky='w', padx=8, pady=(6, 2))

        btns = ttk.Frame(frame)
        btns.grid(row=1, column=0, sticky='w', padx=8, pady=(0, 2))
        ttk.Button(btns, text='? 1', width=8, command=lambda p=int(point): controller.write_keytest_y(p, 1)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btns, text='? 0', width=8, command=lambda p=int(point): controller.write_keytest_y(p, 0)).pack(side=tk.LEFT)

        ttk.Label(frame, textvariable=presenter.keytest_y_lastcmd_vars[i]).grid(row=2, column=0, sticky='w', padx=8, pady=(0, 6))
