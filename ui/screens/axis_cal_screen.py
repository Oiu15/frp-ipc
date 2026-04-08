from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def _make_row(parent: ttk.Frame, row: int, label: str, var: tk.StringVar, status_var: tk.StringVar) -> None:
    ttk.Label(parent, text=label, width=18).grid(row=row, column=0, sticky='e', padx=6, pady=4)
    ttk.Entry(parent, textvariable=var, width=20).grid(row=row, column=1, sticky='w', padx=6, pady=4)
    ttk.Label(parent, textvariable=status_var, width=16).grid(row=row, column=2, sticky='w', padx=6, pady=4)


def build_axis_cal_screen(parent: ttk.Frame, *, presenter, controller, ui) -> None:
    root = ttk.Frame(parent)
    root.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

    ttk.Label(
        root,
        text=(
            'PLC HD ?????????/???'
            'Z ????????sign ?? -1?z_pos ? IPC ????'
        ),
        justify='left',
    ).pack(anchor='w', pady=(0, 10))

    grid = ttk.Frame(root)
    grid.pack(anchor='nw')

    rows = [
        ('sign', 'Sign (-1/+1)'),
        ('off_ax0', 'Off_ax0'),
        ('off_ax1', 'Off_ax1'),
        ('off_ax2', 'Off_ax2'),
        ('off_ax4', 'Off_ax4'),
        ('b14', 'B14'),
        ('b2', 'B2 (AX2->KeepoutCenter)'),
        ('keepout_w', 'Keepout_W (half width)'),
        ('z_pos', 'z_pos (IPC only)'),
    ]

    for idx, (key, label) in enumerate(rows):
        _make_row(grid, idx, label, presenter.axis_cal_vars[key], presenter.axis_cal_field_status_vars[key])

    btns = ttk.Frame(root)
    btns.pack(anchor='w', pady=(12, 0))
    ttk.Button(btns, text='Read', command=controller.axis_cal_read).pack(side=tk.LEFT, padx=6)
    ttk.Button(btns, text='Write', command=controller.axis_cal_write).pack(side=tk.LEFT, padx=6)

    cal_btns = ttk.Frame(root)
    cal_btns.pack(anchor='w', pady=(10, 0))
    ttk.Button(cal_btns, text='????offset', command=controller.axis_cal_capture_offsets).pack(side=tk.LEFT, padx=6)
    ttk.Button(cal_btns, text='??B14', command=controller.axis_cal_calibrate_b14).pack(side=tk.LEFT, padx=6)
    ttk.Button(cal_btns, text='?????', command=controller.axis_cal_calibrate_keepout).pack(side=tk.LEFT, padx=6)
    ttk.Button(cal_btns, text='??Z_Pos??', command=controller.axis_cal_set_zpos_zero).pack(side=tk.LEFT, padx=6)

    stat = ttk.LabelFrame(root, text='????(??)')
    stat.pack(fill=tk.X, expand=False, pady=(14, 0), anchor='w')
    for key in ('off_abs', 'act_abs', 'softlim_pos', 'softlim_neg', 'z_raw', 'keepout_raw', 'keepout_disp', 'z_disp'):
        ttk.Label(stat, textvariable=presenter.axis_cal_status_vars[key], justify='left').pack(anchor='w', padx=10, pady=(6 if key == 'off_abs' else 2, 2 if key != 'z_disp' else 6))

