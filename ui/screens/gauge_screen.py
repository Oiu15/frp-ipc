from __future__ import annotations

"""Gauge/PLC/calibration screen UI only."""

import tkinter as tk
from tkinter import ttk

from config.addresses import DEFAULT_GAUGE_PORT


def build_gauge_screen(parent: ttk.Frame, *, presenter, controller, ui) -> None:
    presenter.ensure_vars(parent)

    pbox = ttk.LabelFrame(parent, text='PLC ???Modbus TCP?')
    pbox.pack(fill=tk.X, pady=(4, 8))
    ttk.Label(pbox, text='IP').grid(row=0, column=0, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(pbox, width=14, textvariable=presenter.ip_var).grid(row=0, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(pbox, text='Port').grid(row=0, column=2, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(pbox, width=6, textvariable=presenter.port_var).grid(row=0, column=3, padx=6, pady=6, sticky='w')
    ttk.Button(pbox, text='??/??', command=controller.apply_plc_connection).grid(row=0, column=4, padx=8, pady=6)
    ttk.Label(pbox, textvariable=presenter.plc_status_var).grid(row=0, column=5, padx=10, pady=6, sticky='w')

    nb = ttk.Notebook(parent)
    nb.pack(fill=tk.X, pady=(4, 8))
    tab_od = ttk.Frame(nb)
    tab_id = ttk.Frame(nb)
    nb.add(tab_od, text='?????/???')
    nb.add(tab_id, text='?????/???')

    gbox = ttk.LabelFrame(tab_od, text='?????? OD, ???')
    gbox.pack(fill=tk.X, pady=(4, 8))
    ttk.Checkbutton(gbox, text='?????', variable=presenter.sim_gauge_var, command=controller.toggle_sim_gauge).grid(row=0, column=0, padx=10, pady=6, sticky='w')
    ttk.Label(gbox, text='??').grid(row=0, column=1, padx=(10, 2), pady=6, sticky='e')
    port_combo = presenter.remember_widget('port_combo', ttk.Combobox(gbox, width=12, state='readonly', values=presenter.list_serial_ports()))
    port_combo.grid(row=0, column=2, padx=6, pady=6, sticky='w')
    port_combo.set(DEFAULT_GAUGE_PORT)
    ttk.Button(gbox, text='??', command=controller.refresh_gauge_ports).grid(row=0, column=3, padx=6, pady=6)
    ttk.Label(gbox, text='???').grid(row=0, column=4, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(gbox, width=8, textvariable=presenter.baud_var).grid(row=0, column=5, padx=6, pady=6, sticky='w')
    ttk.Label(gbox, text='????').grid(row=0, column=6, padx=(10, 2), pady=6, sticky='e')
    cmd_presets = ['M0,1', 'M1,1', 'M0,0', 'M1,0', 'M2,1', 'M2,0']
    req_cmd_combo = ttk.Combobox(gbox, width=10, textvariable=presenter.req_cmd_var, values=cmd_presets, state='normal')
    req_cmd_combo.grid(row=0, column=7, padx=6, pady=6, sticky='w')

    def _on_req_cmd_changed(*_args):
        try:
            presenter.handle_request_command_changed(presenter.req_cmd_var.get())
        except Exception:
            pass

    try:
        presenter.req_cmd_var.trace_add('write', _on_req_cmd_changed)
    except Exception:
        try:
            presenter.req_cmd_var.trace('w', _on_req_cmd_changed)  # type: ignore[arg-type]
        except Exception:
            pass

    ttk.Button(gbox, text='??', command=controller.connect_gauge).grid(row=0, column=8, padx=6, pady=6)
    ttk.Button(gbox, text='??', command=controller.disconnect_gauge).grid(row=0, column=9, padx=6, pady=6)
    ttk.Button(gbox, text='????', command=controller.request_gauge_once).grid(row=0, column=10, padx=6, pady=6)
    ttk.Label(gbox, textvariable=presenter.gauge_conn_var).grid(row=1, column=8, columnspan=3, padx=6, pady=(2, 6), sticky='e')
    ttk.Label(gbox, textvariable=presenter.gauge_last_var).grid(row=1, column=0, columnspan=8, padx=10, pady=(2, 6), sticky='w')
    ttk.Label(gbox, textvariable=presenter.gauge_err_var, foreground='red').grid(row=2, column=0, columnspan=8, padx=10, pady=(0, 6), sticky='w')

    cbox = ttk.LabelFrame(tab_od, text='????')
    cbox.pack(fill=tk.X, pady=(4, 8))
    ttk.Label(cbox, text='????').grid(row=0, column=0, padx=(10, 2), pady=6, sticky='e')
    odcal_cmd_combo = ttk.Combobox(cbox, width=10, textvariable=presenter.odcal_cmd_var, values=cmd_presets, state='normal')
    odcal_cmd_combo.grid(row=0, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(cbox, text='???? D_ref(mm)').grid(row=0, column=2, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(cbox, width=10, textvariable=presenter.odcal_dref_var).grid(row=0, column=3, padx=6, pady=6, sticky='w')
    ttk.Label(cbox, text='????(OUT1?)').grid(row=0, column=4, padx=(10, 2), pady=6, sticky='e')
    odcal_out1_map_combo = ttk.Combobox(cbox, width=4, textvariable=presenter.odcal_map_out1_var, values=['L', 'R'], state='readonly')
    odcal_out1_map_combo.grid(row=0, column=5, padx=6, pady=6, sticky='w')
    ttk.Label(cbox, textvariable=presenter.odcal_out2_hint_var).grid(row=0, column=6, padx=6, pady=6, sticky='w')

    def _refresh_out2_hint(*_args):
        presenter.refresh_out2_hint()

    try:
        presenter.odcal_map_out1_var.trace_add('write', _refresh_out2_hint)
    except Exception:
        try:
            presenter.odcal_map_out1_var.trace('w', _refresh_out2_hint)  # type: ignore[arg-type]
        except Exception:
            pass
    presenter.refresh_out2_hint()

    ttk.Label(cbox, text='???B ???? M0,*?OUT1+OUT2??????????? AX3 ???? deg/s??').grid(row=1, column=0, columnspan=8, padx=10, pady=(2, 6), sticky='w')

    abox = ttk.LabelFrame(tab_od, text='????')
    abox.pack(fill=tk.X, pady=(4, 8))
    ttk.Label(abox, text='????').grid(row=0, column=0, padx=(10, 2), pady=6, sticky='e')
    ttk.Radiobutton(abox, text='????', variable=presenter.odcal_mode_var, value='timed').grid(row=0, column=1, padx=6, pady=6, sticky='w')
    ttk.Radiobutton(abox, text='????', variable=presenter.odcal_mode_var, value='one_rev').grid(row=0, column=2, padx=6, pady=6, sticky='w')
    ttk.Label(abox, text='????(Hz)').grid(row=0, column=3, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(abox, width=8, textvariable=presenter.odcal_hz_var).grid(row=0, column=4, padx=6, pady=6, sticky='w')
    ttk.Label(abox, textvariable=presenter.odcal_duration_label_var).grid(row=0, column=5, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(abox, width=8, textvariable=presenter.odcal_duration_var).grid(row=0, column=6, padx=6, pady=6, sticky='w')
    ttk.Label(abox, text='????(deg/s)').grid(row=0, column=7, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(abox, width=8, textvariable=presenter.odcal_rot_degps_var).grid(row=0, column=8, padx=6, pady=6, sticky='w')

    def _refresh_duration_label(*_args):
        presenter.refresh_odcal_duration_label()

    try:
        presenter.odcal_mode_var.trace_add('write', _refresh_duration_label)
    except Exception:
        try:
            presenter.odcal_mode_var.trace('w', _refresh_duration_label)  # type: ignore[arg-type]
        except Exception:
            pass
    presenter.refresh_odcal_duration_label()

    ttk.Button(abox, text='????', command=presenter.calibration_controller.start_od_b_capture).grid(row=1, column=0, padx=10, pady=6, sticky='w')
    ttk.Button(abox, text='??', command=lambda: presenter.calibration_controller.stop_od_b_capture('manual')).grid(row=1, column=1, padx=6, pady=6, sticky='w')
    ttk.Button(abox, text='?? B', command=presenter.calibration_controller.compute_od_b).grid(row=1, column=2, padx=(16, 6), pady=6, sticky='w')
    ttk.Button(abox, text='??', command=presenter.calibration_controller.apply_od_b).grid(row=1, column=3, padx=6, pady=6, sticky='w')
    ttk.Button(abox, text='??RAW', command=presenter.calibration_controller.export_od_b_raw).grid(row=1, column=4, padx=(16, 6), pady=6, sticky='w')
    ttk.Button(abox, text='??', command=presenter.calibration_controller.clear_od_b_capture).grid(row=1, column=5, padx=6, pady=6, sticky='w')

    adv_frame = ttk.Frame(abox)
    adv_frame.grid(row=3, column=0, columnspan=12, padx=10, pady=(0, 6), sticky='ew')
    ttk.Label(adv_frame, text='????').grid(row=0, column=0, padx=(0, 2), pady=4, sticky='e')
    odcal_angle_src_combo = ttk.Combobox(adv_frame, width=10, textvariable=presenter.odcal_angle_src_var, values=['AX3', '???'], state='readonly')
    odcal_angle_src_combo.grid(row=0, column=1, padx=6, pady=4, sticky='w')
    ttk.Label(adv_frame, text='??/??').grid(row=0, column=2, padx=(10, 2), pady=4, sticky='e')
    odcal_filter_combo = ttk.Combobox(adv_frame, width=10, textvariable=presenter.odcal_filter_var, values=['?', '??(3)', '??(5)'], state='readonly')
    odcal_filter_combo.grid(row=0, column=3, padx=6, pady=4, sticky='w')
    ttk.Label(adv_frame, text='??????(?)').grid(row=0, column=4, padx=(10, 2), pady=4, sticky='e')
    ttk.Entry(adv_frame, width=8, textvariable=presenter.odcal_outlier_sigma_var).grid(row=0, column=5, padx=6, pady=4, sticky='w')
    ttk.Label(adv_frame, text='?????????????????????????????0 ?????????').grid(row=1, column=0, columnspan=8, padx=0, pady=(0, 4), sticky='w')
    ttk.Checkbutton(adv_frame, text='????????????????', variable=presenter.odcal_defect_dyn_enable_var).grid(row=2, column=0, columnspan=6, padx=0, pady=(0, 4), sticky='w')

    def _on_angle_src_change(*_args):
        presenter.handle_odcal_angle_source_changed()

    try:
        presenter.odcal_angle_src_var.trace_add('write', _on_angle_src_change)
    except Exception:
        try:
            presenter.odcal_angle_src_var.trace('w', _on_angle_src_change)  # type: ignore[arg-type]
        except Exception:
            pass

    adv_btn = ttk.Button(abox, text='???? ?', command=lambda: presenter.toggle_odcal_advanced(adv_btn, adv_frame))
    adv_btn.grid(row=1, column=6, padx=(16, 6), pady=6, sticky='w')
    ttk.Button(abox, text='??A', command=controller.learn_odcal_defect_a).grid(row=1, column=7, padx=6, pady=6, sticky='w')
    ttk.Button(abox, text='??B(???)', command=controller.learn_odcal_defect_b).grid(row=1, column=8, padx=6, pady=6, sticky='w')
    ttk.Button(abox, text='?????', command=controller.clear_odcal_defect_template).grid(row=1, column=9, padx=6, pady=6, sticky='w')
    adv_frame.grid_remove()

    ttk.Label(abox, textvariable=presenter.odcal_state_var, width=10).grid(row=2, column=0, padx=10, pady=6, sticky='w')
    ttk.Label(abox, textvariable=presenter.odcal_msg_var).grid(row=2, column=1, columnspan=6, padx=6, pady=6, sticky='w')

    rbox = ttk.LabelFrame(tab_od, text='?????')
    rbox.pack(fill=tk.X, pady=(4, 8))
    ttk.Label(rbox, text='B_candidate').grid(row=0, column=0, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(rbox, textvariable=presenter.odcal_B_candidate_var, width=12).grid(row=0, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text='B_active').grid(row=0, column=2, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(rbox, textvariable=presenter.odcal_B_active_var, width=12).grid(row=0, column=3, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text='??? N').grid(row=0, column=4, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(rbox, textvariable=presenter.odcal_n_var, width=8).grid(row=0, column=5, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text='???').grid(row=0, column=6, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(rbox, textvariable=presenter.odcal_elapsed_var, width=8).grid(row=0, column=7, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text='mean(lL+lR)').grid(row=1, column=0, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(rbox, textvariable=presenter.odcal_sum_mean_var, width=12).grid(row=1, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text='std(lL+lR)').grid(row=1, column=2, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(rbox, textvariable=presenter.odcal_sum_std_var, width=12).grid(row=1, column=3, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text='min/max').grid(row=1, column=4, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(rbox, textvariable=presenter.odcal_sum_min_var, width=12).grid(row=1, column=5, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, textvariable=presenter.odcal_sum_max_var, width=12).grid(row=1, column=6, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text='drop(?GO)').grid(row=1, column=7, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(rbox, textvariable=presenter.odcal_drop_rate_var, width=8).grid(row=1, column=8, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text='????').grid(row=2, column=0, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(rbox, textvariable=presenter.odcal_defect_mode_var, width=10).grid(row=2, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text='shift').grid(row=2, column=2, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(rbox, textvariable=presenter.odcal_defect_shift_var, width=8).grid(row=2, column=3, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text='?').grid(row=2, column=4, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(rbox, textvariable=presenter.odcal_defects_var).grid(row=2, column=5, columnspan=4, padx=6, pady=6, sticky='w')

    dbox = ttk.LabelFrame(tab_id, text='??????CL OUT1~OUT5?')
    dbox.pack(fill=tk.X, pady=(4, 8))

    ibox = ttk.LabelFrame(tab_id, text='?????OUT4?? + OUT5??m?')
    ibox.pack(fill=tk.X, pady=(0, 8))
    ttk.Label(ibox, text='ID_ref /mm').grid(row=0, column=0, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(ibox, width=10, textvariable=presenter.idcal_dref_var).grid(row=0, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(ibox, text='??').grid(row=0, column=2, padx=(10, 2), pady=6, sticky='e')
    ttk.Combobox(ibox, width=10, textvariable=presenter.idcal_mode_var, values=['one_rev', 'timed'], state='readonly').grid(row=0, column=3, padx=6, pady=6, sticky='w')
    ttk.Label(ibox, text='Hz').grid(row=0, column=4, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(ibox, width=6, textvariable=presenter.idcal_hz_var).grid(row=0, column=5, padx=6, pady=6, sticky='w')
    ttk.Label(ibox, text='T /s').grid(row=0, column=6, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(ibox, width=6, textvariable=presenter.idcal_duration_var).grid(row=0, column=7, padx=6, pady=6, sticky='w')
    ttk.Label(ibox, text='AX3 ??? /deg/s').grid(row=0, column=8, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(ibox, width=8, textvariable=presenter.idcal_rot_degps_var).grid(row=0, column=9, padx=6, pady=6, sticky='w')
    ttk.Button(ibox, text='????', command=presenter.calibration_controller.start_id_capture).grid(row=1, column=0, padx=10, pady=6, sticky='w')
    ttk.Button(ibox, text='??', command=presenter.calibration_controller.stop_id_capture).grid(row=1, column=1, padx=6, pady=6, sticky='w')
    ttk.Button(ibox, text='??', command=presenter.calibration_controller.clear_id_capture).grid(row=1, column=2, padx=6, pady=6, sticky='w')
    ttk.Button(ibox, text='??', command=presenter.calibration_controller.compute_id_calibration).grid(row=1, column=3, padx=6, pady=6, sticky='w')
    ttk.Button(ibox, text='??', command=presenter.calibration_controller.apply_id_calibration).grid(row=1, column=4, padx=6, pady=6, sticky='w')
    ttk.Button(ibox, text='??raw', command=presenter.calibration_controller.export_id_raw).grid(row=1, column=5, padx=6, pady=6, sticky='w')
    ttk.Button(ibox, text='??', command=presenter.calibration_controller.verify_id_calibration).grid(row=1, column=6, padx=6, pady=6, sticky='w')
    ttk.Label(ibox, textvariable=presenter.idcal_state_var, width=10).grid(row=1, column=7, padx=(16, 6), pady=6, sticky='w')
    ttk.Label(ibox, textvariable=presenter.idcal_msg_var).grid(row=1, column=8, columnspan=2, padx=6, pady=6, sticky='w')
    ttk.Label(ibox, text='?c_candidate /mm').grid(row=2, column=0, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(ibox, textvariable=presenter.idcal_delta_candidate_var, width=12).grid(row=2, column=1, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text='?c_active /mm').grid(row=2, column=2, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(ibox, textvariable=presenter.idcal_delta_active_var, width=12).grid(row=2, column=3, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text='c_max /mm').grid(row=2, column=4, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(ibox, textvariable=presenter.idcal_cmax_var, width=12).grid(row=2, column=5, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text='m_mean /mm').grid(row=2, column=6, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(ibox, textvariable=presenter.idcal_mmean_var, width=12).grid(row=2, column=7, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text='m_pp /mm').grid(row=2, column=8, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(ibox, textvariable=presenter.idcal_mpp_var, width=12).grid(row=2, column=9, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text='???? 2R /mm').grid(row=3, column=0, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(ibox, textvariable=presenter.idcal_fit_diam_var, width=12).grid(row=3, column=1, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text='?? e /mm').grid(row=3, column=2, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(ibox, textvariable=presenter.idcal_fit_e_var, width=12).grid(row=3, column=3, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text='?? y0 /mm').grid(row=3, column=4, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(ibox, textvariable=presenter.idcal_fit_y0_var, width=12).grid(row=3, column=5, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text='rmse(R?) /mm?').grid(row=3, column=6, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(ibox, textvariable=presenter.idcal_fit_rmse_var, width=12).grid(row=3, column=7, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text='???D /mm').grid(row=4, column=0, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(ibox, textvariable=presenter.idcal_chk_err_var, width=12).grid(row=4, column=1, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text='???').grid(row=4, column=2, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(ibox, textvariable=presenter.idcal_chk_cov_var, width=12).grid(row=4, column=3, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text='N').grid(row=4, column=4, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(ibox, textvariable=presenter.idcal_chk_n_var, width=12).grid(row=4, column=5, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text='d?_max').grid(row=4, column=6, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(ibox, textvariable=presenter.idcal_chk_dtheta_var, width=12).grid(row=4, column=7, padx=6, pady=4, sticky='w')

    sbox = ttk.LabelFrame(tab_id, text='ID Single-Probe Calibration (OUT2/L2)')
    sbox.pack(fill=tk.X, pady=(0, 8))
    ttk.Label(sbox, text='ID_ref /mm').grid(row=0, column=0, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(sbox, width=10, textvariable=presenter.id_single_cal_dref_var).grid(row=0, column=1, padx=6, pady=6, sticky='w')
    ttk.Button(sbox, text='Capture 1 rev', command=presenter.calibration_controller.start_id_single_capture).grid(row=0, column=2, padx=(16, 6), pady=6, sticky='w')
    ttk.Button(sbox, text='Stop', command=lambda: presenter.calibration_controller.stop_id_single_capture('manual')).grid(row=0, column=3, padx=6, pady=6, sticky='w')
    ttk.Button(sbox, text='Compute & Write', command=presenter.calibration_controller.compute_and_write_id_single_calibration).grid(row=0, column=4, padx=(16, 6), pady=6, sticky='w')
    ttk.Label(sbox, textvariable=presenter.id_single_cal_state_var, width=10).grid(row=1, column=0, padx=(10, 2), pady=4, sticky='w')
    ttk.Label(sbox, textvariable=presenter.id_single_cal_msg_var).grid(row=1, column=1, columnspan=3, padx=6, pady=4, sticky='w')
    ttk.Label(sbox, textvariable=presenter.id_single_cal_warn_var, foreground='red').grid(row=1, column=4, padx=6, pady=4, sticky='w')
    ttk.Label(sbox, text='mean(L2_decenter)').grid(row=2, column=0, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(sbox, textvariable=presenter.id_single_cal_mean_var, width=12).grid(row=2, column=1, padx=6, pady=4, sticky='w')
    ttk.Label(sbox, text='B').grid(row=2, column=2, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(sbox, textvariable=presenter.id_single_cal_B_var, width=12).grid(row=2, column=3, padx=6, pady=4, sticky='w')
    ttk.Label(sbox, text='cov').grid(row=2, column=4, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(sbox, textvariable=presenter.id_single_cal_cov_var, width=8).grid(row=2, column=5, padx=6, pady=4, sticky='w')
    ttk.Label(sbox, text='ecc_amp').grid(row=3, column=0, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(sbox, textvariable=presenter.id_single_cal_ecc_amp_var, width=12).grid(row=3, column=1, padx=6, pady=4, sticky='w')
    ttk.Label(sbox, text='ecc_ang(deg)').grid(row=3, column=2, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(sbox, textvariable=presenter.id_single_cal_ecc_ang_var, width=12).grid(row=3, column=3, padx=6, pady=4, sticky='w')

    ttk.Label(dbox, text='OUT1 x1(?) /mm').grid(row=0, column=0, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(dbox, textvariable=presenter.cl_out1_var, width=12).grid(row=0, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text='cnt').grid(row=0, column=2, padx=(6, 2), pady=6, sticky='e')
    ttk.Label(dbox, textvariable=presenter.cl_out1_cnt_var, width=10).grid(row=0, column=3, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text='OUT2 x2(?) /mm').grid(row=0, column=4, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(dbox, textvariable=presenter.cl_out2_var, width=12).grid(row=0, column=5, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text='cnt').grid(row=0, column=6, padx=(6, 2), pady=6, sticky='e')
    ttk.Label(dbox, textvariable=presenter.cl_out2_cnt_var, width=10).grid(row=0, column=7, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text='OUT4 ??ID /mm').grid(row=1, column=0, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(dbox, textvariable=presenter.cl_out4_var, width=12).grid(row=1, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text='cnt').grid(row=1, column=2, padx=(6, 2), pady=6, sticky='e')
    ttk.Label(dbox, textvariable=presenter.cl_out4_cnt_var, width=10).grid(row=1, column=3, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text='OUT5 m(??) /mm').grid(row=1, column=4, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(dbox, textvariable=presenter.cl_out5_var, width=12).grid(row=1, column=5, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text='cnt').grid(row=1, column=6, padx=(6, 2), pady=6, sticky='e')
    ttk.Label(dbox, textvariable=presenter.cl_out5_cnt_var, width=10).grid(row=1, column=7, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text='m?=(x1-x2)/2').grid(row=2, column=0, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(dbox, textvariable=presenter.cl_m_calc_var, width=12).grid(row=2, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text='?=m?-OUT5').grid(row=2, column=4, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(dbox, textvariable=presenter.cl_m_diff_var, width=12).grid(row=2, column=5, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text='OUT3(??/??)').grid(row=3, column=0, padx=(10, 2), pady=(2, 6), sticky='e')
    ttk.Label(dbox, textvariable=presenter.cl_out3_var, width=12).grid(row=3, column=1, padx=6, pady=(2, 6), sticky='w')
    ttk.Label(dbox, text='cnt').grid(row=3, column=2, padx=(6, 2), pady=(2, 6), sticky='e')
    ttk.Label(dbox, textvariable=presenter.cl_out3_cnt_var, width=10).grid(row=3, column=3, padx=6, pady=(2, 6), sticky='w')
    ttk.Label(dbox, text='????????ID??? OUT4?m? ???? OUT5 ??/???????').grid(row=4, column=0, columnspan=8, padx=10, pady=(2, 6), sticky='w')
