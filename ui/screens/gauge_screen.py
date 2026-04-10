from __future__ import annotations

"""Gauge/PLC/calibration screen UI only."""

import tkinter as tk
from tkinter import ttk

from application.state import FIXED_SECTION_PRIMARY_METRICS
from config.addresses import DEFAULT_GAUGE_PORT


def build_gauge_screen(parent: ttk.Frame, *, presenter, controller, ui) -> None:
    outer = ttk.Frame(parent)
    outer.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(outer, highlightthickness=0)
    vscroll = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
    canvas.configure(yscrollcommand=vscroll.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vscroll.pack(side=tk.RIGHT, fill=tk.Y)

    content = ttk.Frame(canvas)
    canvas_window = canvas.create_window((0, 0), window=content, anchor='nw')

    def _sync_scrollregion(_event=None) -> None:
        try:
            canvas.configure(scrollregion=canvas.bbox('all'))
        except Exception:
            pass

    def _sync_content_width(_event=None) -> None:
        try:
            canvas.itemconfigure(canvas_window, width=int(canvas.winfo_width()))
        except Exception:
            pass

    def _on_mousewheel(event) -> str | None:
        try:
            delta = int(getattr(event, 'delta', 0))
            if delta:
                canvas.yview_scroll(int(-delta / 120), 'units')
                return 'break'
        except Exception:
            pass
        return None

    def _on_mousewheel_linux_up(_event) -> str:
        canvas.yview_scroll(-1, 'units')
        return 'break'

    def _on_mousewheel_linux_down(_event) -> str:
        canvas.yview_scroll(1, 'units')
        return 'break'

    def _bind_mousewheel(_event=None) -> None:
        try:
            canvas.bind_all('<MouseWheel>', _on_mousewheel)
            canvas.bind_all('<Button-4>', _on_mousewheel_linux_up)
            canvas.bind_all('<Button-5>', _on_mousewheel_linux_down)
        except Exception:
            pass

    def _unbind_mousewheel(_event=None) -> None:
        try:
            canvas.unbind_all('<MouseWheel>')
            canvas.unbind_all('<Button-4>')
            canvas.unbind_all('<Button-5>')
        except Exception:
            pass

    content.bind('<Configure>', _sync_scrollregion)
    canvas.bind('<Configure>', _sync_content_width)
    canvas.bind('<Enter>', _bind_mousewheel)
    canvas.bind('<Leave>', _unbind_mousewheel)
    content.bind('<Enter>', _bind_mousewheel)
    content.bind('<Leave>', _unbind_mousewheel)
    _sync_content_width()
    _sync_scrollregion()
    presenter.ensure_vars(content)
    pbox = ttk.LabelFrame(content, text="PLC 通信（Modbus TCP）")
    pbox.pack(fill=tk.X, pady=(4, 8))
    ttk.Label(pbox, text='IP').grid(row=0, column=0, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(pbox, width=14, textvariable=presenter.ip_var).grid(row=0, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(pbox, text='Port').grid(row=0, column=2, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(pbox, width=6, textvariable=presenter.port_var).grid(row=0, column=3, padx=6, pady=6, sticky='w')
    ttk.Button(pbox, text='连接/重连', command=controller.apply_plc_connection).grid(row=0, column=4, padx=8, pady=6)
    ttk.Label(pbox, textvariable=presenter.plc_status_var).grid(row=0, column=5, padx=10, pady=6, sticky='w')

    nb = ttk.Notebook(content)
    nb.pack(fill=tk.BOTH, expand=True, pady=(4, 8))
    tab_od = ttk.Frame(nb)
    tab_id = ttk.Frame(nb)
    nb.add(tab_od, text="外径（实时+标定）")
    nb.add(tab_id, text="内径（实时+标定）")

    gbox = ttk.LabelFrame(tab_od, text="测径仪（外径 OD, 串口）")
    gbox.pack(fill=tk.X, pady=(4, 8))
    ttk.Checkbutton(gbox, text='模拟测径仪', variable=presenter.sim_gauge_var, command=controller.toggle_sim_gauge).grid(row=0, column=0, padx=10, pady=6, sticky='w')
    ttk.Label(gbox, text='串口').grid(row=0, column=1, padx=(10, 2), pady=6, sticky='e')
    port_combo = presenter.remember_widget('port_combo', ttk.Combobox(gbox, width=12, state='readonly', values=presenter.list_serial_ports()))
    port_combo.grid(row=0, column=2, padx=6, pady=6, sticky='w')
    port_combo.set(DEFAULT_GAUGE_PORT)
    ttk.Button(gbox, text='刷新', command=controller.refresh_gauge_ports).grid(row=0, column=3, padx=6, pady=6)
    ttk.Label(gbox, text='波特率').grid(row=0, column=4, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(gbox, width=8, textvariable=presenter.baud_var).grid(row=0, column=5, padx=6, pady=6, sticky='w')
    ttk.Label(gbox, text='请求指令').grid(row=0, column=6, padx=(10, 2), pady=6, sticky='e')
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

    ttk.Button(gbox, text='连接', command=controller.connect_gauge).grid(row=0, column=8, padx=6, pady=6)
    ttk.Button(gbox, text='断开', command=controller.disconnect_gauge).grid(row=0, column=9, padx=6, pady=6)
    ttk.Button(gbox, text='请求一次', command=controller.request_gauge_once).grid(row=0, column=10, padx=6, pady=6)
    ttk.Label(gbox, textvariable=presenter.gauge_conn_var).grid(row=1, column=8, columnspan=3, padx=6, pady=(2, 6), sticky='e')
    ttk.Label(gbox, textvariable=presenter.gauge_last_var).grid(row=1, column=0, columnspan=8, padx=10, pady=(2, 6), sticky='w')
    ttk.Label(gbox, textvariable=presenter.gauge_err_var, foreground='red').grid(row=2, column=0, columnspan=8, padx=10, pady=(0, 6), sticky='w')

    cbox = ttk.LabelFrame(tab_od, text="标定参数")
    cbox.pack(fill=tk.X, pady=(4, 8))
    ttk.Label(cbox, text="请求指令").grid(row=0, column=0, padx=(10, 2), pady=6, sticky="e")
    odcal_cmd_combo = ttk.Combobox(cbox, width=10, textvariable=presenter.odcal_cmd_var, values=cmd_presets, state='normal')
    odcal_cmd_combo.grid(row=0, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(cbox, text="环规直径 D_ref(mm)").grid(row=0, column=2, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(cbox, width=10, textvariable=presenter.odcal_dref_var).grid(row=0, column=3, padx=6, pady=6, sticky='w')
    ttk.Label(cbox, text="通道映射(OUT1→)").grid(row=0, column=4, padx=(10, 2), pady=6, sticky="e")
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

    ttk.Label(cbox, text='建议：B 标定使用 M0,*（OUT1+OUT2）。一圈采样会自动控制 AX3 旋转（按 deg/s）。').grid(row=1, column=0, columnspan=8, padx=10, pady=(2, 6), sticky='w')

    abox = ttk.LabelFrame(tab_od, text="采集控制")
    abox.pack(fill=tk.X, pady=(4, 8))
    ttk.Label(abox, text="采集模式").grid(row=0, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Radiobutton(abox, text='定时采样', variable=presenter.odcal_mode_var, value='timed').grid(row=0, column=1, padx=6, pady=6, sticky='w')
    ttk.Radiobutton(abox, text='一圈采样', variable=presenter.odcal_mode_var, value='one_rev').grid(row=0, column=2, padx=6, pady=6, sticky='w')
    ttk.Label(abox, text="采样频率(Hz)").grid(row=0, column=3, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(abox, width=8, textvariable=presenter.odcal_hz_var).grid(row=0, column=4, padx=6, pady=6, sticky='w')
    ttk.Label(abox, textvariable=presenter.odcal_duration_label_var).grid(row=0, column=5, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(abox, width=8, textvariable=presenter.odcal_duration_var).grid(row=0, column=6, padx=6, pady=6, sticky='w')
    ttk.Label(abox, text="旋转速度(deg/s)").grid(row=0, column=7, padx=(10, 2), pady=6, sticky="e")
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

    ttk.Button(abox, text="开始采集", command=presenter.calibration_controller.start_od_b_capture).grid(row=1, column=0, padx=10, pady=6, sticky="w")
    ttk.Button(abox, text='停止', command=lambda: presenter.calibration_controller.stop_od_b_capture('manual')).grid(row=1, column=1, padx=6, pady=6, sticky='w')
    ttk.Button(abox, text="计算 B", command=presenter.calibration_controller.compute_od_b).grid(row=1, column=2, padx=(16, 6), pady=6, sticky="w")
    ttk.Button(abox, text="应用", command=presenter.calibration_controller.apply_od_b).grid(row=1, column=3, padx=6, pady=6, sticky="w")
    ttk.Button(abox, text="导出RAW", command=presenter.calibration_controller.export_od_b_raw).grid(row=1, column=4, padx=(16, 6), pady=6, sticky="w")
    ttk.Button(abox, text="清空", command=presenter.calibration_controller.clear_od_b_capture).grid(row=1, column=5, padx=6, pady=6, sticky="w")

    adv_frame = ttk.Frame(abox)
    adv_frame.grid(row=3, column=0, columnspan=12, padx=10, pady=(0, 6), sticky='ew')
    ttk.Label(adv_frame, text="角度来源").grid(row=0, column=0, padx=(0, 2), pady=4, sticky="e")
    odcal_angle_src_combo = ttk.Combobox(adv_frame, width=10, textvariable=presenter.odcal_angle_src_var, values=['AX3', '无角度'], state='readonly')
    odcal_angle_src_combo.grid(row=0, column=1, padx=6, pady=4, sticky='w')
    ttk.Label(adv_frame, text="去抖/滤波").grid(row=0, column=2, padx=(10, 2), pady=4, sticky="e")
    odcal_filter_combo = ttk.Combobox(adv_frame, width=10, textvariable=presenter.odcal_filter_var, values=['无', '中值(3)', '中值(5)'], state='readonly')
    odcal_filter_combo.grid(row=0, column=3, padx=6, pady=4, sticky='w')
    ttk.Label(adv_frame, text="异常剔除阈值(σ)").grid(row=0, column=4, padx=(10, 2), pady=4, sticky="e")
    ttk.Entry(adv_frame, width=8, textvariable=presenter.odcal_outlier_sigma_var).grid(row=0, column=5, padx=6, pady=4, sticky='w')
    ttk.Label(adv_frame, text='说明：角度=无角度 时，一圈采样会自动切到定时采样；σ=0 表示不做离群剔除。').grid(row=1, column=0, columnspan=8, padx=0, pady=(0, 4), sticky='w')
    ttk.Checkbutton(adv_frame, text='未学习模板时：动态屏蔽最深凹陷段', variable=presenter.odcal_defect_dyn_enable_var).grid(row=2, column=0, columnspan=6, padx=0, pady=(0, 4), sticky='w')

    def _on_angle_src_change(*_args):
        presenter.handle_odcal_angle_source_changed()

    try:
        presenter.odcal_angle_src_var.trace_add('write', _on_angle_src_change)
    except Exception:
        try:
            presenter.odcal_angle_src_var.trace('w', _on_angle_src_change)  # type: ignore[arg-type]
        except Exception:
            pass

    adv_btn = ttk.Button(abox, text='高级参数 ▸', command=lambda: presenter.toggle_odcal_advanced(adv_btn, adv_frame))
    adv_btn.grid(row=1, column=6, padx=(16, 6), pady=6, sticky='w')
    ttk.Button(abox, text='学习A', command=controller.learn_odcal_defect_a).grid(row=1, column=7, padx=6, pady=6, sticky='w')
    ttk.Button(abox, text='学习B(生成表)', command=controller.learn_odcal_defect_b).grid(row=1, column=8, padx=6, pady=6, sticky='w')
    ttk.Button(abox, text='清除凹陷表', command=controller.clear_odcal_defect_template).grid(row=1, column=9, padx=6, pady=6, sticky='w')
    adv_frame.grid_remove()

    ttk.Label(abox, textvariable=presenter.odcal_state_var, width=10).grid(row=2, column=0, padx=10, pady=6, sticky='w')
    ttk.Label(abox, textvariable=presenter.odcal_msg_var).grid(row=2, column=1, columnspan=6, padx=6, pady=6, sticky='w')

    rbox = ttk.LabelFrame(tab_od, text="结果与质量")
    rbox.pack(fill=tk.X, pady=(4, 8))
    ttk.Label(rbox, text='B_candidate').grid(row=0, column=0, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(rbox, textvariable=presenter.odcal_B_candidate_var, width=12).grid(row=0, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text='B_active').grid(row=0, column=2, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(rbox, textvariable=presenter.odcal_B_active_var, width=12).grid(row=0, column=3, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text="样本数 N").grid(row=0, column=4, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(rbox, textvariable=presenter.odcal_n_var, width=8).grid(row=0, column=5, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text="已用时").grid(row=0, column=6, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(rbox, textvariable=presenter.odcal_elapsed_var, width=8).grid(row=0, column=7, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text='mean(lL+lR)').grid(row=1, column=0, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(rbox, textvariable=presenter.odcal_sum_mean_var, width=12).grid(row=1, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text='std(lL+lR)').grid(row=1, column=2, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(rbox, textvariable=presenter.odcal_sum_std_var, width=12).grid(row=1, column=3, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text='min/max').grid(row=1, column=4, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(rbox, textvariable=presenter.odcal_sum_min_var, width=12).grid(row=1, column=5, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, textvariable=presenter.odcal_sum_max_var, width=12).grid(row=1, column=6, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text="drop(非GO)").grid(row=1, column=7, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(rbox, textvariable=presenter.odcal_drop_rate_var, width=8).grid(row=1, column=8, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text="凹陷屏蔽").grid(row=2, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(rbox, textvariable=presenter.odcal_defect_mode_var, width=10).grid(row=2, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text='shift').grid(row=2, column=2, padx=(10, 2), pady=6, sticky='e')
    ttk.Label(rbox, textvariable=presenter.odcal_defect_shift_var, width=8).grid(row=2, column=3, padx=6, pady=6, sticky='w')
    ttk.Label(rbox, text="段").grid(row=2, column=4, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(rbox, textvariable=presenter.odcal_defects_var).grid(row=2, column=5, columnspan=4, padx=6, pady=6, sticky='w')
    vbox = ttk.LabelFrame(tab_od, text="验证调试")
    vbox.pack(fill=tk.X, pady=(4, 8))
    vbox.configure(text="验证调试")
    ttk.Label(vbox, text="section_name").grid(row=0, column=0, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(vbox, width=16, textvariable=presenter.validation_debug_section_name_var).grid(row=0, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(vbox, text="metric_name").grid(row=0, column=2, padx=(10, 2), pady=6, sticky='e')
    ttk.Combobox(
        vbox,
        width=18,
        textvariable=presenter.validation_debug_metric_name_var,
        values=FIXED_SECTION_PRIMARY_METRICS,
        state='readonly',
    ).grid(row=0, column=3, padx=6, pady=6, sticky='w')
    ttk.Label(vbox, text="repeat_count").grid(row=0, column=4, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(vbox, width=8, textvariable=presenter.validation_debug_repeat_count_var).grid(row=0, column=5, padx=6, pady=6, sticky='w')
    start_btn = presenter.remember_widget(
        'validation_debug_start_btn',
        ttk.Button(
            vbox,
            text='开始验证',
            command=lambda: controller.start_fixed_section_repeatability_debug(
                section_name=presenter.validation_debug_section_name_var.get(),
                metric_name=presenter.validation_debug_metric_name_var.get(),
                repeat_count=presenter.validation_debug_repeat_count_var.get(),
                reclamp_enabled=presenter.validation_debug_reclamp_enabled_var.get(),
                rotation_stop_before_measure=presenter.validation_debug_rotation_stop_before_measure_var.get(),
                release_settle_s=presenter.validation_debug_release_settle_s_var.get(),
                clamp_settle_s=presenter.validation_debug_clamp_settle_s_var.get(),
            ),
        ),
    )
    start_btn.grid(row=0, column=6, padx=(12, 10), pady=6, sticky='w')
    ttk.Checkbutton(
        vbox,
        text='启用重复装夹',
        variable=presenter.validation_debug_reclamp_enabled_var,
    ).grid(row=2, column=0, columnspan=2, padx=(10, 2), pady=6, sticky='w')
    ttk.Checkbutton(
        vbox,
        text='测前停转',
        variable=presenter.validation_debug_rotation_stop_before_measure_var,
    ).grid(row=2, column=2, columnspan=2, padx=(10, 2), pady=6, sticky='w')
    ttk.Label(vbox, text='松夹等待(s)').grid(row=2, column=4, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(vbox, width=8, textvariable=presenter.validation_debug_release_settle_s_var).grid(row=2, column=5, padx=6, pady=6, sticky='w')
    ttk.Label(vbox, text='夹紧等待(s)').grid(row=2, column=6, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(vbox, width=8, textvariable=presenter.validation_debug_clamp_settle_s_var).grid(row=2, column=7, padx=(6, 10), pady=6, sticky='w')
    start_btn.configure(text='开始验证')
    ttk.Label(vbox, text='section_name 仅作标签，不触发定位').grid(row=1, column=0, columnspan=8, padx=10, pady=(0, 6), sticky='w')
    ttk.Label(vbox, text='status').grid(row=3, column=0, padx=(10, 2), pady=4, sticky='e')
    try:
        for child in vbox.winfo_children():
            try:
                if isinstance(child, ttk.Label) and int(child.grid_info().get('row', -1)) == 1:
                    child.configure(text='section_name 仅作标签，不触发定位')
            except Exception:
                pass
    except Exception:
        pass
    ttk.Label(vbox, textvariable=presenter.validation_debug_status_var, width=16).grid(row=3, column=1, padx=6, pady=4, sticky='w')
    ttk.Label(vbox, text='phase').grid(row=3, column=2, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(vbox, textvariable=presenter.validation_debug_phase_var, width=16).grid(row=3, column=3, padx=6, pady=4, sticky='w')
    ttk.Label(vbox, text='result').grid(row=3, column=4, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(vbox, textvariable=presenter.validation_debug_result_var).grid(row=3, column=5, columnspan=3, padx=6, pady=4, sticky='w')
    ttk.Label(vbox, text='error').grid(row=4, column=0, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(vbox, textvariable=presenter.validation_debug_error_var, foreground='red').grid(row=4, column=1, columnspan=7, padx=6, pady=4, sticky='w')
    ttk.Label(vbox, text='export').grid(row=5, column=0, padx=(10, 2), pady=(4, 8), sticky='e')
    ttk.Label(vbox, textvariable=presenter.validation_debug_export_path_var).grid(row=5, column=1, columnspan=7, padx=6, pady=(4, 8), sticky='w')

    dbox = ttk.LabelFrame(tab_id, text="位移计实时（CL OUT1~OUT5）")
    dbox.pack(fill=tk.X, pady=(4, 8))

    ibox = ttk.LabelFrame(tab_id, text="内径标定（OUT4弦长 + OUT5偏移m）")
    ibox.pack(fill=tk.X, pady=(0, 8))
    ttk.Label(ibox, text='ID_ref /mm').grid(row=0, column=0, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(ibox, width=10, textvariable=presenter.idcal_dref_var).grid(row=0, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(ibox, text="模式").grid(row=0, column=2, padx=(10, 2), pady=6, sticky="e")
    ttk.Combobox(ibox, width=10, textvariable=presenter.idcal_mode_var, values=['one_rev', 'timed'], state='readonly').grid(row=0, column=3, padx=6, pady=6, sticky='w')
    ttk.Label(ibox, text='Hz').grid(row=0, column=4, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(ibox, width=6, textvariable=presenter.idcal_hz_var).grid(row=0, column=5, padx=6, pady=6, sticky='w')
    ttk.Label(ibox, text='T /s').grid(row=0, column=6, padx=(10, 2), pady=6, sticky='e')
    ttk.Entry(ibox, width=6, textvariable=presenter.idcal_duration_var).grid(row=0, column=7, padx=6, pady=6, sticky='w')
    ttk.Label(ibox, text="AX3 角速度 /deg/s").grid(row=0, column=8, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(ibox, width=8, textvariable=presenter.idcal_rot_degps_var).grid(row=0, column=9, padx=6, pady=6, sticky='w')
    ttk.Button(ibox, text="开始采集", command=presenter.calibration_controller.start_id_capture).grid(row=1, column=0, padx=10, pady=6, sticky="w")
    ttk.Button(ibox, text="停止", command=presenter.calibration_controller.stop_id_capture).grid(row=1, column=1, padx=6, pady=6, sticky="w")
    ttk.Button(ibox, text="清空", command=presenter.calibration_controller.clear_id_capture).grid(row=1, column=2, padx=6, pady=6, sticky="w")
    ttk.Button(ibox, text="计算", command=presenter.calibration_controller.compute_id_calibration).grid(row=1, column=3, padx=6, pady=6, sticky="w")
    ttk.Button(ibox, text="应用", command=presenter.calibration_controller.apply_id_calibration).grid(row=1, column=4, padx=6, pady=6, sticky="w")
    ttk.Button(ibox, text="导出raw", command=presenter.calibration_controller.export_id_raw).grid(row=1, column=5, padx=6, pady=6, sticky="w")
    ttk.Button(ibox, text="复核", command=presenter.calibration_controller.verify_id_calibration).grid(row=1, column=6, padx=6, pady=6, sticky="w")
    ttk.Label(ibox, textvariable=presenter.idcal_state_var, width=10).grid(row=1, column=7, padx=(16, 6), pady=6, sticky='w')
    ttk.Label(ibox, textvariable=presenter.idcal_msg_var).grid(row=1, column=8, columnspan=2, padx=6, pady=6, sticky='w')
    ttk.Label(ibox, text="δc_candidate /mm").grid(row=2, column=0, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=presenter.idcal_delta_candidate_var, width=12).grid(row=2, column=1, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text="δc_active /mm").grid(row=2, column=2, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=presenter.idcal_delta_active_var, width=12).grid(row=2, column=3, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text='c_max /mm').grid(row=2, column=4, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(ibox, textvariable=presenter.idcal_cmax_var, width=12).grid(row=2, column=5, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text='m_mean /mm').grid(row=2, column=6, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(ibox, textvariable=presenter.idcal_mmean_var, width=12).grid(row=2, column=7, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text='m_pp /mm').grid(row=2, column=8, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(ibox, textvariable=presenter.idcal_mpp_var, width=12).grid(row=2, column=9, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text="拟合直径 2R /mm").grid(row=3, column=0, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=presenter.idcal_fit_diam_var, width=12).grid(row=3, column=1, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text="拟合 e /mm").grid(row=3, column=2, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=presenter.idcal_fit_e_var, width=12).grid(row=3, column=3, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text="拟合 y0 /mm").grid(row=3, column=4, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=presenter.idcal_fit_y0_var, width=12).grid(row=3, column=5, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text="rmse(R²) /mm²").grid(row=3, column=6, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=presenter.idcal_fit_rmse_var, width=12).grid(row=3, column=7, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text="复核ΔD /mm").grid(row=4, column=0, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=presenter.idcal_chk_err_var, width=12).grid(row=4, column=1, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text="覆盖率").grid(row=4, column=2, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(ibox, textvariable=presenter.idcal_chk_cov_var, width=12).grid(row=4, column=3, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text='N').grid(row=4, column=4, padx=(10, 2), pady=4, sticky='e')
    ttk.Label(ibox, textvariable=presenter.idcal_chk_n_var, width=12).grid(row=4, column=5, padx=6, pady=4, sticky='w')
    ttk.Label(ibox, text="dθ_max").grid(row=4, column=6, padx=(10, 2), pady=4, sticky="e")
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

    ttk.Label(dbox, text="OUT1 x1(右) /mm").grid(row=0, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(dbox, textvariable=presenter.cl_out1_var, width=12).grid(row=0, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text='cnt').grid(row=0, column=2, padx=(6, 2), pady=6, sticky='e')
    ttk.Label(dbox, textvariable=presenter.cl_out1_cnt_var, width=10).grid(row=0, column=3, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text="OUT2 x2(左) /mm").grid(row=0, column=4, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(dbox, textvariable=presenter.cl_out2_var, width=12).grid(row=0, column=5, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text='cnt').grid(row=0, column=6, padx=(6, 2), pady=6, sticky='e')
    ttk.Label(dbox, textvariable=presenter.cl_out2_cnt_var, width=10).grid(row=0, column=7, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text="OUT4 内径ID /mm").grid(row=1, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(dbox, textvariable=presenter.cl_out4_var, width=12).grid(row=1, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text='cnt').grid(row=1, column=2, padx=(6, 2), pady=6, sticky='e')
    ttk.Label(dbox, textvariable=presenter.cl_out4_cnt_var, width=10).grid(row=1, column=3, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text="OUT5 m(投影) /mm").grid(row=1, column=4, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(dbox, textvariable=presenter.cl_out5_var, width=12).grid(row=1, column=5, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text='cnt').grid(row=1, column=6, padx=(6, 2), pady=6, sticky='e')
    ttk.Label(dbox, textvariable=presenter.cl_out5_cnt_var, width=10).grid(row=1, column=7, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text="m̂=(x1-x2)/2").grid(row=2, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(dbox, textvariable=presenter.cl_m_calc_var, width=12).grid(row=2, column=1, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text="Δ=m̂-OUT5").grid(row=2, column=4, padx=(10, 2), pady=6, sticky="e")
    ttk.Label(dbox, textvariable=presenter.cl_m_diff_var, width=12).grid(row=2, column=5, padx=6, pady=6, sticky='w')
    ttk.Label(dbox, text="OUT3(保留/厚度)").grid(row=3, column=0, padx=(10, 2), pady=(2, 6), sticky="e")
    ttk.Label(dbox, textvariable=presenter.cl_out3_var, width=12).grid(row=3, column=1, padx=6, pady=(2, 6), sticky='w')
    ttk.Label(dbox, text='cnt').grid(row=3, column=2, padx=(6, 2), pady=(2, 6), sticky='e')
    ttk.Label(dbox, textvariable=presenter.cl_out3_cnt_var, width=10).grid(row=3, column=3, padx=6, pady=(2, 6), sticky='w')
    ttk.Label(dbox, text='提示：主流程内径ID默认取 OUT4。m̂用于校验OUT5公式/符号是否一致。').grid(row=4, column=0, columnspan=8, padx=10, pady=(2, 6), sticky='w')
