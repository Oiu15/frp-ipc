from __future__ import annotations

"""Standalone validation screen UI."""

import tkinter as tk
from tkinter import ttk

from application.state import (
    FIXED_SECTION_PRIMARY_METRICS,
    VALIDATION_MOVE_CHANNELS,
    VALIDATION_MOVE_SCENARIOS,
)


def build_validation_screen(parent: ttk.Frame, *, presenter, controller, ui) -> None:
    del ui  # reserved for future screen-local UI helpers

    outer = ttk.Frame(parent)
    outer.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(outer, highlightthickness=0)
    vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vscroll.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vscroll.pack(side=tk.RIGHT, fill=tk.Y)

    content = ttk.Frame(canvas)
    canvas_window = canvas.create_window((0, 0), window=content, anchor="nw")

    def _sync_scrollregion(_event=None) -> None:
        try:
            canvas.configure(scrollregion=canvas.bbox("all"))
        except Exception:
            pass

    def _sync_content_width(_event=None) -> None:
        try:
            canvas.itemconfigure(canvas_window, width=int(canvas.winfo_width()))
        except Exception:
            pass

    def _on_mousewheel(event) -> str | None:
        try:
            delta = int(getattr(event, "delta", 0))
            if delta:
                canvas.yview_scroll(int(-delta / 120), "units")
                return "break"
        except Exception:
            pass
        return None

    def _on_mousewheel_linux_up(_event) -> str:
        canvas.yview_scroll(-1, "units")
        return "break"

    def _on_mousewheel_linux_down(_event) -> str:
        canvas.yview_scroll(1, "units")
        return "break"

    def _bind_mousewheel(_event=None) -> None:
        try:
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel_linux_up)
            canvas.bind_all("<Button-5>", _on_mousewheel_linux_down)
        except Exception:
            pass

    def _unbind_mousewheel(_event=None) -> None:
        try:
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")
        except Exception:
            pass

    content.bind("<Configure>", _sync_scrollregion)
    canvas.bind("<Configure>", _sync_content_width)
    canvas.bind("<Enter>", _bind_mousewheel)
    canvas.bind("<Leave>", _unbind_mousewheel)
    content.bind("<Enter>", _bind_mousewheel)
    content.bind("<Leave>", _unbind_mousewheel)
    _sync_content_width()
    _sync_scrollregion()
    presenter.ensure_vars(content)

    header = ttk.LabelFrame(content, text="Validation")
    header.pack(fill=tk.X, pady=(4, 8))
    ttk.Label(
        header,
        text=(
            "Standalone validation entry. Configure the run here, then start or stop "
            "the existing validation workflow without going through Gauge debug controls."
        ),
        wraplength=900,
        justify="left",
    ).grid(row=0, column=0, padx=10, pady=8, sticky="w")

    param_box = ttk.LabelFrame(content, text="Parameters")
    param_box.pack(fill=tk.X, pady=(4, 8))

    section_choices = presenter.validation_section_choices()
    section_name_combo = presenter.remember_widget(
        "validation_screen_section_name_combo",
        ttk.Combobox(
            param_box,
            width=18,
            textvariable=presenter.validation_debug_section_name_var,
            values=section_choices,
            state="readonly",
        ),
    )
    move_from_combo = presenter.remember_widget(
        "validation_screen_move_from_section_combo",
        ttk.Combobox(
            param_box,
            width=18,
            textvariable=presenter.validation_debug_move_from_section_var,
            values=section_choices,
            state="readonly",
        ),
    )
    move_target_combo = presenter.remember_widget(
        "validation_screen_move_target_section_combo",
        ttk.Combobox(
            param_box,
            width=18,
            textvariable=presenter.validation_debug_move_target_section_var,
            values=section_choices,
            state="readonly",
        ),
    )
    move_return_combo = presenter.remember_widget(
        "validation_screen_move_return_section_combo",
        ttk.Combobox(
            param_box,
            width=18,
            textvariable=presenter.validation_debug_move_return_section_var,
            values=section_choices,
            state="readonly",
        ),
    )

    def _refresh_validation_section_combos(_event=None) -> None:
        choices = presenter.validation_section_choices()
        for combo in (
            section_name_combo,
            move_from_combo,
            move_target_combo,
            move_return_combo,
        ):
            try:
                combo.configure(values=choices)
            except Exception:
                pass
        for var in (
            presenter.validation_debug_section_name_var,
            presenter.validation_debug_move_from_section_var,
            presenter.validation_debug_move_target_section_var,
            presenter.validation_debug_move_return_section_var,
        ):
            try:
                if str(var.get() or "") not in choices and choices:
                    var.set(choices[0])
            except Exception:
                pass

    for combo in (
        section_name_combo,
        move_from_combo,
        move_target_combo,
        move_return_combo,
    ):
        combo.bind("<Button-1>", _refresh_validation_section_combos)
        combo.bind("<FocusIn>", _refresh_validation_section_combos)

    ttk.Label(param_box, text="section").grid(row=0, column=0, padx=(10, 2), pady=6, sticky="e")
    section_name_combo.grid(row=0, column=1, padx=6, pady=6, sticky="w")
    ttk.Label(param_box, text="metric").grid(row=0, column=2, padx=(10, 2), pady=6, sticky="e")
    ttk.Combobox(
        param_box,
        width=18,
        textvariable=presenter.validation_debug_metric_name_var,
        values=FIXED_SECTION_PRIMARY_METRICS,
        state="readonly",
    ).grid(row=0, column=3, padx=6, pady=6, sticky="w")
    ttk.Label(param_box, text="repeat count").grid(row=0, column=4, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(param_box, width=8, textvariable=presenter.validation_debug_repeat_count_var).grid(row=0, column=5, padx=6, pady=6, sticky="w")

    ttk.Checkbutton(
        param_box,
        text="reclamp between repeats",
        variable=presenter.validation_debug_reclamp_between_repeats_var,
    ).grid(row=1, column=0, columnspan=2, padx=(10, 2), pady=6, sticky="w")
    ttk.Checkbutton(
        param_box,
        text="reclamp enabled",
        variable=presenter.validation_debug_reclamp_enabled_var,
    ).grid(row=1, column=2, columnspan=2, padx=(10, 2), pady=6, sticky="w")
    ttk.Checkbutton(
        param_box,
        text="stop rotation before measure",
        variable=presenter.validation_debug_rotation_stop_before_measure_var,
    ).grid(row=1, column=4, columnspan=2, padx=(10, 2), pady=6, sticky="w")

    ttk.Label(param_box, text="release settle (s)").grid(row=2, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(param_box, width=8, textvariable=presenter.validation_debug_release_settle_s_var).grid(row=2, column=1, padx=6, pady=6, sticky="w")
    ttk.Label(param_box, text="clamp settle (s)").grid(row=2, column=2, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(param_box, width=8, textvariable=presenter.validation_debug_clamp_settle_s_var).grid(row=2, column=3, padx=6, pady=6, sticky="w")
    ttk.Label(param_box, text="position settle (s)").grid(row=2, column=4, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(param_box, width=8, textvariable=presenter.validation_debug_position_settle_s_var).grid(row=2, column=5, padx=6, pady=6, sticky="w")

    ttk.Label(param_box, text="sample delay (s)").grid(row=3, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(param_box, width=8, textvariable=presenter.validation_debug_sample_delay_s_var).grid(row=3, column=1, padx=6, pady=6, sticky="w")
    ttk.Label(param_box, text="AX3 speed (deg/s)").grid(row=3, column=2, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(param_box, width=8, textvariable=presenter.validation_debug_ax3_speed_dps_var).grid(row=3, column=3, padx=6, pady=6, sticky="w")
    ttk.Checkbutton(
        param_box,
        text="enable section move",
        variable=presenter.validation_debug_move_enabled_var,
    ).grid(row=3, column=4, columnspan=2, padx=(10, 2), pady=6, sticky="w")

    ttk.Label(param_box, text="move channel").grid(row=4, column=0, padx=(10, 2), pady=6, sticky="e")
    ttk.Combobox(
        param_box,
        width=16,
        textvariable=presenter.validation_debug_move_channel_var,
        values=VALIDATION_MOVE_CHANNELS,
        state="readonly",
    ).grid(row=4, column=1, padx=6, pady=6, sticky="w")
    ttk.Label(param_box, text="move scenario").grid(row=4, column=2, padx=(10, 2), pady=6, sticky="e")
    ttk.Combobox(
        param_box,
        width=22,
        textvariable=presenter.validation_debug_move_scenario_var,
        values=VALIDATION_MOVE_SCENARIOS,
        state="readonly",
    ).grid(row=4, column=3, padx=6, pady=6, sticky="w")
    ttk.Label(param_box, text="move away delta (mm)").grid(row=4, column=4, padx=(10, 2), pady=6, sticky="e")
    ttk.Entry(param_box, width=8, textvariable=presenter.validation_debug_move_away_delta_mm_var).grid(row=4, column=5, padx=6, pady=6, sticky="w")

    ttk.Label(param_box, text="from section").grid(row=5, column=0, padx=(10, 2), pady=6, sticky="e")
    move_from_combo.grid(row=5, column=1, padx=6, pady=6, sticky="w")
    ttk.Label(param_box, text="target section").grid(row=5, column=2, padx=(10, 2), pady=6, sticky="e")
    move_target_combo.grid(row=5, column=3, padx=6, pady=6, sticky="w")
    ttk.Label(param_box, text="return section").grid(row=5, column=4, padx=(10, 2), pady=6, sticky="e")
    move_return_combo.grid(row=5, column=5, padx=6, pady=6, sticky="w")

    run_box = ttk.LabelFrame(content, text="Run Control")
    run_box.pack(fill=tk.X, pady=(4, 8))
    start_btn = presenter.remember_widget(
        "validation_screen_start_btn",
        ttk.Button(
            run_box,
            text="Start",
            command=lambda: controller.start_validation_run(
                section_name=presenter.validation_debug_section_name_var.get(),
                metric_name=presenter.validation_debug_metric_name_var.get(),
                repeat_count=presenter.validation_debug_repeat_count_var.get(),
                reclamp_between_repeats=presenter.validation_debug_reclamp_between_repeats_var.get(),
                reclamp_enabled=presenter.validation_debug_reclamp_enabled_var.get(),
                rotation_stop_before_measure=presenter.validation_debug_rotation_stop_before_measure_var.get(),
                release_settle_s=presenter.validation_debug_release_settle_s_var.get(),
                clamp_settle_s=presenter.validation_debug_clamp_settle_s_var.get(),
                position_settle_s=presenter.validation_debug_position_settle_s_var.get(),
                sample_delay_s=presenter.validation_debug_sample_delay_s_var.get(),
                validation_ax3_speed_dps=presenter.validation_debug_ax3_speed_dps_var.get(),
                move_enabled=presenter.validation_debug_move_enabled_var.get(),
                move_channel=presenter.validation_debug_move_channel_var.get(),
                move_away_delta_mm=presenter.validation_debug_move_away_delta_mm_var.get(),
                move_scenario=presenter.validation_debug_move_scenario_var.get(),
                move_from_section_index=presenter.validation_debug_move_from_section_var.get(),
                move_target_section_index=presenter.validation_debug_move_target_section_var.get(),
                move_return_section_index=presenter.validation_debug_move_return_section_var.get(),
            ),
        ),
    )
    start_btn.grid(row=0, column=0, padx=(10, 6), pady=8, sticky="w")
    presenter.remember_widget(
        "validation_screen_stop_btn",
        ttk.Button(
            run_box,
            text="Stop",
            command=controller.stop_validation_run,
            state="disabled",
        ),
    ).grid(row=0, column=1, padx=6, pady=8, sticky="w")
    ttk.Label(
        run_box,
        text="Opening this page does not start validation. The workflow starts only after Start is pressed.",
        wraplength=800,
        justify="left",
    ).grid(row=0, column=2, padx=(16, 10), pady=8, sticky="w")

    status_box = ttk.LabelFrame(content, text="Status")
    status_box.pack(fill=tk.X, pady=(4, 8))
    ttk.Label(status_box, text="module").grid(row=0, column=0, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(status_box, text="validation", width=16).grid(row=0, column=1, padx=6, pady=4, sticky="w")
    ttk.Label(status_box, text="status").grid(row=0, column=2, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(status_box, textvariable=presenter.validation_debug_status_var, width=18).grid(row=0, column=3, padx=6, pady=4, sticky="w")
    ttk.Label(status_box, text="phase").grid(row=0, column=4, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(status_box, textvariable=presenter.validation_debug_phase_var, width=18).grid(row=0, column=5, padx=6, pady=4, sticky="w")
    ttk.Label(status_box, text="current repeat").grid(row=1, column=0, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(status_box, textvariable=presenter.validation_debug_current_repeat_var, width=18).grid(row=1, column=1, padx=6, pady=4, sticky="w")
    ttk.Label(status_box, text="wait phase").grid(row=1, column=2, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(status_box, textvariable=presenter.validation_debug_wait_phase_var, width=18).grid(row=1, column=3, padx=6, pady=4, sticky="w")
    ttk.Label(status_box, text="wait remaining").grid(row=1, column=4, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(status_box, textvariable=presenter.validation_debug_wait_remaining_s_var, width=18).grid(row=1, column=5, padx=6, pady=4, sticky="w")
    ttk.Label(status_box, text="error").grid(row=2, column=0, padx=(10, 2), pady=4, sticky="ne")
    ttk.Label(
        status_box,
        textvariable=presenter.validation_debug_error_var,
        foreground="red",
        wraplength=900,
        justify="left",
    ).grid(row=2, column=1, columnspan=5, padx=6, pady=4, sticky="w")

    current_box = ttk.LabelFrame(content, text="Current Repeat")
    current_box.pack(fill=tk.X, pady=(4, 8))
    ttk.Label(current_box, text="repeat").grid(row=0, column=0, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(current_box, textvariable=presenter.validation_debug_current_repeat_var, width=18).grid(row=0, column=1, padx=6, pady=4, sticky="w")
    ttk.Label(current_box, text="metric").grid(row=0, column=2, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(current_box, textvariable=presenter.validation_debug_metric_name_var, width=18).grid(row=0, column=3, padx=6, pady=4, sticky="w")
    ttk.Label(current_box, text="metric value").grid(row=1, column=0, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(current_box, textvariable=presenter.validation_current_metric_value_var, width=18).grid(row=1, column=1, padx=6, pady=4, sticky="w")
    ttk.Label(current_box, text="section").grid(row=1, column=2, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(current_box, textvariable=presenter.validation_current_section_var, width=24).grid(row=1, column=3, padx=6, pady=4, sticky="w")
    ttk.Label(current_box, text="z position (mm)").grid(row=2, column=0, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(current_box, textvariable=presenter.validation_current_z_pos_var, width=18).grid(row=2, column=1, padx=6, pady=4, sticky="w")
    ttk.Label(current_box, text="concentricity (mm)").grid(row=2, column=2, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(current_box, textvariable=presenter.validation_current_concentricity_var, width=18).grid(row=2, column=3, padx=6, pady=4, sticky="w")

    running_box = ttk.LabelFrame(content, text="Running Summary")
    running_box.pack(fill=tk.X, pady=(4, 8))
    ttk.Label(running_box, text="count").grid(row=0, column=0, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(running_box, textvariable=presenter.validation_summary_count_var, width=12).grid(row=0, column=1, padx=6, pady=4, sticky="w")
    ttk.Label(running_box, text="mean").grid(row=0, column=2, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(running_box, textvariable=presenter.validation_summary_mean_var, width=16).grid(row=0, column=3, padx=6, pady=4, sticky="w")
    ttk.Label(running_box, text="std").grid(row=0, column=4, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(running_box, textvariable=presenter.validation_summary_std_var, width=16).grid(row=0, column=5, padx=6, pady=4, sticky="w")
    ttk.Label(running_box, text="min").grid(row=1, column=0, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(running_box, textvariable=presenter.validation_summary_min_var, width=16).grid(row=1, column=1, padx=6, pady=4, sticky="w")
    ttk.Label(running_box, text="max").grid(row=1, column=2, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(running_box, textvariable=presenter.validation_summary_max_var, width=16).grid(row=1, column=3, padx=6, pady=4, sticky="w")
    ttk.Label(running_box, text="range").grid(row=1, column=4, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(running_box, textvariable=presenter.validation_summary_range_var, width=16).grid(row=1, column=5, padx=6, pady=4, sticky="w")

    result_box = ttk.LabelFrame(content, text="Final Summary / Export")
    result_box.pack(fill=tk.X, pady=(4, 8))
    ttk.Label(result_box, text="final summary").grid(row=0, column=0, padx=(10, 2), pady=4, sticky="ne")
    ttk.Label(
        result_box,
        textvariable=presenter.validation_debug_result_var,
        wraplength=900,
        justify="left",
    ).grid(row=0, column=1, columnspan=3, padx=6, pady=4, sticky="w")
    ttk.Label(result_box, text="export path").grid(row=1, column=0, padx=(10, 2), pady=4, sticky="ne")
    ttk.Label(
        result_box,
        textvariable=presenter.validation_debug_export_path_var,
        wraplength=900,
        justify="left",
    ).grid(row=1, column=1, columnspan=3, padx=6, pady=4, sticky="w")
    ttk.Label(result_box, text="target position").grid(row=2, column=0, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(result_box, textvariable=presenter.validation_debug_move_target_pos_var, width=24).grid(row=2, column=1, padx=6, pady=4, sticky="w")
    ttk.Label(result_box, text="actual position").grid(row=2, column=2, padx=(10, 2), pady=4, sticky="e")
    ttk.Label(result_box, textvariable=presenter.validation_debug_move_actual_pos_var, width=24).grid(row=2, column=3, padx=6, pady=4, sticky="w")

    _refresh_validation_section_combos()


__all__ = ["build_validation_screen"]
