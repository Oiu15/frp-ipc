from __future__ import annotations

import tkinter as tk
from typing import Any


class GaugeScreenPresenter:
    """Own gauge-screen UI state and translate UI events into controller intents."""

    def __init__(self, host: Any, controller: Any) -> None:
        object.__setattr__(self, 'host', host)
        object.__setattr__(self, 'controller', controller)
        object.__setattr__(self, '_owned_attrs', {})
        object.__setattr__(self, '_widgets', {})

    def _remember(self, name: str, value: Any) -> Any:
        owned = object.__getattribute__(self, '_owned_attrs')
        owned[name] = value
        setattr(self.host, name, value)
        return value

    def remember_widget(self, name: str, widget: Any) -> Any:
        object.__getattribute__(self, '_widgets')[name] = widget
        return widget

    def widget(self, name: str) -> Any:
        return object.__getattribute__(self, '_widgets').get(name)

    def _ensure_var(self, name: str, factory) -> tk.Variable:
        owned = object.__getattribute__(self, '_owned_attrs')
        if name in owned:
            return owned[name]
        try:
            existing = getattr(self.host, name)
        except AttributeError:
            existing = None
        if isinstance(existing, tk.Variable):
            owned[name] = existing
            return existing
        return self._remember(name, factory())

    def ensure_vars(self, master: tk.Misc) -> None:
        self._ensure_var('sim_gauge_var', lambda: tk.IntVar(master=master, value=int(bool(getattr(self.host, 'sim_gauge_enabled', False)))))
        self._ensure_var('baud_var', lambda: tk.StringVar(master=master, value='115200'))
        self._ensure_var('req_cmd_var', lambda: tk.StringVar(master=master, value='M1,1'))
        self._ensure_var('odcal_out2_hint_var', lambda: tk.StringVar(master=master, value='OUT2→R'))
        self._ensure_var('odcal_duration_label_var', lambda: tk.StringVar(master=master, value='时长(s)'))
        self._ensure_var('odcal_adv_open_var', lambda: tk.BooleanVar(master=master, value=False))
        self._ensure_var('validation_debug_section_name_var', lambda: tk.StringVar(master=master, value=''))
        self._ensure_var('validation_debug_metric_name_var', lambda: tk.StringVar(master=master, value='od_avg'))
        self._ensure_var('validation_debug_repeat_count_var', lambda: tk.StringVar(master=master, value='3'))
        self._ensure_var('validation_debug_reclamp_between_repeats_var', lambda: tk.BooleanVar(master=master, value=False))
        self._ensure_var('validation_debug_reclamp_enabled_var', lambda: tk.BooleanVar(master=master, value=False))
        self._ensure_var('validation_debug_rotation_stop_before_measure_var', lambda: tk.BooleanVar(master=master, value=False))
        self._ensure_var('validation_debug_release_settle_s_var', lambda: tk.StringVar(master=master, value='0.0'))
        self._ensure_var('validation_debug_clamp_settle_s_var', lambda: tk.StringVar(master=master, value='0.0'))
        self._ensure_var('validation_debug_status_var', lambda: tk.StringVar(master=master, value='IDLE'))
        self._ensure_var('validation_debug_phase_var', lambda: tk.StringVar(master=master, value='IDLE'))
        self._ensure_var('validation_debug_result_var', lambda: tk.StringVar(master=master, value=''))
        self._ensure_var('validation_debug_error_var', lambda: tk.StringVar(master=master, value=''))
        self._ensure_var('validation_debug_export_path_var', lambda: tk.StringVar(master=master, value=''))
        self.refresh_out2_hint()
        self.refresh_odcal_duration_label()

    def __getattr__(self, name: str) -> Any:
        owned = object.__getattribute__(self, '_owned_attrs')
        if name in owned:
            return owned[name]
        widgets = object.__getattribute__(self, '_widgets')
        if name in widgets:
            return widgets[name]
        attr = getattr(self.host, name)
        if callable(attr) and not (name.startswith('_refresh') or name.startswith('_list')):
            raise AttributeError(name)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {'host', 'controller', '_owned_attrs', '_widgets'}:
            object.__setattr__(self, name, value)
            return
        self._remember(name, value)

    def list_serial_ports(self) -> Any:
        fn = getattr(self.host, '_list_serial_ports', None)
        if callable(fn):
            return fn()
        return []

    def handle_request_command_changed(self, cmd: str) -> Any:
        norm = str(cmd or 'M1,1').strip() or 'M1,1'
        fn = getattr(self.controller, 'set_gauge_request_command', None)
        if callable(fn):
            return fn(norm)
        return None

    def refresh_out2_hint(self) -> None:
        try:
            out1 = (self.host.odcal_map_out1_var.get() or 'L').strip().upper()
        except Exception:
            out1 = 'L'
        out2 = 'R' if out1 == 'L' else 'L'
        self.odcal_out2_hint_var.set(f'OUT2→{out2}')

    def refresh_odcal_duration_label(self) -> None:
        try:
            mode = (self.host.odcal_mode_var.get() or 'timed').strip()
        except Exception:
            mode = 'timed'
        self.odcal_duration_label_var.set('超时(s)' if mode == 'one_rev' else '时长(s)')

    def handle_odcal_angle_source_changed(self) -> None:
        try:
            angle_src = str(self.host.odcal_angle_src_var.get() or 'AX3')
            mode = str(self.host.odcal_mode_var.get() or 'timed')
        except Exception:
            return
        if ('无' in angle_src) and mode == 'one_rev':
            self.host.odcal_mode_var.set('timed')
            self.refresh_odcal_duration_label()

    def toggle_odcal_advanced(self, button: Any, frame: Any) -> None:
        is_open = bool(self.odcal_adv_open_var.get())
        self.odcal_adv_open_var.set(not is_open)
        if self.odcal_adv_open_var.get():
            button.configure(text='高级参数 ▾')
            frame.grid()
        else:
            button.configure(text='高级参数 ▸')
            frame.grid_remove()


__all__ = ['GaugeScreenPresenter']
