from __future__ import annotations

import tkinter as tk
from collections.abc import Iterable
from typing import Any, Protocol


class GaugeScreenViewPort(Protocol):
    def get_var(self, name: str) -> Any: ...
    def get_flag(self, name: str, default: bool = False) -> bool: ...
    def list_serial_ports(self) -> list[str]: ...
    def calibration_controller(self) -> Any: ...


class GaugeCommandPort(Protocol):
    def list_validation_section_choices(self) -> list[str]: ...
    def set_gauge_request_command(self, cmd: str) -> str: ...


class GaugeScreenHostView:
    def __init__(self, app: Any) -> None:
        self._app = app

    def get_var(self, name: str) -> Any:
        ui = getattr(self._app, "ui", None)
        existing = getattr(ui, name, None)
        if existing is not None:
            return existing
        return getattr(self._app, name)

    def get_flag(self, name: str, default: bool = False) -> bool:
        return bool(getattr(self._app, name, default))

    def list_serial_ports(self) -> list[str]:
        fn = getattr(self._app, "_list_serial_ports", None)
        if callable(fn):
            ports = fn()
            if isinstance(ports, Iterable):
                return [str(port) for port in ports]
        return []

    def calibration_controller(self) -> Any:
        return getattr(self._app, "calibration_controller")


class GaugeScreenPresenter:
    """Own gauge-screen UI state and translate UI events into controller intents."""

    _view: GaugeScreenViewPort
    controller: GaugeCommandPort

    def __init__(self, view: Any, controller: GaugeCommandPort) -> None:
        if all(hasattr(view, name) for name in ("get_var", "get_flag", "list_serial_ports")):
            resolved_view = view
        else:
            resolved_view = GaugeScreenHostView(view)
        object.__setattr__(self, '_view', resolved_view)
        object.__setattr__(self, 'controller', controller)
        object.__setattr__(self, '_owned_attrs', {})
        object.__setattr__(self, '_widgets', {})

    @property
    def calibration_controller(self) -> Any:
        provider = getattr(object.__getattribute__(self, '_view'), "calibration_controller", None)
        if callable(provider):
            return provider()
        return object.__getattribute__(self, "controller")

    def _remember(self, name: str, value: Any) -> Any:
        owned = object.__getattribute__(self, '_owned_attrs')
        owned[name] = value
        return value

    def remember_widget(self, name: str, widget: Any) -> Any:
        object.__getattribute__(self, '_widgets')[name] = widget
        return widget

    def widget(self, name: str) -> Any:
        return object.__getattribute__(self, '_widgets').get(name)

    def get_var(self, name: str) -> Any:
        owned = object.__getattribute__(self, '_owned_attrs')
        if name in owned:
            return owned[name]
        return object.__getattribute__(self, '_view').get_var(name)

    def get_flag(self, name: str, default: bool = False) -> bool:
        return bool(object.__getattribute__(self, '_view').get_flag(name, default))

    def _ensure_var(self, name: str, factory) -> tk.Variable:
        owned = object.__getattribute__(self, '_owned_attrs')
        if name in owned:
            return owned[name]
        try:
            existing = object.__getattribute__(self, '_view').get_var(name)
        except AttributeError:
            existing = None
        if isinstance(existing, tk.Variable):
            owned[name] = existing
            return existing
        return self._remember(name, factory())

    def _ui_state_var(self, name: str) -> tk.Variable | None:
        try:
            existing = object.__getattribute__(self, '_view').get_var(name)
        except AttributeError:
            existing = None
        if isinstance(existing, tk.Variable):
            object.__getattribute__(self, "_owned_attrs")[name] = existing
            return existing
        return None

    def _ensure_ui_var(self, name: str, factory) -> tk.Variable:
        existing = self._ui_state_var(name)
        if existing is not None:
            return existing
        return self._ensure_var(name, factory)

    def _ensure_shared_var(self, canonical_name: str, alias_name: str, factory) -> tk.Variable:
        owned = object.__getattribute__(self, '_owned_attrs')
        shared = None
        for name in (canonical_name, alias_name):
            existing = owned.get(name)
            if isinstance(existing, tk.Variable):
                shared = existing
                break
            try:
                existing = object.__getattribute__(self, '_view').get_var(name)
            except AttributeError:
                existing = None
            if isinstance(existing, tk.Variable):
                shared = existing
                break
        if shared is None:
            shared = factory()
        self._remember(canonical_name, shared)
        self._remember(alias_name, shared)
        return shared

    def _ensure_shared_ui_var(self, canonical_name: str, alias_name: str, factory) -> tk.Variable:
        canonical = self._ui_state_var(canonical_name)
        alias = self._ui_state_var(alias_name)
        if canonical is not None and alias is not None:
            return canonical
        return self._ensure_shared_var(canonical_name, alias_name, factory)

    def ensure_vars(self, master: tk.Misc) -> None:
        self._ensure_ui_var('sim_gauge_var', lambda: tk.IntVar(master=master, value=int(bool(object.__getattribute__(self, '_view').get_flag('sim_gauge_enabled', False)))))
        self._ensure_ui_var('baud_var', lambda: tk.StringVar(master=master, value='115200'))
        self._ensure_ui_var('req_cmd_var', lambda: tk.StringVar(master=master, value='M1,1'))
        self._ensure_ui_var('gauge_conn_var', lambda: tk.StringVar(master=master, value='未连接'))
        self._ensure_ui_var('gauge_last_var', lambda: tk.StringVar(master=master, value='Gauge: --'))
        self._ensure_ui_var('gauge_err_var', lambda: tk.StringVar(master=master, value=''))
        self._ensure_ui_var('odcal_out2_hint_var', lambda: tk.StringVar(master=master, value='OUT2→R'))
        self._ensure_ui_var('odcal_duration_label_var', lambda: tk.StringVar(master=master, value='时长(s)'))
        self._ensure_ui_var('odcal_adv_open_var', lambda: tk.BooleanVar(master=master, value=False))
        self._ensure_ui_var('odcal_cmd_var', lambda: tk.StringVar(master=master, value='M0,1'))
        self._ensure_ui_var('odcal_dref_var', lambda: tk.StringVar(master=master, value='180.000'))
        self._ensure_ui_var('odcal_map_out1_var', lambda: tk.StringVar(master=master, value='L'))
        self._ensure_ui_var('odcal_mode_var', lambda: tk.StringVar(master=master, value='timed'))
        self._ensure_ui_var('odcal_hz_var', lambda: tk.StringVar(master=master, value='20'))
        self._ensure_ui_var('odcal_duration_var', lambda: tk.StringVar(master=master, value='10'))
        self._ensure_ui_var('odcal_rot_degps_var', lambda: tk.StringVar(master=master, value='10'))
        self._ensure_ui_var('odcal_angle_src_var', lambda: tk.StringVar(master=master, value='AX3'))
        self._ensure_ui_var('odcal_filter_var', lambda: tk.StringVar(master=master, value='无'))
        self._ensure_ui_var('odcal_outlier_sigma_var', lambda: tk.StringVar(master=master, value='3.0'))
        self._ensure_ui_var('odcal_defect_dyn_enable_var', lambda: tk.IntVar(master=master, value=1))
        self._ensure_ui_var('odcal_state_var', lambda: tk.StringVar(master=master, value='IDLE'))
        self._ensure_ui_var('odcal_msg_var', lambda: tk.StringVar(master=master, value='-'))
        self._ensure_ui_var('odcal_defect_mode_var', lambda: tk.StringVar(master=master, value='OFF'))
        self._ensure_ui_var('odcal_defect_shift_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('odcal_defects_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('odcal_B_candidate_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('odcal_B_active_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('odcal_n_var', lambda: tk.StringVar(master=master, value='0'))
        self._ensure_ui_var('odcal_elapsed_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('odcal_sum_mean_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('odcal_sum_std_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('odcal_sum_min_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('odcal_sum_max_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('odcal_drop_rate_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('idcal_state_var', lambda: tk.StringVar(master=master, value='IDLE'))
        self._ensure_ui_var('idcal_msg_var', lambda: tk.StringVar(master=master, value='-'))
        self._ensure_ui_var('idcal_dref_var', lambda: tk.StringVar(master=master, value='150.000'))
        self._ensure_ui_var('idcal_mode_var', lambda: tk.StringVar(master=master, value='one_rev'))
        self._ensure_ui_var('idcal_hz_var', lambda: tk.StringVar(master=master, value='20'))
        self._ensure_ui_var('idcal_duration_var', lambda: tk.StringVar(master=master, value='10'))
        self._ensure_ui_var('idcal_rot_degps_var', lambda: tk.StringVar(master=master, value='10'))
        self._ensure_ui_var('idcal_delta_candidate_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('idcal_delta_active_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('idcal_cmax_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('idcal_mmean_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('idcal_mpp_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('idcal_fit_diam_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('idcal_fit_e_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('idcal_fit_y0_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('idcal_fit_rmse_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('idcal_chk_err_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('idcal_chk_cov_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('idcal_chk_n_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('idcal_chk_dtheta_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('id_single_cal_state_var', lambda: tk.StringVar(master=master, value='IDLE'))
        self._ensure_ui_var('id_single_cal_msg_var', lambda: tk.StringVar(master=master, value='-'))
        self._ensure_ui_var('id_single_cal_dref_var', lambda: tk.StringVar(master=master, value='150.000'))
        self._ensure_ui_var('id_single_cal_mean_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('id_single_cal_B_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('id_single_cal_ecc_amp_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('id_single_cal_ecc_ang_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('id_single_cal_cov_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('id_single_cal_warn_var', lambda: tk.StringVar(master=master, value=''))
        # geometry_v2 tooling calibration (几何标定 V2 页)
        self._ensure_ui_var('tcal_status_var', lambda: tk.StringVar(master=master, value='未标定'))
        self._ensure_ui_var('tcal_msg_var', lambda: tk.StringVar(master=master, value='-'))
        self._ensure_ui_var('tcal_r_known_var', lambda: tk.StringVar(master=master, value='76.350'))
        self._ensure_ui_var('tcal_d_init_var', lambda: tk.StringVar(master=master, value='140.000'))
        self._ensure_ui_var('tcal_rot_degps_var', lambda: tk.StringVar(master=master, value='10'))
        self._ensure_ui_var('tcal_hz_var', lambda: tk.StringVar(master=master, value='20'))
        self._ensure_ui_var('tcal_id_nsets_var', lambda: tk.StringVar(master=master, value='0'))
        self._ensure_ui_var('tcal_id_s_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('tcal_id_axis_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('tcal_id_q_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('tcal_id_cost_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('tcal_od_psi_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('tcal_selftest_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('tcal_id_Deff_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('tcal_id_s_active_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('tcal_od_psi_active_var', lambda: tk.StringVar(master=master, value='--'))
        # 5c: OD zero / axis / chuck / delta_reg
        self._ensure_ui_var('tcal_known_od_var', lambda: tk.StringVar(master=master, value='190.000'))
        self._ensure_ui_var('tcal_od_b_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('tcal_od_b_active_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('tcal_axis_z_low_var', lambda: tk.StringVar(master=master, value='0.0'))
        self._ensure_ui_var('tcal_axis_z_high_var', lambda: tk.StringVar(master=master, value='1700.0'))
        self._ensure_ui_var('tcal_axis_slope_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('tcal_axis_active_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('tcal_cert_round_var', lambda: tk.StringVar(master=master, value='0.009'))
        self._ensure_ui_var('tcal_chuck_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('tcal_chuck_active_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('tcal_delta_reg_var', lambda: tk.StringVar(master=master, value='--'))
        self._ensure_ui_var('tcal_delta_active_var2', lambda: tk.StringVar(master=master, value='--'))
        validation_var_specs = (
            ('validation_section_name_var', 'validation_debug_section_name_var', lambda: tk.StringVar(master=master, value='')),
            ('validation_metric_name_var', 'validation_debug_metric_name_var', lambda: tk.StringVar(master=master, value='od_avg')),
            ('validation_repeat_count_var', 'validation_debug_repeat_count_var', lambda: tk.StringVar(master=master, value='3')),
            ('validation_reclamp_between_repeats_var', 'validation_debug_reclamp_between_repeats_var', lambda: tk.BooleanVar(master=master, value=False)),
            ('validation_reclamp_enabled_var', 'validation_debug_reclamp_enabled_var', lambda: tk.BooleanVar(master=master, value=False)),
            ('validation_rotation_stop_before_measure_var', 'validation_debug_rotation_stop_before_measure_var', lambda: tk.BooleanVar(master=master, value=False)),
            ('validation_release_settle_s_var', 'validation_debug_release_settle_s_var', lambda: tk.StringVar(master=master, value='0.0')),
            ('validation_clamp_settle_s_var', 'validation_debug_clamp_settle_s_var', lambda: tk.StringVar(master=master, value='0.0')),
            ('validation_position_settle_s_var', 'validation_debug_position_settle_s_var', lambda: tk.StringVar(master=master, value='0.0')),
            ('validation_sample_delay_s_var', 'validation_debug_sample_delay_s_var', lambda: tk.StringVar(master=master, value='0.0')),
            ('validation_ax3_speed_dps_var', 'validation_debug_ax3_speed_dps_var', lambda: tk.StringVar(master=master, value='60.0')),
            ('validation_move_enabled_var', 'validation_debug_move_enabled_var', lambda: tk.BooleanVar(master=master, value=False)),
            ('validation_move_channel_var', 'validation_debug_move_channel_var', lambda: tk.StringVar(master=master, value='od_channel')),
            ('validation_move_away_delta_mm_var', 'validation_debug_move_away_delta_mm_var', lambda: tk.StringVar(master=master, value='0.0')),
            ('validation_move_scenario_var', 'validation_debug_move_scenario_var', lambda: tk.StringVar(master=master, value='distance_round_trip')),
            ('validation_move_from_section_var', 'validation_debug_move_from_section_var', lambda: tk.StringVar(master=master, value='1')),
            ('validation_move_target_section_var', 'validation_debug_move_target_section_var', lambda: tk.StringVar(master=master, value='1')),
            ('validation_move_return_section_var', 'validation_debug_move_return_section_var', lambda: tk.StringVar(master=master, value='1')),
            ('validation_move_target_pos_var', 'validation_debug_move_target_pos_var', lambda: tk.StringVar(master=master, value='')),
            ('validation_move_actual_pos_var', 'validation_debug_move_actual_pos_var', lambda: tk.StringVar(master=master, value='')),
            ('validation_status_var', 'validation_debug_status_var', lambda: tk.StringVar(master=master, value='IDLE')),
            ('validation_phase_var', 'validation_debug_phase_var', lambda: tk.StringVar(master=master, value='IDLE')),
            ('validation_wait_phase_var', 'validation_debug_wait_phase_var', lambda: tk.StringVar(master=master, value='')),
            ('validation_wait_remaining_s_var', 'validation_debug_wait_remaining_s_var', lambda: tk.StringVar(master=master, value='')),
            ('validation_current_repeat_var', 'validation_debug_current_repeat_var', lambda: tk.StringVar(master=master, value='0/0')),
            ('validation_result_var', 'validation_debug_result_var', lambda: tk.StringVar(master=master, value='')),
            ('validation_error_var', 'validation_debug_error_var', lambda: tk.StringVar(master=master, value='')),
            ('validation_export_path_var', 'validation_debug_export_path_var', lambda: tk.StringVar(master=master, value='')),
        )
        for canonical_name, alias_name, factory in validation_var_specs:
            self._ensure_shared_ui_var(canonical_name, alias_name, factory)
        self._ensure_ui_var('validation_current_metric_value_var', lambda: tk.StringVar(master=master, value=''))
        self._ensure_ui_var('validation_current_section_var', lambda: tk.StringVar(master=master, value=''))
        self._ensure_ui_var('validation_current_z_pos_var', lambda: tk.StringVar(master=master, value=''))
        self._ensure_ui_var('validation_current_concentricity_var', lambda: tk.StringVar(master=master, value=''))
        self._ensure_ui_var('validation_summary_count_var', lambda: tk.StringVar(master=master, value='0'))
        self._ensure_ui_var('validation_summary_mean_var', lambda: tk.StringVar(master=master, value=''))
        self._ensure_ui_var('validation_summary_std_var', lambda: tk.StringVar(master=master, value=''))
        self._ensure_ui_var('validation_summary_min_var', lambda: tk.StringVar(master=master, value=''))
        self._ensure_ui_var('validation_summary_max_var', lambda: tk.StringVar(master=master, value=''))
        self._ensure_ui_var('validation_summary_range_var', lambda: tk.StringVar(master=master, value=''))
        self.refresh_out2_hint()
        self.refresh_odcal_duration_label()

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {'_view', 'controller', '_owned_attrs', '_widgets'}:
            object.__setattr__(self, name, value)
            return
        self._remember(name, value)

    def validation_section_choices(self) -> list[str]:
        try:
            raw_values = self.controller.list_validation_section_choices()
            if not isinstance(raw_values, Iterable):
                return ['1']
            values = [str(value) for value in raw_values]
            if values:
                return values
        except Exception:
            pass
        return ['1']

    def list_serial_ports(self) -> Any:
        try:
            return object.__getattribute__(self, '_view').list_serial_ports()
        except Exception:
            return []

    def handle_request_command_changed(self, cmd: str) -> Any:
        norm = str(cmd or 'M1,1').strip() or 'M1,1'
        return self.controller.set_gauge_request_command(norm)

    def refresh_out2_hint(self) -> None:
        try:
            out1 = (self.get_var('odcal_map_out1_var').get() or 'L').strip().upper()
        except Exception:
            out1 = 'L'
        out2 = 'R' if out1 == 'L' else 'L'
        self.get_var('odcal_out2_hint_var').set(f'OUT2→{out2}')

    def refresh_odcal_duration_label(self) -> None:
        try:
            mode = (self.get_var('odcal_mode_var').get() or 'timed').strip()
        except Exception:
            mode = 'timed'
        self.get_var('odcal_duration_label_var').set('超时(s)' if mode == 'one_rev' else '时长(s)')

    def handle_odcal_angle_source_changed(self) -> None:
        try:
            angle_src = str(self.get_var('odcal_angle_src_var').get() or 'AX3')
            mode = str(self.get_var('odcal_mode_var').get() or 'timed')
        except Exception:
            return
        if ('无' in angle_src) and mode == 'one_rev':
            self.get_var('odcal_mode_var').set('timed')
            self.refresh_odcal_duration_label()

    def toggle_odcal_advanced(self, button: Any, frame: Any) -> None:
        is_open = bool(self.get_var('odcal_adv_open_var').get())
        self.get_var('odcal_adv_open_var').set(not is_open)
        if self.get_var('odcal_adv_open_var').get():
            button.configure(text='高级参数 ▾')
            frame.grid()
        else:
            button.configure(text='高级参数 ▸')
            frame.grid_remove()


__all__ = ['GaugeCommandPort', 'GaugeScreenHostView', 'GaugeScreenPresenter', 'GaugeScreenViewPort']
