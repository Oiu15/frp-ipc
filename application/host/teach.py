from __future__ import annotations

"""Teach-position and manual teach-motion mixin for AppHost."""

import time
from typing import TYPE_CHECKING, Any

from config.addresses import AXIS_COUNT, CMD_JOG_B_REQ, CMD_JOG_F_REQ
from core.models import AxisCal, AxisComm, Recipe
from domain.teach_planning import (
    TeachWarning,
    align_by_id_target,
    align_by_od_targets,
    center_position_z_disp,
    end_z_disp,
    relative_move_targets,
    save_section_plan,
    selected_section_targets,
    standby_alignment_plan,
    start_anchor_z_pos,
    teach_position_plan,
)
from utils.logger import log

if TYPE_CHECKING:
    from application.host.ports import TeachHost


class HostTeachMixin:
    """Mixin providing recipe-screen teach actions and teach position displays."""

    axis_cal: AxisCal
    recipe: Recipe
    axis_cal_vars: dict[str, Any]
    axis_cal_field_status_vars: dict[str, Any]
    teach_axes_mode_var: Any
    teach_rel_dist_var: Any
    center_pos_var: Any
    len_z_low_approach_var: Any
    start_info_var: Any
    standby_state_var: Any
    standby_info_var: Any
    teach_mode_var: Any
    teach_align_var: Any
    teach_abs_var: Any
    teach_z_var: Any
    teach_axes_var: Any

    if TYPE_CHECKING:
        def _recipe_ui_widget(self, name: str) -> Any: ...
        def _recipe_apply_from_ui(self) -> Recipe: ...
        def _get_selected_recipe_idx(self) -> int | None: ...
        def _ensure_recipe_section_plan(self, recipe: Recipe | None = None) -> Any: ...
        def _save_taught_section_to_recipe(self, recipe: Recipe, recipe_index: int, z_od_disp: float) -> None: ...
        def _refresh_recipe_table(self) -> None: ...
        def _refresh_length_info(self) -> None: ...
        def get_axis_copy(self, axis: int) -> AxisComm: ...
        def movea_abs(self, axis: int, pos_abs: float, *, context: str = "MoveA") -> None: ...
        def apply_soft_limits_abs(self, axis: int, target_abs: float, *, strict: bool = False, context: str = "") -> float: ...
        def _write_axis_params(self, axis: int) -> None: ...
        def set_cmd_bits(self, axis: int, set_mask: int = 0, clr_mask: int = 0) -> None: ...
        def after(self, ms: Any, func: Any | None = None, *args: Any) -> Any: ...
        def show_error(self, title: str, message: str) -> None: ...
        def show_info(self, title: str, message: str) -> None: ...
        def show_warning(self, title: str, message: str) -> None: ...
        def ask_ok_cancel(self, title: str, message: str) -> bool: ...

    def _on_teach_axes_selected(self, _evt=None):
        """Teach axes mode combobox changed.

        Modes:
          0=OD(AX0)
          1=ID(AX1+AX4)
          2=OD+ID(AX0+AX1+AX4)
          3=Center clamp(AX2)
        """
        combo = self._recipe_ui_widget('teach_axes_combo')
        try:
            i = int(combo.current()) if combo is not None else 2
        except Exception:
            i = 2
        i = max(0, min(3, int(i)))
        try:
            self.teach_axes_mode_var.set(i)
        except Exception:
            pass
        try:
            self.recipe.teach_axes_mode = i
        except Exception:
            pass
        try:
            self._refresh_teach_action_buttons()
        except Exception:
            pass

        self._refresh_teach_pos()



    def _refresh_teach_action_buttons(self) -> None:
        """Refresh teach action buttons according to current teach axis mode.

        - Section-based teach actions (move to selected / save selected) are enabled when
          teach axis is NOT AX2 (mode!=3). AX2 is the center frame, and its positioning is
          managed by dedicated "length/rotate position" controls.
        - Start/End quick moves are disabled when teach axis is AX2.
        """
        btn_move = self._recipe_ui_widget('teach_btn_move')
        btn_update = self._recipe_ui_widget('teach_btn_update')
        btn_goto_start = self._recipe_ui_widget('teach_btn_goto_start')
        btn_goto_end = self._recipe_ui_widget('teach_btn_goto_end')
        if btn_move is None or btn_update is None:
            return

        try:
            mode = int(getattr(self.recipe, "teach_axes_mode", 2))
        except Exception:
            mode = 2

        try:
            btn_move.configure(text="移动示教轴到选中截面", command=self._teach_move_to_selected)
            btn_update.configure(text="保存截面位置", command=self._teach_save_current_to_selected)
        except Exception:
            pass

        st = ("disabled" if mode == 3 else "normal")
        try:
            btn_move.configure(state=st)
            btn_update.configure(state=st)
        except Exception:
            pass

        try:
            st2 = ("disabled" if mode == 3 else "normal")
            if btn_goto_start is not None:
                btn_goto_start.configure(state=st2)
            if btn_goto_end is not None:
                btn_goto_end.configure(state=st2)
        except Exception:
            pass

        # 当示教轴不是 AX2 时启用“选中截面”相关按钮（AX2 时置灰）
        st = ("disabled" if mode == 3 else "normal")
        try:
            if btn_move is not None:
                btn_move.configure(state=st)
            if btn_update is not None:
                btn_update.configure(state=st)
        except Exception:
            pass

        # Start/End 快捷移动：示教轴为 AX2 时置灰
        try:
            st2 = ("disabled" if mode == 3 else "normal")
            if btn_goto_start is not None:
                btn_goto_start.configure(state=st2)
            if btn_goto_end is not None:
                btn_goto_end.configure(state=st2)
        except Exception:
            pass

    def _teach_move_ax2_to_len_pos(self) -> None:
        """Move AX2 to the saved 'length measurement' position."""
        try:
            if not bool(getattr(self.recipe, "ax2_len_valid", False)):
                self.show_warning("中心架位置", "长度测量位尚未设置：请先点击“保存为长度测量位”。")
                return
            a = float(getattr(self.recipe, "ax2_len_abs", 0.0))
            self.movea_abs(2, a)
        except Exception as e:
            self.show_error("中心架移动失败", str(e))

    def _teach_move_ax2_to_rot_pos(self) -> None:
        """Move AX2 to the saved 'rotation measurement' position."""
        try:
            if not bool(getattr(self.recipe, "ax2_rot_valid", False)):
                self.show_warning("中心架位置", "旋转测量位尚未设置：请先点击“保存为旋转测量位”。")
                return
            a = float(getattr(self.recipe, "ax2_rot_abs", 0.0))
            self.movea_abs(2, a)
        except Exception as e:
            self.show_error("中心架移动失败", str(e))
    def _teach_move_to_selected(self):
        try:
            r = self._recipe_apply_from_ui()
            idx = self._get_selected_recipe_idx()
            if idx is None:
                self.show_warning("提示", "请先在表格中选中一个截面")
                return

            mode = int(getattr(self.recipe, 'teach_axes_mode', getattr(r, 'teach_axes_mode', 2)))
            section_plan = self._ensure_recipe_section_plan(r)
            selected_row = section_plan.section_for_recipe_index(idx)

            targets = selected_section_targets(self.axis_cal, mode=mode, selected_row=selected_row)
            for axis, target_abs in targets.axis_items():
                self.movea_abs(axis, target_abs, context='SectionMove')
        except Exception as e:
            self.show_error("示教移动失败", str(e))

    def _teach_save_current_to_selected(self):
        try:
            r = self._recipe_apply_from_ui()
            idx = self._get_selected_recipe_idx()
            if idx is None:
                self.show_warning("提示", "请先在表格中选中一个截面")
                return

            mode = int(getattr(self.recipe, 'teach_axes_mode', getattr(r, 'teach_axes_mode', 2)))

            ac0 = self.get_axis_copy(0)
            ac1 = self.get_axis_copy(1)
            ac2 = self.get_axis_copy(2)
            ac4 = self.get_axis_copy(4)
            plan = save_section_plan(
                self.axis_cal,
                mode=mode,
                ax0_abs=float(ac0.act_pos),
                ax1_abs=float(ac1.act_pos),
                ax2_abs=float(ac2.act_pos),
                ax4_abs=float(ac4.act_pos),
            )
            if TeachWarning.OD_ID_NOT_ALIGNED in plan.warnings:
                try:
                    log("teach: OD/ID not aligned; saving section using OD")
                except Exception:
                    pass

            self._save_taught_section_to_recipe(r, idx, plan.z_od_disp)
            self.recipe = r

            self._refresh_recipe_table()
            self._refresh_teach_pos()
        except Exception as e:
            self.show_error("示教保存失败", str(e))


    def _teach_align_by_od(self):
        """Align ID plane to OD plane (keep AX0, move AX1/AX4)."""
        try:
            ac0 = self.get_axis_copy(0)
            ac1 = self.get_axis_copy(1)
            targets = align_by_od_targets(
                self.axis_cal,
                ax0_abs=float(ac0.act_pos),
                ax1_softlim_pos=getattr(ac1, 'softlim_pos', None),
                ax1_softlim_neg=getattr(ac1, 'softlim_neg', None),
            )
            if targets.ax1_abs is not None:
                self.apply_soft_limits_abs(1, float(targets.ax1_abs), strict=False, context='MoveA')
            if targets.ax4_abs is not None:
                self.apply_soft_limits_abs(4, float(targets.ax4_abs), strict=False, context='MoveA')
            for axis, target_abs in targets.axis_items():
                self.movea_abs(axis, target_abs)
            self._refresh_teach_pos()
        except Exception as e:
            self.show_error("对齐失败(OD基准)", str(e))

    def _teach_align_by_id(self):
        """Align OD plane to ID plane (keep AX1/AX4, move AX0)."""
        try:
            ac1 = self.get_axis_copy(1)
            ac4 = self.get_axis_copy(4)
            self.movea_abs(
                0,
                align_by_id_target(
                    self.axis_cal,
                    ax1_abs=float(ac1.act_pos),
                    ax4_abs=float(ac4.act_pos),
                ),
            )
            self._refresh_teach_pos()
        except Exception as e:
            self.show_error("对齐失败(ID基准)", str(e))

    

    # -------------------------
    # Start anchor (Start)
    # -------------------------
    def _apply_start_anchor_from_recipe(self) -> None:
        """Apply recipe.start_ax0_abs as Z_Pos=0 reference by updating AxisCal.z_pos.

        Notes:
            - AxisCal.z_pos is IPC-only shift (not written to PLC).
            - We define Z_Pos such that: Z_Pos = Z_Raw - z_pos.
              Therefore, to make Start at Z_Pos=0 we set z_pos = Z_Raw(start_abs).
        """
        try:
            r = getattr(self, 'recipe', None)
            if not r or not bool(getattr(r, 'start_valid', False)):
                self.axis_cal.z_pos = 0.0
                try:
                    self.axis_cal_vars['z_pos'].set(f"{self.axis_cal.z_pos:.6f}")
                    self.axis_cal_field_status_vars['z_pos'].set('无配方Start')
                except Exception:
                    pass
                self._refresh_start_pos()
                try:
                    self._refresh_teach_pos()
                except Exception:
                    pass
                return
            self.axis_cal.z_pos = start_anchor_z_pos(self.axis_cal, r)
            try:
                self.axis_cal_vars['z_pos'].set(f"{self.axis_cal.z_pos:.6f}")
                self.axis_cal_field_status_vars['z_pos'].set('配方Start')
            except Exception:
                pass

            # migrate legacy bottom-approach (Z_disp) to absolute AX0 act_pos (one-shot)
            try:
                legacy_z = getattr(self, "_len_low_appr_legacy_z", None)
                if legacy_z is not None:
                    abs_appr = float(self.axis_cal.z_disp_to_abs(0, float(legacy_z)))
                    setattr(r, "len_low_approach_abs", abs_appr)
                    if hasattr(self, "len_z_low_approach_var"):
                        self.len_z_low_approach_var.set(str(abs_appr))
                    self._len_low_appr_legacy_z = None
            except Exception:
                pass

            self._refresh_start_pos()
            try:
                self._refresh_teach_pos()
            except Exception:
                pass
        except Exception:
            # do not block UI
            try:
                self._refresh_start_pos()
            except Exception:
                pass

    def _teach_save_start(self) -> None:
        """Save current AX0 absolute position as measurement start (Start), and bind it to Z_Pos=0."""
        try:
            ac0 = self.get_axis_copy(0)
            self.recipe.start_valid = True
            self.recipe.start_ax0_abs = float(ac0.act_pos)
            self._apply_start_anchor_from_recipe()
            self._refresh_recipe_table()
            self._refresh_teach_pos()
            self.show_info('Start', '已保存测量区间起始位(Start)：Z_Pos=0')
        except Exception as e:
            self.show_error('Start保存失败', str(e))

    def _teach_start_from_standby(self) -> None:
        """Convenience: set Start from already-saved standby pose (AX0 only)."""
        try:
            if not bool(getattr(self.recipe, 'standby_valid', False)):
                self.show_warning('Start', '待定点尚未设置：请先保存待定点。')
                return
            self.recipe.start_valid = True
            self.recipe.start_ax0_abs = float(getattr(self.recipe, 'standby_ax0_abs', 0.0))
            self._apply_start_anchor_from_recipe()
            self._refresh_recipe_table()
            self._refresh_teach_pos()
            self.show_info('Start', '已从待定点同步设置Start：Z_Pos=0')
        except Exception as e:
            self.show_error('Start设置失败', str(e))

    def _teach_goto_start(self) -> None:
        """Move current teach axes to Start (Z_Pos=0).

        Note: Start anchor is defined by AX0 abs stored in recipe, and applied to AxisCal.z_pos.
        In this coordinate, Start corresponds to Z_disp=0.
        """
        try:
            mode = int(getattr(self.recipe, 'teach_axes_mode', 2))
            if mode == 3:
                # AX2: disabled by UI, but keep safe guard here
                return
            if not bool(getattr(self.recipe, 'start_valid', False)):
                self.show_warning('Start', 'Start尚未设置：请先点击“保存为测量区间起始位(Start)”。')
                return

            start_ax0_abs = float(getattr(self.recipe, 'start_ax0_abs', 0.0))
            self._apply_start_anchor_from_recipe()
            z_od_disp = 0.0
            softlims = {
                0: (float(self.get_axis_copy(0).softlim_pos), float(self.get_axis_copy(0).softlim_neg)),
                1: (float(self.get_axis_copy(1).softlim_pos), float(self.get_axis_copy(1).softlim_neg)),
                4: (float(self.get_axis_copy(4).softlim_pos), float(self.get_axis_copy(4).softlim_neg)),
            }
            t = self.axis_cal.od_z_disp_to_targets(z_od_disp, softlims_abs=softlims)
            if mode in (0, 2):
                self.movea_abs(0, start_ax0_abs, context='GotoStart')
            if mode in (1, 2):
                self.movea_abs(1, float(t['ax1_abs']), context='GotoStart')
                self.movea_abs(4, float(t['ax4_abs']), context='GotoStart')
        except Exception as e:
            self.show_error('移动Start失败', str(e))

    def _teach_goto_end(self) -> None:
        """Move current teach axes to End (Z_Pos = measurement total length).

        End is defined as:
          - meas_total_len_mm if > 0
          - else (pipe_len_mm - clamp_occupy_mm)
        (margins are not subtracted).
        """
        try:
            mode = int(getattr(self.recipe, 'teach_axes_mode', 2))
            if mode == 3:
                return
            if not bool(getattr(self.recipe, 'start_valid', False)):
                self.show_warning('End', 'Start尚未设置：请先保存Start，再移动到End。')
                return

            z_od_disp = end_z_disp(self.recipe)
            softlims = {
                0: (float(self.get_axis_copy(0).softlim_pos), float(self.get_axis_copy(0).softlim_neg)),
                1: (float(self.get_axis_copy(1).softlim_pos), float(self.get_axis_copy(1).softlim_neg)),
                4: (float(self.get_axis_copy(4).softlim_pos), float(self.get_axis_copy(4).softlim_neg)),
            }
            t = self.axis_cal.od_z_disp_to_targets(z_od_disp, softlims_abs=softlims)
            if mode in (0, 2):
                self.movea_abs(0, float(t['ax0_abs']), context='GotoEnd')
            if mode in (1, 2):
                self.movea_abs(1, float(t['ax1_abs']), context='GotoEnd')
                self.movea_abs(4, float(t['ax4_abs']), context='GotoEnd')
        except Exception as e:
            self.show_error('移动End失败', str(e))

    def _refresh_start_pos(self) -> None:
        """Refresh Start (measurement anchor) display on the teach page."""
        try:
            if not hasattr(self, 'start_info_var'):
                return
            if not bool(getattr(self.recipe, 'start_valid', False)):
                self.start_info_var.set('Start: 未设置')
                return
            a0 = float(getattr(self.recipe, 'start_ax0_abs', 0.0))
            z_raw = float(self.axis_cal.abs_to_z_raw(0, a0))
            self.start_info_var.set(f"Start: AX0 abs={a0:.3f} | Z_raw={z_raw:.3f} | Z_Pos=0")
        except Exception:
            pass

    # -------------------------
    # Standby point (待定点)
    # -------------------------
    def _teach_save_standby(self):
        """Capture current AX0/AX1/AX4 absolute positions as standby point and store into recipe."""
        try:
            ac0 = self.get_axis_copy(0)
            ac1 = self.get_axis_copy(1)
            ac4 = self.get_axis_copy(4)

            self.recipe.standby_valid = True
            self.recipe.standby_ax0_abs = float(ac0.act_pos)
            self.recipe.standby_ax1_abs = float(ac1.act_pos)
            self.recipe.standby_ax4_abs = float(ac4.act_pos)

            self._refresh_standby_pos()
            self.show_info("待定点", "已保存待定点（请记得保存配方 JSON）")
        except Exception as e:
            self.show_error("待定点保存失败", str(e))

    def _teach_go_standby(self):
        """Move AX0/AX1/AX4 to the stored standby point."""
        try:
            if not bool(getattr(self.recipe, "standby_valid", False)):
                self.show_warning("提示", "待定点尚未设置：请先点击“将当下位置保存为待定位”。")
                return

            a0 = float(getattr(self.recipe, "standby_ax0_abs", 0.0))
            a1 = float(getattr(self.recipe, "standby_ax1_abs", 0.0))
            a4 = float(getattr(self.recipe, "standby_ax4_abs", 0.0))

            # Fire 3 MoveA commands back-to-back (effectively simultaneous)
            self.movea_abs(0, a0)
            self.movea_abs(1, a1)
            self.movea_abs(4, a4)
        except Exception as e:
            self.show_error("回到待定点失败", str(e))

    def _refresh_standby_pos(self):
        """Refresh standby display fields on the teach page."""
        try:
            if not hasattr(self, "standby_info_var"):
                return
            if not bool(getattr(self.recipe, "standby_valid", False)):
                self.standby_state_var.set("未设置")
                self.standby_info_var.set("未设置")
                return

            a0 = float(getattr(self.recipe, "standby_ax0_abs", 0.0))
            a1 = float(getattr(self.recipe, "standby_ax1_abs", 0.0))
            a4 = float(getattr(self.recipe, "standby_ax4_abs", 0.0))

            plan = standby_alignment_plan(
                self.axis_cal,
                ax0_abs=a0,
                ax1_abs=a1,
                ax4_abs=a4,
            )

            self.standby_state_var.set("已设置" + ("（OD/ID对齐）" if plan.aligned else "（OD/ID未对齐）"))
            self.standby_info_var.set(
                "AX0 abs={:.3f}  Z_od={:.3f}\n"
                "AX1 abs={:.3f}  Z1_raw={:.3f}\n"
                "AX4 abs={:.3f}  Z4_raw={:.3f}\n"
                "ID_act={:.3f}  ID_exp={:.3f}  Δ={:.3f}".format(
                    a0,
                    plan.z_od_disp,
                    a1,
                    plan.z1_raw,
                    a4,
                    plan.z4_raw,
                    plan.z_id_disp,
                    plan.z_id_expected_disp,
                    plan.delta,
                )
            )
        except Exception:
            # do not crash UI
            pass


    # -------------------------
    # Center clamp (AX2) positions
    # -------------------------
    def _save_ax2_len_pos(self) -> None:
        """Save current AX2 absolute position as 'length measurement' position in recipe."""
        try:
            act2 = float(self.get_axis_copy(2).act_pos)
            self.recipe.ax2_len_valid = True
            self.recipe.ax2_len_abs = act2
            self._refresh_center_positions()
            self.show_info('中心架位置', '已保存：长度测量位')
        except Exception as e:
            self.show_error('中心架位置保存失败', str(e))

    def _save_ax2_rot_pos(self) -> None:
        """Save current AX2 absolute position as 'rotation measurement' position in recipe."""
        try:
            act2 = float(self.get_axis_copy(2).act_pos)
            self.recipe.ax2_rot_valid = True
            self.recipe.ax2_rot_abs = act2
            self._refresh_center_positions()
            self.show_info('中心架位置', '已保存：旋转测量位')
        except Exception as e:
            self.show_error('中心架位置保存失败', str(e))

    def _refresh_center_positions(self) -> None:
        """Refresh read-only display for AX2 saved positions on recipe screen."""
        if not hasattr(self, 'center_pos_var'):
            return
        cal = self.axis_cal
        lines = []

        if bool(getattr(self.recipe, 'ax2_len_valid', False)):
            a = float(getattr(self.recipe, 'ax2_len_abs', 0.0))
            z = center_position_z_disp(cal, ax2_abs=a)
            lines.append(f"长度测量位: abs={a:.3f}  Z_disp={z:.3f}")
        else:
            lines.append('长度测量位: 未设置')

        if bool(getattr(self.recipe, 'ax2_rot_valid', False)):
            a = float(getattr(self.recipe, 'ax2_rot_abs', 0.0))
            z = center_position_z_disp(cal, ax2_abs=a)
            lines.append(f"旋转测量位: abs={a:.3f}  Z_disp={z:.3f}")
        else:
            lines.append('旋转测量位: 未设置')

        self.center_pos_var.set('\n'.join(lines))


    def _teach_move_relative(self):
        """Relative move for selected teach axes in Z_disp (mm)."""
        try:
            try:
                dz = float(self.teach_rel_dist_var.get())
            except Exception:
                dz = 0.0

            mode = int(getattr(self.recipe, 'teach_axes_mode', 2))

            ac0 = self.get_axis_copy(0)
            ac1 = self.get_axis_copy(1)
            ac2 = self.get_axis_copy(2)
            ac4 = self.get_axis_copy(4)
            targets = relative_move_targets(
                self.axis_cal,
                mode=mode,
                dz=dz,
                ax0_abs=float(ac0.act_pos),
                ax1_abs=float(ac1.act_pos),
                ax2_abs=float(ac2.act_pos),
                ax4_abs=float(ac4.act_pos),
                ax1_softlim_pos=getattr(ac1, 'softlim_pos', None),
                ax1_softlim_neg=getattr(ac1, 'softlim_neg', None),
                ax4_softlim_pos=getattr(ac4, 'softlim_pos', None),
                ax4_softlim_neg=getattr(ac4, 'softlim_neg', None),
            )
            for axis, target_abs in targets.axis_items():
                self.movea_abs(axis, target_abs, context="TeachRel")

            self._refresh_teach_pos()
        except Exception as e:
            self.show_error("相对运动失败", str(e))

    def _teach_jog_hold(self, direction: str, on: bool):
        """Jog for teach panel (press-and-hold).

        - OD: AX0
        - ID: AX1 + AX4 (equal split; when one axis hits a soft limit, the other continues)
        - OD+ID: AX0 + (AX1+AX4)
        - Center clamp: AX2
        """
        mode = int(getattr(self.recipe, 'teach_axes_mode', 2))

        def _jog_axis(ax: int, _direction: str, _on: bool):
            ax = max(0, min(AXIS_COUNT - 1, int(ax)))
            if _on:
                try:
                    self._write_axis_params(ax)
                except Exception:
                    pass
                if _direction == 'rev':
                    self.set_cmd_bits(ax, set_mask=CMD_JOG_B_REQ, clr_mask=CMD_JOG_F_REQ)
                else:
                    self.set_cmd_bits(ax, set_mask=CMD_JOG_F_REQ, clr_mask=CMD_JOG_B_REQ)
            else:
                self.set_cmd_bits(ax, set_mask=0, clr_mask=(CMD_JOG_F_REQ | CMD_JOG_B_REQ))

        # release: stop all active jog bits (safe)
        if not on:
            try:
                self._teach_jog_active = False
            except Exception:
                pass
            if mode in (0, 2):
                _jog_axis(0, direction, False)
            if mode in (1, 2):
                _jog_axis(1, direction, False)
                _jog_axis(4, direction, False)
            if mode == 3:
                _jog_axis(2, direction, False)
            return

        # press: start
        if mode == 3:
            _jog_axis(2, direction, True)
            return

        if mode in (0, 2):
            _jog_axis(0, direction, True)

        if mode in (1, 2):
            # composite jog (AX1+AX4) with periodic limit switching
            self._teach_jog_active = True
            token = time.time()
            self._teach_jog_token = token

            def _raw_range(ax: int):
                cal = self.axis_cal
                ac = self.get_axis_copy(ax)
                try:
                    p = float(getattr(ac, 'softlim_pos', float('nan')))
                    n = float(getattr(ac, 'softlim_neg', float('nan')))
                except Exception:
                    p = float('nan'); n = float('nan')
                if not (p == p and n == n):
                    return (-float('inf'), float('inf'))
                r1 = float(cal.abs_to_z_raw(ax, p))
                r2 = float(cal.abs_to_z_raw(ax, n))
                return (min(r1, r2), max(r1, r2))

            def _tick():
                # stop conditions
                if not bool(getattr(self, '_teach_jog_active', False)):
                    return
                if getattr(self, '_teach_jog_token', None) != token:
                    return

                cal = self.axis_cal

                lo1, hi1 = _raw_range(1)
                lo4, hi4 = _raw_range(4)

                ac1 = self.get_axis_copy(1)
                ac4 = self.get_axis_copy(4)
                z1_raw = float(cal.abs_to_z_raw(1, ac1.act_pos))
                z4_raw = float(cal.abs_to_z_raw(4, ac4.act_pos))

                eps = 0.20
                enable1 = True
                enable4 = True
                if direction == 'fwd':
                    if z1_raw >= hi1 - eps:
                        enable1 = False
                    if z4_raw >= float(hi4) - eps:
                        enable4 = False
                else:  # rev
                    if z1_raw <= float(lo1) + eps:
                        enable1 = False
                    if z4_raw <= float(lo4) + eps:
                        enable4 = False

                # apply (AX4 continues when AX1 is blocked)
                _jog_axis(1, direction, enable1)
                _jog_axis(4, direction, enable4)

                self.after(80, _tick)

            _tick()


    def _refresh_teach_pos(self):
        """Refresh teach panel position labels (OD/ID + alignment)."""
        cal = self.axis_cal

        ac0 = self.get_axis_copy(0)
        ac1 = self.get_axis_copy(1)
        ac4 = self.get_axis_copy(4)
        ac2 = self.get_axis_copy(2)

        act0 = float(ac0.act_pos)
        act1 = float(ac1.act_pos)
        act4 = float(ac4.act_pos)
        act2 = float(ac2.act_pos)

        plan = teach_position_plan(
            cal,
            ax0_abs=act0,
            ax1_abs=act1,
            ax2_abs=act2,
            ax4_abs=act4,
        )
        z0_raw = plan.z0_raw
        z1_raw = plan.z1_raw
        z4_raw = plan.z4_raw
        z2_raw = plan.z2_raw
        zid_raw = plan.zid_raw
        z0_disp = plan.z0_disp
        zid_disp = plan.zid_disp
        z2_disp = plan.z2_disp
        z_id_expect_disp = plan.z_id_expected_disp
        delta = plan.delta
        aligned = plan.aligned

        mode = int(getattr(self.recipe, 'teach_axes_mode', 2))
        mode_text = {0: "外径AX0", 1: "内径AX1+4", 2: "内径+外径AX0+1+4", 3: "中心架AX2"}.get(mode, "-")

        if hasattr(self, "teach_mode_var"):
            self.teach_mode_var.set(f"当前示教轴: {mode_text}")

        if hasattr(self, "teach_align_var"):
            if aligned:
                self.teach_align_var.set(f"OD/ID 对齐: 是  (Δ={delta:+.3f} mm)")
            else:
                self.teach_align_var.set(f"OD/ID 对齐: 否  (Δ={delta:+.3f} mm)")

        if hasattr(self, "teach_abs_var"):
            self.teach_abs_var.set(
                f"绝对位置 abs(mm): AX0={act0:.6f}  AX1={act1:.6f}  AX2={act2:.6f}  AX4={act4:.6f}"
            )

        if hasattr(self, "teach_z_var"):
            self.teach_z_var.set(
                f"Z坐标 Z_disp(mm): OD={z0_disp:.3f}  AX2={z2_disp:.3f}  ID_act={zid_disp:.3f}  ID_exp={z_id_expect_disp:.3f}"
            )

        if hasattr(self, "teach_axes_var"):
            self.teach_axes_var.set(
                f"Z_raw(mm): Z0={z0_raw:.3f}  Z2={z2_raw:.3f}  Z1={z1_raw:.3f}  Z4={z4_raw:.3f}  Zid={zid_raw:.3f}"
            )

        # standby/start display
        try:
            self._refresh_standby_pos()
        except Exception:
            pass
        try:
            self._refresh_start_pos()
        except Exception:
            pass
