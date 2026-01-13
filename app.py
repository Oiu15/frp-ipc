# ./app.py
from __future__ import annotations

"""FRP 管检测 IPC 应用（Tkinter）。

本文件保留“应用层/编排层”的职责：
- Tk 主线程：UI、事件回调、状态刷新
- 驱动层：PLC(Modbus TCP) 与测径仪(Serial) 的后台线程
- 服务层：AutoFlow 自动测量状态机线程

解耦原则：
- 协议常量与地址：./config/addresses.py
- 数据模型：./core/models.py
- IO 驱动：./drivers/*
- 自动测量流程：./services/autoflow_service.py
- UI 构建（四页）：./ui/screens/*
"""

import queue
import threading
import time
from typing import List, Optional, Tuple, Iterable

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from config.addresses import (
    DEFAULT_PLC_IP,
    DEFAULT_PLC_PORT,
    DEFAULT_UNIT_ID,
    DEFAULT_GAUGE_PORT,
    AXIS_NAMES,
    AXIS_COUNT,
    COMM_BASE_D,
    COMM_STRIDE_D,
    CMD_JOG_F_REQ,
    CMD_JOG_B_REQ,
    CMD_VELMOVE_REQ,
    CMD_HALT_REQ,
    CMD_STOP_REQ,
    CMD_RESET_REQ,
    CMD_EN_REQ,
    CMD_MOVEA_REQ,
    MODE_INCH,
    MODE_DIR_REV,
    STS_READY,
    STS_ENABLED,
    STS_BUSY,
    STS_DONE,
    STS_FAULT,
    STS_JOGGING,
    STS_VELRUN,
    STS_INTERLOCK,
    OFF_TGT_POS,
    OFF_TGT_POS2,
    OFF_VEL,
    FLOAT64_WORD_ORDER,
)

from core.models import AxisComm, UiCoord, Recipe, MeasureRow
from drivers.plc_client import (
    PlcWorker,
    CmdWriteRegs,
    CmdSetCmdMask,
    CmdPulseCmdMask,
    CmdWriteModeWord,
    encode_float64_to_4regs,
)
from drivers.gauge_driver import GaugeWorker, list_serial_ports
from services.autoflow_service import AutoFlow

from ui.screens.axis_screen import build_axis_screen
from ui.screens.recipe_screen import build_recipe_screen
from ui.screens.gauge_screen import build_gauge_screen
from ui.screens.main_screen import build_main_screen

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FRP 测量 v0.2")
        self.geometry("1260x820")

        self.ui_q: queue.Queue = queue.Queue()
        self.cmd_q: queue.Queue = queue.Queue()

        self.worker = PlcWorker(self.ui_q, self.cmd_q)
        self.worker.start()

        # Optional gauge worker (real serial)
        self.gauge_worker: Optional[GaugeWorker] = GaugeWorker(self.ui_q)
        self.gauge_worker.start()

        self.axis_idx = tk.IntVar(value=0)
        self.plc_status_var = tk.StringVar(value="PLC: connecting...")

        self.ip_var = tk.StringVar(value=DEFAULT_PLC_IP)
        self.port_var = tk.StringVar(value=str(DEFAULT_PLC_PORT))

        self._axis_snapshot: List[AxisComm] = [AxisComm() for _ in range(AXIS_COUNT)]
        self._snapshot_lock = threading.Lock()

        # UI-only coordinate system
        self.ui_coord = UiCoord(zero_abs=0.0, sign=+1)

        # Recipe (in-memory)
        self.recipe = Recipe()
        self.recipe.section_pos_ui = self.recipe.compute_default_positions_ui()

        # Gauge config (UI)
        self.sim_gauge_enabled = False
        self.gauge_conn_var = tk.StringVar(value="未连接")
        self.gauge_last_var = tk.StringVar(value="Gauge: --")
        self.gauge_err_var = tk.StringVar(value="")

        # Auto
        self._auto_thread: Optional[AutoFlow] = None
        self.auto_state_var = tk.StringVar(value="IDLE")
        self.auto_msg_var = tk.StringVar(value="-")
        self.auto_progress_var = tk.StringVar(value="当前截面: - / 总截面: -")
        self.auto_done_var = tk.StringVar(value="测量完成: 否")
        self.straight_var = tk.StringVar(value="直线度: -")
        self.cov_var = tk.StringVar(value="采样覆盖率: -")

        self._build_ui()
        self.after(60, self._poll_ui_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(200, self._auto_connect_plc)

    def _auto_connect_plc(self):
        """Startup auto-connect kick (non-manual)."""
        try:
            ip = self.ip_var.get().strip() or DEFAULT_PLC_IP
            port = int(self.port_var.get().strip() or str(DEFAULT_PLC_PORT))
        except Exception:
            ip, port = DEFAULT_PLC_IP, DEFAULT_PLC_PORT
        # non-manual: do not reset give-up if already gave up 
        self.worker.request_connect(ip=ip, port=port, manual=False)

    # =========================
    # Close / threading
    # =========================
    def _on_close(self):
        try:
            if self._auto_thread and self._auto_thread.is_alive():
                self._auto_thread.stop()
        except Exception:
            pass
        try:
            self.worker.stop()
        except Exception:
            pass
        try:
            if self.gauge_worker:
                self.gauge_worker.stop()
        except Exception:
            pass
        self.destroy()

    # =========================
    # Build UI
    # =========================
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Label(top, textvariable=self.plc_status_var).pack(side=tk.LEFT)

        cfg = ttk.Frame(top)
        cfg.pack(side=tk.RIGHT)

        ttk.Label(cfg, text="IP").grid(row=0, column=0, padx=4)
        ttk.Entry(cfg, width=14, textvariable=self.ip_var).grid(row=0, column=1, padx=4)

        ttk.Label(cfg, text="Port").grid(row=0, column=2, padx=4)
        ttk.Entry(cfg, width=6, textvariable=self.port_var).grid(row=0, column=3, padx=4)

        ttk.Button(cfg, text="Apply", command=self._apply_conn).grid(row=0, column=4, padx=6)

        nb = ttk.Notebook(self)
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        tab_axis = ttk.Frame(nb)
        tab_recipe = ttk.Frame(nb)
        tab_gauge = ttk.Frame(nb)
        tab_main = ttk.Frame(nb)

        nb.add(tab_axis, text="轴参数/调试")
        nb.add(tab_recipe, text="配方/示教")
        nb.add(tab_gauge, text="测径仪通信")
        nb.add(tab_main, text="主操作/自动测量")

        build_axis_screen(self, tab_axis)
        build_recipe_screen(self, tab_recipe)
        build_gauge_screen(self, tab_gauge)
        build_main_screen(self, tab_main)


    def _apply_conn(self):
        try:
            ip = self.ip_var.get().strip()
            port = int(self.port_var.get().strip())
            if not ip:
                raise ValueError("IP不能为空")
            if port <= 0:
                raise ValueError("Port非法")
        except Exception as e:
            messagebox.showerror("配置错误", str(e))
            return
        self.plc_status_var.set(f"PLC: MANUAL CONNECT... ip={ip}:{port}")
        self.worker.request_connect(ip=ip, port=port, manual=True)

    # =========================
    # Manual tab
    # =========================
    def _build_manual(self, parent: ttk.Frame):
        """(Deprecated) Wrapper for legacy code path."""
        build_axis_screen(self, parent)


    def _set_current_zero(self):
        ax = self._axis()
        ac = self.get_axis_copy(ax)
        self.ui_coord.zero_abs = float(ac.act_pos)
        self.zero_abs_var.set(f"{self.ui_coord.zero_abs:.6f}")
        self._refresh_axis_panel()

    def _set_scan_axis_zero(self):
        scan_ax = int(self.recipe.scan_axis)
        ac = self.get_axis_copy(scan_ax)
        self.ui_coord.zero_abs = float(ac.act_pos)
        self.zero_abs_var.set(f"{self.ui_coord.zero_abs:.6f}")
        self._refresh_axis_panel()
        self._refresh_recipe_table()

    def _on_sign_change(self):
        self.ui_coord.sign = +1 if int(self.sign_var.get()) >= 0 else -1
        self._refresh_axis_panel()
        self._refresh_recipe_table()

    # =========================
    # Recipe tab
    # =========================
    def _build_recipe(self, parent: ttk.Frame):
        """(Deprecated) Wrapper for legacy code path."""
        build_recipe_screen(self, parent)


    def _kv_row(self, parent: ttk.Frame, label: str, var: tk.StringVar, row: int):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="e", padx=6, pady=4
        )
        ttk.Entry(parent, width=18, textvariable=var).grid(
            row=row, column=1, sticky="w", padx=6, pady=4
        )

    def _on_scan_axis_selected(self, _evt=None):
        try:
            i = int(self.scan_axis_combo.get().split(":")[0].strip())
        except Exception:
            i = 4
        self.scan_axis_var.set(i)

    def _recipe_apply_from_ui(self) -> Recipe:
        """Read recipe fields from UI into self.recipe (and return a copy)."""
        r = Recipe()
        r.name = self.recipe_name_var.get().strip() or "默认配方"
        r.pipe_len_mm = float(self.pipe_len_var.get())
        r.clamp_occupy_mm = float(self.clamp_var.get())
        r.margin_head_mm = float(self.margin_h_var.get())
        r.margin_tail_mm = float(self.margin_t_var.get())
        r.section_count = int(float(self.section_n_var.get()))
        r.scan_axis = int(self.scan_axis_var.get())
        r.od_std_mm = float(self.od_std_var.get())
        r.id_std_mm = float(self.id_std_var.get())
        r.od_tol_mm = float(self.od_tol_var.get())
        r.points_per_rev = int(float(self.points_per_rev_var.get()))
        r.min_bin_coverage = float(self.min_cov_var.get())
        r.sample_timeout_s = float(self.sample_timeout_var.get())
        r.max_revolutions = float(self.max_revs_var.get())

        # keep existing taught positions when section_count matches
        if len(self.recipe.section_pos_ui) == r.section_count:
            r.section_pos_ui = list(self.recipe.section_pos_ui)
        else:
            r.section_pos_ui = r.compute_default_positions_ui()

        # save back
        self.recipe = r
        return r

    def _recipe_compute(self):
        try:
            r = self._recipe_apply_from_ui()
            r.section_pos_ui = r.compute_default_positions_ui()
            self.recipe.section_pos_ui = list(r.section_pos_ui)
            self._refresh_recipe_table()
            self._refresh_auto_std_panel()
        except Exception as e:
            messagebox.showerror("配方计算错误", str(e))

    def _recipe_save(self):
        try:
            r = self._recipe_apply_from_ui()
            path = filedialog.asksaveasfilename(
                title="保存配方",
                defaultextension=".json",
                filetypes=[("JSON", "*.json")],
            )
            if not path:
                return
            data = {
                "name": r.name,
                "pipe_len_mm": r.pipe_len_mm,
                "clamp_occupy_mm": r.clamp_occupy_mm,
                "margin_head_mm": r.margin_head_mm,
                "margin_tail_mm": r.margin_tail_mm,
                "section_count": r.section_count,
                "scan_axis": r.scan_axis,
                "od_std_mm": r.od_std_mm,
                "id_std_mm": r.id_std_mm,
                "od_tol_mm": r.od_tol_mm,
                "points_per_rev": r.points_per_rev,
                "section_pos_ui": r.section_pos_ui,
                "ui_zero_abs": self.ui_coord.zero_abs,
                "ui_sign": self.ui_coord.sign,
            }
            with open(path, "w", encoding="utf-8") as f:
                import json

                json.dump(data, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("保存成功", f"已保存：{path}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def _recipe_load(self):
        try:
            path = filedialog.askopenfilename(
                title="加载配方",
                filetypes=[("JSON", "*.json")],
            )
            if not path:
                return
            import json

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # load recipe
            self.recipe_name_var.set(str(data.get("name", "默认配方")))
            self.pipe_len_var.set(str(data.get("pipe_len_mm", 1700.0)))
            self.clamp_var.set(str(data.get("clamp_occupy_mm", 300.0)))
            self.margin_h_var.set(str(data.get("margin_head_mm", 20.0)))
            self.margin_t_var.set(str(data.get("margin_tail_mm", 20.0)))
            self.section_n_var.set(str(data.get("section_count", 12)))
            scan_axis = int(data.get("scan_axis", 4))
            self.scan_axis_var.set(scan_axis)
            self.scan_axis_combo.current(max(0, min(AXIS_COUNT - 1, scan_axis)))
            self.od_std_var.set(str(data.get("od_std_mm", 187.3)))
            self.id_std_var.set(str(data.get("id_std_mm", 152.7)))
            self.od_tol_var.set(str(data.get("od_tol_mm", 0.1)))
            # 每圈采样点数：优先 points_per_rev；兼容旧字段 sample_count（旧版本是“每截面采样(M)”）
            if "points_per_rev" in data:
                self.points_per_rev_var.set(str(data.get("points_per_rev", 120)))
            else:
                self.points_per_rev_var.set(str(data.get("sample_count", 120)))

            # 等角采样参数（缺省值兼容）
            self.min_cov_var.set(str(data.get('min_bin_coverage', 0.95)))
            self.sample_timeout_var.set(str(data.get('sample_timeout_s', 5.0)))
            self.max_revs_var.set(str(data.get('max_revolutions', 2.0)))

            # coord system
            self.ui_coord.zero_abs = float(
                data.get("ui_zero_abs", self.ui_coord.zero_abs)
            )
            self.ui_coord.sign = int(data.get("ui_sign", self.ui_coord.sign))
            self.zero_abs_var.set(f"{self.ui_coord.zero_abs:.6f}")
            self.sign_var.set(+1 if self.ui_coord.sign >= 0 else -1)

            # positions
            pos = data.get("section_pos_ui", [])
            if isinstance(pos, list):
                self.recipe.section_pos_ui = [float(x) for x in pos]

            self._recipe_apply_from_ui()
            self._refresh_recipe_table()
            self._refresh_auto_std_panel()
            messagebox.showinfo("加载成功", f"已加载：{path}")
        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def _refresh_recipe_table(self):
        try:
            self.recipe_tree.delete(*self.recipe_tree.get_children())
        except Exception:
            return

        r = self.recipe
        # Ensure positions length
        if len(r.section_pos_ui) != r.section_count:
            r.section_pos_ui = r.compute_default_positions_ui()

        for i, x_ui in enumerate(r.section_pos_ui):
            x_abs = self.ui_coord.ui_to_abs(x_ui)
            src = (
                "示教/保留"
                if hasattr(self, "_taught_mark")
                and getattr(self, "_taught_mark", {}).get(i, False)
                else "计算"
            )
            self.recipe_tree.insert(
                "", "end", values=(i, f"{x_ui:.3f}", f"{x_abs:.3f}", src)
            )

    def _get_selected_recipe_idx(self) -> Optional[int]:
        sel = self.recipe_tree.selection()
        if not sel:
            return None
        item = sel[0]
        vals = self.recipe_tree.item(item, "values")
        if not vals:
            return None
        try:
            return int(vals[0])
        except Exception:
            return None

    def _teach_move_to_selected(self):
        try:
            r = self._recipe_apply_from_ui()
            idx = self._get_selected_recipe_idx()
            if idx is None:
                messagebox.showwarning("提示", "请先在表格中选中一个截面")
                return
            scan_ax = int(r.scan_axis)
            x_ui = float(r.section_pos_ui[idx])
            x_abs = self.ui_coord.ui_to_abs(x_ui)
            self.movea_abs(scan_ax, x_abs)
        except Exception as e:
            messagebox.showerror("示教移动失败", str(e))

    def _teach_save_current_to_selected(self):
        try:
            r = self._recipe_apply_from_ui()
            idx = self._get_selected_recipe_idx()
            if idx is None:
                messagebox.showwarning("提示", "请先在表格中选中一个截面")
                return
            scan_ax = int(r.scan_axis)
            ac = self.get_axis_copy(scan_ax)
            x_ui = self.ui_coord.abs_to_ui(ac.act_pos)
            r.section_pos_ui[idx] = float(x_ui)
            self.recipe.section_pos_ui = list(r.section_pos_ui)

            # mark taught
            if not hasattr(self, "_taught_mark"):
                self._taught_mark = {}
            self._taught_mark[idx] = True

            self._refresh_recipe_table()
        except Exception as e:
            messagebox.showerror("示教保存失败", str(e))

    def _teach_jog_hold(self, direction: str, on: bool):
        """Jog for teach panel, always on scan_axis."""
        try:
            scan_ax = int(self.recipe.scan_axis)
        except Exception:
            scan_ax = 0
        scan_ax = max(0, min(AXIS_COUNT - 1, scan_ax))

        # ensure MODE_INCH is off for jog
        cur_mode = self.get_axis_copy(scan_ax).mode & 0xFFFF
        if cur_mode & MODE_INCH:
            self._write_mode_word(scan_ax, (cur_mode & (~MODE_INCH)) & 0xFFFF)

        if on:
            try:
                vel, acc, dec, jerk = self._read_common_params()
            except Exception:
                vel, acc, dec, jerk = 100, 200, 200, 500
            base = self._base(scan_ax)
            self._write_regs(base + OFF_VEL, [vel, acc, dec, jerk])

            if direction == "rev":
                self.set_cmd_bits(
                    scan_ax, set_mask=CMD_JOG_B_REQ, clr_mask=CMD_JOG_F_REQ
                )
            else:
                self.set_cmd_bits(
                    scan_ax, set_mask=CMD_JOG_F_REQ, clr_mask=CMD_JOG_B_REQ
                )
        else:
            self.set_cmd_bits(
                scan_ax, set_mask=0, clr_mask=(CMD_JOG_F_REQ | CMD_JOG_B_REQ)
            )

    def _refresh_teach_pos(self):
        """Refresh teach panel position labels (scan axis)."""
        try:
            scan_ax = int(self.recipe.scan_axis)
        except Exception:
            scan_ax = 0
        scan_ax = max(0, min(AXIS_COUNT - 1, scan_ax))

        ac = self.get_axis_copy(scan_ax)
        ui_pos = self.ui_coord.abs_to_ui(ac.act_pos)

        # teach labels exist only after recipe tab built
        if hasattr(self, "teach_abs_var"):
            self.teach_abs_var.set(f"当前位置 abs: {ac.act_pos:.6f} (AX{scan_ax})")
        if hasattr(self, "teach_ui_var"):
            self.teach_ui_var.set(
                f"当前位置 UI : {ui_pos:.3f}    (ZeroAbs={self.ui_coord.zero_abs:.3f}, sign={self.ui_coord.sign:+d})"
            )

    # =========================
    # Auto tab
    # =========================
    def _build_auto(self, parent: ttk.Frame):
        """(Deprecated) Wrapper for legacy code path."""
        build_main_screen(self, parent)


    def _refresh_auto_std_panel(self):
        r = self.recipe
        self.lbl_od_std.config(text=f"{r.od_std_mm:.3f} mm   (tol ±{r.od_tol_mm:.3f})")
        self.lbl_id_std.config(text=f"{r.id_std_mm:.3f} mm")

    def _on_sim_gauge_toggle(self):
        self.sim_gauge_enabled = bool(self.sim_gauge_var.get())

    def _list_serial_ports(self) -> List[str]:
        """Return list of available serial ports."""
        return list_serial_ports()


    def _refresh_ports(self):
        ports = self._list_serial_ports()
        self.port_combo.configure(values=ports)

        cur = (self.port_combo.get() or "").strip()

        if not ports:
            self.port_combo.set(DEFAULT_GAUGE_PORT)
            return

        if cur and (cur in ports):
            self.port_combo.set(cur)
            return

        if DEFAULT_GAUGE_PORT in ports:
            self.port_combo.set(DEFAULT_GAUGE_PORT)
            return

        self.port_combo.set(ports[0])

    def _gauge_connect(self):
        """连接测径仪（只在需要时打开串口）。
        说明：
        - 重复点击“连接”不会重复 open 串口，只会更新参数（避免 Windows 下 PermissionError(13)）。
        - 会自动关闭“模拟测径仪”开关。
        """
        try:
            #if serial is None:
            #    raise RuntimeError("pyserial 未安装。")

            port = self.port_combo.get().strip() or DEFAULT_GAUGE_PORT
            baud = int(self.baud_var.get().strip() or "9600")
            cmd = (self.req_cmd_var.get() or "M1,0").strip()

            # 选择真实测径仪时，自动关闭模拟
            self.sim_gauge_var.set(0)
            self.sim_gauge_enabled = False

            # UI 立即给一个“连接中”的可见反馈；真正成功/失败由 gauge_conn/gauge_err 更新
            self.gauge_conn_var.set(f"串口: 连接中... ({port}@{baud})")
            self.gauge_err_var.set("")

            self.gauge_worker.configure(
                enabled=True,
                port=port,
                baud=baud,
                timeout_s=0.5,
                eol="\r",
                request_cmd=cmd,
                bytesize=8,
                parity="N",
                stopbits=1,
            )
        except Exception as e:
            self.gauge_conn_var.set("串口: 未连接")
            messagebox.showerror("连接测径仪失败", str(e))

    def _gauge_disconnect(self):
        """断开测径仪串口。"""
        try:
            self.gauge_worker.configure(
                enabled=False,
                port="",
                baud=9600,
                timeout_s=0.5,
                eol="\r",
                request_cmd="",
            )
            self.gauge_conn_var.set("串口: 未连接")
            self.gauge_err_var.set("已断开")
        except Exception:
            self.gauge_conn_var.set("串口: 未连接")

    def _gauge_request_once(self):
        """发送一次测径仪请求命令（默认 M1,0\\r）。
        - 需要先“连接”，否则会提示 not enabled。
        - 返回数据由后台线程解析后，自动更新 Gauge: OD。
        """
        try:
            self.gauge_worker.send_request()
        except Exception as e:
            self.gauge_err_var.set(f"Gauge ERROR: {e}")

    # =========================
    # Auto actions
    # =========================
    def _auto_start(self):
        try:
            # update recipe first
            self._auto_clear_ui()
            self._recipe_apply_from_ui()
            self._refresh_auto_std_panel()

            if self._auto_thread and self._auto_thread.is_alive():
                messagebox.showwarning("提示", "自动测量已在运行")
                return
            self._auto_thread = AutoFlow(self)
            self._auto_thread.start()
        except Exception as e:
            messagebox.showerror("启动失败", str(e))

    def _auto_stop(self):
        try:
            if self._auto_thread and self._auto_thread.is_alive():
                self._auto_thread.stop()
                # Immediately stop axis motions on PLC side to avoid "in-position timeout" -> ERR.
                self.abort_motion()
        except Exception:
            pass

    def _auto_clear_ui(self):
        self.result_tree.delete(*self.result_tree.get_children())
        self.straight_var.set("直线度: -")
        self.cov_var.set("采样覆盖率: -")
        self.auto_progress_var.set("当前截面: - / 总截面: -")
        self.auto_done_var.set("测量完成: 否")

    # =========================
    # Motion abort (used by Auto STOP)
    # =========================
    def abort_motion(self, axes: Optional[Iterable[int]] = None):
        """Immediately stop axis motions on PLC side.

        Strategy:
        - clear any level-type motion request bits (JOG / VELMOVE)
        - pulse HALT then STOP (both are supported in your command word)
        """
        if axes is None:
            axes = range(AXIS_COUNT)

        # Best-effort: drop queued motion commands so STOP/HALT reaches PLC ASAP.
        try:
            while True:
                self.cmd_q.get_nowait()
        except queue.Empty:
            pass

        clr = CMD_JOG_F_REQ | CMD_JOG_B_REQ | CMD_VELMOVE_REQ
        for ax in axes:
            try:
                ax_i = max(0, min(AXIS_COUNT - 1, int(ax)))
                # clear level bits first
                self.set_cmd_bits(ax_i, set_mask=0, clr_mask=clr)
                # then request stop/halt (pulse)
                self._pulse_cmd_bits(ax_i, CMD_HALT_REQ)
                self._pulse_cmd_bits(ax_i, CMD_STOP_REQ)
            except Exception:
                pass

    # =========================
    # Helper: labeled entry
    # =========================
    def _labeled_entry(
        self, parent: ttk.Frame, label: str, default: str, col: int
    ) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(
            row=0,
            column=col * 2,
            padx=(10 if col == 0 else 6, 2),
            pady=6,
            sticky="e",
        )
        ent = ttk.Entry(parent, width=14)
        ent.grid(row=0, column=col * 2 + 1, padx=(0, 6), pady=6, sticky="w")
        ent.insert(0, default)
        return ent

    # =========================
    # Axis snapshot helpers (thread-safe)
    # =========================
    def get_axis_copy(self, axis: int) -> AxisComm:
        axis = max(0, min(AXIS_COUNT - 1, int(axis)))
        with self._snapshot_lock:
            ac = self._axis_snapshot[axis]
            return AxisComm(**ac.__dict__)

    def get_recipe_copy(self) -> Recipe:
        # minimal deep copy
        r = self.recipe
        rr = Recipe(**{k: getattr(r, k) for k in r.__dataclass_fields__.keys()})
        rr.section_pos_ui = list(r.section_pos_ui)
        return rr

    # =========================
    # Low-level write helpers
    # =========================
    def _base(self, axis: int) -> int:
        return COMM_BASE_D + COMM_STRIDE_D * axis

    def _write_regs(self, d_addr: int, values: List[int]):
        self.cmd_q.put(CmdWriteRegs(d_addr=d_addr, values=values))

    def set_cmd_bits(self, axis: int, set_mask: int = 0, clr_mask: int = 0):
        self.cmd_q.put(CmdSetCmdMask(axis=axis, set_mask=set_mask, clr_mask=clr_mask))

    def _pulse_cmd_bits(self, axis: int, pulse_mask: int, pulse_ms: int = 120):
        self.cmd_q.put(
            CmdPulseCmdMask(axis=axis, pulse_mask=pulse_mask, pulse_ms=pulse_ms)
        )

    def _write_mode_word(self, axis: int, mode_word: int):
        self.cmd_q.put(CmdWriteModeWord(axis=axis, mode_word=mode_word))

    def _read_common_params(self) -> tuple[int, int, int, int]:
        vel = int(float(self.ent_vel.get().strip()))
        acc = int(float(self.ent_acc.get().strip()))
        dec = int(float(self.ent_dec.get().strip()))
        jerk = int(float(self.ent_jerk.get().strip()))
        return vel, acc, dec, jerk

    def _write_common_params(self):
        ax = self._axis()
        try:
            vel, acc, dec, jerk = self._read_common_params()
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return

        base = self._base(ax)
        self._write_regs(base + OFF_VEL, [vel, acc, dec, jerk])

    # =========================
    # Programmatic MoveA (abs)
    # =========================
    def movea_abs(self, axis: int, pos_abs: float):
        axis = max(0, min(AXIS_COUNT - 1, int(axis)))

        try:
            vel, acc, dec, jerk = self._read_common_params()
        except Exception:
            vel, acc, dec, jerk = 100, 200, 200, 500

        regs = encode_float64_to_4regs(float(pos_abs), FLOAT64_WORD_ORDER)
        base = self._base(axis)

        self._write_regs(base + OFF_TGT_POS, regs)
        self._write_regs(base + OFF_VEL, [vel, acc, dec, jerk])
        self._pulse_cmd_bits(axis, CMD_MOVEA_REQ)

    # =========================
    # UI actions (manual)
    # =========================
    def _axis(self) -> int:
        i = int(self.axis_idx.get())
        return max(0, min(AXIS_COUNT - 1, i))

    def _on_axis_selected(self, _evt=None):
        try:
            i = int(self.axis_combo.get().split(":")[0].strip())
        except Exception:
            i = 0
        self.axis_idx.set(i)
        self._refresh_axis_panel()

    def _poll_ui_queue(self):
        try:
            while True:
                k, payload = self.ui_q.get_nowait()

                if k == "plc_ok":
                    self.plc_status_var.set(
                        f"PLC: OK   {time.strftime('%H:%M:%S')}   ip={self.worker.ip}:{self.worker.port}   unit={self.worker.unit_id}"
                    )
                    with self._snapshot_lock:
                        self._axis_snapshot = payload["axes"]
                    self._refresh_axis_panel()

                elif k == "plc_err":
                    err = payload.get("err", "")
                    retry = payload.get("retry", None)
                    mx = payload.get("max", None)
                    backoff_s = payload.get("backoff_s", None)
                    if retry is not None and mx is not None and backoff_s is not None:
                        self.plc_status_var.set(
                            f"PLC: ERROR  {err}   (retry {retry}/{mx}, next in {backoff_s}s)"
                        )
                    else:
                        self.plc_status_var.set(f"PLC: ERROR   {err}")
                
                elif k == "plc_giveup":
                    retry = payload.get("retry", 0)
                    mx = payload.get("max", 0)
                    self.plc_status_var.set(
                        f"PLC: GIVE UP after {retry}/{mx}. Click Apply to reconnect."
                    )

                elif k == "plc_manual":
                    ip = payload.get("ip", "")
                    port = payload.get("port", "")
                    self.plc_status_var.set(f"PLC: MANUAL CONNECT... ip={ip}:{port}")
                    
                elif k == "gauge_conn":
                    if payload.get("connected"):
                        port = payload.get("port", "")
                        baud = payload.get("baud", "")
                        self.gauge_conn_var.set(f"串口: 已连接 ({port}@{baud})")
                    else:
                        self.gauge_conn_var.set("串口: 未连接")

                elif k == "gauge_tx":
                    # 可选：显示最近一次发送的请求（避免刷屏，只做轻提示）
                    cmd = payload.get("cmd", "")
                    if cmd:
                        self.gauge_err_var.set(f"已发送: {cmd}")

                elif k == "gauge_ok":
                    self.gauge_last_var.set(
                        f"Gauge: OD={payload['od']:.4f} mm   raw={payload.get('raw','')}"
                    )
                    self.gauge_err_var.set("")

                elif k == "gauge_raw":
                    # only update if no parsed value is flowing
                    pass

                elif k == "gauge_err":
                    self.gauge_err_var.set(f"Gauge ERROR: {payload.get('err')}")

                elif k == "auto_clear":
                    self._auto_clear_ui()

                elif k == "auto_progress":
                    idx = int(payload.get("idx", 0))
                    total = int(payload.get("total", 0))
                    # UI uses 1-based section index
                    self.auto_progress_var.set(f"当前截面: {idx + 1} / 总截面: {total}")
                    self.auto_done_var.set("测量完成: 否")

                elif k == "auto_cov":
                    cov = payload.get("cov", None)
                    miss = payload.get("miss", None)
                    reason = str(payload.get("reason", "") or "")
                    revs = payload.get("revs", None)
                    elapsed = payload.get("elapsed", None)

                    reason_txt = ""
                    if reason:
                        mapping = {"COV": "覆盖率达标", "TIMEOUT": "超时退出", "REV": "圈数到达"}
                        reason_txt = mapping.get(reason.upper(), reason)

                    if cov is None:
                        self.cov_var.set("采样覆盖率: -")
                    else:
                        parts = [f"采样覆盖率: {float(cov)*100:.1f}%"]
                        if miss is not None:
                            parts.append(f"缺失bin: {int(miss)}")
                        if revs is not None:
                            parts.append(f"圈数≈{float(revs):.2f}")
                        if elapsed is not None:
                            parts.append(f"用时{float(elapsed):.2f}s")
                        if reason_txt:
                            parts.append(f"结束:{reason_txt}")
                        self.cov_var.set("  ".join(parts))

                elif k == "auto_straightness":
                    # overall straightness result
                    val = payload.get("straightness", None)
                    if val is None:
                        self.straight_var.set("直线度: -")
                    else:
                        self.straight_var.set(f"直线度: {float(val):.3f} mm")

                elif k == "auto_row":
                    row: MeasureRow = payload["row"]
                    self._append_result_row(row)

                elif k == "auto_state":
                    st = payload.get("state", "IDLE")
                    msg = payload.get("msg", "-")
                    self.auto_state_var.set(str(st))
                    self.auto_msg_var.set(str(msg))
                    if st == "DONE":
                        self.auto_done_var.set("测量完成: 是")
                    elif st in ("ERR", "STOP"):
                        self.auto_done_var.set("测量完成: 否")

        except queue.Empty:
            pass
        self.after(60, self._poll_ui_queue)

    def _append_result_row(self, row: MeasureRow):
        ok_txt = "合格" if row.ok else "不合格"
        self.result_tree.insert(
            "",
            "end",
            values=(
                row.idx,
                f"{row.x_ui:.3f}",
                f"{row.od_avg:.3f}",
                f"{row.od_max:.3f}",
                f"{row.od_min:.3f}",
                f"{row.dev:+.3f}",
                f"{row.od_round:.3f}",
                ok_txt,
            ),
        )

    def _refresh_axis_panel(self):
        ax = self._axis()
        ac = self.get_axis_copy(ax)

        ui_pos = self.ui_coord.abs_to_ui(ac.act_pos)
        self.lbl_actpos.config(
            text=f"Act_Pos(abs): {ac.act_pos:.6f}    Act_Vel: {ac.act_vel:.6f}"
        )
        self.lbl_uipos.config(
            text=f"UI_Pos(相对): {ui_pos:.3f}    (ZeroAbs={self.ui_coord.zero_abs:.3f}, sign={self.ui_coord.sign:+d})"
        )

        self.lbl_err.config(text=f"ErrCode: {ac.err}    Warn: {ac.warn}")
        self.lbl_sts.config(text=f"Sts: 0x{ac.sts:04X}    Diag: 0x{ac.diag:04X}")

        flags = [
            ("R", bool(ac.sts & STS_READY)),
            ("EN", bool(ac.sts & STS_ENABLED)),
            ("BSY", bool(ac.sts & STS_BUSY)),
            ("DONE", bool(ac.sts & STS_DONE)),
            ("FLT", bool(ac.sts & STS_FAULT)),
            ("JOG", bool(ac.sts & STS_JOGGING)),
            ("VEL", bool(ac.sts & STS_VELRUN)),
            ("ILK", bool(ac.sts & STS_INTERLOCK)),
        ]
        self.lbl_flags.config(
            text="READY/EN/BSY/DONE/FLT/JOG/VEL/ILK: "
            + "  ".join([f"{k}={1 if v else 0}" for k, v in flags])
        )
        self.lbl_stid.config(
            text=f"St_ID: {ac.st_id}    Seq/Ack: {ac.seq}/{ac.seq_ack}"
        )
        self.lbl_cmd.config(text=f"Cmd: 0x{ac.cmd:04X}    Mode: 0x{ac.mode:04X}")

        # reflect UI toggles
        self.power_var.set(1 if (ac.cmd & CMD_EN_REQ) else 0)
        self.rev_var.set(1 if (ac.mode & MODE_DIR_REV) else 0)

        # keep teach panel synced
        self._refresh_teach_pos()

    def _on_power_toggle(self):
        ax = self._axis()
        if self.power_var.get():
            self.set_cmd_bits(ax, set_mask=CMD_EN_REQ, clr_mask=0)
        else:
            self.set_cmd_bits(ax, set_mask=0, clr_mask=CMD_EN_REQ)

    def _on_rev_toggle(self):
        ax = self._axis()
        cur_mode = self.get_axis_copy(ax).mode & 0xFFFF
        if self.rev_var.get():
            new_mode = (cur_mode | MODE_DIR_REV) & 0xFFFF
        else:
            new_mode = (cur_mode & (~MODE_DIR_REV)) & 0xFFFF
        self._write_mode_word(ax, new_mode)

    def _do_reset(self):
        ax = self._axis()
        self._pulse_cmd_bits(ax, CMD_RESET_REQ)

    def _do_stop(self):
        ax = self._axis()
        self._pulse_cmd_bits(ax, CMD_STOP_REQ)

    def _do_halt(self):
        ax = self._axis()
        self._pulse_cmd_bits(ax, CMD_HALT_REQ)

    def _do_movea(self):
        ax = self._axis()
        try:
            pos = float(self.ent_pos.get().strip())
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return
        self.movea_abs(ax, pos)

    def _do_vel_start(self):
        ax = self._axis()

        try:
            vel, acc, dec, jerk = self._read_common_params()
        except Exception:
            vel, acc, dec, jerk = 100, 200, 200, 500

        base = self._base(ax)
        self._write_regs(base + OFF_VEL, [vel, acc, dec, jerk])

        # VelMove 要求 MODE_INCH=0
        cur_mode = self.get_axis_copy(ax).mode & 0xFFFF
        if cur_mode & MODE_INCH:
            self._write_mode_word(ax, cur_mode & (~MODE_INCH))

        # 电平启动
        self.set_cmd_bits(ax, set_mask=CMD_VELMOVE_REQ, clr_mask=0)

    def _do_vel_stop(self):
        ax = self._axis()
        # 清电平
        self.set_cmd_bits(ax, set_mask=0, clr_mask=CMD_VELMOVE_REQ)

    def _jog_hold(self, direction: str, on: bool):
        ax = self._axis()

        # ensure MODE_INCH is off for jog
        cur_mode = self.get_axis_copy(ax).mode & 0xFFFF
        if cur_mode & MODE_INCH:
            self._write_mode_word(ax, (cur_mode & (~MODE_INCH)) & 0xFFFF)

        if on:
            try:
                vel, acc, dec, jerk = self._read_common_params()
            except Exception:
                vel, acc, dec, jerk = 100, 200, 200, 500
            base = self._base(ax)
            self._write_regs(base + OFF_VEL, [vel, acc, dec, jerk])

            if direction == "rev":
                self.set_cmd_bits(ax, set_mask=CMD_JOG_B_REQ, clr_mask=CMD_JOG_F_REQ)
            else:
                self.set_cmd_bits(ax, set_mask=CMD_JOG_F_REQ, clr_mask=CMD_JOG_B_REQ)
        else:
            self.set_cmd_bits(ax, set_mask=0, clr_mask=(CMD_JOG_F_REQ | CMD_JOG_B_REQ))

    def _do_inch(self, direction: str):
        ax = self._axis()

        # write distance to Tgt_Pos2
        try:
            dis = float(self.ent_step.get().strip())
            vel, acc, dec, jerk = self._read_common_params()
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return

        dis_regs = encode_float64_to_4regs(dis, FLOAT64_WORD_ORDER)

        base = self._base(ax)
        self._write_regs(base + OFF_TGT_POS2, dis_regs)
        self._write_regs(base + OFF_VEL, [vel, acc, dec, jerk])

        self.set_cmd_bits(ax, set_mask=0, clr_mask=(CMD_JOG_F_REQ | CMD_JOG_B_REQ))

        cur_mode = self.get_axis_copy(ax).mode & 0xFFFF
        mode_with_inch = (cur_mode | MODE_INCH) & 0xFFFF
        self._write_mode_word(ax, mode_with_inch)

        if direction == "rev":
            self._pulse_cmd_bits(ax, CMD_JOG_B_REQ, pulse_ms=800)
        else:
            self._pulse_cmd_bits(ax, CMD_JOG_F_REQ, pulse_ms=800)

        def _restore():
            time.sleep(1.2)
            self._write_mode_word(ax, cur_mode)

        threading.Thread(target=_restore, daemon=True).start()

    # =========================
    # Simulated gauge
    # =========================
    def simulate_gauge_once(self, recipe: Recipe) -> Tuple[float, str]:
        """Generate OD value near od_std with small deterministic-ish noise."""
        od_noise = (0.5 - (time.time() * 997) % 1.0) * 0.02  # ~±0.01mm
        od = float(recipe.od_std_mm) + float(od_noise)
        raw = f"M1,{od:+.4f}"
        return float(od), raw



def main():
    App().mainloop()


if __name__ == "__main__":
    main()
