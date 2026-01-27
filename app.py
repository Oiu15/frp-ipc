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
import os
from pathlib import Path
import datetime
import uuid
import json
import csv
import platform
import re

from utils.logger import init_log, log, log_exc
from typing import List, Optional, Tuple, Iterable

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from config.addresses import (
    DEFAULT_PLC_IP,
    DEFAULT_PLC_PORT,
    DEFAULT_GAUGE_PORT,
    AXIS_NAMES,
    AXIS_COUNT,
    axis_base,
    # cmd bits
    CMD_JOG_F_REQ,
    CMD_JOG_B_REQ,
    CMD_VELMOVE_REQ,
    CMD_HALT_REQ,
    CMD_STOP_REQ,
    CMD_RESET_REQ,
    CMD_EN_REQ,
    CMD_MOVEA_REQ,
    CMD_MOVER_REQ,
    # dir enum
    DIR_NONE,
    DIR_POS,
    DIR_NEG,
    DIR_SHORTEST,
    DIR_CURRENT,
    # offsets
    OFF_ACT_POS,
    OFF_POS_MOVEA,
    OFF_POS_MOVER,
    OFF_DIR_MOVER,
    OFF_VEL_MOVEA,
    OFF_VEL_MOVER,
    OFF_VEL_JOG,
    OFF_VEL_VELMOVE,
    OFF_ACC,
    OFF_DEC,
    OFF_JERK,
    # float word order
    FLOAT64_WORD_ORDER,
    # CL (Keyence) input mapping
    CL_IN_BASE_D,
    CL_OUT3_WORD_OFF,
    CL_OUT3_UPD_WORD_OFF,
    CL_OUT_SCALE_MM,
    CL_OUT_INVALID,
    CL_OUT_STANDBY,
    CL_OUT_POS_OVER,
    CL_OUT_NEG_OVER,
    # legacy aliases (still referenced by some code paths)
    OFF_TGT_POS,
    OFF_TGT_POS2,
    OFF_VEL,
    FLOAT64_WORD_ORDER,
    AXISCAL_MB_BASE,
    AXISCAL_WORDS,
    LINEAR_AXES,
    KEYTEST_X_BASE_COIL,
    KEYTEST_Y_BASE_COIL,
    KEYTEST_X_POINTS,
    KEYTEST_Y_POINTS,
)

from core.models import AxisComm, UiCoord, Recipe, MeasureRow, AxisCal
from core.recipe_store import RecipeStore
from drivers.plc_client import (
    PlcWorker,
    CmdWriteRegs,
    CmdReadRegs,
    CmdSetCmdMask,
    CmdPulseCmdMask,
    CmdWriteCoil,
    encode_float64_to_4regs,
    decode_float64_from_4regs,
)
from drivers.gauge_driver import GaugeWorker, list_serial_ports
from services.autoflow_service import AutoFlow

from ui.screens.axis_screen import build_axis_screen
from ui.screens.axis_cal_screen import build_axis_cal_screen
from ui.screens.recipe_screen import build_recipe_screen
from ui.screens.gauge_screen import build_gauge_screen
from ui.screens.main_screen import build_main_screen
from ui.screens.key_test_screen import build_key_test_screen


SOFTWARE_VERSION = "ipc_lgth_f1_14"

# ------------------------------
# UI event logging filter
# ------------------------------
# Keep logs useful: only record selected high-level UI events.
# (High-frequency events such as per-cycle PLC snapshots are intentionally excluded.)
LOG_UI_EVENT_FILTER = {
    "auto_state",
    "auto_progress",
    "auto_cov",
    "auto_row",
    "auto_postcalc",
    "auto_straightness",
    "auto_clear",
    "gauge_err",
    "plc_err",
}


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        try:
            init_log(filename=str(self._app_root_dir() / "logs" / f"log-{datetime.date.today().strftime('%Y-%m-%d')}.txt"), overwrite=False)
            log("APP_START", cwd=os.getcwd())
        except Exception:
            pass
        self.title(f"FRP 测量 {SOFTWARE_VERSION}")
        self.geometry("1260x820")

        try:
            self.state("zoomed")
        except Exception:
            try:
                w = self.winfo_screenwidth()
                h = self.winfo_screenheight()
                self.geometry(f"{w}x{h}+0+0")
            except Exception:
                pass

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

        # ------------------------------
        # Sync PLC reads (used by AutoFlow sampling to bind (OD, ID, θ))
        # ------------------------------
        # tag -> {"evt": threading.Event, "regs": List[int] | None}
        self._sync_reads = {}
        self._sync_reads_lock = threading.Lock()

        # Latest CL (ID, OUT3) snapshot from background polling (for UI / fallback)
        self._cl_out3_mm_latest: Optional[float] = None
        self._cl_out3_raw_latest: Optional[int] = None
        self._cl_out3_cnt_latest: Optional[int] = None
        self._cl_out3_ts_latest: float = 0.0


        # Per-axis pending flags for level commands (e.g., Enable) to avoid UI flip-flop
        self._power_cmd_pending = [0.0 for _ in range(AXIS_COUNT)]

        # UI-only coordinate system
        self.ui_coord = UiCoord(zero_abs=0.0, sign=+1)

        # ------------------------------
        # Key test (PLC X/Y points via Modbus coils)
        # ------------------------------
        # X 点：只读显示（物理输入）
        self.keytest_x_vars = [tk.IntVar(value=0) for _ in range(len(KEYTEST_X_POINTS))]
        # Y 点：读状态 + 单次写入（0/1）
        self.keytest_y_vars = [tk.IntVar(value=0) for _ in range(len(KEYTEST_Y_POINTS))]
        # 上次写入操作（仅提示；写入是否生效以读回状态为准）
        self.keytest_y_lastcmd_vars = [tk.StringVar(value="--") for _ in range(len(KEYTEST_Y_POINTS))]

        # Raw polled bits cache (for debugging / future extensions)
        self._keytest_x_bits = None  # type: ignore
        self._keytest_y_bits = None  # type: ignore

        # Axis calibration block (stored in PLC HD area)
        # Note: z_pos is IPC-only temporary shift, not written to PLC.
        self.axis_cal = AxisCal()  # sign defaults to -1
        self.axis_cal_vars = {
            "sign": tk.StringVar(value=str(self.axis_cal.sign)),
            "off_ax0": tk.StringVar(value=f"{self.axis_cal.off_ax0:.6f}"),
            "off_ax1": tk.StringVar(value=f"{self.axis_cal.off_ax1:.6f}"),
            "off_ax2": tk.StringVar(value=f"{self.axis_cal.off_ax2:.6f}"),
            "off_ax4": tk.StringVar(value=f"{self.axis_cal.off_ax4:.6f}"),
            "b14": tk.StringVar(value=f"{self.axis_cal.b14:.6f}"),
            "b2": tk.StringVar(value=f"{getattr(self.axis_cal, 'b2', 0.0):.6f}"),
            "keepout_w": tk.StringVar(value=f"{getattr(self.axis_cal, 'keepout_w', 0.0):.6f}"),
            "z_pos": tk.StringVar(value=f"{self.axis_cal.z_pos:.6f}"),
        }

        # Per-field status next to each AxisCal entry.
        # Semantics:
        # - 未读取: no data loaded yet
        # - 已读取: values filled from PLC read
        # - 已采集/未写入: capture/calibrate filled the entry but not persisted to PLC
        # - 写入中: write requested, awaiting verify
        # - 写入成功 / 写入失败: result of write+readback comparison
        self.axis_cal_field_status_vars = {
            "sign": tk.StringVar(value="未读取"),
            "off_ax0": tk.StringVar(value="未读取"),
            "off_ax1": tk.StringVar(value="未读取"),
            "off_ax2": tk.StringVar(value="未读取"),
            "off_ax4": tk.StringVar(value="未读取"),
            "b14": tk.StringVar(value="未读取"),
            "b2": tk.StringVar(value="未读取"),
            "keepout_w": tk.StringVar(value="未读取"),
            "z_pos": tk.StringVar(value="默认0"),
        }

        # AxisCal status / read-only display area (updated on PLC snapshots)
        self.axis_cal_status_vars = {
            "off_abs": tk.StringVar(value="-"),
            "act_abs": tk.StringVar(value="-"),
            "softlim_pos": tk.StringVar(value="-"),
            "softlim_neg": tk.StringVar(value="-"),
            "z_raw": tk.StringVar(value="-"),
            "z_disp": tk.StringVar(value="-"),
            "keepout_raw": tk.StringVar(value="-"),
            "keepout_disp": tk.StringVar(value="-"),
        }

        # Recipe (in-memory)
        self.recipe = Recipe()
        # Default Z_Pos section positions
        self.recipe.section_pos_z = self.recipe.compute_default_positions_z()
        # Keep legacy field aligned (deprecated)
        self.recipe.section_pos_ui = list(self.recipe.section_pos_z)

        
        # Recipe store (persistent, user directory)
        try:
            self.recipe_store = RecipeStore(RecipeStore.default_root("FRP_IPC"))
        except Exception:
            # fallback to local directory
            self.recipe_store = RecipeStore(Path("./data/recipes"))

        # Gauge config (UI)
        self.sim_gauge_enabled = False
        # Displacement meter (ID) - simulation only for now
        self.sim_disp_enabled = False
        self.gauge_conn_var = tk.StringVar(value="未连接")
        self.gauge_last_var = tk.StringVar(value="Gauge: --")
        self.gauge_err_var = tk.StringVar(value="")

        # CL (ID via PLC mapped registers, OUT3)
        self.cl_id_var = tk.StringVar(value="--")
        self.cl_cnt_var = tk.StringVar(value="--")
        self.id_n_var = tk.StringVar(value="0")
        self.id_avg_var = tk.StringVar(value="--")
        self.id_dev_var = tk.StringVar(value="--")
        self.id_round_var = tk.StringVar(value="--")

        # ID sample window (for avg/dev/roundness)
        import collections as _collections
        self._id_samples = _collections.deque(maxlen=300)
        self._last_cl_cnt = None

        # Auto
        self._auto_thread: Optional[AutoFlow] = None
        # Result table item ids (Treeview iids), in insertion order
        self._result_iids: list[str] = []
        self.auto_state_var = tk.StringVar(value="IDLE")
        self.auto_msg_var = tk.StringVar(value="-")
        self.auto_progress_var = tk.StringVar(value="当前截面: - / 总截面: -")
        self.auto_done_var = tk.StringVar(value="测量完成: 否")
        self.straight_var = tk.StringVar(value="直线度：--（外圆）/ --（内圆） || 整体同心度：--")
        self.cov_var = tk.StringVar(value="采样覆盖率：--")

        # ------------------------------
        # Run/Export (MSA)
        # ------------------------------
        self.pipe_sn_var = tk.StringVar(value="--")  # 流水号 (date + recipe + seq)
        self._run_serial: Optional[str] = None
        self._run_id: Optional[str] = None
        self._run_start_ts: Optional[float] = None
        self._run_end_ts: Optional[float] = None
        self._auto_rows: list[MeasureRow] = []
        self._auto_raw_points: list[dict] = []
        self._auto_export_done: bool = False

        # Per-section sampling coverage/info cache (key: 1-based section index)
        self._section_cov_info: dict[int, dict] = {}
        self._auto_cur_sec_idx: Optional[int] = None
        self._selected_sec_idx: Optional[int] = None
        self._axis_dist: Optional[float] = None

        self._build_ui()
        self.after(60, self._poll_ui_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(200, self._auto_connect_plc)
        self.after(250, self._auto_connect_gauge)

        # f2: one-shot debug read of axis calibration block (HD1000..)
        # Issued once after PLC connection becomes OK.
        self._dbg_axis_cal_sent = False

        # f4_1: write-then-readback verification for axis calibration block
        self._axis_cal_write_expect_regs: Optional[List[int]] = None
        self._axis_cal_write_pending = False

    def _dbg_read_axis_cal(self):
        """Issue a one-shot read of the axis calibration block for f2 validation.

        Note: In f2 it is normally triggered once after PLC connects OK (see plc_ok handler).
        """
        try:
            self.cmd_q.put(CmdReadRegs(AXISCAL_MB_BASE, AXISCAL_WORDS, "axis_cal"))
            print(
                f"[axis_cal] request read: addr={AXISCAL_MB_BASE} count={AXISCAL_WORDS}"
            )
        except Exception as e:
            print(f"[axis_cal] enqueue read failed: {e}")

    def _auto_connect_plc(self):
        """Startup auto-connect kick (non-manual)."""
        try:
            ip = self.ip_var.get().strip() or DEFAULT_PLC_IP
            port = int(self.port_var.get().strip() or str(DEFAULT_PLC_PORT))
        except Exception:
            ip, port = DEFAULT_PLC_IP, DEFAULT_PLC_PORT
        # non-manual: do not reset give-up if already gave up
        self.worker.request_connect(ip=ip, port=port, manual=False)

    def _auto_connect_gauge(self):
        """Startup auto-connect gauge once (COM2). Fail -> no retry."""
        if not self.gauge_worker:
            return
        port = "COM2"
        try:
            baud = int(
                (
                    getattr(self, "baud_var", None).get()
                    if hasattr(self, "baud_var")
                    else "9600"
                )
                or "9600"
            )
        except Exception:
            baud = 9600

        # 给UI一个立即反馈（不依赖线程回报）
        self.gauge_conn_var.set(f"串口: 连接中... ({port}@{baud})")
        self.gauge_err_var.set("")

        try:
            self.gauge_worker.configure(
                enabled=True,
                port=port,
                baud=baud,
                timeout_s=0.5,
                eol="\r",
                request_cmd=(
                    self.req_cmd_var.get().strip()
                    if hasattr(self, "req_cmd_var")
                    else "M1,0"
                ),
                bytesize=8,
                parity="N",
                stopbits=1,
            )
        except Exception as e:
            # 失败：禁用worker
            try:
                self.gauge_worker.configure(
                    enabled=False,
                    port="",
                    baud=9600,
                    timeout_s=0.5,
                    eol="\r",
                    request_cmd="",
                )
            except Exception:
                pass
            self.gauge_conn_var.set("串口: 未连接")
            self.gauge_err_var.set(f"启动自动连接失败: {e}")

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

        # PLC status is kept on the top bar; connect controls are moved into the “外设通信” tab.
        ttk.Label(top, textvariable=self.plc_status_var).pack(side=tk.LEFT)


        nb = ttk.Notebook(self)
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        tab_axis_cal = ttk.Frame(nb)
        tab_axis = ttk.Frame(nb)
        tab_recipe = ttk.Frame(nb)
        tab_gauge = ttk.Frame(nb)
        tab_keytest = ttk.Frame(nb)
        tab_main = ttk.Frame(nb)

        nb.add(tab_axis_cal, text="轴位标定")
        nb.add(tab_axis, text="轴参数/调试")
        nb.add(tab_recipe, text="配方/示教")
        nb.add(tab_gauge, text="外设通信")
        nb.add(tab_keytest, text="按键测试")
        nb.add(tab_main, text="主操作/自动测量")

        build_axis_cal_screen(self, tab_axis_cal)
        build_axis_screen(self, tab_axis)
        build_recipe_screen(self, tab_recipe)
        build_gauge_screen(self, tab_gauge)
        build_key_test_screen(self, tab_keytest)
        build_main_screen(self, tab_main)

        # init recipe store UI (dropdown, last recipe)
        try:
            self._recipe_store_init()
        except Exception:
            pass

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
    # Key test (PLC X/Y points via Modbus coils)
    # =========================
    def _keytest_write_y(self, y_point: int, value: int) -> None:
        """One-shot write to Y coil.

        - 写入与状态显示分离：写入后是否生效，以读回状态为准。
        - 不做持续写入，避免与 PLC 内部逻辑冲突。
        """
        try:
            y_point = int(y_point)
            value = 1 if int(value) != 0 else 0

            # NOTE:
            # X/Y 点采用“八进制标签”，线圈地址空间中没有 8/9。
            # 因此：Y10 的线圈地址 = BASE + 8；Y15 = BASE + 13。
            if y_point < 8:
                idx = y_point
            else:
                idx = y_point - 2  # skip 8/9
            coil = int(KEYTEST_Y_BASE_COIL) + int(idx)
            self.cmd_q.put(CmdWriteCoil(coil_addr=coil, value=value))
            # record last cmd
            try:
                idx = KEYTEST_Y_POINTS.index(y_point)
                ts = time.strftime("%H:%M:%S")
                self.keytest_y_lastcmd_vars[idx].set(f"写{value} @{ts}")
            except Exception:
                pass
        except Exception:
            pass

    def _keytest_apply_bits(self, x_bits, y_bits) -> None:
        """Update UI from polled coil bits."""
        try:
            self._keytest_x_bits = x_bits
            self._keytest_y_bits = y_bits

            # X points
            if isinstance(x_bits, (list, tuple)):
                for i, p in enumerate(KEYTEST_X_POINTS):
                    try:
                        pp = int(p)
                        if pp < 8:
                            idx = pp
                        else:
                            idx = pp - 2  # skip 8/9
                        v = 1 if bool(x_bits[int(idx)]) else 0
                        self.keytest_x_vars[i].set(v)
                    except Exception:
                        pass

            # Y points
            if isinstance(y_bits, (list, tuple)):
                for i, p in enumerate(KEYTEST_Y_POINTS):
                    try:
                        pp = int(p)
                        if pp < 8:
                            idx = pp
                        else:
                            idx = pp - 2  # skip 8/9
                        v = 1 if bool(y_bits[int(idx)]) else 0
                        self.keytest_y_vars[i].set(v)
                    except Exception:
                        pass
        except Exception:
            pass

    
    def _refresh_id_stats(self) -> None:
        """Compute ID metrics from recent CL OUT3 samples.

        Metrics (aligned with OD semantics):
        - 平均内径: mean(ID)
        - 内径偏差: mean(ID) - recipe.id_std_mm
        - 内径真圆度: max(ID) - min(ID)
        """
        try:
            samples = list(getattr(self, "_id_samples", []) or [])
            n = len(samples)
            self.id_n_var.set(str(n))
            if n <= 0:
                self.id_avg_var.set("--")
                self.id_dev_var.set("--")
                self.id_round_var.set("--")
                return

            avg = float(sum(samples) / n)
            mn = float(min(samples))
            mx = float(max(samples))
            roundness = mx - mn

            # deviation against recipe standard
            try:
                std = float(getattr(self, "recipe", None).id_std_mm)  # type: ignore[attr-defined]
            except Exception:
                std = float(getattr(getattr(self, "recipe", None), "id_std_mm", 0.0) or 0.0)

            dev = avg - std

            self.id_avg_var.set(f"{avg:.3f}")
            self.id_dev_var.set(f"{dev:+.3f}")
            self.id_round_var.set(f"{roundness:.3f}")
        except Exception:
            pass

# =========================
    # Axis calibration (HD block)
    # =========================
    def _axis_cal_set_field_status(self, keys: Iterable[str], text: str) -> None:
        """Update per-field status label(s) on the AxisCal page."""
        sv = getattr(self, "axis_cal_field_status_vars", None)
        if not isinstance(sv, dict):
            return
        for k in keys:
            if k in sv:
                try:
                    sv[k].set(text)
                except Exception:
                    pass

    def _axis_cal_from_ui(self) -> AxisCal:
        """Build an AxisCal instance from UI entry variables.

        Note: z_pos is IPC-only (will not be written to PLC), but we keep it in memory.
        """

        def _f(key: str, default: float = 0.0) -> float:
            try:
                return float(self.axis_cal_vars[key].get().strip())
            except Exception:
                return float(default)

        def _i(key: str, default: int = -1) -> int:
            try:
                return int(float(self.axis_cal_vars[key].get().strip()))
            except Exception:
                return int(default)

        cal = AxisCal(
            sign=-1 if _i("sign", -1) < 0 else +1,
            off_ax0=_f("off_ax0"),
            off_ax1=_f("off_ax1"),
            off_ax2=_f("off_ax2"),
            off_ax4=_f("off_ax4"),
            b14=_f("b14"),
            b2=_f("b2"),
            keepout_w=_f("keepout_w"),
            z_pos=_f("z_pos"),
        )
        return cal

    def _axis_cal_to_ui(self, cal: AxisCal) -> None:
        """Push an AxisCal instance into UI entry variables."""
        try:
            self.axis_cal_vars["sign"].set(str(int(cal.sign)))
            self.axis_cal_vars["off_ax0"].set(f"{cal.off_ax0:.6f}")
            self.axis_cal_vars["off_ax1"].set(f"{cal.off_ax1:.6f}")
            self.axis_cal_vars["off_ax2"].set(f"{cal.off_ax2:.6f}")
            self.axis_cal_vars["off_ax4"].set(f"{cal.off_ax4:.6f}")
            self.axis_cal_vars["b14"].set(f"{cal.b14:.6f}")
            self.axis_cal_vars["b2"].set(f"{cal.b2:.6f}")
            self.axis_cal_vars["keepout_w"].set(f"{cal.keepout_w:.6f}")
            # z_pos is IPC-only
            self.axis_cal_vars["z_pos"].set(f"{cal.z_pos:.6f}")
        except Exception:
            pass

    def axis_cal_read(self) -> None:
        """Read the axis calibration block from PLC (HD1000..)."""
        try:
            self._axis_cal_set_field_status(
                ["sign", "off_ax0", "off_ax1", "off_ax2", "off_ax4", "b14", "b2", "keepout_w"],
                "读取中",
            )
            self.cmd_q.put(CmdReadRegs(AXISCAL_MB_BASE, AXISCAL_WORDS, "axis_cal"))
            print(f"[axis_cal] request read: addr={AXISCAL_MB_BASE} count={AXISCAL_WORDS}")
        except Exception as e:
            print(f"[axis_cal] enqueue read failed: {e}")

    def axis_cal_write(self) -> None:
        """Write the axis calibration block to PLC (HD1000..).

        Note: z_pos will NOT be written to PLC.
        """
        try:
            cal = self._axis_cal_from_ui()
            # Keep IPC copy
            self.axis_cal = cal
            regs = cal.to_regs()
            # Enqueue write then read back to verify
            self._axis_cal_write_expect_regs = list(regs)
            self._axis_cal_set_field_status(
                ["sign", "off_ax0", "off_ax1", "off_ax2", "off_ax4", "b14", "b2", "keepout_w"],
                "写入中",
            )
            self.cmd_q.put(CmdWriteRegs(AXISCAL_MB_BASE, regs))
            self.cmd_q.put(CmdReadRegs(AXISCAL_MB_BASE, AXISCAL_WORDS, "axis_cal_verify"))
            print(
                f"[axis_cal] write+verify: addr={AXISCAL_MB_BASE} words={len(regs)} "
                f"(will read back {AXISCAL_WORDS} words)"
            )
        except Exception as e:
            self._axis_cal_set_field_status(
                ["sign", "off_ax0", "off_ax1", "off_ax2", "off_ax4", "b14", "b2", "keepout_w"],
                "写入失败",
            )
            print(f"[axis_cal] write failed: {e}")

    def axis_cal_capture_offsets(self) -> None:
        """Capture Off_AX0/1/2/4 from current axis feedback (Act_Pos).

        Semantics:
        - Off_AXn is defined as the servo feedback position (abs) at Z_raw == 0.
        - Capturing Off_AXn at the current position makes current Z_raw become 0.

        This function only updates IPC UI/in-memory values. Use "Write" to persist to PLC.
        """
        try:
            cal = self._axis_cal_from_ui()
            act0 = float(self.get_axis_copy(0).act_pos)
            act1 = float(self.get_axis_copy(1).act_pos)
            act2 = float(self.get_axis_copy(2).act_pos)
            act4 = float(self.get_axis_copy(4).act_pos)

            cal.off_ax0 = act0
            cal.off_ax1 = act1
            cal.off_ax2 = act2
            cal.off_ax4 = act4

            self._axis_cal_set_field_status(
                ["off_ax0", "off_ax1", "off_ax2", "off_ax4"],
                "已采集/未写入",
            )

            self.axis_cal = cal
            self._axis_cal_to_ui(cal)
            self.axis_cal_refresh_status()
            print(
                "[axis_cal] capture offsets: "
                f"off_ax0={act0:.6f} off_ax1={act1:.6f} off_ax2={act2:.6f} off_ax4={act4:.6f}"
            )
        except Exception as e:
            print(f"[axis_cal] capture offsets failed: {e}")

    def axis_cal_calibrate_b14(self) -> None:
        """Calibrate B14 based on current OD/ID plane alignment.

        Uses current feedback positions:
            z_od_raw = Z0_raw (from AX0)
            z_id_raw = Z1_raw + Z4_raw (AX1 + AX4)
            B14 = z_id_raw - z_od_raw

        Only updates IPC UI/in-memory values. Use "Write" to persist to PLC.
        """
        try:
            cal = self._axis_cal_from_ui()
            act0 = float(self.get_axis_copy(0).act_pos)
            act1 = float(self.get_axis_copy(1).act_pos)
            act4 = float(self.get_axis_copy(4).act_pos)

            z0_raw = cal.abs_to_z_raw(0, act0)
            z1_raw = cal.abs_to_z_raw(1, act1)
            z4_raw = cal.abs_to_z_raw(4, act4)
            zid_raw = z1_raw + z4_raw

            cal.b14 = float(zid_raw - z0_raw)
            self._axis_cal_set_field_status(["b14"], "已标定/未写入")
            self.axis_cal = cal
            self._axis_cal_to_ui(cal)
            self.axis_cal_refresh_status()
            print(
                "[axis_cal] calibrate B14: "
                f"z0_raw={z0_raw:.6f} zid_raw={zid_raw:.6f} -> b14={cal.b14:.6f}"
            )
        except Exception as e:
            print(f"[axis_cal] calibrate B14 failed: {e}")

    def axis_cal_calibrate_keepout(self) -> None:
        """Calibrate AX2 keepout parameters (b2, keepout_w) based on current AX0/AX1/AX2 positions.

        Intended workflow:
        - Move AX2 (center clamp) to the working position you want to bind.
        - Move AX0 to one keepout boundary and AX1 (or AX1+AX4 combined) to the other boundary.
        - Click this button to compute:
            keepout_w = (z_high - z_low) / 2
            b2        = z_center - z2_raw

        Notes:
        - Uses Z_raw coordinates.
        - Only updates IPC UI/in-memory values. Use "Write" to persist to PLC.
        """
        try:
            cal = self._axis_cal_from_ui()

            act0 = float(self.get_axis_copy(0).act_pos)
            act1 = float(self.get_axis_copy(1).act_pos)
            act2 = float(self.get_axis_copy(2).act_pos)

            z0_raw = cal.abs_to_z_raw(0, act0)
            z1_raw = cal.abs_to_z_raw(1, act1)
            z2_raw = cal.abs_to_z_raw(2, act2)

            z_low = min(z0_raw, z1_raw)
            z_high = max(z0_raw, z1_raw)

            zc = 0.5 * (z_low + z_high)
            w = 0.5 * (z_high - z_low)

            cal.keepout_w = float(abs(w))
            cal.b2 = float(zc - z2_raw)

            self._axis_cal_set_field_status(["b2", "keepout_w"], "已标定/未写入")
            self.axis_cal = cal
            self._axis_cal_to_ui(cal)
            self.axis_cal_refresh_status()

            print(
                "[axis_cal] calibrate keepout: "
                f"z0_raw={z0_raw:.6f} z1_raw={z1_raw:.6f} z2_raw={z2_raw:.6f} "
                f"-> z_low={z_low:.6f} z_high={z_high:.6f} "
                f"b2={cal.b2:.6f} keepout_w={cal.keepout_w:.6f}"
            )
        except Exception as e:
            print(f"[axis_cal] calibrate keepout failed: {e}")

    def axis_cal_set_zpos_zero(self) -> None:
        """Set IPC-only z_pos so that current OD plane shows Z_disp == 0.

        z_pos is defined as a UI shift:
            z_disp = z_raw - z_pos
        Thus setting z_pos = current z_od_raw makes current z_od_disp == 0.
        """
        try:
            cal = self._axis_cal_from_ui()
            act0 = float(self.get_axis_copy(0).act_pos)
            z0_raw = cal.abs_to_z_raw(0, act0)
            cal.z_pos = float(z0_raw)
            self._axis_cal_set_field_status(["z_pos"], "已设置/未写入")
            self.axis_cal = cal
            self._axis_cal_to_ui(cal)
            self.axis_cal_refresh_status()
            print(f"[axis_cal] set z_pos: z_pos={cal.z_pos:.6f} (OD disp -> 0)")
        except Exception as e:
            print(f"[axis_cal] set z_pos failed: {e}")

    def axis_cal_refresh_status(self) -> None:
        """Refresh read-only display block on the AxisCal screen.

        This is a pure UI helper that computes current Z_raw/Z_disp from snapshots.
        It is safe to call frequently.
        """
        v = getattr(self, "axis_cal_status_vars", None)
        if not isinstance(v, dict):
            return

        try:
            cal = self._axis_cal_from_ui()
        except Exception:
            cal = getattr(self, "axis_cal", AxisCal())

        try:
            act0 = float(self.get_axis_copy(0).act_pos)
            act1 = float(self.get_axis_copy(1).act_pos)
            act2 = float(self.get_axis_copy(2).act_pos)
            act4 = float(self.get_axis_copy(4).act_pos)
        except Exception:
            return

        # Current Z_raw
        z0_raw = cal.abs_to_z_raw(0, act0)
        z1_raw = cal.abs_to_z_raw(1, act1)
        z2_raw = cal.abs_to_z_raw(2, act2)
        z4_raw = cal.abs_to_z_raw(4, act4)
        zid_raw = z1_raw + z4_raw

        # Current Z_disp
        z0_disp = cal.z_raw_to_z_disp(z0_raw)
        z1_disp = cal.z_raw_to_z_disp(z1_raw)
        z2_disp = cal.z_raw_to_z_disp(z2_raw)
        z4_disp = cal.z_raw_to_z_disp(z4_raw)
        zid_disp = cal.z_raw_to_z_disp(zid_raw)

        # Alignment check (OD/ID planes)
        # When aligned: (Z_id_raw - Z_od_raw) ~= B14
        delta = (zid_raw - z0_raw) - float(cal.b14)
        tol = 0.50  # mm, pragmatic default
        aligned = abs(delta) <= tol

        # Keepout bounds (derived from live AX2 feedback)
        z_center = z2_raw + float(getattr(cal, 'b2', 0.0))
        w = float(getattr(cal, 'keepout_w', 0.0))
        z_low_k = z_center - w
        z_high_k = z_center + w
        z_low_disp = cal.z_raw_to_z_disp(z_low_k)
        z_high_disp = cal.z_raw_to_z_disp(z_high_k)

        try:
            v["off_abs"].set(
                "已标定 Off(abs@Z_raw=0): "
                f"AX0={cal.off_ax0:.3f}  AX1={cal.off_ax1:.3f}  AX2={cal.off_ax2:.3f}  AX4={cal.off_ax4:.3f}"
            )
            v["act_abs"].set(
                "当前 Act_Pos(abs): "
                f"AX0={act0:.3f}  AX1={act1:.3f}  AX2={act2:.3f}  AX4={act4:.3f}"
            )
            # Soft limits (absolute position) are now polled inside AXIS_Ctrl block:
            #   Softlim_pos @ OFF 52 (D152 for AX0)
            #   Softlim_neg @ OFF 56 (D156 for AX0)
            try:
                pos_parts = []
                neg_parts = []
                for ax in LINEAR_AXES:
                    ac = self.get_axis_copy(ax)
                    try:
                        p = float(getattr(ac, "softlim_pos", float("nan")))
                        n = float(getattr(ac, "softlim_neg", float("nan")))
                        if p == p and n == n:  # not NaN
                            pos_parts.append(f"AX{ax}={p:.3f}")
                            neg_parts.append(f"AX{ax}={n:.3f}")
                        else:
                            pos_parts.append(f"AX{ax}=--")
                            neg_parts.append(f"AX{ax}=--")
                    except Exception:
                        pos_parts.append(f"AX{ax}=--")
                        neg_parts.append(f"AX{ax}=--")
                v["softlim_pos"].set("软限位+(abs): " + "  ".join(pos_parts) if pos_parts else "软限位+(abs): -")
                v["softlim_neg"].set("软限位-(abs): " + "  ".join(neg_parts) if neg_parts else "软限位-(abs): -")
            except Exception:
                pass

            v["z_raw"].set(
                "当前 Z_raw(mm): "
                f"Z0={z0_raw:.3f}  Z1={z1_raw:.3f}  Z2={z2_raw:.3f}  Z4={z4_raw:.3f}  Zid={zid_raw:.3f}"
            )
            v["keepout_raw"].set(
                "避让区 Z_raw(mm): "
                f"Zc={z_center:.3f}  W={w:.3f}  z_low={z_low_k:.3f}  z_high={z_high_k:.3f}"
            )
            v["keepout_disp"].set(
                "避让区 Z_disp(mm): "
                f"z_low={z_low_disp:.3f}  z_high={z_high_disp:.3f}"
            )


            if aligned:
                v["z_disp"].set(
                    "当前 Z_disp(mm): "
                    f"Zod={z0_disp:.3f}  Zid={zid_disp:.3f}  (Δ={(delta):+.3f}mm)"
                )
            else:
                v["z_disp"].set(
                    "OD与ID测量截面未对齐  "
                    f"(Δ={(delta):+.3f}mm)  "
                    f"Zod_disp={z0_disp:.3f}  Zid_disp={zid_disp:.3f}"
                )
        except Exception:
            # Never crash UI for status updates
            pass

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
        """Set current OD plane (AX0) as Z_disp = 0 by updating IPC z_pos."""
        ac0 = self.get_axis_copy(0)
        z_raw = self.axis_cal.abs_to_z_raw(0, ac0.act_pos)
        self.axis_cal.z_pos = float(z_raw)
        self._refresh_recipe_table()
        self._refresh_teach_pos()

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

    def _on_teach_axes_selected(self, _evt=None):
        """Teach axes mode combobox changed.

        Modes:
          0=OD(AX0)
          1=ID(AX1+AX4)
          2=OD+ID(AX0+AX1+AX4)
          3=Center clamp(AX2)
        """
        try:
            i = int(self.teach_axes_combo.current())
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

        - Normal modes (0/1/2): section-based teach actions (move to selected / update selected).
        - Center clamp mode (3): repurpose the first two buttons as:
            1) Move AX2 to 'length measurement' position
            2) Move AX2 to 'rotation measurement' position
        """
        if (not hasattr(self, "teach_btn_move")) or (not hasattr(self, "teach_btn_update")):
            return

        try:
            mode = int(getattr(self.recipe, "teach_axes_mode", 2))
        except Exception:
            mode = 2

        if mode == 3:
            self.teach_btn_move.configure(
                text="移动到长度测量位",
                command=self._teach_move_ax2_to_len_pos,
            )
            self.teach_btn_update.configure(
                text="移动到旋转测量位",
                command=self._teach_move_ax2_to_rot_pos,
            )
        else:
            self.teach_btn_move.configure(
                text="移动示教轴到选中截面",
                command=self._teach_move_to_selected,
            )
            self.teach_btn_update.configure(
                text="将当前示教轴位置更新",
                command=self._teach_save_current_to_selected,
            )

    def _teach_move_ax2_to_len_pos(self) -> None:
        """Move AX2 to the saved 'length measurement' position."""
        try:
            if not bool(getattr(self.recipe, "ax2_len_valid", False)):
                messagebox.showwarning("中心架位置", "长度测量位尚未设置：请先点击“保存为长度测量位”。")
                return
            a = float(getattr(self.recipe, "ax2_len_abs", 0.0))
            self.movea_abs(2, a)
        except Exception as e:
            messagebox.showerror("中心架移动失败", str(e))

    def _teach_move_ax2_to_rot_pos(self) -> None:
        """Move AX2 to the saved 'rotation measurement' position."""
        try:
            if not bool(getattr(self.recipe, "ax2_rot_valid", False)):
                messagebox.showwarning("中心架位置", "旋转测量位尚未设置：请先点击“保存为旋转测量位”。")
                return
            a = float(getattr(self.recipe, "ax2_rot_abs", 0.0))
            self.movea_abs(2, a)
        except Exception as e:
            messagebox.showerror("中心架移动失败", str(e))
    def _recipe_apply_from_ui(self) -> Recipe:
        """Read recipe fields from UI into self.recipe (and return a copy)."""
        r = Recipe()
        r.name = self.recipe_name_var.get().strip() or "默认配方"
        r.pipe_len_mm = float(self.pipe_len_var.get())
        r.clamp_occupy_mm = float(self.clamp_var.get())
        r.margin_head_mm = float(self.margin_h_var.get())
        r.margin_tail_mm = float(self.margin_t_var.get())
        r.section_count = int(float(self.section_n_var.get()))
        # 扫描轴（用于自动流程）固定为 AX0；示教轴单独用 teach_axes_mode 控制
        r.scan_axis = 0
        try:
            r.teach_axes_mode = int(self.teach_axes_mode_var.get())
        except Exception:
            r.teach_axes_mode = int(getattr(self.recipe, 'teach_axes_mode', 2))
        r.od_std_mm = float(self.od_std_var.get())
        r.id_std_mm = float(self.id_std_var.get())
        r.od_tol_mm = float(self.od_tol_var.get())
        r.points_per_rev = int(float(self.points_per_rev_var.get()))
        r.min_bin_coverage = float(self.min_cov_var.get())
        r.sample_timeout_s = float(self.sample_timeout_var.get())
        r.max_revolutions = float(self.max_revs_var.get())

        # fit strategy
        try:
            r.fit_strategy = str(self.fit_strategy_var.get())
        except Exception:
            r.fit_strategy = str(getattr(self.recipe, "fit_strategy", "b 原始点按bin权重均衡"))

        # keep existing taught positions when section_count matches
        if len(getattr(self.recipe, 'section_pos_z', [])) == r.section_count:
            r.section_pos_z = list(self.recipe.section_pos_z)
        else:
            r.section_pos_z = r.compute_default_positions_z()

        # keep legacy aligned (deprecated)
        r.section_pos_ui = list(r.section_pos_z)

        # Standby point (待定点) is not edited in the recipe UI inputs, keep current value.
        try:
            r.standby_valid = bool(getattr(self.recipe, "standby_valid", False))
            r.standby_ax0_abs = float(getattr(self.recipe, "standby_ax0_abs", 0.0))
            r.standby_ax1_abs = float(getattr(self.recipe, "standby_ax1_abs", 0.0))
            r.standby_ax4_abs = float(getattr(self.recipe, "standby_ax4_abs", 0.0))
        except Exception:
            pass


        # Center clamp positions (AX2) are set via buttons; keep current value.
        try:
            r.ax2_len_valid = bool(getattr(self.recipe, 'ax2_len_valid', False))
            r.ax2_len_abs = float(getattr(self.recipe, 'ax2_len_abs', 0.0))
            r.ax2_rot_valid = bool(getattr(self.recipe, 'ax2_rot_valid', False))
            r.ax2_rot_abs = float(getattr(self.recipe, 'ax2_rot_abs', 0.0))
        except Exception:
            pass
        # save back
        self.recipe = r
        return r

    def _recipe_compute(self):
        try:
            r = self._recipe_apply_from_ui()
            r.section_pos_z = r.compute_default_positions_z()
            self.recipe.section_pos_z = list(r.section_pos_z)
            self.recipe.section_pos_ui = list(self.recipe.section_pos_z)
            self._refresh_recipe_table()
            self._refresh_auto_std_panel()

            try:
                r = self.get_recipe_copy()
                log("AUTO_START", section_count=getattr(r,'section_count',None), points_per_rev=getattr(r,'points_per_rev',None), min_bin_coverage=getattr(r,'min_bin_coverage',None), timeout_s=getattr(r,'sample_timeout_s',None), max_revolutions=getattr(r,'max_revolutions',None))
            except Exception:
                log("AUTO_START", section_count="?")
        except Exception as e:
            messagebox.showerror("配方计算错误", str(e))

    
    # -------------------------
    # Recipe store (backend)
    # -------------------------
    def _recipe_store_init(self) -> None:
        """Initialize recipe dropdown and auto-load last recipe."""
        self._recipe_refresh_dropdown()
        # auto load last
        last = None
        try:
            idx = self.recipe_store.load_index()
            last = str(idx.get("last_recipe", "")).strip() or None
        except Exception:
            last = None

        # prefer last recipe if exists; otherwise keep current UI values
        if last:
            try:
                self._recipe_load_from_store(last, show_msg=False)
                return
            except Exception:
                pass

        # if current name exists in store, load it (for consistency)
        try:
            cur = str(self.recipe_name_var.get()).strip()
            if cur:
                self._recipe_load_from_store(cur, show_msg=False)
        except Exception:
            pass

    def _recipe_refresh_dropdown(self) -> None:
        """Refresh recipe name combobox values."""
        try:
            names = self.recipe_store.list_names()
        except Exception:
            names = []
        if "默认配方" not in names:
            names.insert(0, "默认配方")
        try:
            self.recipe_name_combo["values"] = names
        except Exception:
            pass

    def _on_recipe_selected(self, _evt=None) -> None:
        """Recipe combobox selected -> auto load."""
        name = str(self.recipe_name_var.get()).strip()
        if not name:
            return
        try:
            self._recipe_load_from_store(name, show_msg=False)
        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def _on_recipe_enter(self, _evt=None) -> None:
        """Enter in recipe name: if exists, load; otherwise treat as new name."""
        name = str(self.recipe_name_var.get()).strip()
        if not name:
            return
        try:
            self._recipe_load_from_store(name, show_msg=False)
        except FileNotFoundError:
            return
        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def _recipe_dump_dict(self, r: Recipe) -> dict:
        return {
            "name": r.name,
            "pipe_len_mm": r.pipe_len_mm,
            "clamp_occupy_mm": r.clamp_occupy_mm,
            "margin_head_mm": r.margin_head_mm,
            "margin_tail_mm": r.margin_tail_mm,
            "section_count": r.section_count,
            "scan_axis": r.scan_axis,
            "teach_axes_mode": int(getattr(r, "teach_axes_mode", 2)),
            "od_std_mm": r.od_std_mm,
            "id_std_mm": r.id_std_mm,
            "od_tol_mm": r.od_tol_mm,
            # Sampling params (persisted)
            "points_per_rev": r.points_per_rev,
            "sample_coverage": r.min_bin_coverage,
            "section_timeout_s": r.sample_timeout_s,
            "max_revs": r.max_revolutions,
            "fit_strategy": str(getattr(r, "fit_strategy", "b 原始点按bin权重均衡")),
            # Taught section positions (Z_Pos, mm)
            "section_pos_z": getattr(r, "section_pos_z", []),
            # Standby point (absolute)
            "standby_valid": bool(getattr(r, "standby_valid", False)),
            "standby_ax0_abs": float(getattr(r, "standby_ax0_abs", 0.0)),
            "standby_ax1_abs": float(getattr(r, "standby_ax1_abs", 0.0)),
            "standby_ax4_abs": float(getattr(r, "standby_ax4_abs", 0.0)),
            # Center clamp (AX2) saved positions
            "ax2_len_valid": bool(getattr(r, "ax2_len_valid", False)),
            "ax2_len_abs": float(getattr(r, "ax2_len_abs", 0.0)),
            "ax2_rot_valid": bool(getattr(r, "ax2_rot_valid", False)),
            "ax2_rot_abs": float(getattr(r, "ax2_rot_abs", 0.0)),
        }

    def _recipe_apply_data_to_ui(self, data: dict) -> None:
        """Apply recipe dict to UI vars and internal recipe object (no dialogs)."""
        self.recipe_name_var.set(str(data.get("name", "默认配方")))
        self.pipe_len_var.set(str(data.get("pipe_len_mm", 1700.0)))
        self.clamp_var.set(str(data.get("clamp_occupy_mm", 300.0)))
        self.margin_h_var.set(str(data.get("margin_head_mm", 20.0)))
        self.margin_t_var.set(str(data.get("margin_tail_mm", 20.0)))
        self.section_n_var.set(str(data.get("section_count", 12)))

        # scan_axis fixed to AX0 for this project path
        try:
            _scan_axis = int(data.get("scan_axis", 0))
        except Exception:
            _scan_axis = 0
        self.recipe.scan_axis = 0

        teach_mode = int(data.get("teach_axes_mode", getattr(self.recipe, "teach_axes_mode", 2)))
        try:
            teach_mode = max(0, min(3, teach_mode))
            self.teach_axes_mode_var.set(teach_mode)
            self.teach_axes_combo.current(teach_mode)
        except Exception:
            pass

        self.od_std_var.set(str(data.get("od_std_mm", 187.3)))
        self.id_std_var.set(str(data.get("id_std_mm", 152.7)))
        self.od_tol_var.set(str(data.get("od_tol_mm", 0.1)))

        # points per rev (compatible)
        if "points_per_rev" in data:
            self.points_per_rev_var.set(str(data.get("points_per_rev", 120)))
        else:
            self.points_per_rev_var.set(str(data.get("sample_count", 120)))

        # sampling params (compatible)
        self.min_cov_var.set(
            str(
                data.get(
                    "sample_coverage",
                    data.get("min_bin_coverage", getattr(self.recipe, "min_bin_coverage", 0.95)),
                )
            )
        )
        self.sample_timeout_var.set(
            str(
                data.get(
                    "section_timeout_s",
                    data.get("sample_timeout_s", getattr(self.recipe, "sample_timeout_s", 5.0)),
                )
            )
        )
        self.max_revs_var.set(
            str(
                data.get(
                    "max_revs",
                    data.get("max_revolutions", getattr(self.recipe, "max_revolutions", 2.0)),
                )
            )
        )
        # fit strategy (persisted)
        try:
            fs = str(data.get("fit_strategy", getattr(self.recipe, "fit_strategy", "b 原始点按bin权重均衡")))
            if hasattr(self, "fit_strategy_var"):
                self.fit_strategy_var.set(fs)
            try:
                if hasattr(self, "fit_strategy_combo") and self.fit_strategy_combo is not None:
                    vals = list(self.fit_strategy_combo.cget("values") or [])
                    if fs in vals:
                        self.fit_strategy_combo.current(vals.index(fs))
            except Exception:
                pass
        except Exception:
            pass


        # positions
        pos_z = data.get("section_pos_z", [])
        pos_ui = data.get("section_pos_ui", [])
        if isinstance(pos_z, list) and pos_z:
            self.recipe.section_pos_z = [float(x) for x in pos_z]
        elif isinstance(pos_ui, list) and pos_ui:
            self.recipe.section_pos_z = [float(x) for x in pos_ui]
        else:
            self.recipe.section_pos_z = self.recipe.compute_default_positions_z()

        # keep legacy aligned (deprecated)
        self.recipe.section_pos_ui = list(self.recipe.section_pos_z)

        # standby (absolute)
        try:
            self.recipe.standby_valid = bool(
                data.get("standby_valid", getattr(self.recipe, "standby_valid", False))
            )
            self.recipe.standby_ax0_abs = float(
                data.get("standby_ax0_abs", getattr(self.recipe, "standby_ax0_abs", 0.0))
            )
            self.recipe.standby_ax1_abs = float(
                data.get("standby_ax1_abs", getattr(self.recipe, "standby_ax1_abs", 0.0))
            )
            self.recipe.standby_ax4_abs = float(
                data.get("standby_ax4_abs", getattr(self.recipe, "standby_ax4_abs", 0.0))
            )
        except Exception:
            pass


        # Center clamp (AX2) saved positions
        try:
            self.recipe.ax2_len_valid = bool(data.get("ax2_len_valid", getattr(self.recipe, "ax2_len_valid", False)))
            self.recipe.ax2_len_abs = float(data.get("ax2_len_abs", getattr(self.recipe, "ax2_len_abs", 0.0)))
            self.recipe.ax2_rot_valid = bool(data.get("ax2_rot_valid", getattr(self.recipe, "ax2_rot_valid", False)))
            self.recipe.ax2_rot_abs = float(data.get("ax2_rot_abs", getattr(self.recipe, "ax2_rot_abs", 0.0)))
        except Exception:
            pass
        # commit to self.recipe from UI and refresh UI tables/panels
        self._recipe_apply_from_ui()
        self._refresh_recipe_table()
        self._refresh_auto_std_panel()
        try:
            self._refresh_standby_pos()
        except Exception:
            pass

        try:
            self._refresh_center_positions()
        except Exception:
            pass

    def _recipe_load_from_store(self, name: str, *, show_msg: bool = False) -> None:
        data = self.recipe_store.load(name)
        self._recipe_apply_data_to_ui(data)
        # refresh dropdown in case new files appear
        self._recipe_refresh_dropdown()
        # remember last
        try:
            self.recipe_store.save_index({"last_recipe": str(self.recipe_name_var.get()).strip()})
        except Exception:
            pass
        if show_msg:
            messagebox.showinfo("加载成功", f"已加载配方：{self.recipe_name_var.get()}")

    def _recipe_save_backend(self) -> None:
        try:
            r = self._recipe_apply_from_ui()
            data = self._recipe_dump_dict(r)
            safe = self.recipe_store.save(r.name, data)
            # sync name if sanitized
            if safe != r.name:
                self.recipe_name_var.set(safe)
                try:
                    self.recipe.name = safe
                except Exception:
                    pass
            self._recipe_refresh_dropdown()
            try:
                self.recipe_store.save_index({"last_recipe": safe})
            except Exception:
                pass
            save_path = self.recipe_store.root / f"{safe}.json"
            messagebox.showinfo("保存成功", f"已保存：{save_path}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def _recipe_delete_backend(self) -> None:
        name = str(self.recipe_name_var.get()).strip()
        if not name:
            return
        if name == "默认配方":
            messagebox.showinfo("提示", "默认配方不允许删除。")
            return
        if not messagebox.askyesno("确认删除", f"确定删除配方“{name}”吗？此操作不可恢复。"):
            return
        try:
            self.recipe_store.delete(name)
            self._recipe_refresh_dropdown()
            # reset to default values (do not auto-create file)
            self.recipe_name_var.set("默认配方")
            self._recipe_apply_data_to_ui(self._recipe_dump_dict(Recipe()))
            try:
                self.recipe_store.save_index({"last_recipe": "默认配方"})
            except Exception:
                pass
            messagebox.showinfo("删除成功", f"已删除配方：{name}")
        except Exception as e:
            messagebox.showerror("删除失败", str(e))


    def _recipe_export_json(self):
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
                "teach_axes_mode": int(getattr(r, 'teach_axes_mode', 2)),
                "od_std_mm": r.od_std_mm,
                "id_std_mm": r.id_std_mm,
                "od_tol_mm": r.od_tol_mm,
                # Sampling params (persisted)
                "points_per_rev": r.points_per_rev,
                "sample_coverage": r.min_bin_coverage,
                "section_timeout_s": r.sample_timeout_s,
                "max_revs": r.max_revolutions,
                # Taught section positions (Z_Pos, mm)
                "section_pos_z": getattr(r, "section_pos_z", []),
                # Standby point (absolute)
                "standby_valid": bool(getattr(r, "standby_valid", False)),
                "standby_ax0_abs": float(getattr(r, "standby_ax0_abs", 0.0)),
                "standby_ax1_abs": float(getattr(r, "standby_ax1_abs", 0.0)),
                "standby_ax4_abs": float(getattr(r, "standby_ax4_abs", 0.0)),
                # legacy fields are intentionally omitted (UI_Pos/ui_coord are deprecated)
            }
            with open(path, "w", encoding="utf-8") as f:
                import json

                json.dump(data, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("保存成功", f"已保存：{path}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def _recipe_import_json(self):
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
            # scan_axis is fixed to AX0 for this project path
            scan_axis = int(data.get("scan_axis", 0))
            self.recipe.scan_axis = 0
            teach_mode = int(data.get("teach_axes_mode", getattr(self.recipe, 'teach_axes_mode', 2)))
            try:
                self.teach_axes_mode_var.set(max(0, min(3, teach_mode)))
                self.teach_axes_combo.current(max(0, min(3, teach_mode)))
            except Exception:
                pass
            self.od_std_var.set(str(data.get("od_std_mm", 187.3)))
            self.id_std_var.set(str(data.get("id_std_mm", 152.7)))
            self.od_tol_var.set(str(data.get("od_tol_mm", 0.1)))
            # 每圈采样点数：优先 points_per_rev；兼容旧字段 sample_count（旧版本是“每截面采样(M)”）
            if "points_per_rev" in data:
                self.points_per_rev_var.set(str(data.get("points_per_rev", 120)))
            else:
                self.points_per_rev_var.set(str(data.get("sample_count", 120)))

            # 等角采样参数（缺省值兼容）
            # New keys (f7.1): sample_coverage/section_timeout_s/max_revs
            # Backward compatible with older keys: min_bin_coverage/sample_timeout_s/max_revolutions
            self.min_cov_var.set(
                str(
                    data.get(
                        "sample_coverage",
                        data.get("min_bin_coverage", getattr(self.recipe, "min_bin_coverage", 0.95)),
                    )
                )
            )
            self.sample_timeout_var.set(
                str(
                    data.get(
                        "section_timeout_s",
                        data.get("sample_timeout_s", getattr(self.recipe, "sample_timeout_s", 5.0)),
                    )
                )
            )
            self.max_revs_var.set(
                str(
                    data.get(
                        "max_revs",
                        data.get("max_revolutions", getattr(self.recipe, "max_revolutions", 2.0)),
                    )
                )
            )

            # positions
            pos_z = data.get("section_pos_z", [])
            pos_ui = data.get("section_pos_ui", [])
            if isinstance(pos_z, list) and pos_z:
                self.recipe.section_pos_z = [float(x) for x in pos_z]
            elif isinstance(pos_ui, list) and pos_ui:
                # legacy fallback
                self.recipe.section_pos_z = [float(x) for x in pos_ui]
            else:
                self.recipe.section_pos_z = self.recipe.compute_default_positions_z()

            # keep legacy aligned (deprecated)
            self.recipe.section_pos_ui = list(self.recipe.section_pos_z)

            # standby (absolute)
            try:
                self.recipe.standby_valid = bool(data.get("standby_valid", getattr(self.recipe, "standby_valid", False)))
                self.recipe.standby_ax0_abs = float(data.get("standby_ax0_abs", getattr(self.recipe, "standby_ax0_abs", 0.0)))
                self.recipe.standby_ax1_abs = float(data.get("standby_ax1_abs", getattr(self.recipe, "standby_ax1_abs", 0.0)))
                self.recipe.standby_ax4_abs = float(data.get("standby_ax4_abs", getattr(self.recipe, "standby_ax4_abs", 0.0)))
            except Exception:
                pass

            self._recipe_apply_from_ui()
            self._refresh_recipe_table()
            self._refresh_auto_std_panel()
            try:
                self._refresh_standby_pos()
            except Exception:
                pass
            messagebox.showinfo("加载成功", f"已加载：{path}")
        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def _refresh_recipe_table(self):
        try:
            self.recipe_tree.delete(*self.recipe_tree.get_children())
        except Exception:
            return

        r = self.recipe

        # Ensure positions length (Z_Pos)
        if len(getattr(r, 'section_pos_z', [])) != int(r.section_count):
            r.section_pos_z = r.compute_default_positions_z()

        # Keep legacy aligned (deprecated)
        try:
            r.section_pos_ui = list(r.section_pos_z)
        except Exception:
            pass

        for i, z_od_disp in enumerate(r.section_pos_z):
            z_od_disp = float(z_od_disp)
            # 由 OD 截面位置推导：AX0/AX1/AX4 目标 abs 以及 ID 位置
            try:
                # For section planning, keepout reference should use AX2 rotation measurement position (if set).
                ax2_abs = float(self._get_ax2_keepout_ref_abs(prefer_rot=True))
                # Soft limits (abs) for target solving (OD clamp + ID split)
                softlims = {
                    0: (float(self.get_axis_copy(0).softlim_pos), float(self.get_axis_copy(0).softlim_neg)),
                    1: (float(self.get_axis_copy(1).softlim_pos), float(self.get_axis_copy(1).softlim_neg)),
                    4: (float(self.get_axis_copy(4).softlim_pos), float(self.get_axis_copy(4).softlim_neg)),
                }
                t = self.axis_cal.od_z_disp_to_targets(z_od_disp, ax2_abs=ax2_abs, softlims_abs=softlims)
                ax0_abs = float(t["ax0_abs"])
                ax1_abs = float(t["ax1_abs"])
                ax4_abs = float(t["ax4_abs"])
                z_id_disp = float(t["z_id_disp"])
            except Exception:
                ax0_abs, ax1_abs, ax4_abs, z_id_disp = 0.0, 0.0, 0.0, z_od_disp + float(getattr(self.axis_cal, 'b14', 0.0))

            src = (
                "示教/保留"
                if hasattr(self, "_taught_mark")
                and getattr(self, "_taught_mark", {}).get(i, False)
                else "计算"
            )
            self.recipe_tree.insert(
                "",
                "end",
                values=(
                    i,
                    f"{z_od_disp:.3f}",
                    f"{z_id_disp:.3f}",
                    f"{ax0_abs:.3f}",
                    f"{ax1_abs:.3f}",
                    f"{ax4_abs:.3f}",
                    src,
                ),
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

            z_od_disp = float(r.section_pos_z[idx])
            mode = int(getattr(self.recipe, 'teach_axes_mode', getattr(r, 'teach_axes_mode', 2)))
            # For section moves, keepout reference should use AX2 rotation measurement position (if set).
            ax2_abs = float(self._get_ax2_keepout_ref_abs(prefer_rot=True))

            # Soft limits (abs) for target solving (OD clamp + ID split)
            softlims = {
                0: (float(self.get_axis_copy(0).softlim_pos), float(self.get_axis_copy(0).softlim_neg)),
                1: (float(self.get_axis_copy(1).softlim_pos), float(self.get_axis_copy(1).softlim_neg)),
                4: (float(self.get_axis_copy(4).softlim_pos), float(self.get_axis_copy(4).softlim_neg)),
            }

            t = self.axis_cal.od_z_disp_to_targets(z_od_disp, ax2_abs=ax2_abs, softlims_abs=softlims)

            # Move selected teach axes
            if mode in (0, 2):
                self.movea_abs(0, float(t['ax0_abs']), context='SectionMove')
            if mode in (1, 2):
                self.movea_abs(1, float(t['ax1_abs']), context='SectionMove')
                self.movea_abs(4, float(t['ax4_abs']), context='SectionMove')
        except Exception as e:
            messagebox.showerror("示教移动失败", str(e))

    def _teach_save_current_to_selected(self):
        try:
            r = self._recipe_apply_from_ui()
            idx = self._get_selected_recipe_idx()
            if idx is None:
                messagebox.showwarning("提示", "请先在表格中选中一个截面")
                return

            mode = int(getattr(self.recipe, 'teach_axes_mode', getattr(r, 'teach_axes_mode', 2)))

            ac0 = self.get_axis_copy(0)
            ac1 = self.get_axis_copy(1)
            ac4 = self.get_axis_copy(4)

            z_od_from_od = self.axis_cal.abs_to_z_disp(0, ac0.act_pos)
            # ID composite: Zid_disp = z_raw_to_z_disp(Z1_raw+Z4_raw)
            z1_raw = self.axis_cal.abs_to_z_raw(1, ac1.act_pos)
            z4_raw = self.axis_cal.abs_to_z_raw(4, ac4.act_pos)
            zid_raw = float(z1_raw) + float(z4_raw)
            zid_disp = self.axis_cal.z_raw_to_z_disp(zid_raw)
            z_od_from_id = float(zid_disp) - float(self.axis_cal.b14)

            tol = 0.50
            if mode == 0:
                z_od_disp = float(z_od_from_od)
            elif mode == 1:
                z_od_disp = float(z_od_from_id)
            else:
                # Both selected: prefer OD; if misaligned, keep OD and warn
                z_od_disp = float(z_od_from_od)
                if abs(float(z_od_from_od) - float(z_od_from_id)) > tol:
                    try:
                        self.log("teach: OD/ID not aligned; saving section using OD")
                    except Exception:
                        pass

            r.section_pos_z[idx] = float(z_od_disp)
            self.recipe.section_pos_z = list(r.section_pos_z)
            self.recipe.section_pos_ui = list(self.recipe.section_pos_z)  # legacy

            # mark taught
            if not hasattr(self, "_taught_mark"):
                self._taught_mark = {}
            self._taught_mark[idx] = True

            self._refresh_recipe_table()
            self._refresh_teach_pos()
        except Exception as e:
            messagebox.showerror("示教保存失败", str(e))


    def _keepout_handoff_raw(self, cal: AxisCal) -> float:
        """Keepout handoff boundary (Z_raw) for AX1 forward travel.

        Collision direction (empirical): AX1 moving in negative abs makes Z_raw increase.
        Therefore AX1 maximum allowed Z_raw near AX2 is (Zc + W); any deeper part is
        assigned to AX4.
        """
        z2_raw = cal.abs_to_z_raw(2, self._get_ax2_keepout_ref_abs())
        zc = z2_raw + cal.b2
        w = cal.keepout_w
        return zc + w

    def _teach_align_by_od(self):
        """Align ID plane to OD plane (keep AX0, move AX1/AX4)."""
        try:
            ac0 = self.get_axis_copy(0)
            z0_raw = self.axis_cal.abs_to_z_raw(0, ac0.act_pos)
            z_id_raw_tgt = float(z0_raw) + float(self.axis_cal.b14)

            # split into AX1/AX4 raw by handoff
            if float(z_id_raw_tgt) <= float(self._keepout_handoff_raw(self.axis_cal)):
                z1_raw_tgt = float(z_id_raw_tgt)
                z4_raw_tgt = 0.0
            else:
                z1_raw_tgt = float(self._keepout_handoff_raw(self.axis_cal))
                z4_raw_tgt = float(z_id_raw_tgt) - float(self._keepout_handoff_raw(self.axis_cal))

            self.movea_abs(1, float(self.axis_cal.z_raw_to_abs(1, z1_raw_tgt)))
            self.movea_abs(4, float(self.axis_cal.z_raw_to_abs(4, z4_raw_tgt)))
            self._refresh_teach_pos()
        except Exception as e:
            messagebox.showerror("对齐失败(OD基准)", str(e))

    def _teach_align_by_id(self):
        """Align OD plane to ID plane (keep AX1/AX4, move AX0)."""
        try:
            ac1 = self.get_axis_copy(1)
            ac4 = self.get_axis_copy(4)
            z1_raw = self.axis_cal.abs_to_z_raw(1, ac1.act_pos)
            z4_raw = self.axis_cal.abs_to_z_raw(4, ac4.act_pos)
            zid_raw = float(z1_raw) + float(z4_raw)

            z_od_raw_tgt = float(zid_raw) - float(self.axis_cal.b14)
            self.movea_abs(0, float(self.axis_cal.z_raw_to_abs(0, z_od_raw_tgt)))
            self._refresh_teach_pos()
        except Exception as e:
            messagebox.showerror("对齐失败(ID基准)", str(e))

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
            messagebox.showinfo("待定点", "已保存待定点（请记得保存配方 JSON）")
        except Exception as e:
            messagebox.showerror("待定点保存失败", str(e))

    def _teach_go_standby(self):
        """Move AX0/AX1/AX4 to the stored standby point."""
        try:
            if not bool(getattr(self.recipe, "standby_valid", False)):
                messagebox.showwarning("提示", "待定点尚未设置：请先点击“将当下位置保存为待定位”。")
                return

            a0 = float(getattr(self.recipe, "standby_ax0_abs", 0.0))
            a1 = float(getattr(self.recipe, "standby_ax1_abs", 0.0))
            a4 = float(getattr(self.recipe, "standby_ax4_abs", 0.0))

            # Fire 3 MoveA commands back-to-back (effectively simultaneous)
            self.movea_abs(0, a0)
            self.movea_abs(1, a1)
            self.movea_abs(4, a4)
        except Exception as e:
            messagebox.showerror("回到待定点失败", str(e))

    def _refresh_standby_pos(self):
        """Refresh standby display fields on the teach page."""
        try:
            if not hasattr(self, "standby_info_var"):
                return
            if not bool(getattr(self.recipe, "standby_valid", False)):
                self.standby_state_var.set("未设置")
                self.standby_info_var.set("未设置")
                return

            cal = self.axis_cal
            a0 = float(getattr(self.recipe, "standby_ax0_abs", 0.0))
            a1 = float(getattr(self.recipe, "standby_ax1_abs", 0.0))
            a4 = float(getattr(self.recipe, "standby_ax4_abs", 0.0))

            z0_disp = float(cal.abs_to_z_disp(0, a0))
            z1_raw = float(cal.abs_to_z_raw(1, a1))
            z4_raw = float(cal.abs_to_z_raw(4, a4))
            zid_raw = z1_raw + z4_raw
            zid_disp = float(cal.z_raw_to_z_disp(zid_raw))
            zid_exp = float(z0_disp) + float(cal.b14)
            dz = float(zid_disp) - float(zid_exp)
            aligned = abs(dz) <= 0.50

            self.standby_state_var.set("已设置" + ("（OD/ID对齐）" if aligned else "（OD/ID未对齐）"))
            self.standby_info_var.set(
                "AX0 abs={:.3f}  Z_od={:.3f}\n"
                "AX1 abs={:.3f}  Z1_raw={:.3f}\n"
                "AX4 abs={:.3f}  Z4_raw={:.3f}\n"
                "ID_act={:.3f}  ID_exp={:.3f}  Δ={:.3f}".format(
                    a0,
                    z0_disp,
                    a1,
                    z1_raw,
                    a4,
                    z4_raw,
                    zid_disp,
                    zid_exp,
                    dz,
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
            messagebox.showinfo('中心架位置', '已保存：长度测量位')
        except Exception as e:
            messagebox.showerror('中心架位置保存失败', str(e))

    def _save_ax2_rot_pos(self) -> None:
        """Save current AX2 absolute position as 'rotation measurement' position in recipe."""
        try:
            act2 = float(self.get_axis_copy(2).act_pos)
            self.recipe.ax2_rot_valid = True
            self.recipe.ax2_rot_abs = act2
            self._refresh_center_positions()
            messagebox.showinfo('中心架位置', '已保存：旋转测量位')
        except Exception as e:
            messagebox.showerror('中心架位置保存失败', str(e))

    def _refresh_center_positions(self) -> None:
        """Refresh read-only display for AX2 saved positions on recipe screen."""
        if not hasattr(self, 'center_pos_var'):
            return
        cal = self.axis_cal
        lines = []

        if bool(getattr(self.recipe, 'ax2_len_valid', False)):
            a = float(getattr(self.recipe, 'ax2_len_abs', 0.0))
            z = float(cal.abs_to_z_disp(2, a))
            lines.append(f"长度测量位: abs={a:.3f}  Z_disp={z:.3f}")
        else:
            lines.append('长度测量位: 未设置')

        if bool(getattr(self.recipe, 'ax2_rot_valid', False)):
            a = float(getattr(self.recipe, 'ax2_rot_abs', 0.0))
            z = float(cal.abs_to_z_disp(2, a))
            lines.append(f"旋转测量位: abs={a:.3f}  Z_disp={z:.3f}")
        else:
            lines.append('旋转测量位: 未设置')

        self.center_pos_var.set('\n'.join(lines))

    def _get_ax2_keepout_ref_abs(self, prefer_rot: bool = True) -> float:
        """AX2 absolute position used as reference for keepout computation.

        prefer_rot=True: use saved AX2 rotation measurement position (recipe.ax2_rot_abs) when valid,
                        otherwise fall back to current AX2 actual position.
        prefer_rot=False: always use current AX2 actual position.
        """
        if prefer_rot:
            try:
                if bool(getattr(self.recipe, 'ax2_rot_valid', False)):
                    return float(getattr(self.recipe, 'ax2_rot_abs', 0.0))
            except Exception:
                pass
        try:
            return float(self.get_axis_copy(2).act_pos)
        except Exception:
            return 0.0
    def _ctx_use_ax2_rot_ref(self, context: str) -> bool:
        ctx = (context or '').lower()
        return (
            'section' in ctx
            or 'auto_sec' in ctx
            or 'recipe' in ctx
            or 'teach_sec' in ctx
            or 'sec_' in ctx
        )

    def _teach_move_relative(self):
        """Relative move for selected teach axes in Z_disp (mm)."""
        try:
            try:
                dz = float(self.teach_rel_dist_var.get())
            except Exception:
                dz = 0.0

            mode = int(getattr(self.recipe, 'teach_axes_mode', 2))

            # Center clamp AX2
            if mode == 3:
                ac2 = self.get_axis_copy(2)
                z2_disp = float(self.axis_cal.abs_to_z_disp(2, ac2.act_pos))
                z2_tgt_disp = z2_disp + dz
                self.movea_abs(2, float(self.axis_cal.z_disp_to_abs(2, z2_tgt_disp)), context="TeachRel")

            # OD
            if mode in (0, 2):
                ac0 = self.get_axis_copy(0)
                z0_disp = float(self.axis_cal.abs_to_z_disp(0, ac0.act_pos))
                z0_tgt_disp = z0_disp + dz
                self.movea_abs(0, float(self.axis_cal.z_disp_to_abs(0, z0_tgt_disp)), context="TeachRel")

            # ID composite (AX1 + AX4): equal split, overflow to AX4 when AX1 hits keepout/soft limit
            if mode in (1, 2):
                cal = self.axis_cal
                ac1 = self.get_axis_copy(1)
                ac4 = self.get_axis_copy(4)

                z1_raw = float(cal.abs_to_z_raw(1, ac1.act_pos))
                z4_raw = float(cal.abs_to_z_raw(4, ac4.act_pos))
                zid_raw = z1_raw + z4_raw

                zid_disp = float(cal.z_raw_to_z_disp(zid_raw))
                zid_tgt_disp = zid_disp + dz
                zid_tgt_raw = float(cal.z_disp_to_z_raw(zid_tgt_disp))

                # --- raw ranges from soft limits (abs) ---
                def _raw_range(ax: int):
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

                lo1, hi1 = _raw_range(1)
                lo4, hi4 = _raw_range(4)

                # --- keepout (use AX2 rotation measurement ref when available) ---
                ax2_abs_ref = float(self.get_axis_copy(2).act_pos)
                try:
                    z2_raw_ref = float(cal.abs_to_z_raw(2, ax2_abs_ref))
                    zc = float(z2_raw_ref + cal.b2)
                    w = float(cal.keepout_w)
                    if abs(w) >= 1e-6:
                        hi1 = min(hi1, float(zc + w))
                except Exception:
                    pass

                # --- equal split + overflow ---
                delta_raw = float(zid_tgt_raw) - float(zid_raw)
                z1_des = float(z1_raw) + 0.5 * delta_raw
                z1_tgt = max(float(lo1), min(float(hi1), float(z1_des)))
                used1 = float(z1_tgt) - float(z1_raw)

                rem = float(delta_raw) - float(used1)
                z4_des = float(z4_raw) + float(rem)
                z4_tgt = max(float(lo4), min(float(hi4), float(z4_des)))

                self.movea_abs(1, float(cal.z_raw_to_abs(1, z1_tgt)), context="TeachRel")
                self.movea_abs(4, float(cal.z_raw_to_abs(4, z4_tgt)), context="TeachRel")

            self._refresh_teach_pos()
        except Exception as e:
            messagebox.showerror("相对运动失败", str(e))

    def _teach_jog_hold(self, direction: str, on: bool):
        """Jog for teach panel (press-and-hold).

        - OD: AX0
        - ID: AX1 + AX4 (equal split; when AX1 hits keepout/soft limit, AX4 continues)
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

                # keepout upper bound for AX1 (use AX2 rotation measurement ref when available)
                keepout_hi = float('inf')
                try:
                    ax2_abs_ref = float(self.get_axis_copy(2).act_pos)
                    z2_raw_ref = float(cal.abs_to_z_raw(2, ax2_abs_ref))
                    zc = float(z2_raw_ref + cal.b2)
                    w = float(cal.keepout_w)
                    if abs(w) >= 1e-6:
                        keepout_hi = float(zc + w)
                except Exception:
                    pass

                lo1, hi1 = _raw_range(1)
                lo4, hi4 = _raw_range(4)
                hi1 = min(float(hi1), float(keepout_hi))

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

        z0_raw = float(cal.abs_to_z_raw(0, act0))
        z1_raw = float(cal.abs_to_z_raw(1, act1))
        z4_raw = float(cal.abs_to_z_raw(4, act4))
        z2_raw = float(cal.abs_to_z_raw(2, act2))
        zid_raw = z1_raw + z4_raw

        z0_disp = float(cal.z_raw_to_z_disp(z0_raw))
        zid_disp = float(cal.z_raw_to_z_disp(zid_raw))
        z2_disp = float(cal.z_raw_to_z_disp(z2_raw))
        z_id_expect_raw = z0_raw + float(cal.b14)
        z_id_expect_disp = float(cal.z_raw_to_z_disp(z_id_expect_raw))

        delta = float(zid_raw) - float(z_id_expect_raw)
        tol = 0.50
        aligned = abs(delta) <= tol

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

        # standby display
        try:
            self._refresh_standby_pos()
        except Exception:
            pass

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

    def _on_sim_disp_toggle(self):
        """Toggle simulated displacement meter (ID).

        Current phase: simulation only.
        """
        try:
            self.sim_disp_enabled = bool(self.sim_disp_var.get())
        except Exception:
            self.sim_disp_enabled = False

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
            # if serial is None:
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
            # create a new RunId/Serial (流水号) for this measurement
            self._prepare_new_run()
            self._auto_thread = AutoFlow(self)
            self._auto_thread.start()
        except Exception as e:
            messagebox.showerror("启动失败", str(e))

    def _auto_stop(self):
        try:
            log("AUTO_STOP")
            if self._auto_thread and self._auto_thread.is_alive():
                self._auto_thread.stop()
                # Immediately stop axis motions on PLC side to avoid "in-position timeout" -> ERR.
                self.abort_motion()
        except Exception:
            pass

    def _auto_clear_ui(self):
        self.result_tree.delete(*self.result_tree.get_children())
        try:
            self._result_iids.clear()
        except Exception:
            self._result_iids = []
        self.straight_var.set("直线度：--（外圆）/ --（内圆） || 整体同心度：--")
        self.cov_var.set("采样覆盖率：--")
        # clear per-section coverage cache & selections
        try:
            self._section_cov_info.clear()
        except Exception:
            self._section_cov_info = {}
        self._auto_cur_sec_idx = None
        self._selected_sec_idx = None
        self._axis_dist = None
        self.auto_progress_var.set("当前截面: - / 总截面: -")
        self.auto_done_var.set("测量完成: 否")
        # clear run data caches
        try:
            self._auto_rows.clear()
            self._auto_raw_points.clear()
        except Exception:
            self._auto_rows = []
            self._auto_raw_points = []
        self._auto_export_done = False

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
            # Defensive copy: snapshot may carry extra attrs (setattr) across protocol revisions.
            try:
                allowed = set(AxisComm.__dataclass_fields__.keys())
                data = {k: v for k, v in ac.__dict__.items() if k in allowed}
                return AxisComm(**data)
            except Exception:
                # Fallback to a plain construction to avoid UI crash.
                return AxisComm()

    def get_recipe_copy(self) -> Recipe:
        # minimal deep copy
        r = self.recipe
        rr = Recipe(**{k: getattr(r, k) for k in r.__dataclass_fields__.keys()})
        rr.section_pos_ui = list(r.section_pos_ui)
        rr.section_pos_z = list(getattr(r, "section_pos_z", []) or [])
        return rr

    # =========================
    # Low-level write helpers
    # =========================
    def _base(self, axis: int) -> int:
        return axis_base(axis)

    def _write_regs(self, d_addr: int, values: List[int]):
        self.cmd_q.put(CmdWriteRegs(d_addr=d_addr, values=values))

    def _read_regs_sync(self, d_addr: int, count: int, timeout_s: float = 0.35) -> Optional[List[int]]:
        """Synchronous Modbus holding-register read via PlcWorker.

        This is used by AutoFlow to obtain a tighter snapshot for binding samples:
        (theta from AX3 act_pos, ID from CL OUT3) at the moment an OD sample arrives.
        """
        tag = f"sync:{time.time_ns()}"
        evt = threading.Event()
        with self._sync_reads_lock:
            self._sync_reads[tag] = {"evt": evt, "regs": None}
        try:
            self.cmd_q.put(CmdReadRegs(d_addr, int(count), tag))
        except Exception:
            with self._sync_reads_lock:
                self._sync_reads.pop(tag, None)
            return None

        if not evt.wait(float(timeout_s)):
            try:
                log("SYNC_READ_TIMEOUT", d_addr=d_addr, count=count, timeout_s=timeout_s)
            except Exception:
                pass
            with self._sync_reads_lock:
                self._sync_reads.pop(tag, None)
            return None

        with self._sync_reads_lock:
            slot = self._sync_reads.pop(tag, None)
        if not slot:
            return None
        regs = slot.get("regs", None)
        try:
            if regs is not None:
                log("SYNC_READ_OK", d_addr=d_addr, count=count)
        except Exception:
            pass
        if regs is None:
            return None
        try:
            return list(regs)
        except Exception:
            return None

    def _decode_fp64_4regs(self, regs: List[int]) -> float:
        try:
            return float(decode_float64_from_4regs(list(regs[:4]), FLOAT64_WORD_ORDER))
        except Exception:
            return 0.0

    def read_axis_act_pos_deg_sync(self, axis: int = 3, timeout_s: float = 0.35) -> Optional[float]:
        """Read AXn act_pos (FP64) on-demand and return degrees in [0, 360)."""
        try:
            base = self._base(int(axis))
            regs = self._read_regs_sync(base + OFF_ACT_POS, 4, timeout_s=timeout_s)
            if not regs:
                return None
            v = self._decode_fp64_4regs(regs)
            return float(v) % 360.0
        except Exception:
            return None

    def read_cl_out3_sync(self, timeout_s: float = 0.35) -> Tuple[Optional[float], Optional[int], Optional[int]]:
        """Read CL OUT3 (DINT32) and update counter (UINT32) on-demand.

        Returns: (id_mm, raw_dint, upd_cnt)
        id_mm is None when raw indicates invalid/standby/over-range or read fails.
        """
        try:
            regs = self._read_regs_sync(CL_IN_BASE_D + CL_OUT3_WORD_OFF, 2, timeout_s=timeout_s)
            regs2 = self._read_regs_sync(CL_IN_BASE_D + CL_OUT3_UPD_WORD_OFF, 2, timeout_s=timeout_s)
            raw = None
            cnt = None
            if regs and len(regs) >= 2:
                u32 = int(regs[0] & 0xFFFF) | (int(regs[1] & 0xFFFF) << 16)
                raw = u32 - 0x100000000 if (u32 & 0x80000000) else u32
                raw = int(raw)
            if regs2 and len(regs2) >= 2:
                cnt = int(regs2[0] & 0xFFFF) | (int(regs2[1] & 0xFFFF) << 16)

            id_mm = None
            if raw is not None and raw not in {CL_OUT_INVALID, CL_OUT_STANDBY, CL_OUT_POS_OVER, CL_OUT_NEG_OVER}:
                id_mm = float(raw) * float(CL_OUT_SCALE_MM)
            return (id_mm, raw, cnt)
        except Exception:
            return (None, None, None)


    def set_cmd_bits(self, axis: int, set_mask: int = 0, clr_mask: int = 0):
        self.cmd_q.put(CmdSetCmdMask(axis=axis, set_mask=set_mask, clr_mask=clr_mask))

    def _pulse_cmd_bits(self, axis: int, pulse_mask: int, pulse_ms: int = 120):
        self.cmd_q.put(
            CmdPulseCmdMask(axis=axis, pulse_mask=pulse_mask, pulse_ms=pulse_ms)
        )


    def _parse_float(self, s: str, default: float) -> float:
        try:
            return float(str(s).strip())
        except Exception:
            return float(default)

    def _read_axis_params_from_ui(self) -> tuple[float, float, float, float, int, float, float, float]:
        """Read per-axis motion parameters from UI entries.

        Returns:
            (vel_movea, vel_mover, vel_jog, vel_velmove, dir_mover, acc, dec, jerk)
        """
        # New UI (recommended)
        if hasattr(self, 'ent_vel_movea'):
            vel_movea = self._parse_float(getattr(self, 'ent_vel_movea').get(), 100.0)
            vel_mover = self._parse_float(getattr(self, 'ent_vel_mover').get(), vel_movea)
            vel_jog = self._parse_float(getattr(self, 'ent_vel_jog').get(), 80.0)
            vel_velmove = self._parse_float(getattr(self, 'ent_vel_velmove').get(), 200.0)
            acc = self._parse_float(getattr(self, 'ent_acc').get(), 200.0)
            dec = self._parse_float(getattr(self, 'ent_dec').get(), 200.0)
            jerk = self._parse_float(getattr(self, 'ent_jerk').get(), 500.0)

            dir_mover = DIR_NONE
            if hasattr(self, 'dir_mover_var'):
                try:
                    dir_mover = int(getattr(self, 'dir_mover_var').get())
                except Exception:
                    dir_mover = DIR_NONE
            elif hasattr(self, 'cmb_dir_mover'):
                try:
                    txt = str(getattr(self, 'cmb_dir_mover').get())
                    dir_mover = int(txt.split(':')[0].strip())
                except Exception:
                    dir_mover = DIR_NONE

            return (
                float(vel_movea),
                float(vel_mover),
                float(vel_jog),
                float(vel_velmove),
                int(dir_mover),
                float(acc),
                float(dec),
                float(jerk),
            )

        # Legacy UI fallback: one vel + acc/dec/jerk
        vel = self._parse_float(getattr(self, 'ent_vel').get(), 100.0) if hasattr(self, 'ent_vel') else 100.0
        acc = self._parse_float(getattr(self, 'ent_acc').get(), 200.0) if hasattr(self, 'ent_acc') else 200.0
        dec = self._parse_float(getattr(self, 'ent_dec').get(), 200.0) if hasattr(self, 'ent_dec') else 200.0
        jerk = self._parse_float(getattr(self, 'ent_jerk').get(), 500.0) if hasattr(self, 'ent_jerk') else 500.0
        return float(vel), float(vel), float(vel), float(vel), DIR_NONE, float(acc), float(dec), float(jerk)

    def _write_axis_params(self, axis: int):
        """Write motion parameters into Axis_Ctrl (FP64 + Dir word)."""
        axis = max(0, min(AXIS_COUNT - 1, int(axis)))
        (
            vel_movea,
            vel_mover,
            vel_jog,
            vel_velmove,
            dir_mover,
            acc,
            dec,
            jerk,
        ) = self._read_axis_params_from_ui()

        base = self._base(axis)

        # Dir_MoveR (UINT)
        self._write_regs(base + OFF_DIR_MOVER, [int(dir_mover) & 0xFFFF])

        # FP64 setpoints
        self._write_regs(base + OFF_VEL_MOVEA, encode_float64_to_4regs(vel_movea, FLOAT64_WORD_ORDER))
        self._write_regs(base + OFF_VEL_MOVER, encode_float64_to_4regs(vel_mover, FLOAT64_WORD_ORDER))
        self._write_regs(base + OFF_VEL_JOG, encode_float64_to_4regs(vel_jog, FLOAT64_WORD_ORDER))
        self._write_regs(base + OFF_VEL_VELMOVE, encode_float64_to_4regs(vel_velmove, FLOAT64_WORD_ORDER))
        self._write_regs(base + OFF_ACC, encode_float64_to_4regs(acc, FLOAT64_WORD_ORDER))
        self._write_regs(base + OFF_DEC, encode_float64_to_4regs(dec, FLOAT64_WORD_ORDER))
        self._write_regs(base + OFF_JERK, encode_float64_to_4regs(jerk, FLOAT64_WORD_ORDER))

    # Backward compatible helper (legacy name)
    def _read_common_params(self) -> tuple[float, float, float, float]:
        vel_movea, _, vel_jog, _, _, acc, dec, jerk = self._read_axis_params_from_ui()
        # legacy returns (vel, acc, dec, jerk)
        return float(vel_movea), float(acc), float(dec), float(jerk)

    def _write_common_params(self):
        ax = self._axis()
        self._write_axis_params(ax)



    def apply_soft_limits_abs(
        self,
        axis: int,
        target_abs: float,
        *,
        strict: bool = False,
        context: str = "",
    ) -> float:
        """Apply motion safety limits to an absolute target position.

        Limits applied (when available):
        1) Axis soft limits (AX0/1/2/4): clamp to [min(softlim_pos, softlim_neg), max(...)].
        2) Dynamic keepout vs AX2 (AX0 and AX1 only):
           - AX0: moving +abs => Z_raw decreases; forbid Z0_raw < (Zc - W) => abs must be <= abs_at(Zc - W)
           - AX1: moving -abs => Z_raw increases; forbid Z1_raw > (Zc + W) => abs must be >= abs_at(Zc + W)

        - strict=True: raise RuntimeError when out-of-range (AutoFlow)
        - strict=False: clamp and log (manual operations)
        """
        ax = int(axis)
        t = float(target_abs)
        if ax not in LINEAR_AXES:
            return t

        # ---------------- soft limits ----------------
        lo = -float('inf')
        hi = float('inf')
        ac = self.get_axis_copy(ax)
        try:
            p = float(getattr(ac, 'softlim_pos', float('nan')))
            n = float(getattr(ac, 'softlim_neg', float('nan')))
        except Exception:
            p = float('nan')
            n = float('nan')

        if (p == p) and (n == n) and (abs(p) + abs(n) >= 1e-6):
            lo, hi = (min(p, n), max(p, n))
            if hi - lo < 1e-9:
                lo, hi = (-float('inf'), float('inf'))

        # ---------------- dynamic keepout ----------------
        try:
            if ax in (0, 1):
                ax2_abs = float(self.get_axis_copy(2).act_pos) if self._ctx_use_ax2_rot_ref(context) else float(self.get_axis_copy(2).act_pos)
                z2_raw = float(self.axis_cal.abs_to_z_raw(2, ax2_abs))
                zc = float(z2_raw + self.axis_cal.b2)
                w = float(self.axis_cal.keepout_w)
                if abs(w) >= 1e-6:
                    keepout_low = zc - w
                    keepout_high = zc + w
                    if ax == 0:
                        # AX0 cannot go to Z0_raw < keepout_low
                        abs_max = float(self.axis_cal.z_raw_to_abs(0, keepout_low))
                        hi = min(hi, abs_max)
                    else:
                        # AX1 cannot go to Z1_raw > keepout_high
                        abs_min = float(self.axis_cal.z_raw_to_abs(1, keepout_high))
                        lo = max(lo, abs_min)
        except Exception:
            pass

        # no valid limits at all
        if lo == -float('inf') and hi == float('inf'):
            return t

        # interval sanity
        if lo > hi:
            # Degenerate constraints: do not clamp to nonsense.
            return t

        if t < lo or t > hi:
            if strict:
                raise RuntimeError(
                    f"AX{ax} 目标位置超限: tgt={t:.3f}, lim=[{lo:.3f},{hi:.3f}] ({context})"
                )
            t2 = min(max(t, lo), hi)
            try:
                log(
                    "MOTION_LIM_CLAMP",
                    axis=ax,
                    tgt=t,
                    clamped=t2,
                    lim_lo=lo,
                    lim_hi=hi,
                    ctx=context,
                )
            except Exception:
                pass
            return float(t2)

        return t

    def movea_abs(self, axis: int, pos_abs: float, *, context: str = 'MoveA'):
        axis = max(0, min(AXIS_COUNT - 1, int(axis)))
        pos_abs = self.apply_soft_limits_abs(axis, float(pos_abs), strict=False, context=str(context))

        base = self._base(axis)

        # write setpoints
        self._write_regs(base + OFF_POS_MOVEA, encode_float64_to_4regs(float(pos_abs), FLOAT64_WORD_ORDER))
        self._write_axis_params(axis)

        # pulse command
        self._pulse_cmd_bits(axis, CMD_MOVEA_REQ)

    # =========================
    # UI actions (manual)
    # =========================
    def _axis(self) -> int:
        i = int(self.axis_idx.get())
        return max(0, min(AXIS_COUNT - 1, i))

    def _on_axis_selected(self, _evt=None):
        """Legacy handler (axis_combo removed).

        Axis selection is handled by AxisScreen notebook tabs.
        This function is kept to avoid accidental callback breakage.
        """
        self._refresh_axis_panel()

    def _poll_ui_queue(self):
        try:
            while True:
                k, payload = self.ui_q.get_nowait()

                # lightweight workflow logging (avoid high-frequency spam)
                try:
                    if k in LOG_UI_EVENT_FILTER:
                        if k == "auto_row":
                            row = payload.get("row", None)
                            if row is not None:
                                log("UI_AUTO_ROW", idx=getattr(row, "idx", None), od_dev=getattr(row, "od_dev", None), od_runout=getattr(row, "od_runout", None), od_round=getattr(row, "od_round", None), id_dev=getattr(row, "id_dev", None), id_runout=getattr(row, "id_runout", None), id_round=getattr(row, "id_round", None), concentricity=getattr(row, "concentricity", None), ok=getattr(row, "ok", None))
                            else:
                                log("UI_EVT", k=k)
                        elif k == "auto_state":
                            log("UI_AUTO_STATE", state=payload.get("state", None), msg=payload.get("msg", None))
                        elif k == "auto_progress":
                            log("UI_AUTO_PROGRESS", idx=payload.get("idx", None), total=payload.get("total", None), x_ui=payload.get("x_ui", None), x_abs=payload.get("x_abs", None))
                        elif k == "auto_cov":
                            log("UI_AUTO_COV", idx=payload.get("idx", None), cov=payload.get("cov", None), miss=payload.get("miss", None), reason=payload.get("reason", None), revs=payload.get("revs", None), elapsed=payload.get("elapsed", None))
                        elif k == "auto_postcalc":
                            log("UI_AUTO_POSTCALC", ecc_od=payload.get("ecc_od", None), ecc_id=payload.get("ecc_id", None), straight_od=payload.get("straight_od", None), straight_id=payload.get("straight_id", None), axis_dist=payload.get("axis_dist", None))
                        elif k == "auto_straightness":
                            log("UI_AUTO_STRAIGHT", straight_od=payload.get("straight_od", None), straight_id=payload.get("straight_id", None), axis_dist=payload.get("axis_dist", None))
                        elif k == "auto_clear":
                            log("UI_AUTO_CLEAR")
                        elif k == "gauge_err":
                            log("UI_GAUGE_ERR", err=payload.get("err", None))
                        elif k == "gauge_conn":
                            log("UI_GAUGE_CONN", connected=payload.get("connected", None), port=payload.get("port", None), baud=payload.get("baud", None))
                        elif k == "plc_err":
                            log("UI_PLC_ERR", err=payload.get("err", None), retry=payload.get("retry", None), max=payload.get("max", None), backoff_s=payload.get("backoff_s", None))
                        elif k == "plc_giveup":
                            log("UI_PLC_GIVEUP", retry=payload.get("retry", None), max=payload.get("max", None))
                        elif k == "plc_manual":
                            log("UI_PLC_MANUAL", ip=payload.get("ip", None), port=payload.get("port", None))
                        else:
                            log("UI_EVT", k=k)
                except Exception:
                    pass


                if k == "plc_ok":
                    self.plc_status_var.set(
                        f"PLC: OK   {time.strftime('%H:%M:%S')}   ip={self.worker.ip}:{self.worker.port}   unit={self.worker.unit_id}"
                    )
                    with self._snapshot_lock:
                        self._axis_snapshot = payload["axes"]

                    # CL (ID) live values
                    try:
                        cl_out3_mm = payload.get("cl_out3_mm", None)
                        cl_out3_raw = payload.get("cl_out3_raw", None)
                        cl_cnt = payload.get("cl_out3_cnt", None)
                        # keep latest snapshot for sampling/fallback
                        try:
                            self._cl_out3_mm_latest = None if cl_out3_mm is None else float(cl_out3_mm)
                            self._cl_out3_raw_latest = None if cl_out3_raw is None else int(cl_out3_raw)
                            self._cl_out3_cnt_latest = None if cl_cnt is None else int(cl_cnt)
                            self._cl_out3_ts_latest = float(time.time())
                        except Exception:
                            pass
                        if cl_out3_mm is None:
                            # show raw if available (special values), else "--"
                            if cl_out3_raw is not None:
                                self.cl_id_var.set(str(int(cl_out3_raw)))
                            else:
                                self.cl_id_var.set("--")
                        else:
                            # OUT3 (DINT) scaled to mm
                            self.cl_id_var.set(f"{float(cl_out3_mm):.2f}")
                        if cl_cnt is None:
                            self.cl_cnt_var.set("--")
                        else:
                            self.cl_cnt_var.set(str(int(cl_cnt)))

                        # Update ID sample window only on counter change (new sample)
                        if cl_cnt is not None and cl_out3_mm is not None:
                            if self._last_cl_cnt is None or int(cl_cnt) != int(self._last_cl_cnt):
                                self._last_cl_cnt = int(cl_cnt)
                                self._id_samples.append(float(cl_out3_mm))
                                self._refresh_id_stats()
                    except Exception:
                        pass

                    # Key test coils (X/Y)
                    try:
                        self._keytest_apply_bits(
                            payload.get("keytest_x_bits", None),
                            payload.get("keytest_y_bits", None),
                        )
                    except Exception:
                        pass
                    # f2 validation: issue one-shot read after first successful PLC connection
                    if not getattr(self, "_dbg_axis_cal_sent", False):
                        try:
                            self.cmd_q.put(CmdReadRegs(AXISCAL_MB_BASE, AXISCAL_WORDS, "axis_cal"))
                            self._dbg_axis_cal_sent = True
                            print(f"[axis_cal] request read(after plc_ok): addr={AXISCAL_MB_BASE} count={AXISCAL_WORDS}")
                        except Exception as e:
                            print(f"[axis_cal] enqueue read failed(after plc_ok): {e}")
                    self._refresh_axis_panel()
                    # Keep AxisCal read-only status in sync with latest feedback
                    self.axis_cal_refresh_status()

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

                elif k == "plc_read":
                    tag = payload.get("tag", "")
                    d_addr = payload.get("d_addr", None)
                    count = payload.get("count", None)
                    regs = payload.get("regs", [])

                    # sync reads (AutoFlow sampling)
                    if isinstance(tag, str) and tag.startswith("sync:"):
                        try:
                            with self._sync_reads_lock:
                                slot = self._sync_reads.get(tag, None)
                                if slot is not None:
                                    slot["regs"] = list(regs)
                                    try:
                                        slot["evt"].set()
                                    except Exception:
                                        pass
                        finally:
                            # Do not fall through to axis_cal parsing
                            continue

                    # f2/f3/f4: parse axis calibration block if requested
                    if tag == "axis_cal" or tag == "axis_cal_verify":
                        try:
                            cal = AxisCal.from_regs(regs)

                            if tag == "axis_cal_verify":
                                exp = getattr(self, "_axis_cal_write_expect_regs", None)
                                ok = exp is not None and list(exp) == list(regs)

                                if ok:
                                    # success: accept PLC readback and refresh UI
                                    self.axis_cal = cal
                                    self._axis_cal_to_ui(cal)
                                    self.axis_cal_refresh_status()
                                    self._axis_cal_set_field_status(
                                        [
                                            "sign",
                                            "off_ax0",
                                            "off_ax1",
                                            "off_ax2",
                                            "off_ax4",
                                            "b14",
                                            "b2",
                                        ],
                                        "写入成功",
                                    )
                                    print(
                                        "[axis_cal] verify OK; readback matches written regs. "
                                        f"sign={cal.sign} off_ax0={cal.off_ax0:.6f} off_ax1={cal.off_ax1:.6f} "
                                        f"off_ax2={cal.off_ax2:.6f} off_ax4={cal.off_ax4:.6f} "
                                        f"b14={cal.b14:.6f} keepout_handoff={self._keepout_handoff_raw(cal):.6f}"
                                    )
                                else:
                                    # failure: report mismatch indices (do not overwrite UI)
                                    self._axis_cal_set_field_status(
                                        [
                                            "sign",
                                            "off_ax0",
                                            "off_ax1",
                                            "off_ax2",
                                            "off_ax4",
                                            "b14",
                                            "b2",
                                        ],
                                        "写入失败",
                                    )
                                    mism = []
                                    if exp is not None:
                                        for i, (a, b) in enumerate(zip(exp, regs)):
                                            if a != b:
                                                mism.append((i, a, b))
                                    print(
                                        "[axis_cal] verify FAIL; readback differs from written regs. "
                                        f"mismatch_count={len(mism)}"
                                    )
                                    if mism:
                                        # print first few mismatches for diagnosis
                                        for i, a, b in mism[:8]:
                                            print(f"  - idx {i}: expect={a} got={b}")

                                # one-shot: clear expectation regardless of result
                                self._axis_cal_write_expect_regs = None

                            else:
                                # Normal read: keep in-memory copy and refresh calibration UI
                                self.axis_cal = cal
                                self._axis_cal_to_ui(cal)
                                self.axis_cal_refresh_status()
                                self._axis_cal_set_field_status(
                                    [
                                        "sign",
                                        "off_ax0",
                                        "off_ax1",
                                        "off_ax2",
                                        "off_ax4",
                                        "b14",
                                        "b2",
                                        "keepout_w",
                                    ],
                                    "已读取",
                                )
                                print(
                                    "[axis_cal] parsed "
                                    f"sign={cal.sign} "
                                    f"off_ax0={cal.off_ax0:.6f} off_ax1={cal.off_ax1:.6f} "
                                    f"off_ax2={cal.off_ax2:.6f} off_ax4={cal.off_ax4:.6f} "
                                    f"b14={cal.b14:.6f} keepout_handoff={self._keepout_handoff_raw(cal):.6f}"
                                )
                        except Exception as e:
                            print(f"[axis_cal] parse failed: {e}")

                    # Always keep the raw dump for low-level diagnostics
                    print(f"[plc_read] tag={tag} addr={d_addr} count={count} regs={regs}")

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
                    self._auto_cur_sec_idx = idx + 1
                    self.auto_progress_var.set(f"当前截面: {idx + 1} / 总截面: {total}")
                    self.auto_done_var.set("测量完成: 否")

                elif k == "auto_cov":
                    # Coverage info may optionally carry a 1-based section idx.
                    sec_idx = payload.get("idx", None)
                    cov = payload.get("cov", None)
                    miss = payload.get("miss", None)
                    reason = str(payload.get("reason", "") or "")
                    revs = payload.get("revs", None)
                    elapsed = payload.get("elapsed", None)

                    try:
                        sec_idx_int = int(sec_idx) if sec_idx is not None else (int(self._auto_cur_sec_idx) if self._auto_cur_sec_idx is not None else None)
                    except Exception:
                        sec_idx_int = int(self._auto_cur_sec_idx) if self._auto_cur_sec_idx is not None else None

                    info = {
                        "cov": cov,
                        "miss": miss,
                        "max_gap_deg": payload.get("max_gap_deg", None),
                        "reason": reason,
                        "revs": revs,
                        "elapsed": elapsed,
                    }

                    if sec_idx_int is not None:
                        self._section_cov_info[int(sec_idx_int)] = info

                    txt = self._format_cov_info(info)
                    # If user selected a section row, keep showing that row's info
                    # unless the update corresponds to the same section.
                    if (self._selected_sec_idx is None) or (sec_idx_int is None) or (int(self._selected_sec_idx) == int(sec_idx_int)):
                        self.cov_var.set(txt)

                elif k == "auto_straightness":
                    # overall straightness result (outer/inner)
                    od = payload.get("straight_od", payload.get("straightness", None))
                    idv = payload.get("straight_id", None)
                    axis_dist = payload.get("axis_dist", None)
                    if axis_dist is not None:
                        try:
                            self._axis_dist = float(axis_dist)
                        except Exception:
                            self._axis_dist = None

                    self._set_straight_label(od, idv, self._axis_dist)

                elif k == "auto_postcalc":
                    # post-calculated eccentricity + straightness
                    ecc_od = payload.get("ecc_od", []) or []
                    ecc_id = payload.get("ecc_id", []) or []
                    od = payload.get("straight_od", None)
                    idv = payload.get("straight_id", None)
                    axis_dist = payload.get("axis_dist", None)
                    if axis_dist is not None:
                        try:
                            self._axis_dist = float(axis_dist)
                        except Exception:
                            self._axis_dist = None

                    self._set_straight_label(od, idv, self._axis_dist)

                    try:
                        n = min(len(self._result_iids), len(ecc_od), len(ecc_id))
                        for i in range(n):
                            iid = self._result_iids[i]
                            self.result_tree.set(iid, "od_ecc", f"{float(ecc_od[i]):.3f}")
                            self.result_tree.set(iid, "id_ecc", f"{float(ecc_id[i]):.3f}")
                        # update cached rows for export
                        try:
                            n2 = min(len(self._auto_rows), len(ecc_od), len(ecc_id))
                            for i2 in range(n2):
                                try:
                                    self._auto_rows[i2].od_ecc = float(ecc_od[i2])
                                except Exception:
                                    pass
                                try:
                                    self._auto_rows[i2].id_ecc = float(ecc_id[i2])
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    except Exception:
                        pass

                elif k == "auto_raw_points":
                    pts = payload.get("points", []) or []
                    try:
                        if isinstance(pts, list):
                            self._auto_raw_points.extend([p for p in pts if isinstance(p, dict)])
                    except Exception:
                        pass

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
                        # auto export once per run
                        if not getattr(self, "_auto_export_done", False):
                            try:
                                self._run_end_ts = float(time.time())
                            except Exception:
                                self._run_end_ts = None
                            ok, emsg = self._export_current_run()
                            self._auto_export_done = True if ok else False
                            try:
                                # keep UI info concise
                                self.auto_msg_var.set(str(emsg))
                            except Exception:
                                pass
                    elif st in ("ERR", "STOP"):
                        self.auto_done_var.set("测量完成: 否")

        except queue.Empty:
            pass
        self.after(60, self._poll_ui_queue)

    def _append_result_row(self, row: MeasureRow):
        od_ecc_txt = "--" if getattr(row, "od_ecc", None) is None else f"{float(row.od_ecc):.3f}"
        id_ecc_txt = "--" if getattr(row, "id_ecc", None) is None else f"{float(row.id_ecc):.3f}"

        iid = self.result_tree.insert(
            "",
            "end",
            values=(
                row.idx,
                f"{row.x_ui:.3f}",
                f"{row.od_dev:+.3f}",
                f"{float(getattr(row, 'od_runout', 0.0)):.3f}",
                f"{row.od_round:.3f}",
                f"{row.id_dev:+.3f}",
                f"{float(getattr(row, 'id_runout', 0.0)):.3f}",
                f"{row.id_round:.3f}",
                f"{row.concentricity:.3f}",
                od_ecc_txt,
                id_ecc_txt,
            ),
        )
        try:
            self._result_iids.append(str(iid))
        except Exception:
            pass

        try:
            self._auto_rows.append(row)
        except Exception:
            pass


    # =========================
    # RunId / Serial / Export helpers
    # =========================
    def _sanitize_recipe_key(self, name: str) -> str:
        """Recipe key used in serial/filenames (keep readable but filesystem-safe)."""
        s = str(name or "").strip()
        if not s:
            s = "recipe"
        # allow letters/digits/underscore/hyphen and common CJK; replace others with '_'
        s2 = []
        for ch in s:
            o = ord(ch)
            if ch.isalnum() or ch in "_-":
                s2.append(ch)
            elif 0x4E00 <= o <= 0x9FFF:  # CJK Unified Ideographs
                s2.append(ch)
            else:
                s2.append("_")
        out = "".join(s2)
        out = re.sub(r"_+", "_", out).strip("_")
        return out[:24] if out else "recipe"

    def _app_root_dir(self) -> Path:
        # C:\Users\<username>\FRP_IPC
        try:
            return Path.home() / "FRP_IPC"
        except Exception:
            return Path("./FRP_IPC")

    def _exports_root_dir(self) -> Path:
        return self._app_root_dir() / "exports"

    def _counter_file(self) -> Path:
        return self._app_root_dir() / "run_counter.json"

    def _load_run_counters(self) -> dict:
        p = self._counter_file()
        try:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
        except Exception:
            pass
        return {}

    def _save_run_counters(self, data: dict) -> None:
        p = self._counter_file()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _next_serial(self, recipe_name: str) -> str:
        today = datetime.date.today()
        day_tag = today.strftime("%Y%m%d")
        recipe_key = self._sanitize_recipe_key(recipe_name)
        counters = self._load_run_counters()
        day_map = counters.get(day_tag, {})
        try:
            seq = int(day_map.get(recipe_key, 0)) + 1
        except Exception:
            seq = 1
        day_map[recipe_key] = seq
        counters[day_tag] = day_map
        self._save_run_counters(counters)
        return f"{day_tag}-{recipe_key}-{seq:03d}"

    def _get_device_code(self) -> str:
        """Best-effort stable device code (used in export meta)."""
        # Prefer Windows MachineGuid when available.
        try:
            import winreg  # type: ignore
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography")
            v, _t = winreg.QueryValueEx(k, "MachineGuid")
            if v:
                return str(v)
        except Exception:
            pass
        # Fallback: hostname + MAC
        try:
            import uuid as _uuid
            mac = _uuid.getnode()
            return f"{platform.node()}-{mac:012x}"
        except Exception:
            return platform.node()

    def _prepare_new_run(self) -> None:
        """Allocate a new Serial/RunId for the next Auto measurement."""
        try:
            recipe_name = str(getattr(self.recipe, "name", "默认配方") or "默认配方")
        except Exception:
            recipe_name = "默认配方"
        serial = self._next_serial(recipe_name)
        self._run_serial = serial
        self._run_id = str(uuid.uuid4())
        self._run_start_ts = float(time.time())
        self._run_end_ts = None
        self._auto_export_done = False
        # reset caches for this run
        try:
            self._auto_rows.clear()
            self._auto_raw_points.clear()
        except Exception:
            self._auto_rows = []
            self._auto_raw_points = []
        try:
            self.pipe_sn_var.set(serial)
        except Exception:
            pass

    def _export_current_run(self) -> tuple[bool, str]:
        """Export current run to exports directory. Returns (ok, msg)."""
        if not self._run_serial or not self._run_id or not self._run_start_ts:
            return False, "未生成流水号/RunId，无法导出。"
        try:
            start_ts = float(self._run_start_ts)
            end_ts = float(self._run_end_ts or time.time())
            day_dir = self._exports_root_dir() / datetime.date.fromtimestamp(start_ts).strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)

            serial = self._run_serial
            run_id = self._run_id

            # Section results CSV
            run_dir = day_dir / serial
            run_dir.mkdir(parents=True, exist_ok=True)

            section_csv = run_dir / "section_results.csv"
            rows = list(self._auto_rows or [])
            with open(section_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "serial", "run_id",
                    "start_time", "end_time", "duration_s",
                    "section_idx", "z_pos_mm",
                    "od_avg_mm", "od_dev_mm", "od_runout_mm", "od_round_mm",
                    "id_avg_mm", "id_dev_mm", "id_runout_mm", "id_round_mm",
                    "concentricity_mm", "od_ecc_mm", "id_ecc_mm",
                    "raw",
                ])
                for r in rows:
                    w.writerow([
                        serial, run_id,
                        datetime.datetime.fromtimestamp(start_ts).isoformat(sep=" ", timespec="seconds"),
                        datetime.datetime.fromtimestamp(end_ts).isoformat(sep=" ", timespec="seconds"),
                        f"{(end_ts-start_ts):.3f}",
                        int(getattr(r, "idx", 0)),
                        float(getattr(r, "x_ui", 0.0)),
                        float(getattr(r, "od_avg", 0.0)),
                        float(getattr(r, "od_dev", 0.0)),
                        float(getattr(r, "od_runout", 0.0)),
                        float(getattr(r, "od_round", 0.0)),
                        float(getattr(r, "id_avg", 0.0)),
                        float(getattr(r, "id_dev", 0.0)),
                        float(getattr(r, "id_runout", 0.0)),
                        float(getattr(r, "id_round", 0.0)),
                        float(getattr(r, "concentricity", 0.0)),
                        "" if getattr(r, "od_ecc", None) is None else float(getattr(r, "od_ecc", 0.0)),
                        "" if getattr(r, "id_ecc", None) is None else float(getattr(r, "id_ecc", 0.0)),
                        str(getattr(r, "raw", "") or ""),
                    ])

            # Raw points CSV
            raw_csv = run_dir / "raw_points.csv"
            pts = list(self._auto_raw_points or [])
            with open(raw_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "serial", "run_id",
                    "section_idx", "z_pos_mm", "sample_idx",
                    "ts", "theta_deg", "bin",
                    "od_mm", "id_mm", "cl_cnt",
                    "raw_od", "raw_id",
                ])
                for p in pts:
                    if not isinstance(p, dict):
                        continue
                    w.writerow([
                        serial, run_id,
                        p.get("section_idx", ""),
                        p.get("z_pos_mm", ""),
                        p.get("sample_idx", ""),
                        p.get("ts", ""),
                        p.get("theta_deg", ""),
                        p.get("bin", ""),
                        p.get("od_mm", ""),
                        p.get("id_mm", ""),
                        p.get("cl_cnt", ""),
                        p.get("raw_od", ""),
                        p.get("raw_id", ""),
                    ])

            # Meta JSON
            meta_path = run_dir / "meta.json"
            try:
                rcp = self.get_recipe_copy()
                recipe_snapshot = self._recipe_dump_dict(rcp)
            except Exception:
                recipe_snapshot = {}
            meta = {
                "serial": serial,
                "run_id": run_id,
                "start_time": datetime.datetime.fromtimestamp(start_ts).isoformat(sep=" ", timespec="seconds"),
                "end_time": datetime.datetime.fromtimestamp(end_ts).isoformat(sep=" ", timespec="seconds"),
                "duration_s": float(end_ts - start_ts),
                "recipe": recipe_snapshot,
                "device_code": self._get_device_code(),
                "software_version": str(SOFTWARE_VERSION),
                "plc": {
                    "ip": getattr(self.worker, "ip", ""),
                    "port": getattr(self.worker, "port", ""),
                    "unit": getattr(self.worker, "unit_id", ""),
                },
                "gauge": {
                    "enabled": bool(getattr(self, "sim_gauge_enabled", False)) is False,
                    "port": getattr(self.gauge_worker, "port", None) if getattr(self, "gauge_worker", None) is not None else None,
                },
                "exports": {
                    "section_results_csv": str(section_csv),
                    "raw_points_csv": str(raw_csv),
                    "meta_json": str(meta_path),
                },
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            return True, f"已导出：{run_dir}"
        except Exception as e:
            return False, f"导出失败：{e}"

    # =========================
    # Auto result helpers
    # =========================
    def _format_cov_info(self, info: dict) -> str:
        cov = info.get("cov", None)
        miss = info.get("miss", None)
        reason = str(info.get("reason", "") or "")
        revs = info.get("revs", None)
        elapsed = info.get("elapsed", None)

        reason_txt = ""
        if reason:
            mapping = {
                "COV": "覆盖率达标",
                "TIMEOUT": "超时退出",
                "REV": "圈数到达",
            }
            reason_txt = mapping.get(reason.upper(), reason)

        if cov is None:
            return "采样覆盖率：--"

        parts = [f"采样覆盖率：{float(cov) * 100:.1f}%"]
        if miss is not None:
            try:
                parts.append(f"缺失bin: {int(miss)}")
            except Exception:
                pass
        max_gap = info.get("max_gap_deg", None)
        if max_gap is not None:
            try:
                parts.append(f"最大空窗角: {float(max_gap):.1f}°")
            except Exception:
                pass
        if revs is not None:
            try:
                parts.append(f"圈数≈{float(revs):.2f}")
            except Exception:
                pass
        if elapsed is not None:
            try:
                parts.append(f"用时{float(elapsed):.2f}s")
            except Exception:
                pass
        if reason_txt:
            parts.append(f"结束:{reason_txt}")
        return "  ".join(parts)

    def _set_straight_label(self, straight_od, straight_id, axis_dist) -> None:
        if straight_od is None and straight_id is None and axis_dist is None:
            self.straight_var.set("直线度：--（外圆）/ --（内圆） || 整体同心度：--")
            return

        od_txt = "--" if straight_od is None else f"{float(straight_od):.3f}"
        id_txt = "--" if straight_id is None else f"{float(straight_id):.3f}"
        ax_txt = "--" if axis_dist is None else f"{float(axis_dist):.3f}"
        self.straight_var.set(f"直线度：{od_txt}（外圆）/ {id_txt}（内圆） || 整体同心度：{ax_txt}")

    def _show_cov_for_section(self, sec_idx: int) -> None:
        info = self._section_cov_info.get(int(sec_idx))
        if not info:
            self.cov_var.set("采样覆盖率：--")
            return
        self.cov_var.set(self._format_cov_info(info))

    def _on_result_select(self, event=None):
        """When user selects a section row, show that section's sampling coverage/info."""
        try:
            sel = self.result_tree.selection()
            if not sel:
                self._selected_sec_idx = None
                # fallback to current section (or keep last shown)
                if self._auto_cur_sec_idx is not None:
                    self._show_cov_for_section(int(self._auto_cur_sec_idx))
                return

            iid = sel[0]
            vals = self.result_tree.item(iid, "values")
            if not vals:
                return
            sec_idx = int(vals[0])
            self._selected_sec_idx = sec_idx
            self._show_cov_for_section(sec_idx)
        except Exception:
            pass


    def _refresh_axis_panel(self):
        ax = self._axis()
        ac = self.get_axis_copy(ax)

        # Act_Pos is the only guaranteed feedback in Axis_Ctrl
        ui_pos = self.ui_coord.abs_to_ui(getattr(ac, 'act_pos', 0.0))
        act_pos = float(getattr(ac, 'act_pos', 0.0) or 0.0)

        if hasattr(self, 'lbl_actpos'):
            self.lbl_actpos.config(text=f"Act_Pos(abs): {act_pos:.6f}")
        if hasattr(self, 'lbl_uipos'):
            self.lbl_uipos.config(
                text=f"UI_Pos(相对): {ui_pos:.3f}    (ZeroAbs={self.ui_coord.zero_abs:.3f}, sign={self.ui_coord.sign:+d})"
            )

        err = int(getattr(ac, 'err', 0) or 0)
        warn = int(getattr(ac, 'warn', 0) or 0)
        sts = int(getattr(ac, 'sts', 0) or 0)
        st_id = int(getattr(ac, 'st_id', 0) or 0)
        seq = int(getattr(ac, 'seq', 0) or 0)
        seq_ack = int(getattr(ac, 'seq_ack', 0) or 0)

        if hasattr(self, 'lbl_err'):
            self.lbl_err.config(text=f"ErrCode: {err}    Warn: {warn}")
        if hasattr(self, 'lbl_sts'):
            self.lbl_sts.config(text=f"Sts(raw_state): {sts}    (0..8)")
        if hasattr(self, 'lbl_stid'):
            self.lbl_stid.config(text=f"St_ID: {st_id}    Seq/Ack: {seq}/{seq_ack}")
        if hasattr(self, 'lbl_cmd'):
            self.lbl_cmd.config(text=f"Cmd: 0x{int(getattr(ac, 'cmd', 0) or 0):04X}")
        if hasattr(self, 'lbl_flags'):
            self.lbl_flags.config(text="")

        # UI显示使能：Sts==0 视为未使能，其余视为已使能（含错误态）
        # 为避免用户点击 Enable 后在反馈尚未更新前被刷新逻辑立即“打回”，
        # 在短暂的 pending 窗口内不强制覆盖 power_var。
        pend_t = 0.0
        try:
            pend_t = float(self._power_cmd_pending[ax])
        except Exception:
            pend_t = 0.0
        if (time.time() - pend_t) > 0.6:
            if hasattr(self, 'power_var'):
                self.power_var.set(1 if sts != 0 else 0)

        # keep teach panel synced
        self._refresh_teach_pos()

    def _on_power_toggle(self):
        ax = self._axis()
        want_en = 1 if int(self.power_var.get() or 0) else 0

        # Enable/Disable 为电平命令（LEVEL）。
        if want_en:
            self.set_cmd_bits(ax, set_mask=CMD_EN_REQ, clr_mask=0)
        else:
            self.set_cmd_bits(ax, set_mask=0, clr_mask=CMD_EN_REQ)

        # 记录一次 pending，允许 UI 暂时保持用户意图，等待 PLC 反馈刷新
        try:
            self._power_cmd_pending[ax] = time.time()
        except Exception:
            pass

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


    def _do_mover(self):
        ax = self._axis()
        try:
            # New UI uses ent_pos_r; keep compatibility with older name ent_pos2
            if hasattr(self, 'ent_pos_r'):
                dis = float(getattr(self, 'ent_pos_r').get().strip())
            elif hasattr(self, 'ent_pos2'):
                dis = float(getattr(self, 'ent_pos2').get().strip())
            else:
                raise ValueError('未找到 MoveR 位移输入框(ent_pos_r)')
        except Exception as e:
            messagebox.showerror('参数错误', str(e))
            return

        base = self._base(ax)
        # Pos_MoveR (relative displacement)
        self._write_regs(
            base + OFF_POS_MOVER,
            encode_float64_to_4regs(float(dis), FLOAT64_WORD_ORDER),
        )
        # Dir_MoveR + velocities/acc/dec/jerk
        self._write_axis_params(ax)
        # pulse MoveR
        self._pulse_cmd_bits(ax, CMD_MOVER_REQ)

    def _do_vel_start(self):
        ax = self._axis()
        # write params (Vel_VelMove etc.)
        self._write_axis_params(ax)
        # VelMove is LEVEL command
        self.set_cmd_bits(ax, set_mask=CMD_VELMOVE_REQ, clr_mask=0)

    def _do_vel_stop(self):
        ax = self._axis()
        # clear level bit first
        self.set_cmd_bits(ax, set_mask=0, clr_mask=CMD_VELMOVE_REQ)
        # then request STOP to decelerate
        self._pulse_cmd_bits(ax, CMD_STOP_REQ)

    def _jog_hold(self, direction: str, on: bool):

        ax = self._axis()

        if on:
            try:
                vel, acc, dec, jerk = self._read_common_params()
            except Exception:
                vel, acc, dec, jerk = 100, 200, 200, 500
            self._write_axis_params(ax)

            if direction == "rev":
                self.set_cmd_bits(ax, set_mask=CMD_JOG_B_REQ, clr_mask=CMD_JOG_F_REQ)
            else:
                self.set_cmd_bits(ax, set_mask=CMD_JOG_F_REQ, clr_mask=CMD_JOG_B_REQ)
        else:
            self.set_cmd_bits(ax, set_mask=0, clr_mask=(CMD_JOG_F_REQ | CMD_JOG_B_REQ))

    def _do_inch(self, direction: str):
        messagebox.showinfo('提示', 'Inch 功能已在新版 axiscore 中移除。请使用 MoveR 或 Jog。')

    # =========================
    # Simulated gauge
    # =========================
    def simulate_gauge_once(self, recipe: Recipe) -> Tuple[float, str]:
        """Generate OD value near od_std with small deterministic-ish noise."""
        od_noise = (0.5 - (time.time() * 997) % 1.0) * 0.02  # ~±0.01mm
        od = float(recipe.od_std_mm) + float(od_noise)
        raw = f"M1,{od:+.4f}"
        return float(od), raw

    def simulate_disp_once(self, recipe: Recipe) -> Tuple[float, str]:
        """Generate ID value near id_std with small deterministic-ish noise.

        Phase f8: simulation only. The returned value is treated as *diameter* (mm).
        """
        id_noise = (0.5 - (time.time() * 733) % 1.0) * 0.02  # ~±0.01mm
        id_mm = float(recipe.id_std_mm) + float(id_noise)
        raw = f"D1,{id_mm:+.4f}"
        return float(id_mm), raw


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
