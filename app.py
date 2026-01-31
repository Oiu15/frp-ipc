# ./app.py
from __future__ import annotations

import numpy as np
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
import math

from utils.logger import init_log, log, log_exc
from typing import Any, List, Optional, Tuple, Iterable

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.font as tkfont

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
    CL_OUT1_WORD_OFF,
    CL_OUT2_WORD_OFF,
    CL_OUT4_WORD_OFF,
    CL_OUT5_WORD_OFF,
    CL_OUT1_UPD_WORD_OFF,
    CL_OUT2_UPD_WORD_OFF,
    CL_OUT4_UPD_WORD_OFF,
    CL_OUT5_UPD_WORD_OFF,
    CL_OUT_MEAS_BLOCK_OFF,
    CL_OUT_MEAS_BLOCK_WORDS,
    CL_OUT_CNT_BLOCK_OFF,
    CL_OUT_CNT_BLOCK_WORDS,
    CL_ID_WORD_OFF,
    CL_ID_UPD_WORD_OFF,
    CL_OUT_SCALE_MM,
    CL_OUT1_SCALE_MM,
    CL_OUT2_SCALE_MM,
    CL_OUT3_SCALE_MM,
    CL_OUT4_SCALE_MM,
    CL_OUT5_SCALE_MM,
    CL_ID_SCALE_MM,
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
    CmdSetPollProfile,
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


SOFTWARE_VERSION = "ipc_nid_f3_7"
# AX0 soft limits (absolute position, mm). Used for Z_disp travel estimation when PLC is offline.
# If PLC provides non-zero soft limits, those values will take precedence.
AX0_SOFTLIM_NEG_ABS = -350.0
AX0_SOFTLIM_POS_ABS = 1200.0

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
    "auto_len",
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

        # Rolling error banner (top bar, red marquee)
        self.err_banner_var = tk.StringVar(value="")
        self._err_banner_src: str = ""
        self._err_banner_pos: int = 0
        self._err_banner_gap: str = "   |   "
        self._err_banner_min_update_ts: float = 0.0

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

        # Latest CL (ID, OUT4) snapshot from background polling (for UI / fallback)
        self._cl_id_mm_latest: Optional[float] = None
        self._cl_id_raw_latest: Optional[int] = None
        self._cl_id_cnt_latest: Optional[int] = None
        self._cl_id_ts_latest: float = 0.0


        # Latest CL OUT snapshots from background polling (for ID calibration without sync reads)
        self._cl_out1_mm_latest: Optional[float] = None
        self._cl_out1_raw_latest: Optional[int] = None
        self._cl_out1_cnt_latest: Optional[int] = None
        self._cl_out2_mm_latest: Optional[float] = None
        self._cl_out2_raw_latest: Optional[int] = None
        self._cl_out2_cnt_latest: Optional[int] = None
        self._cl_out4_mm_latest: Optional[float] = None
        self._cl_out4_raw_latest: Optional[int] = None
        self._cl_out4_cnt_latest: Optional[int] = None
        self._cl_out5_mm_latest: Optional[float] = None
        self._cl_out5_raw_latest: Optional[int] = None
        self._cl_out5_cnt_latest: Optional[int] = None
        self._cl_out_ts_latest: float = 0.0

        # Last requested PLC polling profile as tracked by IPC (normal|sampling)
        self._plc_poll_profile_req: str = 'normal'


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

        # Cached X/Y point states for thread-safe access (AutoFlow/background threads)
        self._keytest_bits_lock = threading.Lock()
        self._keytest_x_points_state = [0 for _ in range(len(KEYTEST_X_POINTS))]
        self._keytest_y_points_state = [0 for _ in range(len(KEYTEST_Y_POINTS))]

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

        # ------------------------------
        # OD Calibration (B) UI state
        # ------------------------------
        # 说明：
        # - B 值属于“工装/安装状态”的参数，不应散落在配方中。
        # - f2_0 主要落地 UI 布局与接口；采集/计算做最小可用实现（按定时采样）。
        self.odcal_state_var = tk.StringVar(value="IDLE")
        self.odcal_msg_var = tk.StringVar(value="-")
        self.odcal_cmd_var = tk.StringVar(value="M0,1")
        self.odcal_dref_var = tk.StringVar(value="180.000")
        self.odcal_map_out1_var = tk.StringVar(value="L")  # OUT1 -> L/R

        self.odcal_mode_var = tk.StringVar(value="timed")  # timed | one_rev
        self.odcal_hz_var = tk.StringVar(value="20")
        self.odcal_duration_var = tk.StringVar(value="10")
        # AX3 rotation speed for one-rev capture (deg/s)
        self.odcal_rot_degps_var = tk.StringVar(value="10")

        # Advanced sampling parameters (folded UI)
        # - 角度来源：AX3 编码器 / 无角度
        # - 去抖/滤波：用于降低抖动噪声（先对 sum=lL+lR 处理，后续可扩展到 v1/v2）
        # - 异常剔除阈值：基于 sigma 的离群点剔除
        self.odcal_angle_src_var = tk.StringVar(value="AX3")  # AX3 | NONE
        self.odcal_filter_var = tk.StringVar(value="无")  # 无 | 中值(3) | 中值(5)
        self.odcal_outlier_sigma_var = tk.StringVar(value="3.0")

        # capture-time snapshot
        self._odcal_angle_enabled: bool = True
        self._odcal_filter_mode: str = "无"
        self._odcal_outlier_sigma: float = 3.0

        # Results
        self.odcal_B_candidate_var = tk.StringVar(value="--")
        self.odcal_B_active_var = tk.StringVar(value="--")
        self.odcal_n_var = tk.StringVar(value="0")
        self.odcal_elapsed_var = tk.StringVar(value="--")

        # Quality stats (sum = lL+lR)
        self.odcal_sum_mean_var = tk.StringVar(value="--")
        self.odcal_sum_std_var = tk.StringVar(value="--")
        self.odcal_sum_min_var = tk.StringVar(value="--")
        self.odcal_sum_max_var = tk.StringVar(value="--")
        self.odcal_drop_rate_var = tk.StringVar(value="--")

        # in-memory capture buffer
        self._odcal_capturing: bool = False
        self._odcal_points: list[dict] = []
        self._odcal_drop_cnt: int = 0
        self._odcal_start_ts: Optional[float] = None
        self._odcal_after_id: Optional[str] = None
        self._odcal_stop_at_ts: Optional[float] = None

        # one-rev capture state (bind samples to AX3 angle)
        self._odcal_one_rev: bool = False
        self._odcal_ax3_rotating: bool = False
        self._odcal_ax3_speed_degps: float = 0.0
        self._odcal_theta_start: Optional[float] = None
        self._odcal_theta_last: Optional[float] = None
        self._odcal_theta_unwrap: float = 0.0
        self._odcal_rev_progress_deg: float = 0.0
        self._odcal_rev_target_deg: float = 360.0
        self._odcal_stop_reason: str = ""

        # Load last applied B (if any)
        try:
            self._odcal_load_active()
        except Exception:
            pass
        self._odcal_B_candidate: Optional[float] = None

        # Load B_active if exists
        try:
            self._odcal_load_active()
        except Exception:
            pass

        # CL (Keyence CL-3000) via PLC mapped registers (OUT1..OUT5)
        # 约定：OUT1=x1(右测头原始位移), OUT2=x2(左测头原始位移), OUT3=保留/厚度, OUT4=内径(ID)直接值, OUT5=m(偏心投影)
        #
        # 兼容：cl_id_var/cl_cnt_var 作为 “ID(OUT4)” 的显示/统计入口。
        self.cl_id_var = tk.StringVar(value="--")  # OUT4 (ID) mm or raw
        self.cl_cnt_var = tk.StringVar(value="--")  # OUT4 update counter

        self.cl_out1_var = tk.StringVar(value="--")
        self.cl_out2_var = tk.StringVar(value="--")
        self.cl_out3_var = tk.StringVar(value="--")
        self.cl_out4_var = tk.StringVar(value="--")
        self.cl_out5_var = tk.StringVar(value="--")

        self.cl_out1_cnt_var = tk.StringVar(value="--")
        self.cl_out2_cnt_var = tk.StringVar(value="--")
        self.cl_out3_cnt_var = tk.StringVar(value="--")
        self.cl_out4_cnt_var = tk.StringVar(value="--")
        self.cl_out5_cnt_var = tk.StringVar(value="--")

        self.cl_m_calc_var = tk.StringVar(value="--")  # (x1+x2)/2 from IPC
        self.cl_m_diff_var = tk.StringVar(value="--")  # m_calc - out5
        self.id_n_var = tk.StringVar(value="0")
        self.id_avg_var = tk.StringVar(value="--")
        self.id_dev_var = tk.StringVar(value="--")
        self.id_round_var = tk.StringVar(value="--")

        # ID sample window (for avg/dev/roundness)
        import collections as _collections
        self._id_samples = _collections.deque(maxlen=300)
        self._last_cl_cnt = None
        # ------------------------------
        # ID Calibration (Chord + m) UI state
        # ------------------------------
        # 说明：
        # - 当前 CL 输出的 OUT4 为“弦长 c”，OUT5 为“弦中点在测量线上的偏移 m”（约定 m=(x1-x2)/2）。
        # - 仅用 c 当作直径会系统性偏小（除非测量线恰好过圆心）。
        # - 本标定主要解决：OUT4(c) 的零点/比例偏差（用已知环规 ID_ref 进行修正），并记录 m 的统计量用于装调参考。
        self.idcal_state_var = tk.StringVar(value="IDLE")
        self.idcal_msg_var = tk.StringVar(value="-")
        self.idcal_dref_var = tk.StringVar(value="150.000")  # 内径环规标称值

        self.idcal_mode_var = tk.StringVar(value="one_rev")  # timed | one_rev
        self.idcal_hz_var = tk.StringVar(value="20")
        self.idcal_duration_var = tk.StringVar(value="10")
        self.idcal_rot_degps_var = tk.StringVar(value="10")  # one_rev: AX3 角速度

        # Results
        self.idcal_delta_candidate_var = tk.StringVar(value="--")  # 对 OUT4(c) 的加法修正量 δc
        self.idcal_delta_active_var = tk.StringVar(value="--")
        self.idcal_cmax_var = tk.StringVar(value="--")
        self.idcal_mmean_var = tk.StringVar(value="--")
        self.idcal_mpp_var = tk.StringVar(value="--")
        self.idcal_fit_diam_var = tk.StringVar(value="--")  # 2R (after δc)
        self.idcal_fit_e_var = tk.StringVar(value="--")     # e from m(θ)
        self.idcal_fit_y0_var = tk.StringVar(value="--")    # y0 fitted
        self.idcal_fit_rmse_var = tk.StringVar(value="--")

        # Verify (复核) results - does not modify δc
        self.idcal_chk_err_var = tk.StringVar(value="--")     # D_fit(active) - D_ref
        self.idcal_chk_cov_var = tk.StringVar(value="--")     # theta coverage %
        self.idcal_chk_n_var = tk.StringVar(value="--")       # sample count
        self.idcal_chk_dtheta_var = tk.StringVar(value="--")  # max |Δθ| between samples

        # Verify state
        self._idcal_verify_pending: bool = False
        self._idcal_verify_delta: Optional[float] = None
        self._idcal_verify_dref: Optional[float] = None

        # in-memory capture buffer
        self._idcal_capturing: bool = False
        self._idcal_points: list[dict] = []
        self._idcal_start_ts: Optional[float] = None
        self._idcal_after_id: Optional[str] = None
        self._idcal_stop_at_ts: Optional[float] = None

        # one-rev capture state
        self._idcal_one_rev: bool = False
        self._idcal_ax3_rotating: bool = False
        self._idcal_ax3_speed_degps: float = 0.0
        self._idcal_theta_start: Optional[float] = None
        self._idcal_theta_last: Optional[float] = None
        self._idcal_theta_unwrap: float = 0.0
        self._idcal_rev_progress_deg: float = 0.0
        self._idcal_rev_target_deg: float = 360.0
        self._idcal_stop_reason: str = ""

        # Load last applied ID calibration (if any)
        try:
            self._idcal_load_active()
        except Exception:
            pass
        self._idcal_delta_candidate: Optional[float] = None



        # Auto
        self._auto_thread: Optional[AutoFlow] = None
        # Result table item ids (Treeview iids), in insertion order
        self._result_iids: list[str] = []
        self.auto_state_var = tk.StringVar(value="IDLE")
        self.auto_msg_var = tk.StringVar(value="-")
        self.auto_progress_var = tk.StringVar(value="当前截面: - / 总截面: -")
        self.auto_done_var = tk.StringVar(value="测量完成: 否")
        # Summary text lines (main screen)
        self.straight_var = tk.StringVar(value="直线度   --（外圆） | --（内圆）")
        self.conc_var = tk.StringVar(value="整体同心度   --")
        self.cov_var = tk.StringVar(value="采样覆盖率：--")

        # Summary split vars (main screen)
        self.straight_od_var = tk.StringVar(value="--")
        self.straight_id_var = tk.StringVar(value="--")
        self.axis_dist_var = tk.StringVar(value="--")
        # scheme-3 overall concentricity metrics
        self.conc_max_var = tk.StringVar(value="--")
        self.axis_span_max_var = tk.StringVar(value="--")
        self.od_tilt_var = tk.StringVar(value="--")
        self.od_endoff_var = tk.StringVar(value="--")
        self.id_tilt_var = tk.StringVar(value="--")
        self.id_endoff_var = tk.StringVar(value="--")

        # Operator confirm (modal dialog) infra for AutoFlow
        self._op_confirm_lock = threading.Lock()
        self._op_confirm_token = None
        self._op_confirm_evt = None
        self._op_confirm_result = None
        self._op_confirm_popup = None

        # ------------------------------
        # Run/Export (MSA)
        # ------------------------------
        self.pipe_sn_var = tk.StringVar(value="--")  # 流水号 (date + recipe + seq)
        self.meas_seq_var = tk.StringVar(value="--")  # 测量计数（当日序号）
        self.meas_start_var = tk.StringVar(value="--")  # 开始时间 (HH:MM:SS)
        self.meas_elapsed_var = tk.StringVar(value="--")  # 耗时 (HH:MM:SS)

        # Summary (main screen)
        self.max_od_dev_var = tk.StringVar(value="--")
        self.max_id_dev_var = tk.StringVar(value="--")
        self.max_od_round_var = tk.StringVar(value="--")
        self.max_id_round_var = tk.StringVar(value="--")
        self.od_mean_var = tk.StringVar(value="--")
        self.od_dpp_var = tk.StringVar(value="--")
        self.od_e_var = tk.StringVar(value="--")
        self.id_mean_var = tk.StringVar(value="--")
        self.id_dpp_var = tk.StringVar(value="--")
        # Optional: length measurement summary (main screen)
        self.len_meas_var = tk.StringVar(value="--")

        self._max_od_dev = None
        self._max_id_dev = None
        self._max_od_round = None
        self._max_id_round = None
        self._run_serial: Optional[str] = None
        self._run_id: Optional[str] = None
        self._run_start_ts: Optional[float] = None
        self._run_end_ts: Optional[float] = None
        self._auto_rows: list[MeasureRow] = []
        self._auto_raw_points: list[dict] = []
        self._auto_export_done: bool = False

        # Summary extrema caches (computed from per-section results)
        self._max_od_dev_abs: Optional[float] = None
        self._max_id_dev_abs: Optional[float] = None
        self._max_od_round: Optional[float] = None
        self._max_id_round: Optional[float] = None

        # Per-section sampling coverage/info cache (key: 1-based section index)
        self._section_cov_info: dict[int, dict] = {}
        # Map 1-based section index -> Treeview iid (used to update cov columns asynchronously)
        self._sec_iid_map: dict[int, str] = {}
        self._auto_cur_sec_idx: Optional[int] = None
        self._selected_sec_idx: Optional[int] = None
        self._axis_dist: Optional[float] = None
        self._conc_max: Optional[float] = None
        self._axis_span_max: Optional[float] = None

        # Last overall metrics (for summary at DONE)
        self._last_straight_od: Optional[float] = None
        self._last_straight_id: Optional[float] = None
        self._last_axis_dist: Optional[float] = None
        self._last_conc_max: Optional[float] = None
        self._last_axis_span_max: Optional[float] = None
        self._last_od_tilt_deg: Optional[float] = None
        self._last_od_end_off_mm: Optional[float] = None
        self._last_od_slope: Optional[float] = None
        self._last_id_tilt_deg: Optional[float] = None
        self._last_id_end_off_mm: Optional[float] = None
        self._last_id_slope: Optional[float] = None
        self._run_summary: dict = {}

        # Auto length result produced by AutoFlow (optional)
        self._run_len_result: Optional[dict] = None

        self._build_ui()
        # start rolling error banner ticker
        self.after(180, self._tick_error_banner)
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
                    else "M1,1"
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

        # Top bar: left = PLC status; right = rolling error banner.
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)
        ttk.Label(top, textvariable=self.plc_status_var).grid(row=0, column=0, sticky="w")
        self._err_banner_lbl = tk.Label(
            top,
            textvariable=self.err_banner_var,
            fg="red",
            anchor="e",
            justify="right",
        )
        self._err_banner_lbl.grid(row=0, column=1, sticky="e")


        nb = ttk.Notebook(self)
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        # keep a reference for future extensions
        self._notebook = nb

        tab_main = ttk.Frame(nb)
        tab_axis_cal = ttk.Frame(nb)
        tab_axis = ttk.Frame(nb)
        tab_recipe = ttk.Frame(nb)
        tab_gauge = ttk.Frame(nb)
        tab_keytest = ttk.Frame(nb)

        # Main operation tab first (left-most) and selected by default.
        nb.add(tab_main, text="主操作/自动测量")
        nb.add(tab_axis_cal, text="轴位标定")
        nb.add(tab_axis, text="轴参数/调试")
        nb.add(tab_recipe, text="配方/示教")
        nb.add(tab_gauge, text="外设通信")
        nb.add(tab_keytest, text="按键测试")

        build_main_screen(self, tab_main)
        build_axis_cal_screen(self, tab_axis_cal)
        build_axis_screen(self, tab_axis)
        build_recipe_screen(self, tab_recipe)
        build_gauge_screen(self, tab_gauge)
        build_key_test_screen(self, tab_keytest)

        try:
            nb.select(tab_main)
        except Exception:
            pass

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
    # Top error banner (marquee)
    # =========================
    def _collect_top_errors(self) -> list[str]:
        """Collect runtime errors for top-bar banner.

        Goal: operators can notice axis errors without opening pages.
        Keep the message terse; it scrolls in red.
        """
        msgs: list[str] = []

        # Axis errors/warnings
        try:
            with self._snapshot_lock:
                axes = list(self._axis_snapshot)
            for i, ax in enumerate(axes):
                e = int(getattr(ax, "err", 0) or 0)
                w = int(getattr(ax, "warn", 0) or 0)
                if e:
                    msgs.append(f"AX{i} ERR={e}")
                if w:
                    msgs.append(f"AX{i} WARN={w}")
        except Exception:
            pass

        # Gauge connection error (if any)
        try:
            gerr = str(getattr(self, "gauge_err_var", tk.StringVar()).get()).strip()
            if gerr and gerr != "-":
                msgs.append(f"GAUGE: {gerr}")
        except Exception:
            pass

        # Auto-flow error state (if any)
        try:
            st = str(getattr(self, "auto_state_var", tk.StringVar()).get()).strip()
            if st.upper().startswith("ERR"):
                amsg = str(getattr(self, "auto_msg_var", tk.StringVar()).get()).strip()
                if amsg and amsg != "-":
                    msgs.append(f"AUTO: {amsg}")
                else:
                    msgs.append("AUTO: ERR")
        except Exception:
            pass

        return msgs

    def _update_error_banner_source(self):
        try:
            msgs = self._collect_top_errors()
            src = self._err_banner_gap.join(msgs) if msgs else ""
        except Exception:
            src = ""

        if src != self._err_banner_src:
            self._err_banner_src = src
            self._err_banner_pos = 0

    def _tick_error_banner(self):
        """Periodic marquee refresh."""
        try:
            self._update_error_banner_source()
            src = self._err_banner_src
            if not src:
                self.err_banner_var.set("")
            else:
                lbl = getattr(self, "_err_banner_lbl", None)
                # estimate visible character count from pixel width
                try:
                    wpx = int(lbl.winfo_width()) if lbl is not None else 600
                except Exception:
                    wpx = 600
                try:
                    fnt = tkfont.Font(font=lbl.cget("font")) if lbl is not None else tkfont.nametofont("TkDefaultFont")
                    ch_px = max(1, int(fnt.measure("0")))
                except Exception:
                    ch_px = 8
                n = max(24, int(wpx / ch_px))

                if len(src) <= n:
                    self.err_banner_var.set(src)
                else:
                    loop = src + self._err_banner_gap
                    L = len(loop)
                    if L <= 0:
                        self.err_banner_var.set(src)
                    else:
                        pos = int(self._err_banner_pos) % L
                        if pos + n <= L:
                            view = loop[pos:pos + n]
                        else:
                            view = loop[pos:] + loop[: (pos + n - L)]
                        self.err_banner_var.set(view)
                        self._err_banner_pos = (pos + 1) % L
        except Exception:
            # keep banner silent; do not crash UI
            pass
        finally:
            # keep running
            try:
                self.after(130, self._tick_error_banner)
            except Exception:
                pass

    # =========================
    # Key test (PLC X/Y points via Modbus coils)
    # =========================
    def plc_write_y_point(self, y_point: int, value: int) -> None:
        """Thread-safe one-shot write to a Y point (Modbus coil). Safe to call from any thread."""
        try:
            y_point = int(y_point)
            value = 1 if int(value) != 0 else 0
            # X/Y points are octal labels: no 8/9 in coil address space.
            if y_point < 8:
                idx = y_point
            else:
                idx = y_point - 2  # skip 8/9
            coil = int(KEYTEST_Y_BASE_COIL) + int(idx)
            self.cmd_q.put(CmdWriteCoil(coil_addr=coil, value=value))
        except Exception:
            pass

    def get_x_point(self, x_point: int) -> int:
        """Get cached X point value (0/1). Safe to call from any thread."""
        try:
            x_point = int(x_point)
            i = KEYTEST_X_POINTS.index(x_point)
        except Exception:
            return 0
        try:
            with self._keytest_bits_lock:
                arr = self._keytest_x_points_state
                return int(arr[i]) if 0 <= i < len(arr) else 0
        except Exception:
            return 0

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
            self.plc_write_y_point(y_point, value)
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
        """Update UI and cached X/Y states from polled coil bits.

        Also handles edge shortcuts:
        - X2 rising: start AutoFlow (same as clicking 'Start')
        - X3 rising: confirm the active operator dialog (if any)
        """
        try:
            self._keytest_x_bits = x_bits
            self._keytest_y_bits = y_bits

            cur_x = [0 for _ in range(len(KEYTEST_X_POINTS))]
            cur_y = [0 for _ in range(len(KEYTEST_Y_POINTS))]

            if isinstance(x_bits, (list, tuple)):
                for i, p in enumerate(KEYTEST_X_POINTS):
                    try:
                        pp = int(p)
                        idx = pp if pp < 8 else pp - 2
                        v = 1 if bool(x_bits[int(idx)]) else 0
                        cur_x[i] = v
                        self.keytest_x_vars[i].set(v)
                    except Exception:
                        pass

            if isinstance(y_bits, (list, tuple)):
                for i, p in enumerate(KEYTEST_Y_POINTS):
                    try:
                        pp = int(p)
                        idx = pp if pp < 8 else pp - 2
                        v = 1 if bool(y_bits[int(idx)]) else 0
                        cur_y[i] = v
                        self.keytest_y_vars[i].set(v)
                    except Exception:
                        pass

            start_edge = False
            confirm_edge = False
            with self._keytest_bits_lock:
                prev_x = list(self._keytest_x_points_state)
                self._keytest_x_points_state = list(cur_x)
                self._keytest_y_points_state = list(cur_y)

            try:
                i2 = KEYTEST_X_POINTS.index(2)
                if i2 < len(prev_x):
                    start_edge = (prev_x[i2] == 0) and (cur_x[i2] == 1)
            except Exception:
                pass
            try:
                i3 = KEYTEST_X_POINTS.index(3)
                if i3 < len(prev_x):
                    confirm_edge = (prev_x[i3] == 0) and (cur_x[i3] == 1)
            except Exception:
                pass

            if start_edge:
                try:
                    self._auto_start()
                except Exception:
                    pass

            if confirm_edge:
                try:
                    self._op_confirm_set('confirm')
                except Exception:
                    pass
        except Exception:
            pass

    # =========================
    # Operator confirm (no clamp feedback)
    # =========================
    def operator_confirm(self, title: str, message: str, *, allow_stop: bool = True, timeout_s: float | None = None) -> str:
        """Block current (non-UI) thread until operator confirms or stops.

        Returns: 'confirm' | 'stop' | 'timeout'.
        """
        try:
            if threading.current_thread() is threading.main_thread():
                ok = messagebox.askokcancel(title or 'Confirm', message)
                return 'confirm' if ok else 'stop'

            token = str(uuid.uuid4())
            evt = threading.Event()
            with self._op_confirm_lock:
                self._op_confirm_token = token
                self._op_confirm_evt = evt
                self._op_confirm_result = None

            self.ui_q.put(("op_confirm_show", {"token": token, "title": title, "message": message, "allow_stop": bool(allow_stop)}))

            if timeout_s is None:
                evt.wait()
            else:
                if not evt.wait(float(timeout_s)):
                    with self._op_confirm_lock:
                        if self._op_confirm_token == token and self._op_confirm_result is None:
                            self._op_confirm_result = 'timeout'
                            try:
                                evt.set()
                            except Exception:
                                pass
                    self.ui_q.put(("op_confirm_close", {"token": token}))

            with self._op_confirm_lock:
                res = self._op_confirm_result or 'timeout'
                if self._op_confirm_token == token:
                    self._op_confirm_token = None
                    self._op_confirm_evt = None
                    self._op_confirm_result = None
            return str(res)
        except Exception:
            return 'timeout'

    def _show_op_confirm_popup(self, token: str, title: str, message: str, allow_stop: bool) -> None:
        try:
            # close previous if any
            try:
                if self._op_confirm_popup is not None and self._op_confirm_popup.winfo_exists():
                    self._op_confirm_popup.destroy()
            except Exception:
                pass

            top = tk.Toplevel(self)
            self._op_confirm_popup = top
            top.title(title or '操作员确认')
            top.transient(self)
            try:
                top.grab_set()
            except Exception:
                pass

            frm = ttk.Frame(top, padding=12)
            frm.pack(fill='both', expand=True)

            lbl = ttk.Label(frm, text=message or '', wraplength=520, justify='left')
            lbl.pack(fill='x', pady=(0, 8))

            hint = ttk.Label(frm, text='提示：可按 X3 进行确认。', foreground='#666')
            hint.pack(fill='x', pady=(0, 10))

            btn_row = ttk.Frame(frm)
            btn_row.pack(fill='x')

            def _on_confirm():
                self._op_confirm_set('confirm', token=token)

            def _on_stop():
                try:
                    self._auto_stop()
                except Exception:
                    pass
                self._op_confirm_set('stop', token=token)

            b1 = ttk.Button(btn_row, text='确认夹紧 (X3)', command=_on_confirm)
            b1.pack(side='left', padx=(0, 8))

            if allow_stop:
                b2 = ttk.Button(btn_row, text='停止流程', command=_on_stop)
                b2.pack(side='left')

            def _on_close():
                # treat closing as stop
                _on_stop()

            try:
                top.protocol('WM_DELETE_WINDOW', _on_close)
                top.bind('<Return>', lambda _e: _on_confirm())
                top.bind('<Escape>', lambda _e: _on_stop())
            except Exception:
                pass

            try:
                b1.focus_set()
            except Exception:
                pass

        except Exception:
            pass

    def _close_op_confirm_popup(self, token: str) -> None:
        try:
            with self._op_confirm_lock:
                cur = self._op_confirm_token
            if token and cur and token != cur:
                return
            pop = self._op_confirm_popup
            if pop is not None and pop.winfo_exists():
                try:
                    pop.grab_release()
                except Exception:
                    pass
                try:
                    pop.destroy()
                except Exception:
                    pass
            self._op_confirm_popup = None
        except Exception:
            pass

    def _op_confirm_set(self, result: str, token: str | None = None) -> None:
        try:
            with self._op_confirm_lock:
                cur = self._op_confirm_token
                evt = self._op_confirm_evt
                if cur is None:
                    return
                if token is not None and token != cur:
                    return
                self._op_confirm_result = str(result)
            try:
                if evt is not None:
                    evt.set()
            except Exception:
                pass
            # Close popup on UI thread
            try:
                pop = self._op_confirm_popup
                if pop is not None and pop.winfo_exists():
                    try:
                        pop.grab_release()
                    except Exception:
                        pass
                    pop.destroy()
            except Exception:
                pass
            self._op_confirm_popup = None
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

        # OD algorithm switch
        try:
            r.od_use_edges = bool(self.od_use_edges_var.get())
        except Exception:
            r.od_use_edges = bool(getattr(self.recipe, "od_use_edges", False))

        # ID algorithm switch (recipe only; auto-flow hookup later)
        try:
            r.id_use_fit = bool(getattr(self, 'id_use_fit_var').get())
        except Exception:
            r.id_use_fit = bool(getattr(self.recipe, 'id_use_fit', False))

        # length measurement (optional)
        try:
            r.len_enable = bool(getattr(self, "len_enable_var").get())
            r.len_z_low_approach = float(getattr(self, "len_z_low_approach_var").get())
            r.len_low_search_dist = float(getattr(self, "len_low_search_dist_var").get())
            r.len_high_search_dist = float(getattr(self, "len_high_search_dist_var").get())
            r.len_search_vel = float(getattr(self, "len_search_vel_var").get())
            r.len_search_timeout_s = float(getattr(self, "len_search_timeout_var").get())
            r.len_tol_mm = float(getattr(self, "len_tol_var").get())
            # advanced
            r.len_high_margin = float(getattr(self, "len_high_margin_var").get())
            r.len_debounce_k = int(float(getattr(self, "len_debounce_k_var").get()))
            r.len_max_stale_ms = int(float(getattr(self, "len_max_stale_ms_var").get()))
            r.len_backoff_mm = float(getattr(self, "len_backoff_var").get())
        except Exception:
            # Keep current recipe values when UI vars are not available (backward compatible)
            try:
                for k in (
                    "len_enable",
                    "len_z_low_approach",
                    "len_low_search_dist",
                    "len_high_search_dist",
                    "len_search_vel",
                    "len_search_timeout_s",
                    "len_tol_mm",
                    "len_high_margin",
                    "len_debounce_k",
                    "len_max_stale_ms",
                    "len_backoff_mm",
                ):
                    if hasattr(self.recipe, k):
                        setattr(r, k, getattr(self.recipe, k))
            except Exception:
                pass

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
            "od_use_edges": bool(getattr(r, "od_use_edges", False)),
            "id_use_fit": bool(getattr(r, 'id_use_fit', False)),

            # Length measurement (OD gauge edge search)
            "len_enable": bool(getattr(r, "len_enable", False)),
            "len_z_low_approach": float(getattr(r, "len_z_low_approach", 1300.0)),
            "len_low_search_dist": float(getattr(r, "len_low_search_dist", 220.0)),
            "len_high_search_dist": float(getattr(r, "len_high_search_dist", 220.0)),
            "len_search_vel": float(getattr(r, "len_search_vel", 5.0)),
            "len_search_timeout_s": float(getattr(r, "len_search_timeout_s", 12.0)),
            "len_tol_mm": float(getattr(r, "len_tol_mm", 20.0)),
            "len_high_margin": float(getattr(r, "len_high_margin", 20.0)),
            "len_debounce_k": int(getattr(r, "len_debounce_k", 6)),
            "len_max_stale_ms": int(getattr(r, "len_max_stale_ms", 300)),
            "len_backoff_mm": float(getattr(r, "len_backoff_mm", 2.0)),
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
            fs = str(
                data.get(
                    "fit_strategy",
                    getattr(self.recipe, "fit_strategy", "b 原始点按bin权重均衡"),
                )
            )
            if hasattr(self, "fit_strategy_var"):
                self.fit_strategy_var.set(fs)
            if hasattr(self, "fit_strategy_combo") and self.fit_strategy_combo is not None:
                vals = list(self.fit_strategy_combo.cget("values") or [])
                if fs in vals:
                    self.fit_strategy_combo.current(vals.index(fs))
        except Exception:
            pass

        # OD algorithm switch (persisted)
        try:
            use_edges = bool(
                data.get(
                    "od_use_edges",
                    data.get(
                        "od_algo_edges",
                        getattr(self.recipe, "od_use_edges", False),
                    ),
                )
            )
            if hasattr(self, "od_use_edges_var"):
                self.od_use_edges_var.set(bool(use_edges))
            setattr(self.recipe, "od_use_edges", bool(use_edges))

        except Exception:
            pass

        # ID algorithm switch (persisted)
        try:
            use_fit = bool(
                data.get(
                    'id_use_fit',
                    data.get('id_algo_fit', getattr(self.recipe, 'id_use_fit', False)),
                )
            )
            if hasattr(self, 'id_use_fit_var'):
                self.id_use_fit_var.set(bool(use_fit))
            setattr(self.recipe, 'id_use_fit', bool(use_fit))
        except Exception:
            pass

        # length measurement (optional)
        try:
            self.recipe.len_enable = bool(data.get("len_enable", getattr(self.recipe, "len_enable", False)))
            self.recipe.len_z_low_approach = float(data.get("len_z_low_approach", getattr(self.recipe, "len_z_low_approach", 1300.0)))
            self.recipe.len_low_search_dist = float(data.get("len_low_search_dist", getattr(self.recipe, "len_low_search_dist", 220.0)))
            self.recipe.len_high_search_dist = float(data.get("len_high_search_dist", getattr(self.recipe, "len_high_search_dist", 220.0)))
            self.recipe.len_search_vel = float(data.get("len_search_vel", getattr(self.recipe, "len_search_vel", 5.0)))
            self.recipe.len_search_timeout_s = float(data.get("len_search_timeout_s", getattr(self.recipe, "len_search_timeout_s", 12.0)))
            self.recipe.len_tol_mm = float(data.get("len_tol_mm", getattr(self.recipe, "len_tol_mm", 20.0)))

            self.recipe.len_high_margin = float(data.get("len_high_margin", getattr(self.recipe, "len_high_margin", 20.0)))
            self.recipe.len_debounce_k = int(data.get("len_debounce_k", getattr(self.recipe, "len_debounce_k", 6)))
            self.recipe.len_max_stale_ms = int(data.get("len_max_stale_ms", getattr(self.recipe, "len_max_stale_ms", 300)))
            self.recipe.len_backoff_mm = float(data.get("len_backoff_mm", getattr(self.recipe, "len_backoff_mm", 2.0)))

            if hasattr(self, "len_enable_var"):
                self.len_enable_var.set(bool(self.recipe.len_enable))
            if hasattr(self, "len_z_low_approach_var"):
                self.len_z_low_approach_var.set(str(self.recipe.len_z_low_approach))
            if hasattr(self, "len_low_search_dist_var"):
                self.len_low_search_dist_var.set(str(self.recipe.len_low_search_dist))
            if hasattr(self, "len_high_search_dist_var"):
                self.len_high_search_dist_var.set(str(self.recipe.len_high_search_dist))
            if hasattr(self, "len_search_vel_var"):
                self.len_search_vel_var.set(str(self.recipe.len_search_vel))
            if hasattr(self, "len_search_timeout_var"):
                self.len_search_timeout_var.set(str(self.recipe.len_search_timeout_s))
            if hasattr(self, "len_tol_var"):
                self.len_tol_var.set(str(self.recipe.len_tol_mm))
            if hasattr(self, "len_high_margin_var"):
                self.len_high_margin_var.set(str(self.recipe.len_high_margin))
            if hasattr(self, "len_debounce_k_var"):
                self.len_debounce_k_var.set(str(self.recipe.len_debounce_k))
            if hasattr(self, "len_max_stale_ms_var"):
                self.len_max_stale_ms_var.set(str(self.recipe.len_max_stale_ms))
            if hasattr(self, "len_backoff_var"):
                self.len_backoff_var.set(str(self.recipe.len_backoff_mm))

            try:
                self._refresh_length_info()
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
                "fit_strategy": str(getattr(r, "fit_strategy", "b 原始点按bin权重均衡")),

                # Length measurement
                "len_enable": bool(getattr(r, "len_enable", False)),
                "len_z_low_approach": float(getattr(r, "len_z_low_approach", 1300.0)),
                "len_low_search_dist": float(getattr(r, "len_low_search_dist", 220.0)),
                "len_high_search_dist": float(getattr(r, "len_high_search_dist", 220.0)),
                "len_search_vel": float(getattr(r, "len_search_vel", 5.0)),
                "len_search_timeout_s": float(getattr(r, "len_search_timeout_s", 12.0)),
                "len_tol_mm": float(getattr(r, "len_tol_mm", 20.0)),
                "len_high_margin": float(getattr(r, "len_high_margin", 20.0)),
                "len_debounce_k": int(getattr(r, "len_debounce_k", 6)),
                "len_max_stale_ms": int(getattr(r, "len_max_stale_ms", 300)),
                "len_backoff_mm": float(getattr(r, "len_backoff_mm", 2.0)),
                # Taught section positions (Z_Pos, mm)
                "section_pos_z": getattr(r, "section_pos_z", []),
                # Standby point (absolute)
                "standby_valid": bool(getattr(r, "standby_valid", False)),
                "standby_ax0_abs": float(getattr(r, "standby_ax0_abs", 0.0)),
                "standby_ax1_abs": float(getattr(r, "standby_ax1_abs", 0.0)),
                "standby_ax4_abs": float(getattr(r, "standby_ax4_abs", 0.0)),
                # Center clamp (AX2)
                "ax2_len_valid": bool(getattr(r, "ax2_len_valid", False)),
                "ax2_len_abs": float(getattr(r, "ax2_len_abs", 0.0)),
                "ax2_rot_valid": bool(getattr(r, "ax2_rot_valid", False)),
                "ax2_rot_abs": float(getattr(r, "ax2_rot_abs", 0.0)),
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

            # fit strategy (optional)
            try:
                fs = str(data.get("fit_strategy", getattr(self.recipe, "fit_strategy", "b 原始点按bin权重均衡")))
                if hasattr(self, "fit_strategy_var"):
                    self.fit_strategy_var.set(fs)
            except Exception:
                pass

            # OD algorithm switch (optional/persisted)
            try:
                use_edges = bool(
                    data.get(
                        "od_use_edges",
                        data.get(
                            "od_algo_edges",
                            getattr(self.recipe, "od_use_edges", False),
                        ),
                    )
                )
                if hasattr(self, "od_use_edges_var"):
                    self.od_use_edges_var.set(bool(use_edges))
                setattr(self.recipe, "od_use_edges", bool(use_edges))
            except Exception:
                pass

            # length measurement (optional)
            try:
                self.recipe.len_enable = bool(data.get("len_enable", getattr(self.recipe, "len_enable", False)))
                self.recipe.len_z_low_approach = float(data.get("len_z_low_approach", getattr(self.recipe, "len_z_low_approach", 1300.0)))
                self.recipe.len_low_search_dist = float(data.get("len_low_search_dist", getattr(self.recipe, "len_low_search_dist", 220.0)))
                self.recipe.len_high_search_dist = float(data.get("len_high_search_dist", getattr(self.recipe, "len_high_search_dist", 220.0)))
                self.recipe.len_search_vel = float(data.get("len_search_vel", getattr(self.recipe, "len_search_vel", 5.0)))
                self.recipe.len_search_timeout_s = float(data.get("len_search_timeout_s", getattr(self.recipe, "len_search_timeout_s", 12.0)))
                self.recipe.len_tol_mm = float(data.get("len_tol_mm", getattr(self.recipe, "len_tol_mm", 20.0)))
                self.recipe.len_high_margin = float(data.get("len_high_margin", getattr(self.recipe, "len_high_margin", 20.0)))
                self.recipe.len_debounce_k = int(data.get("len_debounce_k", getattr(self.recipe, "len_debounce_k", 6)))
                self.recipe.len_max_stale_ms = int(data.get("len_max_stale_ms", getattr(self.recipe, "len_max_stale_ms", 300)))
                self.recipe.len_backoff_mm = float(data.get("len_backoff_mm", getattr(self.recipe, "len_backoff_mm", 2.0)))

                if hasattr(self, "len_enable_var"):
                    self.len_enable_var.set(bool(self.recipe.len_enable))
                if hasattr(self, "len_z_low_approach_var"):
                    self.len_z_low_approach_var.set(str(self.recipe.len_z_low_approach))
                if hasattr(self, "len_low_search_dist_var"):
                    self.len_low_search_dist_var.set(str(self.recipe.len_low_search_dist))
                if hasattr(self, "len_high_search_dist_var"):
                    self.len_high_search_dist_var.set(str(self.recipe.len_high_search_dist))
                if hasattr(self, "len_search_vel_var"):
                    self.len_search_vel_var.set(str(self.recipe.len_search_vel))
                if hasattr(self, "len_search_timeout_var"):
                    self.len_search_timeout_var.set(str(self.recipe.len_search_timeout_s))
                if hasattr(self, "len_tol_var"):
                    self.len_tol_var.set(str(self.recipe.len_tol_mm))
                if hasattr(self, "len_high_margin_var"):
                    self.len_high_margin_var.set(str(self.recipe.len_high_margin))
                if hasattr(self, "len_debounce_k_var"):
                    self.len_debounce_k_var.set(str(self.recipe.len_debounce_k))
                if hasattr(self, "len_max_stale_ms_var"):
                    self.len_max_stale_ms_var.set(str(self.recipe.len_max_stale_ms))
                if hasattr(self, "len_backoff_var"):
                    self.len_backoff_var.set(str(self.recipe.len_backoff_mm))
            except Exception:
                pass

            # Center clamp (AX2) (optional)
            try:
                self.recipe.ax2_len_valid = bool(data.get("ax2_len_valid", getattr(self.recipe, "ax2_len_valid", False)))
                self.recipe.ax2_len_abs = float(data.get("ax2_len_abs", getattr(self.recipe, "ax2_len_abs", 0.0)))
                self.recipe.ax2_rot_valid = bool(data.get("ax2_rot_valid", getattr(self.recipe, "ax2_rot_valid", False)))
                self.recipe.ax2_rot_abs = float(data.get("ax2_rot_abs", getattr(self.recipe, "ax2_rot_abs", 0.0)))
                try:
                    self._refresh_center_positions()
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

    # =========================
    # Length measurement helpers
    # =========================
    def _len_pick_low_approach(self) -> None:
        """Pick current AX0 position as 'bottom approach' Z_disp for length measurement."""
        try:
            if not hasattr(self, "len_z_low_approach_var"):
                return
            a0 = float(self.get_axis_copy(0).act_pos)
            z = float(self.axis_cal.abs_to_z_disp(0, a0))
            self.len_z_low_approach_var.set(f"{z:.3f}")
            self._refresh_length_info()
        except Exception as e:
            messagebox.showerror("长度测量", f"取当前位置失败: {e}")

    def _get_ax0_softlims_abs(self) -> Tuple[float, float]:
        """Return (abs_min, abs_max) soft limits for AX0."""
        try:
            ac0 = self.get_axis_copy(0)
            p = float(getattr(ac0, "softlim_pos", 0.0))
            n = float(getattr(ac0, "softlim_neg", 0.0))
            # When PLC is disconnected, some values may be 0.
            if abs(p) < 1e-6 and abs(n) < 1e-6:
                raise ValueError
            if abs(p - n) < 1e-6:
                raise ValueError
            return (min(p, n), max(p, n))
        except Exception:
            return (AX0_SOFTLIM_NEG_ABS, AX0_SOFTLIM_POS_ABS)

    def _get_ax0_z_disp_limits(self) -> Tuple[float, float, float]:
        """Return (z_min, z_max, travel) in Z_disp(mm) for AX0."""
        lo_abs, hi_abs = self._get_ax0_softlims_abs()
        z1 = float(self.axis_cal.abs_to_z_disp(0, lo_abs))
        z2 = float(self.axis_cal.abs_to_z_disp(0, hi_abs))
        z_min = min(z1, z2)
        z_max = max(z1, z2)
        return (z_min, z_max, max(0.0, z_max - z_min))

    def _refresh_length_info(self) -> None:
        """Refresh length measurement read-only info (Lmax/status) on recipe screen."""
        if not hasattr(self, "len_info_var") or not hasattr(self, "len_status_var"):
            return
        try:
            enabled = False
            try:
                enabled = bool(self.len_enable_var.get())
            except Exception:
                enabled = bool(getattr(self.recipe, "len_enable", False))

            z_min, z_max, travel = self._get_ax0_z_disp_limits()

            # Parse operator inputs
            def _f(v, d=0.0):
                try:
                    return float(v)
                except Exception:
                    return float(d)

            z_low_appr = _f(getattr(self, "len_z_low_approach_var", tk.StringVar(value="0")).get(), 0.0)
            d_low = _f(getattr(self, "len_low_search_dist_var", tk.StringVar(value="0")).get(), 0.0)
            d_high = _f(getattr(self, "len_high_search_dist_var", tk.StringVar(value="0")).get(), 0.0)
            hi_margin = _f(getattr(self, "len_high_margin_var", tk.StringVar(value="0")).get(), 0.0)
            pipe_len = _f(getattr(self, "pipe_len_var", tk.StringVar(value="0")).get(), 0.0)

            # Conservative Lmax estimation based on current approach/search settings
            z_low_edge_max = min(z_max, z_low_appr + d_low)
            lmax = z_low_edge_max + hi_margin - d_high - z_min
            if lmax < 0:
                lmax = 0.0

            self.len_info_var.set(f"{lmax:.0f}")

            if not enabled:
                self.len_status_var.set("未启用")
                return

            # Basic sanity checks
            if not (z_min <= z_low_appr <= z_max):
                self.len_status_var.set("底边接近位超出行程")
                return
            if z_low_appr + d_low > z_max + 1e-6:
                self.len_status_var.set("底边慢搜超出行程")
                return
            if lmax <= 1.0:
                self.len_status_var.set("行程不足")
                return
            if pipe_len > lmax + 1e-6:
                self.len_status_var.set(f"将跳过(管长>{lmax:.0f})")
                return

            # OK
            self.len_status_var.set("OK")
        except Exception:
            # Keep UI robust: do not raise from refresh
            try:
                self.len_info_var.set("--")
                self.len_status_var.set("--")
            except Exception:
                pass

    # =========================
    # Teach: Length edge search (manual debug)
    # =========================
    def _len_try_update_measured_length(self) -> None:
        """If both edges are known, compute pipe length and update UI vars."""
        try:
            if not hasattr(self, 'len_edge_low_var') or not hasattr(self, 'len_edge_high_var'):
                return
            try:
                z_low = float(str(self.len_edge_low_var.get()).strip())
                z_high = float(str(self.len_edge_high_var.get()).strip())
            except Exception:
                return
            L = float(z_low - z_high)
            if L <= 0 or (not math.isfinite(L)):
                return
            if hasattr(self, 'len_edge_len_var'):
                self.len_edge_len_var.set(f"{L:.3f}")
            if hasattr(self, 'len_edge_state_var'):
                self.len_edge_state_var.set(f"边沿已锁定：L={L:.1f} mm")
        except Exception:
            pass

    def _teach_len_search_low_toggle(self) -> None:
        """Toggle bottom-edge search thread (GO -> non-GO)."""
        try:
            th = getattr(self, '_len_edge_search_thread', None)
            if th is not None and getattr(th, 'is_alive', lambda: False)():
                # request stop
                evt = getattr(self, '_len_edge_search_stop_evt', None)
                if evt is not None:
                    evt.set()
                try:
                    if hasattr(self, 'len_edge_state_var'):
                        self.len_edge_state_var.set('底边搜索：停止中...')
                    if hasattr(self, 'btn_len_search_low'):
                        self.btn_len_search_low.configure(text='尝试搜索底边(GO→非GO)')
                except Exception:
                    pass
                return

            # start new
            stop_evt = threading.Event()
            self._len_edge_search_stop_evt = stop_evt
            th = threading.Thread(target=self._teach_len_search_low_worker, args=(stop_evt,), daemon=True)
            self._len_edge_search_thread = th

            try:
                if hasattr(self, 'btn_len_search_low'):
                    self.btn_len_search_low.configure(text='停止搜索底边')
                if hasattr(self, 'len_edge_state_var'):
                    self.len_edge_state_var.set('底边搜索：准备...')
            except Exception:
                pass

            th.start()
        except Exception as e:
            messagebox.showerror('底边搜索', str(e))

    def _teach_len_search_high_toggle(self) -> None:
        """Toggle top-edge search thread (valid -> invalid)."""
        try:
            th = getattr(self, '_len_edge_search_high_thread', None)
            if th is not None and getattr(th, 'is_alive', lambda: False)():
                evt = getattr(self, '_len_edge_search_high_stop_evt', None)
                if evt is not None:
                    evt.set()
                try:
                    if hasattr(self, 'len_edge_state_var'):
                        self.len_edge_state_var.set('顶边搜索：停止中...')
                    if hasattr(self, 'btn_len_search_high'):
                        self.btn_len_search_high.configure(text='尝试搜索顶边(有效→无效)')
                except Exception:
                    pass
                return

            # Do not run concurrently with bottom search
            try:
                th_low = getattr(self, '_len_edge_search_thread', None)
                if th_low is not None and getattr(th_low, 'is_alive', lambda: False)():
                    evt_low = getattr(self, '_len_edge_search_stop_evt', None)
                    if evt_low is not None:
                        evt_low.set()
            except Exception:
                pass

            stop_evt = threading.Event()
            self._len_edge_search_high_stop_evt = stop_evt
            th = threading.Thread(target=self._teach_len_search_high_worker, args=(stop_evt,), daemon=True)
            self._len_edge_search_high_thread = th

            try:
                if hasattr(self, 'btn_len_search_high'):
                    self.btn_len_search_high.configure(text='停止搜索顶边')
                if hasattr(self, 'len_edge_state_var'):
                    self.len_edge_state_var.set('顶边搜索：准备...')
            except Exception:
                pass

            th.start()
        except Exception as e:
            messagebox.showerror('顶边搜索', str(e))

    def _ui_set(self, var: tk.Variable, value: str) -> None:
        """Thread-safe tk variable update."""
        try:
            self.after(0, lambda: var.set(value))
        except Exception:
            try:
                var.set(value)
            except Exception:
                pass

    def _ui_btn_text(self, btn: Any, text: str) -> None:
        """Thread-safe button text update."""
        try:
            self.after(0, lambda: btn.configure(text=text))
        except Exception:
            try:
                btn.configure(text=text)
            except Exception:
                pass

    def _velmove_start_axis(self, axis: int, vel_velmove: float, *, acc: float = 80.0, dec: float = 80.0, jerk: float = 300.0) -> None:
        """Start VelMove for a given axis with explicit setpoints (without relying on axis debug UI)."""
        ax = max(0, min(AXIS_COUNT - 1, int(axis)))
        base = self._base(ax)
        # write FP64 setpoints
        self._write_regs(base + OFF_VEL_VELMOVE, encode_float64_to_4regs(float(vel_velmove), FLOAT64_WORD_ORDER))
        self._write_regs(base + OFF_ACC, encode_float64_to_4regs(float(acc), FLOAT64_WORD_ORDER))
        self._write_regs(base + OFF_DEC, encode_float64_to_4regs(float(dec), FLOAT64_WORD_ORDER))
        self._write_regs(base + OFF_JERK, encode_float64_to_4regs(float(jerk), FLOAT64_WORD_ORDER))
        # clear other level commands (jog) then set velmove
        try:
            self.set_cmd_bits(ax, set_mask=0, clr_mask=(CMD_JOG_F_REQ | CMD_JOG_B_REQ))
        except Exception:
            pass
        self.set_cmd_bits(ax, set_mask=CMD_VELMOVE_REQ, clr_mask=0)

    def _velmove_stop_axis(self, axis: int) -> None:
        """Stop VelMove for a given axis (clear level bit + STOP pulse)."""
        ax = max(0, min(AXIS_COUNT - 1, int(axis)))
        try:
            self.set_cmd_bits(ax, set_mask=0, clr_mask=CMD_VELMOVE_REQ)
        except Exception:
            pass
        try:
            self._pulse_cmd_bits(ax, CMD_STOP_REQ)
        except Exception:
            pass

    def _teach_len_search_low_worker(self, stop_evt: threading.Event) -> None:
        """Worker thread: move to approach, VelMove +Z_disp, detect GO->nonGO, lock Z_disp."""
        # local UI helpers
        def ui_msg(msg: str) -> None:
            try:
                if hasattr(self, 'len_edge_state_var'):
                    self._ui_set(self.len_edge_state_var, msg)
            except Exception:
                pass

        def ui_done_btn() -> None:
            try:
                if hasattr(self, 'btn_len_search_low'):
                    self._ui_btn_text(self.btn_len_search_low, '尝试搜索底边(GO→非GO)')
            except Exception:
                pass

        found = False
        edge_z = None

        try:
            # --- validations ---
            if bool(getattr(self, 'sim_gauge_enabled', False)) or (hasattr(self, 'sim_gauge_var') and int(self.sim_gauge_var.get() or 0) == 1):
                ui_msg('底边搜索：模拟测径仪不支持比较器(GO)')
                return

            gw = getattr(self, 'gauge_worker', None)
            if gw is None or (not getattr(gw, 'enabled', False)):
                ui_msg('底边搜索：请先连接测径仪(串口)')
                return

            # Require comparator mode (M1,1) to get judge
            try:
                req_cmd = str(getattr(gw, 'request_cmd', '') or '').upper().replace(' ', '')
            except Exception:
                req_cmd = ''
            if (',1' not in req_cmd) and ('M0,1' not in req_cmd) and ('M1,1' not in req_cmd):
                ui_msg('底边搜索：请将测径仪请求设为 M1,1 或 M0,1 (需包含比较器字段)')
                return

            # AX0 must be enabled
            ac0 = self.get_axis_copy(0)
            if int(getattr(ac0, 'sts', 0) or 0) == 0:
                ui_msg('底边搜索：请先使能 AX0')
                return

            # Read parameters from UI/recipe
            def _f(var, d=0.0):
                try:
                    return float(var.get())
                except Exception:
                    try:
                        return float(var)
                    except Exception:
                        return float(d)

            z_appr = _f(getattr(self, 'len_z_low_approach_var', 0.0), 0.0)
            d_max = max(0.0, _f(getattr(self, 'len_low_search_dist_var', 0.0), 0.0))
            v_z = abs(_f(getattr(self, 'len_search_vel_var', 10.0), 10.0))
            timeout_s = max(1.0, _f(getattr(self, 'len_search_timeout_var', 8.0), 8.0))
            tol_z = max(0.1, _f(getattr(self, 'len_tol_var', 0.5), 0.5))
            deb_k = 2
            try:
                deb_k = int(float(getattr(self, 'len_debounce_k_var').get()))
            except Exception:
                deb_k = 2

            # Move to approach (absolute)
            ui_msg('底边搜索：移动到接近位...')
            abs_tgt = float(self.axis_cal.z_disp_to_abs(0, z_appr))
            self.movea_abs(0, abs_tgt, context='LenEdgeLowAppr')

            # Wait until close to approach
            t0 = time.time()
            z_now = float(self.axis_cal.abs_to_z_disp(0, self.get_axis_copy(0).act_pos))
            while (not stop_evt.is_set()) and (time.time() - t0 < 15.0):
                ac0 = self.get_axis_copy(0)
                if int(getattr(ac0, 'err', 0) or 0) != 0:
                    ui_msg(f"底边搜索：AX0错误({int(getattr(ac0,'err',0) or 0)})")
                    return
                z_now = float(self.axis_cal.abs_to_z_disp(0, ac0.act_pos))
                if abs(z_now - float(z_appr)) <= max(0.5, tol_z):
                    break
                time.sleep(0.05)

            if stop_evt.is_set():
                ui_msg('底边搜索：已停止')
                return

            if abs(z_now - float(z_appr)) > max(0.8, tol_z * 2.0):
                ui_msg('底边搜索：到达接近位超时')
                return

            # Start VelMove to +Z_disp direction
            ui_msg('底边搜索：慢速搜索中...')
            z_start = float(self.axis_cal.abs_to_z_disp(0, self.get_axis_copy(0).act_pos))
            # Desired +Z_disp -> abs velocity = v_z * sign_eff
            vel_abs = float(v_z) * float(self.axis_cal.sign_eff(0))
            self._velmove_start_axis(0, vel_abs, acc=80.0, dec=80.0, jerk=300.0)

            # Edge detection loop
            t_search0 = time.time()
            last_ts = 0.0
            non_go_cnt = 0
            unk_cnt = 0
            first_non_go_z = None

            while not stop_evt.is_set():
                # check distance/timeout
                ac0 = self.get_axis_copy(0)
                z_cur = float(self.axis_cal.abs_to_z_disp(0, ac0.act_pos))
                if (z_cur - z_start) >= (d_max - 1e-6) and d_max > 0:
                    ui_msg('底边搜索：未找到(到达最大距离)')
                    break
                if (time.time() - t_search0) >= timeout_s:
                    ui_msg('底边搜索：未找到(超时)')
                    break
                if int(getattr(ac0, 'err', 0) or 0) != 0:
                    ui_msg(f"底边搜索：AX0错误({int(getattr(ac0,'err',0) or 0)})")
                    break

                # request gauge once; parser thread updates gw.last
                try:
                    gw.send_request()
                except Exception:
                    pass

                # small sleep to allow serial IO
                time.sleep(0.06)

                s = None
                try:
                    s = gw.get_last()
                except Exception:
                    s = None
                if s is None:
                    continue
                if float(getattr(s, 'ts', 0.0) or 0.0) <= last_ts:
                    continue
                last_ts = float(getattr(s, 'ts', 0.0) or 0.0)

                j = str(getattr(s, 'judge', 'UNK') or 'UNK').strip().upper()
                if j == 'UNK':
                    unk_cnt += 1
                    if unk_cnt >= 8:
                        ui_msg('底边搜索：未收到比较器(GO)字段，请确认请求为 M1,1')
                        break
                    continue
                unk_cnt = 0

                if j != 'GO':
                    non_go_cnt += 1
                    if first_non_go_z is None:
                        first_non_go_z = z_cur
                    # debounce
                    if non_go_cnt >= max(1, int(deb_k)):
                        found = True
                        edge_z = float(first_non_go_z)
                        ui_msg(f"底边搜索：锁定 {edge_z:.3f} (judge={j})")
                        break
                else:
                    non_go_cnt = 0
                    first_non_go_z = None

            # stop motion always
            self._velmove_stop_axis(0)
            time.sleep(0.15)

            if found and edge_z is not None:
                try:
                    if hasattr(self, 'len_edge_low_var'):
                        self._ui_set(self.len_edge_low_var, f"{float(edge_z):.3f}")
                    # If both edges are known, update measured length
                    try:
                        self.after(0, self._len_try_update_measured_length)
                    except Exception:
                        pass
                except Exception:
                    pass
            else:
                # keep previous value, only state message already set
                pass

        except Exception as e:
            try:
                ui_msg(f"底边搜索：异常 {e}")
            except Exception:
                pass
            try:
                self._velmove_stop_axis(0)
            except Exception:
                pass
        finally:
            ui_done_btn()


    def _teach_len_search_high_worker(self, stop_evt: threading.Event) -> None:
        """Worker thread: search top edge (valid -> invalid) and lock AX0 Z_disp.

        机制说明：
        - 测径仪到达管端外侧后，经常会返回 "--.---" 之类的无效值；当前驱动会严格解析数值，
          无效帧不会更新 last sample。
        - 因此这里用“有效数据停更超过阈值(max_stale_ms)”来判断 invalid。
        """

        def ui_msg(msg: str) -> None:
            try:
                if hasattr(self, 'len_edge_state_var'):
                    self._ui_set(self.len_edge_state_var, msg)
            except Exception:
                pass

        def ui_done_btn() -> None:
            try:
                if hasattr(self, 'btn_len_search_high'):
                    self._ui_btn_text(self.btn_len_search_high, '尝试搜索顶边(有效→无效)')
            except Exception:
                pass

        found = False
        edge_z = None

        try:
            # --- validations ---
            if bool(getattr(self, 'sim_gauge_enabled', False)) or (hasattr(self, 'sim_gauge_var') and int(self.sim_gauge_var.get() or 0) == 1):
                ui_msg('顶边搜索：模拟测径仪不支持')
                return

            gw = getattr(self, 'gauge_worker', None)
            if gw is None or (not getattr(gw, 'enabled', False)):
                ui_msg('顶边搜索：请先连接测径仪(串口)')
                return

            # AX0 must be enabled
            ac0 = self.get_axis_copy(0)
            if int(getattr(ac0, 'sts', 0) or 0) == 0:
                ui_msg('顶边搜索：请先使能 AX0')
                return

            # Require bottom edge known
            if (not hasattr(self, 'len_edge_low_var')) or (str(self.len_edge_low_var.get()).strip() in ('', '--')):
                ui_msg('顶边搜索：请先搜索底边')
                return
            try:
                z_low_edge = float(str(self.len_edge_low_var.get()).strip())
            except Exception:
                ui_msg('顶边搜索：底边数据无效')
                return

            # Read parameters from UI/recipe
            def _f(var, d=0.0):
                try:
                    return float(var.get())
                except Exception:
                    try:
                        return float(var)
                    except Exception:
                        return float(d)

            pipe_len = max(0.0, _f(getattr(self, 'pipe_len_var', 0.0), 0.0))
            hi_margin = _f(getattr(self, 'len_high_margin_var', 0.0), 0.0)
            d_max = max(0.0, _f(getattr(self, 'len_high_search_dist_var', 0.0), 0.0))
            v_z = abs(_f(getattr(self, 'len_search_vel_var', 10.0), 10.0))
            timeout_s = max(1.0, _f(getattr(self, 'len_search_timeout_var', 8.0), 8.0))
            tol_z = max(0.1, _f(getattr(self, 'len_tol_var', 0.5), 0.5))
            backoff_mm = max(0.0, _f(getattr(self, 'len_backoff_var', 0.0), 0.0))

            deb_k = 2
            try:
                deb_k = int(float(getattr(self, 'len_debounce_k_var').get()))
            except Exception:
                deb_k = 2

            max_stale_ms = 300.0
            try:
                max_stale_ms = float(getattr(self, 'len_max_stale_ms_var').get())
            except Exception:
                max_stale_ms = 300.0
            max_stale_s = max(0.05, float(max_stale_ms) / 1000.0)

            # Compute approach point for top edge (in Z_disp)
            # Top edge should be about: z_high = z_low - pipe_len
            if pipe_len <= 1e-6:
                ui_msg('顶边搜索：管长(配方)为0')
                return
            z_appr = float(z_low_edge - pipe_len + hi_margin)

            # Clamp to travel limits
            z_min, z_max, _travel = self._get_ax0_z_disp_limits()
            z_appr = max(float(z_min), min(float(z_max), float(z_appr)))

            # Move to approach
            ui_msg('顶边搜索：移动到接近位...')
            abs_tgt = float(self.axis_cal.z_disp_to_abs(0, z_appr))
            self.movea_abs(0, abs_tgt, context='LenEdgeHighAppr')

            t0 = time.time()
            z_now = float(self.axis_cal.abs_to_z_disp(0, self.get_axis_copy(0).act_pos))
            while (not stop_evt.is_set()) and (time.time() - t0 < 15.0):
                ac0 = self.get_axis_copy(0)
                if int(getattr(ac0, 'err', 0) or 0) != 0:
                    ui_msg(f"顶边搜索：AX0错误({int(getattr(ac0,'err',0) or 0)})")
                    return
                z_now = float(self.axis_cal.abs_to_z_disp(0, ac0.act_pos))
                if abs(z_now - float(z_appr)) <= max(0.5, tol_z):
                    break
                time.sleep(0.05)

            if stop_evt.is_set():
                ui_msg('顶边搜索：已停止')
                return

            if abs(z_now - float(z_appr)) > max(0.8, tol_z * 2.0):
                ui_msg('顶边搜索：到达接近位超时')
                return

            # Pre-check: ensure we can get valid samples at approach
            ui_msg('顶边搜索：确认有效测量...')
            last_ts = 0.0
            last_valid_ts = 0.0
            last_valid_z = float(z_now)
            ok = False
            tchk = time.time()
            while (not stop_evt.is_set()) and (time.time() - tchk < 1.5):
                try:
                    gw.send_request()
                except Exception:
                    pass
                time.sleep(0.06)
                s = None
                try:
                    s = gw.get_last()
                except Exception:
                    s = None
                if s is None:
                    continue
                ts = float(getattr(s, 'ts', 0.0) or 0.0)
                if ts > last_ts:
                    last_ts = ts
                    last_valid_ts = ts
                    last_valid_z = float(z_now)
                    ok = True
                    break

            if stop_evt.is_set():
                ui_msg('顶边搜索：已停止')
                return
            if not ok:
                ui_msg('顶边搜索：未收到有效测量值(请检查测径仪/串口)')
                return

            # Start VelMove to -Z_disp direction
            ui_msg('顶边搜索：慢速搜索中...')
            z_start = float(self.axis_cal.abs_to_z_disp(0, self.get_axis_copy(0).act_pos))
            vel_abs = -float(v_z) * float(self.axis_cal.sign_eff(0))
            self._velmove_start_axis(0, vel_abs, acc=80.0, dec=80.0, jerk=300.0)

            t_search0 = time.time()
            invalid_cnt = 0

            while not stop_evt.is_set():
                ac0 = self.get_axis_copy(0)
                z_cur = float(self.axis_cal.abs_to_z_disp(0, ac0.act_pos))
                if (z_start - z_cur) >= (d_max - 1e-6) and d_max > 0:
                    ui_msg('顶边搜索：未找到(到达最大距离)')
                    break
                if (time.time() - t_search0) >= timeout_s:
                    ui_msg('顶边搜索：未找到(超时)')
                    break
                if int(getattr(ac0, 'err', 0) or 0) != 0:
                    ui_msg(f"顶边搜索：AX0错误({int(getattr(ac0,'err',0) or 0)})")
                    break

                # request gauge once
                try:
                    gw.send_request()
                except Exception:
                    pass
                time.sleep(0.06)

                s = None
                try:
                    s = gw.get_last()
                except Exception:
                    s = None

                if s is not None:
                    ts = float(getattr(s, 'ts', 0.0) or 0.0)
                    if ts > last_ts:
                        last_ts = ts
                        last_valid_ts = ts
                        last_valid_z = float(z_cur)
                        invalid_cnt = 0
                        continue

                # No new valid sample: judge stale
                if last_valid_ts > 0 and (time.time() - float(last_valid_ts)) >= max_stale_s:
                    invalid_cnt += 1
                else:
                    invalid_cnt = 0

                if invalid_cnt >= max(1, int(deb_k)):
                    found = True
                    edge_z = float(last_valid_z)
                    ui_msg(f"顶边搜索：锁定 {edge_z:.3f} (valid→invalid)")
                    break

            # stop motion always
            self._velmove_stop_axis(0)
            time.sleep(0.15)

            if found and edge_z is not None:
                try:
                    if hasattr(self, 'len_edge_high_var'):
                        self._ui_set(self.len_edge_high_var, f"{float(edge_z):.3f}")
                except Exception:
                    pass
                # Optional backoff to stay inside the tube (towards +Z_disp)
                if backoff_mm > 1e-6:
                    try:
                        z_back = max(float(z_min), min(float(z_max), float(edge_z) + float(backoff_mm)))
                        self.movea_abs(0, float(self.axis_cal.z_disp_to_abs(0, z_back)), context='LenEdgeHighBackoff')
                    except Exception:
                        pass
                # Update measured length if possible
                try:
                    self.after(0, self._len_try_update_measured_length)
                except Exception:
                    pass

        except Exception as e:
            try:
                ui_msg(f"顶边搜索：异常 {e}")
            except Exception:
                pass
            try:
                self._velmove_stop_axis(0)
            except Exception:
                pass
        finally:
            ui_done_btn()


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
            cmd = (self.req_cmd_var.get() or "M1,1").strip()

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
        """发送一次测径仪请求命令（默认 M1,1\\r：包含鉴别结果）。
        - 需要先“连接”，否则会提示 not enabled。
        - 返回数据由后台线程解析后，自动更新 Gauge: OD。
        """
        try:
            # NOTE:
            # 请求指令在 UI 下拉中可随时更改（例如从 M1,1 -> M0,1），
            # 但 GaugeWorker.request_cmd 只会在 configure() 时更新。
            # 因此这里在“请求一次”前强制同步最新指令，避免出现：
            #   UI 显示 M0,1 但实际仍发送 M1,1，导致返回帧只有 OUT1。
            try:
                cmd = (self.req_cmd_var.get() if hasattr(self, "req_cmd_var") else "")
                cmd = (cmd or "M1,1").strip()
                if getattr(self, "gauge_worker", None) is not None:
                    self.gauge_worker.request_cmd = cmd
            except Exception:
                pass

            self.gauge_worker.send_request()
        except Exception as e:
            self.gauge_err_var.set(f"Gauge ERROR: {e}")


    # =========================
    # OD Calibration (B)
    # =========================
    def _odcal_get_ax3_pos(self) -> Optional[float]:
        """Read AX3 act_pos (deg) from the latest PLC snapshot.

        Returns None if not available.
        """
        try:
            ac = self.get_axis_copy(3)
            return float(getattr(ac, "act_pos", 0.0) or 0.0)
        except Exception:
            return None

    def _odcal_update_rev_progress(self, theta_deg: Optional[float]) -> float:
        """Update internal unwrapped angle and return progress (deg) since start.

        Handles both continuous theta and 0..360 wrap-around.
        """
        if theta_deg is None:
            return float(self._odcal_rev_progress_deg or 0.0)

        try:
            th = float(theta_deg)
        except Exception:
            return float(self._odcal_rev_progress_deg or 0.0)

        if self._odcal_theta_start is None:
            self._odcal_theta_start = th
            self._odcal_theta_last = th
            self._odcal_theta_unwrap = th
            self._odcal_rev_progress_deg = 0.0
            return 0.0

        last = float(self._odcal_theta_last if self._odcal_theta_last is not None else th)
        dp = th - last
        # unwrap for 0..360 style angle
        if dp < -180.0:
            dp += 360.0
        elif dp > 180.0:
            dp -= 360.0

        self._odcal_theta_unwrap = float(self._odcal_theta_unwrap) + dp
        self._odcal_theta_last = th
        self._odcal_rev_progress_deg = float(self._odcal_theta_unwrap) - float(self._odcal_theta_start)
        return float(self._odcal_rev_progress_deg)

    def _odcal_rev_done(self) -> bool:
        """True if one-rev capture has reached target angle."""
        try:
            tgt = float(self._odcal_rev_target_deg or 360.0)
            prog = float(self._odcal_rev_progress_deg or 0.0)
            # tolerate small overshoot/undershoot
            return abs(prog) >= (tgt - 1.0)
        except Exception:
            return False

    def _odcal_start_ax3_rotation(self, speed_degps: float) -> None:
        """Start AX3 rotation using VelMove (deg/s)."""
        try:
            # Try to keep AX3 enabled during capture.
            try:
                self.set_cmd_bits(3, set_mask=CMD_EN_REQ, clr_mask=0)
            except Exception:
                pass
            self._velmove_start_axis(3, float(speed_degps))
            self._odcal_ax3_rotating = True
        except Exception:
            self._odcal_ax3_rotating = False
            raise

    def _odcal_stop_ax3_rotation(self) -> None:
        """Stop AX3 rotation if it was started by OD calibration."""
        try:
            if not bool(self._odcal_ax3_rotating):
                return
            self._velmove_stop_axis(3)
        finally:
            self._odcal_ax3_rotating = False

    def _odcal_start_capture(self):
        """Start OD calibration capture.

        f2_0：
        - 主要实现 UI 与接口。
        - 采集实现为“定时采样”：以一定频率发送 gauge 请求（推荐 M0,1），
          在 gauge_ok 回调中累积两路数据。
        """
        try:
            if self._odcal_capturing:
                return

            mode = (self.odcal_mode_var.get() or "timed").strip()

            # Advanced params snapshot (keep stable during capture)
            angle_src = str(self.odcal_angle_src_var.get() or "AX3").strip()
            no_angle = ("无" in angle_src) or (angle_src.upper() == "NONE")
            self._odcal_angle_enabled = (not no_angle)

            self._odcal_filter_mode = str(self.odcal_filter_var.get() or "无").strip()
            try:
                self._odcal_outlier_sigma = float(self._parse_float(self.odcal_outlier_sigma_var.get(), 3.0))
            except Exception:
                self._odcal_outlier_sigma = 3.0

            # one-rev needs theta
            if mode == "one_rev" and no_angle:
                # auto fallback to timed, to avoid confusing users
                mode = "timed"
                try:
                    self.odcal_mode_var.set("timed")
                except Exception:
                    pass
                self.odcal_msg_var.set("角度来源=无角度：已自动切换为定时采样。")

            self._odcal_one_rev = (mode == "one_rev")
            self._odcal_stop_reason = ""

            cmd = (self.odcal_cmd_var.get() or "M0,1").strip()
            if getattr(self, "gauge_worker", None) is not None:
                self.gauge_worker.request_cmd = cmd

            # Basic validation: calibration wants two outputs.
            if not cmd.upper().startswith("M0"):
                self.odcal_msg_var.set("提示：标定 B 建议使用 M0,*（同时输出 OUT1+OUT2）。")

            hz = max(1.0, float(self._parse_float(self.odcal_hz_var.get(), 20.0)))
            dur = max(0.5, float(self._parse_float(self.odcal_duration_var.get(), 10.0)))

            # reset buffers + one-rev trackers
            self._odcal_points = []
            self._odcal_drop_cnt = 0
            self._odcal_theta_start = None
            self._odcal_theta_last = None
            self._odcal_theta_unwrap = 0.0
            self._odcal_rev_progress_deg = 0.0
            self._odcal_ax3_rotating = False
            try:
                self._odcal_ax3_speed_degps = float(self._parse_float(self.odcal_rot_degps_var.get(), 10.0))
            except Exception:
                self._odcal_ax3_speed_degps = 10.0

            self._odcal_capturing = True
            self._odcal_start_ts = time.time()
            self._odcal_stop_at_ts = self._odcal_start_ts + dur  # timed: duration; one_rev: timeout

            self.odcal_state_var.set("CAPTURING")
            if self._odcal_one_rev:
                # start AX3 rotation (deg/s)
                spd = float(self._odcal_ax3_speed_degps)
                self._odcal_start_ax3_rotation(spd)
                self.odcal_msg_var.set(f"一圈采样... cmd={cmd}  {hz:.1f}Hz  spd={spd:.2f}deg/s  timeout={dur:.1f}s")
            else:
                self.odcal_msg_var.set(f"采集中... cmd={cmd}  {hz:.1f}Hz x {dur:.1f}s")

            self.odcal_n_var.set("0")
            self.odcal_elapsed_var.set("0.0s")
            self.odcal_B_candidate_var.set("--")

            # schedule tick loop (send requests)
            self._odcal_tick(hz=hz)
        except Exception as e:
            try:
                self._odcal_stop_ax3_rotation()
            except Exception:
                pass
            self.odcal_state_var.set("ERROR")
            self.odcal_msg_var.set(f"启动采集失败: {e}")
            self._odcal_capturing = False

    def _odcal_stop_capture(self, reason: str = ""):
        """Stop capture, cancel tick, and stop AX3 rotation if needed."""
        try:
            if not self._odcal_capturing:
                return

            self._odcal_capturing = False
            self._odcal_stop_reason = str(reason or self._odcal_stop_reason or "")

            # stop AX3 if we started it
            try:
                self._odcal_stop_ax3_rotation()
            except Exception:
                pass

            if self._odcal_after_id:
                try:
                    self.after_cancel(self._odcal_after_id)
                except Exception:
                    pass
                self._odcal_after_id = None

            self._odcal_stop_at_ts = None
            self.odcal_state_var.set("DONE")
            msg = "采集完成，可计算 B"
            if self._odcal_stop_reason:
                if self._odcal_stop_reason == "timeout":
                    msg = "采集结束：超时停止，可计算 B"
                elif self._odcal_stop_reason == "one_rev":
                    msg = "采集结束：完成一圈，可计算 B"
                elif self._odcal_stop_reason == "manual":
                    msg = "采集结束：手动停止，可计算 B"
            self.odcal_msg_var.set(msg)
            self._odcal_update_stats()
        except Exception:
            pass

    def _odcal_clear(self):
        try:
            try:
                self._odcal_stop_ax3_rotation()
            except Exception:
                pass
            self._odcal_points = []
            self._odcal_drop_cnt = 0
            self._odcal_capturing = False
            self._odcal_start_ts = None
            self._odcal_stop_at_ts = None
            self._odcal_one_rev = False
            self._odcal_stop_reason = ""
            self._odcal_theta_start = None
            self._odcal_theta_last = None
            self._odcal_theta_unwrap = 0.0
            self._odcal_rev_progress_deg = 0.0
            self.odcal_state_var.set("IDLE")
            self.odcal_msg_var.set("-")
            self.odcal_B_candidate_var.set("--")
            self.odcal_n_var.set("0")
            self.odcal_elapsed_var.set("--")
            self.odcal_sum_mean_var.set("--")
            self.odcal_sum_std_var.set("--")
            self.odcal_sum_min_var.set("--")
            self.odcal_sum_max_var.set("--")
            self.odcal_drop_rate_var.set("--")
        except Exception:
            pass


    def _odcal_prepare_sums(self) -> tuple[list[float], dict]:
        """Collect sum=lL+lR series and apply optional filtering/outlier removal.

        Filtering/outlier params are snapshotted at capture start (self._odcal_filter_mode etc.).
        Returns (sums_kept, meta).
        """
        sums_raw: list[float] = []
        for pt in self._odcal_points:
            v1 = pt.get("v1", None)
            v2 = pt.get("v2", None)
            if v1 is None or v2 is None:
                continue
            try:
                sums_raw.append(float(v1) + float(v2))
            except Exception:
                continue

        meta = {
            "n_raw": int(len(sums_raw)),
            "n_kept": int(len(sums_raw)),
            "filter": str(self._odcal_filter_mode or "无"),
            "sigma": float(self._odcal_outlier_sigma or 0.0),
        }
        if not sums_raw:
            return [], meta

        sums = list(sums_raw)

        # 1) debounce/filter (median)
        fmode = str(self._odcal_filter_mode or "无").strip()
        k = 0
        if "中值" in fmode:
            if "5" in fmode:
                k = 5
            else:
                k = 3
        if k >= 3:
            half = k // 2
            out = []
            n = len(sums)
            for i in range(n):
                lo = 0 if i - half < 0 else i - half
                hi = n if i + half + 1 > n else i + half + 1
                win = sorted(sums[lo:hi])
                out.append(win[len(win)//2])
            sums = out

        # 2) outlier removal by sigma
        sigma = float(self._odcal_outlier_sigma or 0.0)
        if sigma > 0.0 and len(sums) >= 6:
            mean = sum(sums) / len(sums)
            var = sum((x - mean) ** 2 for x in sums) / max(1, (len(sums) - 1))
            std = var ** 0.5
            if std > 1e-12:
                kept = [x for x in sums if abs(x - mean) <= sigma * std]
                # avoid empty result
                if len(kept) >= 3:
                    sums = kept

        meta["n_kept"] = int(len(sums))
        return sums, meta

    def _odcal_compute(self):
        """Compute B_candidate using current captured points."""
        try:
            dref = float(self._parse_float(self.odcal_dref_var.get(), 180.0))
            sums, meta = self._odcal_prepare_sums()
            if not sums:
                self.odcal_msg_var.set("无法计算：当前采集数据没有两路（OUT1+OUT2）值。请使用 M0,* 采集。")
                self.odcal_state_var.set("ERROR")
                return


            mean_sum = sum(sums) / len(sums)
            B = dref + mean_sum
            self.odcal_B_candidate_var.set(f"{B:.5f}")
            self.odcal_state_var.set("DONE")
            self.odcal_msg_var.set("已计算 B_candidate，可应用")
            self._odcal_update_stats()
        except Exception as e:
            self.odcal_state_var.set("ERROR")
            self.odcal_msg_var.set(f"计算失败: {e}")

    def _odcal_apply(self):
        """Apply B_candidate as active, save to calibration.json."""
        try:
            s = (self.odcal_B_candidate_var.get() or "").strip()
            B = float(s)
        except Exception:
            self.odcal_msg_var.set("无有效 B_candidate")
            return

        try:
            dref = float(self._parse_float(self.odcal_dref_var.get(), 180.0))
            cmd = (self.odcal_cmd_var.get() or "M0,1").strip()
            out1_map = (self.odcal_map_out1_var.get() or "L").strip().upper()
            data = self._odcal_build_record(B_active=B, D_ref=dref, cmd_used=cmd, out1_map=out1_map)
            self._odcal_save_active(data)
            self.odcal_B_active_var.set(f"{B:.5f}")
            self.odcal_state_var.set("APPLIED")
            self.odcal_msg_var.set("已应用并保存")
        except Exception as e:
            self.odcal_state_var.set("ERROR")
            self.odcal_msg_var.set(f"应用失败: {e}")

    def _odcal_export_raw(self):
        """Export raw captured points as CSV (debug)."""
        try:
            if not self._odcal_points:
                self.odcal_msg_var.set("无数据可导出")
                return
            out_dir = self._app_root_dir() / "exports" / "od_calib"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            p = out_dir / f"od_calib_raw_{ts}.csv"
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ts", "theta", "theta_rel", "raw", "v1", "j1", "v2", "j2"])
                for r in self._odcal_points:
                    w.writerow([
                        r.get("ts", ""),
                        r.get("theta", ""),
                        r.get("theta_rel", ""),
                        r.get("raw", ""),
                        r.get("v1", ""),
                        r.get("j1", ""),
                        r.get("v2", ""),
                        r.get("j2", ""),
                    ])
            self.odcal_msg_var.set(f"已导出: {p}")
        except Exception as e:
            self.odcal_msg_var.set(f"导出失败: {e}")

    def _odcal_tick(self, hz: float = 20.0):
        """Periodic tick: send gauge request and stop on timeout."""
        if not self._odcal_capturing:
            return
        now = time.time()
        if self._odcal_stop_at_ts is not None and now >= self._odcal_stop_at_ts:
            # timed: duration reached; one_rev: treat duration as timeout
            self._odcal_stop_capture("timeout" if self._odcal_one_rev else "")
            return

        # update elapsed
        if self._odcal_start_ts is not None:
            self.odcal_elapsed_var.set(f"{(now - self._odcal_start_ts):.1f}s")

        # one-rev progress check (based on AX3 angle)
        if self._odcal_one_rev:
            try:
                th = self._odcal_get_ax3_pos()
                self._odcal_update_rev_progress(th)
                if self._odcal_rev_done():
                    self._odcal_stop_capture("one_rev")
                    return
            except Exception:
                pass

        try:
            # send one request
            if getattr(self, "gauge_worker", None) is not None:
                self.gauge_worker.send_request()
        except Exception:
            pass

        # schedule next
        dt_ms = int(max(20.0, 1000.0 / float(hz)))
        try:
            self._odcal_after_id = self.after(dt_ms, lambda: self._odcal_tick(hz=hz))
        except Exception:
            self._odcal_after_id = None

    def _odcal_on_gauge_sample(self, payload: dict):
        """Hooked from UI queue 'gauge_ok'. Accumulate points when capturing."""
        if not self._odcal_capturing:
            return
        try:
            v1 = payload.get("od", None)
            v2 = payload.get("od2", None)
            j1 = str(payload.get("judge", "") or "").strip().upper()
            j2 = str(payload.get("judge2", "") or "").strip().upper()
            raw = str(payload.get("raw", "") or "").strip()
            ts = float(payload.get("ts", time.time()))

            # Bind sample to AX3 angle (deg) if enabled.
            theta = None
            theta_rel = None
            if bool(getattr(self, "_odcal_angle_enabled", True)):
                try:
                    theta = self._odcal_get_ax3_pos()
                    if self._odcal_one_rev:
                        theta_rel = self._odcal_update_rev_progress(theta)
                except Exception:
                    theta = None
                    theta_rel = None

            # basic validity: if judge exists and not GO, count as drop
            if j1 and j1 != "GO":
                self._odcal_drop_cnt += 1
            if j2 and j2 != "GO":
                self._odcal_drop_cnt += 1

            self._odcal_points.append(
                {
                    "ts": ts,
                    "raw": raw,
                    "v1": v1,
                    "j1": j1,
                    "v2": v2,
                    "j2": j2,
                    "theta": theta,
                    "theta_rel": theta_rel,
                }
            )
            self.odcal_n_var.set(str(len(self._odcal_points)))

            # If one_rev already reached, stop right after collecting this sample.
            if self._odcal_one_rev:
                try:
                    if self._odcal_rev_done():
                        self._odcal_stop_capture("one_rev")
                except Exception:
                    pass
        except Exception:
            pass

    def _odcal_update_stats(self):
        try:
            n = len(self._odcal_points)
            if n <= 0:
                return
            sums, meta = self._odcal_prepare_sums()
            if not sums:
                # only update counts
                dr = (self._odcal_drop_cnt / max(1, n))
                self.odcal_drop_rate_var.set(f"{dr*100:.1f}%")
                return

            mean_sum = sum(sums) / len(sums)
            # std
            var = sum((x - mean_sum) ** 2 for x in sums) / max(1, (len(sums) - 1))
            std_sum = math.sqrt(var)
            self.odcal_sum_mean_var.set(f"{mean_sum:.5f}")
            self.odcal_sum_std_var.set(f"{std_sum:.5f}")
            self.odcal_sum_min_var.set(f"{min(sums):.5f}")
            self.odcal_sum_max_var.set(f"{max(sums):.5f}")

            dr = (self._odcal_drop_cnt / max(1, n))
            self.odcal_drop_rate_var.set(f"{dr*100:.1f}%")
        except Exception:
            pass

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

    def _auto_clear_ui(self, preserve_run: bool = False):
        self.result_tree.delete(*self.result_tree.get_children())
        try:
            self._result_iids.clear()
        except Exception:
            self._result_iids = []
        self.straight_var.set("直线度   --（外圆） | --（内圆）")
        try:
            self.conc_var.set("整体同心度   --")
        except Exception:
            pass
        self.cov_var.set("采样覆盖率：--")
        # clear per-section coverage cache & selections
        try:
            self._section_cov_info.clear()
        except Exception:
            self._section_cov_info = {}
        try:
            self._sec_iid_map.clear()
        except Exception:
            self._sec_iid_map = {}
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

        # reset main-screen time & summary display
        if not preserve_run:
            self._run_start_ts = None
            self._run_end_ts = None
            try:
                self.meas_seq_var.set("--")
                self.meas_start_var.set("--")
                self.meas_elapsed_var.set("--")
            except Exception:
                pass
        else:
            # Keep serial/run_id/start time; only reset end/elapsed.
            try:
                self._run_end_ts = None
            except Exception:
                pass
            try:
                if hasattr(self, "meas_elapsed_var"):
                    self.meas_elapsed_var.set("00:00:00")
            except Exception:
                pass
        self._reset_summary_extrema()

        self._last_straight_od = None
        self._last_straight_id = None
        self._last_axis_dist = None
        self._run_summary = {}
        self._run_len_result = None

        # clear main-screen length display
        try:
            if hasattr(self, 'len_meas_var'):
                self.len_meas_var.set("--")
        except Exception:
            pass

        # auto length result is per-run; clear at the start of a new run
        self._run_len_result = None
        try:
            if hasattr(self, 'len_edge_state_var'):
                self.len_edge_state_var.set("--")
            if hasattr(self, 'len_edge_low_var'):
                self.len_edge_low_var.set("--")
            if hasattr(self, 'len_edge_high_var'):
                self.len_edge_high_var.set("--")
            if hasattr(self, 'len_edge_len_var'):
                self.len_edge_len_var.set("--")
        except Exception:
            pass


    # =========================
    # Main screen: time/summary helpers
    # =========================
    @staticmethod
    def _fmt_hhmmss(seconds: float) -> str:
        try:
            s = int(round(float(seconds)))
            if s < 0:
                s = 0
            h = s // 3600
            m = (s % 3600) // 60
            ss = s % 60
            return f"{h:02d}:{m:02d}:{ss:02d}"
        except Exception:
            return "--"

    def _refresh_run_time_ui(self) -> None:
        """Refresh main screen run start/elapsed time vars."""
        if not getattr(self, "_run_start_ts", None):
            try:
                self.meas_start_var.set("--")
                self.meas_elapsed_var.set("--")
            except Exception:
                pass
            return

        try:
            start_ts = float(self._run_start_ts)
        except Exception:
            return

        try:
            import datetime as _dt
            self.meas_start_var.set(_dt.datetime.fromtimestamp(start_ts).strftime("%H:%M:%S"))
        except Exception:
            pass

        try:
            end_ts = float(self._run_end_ts) if getattr(self, "_run_end_ts", None) else float(time.time())
            dur = max(0.0, end_ts - start_ts)
            self.meas_elapsed_var.set(self._fmt_hhmmss(dur))
        except Exception:
            pass

    def _reset_summary_extrema(self) -> None:
        """Reset max deviation/roundness shown in main screen summary panel."""
        self._max_od_dev_abs = None
        self._max_id_dev_abs = None
        self._max_od_round = None
        self._max_id_round = None
        try:
            self.max_od_dev_var.set("--")
            self.max_id_dev_var.set("--")
            self.max_od_round_var.set("--")
            self.max_id_round_var.set("--")
            self.od_mean_var.set("--")
            self.od_dpp_var.set("--")
            self.od_e_var.set("--")
            self.id_mean_var.set("--")
            self.id_dpp_var.set("--")
        except Exception:
            pass

    def _update_summary_extrema_from_row(self, row: "MeasureRow") -> None:
        """Update summary max values based on a newly appended section row."""
        if row is None:
            return

        # NOTE:
        # `MeasureRow.ok` is used for judgement (tolerance pass/fail) in AutoFlow.
        # Summary extrema should reflect measured data even when judgement is NG.
        import math

        def _to_float(v):
            try:
                x = float(v)
                if not math.isfinite(x):
                    return None
                return x
            except Exception:
                return None

        def _upd_max(cur, val):
            v = _to_float(val)
            if v is None:
                return cur
            if cur is None or v > cur:
                return v
            return cur

        try:
            od_dev = _to_float(getattr(row, "od_dev", None))
            id_dev = _to_float(getattr(row, "id_dev", None))
            od_round = _to_float(getattr(row, "od_round", None))
            id_round = _to_float(getattr(row, "id_round", None))

            if od_dev is not None:
                self._max_od_dev_abs = _upd_max(self._max_od_dev_abs, abs(od_dev))
            if id_dev is not None:
                self._max_id_dev_abs = _upd_max(self._max_id_dev_abs, abs(id_dev))
            if od_round is not None:
                self._max_od_round = _upd_max(self._max_od_round, od_round)
            if id_round is not None:
                self._max_id_round = _upd_max(self._max_id_round, id_round)

            if self._max_od_dev_abs is not None:
                self.max_od_dev_var.set(f"{self._max_od_dev_abs:.3f} mm")
            if self._max_id_dev_abs is not None:
                self.max_id_dev_var.set(f"{self._max_id_dev_abs:.3f} mm")
            if self._max_od_round is not None:
                self.max_od_round_var.set(f"{self._max_od_round:.3f} mm")
            if self._max_id_round is not None:
                self.max_id_round_var.set(f"{self._max_id_round:.3f} mm")
        except Exception:
            pass

    def _calc_run_summary(self) -> dict:
        """Calculate run-level summary from current section rows (in-memory only).

        Rules:
        - If no rows: summary invalid (reason=截面结果为空)
        - Summary is computed from numeric fields; judgement (row.ok) does NOT affect summarization.
        - If no numeric value can be extracted: summary invalid (reason=无有效数据)
        """
        rows = list(getattr(self, '_auto_rows', []) or [])
        if not rows:
            return {
                'ok': False,
                'reason': '截面结果为空',
            }

        import math

        def _to_float(v):
            try:
                x = float(v)
                if not math.isfinite(x):
                    return None
                return x
            except Exception:
                return None

        od_dev_abs_vals = []
        id_dev_abs_vals = []
        od_round_vals = []
        id_round_vals = []
        od_avg_vals = []
        od_runout_vals = []
        id_avg_vals = []
        conc_vals = []

        # Also keep judgement stats for debugging / future UI, but do not use it to decide summary ok.
        judge_total = 0
        judge_ok_cnt = 0

        for r in rows:
            try:
                judge_total += 1
                if bool(getattr(r, 'ok', True)):
                    judge_ok_cnt += 1
            except Exception:
                pass

            od_dev = _to_float(getattr(r, 'od_dev', None))
            id_dev = _to_float(getattr(r, 'id_dev', None))
            od_round = _to_float(getattr(r, 'od_round', None))
            id_round = _to_float(getattr(r, 'id_round', None))
            id_avg = _to_float(getattr(r, 'id_avg', None))
            od_avg = _to_float(getattr(r, 'od_avg', None))
            od_runout = _to_float(getattr(r, 'od_runout', None))
            conc = _to_float(getattr(r, 'concentricity', None))

            if od_dev is not None:
                od_dev_abs_vals.append(abs(od_dev))
            if id_dev is not None:
                id_dev_abs_vals.append(abs(id_dev))
            if od_round is not None:
                od_round_vals.append(od_round)
            if id_round is not None:
                id_round_vals.append(id_round)
            if od_avg is not None:
                od_avg_vals.append(od_avg)
            if id_avg is not None:
                id_avg_vals.append(id_avg)
            if od_runout is not None:
                od_runout_vals.append(od_runout)
            if conc is not None:
                conc_vals.append(conc)

        if not (od_dev_abs_vals or id_dev_abs_vals or od_round_vals or id_round_vals or od_avg_vals or od_runout_vals or conc_vals):
            return {
                'ok': False,
                'reason': '无有效数据',
            }

        max_od_dev = max(od_dev_abs_vals) if od_dev_abs_vals else None
        max_id_dev = max(id_dev_abs_vals) if id_dev_abs_vals else None
        max_od_round = max(od_round_vals) if od_round_vals else None
        max_id_round = max(id_round_vals) if id_round_vals else None
        max_od_round = max(od_round_vals) if od_round_vals else None
        max_id_round = max(id_round_vals) if id_round_vals else None

        od_mean = (sum(od_avg_vals) / len(od_avg_vals)) if od_avg_vals else None
        od_d_pp = float(max_od_round) if max_od_round is not None else None
        id_mean = (sum(id_avg_vals) / len(id_avg_vals)) if id_avg_vals else None
        id_d_pp = float(max_id_round) if max_id_round is not None else None

        conc_max = max(conc_vals) if conc_vals else None

        od_e = None
        try:
            if bool(getattr(self.recipe, 'od_use_edges', False)) and od_runout_vals:
                od_e = float(max(od_runout_vals)) / 2.0
        except Exception:
            od_e = None

        return {
            'ok': True,
            'reason': '',
            'max_od_dev_abs': float(max_od_dev) if max_od_dev is not None else None,
            'max_id_dev_abs': float(max_id_dev) if max_id_dev is not None else None,
            'max_od_round': float(max_od_round) if max_od_round is not None else None,
            'max_id_round': float(max_id_round) if max_id_round is not None else None,
            'od_mean': float(od_mean) if od_mean is not None else None,
            'od_d_pp': float(od_d_pp) if od_d_pp is not None else None,
            'od_e': float(od_e) if od_e is not None else None,
            'id_mean': float(id_mean) if id_mean is not None else None,
            'id_d_pp': float(id_d_pp) if id_d_pp is not None else None,
            'straight_od': getattr(self, '_last_straight_od', None),
            'straight_id': getattr(self, '_last_straight_id', None),
            'axis_dist': getattr(self, '_last_axis_dist', None),
            'conc_max': float(conc_max) if conc_max is not None else getattr(self, '_last_conc_max', None),
            'axis_span_max': getattr(self, '_last_axis_span_max', None),
            'od_tilt_deg': getattr(self, '_last_od_tilt_deg', None),
            'od_end_off_mm': getattr(self, '_last_od_end_off_mm', None),
            'od_slope': getattr(self, '_last_od_slope', None),
            'id_tilt_deg': getattr(self, '_last_id_tilt_deg', None),
            'id_end_off_mm': getattr(self, '_last_id_end_off_mm', None),
            'id_slope': getattr(self, '_last_id_slope', None),
            'judge_ok_cnt': int(judge_ok_cnt),
            'judge_total': int(judge_total),
        }

    def _apply_run_summary_to_ui(self, summary: dict) -> None:
        """Apply computed summary to main-screen result panel.

        This function only updates UI (StringVar/labels).
        """
        try:
            self._run_summary = dict(summary or {})
        except Exception:
            self._run_summary = {}

        if not summary or not bool(summary.get('ok', False)):
            reason = str((summary or {}).get('reason', '') or '')
            # reset result panel fields
            try:
                self._set_straight_label(None, None, None)
            except Exception:
                pass
            try:
                self.max_od_dev_var.set('--')
                self.max_id_dev_var.set('--')
                self.max_od_round_var.set('--')
                self.max_id_round_var.set('--')
                self.od_mean_var.set('--')
                self.od_dpp_var.set('--')
                self.od_e_var.set('--')
                self.id_mean_var.set('--')
                self.id_dpp_var.set('--')
                self.od_tilt_var.set('--')
                self.od_endoff_var.set('--')
                self.id_tilt_var.set('--')
                self.id_endoff_var.set('--')
            except Exception:
                pass
            if reason:
                try:
                    cur = str(self.auto_msg_var.get() or '')
                    if cur in ('-', '', 'None'):
                        self.auto_msg_var.set(f'汇总失败: {reason}')
                    elif '汇总失败' not in cur:
                        self.auto_msg_var.set(f'{cur} | 汇总失败: {reason}')
                except Exception:
                    pass
            return

        # straightness + axis distance
        try:
            self._set_straight_label(
                summary.get('straight_od'),
                summary.get('straight_id'),
                summary.get('axis_dist'),
                summary.get('conc_max'),
                summary.get('axis_span_max'),
            )
        except Exception:
            pass

        def _set_var(var, val, unit=' mm'):
            try:
                if val is None:
                    var.set('--')
                else:
                    var.set(f'{float(val):.3f}{unit}')
            except Exception:
                try:
                    var.set('--')
                except Exception:
                    pass

        _set_var(self.max_od_dev_var, summary.get('max_od_dev_abs'))
        _set_var(self.max_id_dev_var, summary.get('max_id_dev_abs'))
        _set_var(self.max_od_round_var, summary.get('max_od_round'))
        _set_var(self.max_id_round_var, summary.get('max_id_round'))

        _set_var(self.od_mean_var, summary.get('od_mean'), unit=' mm')
        _set_var(self.od_dpp_var, summary.get('od_d_pp'), unit=' mm')
        _set_var(self.od_e_var, summary.get('od_e'), unit=' mm')

        _set_var(self.id_mean_var, summary.get('id_mean'), unit=' mm')
        _set_var(self.id_dpp_var, summary.get('id_d_pp'), unit=' mm')

        # axis-line orientation
        # NOTE: tilt angles are typically very small (<<0.1°). Show 3 decimals to avoid displaying 0.00°.
        try:
            self.od_tilt_var.set("--" if summary.get('od_tilt_deg') is None else f"{float(summary.get('od_tilt_deg')):.3f}°")
            self.od_endoff_var.set("--" if summary.get('od_end_off_mm') is None else f"{float(summary.get('od_end_off_mm')):.3f} mm")
            self.id_tilt_var.set("--" if summary.get('id_tilt_deg') is None else f"{float(summary.get('id_tilt_deg')):.3f}°")
            self.id_endoff_var.set("--" if summary.get('id_end_off_mm') is None else f"{float(summary.get('id_end_off_mm')):.3f} mm")
        except Exception:
            pass

    def _compute_and_apply_run_summary(self) -> None:
        """Compute and apply summary (best-effort).

        Called on DONE, and may be called again if late post-calc data arrives.
        """
        try:
            s = self._calc_run_summary()
            self._apply_run_summary_to_ui(s)
        except Exception as e:
            try:
                self._apply_run_summary_to_ui({'ok': False, 'reason': f'异常: {e}'})
            except Exception:
                pass

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

    def read_cl_id_sync(self, timeout_s: float = 0.35) -> Tuple[Optional[float], Optional[int], Optional[int]]:

        """Read CL ID (OUT4, DINT32) and update counter (UINT32) on-demand.


        Returns: (id_mm, raw_dint, upd_cnt)

        id_mm is None when raw indicates invalid/standby/over-range or read fails.

        """

        try:

            regs = self._read_regs_sync(CL_IN_BASE_D + CL_ID_WORD_OFF, 2, timeout_s=timeout_s)

            regs2 = self._read_regs_sync(CL_IN_BASE_D + CL_ID_UPD_WORD_OFF, 2, timeout_s=timeout_s)

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

                # ID uses OUT4 scale (typically 0.001 mm/LSB)
                try:
                    from config.addresses import CL_ID_SCALE_MM
                    id_mm = float(raw) * float(CL_ID_SCALE_MM)
                except Exception:
                    id_mm = float(raw) * float(CL_OUT_SCALE_MM)

            # Apply active ID calibration (δc) to chord OUT4.
            # Note: OUT4 is chord length, not true diameter.
            try:
                delta = float(self.idcal_delta_active_var.get())
                id_mm += delta
            except Exception:
                pass

            return (id_mm, raw, cnt)

        except Exception:

            return (None, None, None)



    def read_cl_out3_sync(self, timeout_s: float = 0.35) -> Tuple[Optional[float], Optional[int], Optional[int]]:

        """Backward-compatible alias of read_cl_id_sync().


        Historical versions used OUT3 as ID; current mapping uses OUT4.

        """

        return self.read_cl_id_sync(timeout_s=timeout_s)


    def set_cmd_bits(self, axis: int, set_mask: int = 0, clr_mask: int = 0):
        self.cmd_q.put(CmdSetCmdMask(axis=axis, set_mask=set_mask, clr_mask=clr_mask))

    def set_plc_poll_profile(self, profile: str = "normal") -> None:
        # Set PLC worker background polling profile.
        # profile:
        #   - 'normal': poll all axes + CL + keytest
        #   - 'sampling': poll only AX3 and disable CL/keytest background polling
        try:
            prof = str(profile or 'normal').strip().lower()
            if prof not in ('normal', 'sampling'):
                prof = 'normal'
            self._plc_poll_profile_req = prof
            self.cmd_q.put(CmdSetPollProfile(profile=prof))
        except Exception:
            pass


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
                # Keepout reference must be consistent with section/teach/auto computations.
                # In those contexts we prefer the taught AX2 rotation measurement position when valid.
                ax2_abs = float(
                    self._get_ax2_keepout_ref_abs(prefer_rot=self._ctx_use_ax2_rot_ref(context))
                )
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

                    # CL (Keyence) live values (OUT1..OUT5)
                    try:
                        out1_mm = payload.get('cl_out1_mm', None)
                        out1_raw = payload.get('cl_out1_raw', None)
                        out1_cnt = payload.get('cl_out1_cnt', None)
                        out2_mm = payload.get('cl_out2_mm', None)
                        out2_raw = payload.get('cl_out2_raw', None)
                        out2_cnt = payload.get('cl_out2_cnt', None)
                        out3_mm = payload.get('cl_out3_mm', None)
                        out3_raw = payload.get('cl_out3_raw', None)
                        out3_cnt = payload.get('cl_out3_cnt', None)
                        out4_mm = payload.get('cl_out4_mm', None)
                        out4_raw = payload.get('cl_out4_raw', None)
                        out4_cnt = payload.get('cl_out4_cnt', None)
                        out5_mm = payload.get('cl_out5_mm', None)
                        out5_raw = payload.get('cl_out5_raw', None)
                        out5_cnt = payload.get('cl_out5_cnt', None)

                        # keep latest CL snapshot for sampling/fallback (ID = OUT4)
                        try:
                            ts_now = float(time.time())
                            self._cl_id_mm_latest = None if out4_mm is None else float(out4_mm)
                            self._cl_id_raw_latest = None if out4_raw is None else int(out4_raw)
                            self._cl_id_cnt_latest = None if out4_cnt is None else int(out4_cnt)
                            self._cl_id_ts_latest = ts_now

                            self._cl_out1_mm_latest = None if out1_mm is None else float(out1_mm)
                            self._cl_out1_raw_latest = None if out1_raw is None else int(out1_raw)
                            self._cl_out1_cnt_latest = None if out1_cnt is None else int(out1_cnt)
                            self._cl_out2_mm_latest = None if out2_mm is None else float(out2_mm)
                            self._cl_out2_raw_latest = None if out2_raw is None else int(out2_raw)
                            self._cl_out2_cnt_latest = None if out2_cnt is None else int(out2_cnt)
                            self._cl_out4_mm_latest = None if out4_mm is None else float(out4_mm)
                            self._cl_out4_raw_latest = None if out4_raw is None else int(out4_raw)
                            self._cl_out4_cnt_latest = None if out4_cnt is None else int(out4_cnt)
                            self._cl_out5_mm_latest = None if out5_mm is None else float(out5_mm)
                            self._cl_out5_raw_latest = None if out5_raw is None else int(out5_raw)
                            self._cl_out5_cnt_latest = None if out5_cnt is None else int(out5_cnt)
                            self._cl_out_ts_latest = ts_now
                        except Exception:
                            pass

                        def _fmt(mm, raw, ndigits: int) -> str:
                            if mm is None:
                                return "--" if raw is None else str(int(raw))
                            return f"{float(mm):.{ndigits}f}"

                        # Update display vars
                        # CL-NavigatorN: OUT1/OUT2/OUT5 typically show 4 decimals; OUT3/OUT4 show 3 decimals.
                        self.cl_out1_var.set(_fmt(out1_mm, out1_raw, 4))
                        self.cl_out2_var.set(_fmt(out2_mm, out2_raw, 4))
                        self.cl_out3_var.set(_fmt(out3_mm, out3_raw, 3))
                        self.cl_out4_var.set(_fmt(out4_mm, out4_raw, 3))  # ID direct
                        self.cl_out5_var.set(_fmt(out5_mm, out5_raw, 4))

                        self.cl_out1_cnt_var.set("--" if out1_cnt is None else str(int(out1_cnt)))
                        self.cl_out2_cnt_var.set("--" if out2_cnt is None else str(int(out2_cnt)))
                        self.cl_out3_cnt_var.set("--" if out3_cnt is None else str(int(out3_cnt)))
                        self.cl_out4_cnt_var.set("--" if out4_cnt is None else str(int(out4_cnt)))
                        self.cl_out5_cnt_var.set("--" if out5_cnt is None else str(int(out5_cnt)))

                        # Backward compatible mirrors
                        self.cl_id_var.set(self.cl_out4_var.get())
                        self.cl_cnt_var.set("--" if out4_cnt is None else str(int(out4_cnt)))

                        # m-hat computation (match CL OUT5 formula by default): m̂ = (x1 - x2)/2
                        if out1_mm is not None and out2_mm is not None:
                            m_hat = 0.5 * float(out1_mm) - 0.5 * float(out2_mm)
                            self.cl_m_calc_var.set(f"{m_hat:.4f}")
                            if out5_mm is not None:
                                self.cl_m_diff_var.set(f"{(m_hat - float(out5_mm)):.4f}")
                            else:
                                self.cl_m_diff_var.set("--")
                        else:
                            self.cl_m_calc_var.set("--")
                            self.cl_m_diff_var.set("--")

                        # Update ID sample window only on counter change (new sample) - use OUT4
                        if out4_cnt is not None and out4_mm is not None:
                            if self._last_cl_cnt is None or int(out4_cnt) != int(self._last_cl_cnt):
                                self._last_cl_cnt = int(out4_cnt)
                                self._id_samples.append(float(out4_mm))
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


                elif k == "op_confirm_show":
                    try:
                        self._show_op_confirm_popup(
                            token=str(payload.get('token', '')),
                            title=str(payload.get('title', '操作员确认')),
                            message=str(payload.get('message', '')),
                            allow_stop=bool(payload.get('allow_stop', True)),
                        )
                    except Exception:
                        pass

                elif k == "op_confirm_close":
                    try:
                        self._close_op_confirm_popup(str(payload.get('token', '')))
                    except Exception:
                        pass

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
                    # OUT1 always present; OUT2 optional when using M0,*
                    od1 = payload.get("od", None)
                    od2 = payload.get("od2", None)
                    j1 = str(payload.get("judge", "") or "").strip()
                    j2 = str(payload.get("judge2", "") or "").strip()

                    raw = str(payload.get("raw", "") or "").strip()
                    raw_head = raw.upper().split(",", 1)[0] if raw else ""

                    jtxt1 = f" judge={j1}" if j1 else ""

                    # 显示策略：
                    # - M1: 仅 OUT1
                    # - M2: 仅 OUT2（设备返回值仍放在 od 字段里，这里按 OUT2 显示）
                    # - M0: OUT1 + OUT2
                    if raw_head == "M2" and od2 is None:
                        # 单独读取 OUT2 的模式：M2 返回值仍放在 od 字段
                        self.gauge_last_var.set(
                            f"Gauge: OUT2={float(od1):.4f} mm{jtxt1}   raw={raw}"
                        )
                    elif od2 is None:
                        # 单通道：仅 OUT1
                        self.gauge_last_var.set(
                            f"Gauge: OUT1={float(od1):.4f} mm{jtxt1}   raw={raw}"
                        )
                    else:
                        # 双通道：OUT1 + OUT2
                        jtxt2 = f" judge={j2}" if j2 else ""

                        # 若已标定 B，则给出基于 (OUT1+OUT2) 的外径 OD(B)
                        od_b_txt = ""
                        try:
                            b_txt = str(self.odcal_B_active_var.get() if hasattr(self, "odcal_B_active_var") else "").strip()
                            b = float(b_txt) if b_txt and b_txt != "--" else None
                        except Exception:
                            b = None

                        if b is not None:
                            try:
                                l_sum = float(od1) + float(od2)
                                od_b = float(b) - float(l_sum)
                                od_b_txt = f" | OD(B)={od_b:.4f} mm"
                            except Exception:
                                od_b_txt = " | OD(B)=--"
                        else:
                            od_b_txt = " | OD(B)=--"

                        self.gauge_last_var.set(
                            f"Gauge: OUT1={float(od1):.4f} mm{jtxt1} | OUT2={float(od2):.4f} mm{jtxt2}{od_b_txt}   raw={raw}"
                        )
                    self.gauge_err_var.set("")

                    # OD Calibration: consume samples when capturing
                    try:
                        self._odcal_on_gauge_sample(payload)
                    except Exception:
                        pass

                elif k == "gauge_raw":
                    # only update if no parsed value is flowing
                    pass

                elif k == "gauge_err":
                    self.gauge_err_var.set(f"Gauge ERROR: {payload.get('err')}")

                elif k == "auto_clear":
                    # AutoFlow sends auto_clear at the beginning of a run; do NOT wipe run identity/timestamps.
                    self._auto_clear_ui(preserve_run=True)

                elif k == "auto_len":
                    # Published by AutoFlow after S30 (length measurement)
                    p = payload if isinstance(payload, dict) else {}
                    try:
                        self._run_len_result = dict(p)
                    except Exception:
                        self._run_len_result = None

                    ok = bool(p.get("ok", False))
                    skipped = bool(p.get("skipped", False))
                    reason = str(p.get("reason", "") or "")
                    z_low = p.get("z_low", None)
                    z_high = p.get("z_high", None)
                    length_mm = p.get("length_mm", None)

                    # Update main-screen summary (测量结果) if present
                    try:
                        if hasattr(self, 'len_meas_var'):
                            enabled = bool(p.get('enabled', False))
                            if not enabled:
                                self.len_meas_var.set("未启用")
                            else:
                                if skipped:
                                    self.len_meas_var.set(f"跳过（{reason}）" if reason else "跳过")
                                elif ok and length_mm is not None:
                                    # show value + deviation to recipe target (if available)
                                    try:
                                        exp = float(getattr(self.recipe, 'pipe_len_mm', 0.0) or 0.0)
                                    except Exception:
                                        exp = 0.0
                                    try:
                                        tol = float(getattr(self.recipe, 'len_tol_mm', 0.0) or 0.0)
                                    except Exception:
                                        tol = 0.0
                                    try:
                                        l = float(length_mm)
                                    except Exception:
                                        l = None
                                    if l is None:
                                        self.len_meas_var.set("--")
                                    else:
                                        if exp > 1e-6:
                                            dev = l - exp
                                            if tol > 1e-6:
                                                judge_txt = "OK" if abs(dev) <= tol else "NG"
                                                self.len_meas_var.set(f"{l:.3f} mm  (Δ {dev:+.3f}, tol ±{tol:.1f})  {judge_txt}")
                                            else:
                                                self.len_meas_var.set(f"{l:.3f} mm  (Δ {dev:+.3f})")
                                        else:
                                            self.len_meas_var.set(f"{l:.3f} mm")
                                else:
                                    self.len_meas_var.set(f"失败（{reason}）" if reason else "失败")
                    except Exception:
                        pass

                    # Update recipe-screen length widgets if present
                    try:
                        if hasattr(self, 'len_edge_state_var'):
                            if skipped:
                                self.len_edge_state_var.set(f"自动长度：跳过（{reason}）" if reason else "自动长度：跳过")
                            elif ok:
                                self.len_edge_state_var.set("自动长度：OK")
                            else:
                                self.len_edge_state_var.set(f"自动长度：失败（{reason}）" if reason else "自动长度：失败")
                        if hasattr(self, 'len_edge_low_var'):
                            self.len_edge_low_var.set(f"{float(z_low):.3f}" if z_low is not None else "--")
                        if hasattr(self, 'len_edge_high_var'):
                            self.len_edge_high_var.set(f"{float(z_high):.3f}" if z_high is not None else "--")
                        if hasattr(self, 'len_edge_len_var'):
                            self.len_edge_len_var.set(f"{float(length_mm):.3f}" if length_mm is not None else "--")
                    except Exception:
                        pass

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
                        # update table row cov columns if the row already exists
                        try:
                            self._update_result_row_cov(int(sec_idx_int), info)
                        except Exception:
                            pass

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
                    conc_max = payload.get("conc_max", None)
                    axis_span_max = payload.get("axis_span_max", None)

                    # optional axis-line orientation (tilt/end offset)
                    od_tilt = payload.get("od_tilt_deg", None)
                    od_end = payload.get("od_end_off_mm", None)
                    od_slope = payload.get("od_slope", None)
                    id_tilt = payload.get("id_tilt_deg", None)
                    id_end = payload.get("id_end_off_mm", None)
                    id_slope = payload.get("id_slope", None)
                    if axis_dist is not None:
                        try:
                            self._axis_dist = float(axis_dist)
                        except Exception:
                            self._axis_dist = None
                    if conc_max is not None:
                        try:
                            self._conc_max = float(conc_max)
                        except Exception:
                            self._conc_max = None
                    if axis_span_max is not None:
                        try:
                            self._axis_span_max = float(axis_span_max)
                        except Exception:
                            self._axis_span_max = None

                    self._set_straight_label(od, idv, self._axis_dist, self._conc_max, self._axis_span_max)

                    # cache for run-level summary
                    try:
                        self._last_straight_od = None if od is None else float(od)
                    except Exception:
                        self._last_straight_od = None
                    try:
                        self._last_straight_id = None if idv is None else float(idv)
                    except Exception:
                        self._last_straight_id = None
                    try:
                        self._last_axis_dist = None if self._axis_dist is None else float(self._axis_dist)
                    except Exception:
                        self._last_axis_dist = None
                    try:
                        self._last_conc_max = None if self._conc_max is None else float(self._conc_max)
                    except Exception:
                        self._last_conc_max = None
                    try:
                        self._last_axis_span_max = None if self._axis_span_max is None else float(self._axis_span_max)
                    except Exception:
                        self._last_axis_span_max = None

                    # cache axis-line orientation
                    try:
                        self._last_od_tilt_deg = None if od_tilt is None else float(od_tilt)
                    except Exception:
                        self._last_od_tilt_deg = None
                    try:
                        self._last_od_end_off_mm = None if od_end is None else float(od_end)
                    except Exception:
                        self._last_od_end_off_mm = None
                    try:
                        self._last_od_slope = None if od_slope is None else float(od_slope)
                    except Exception:
                        self._last_od_slope = None
                    try:
                        self._last_id_tilt_deg = None if id_tilt is None else float(id_tilt)
                    except Exception:
                        self._last_id_tilt_deg = None
                    try:
                        self._last_id_end_off_mm = None if id_end is None else float(id_end)
                    except Exception:
                        self._last_id_end_off_mm = None
                    try:
                        self._last_id_slope = None if id_slope is None else float(id_slope)
                    except Exception:
                        self._last_id_slope = None

                    # reflect to UI vars (even before DONE)
                    try:
                        self.od_tilt_var.set("--" if self._last_od_tilt_deg is None else f"{float(self._last_od_tilt_deg):.3f}°")
                        self.od_endoff_var.set("--" if self._last_od_end_off_mm is None else f"{float(self._last_od_end_off_mm):.3f} mm")
                        self.id_tilt_var.set("--" if self._last_id_tilt_deg is None else f"{float(self._last_id_tilt_deg):.3f}°")
                        self.id_endoff_var.set("--" if self._last_id_end_off_mm is None else f"{float(self._last_id_end_off_mm):.3f} mm")
                    except Exception:
                        pass

                    # if DONE already, refresh summary (postcalc may arrive late)
                    try:
                        if str(self.auto_state_var.get() or '') == 'DONE':
                            self._compute_and_apply_run_summary()
                            try:
                                self._export_daily_summary_csv(status='DONE')
                            except Exception:
                                pass
                    except Exception:
                        pass

                elif k == "auto_postcalc":
                    # post-calculated eccentricity + straightness
                    ecc_od = payload.get("ecc_od", []) or []
                    ecc_id = payload.get("ecc_id", []) or []
                    od = payload.get("straight_od", None)
                    idv = payload.get("straight_id", None)
                    axis_dist = payload.get("axis_dist", None)
                    conc_max = payload.get("conc_max", None)
                    axis_span_max = payload.get("axis_span_max", None)

                    # optional axis-line orientation (tilt/end offset)
                    od_tilt = payload.get("od_tilt_deg", None)
                    od_end = payload.get("od_end_off_mm", None)
                    od_slope = payload.get("od_slope", None)
                    id_tilt = payload.get("id_tilt_deg", None)
                    id_end = payload.get("id_end_off_mm", None)
                    id_slope = payload.get("id_slope", None)
                    if axis_dist is not None:
                        try:
                            self._axis_dist = float(axis_dist)
                        except Exception:
                            self._axis_dist = None
                    if conc_max is not None:
                        try:
                            self._conc_max = float(conc_max)
                        except Exception:
                            self._conc_max = None
                    if axis_span_max is not None:
                        try:
                            self._axis_span_max = float(axis_span_max)
                        except Exception:
                            self._axis_span_max = None

                    self._set_straight_label(od, idv, self._axis_dist, self._conc_max, self._axis_span_max)

                    # cache for run-level summary
                    try:
                        self._last_straight_od = None if od is None else float(od)
                    except Exception:
                        self._last_straight_od = None
                    try:
                        self._last_straight_id = None if idv is None else float(idv)
                    except Exception:
                        self._last_straight_id = None
                    try:
                        self._last_axis_dist = None if self._axis_dist is None else float(self._axis_dist)
                    except Exception:
                        self._last_axis_dist = None
                    try:
                        self._last_conc_max = None if self._conc_max is None else float(self._conc_max)
                    except Exception:
                        self._last_conc_max = None
                    try:
                        self._last_axis_span_max = None if self._axis_span_max is None else float(self._axis_span_max)
                    except Exception:
                        self._last_axis_span_max = None

                    # cache axis-line orientation
                    try:
                        self._last_od_tilt_deg = None if od_tilt is None else float(od_tilt)
                    except Exception:
                        self._last_od_tilt_deg = None
                    try:
                        self._last_od_end_off_mm = None if od_end is None else float(od_end)
                    except Exception:
                        self._last_od_end_off_mm = None
                    try:
                        self._last_od_slope = None if od_slope is None else float(od_slope)
                    except Exception:
                        self._last_od_slope = None
                    try:
                        self._last_id_tilt_deg = None if id_tilt is None else float(id_tilt)
                    except Exception:
                        self._last_id_tilt_deg = None
                    try:
                        self._last_id_end_off_mm = None if id_end is None else float(id_end)
                    except Exception:
                        self._last_id_end_off_mm = None
                    try:
                        self._last_id_slope = None if id_slope is None else float(id_slope)
                    except Exception:
                        self._last_id_slope = None

                    # reflect to UI vars (even before DONE)
                    try:
                        self.od_tilt_var.set("--" if self._last_od_tilt_deg is None else f"{float(self._last_od_tilt_deg):.3f}°")
                        self.od_endoff_var.set("--" if self._last_od_end_off_mm is None else f"{float(self._last_od_end_off_mm):.3f} mm")
                        self.id_tilt_var.set("--" if self._last_id_tilt_deg is None else f"{float(self._last_id_tilt_deg):.3f}°")
                        self.id_endoff_var.set("--" if self._last_id_end_off_mm is None else f"{float(self._last_id_end_off_mm):.3f} mm")
                    except Exception:
                        pass

                    # if DONE already, refresh summary (postcalc may arrive late)
                    try:
                        if str(self.auto_state_var.get() or '') == 'DONE':
                            self._compute_and_apply_run_summary()
                            try:
                                self._export_daily_summary_csv(status='DONE')
                            except Exception:
                                pass
                    except Exception:
                        pass

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
                            try:
                                self._compute_and_apply_run_summary()
                            except Exception:
                                pass
                    elif st in ("ERR", "STOP"):
                        self.auto_done_var.set("测量完成: 否")
                        # freeze elapsed time on abnormal end
                        if getattr(self, '_run_end_ts', None) is None and getattr(self, '_run_start_ts', None):
                            try:
                                self._run_end_ts = float(time.time())
                            except Exception:
                                pass

        except queue.Empty:
            pass
        try:
            self._refresh_run_time_ui()
        except Exception:
            pass
        self.after(60, self._poll_ui_queue)

    def _append_result_row(self, row: MeasureRow):
        od_ecc_txt = "--" if getattr(row, "od_ecc", None) is None else f"{float(row.od_ecc):.3f}"
        id_ecc_txt = "--" if getattr(row, "id_ecc", None) is None else f"{float(row.id_ecc):.3f}"

        od_e_txt = "--" if getattr(row, "od_e", None) is None else f"{float(getattr(row, 'od_e', 0.0)):.3f}"
        od_phi_txt = "--" if getattr(row, "od_phi_deg", None) is None else f"{float(getattr(row, 'od_phi_deg', 0.0)):+.1f}"

        id_e_txt = "--" if getattr(row, "id_e", None) is None else f"{float(getattr(row, 'id_e', 0.0)):.3f}"
        id_phi_txt = "--" if getattr(row, "id_phi_deg", None) is None else f"{float(getattr(row, 'id_phi_deg', 0.0)):+.1f}"

        # fill cov columns if available (auto_cov message may arrive before/after auto_row)
        cov_info = self._section_cov_info.get(int(getattr(row, "idx", 0) or 0), {})
        cov_cols = self._format_cov_cols(cov_info)

        iid = self.result_tree.insert(
            "",
            "end",
            values=(
                row.idx,
                f"{row.x_ui:.3f}",
                f"{row.od_dev:+.3f}",
                f"{float(getattr(row, 'od_runout', 0.0)):.3f}",
                f"{row.od_round:.3f}",
                od_e_txt,
                od_phi_txt,
                od_ecc_txt,
                f"{row.id_dev:+.3f}",
                f"{float(getattr(row, 'id_runout', 0.0)):.3f}",
                f"{row.id_round:.3f}",
                id_e_txt,
                id_phi_txt,
                id_ecc_txt,
                f"{row.concentricity:.3f}",
                *cov_cols,
            ),
        )
        try:
            self._sec_iid_map[int(row.idx)] = str(iid)
        except Exception:
            pass
        try:
            self._result_iids.append(str(iid))
        except Exception:
            pass

        try:
            self._auto_rows.append(row)
        except Exception:
            pass

        # update main summary extrema
        self._update_summary_extrema_from_row(row)


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

    # ------------------------------
    # OD Calibration (B) persistence
    # ------------------------------
    def _odcal_file(self) -> Path:
        return self._app_root_dir() / "calibration" / "od_calibration.json"

    def _odcal_history_file(self) -> Path:
        return self._app_root_dir() / "calibration" / "od_calibration_history.jsonl"

    def _odcal_build_record(self, B_active: float, D_ref: float, cmd_used: str, out1_map: str) -> dict:
        """Build a calibration record (for save/history).

        Notes:
        - f2_0: 仅保存最必要的字段，为后续算法扩展预留 stats 字段。
        """
        try:
            stats = {
                "n": int(len(self._odcal_points) if hasattr(self, "_odcal_points") else 0),
                "mean_sum": self.odcal_sum_mean_var.get(),
                "std_sum": self.odcal_sum_std_var.get(),
                "min_sum": self.odcal_sum_min_var.get(),
                "max_sum": self.odcal_sum_max_var.get(),
                "drop_rate": self.odcal_drop_rate_var.get(),
            }
        except Exception:
            stats = {}

        return {
            "B_active": float(B_active),
            "D_ref": float(D_ref),
            "cmd_used": str(cmd_used or ""),
            "out_map": {"OUT1": str(out1_map or "L"), "OUT2": ("R" if str(out1_map or "L").upper() == "L" else "L")},
            "params": {
                "angle_src": str(getattr(self, "odcal_angle_src_var", None).get() if hasattr(self, "odcal_angle_src_var") else "AX3"),
                "filter": str(getattr(self, "odcal_filter_var", None).get() if hasattr(self, "odcal_filter_var") else "无"),
                "outlier_sigma": str(getattr(self, "odcal_outlier_sigma_var", None).get() if hasattr(self, "odcal_outlier_sigma_var") else "3.0"),
            },
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "stats": stats,
        }

    def _odcal_save_active(self, data: dict) -> None:
        p = self._odcal_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data or {}, f, ensure_ascii=False, indent=2)

        # append history
        try:
            hp = self._odcal_history_file()
            hp.parent.mkdir(parents=True, exist_ok=True)
            with open(hp, "a", encoding="utf-8") as f:
                f.write(json.dumps(data or {}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _odcal_load_active(self) -> None:
        p = self._odcal_file()
        if not p.exists():
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception:
            return
        try:
            b = data.get("B_active", None)
            if b is not None:
                self.odcal_B_active_var.set(f"{float(b):.5f}")
        except Exception:
            pass

        # Also prefill UI inputs for convenience
        try:
            dref = data.get("D_ref", None)
            if dref is not None:
                self.odcal_dref_var.set(f"{float(dref):.3f}")
        except Exception:
            pass
        try:
            cmd = str(data.get("cmd_used", "") or "").strip()
            if cmd:
                self.odcal_cmd_var.set(cmd)
        except Exception:
            pass
        try:
            out1 = str((data.get("out_map", {}) or {}).get("OUT1", "L") or "L").upper()
            if out1 in ("L", "R"):
                self.odcal_map_out1_var.set(out1)
        except Exception:
            pass

        # Prefill advanced params if present
        try:
            params = data.get("params", {}) or {}
            ang = str(params.get("angle_src", "") or "").strip()
            if ang:
                # accept AX3/NONE/无角度
                if ("无" in ang) or (ang.upper() == "NONE"):
                    self.odcal_angle_src_var.set("无角度")
                else:
                    self.odcal_angle_src_var.set("AX3")
            flt = str(params.get("filter", "") or "").strip()
            if flt:
                self.odcal_filter_var.set(flt)
            sig = params.get("outlier_sigma", None)
            if sig is not None:
                self.odcal_outlier_sigma_var.set(str(sig))
        except Exception:
            pass


    # ------------------------------
    # ID Calibration helpers (Chord OUT4 + m OUT5)
    # ------------------------------
    def _idcal_file(self) -> Path:
        return self._app_root_dir() / "calibration" / "id_calibration.json"

    def _idcal_history_file(self) -> Path:
        return self._app_root_dir() / "calibration" / "id_calibration_history.jsonl"

    def _idcal_save_active(self, data: dict) -> None:
        p = self._idcal_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # append history
        try:
            with open(self._idcal_history_file(), "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _idcal_load_active(self) -> None:
        p = self._idcal_file()
        if not p.exists():
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception:
            return
        try:
            delta = data.get("delta_c_mm", None)
            if delta is not None:
                self.idcal_delta_active_var.set(f"{float(delta):.4f}")
        except Exception:
            pass
        try:
            dref = data.get("D_ref", None)
            if dref is not None:
                self.idcal_dref_var.set(f"{float(dref):.3f}")
        except Exception:
            pass

    def read_cl_out145_sync(self, timeout_s: float = 0.5):
        """Read CL OUT1/OUT2/OUT4/OUT5 on-demand.

        Returns: (x1_mm, x2_mm, c_mm, m_mm, raw_dict, cnt_dict)
        """
        try:
            regs = self._read_regs_sync(CL_IN_BASE_D + CL_OUT1_WORD_OFF, 10, timeout_s=timeout_s)
            if not regs:
                return None, None, None, None, {}, {}
            regs_cnt = self._read_regs_sync(CL_IN_BASE_D + CL_OUT1_UPD_WORD_OFF, 10, timeout_s=timeout_s) or [0] * 10

            def _s32(lo, hi):
                v = ((int(hi) & 0xFFFF) << 16) | (int(lo) & 0xFFFF)
                if v & 0x80000000:
                    v -= 0x100000000
                return int(v)

            def _u32(lo, hi):
                return ((int(hi) & 0xFFFF) << 16) | (int(lo) & 0xFFFF)

            raw = {
                "out1": _s32(regs[0], regs[1]),
                "out2": _s32(regs[2], regs[3]),
                "out3": _s32(regs[4], regs[5]),
                "out4": _s32(regs[6], regs[7]),
                "out5": _s32(regs[8], regs[9]),
            }
            cnt = {
                "out1": _u32(regs_cnt[0], regs_cnt[1]) if len(regs_cnt) >= 2 else 0,
                "out2": _u32(regs_cnt[2], regs_cnt[3]) if len(regs_cnt) >= 4 else 0,
                "out3": _u32(regs_cnt[4], regs_cnt[5]) if len(regs_cnt) >= 6 else 0,
                "out4": _u32(regs_cnt[6], regs_cnt[7]) if len(regs_cnt) >= 8 else 0,
                "out5": _u32(regs_cnt[8], regs_cnt[9]) if len(regs_cnt) >= 10 else 0,
            }

            if raw["out4"] in (CL_OUT_INVALID, CL_OUT_STANDBY, CL_OUT_POS_OVER, CL_OUT_NEG_OVER):
                c_mm = None
            else:
                c_mm = float(raw["out4"]) * float(CL_OUT4_SCALE_MM)
            x1_mm = None if raw["out1"] in (CL_OUT_INVALID, CL_OUT_STANDBY, CL_OUT_POS_OVER, CL_OUT_NEG_OVER) else float(raw["out1"]) * float(CL_OUT1_SCALE_MM)
            x2_mm = None if raw["out2"] in (CL_OUT_INVALID, CL_OUT_STANDBY, CL_OUT_POS_OVER, CL_OUT_NEG_OVER) else float(raw["out2"]) * float(CL_OUT2_SCALE_MM)
            m_mm = None if raw["out5"] in (CL_OUT_INVALID, CL_OUT_STANDBY, CL_OUT_POS_OVER, CL_OUT_NEG_OVER) else float(raw["out5"]) * float(CL_OUT5_SCALE_MM)

            return x1_mm, x2_mm, c_mm, m_mm, raw, cnt
        except Exception:
            return None, None, None, None, {}, {}

    def get_cl_out145_cached(self):
        # Get latest CL OUT1/OUT2/OUT4/OUT5 from background polling.
        # Returns: (x1_mm, x2_mm, c_mm, m_mm, raw_dict, cnt_dict)
        try:
            x1 = self._cl_out1_mm_latest
            x2 = self._cl_out2_mm_latest
            c = self._cl_out4_mm_latest
            m5 = self._cl_out5_mm_latest
            raw = {
                'out1': self._cl_out1_raw_latest,
                'out2': self._cl_out2_raw_latest,
                'out4': self._cl_out4_raw_latest,
                'out5': self._cl_out5_raw_latest,
            }
            cnt = {
                'out1': self._cl_out1_cnt_latest,
                'out2': self._cl_out2_cnt_latest,
                'out4': self._cl_out4_cnt_latest,
                'out5': self._cl_out5_cnt_latest,
            }
            # Prefer m_hat computed from OUT1/OUT2 to avoid OUT5 counter lag.
            m = None
            if x1 is not None and x2 is not None:
                try:
                    m = 0.5 * float(x1) - 0.5 * float(x2)
                except Exception:
                    m = None
            if m is None:
                m = m5
            return x1, x2, c, m, raw, cnt
        except Exception:
            return None, None, None, None, {}, {}

    def _idcal_start_ax3_rotation(self, speed_degps: float) -> None:
        try:
            try:
                self.set_cmd_bits(3, set_mask=CMD_EN_REQ, clr_mask=0)
            except Exception:
                pass
            self._velmove_start_axis(3, float(speed_degps))
            self._idcal_ax3_rotating = True
        except Exception:
            self._idcal_ax3_rotating = False
            raise

    def _idcal_stop_ax3_rotation(self) -> None:
        try:
            if not bool(self._idcal_ax3_rotating):
                return
            self._velmove_stop_axis(3)
        finally:
            self._idcal_ax3_rotating = False

    def _idcal_update_rev_progress(self, theta_deg: float) -> None:
        if self._idcal_theta_start is None:
            self._idcal_theta_start = float(theta_deg)
            self._idcal_theta_last = float(theta_deg)
            self._idcal_theta_unwrap = 0.0
            self._idcal_rev_progress_deg = 0.0
            return
        last = float(self._idcal_theta_last if self._idcal_theta_last is not None else theta_deg)
        cur = float(theta_deg)
        d = cur - last
        if d < -180.0:
            d += 360.0
        elif d > 180.0:
            d -= 360.0
        self._idcal_theta_unwrap += d
        self._idcal_theta_last = cur
        self._idcal_rev_progress_deg = abs(self._idcal_theta_unwrap)

    def _idcal_rev_done(self) -> bool:
        return bool(self._idcal_rev_progress_deg >= float(self._idcal_rev_target_deg))

    def _idcal_clear(self) -> None:
        self._idcal_points = []
        self._idcal_start_ts = None
        self._idcal_stop_at_ts = None
        self._idcal_theta_start = None
        self._idcal_theta_last = None
        self._idcal_theta_unwrap = 0.0
        self._idcal_rev_progress_deg = 0.0
        self.idcal_delta_candidate_var.set("--")
        self.idcal_cmax_var.set("--")
        self.idcal_mmean_var.set("--")
        self.idcal_mpp_var.set("--")
        self.idcal_fit_diam_var.set("--")
        self.idcal_fit_e_var.set("--")
        self.idcal_fit_y0_var.set("--")
        self.idcal_fit_rmse_var.set("--")
        self.idcal_chk_err_var.set("--")
        self.idcal_chk_cov_var.set("--")
        self.idcal_chk_n_var.set("--")
        self.idcal_chk_dtheta_var.set("--")
        self.idcal_state_var.set("IDLE")
        self.idcal_msg_var.set("已清空")

    def _idcal_stop_capture(self) -> None:
        self._idcal_capturing = False
        try:
            if self._idcal_after_id is not None:
                self.after_cancel(self._idcal_after_id)
        except Exception:
            pass
        self._idcal_after_id = None
        try:
            if self._idcal_one_rev:
                self._idcal_stop_ax3_rotation()
        except Exception:
            pass
        self._idcal_one_rev = False
        # restore poll profile (in case AutoFlow had set it)
        try:
            prev = getattr(self, '_idcal_prev_poll_profile', None)
            if prev:
                self.set_plc_poll_profile(prev)
        except Exception:
            pass
        self._idcal_prev_poll_profile = None

        # If this stop belongs to a verify run, compute check result now.
        if getattr(self, "_idcal_verify_pending", False):
            try:
                self._idcal_verify_pending = False
                self._idcal_verify_compute()
                return
            except Exception as e:
                self.idcal_state_var.set("ERR")
                self.idcal_msg_var.set(f"复核失败: {e}")
                return

        self.idcal_state_var.set("STOP")
        self.idcal_msg_var.set(self._idcal_stop_reason or "已停止")

    def _idcal_start_capture(self) -> None:
        if self._idcal_capturing:
            return

        # Ensure CL is polled (no sync reads during capture).
        try:
            self._idcal_prev_poll_profile = getattr(self, '_plc_poll_profile_req', 'normal')
            self.set_plc_poll_profile('normal')
        except Exception:
            self._idcal_prev_poll_profile = None

        # Gate sampling by OUT4 update counter to avoid duplicates.
        self._idcal_last_out4_cnt = None
        self._idcal_one_rev_timeout_ts = None
        mode = str(self.idcal_mode_var.get() or "one_rev").strip()
        force_one_rev = bool(getattr(self, '_idcal_force_one_rev', False))
        self._idcal_force_one_rev = False
        self._idcal_one_rev = force_one_rev or (mode == "one_rev")
        self._idcal_points = []
        self._idcal_start_ts = time.time()
        self._idcal_stop_reason = ""

        if self._idcal_one_rev:
            # one_rev safety timeout: stop even if theta is unavailable
            try:
                spd_tmp = float(self._parse_float(self.idcal_rot_degps_var.get(), 10.0))
                spd_abs = abs(spd_tmp) if abs(spd_tmp) > 1e-6 else 10.0
                req_s = 360.0 / spd_abs
                # default timeout: generous margin (for missing θ / slow refresh)
                timeout_s = max(8.0, 2.5 * req_s)
                # If user provided a reasonable timeout (>= one rev time), respect it as an upper bound.
                try:
                    user_t = float(self._parse_float(self.idcal_duration_var.get(), timeout_s))
                    if math.isfinite(user_t) and (user_t >= req_s):
                        timeout_s = max(8.0, user_t)
                except Exception:
                    pass
                self._idcal_one_rev_timeout_ts = (self._idcal_start_ts or time.time()) + float(timeout_s)
            except Exception:
                self._idcal_one_rev_timeout_ts = (self._idcal_start_ts or time.time()) + 60.0
            try:
                spd = float(self._parse_float(self.idcal_rot_degps_var.get(), 10.0))
                self._idcal_ax3_speed_degps = spd
                self._idcal_start_ax3_rotation(spd)
            except Exception as e:
                self.idcal_state_var.set("ERR")
                self.idcal_msg_var.set(f"启动AX3失败: {e}")
                return
        else:
            try:
                dur = float(self._parse_float(self.idcal_duration_var.get(), 10.0))
                self._idcal_stop_at_ts = (self._idcal_start_ts or time.time()) + max(0.5, dur)
            except Exception:
                self._idcal_stop_at_ts = None

        self._idcal_theta_start = None
        self._idcal_theta_last = None
        self._idcal_theta_unwrap = 0.0
        self._idcal_rev_progress_deg = 0.0

        self._idcal_capturing = True
        self.idcal_state_var.set("CAPTURING")
        self.idcal_msg_var.set("采集中...")
        self._idcal_tick()

    def _idcal_tick(self) -> None:
        if not self._idcal_capturing:
            return

        now = time.time()

        # Timed mode stop
        if (not self._idcal_one_rev) and (self._idcal_stop_at_ts is not None):
            if now >= float(self._idcal_stop_at_ts):
                self._idcal_stop_reason = '定时结束'
                self._idcal_stop_capture()
                return

        # one_rev safety timeout
        if self._idcal_one_rev and (getattr(self, '_idcal_one_rev_timeout_ts', None) is not None):
            try:
                if now >= float(self._idcal_one_rev_timeout_ts):
                    self._idcal_stop_reason = '一圈超时(θ无效/刷新慢)'
                    self._idcal_stop_capture()
                    return
            except Exception:
                pass

        # Read cached theta from background snapshot (avoid sync Modbus reads)
        theta_deg = float('nan')
        try:
            with self._snapshot_lock:
                theta_deg = float(self._axis_snapshot[3].act_pos)
        except Exception:
            pass

        if self._idcal_one_rev and math.isfinite(theta_deg):
            self._idcal_update_rev_progress(float(theta_deg))
            if self._idcal_rev_done():
                self._idcal_stop_reason = '已采满一圈'
                self._idcal_stop_capture()
                return

        # Cached CL OUTs
        x1_mm, x2_mm, c_mm, m_mm, raw, cnt = self.get_cl_out145_cached()
        out4_cnt = None
        try:
            out4_cnt = cnt.get('out4', None) if isinstance(cnt, dict) else None
        except Exception:
            out4_cnt = None

        # Gate by OUT4 counter change
        accept = False
        if (c_mm is not None) and (m_mm is not None):
            if out4_cnt is None:
                accept = True
            else:
                last = getattr(self, '_idcal_last_out4_cnt', None)
                accept = (last is None) or (int(out4_cnt) != int(last))
            if accept and out4_cnt is not None:
                self._idcal_last_out4_cnt = int(out4_cnt)

        if accept:
            self._idcal_points.append({
                'ts': now,
                'theta_deg': float(theta_deg),
                'x1_mm': x1_mm,
                'x2_mm': x2_mm,
                'c_mm': float(c_mm),
                'm_mm': float(m_mm),
                'raw': raw,
                'cnt': cnt,
            })

            # lightweight live stats
            try:
                cs = [p['c_mm'] for p in self._idcal_points if p.get('c_mm') is not None]
                ms = [p['m_mm'] for p in self._idcal_points if p.get('m_mm') is not None]
                if cs:
                    self.idcal_cmax_var.set(f"{max(cs):.3f}")
                if ms:
                    self.idcal_mmean_var.set(f"{(sum(ms)/len(ms)):.4f}")
                    self.idcal_mpp_var.set(f"{(max(ms)-min(ms)):.4f}")
            except Exception:
                pass

        # schedule next
        try:
            hz = float(self._parse_float(self.idcal_hz_var.get(), 20.0))
            hz = max(1.0, min(100.0, hz))
        except Exception:
            hz = 20.0
        period_ms = int(max(5, round(1000.0 / hz)))
        self._idcal_after_id = self.after(period_ms, self._idcal_tick)

    @staticmethod
    def _lsq_fit_cos_sin(theta_rad: np.ndarray, y: np.ndarray):
        X = np.column_stack([np.ones_like(theta_rad), np.cos(theta_rad), np.sin(theta_rad)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        x0, A, B = float(beta[0]), float(beta[1]), float(beta[2])
        return x0, A, B

    def _idcal_fit_diameter(self, theta_deg: np.ndarray, c_mm: np.ndarray, m_mm: np.ndarray, delta_c: float):
        th = np.deg2rad(theta_deg.astype(float))
        m = m_mm.astype(float)
        x0, A, B = self._lsq_fit_cos_sin(th, m)
        e = float(math.hypot(A, B))
        phi = float(math.atan2(-B, A))  # m = x0 + e*cos(theta+phi)

        s = np.sin(th + phi)
        c_corr = np.clip(c_mm.astype(float) + float(delta_c), 0.001, None)
        Z = (0.5 * c_corr) ** 2 + (e * s) ** 2
        X2 = np.column_stack([np.ones_like(s), (-2.0 * e * s)])
        beta2, *_ = np.linalg.lstsq(X2, Z, rcond=None)
        R2p = float(beta2[0])
        y0 = float(beta2[1])
        R2 = float(R2p + y0 * y0)
        R = float(math.sqrt(max(R2, 0.0)))

        pred_R2 = (0.5 * c_corr) ** 2 + (y0 + e * s) ** 2
        rmse_R2 = float(math.sqrt(max(0.0, float(np.mean((pred_R2 - R2) ** 2)))))
        return {"R": R, "diam": 2.0 * R, "e": e, "phi_rad": phi, "x0": x0, "y0": y0, "rmse_R2": rmse_R2}

    def _idcal_compute(self) -> None:
        if not self._idcal_points:
            self.idcal_state_var.set("ERR")
            self.idcal_msg_var.set("无数据")
            return
        try:
            dref = float(self._parse_float(self.idcal_dref_var.get(), 150.0))
        except Exception:
            dref = 150.0

        pts = [p for p in self._idcal_points if (p.get("theta_deg") is not None and math.isfinite(float(p.get("theta_deg"))) and p.get("c_mm") is not None and p.get("m_mm") is not None)]
        if len(pts) < 20:
            cs = [p["c_mm"] for p in self._idcal_points if p.get("c_mm") is not None]
            if not cs:
                self.idcal_state_var.set("ERR")
                self.idcal_msg_var.set("无有效OUT4")
                return
            cmax = float(max(cs))
            delta = float(dref - cmax)
            self._idcal_delta_candidate = delta
            self.idcal_delta_candidate_var.set(f"{delta:.4f}")
            self.idcal_state_var.set("READY")
            self.idcal_msg_var.set("样本不足，采用 c_max 标定")
            return

        theta = np.array([p["theta_deg"] for p in pts], dtype=float)
        c = np.array([p["c_mm"] for p in pts], dtype=float)
        m = np.array([p["m_mm"] for p in pts], dtype=float)

        cmax = float(np.max(c))
        self.idcal_cmax_var.set(f"{cmax:.3f}")
        self.idcal_mmean_var.set(f"{float(np.mean(m)):.4f}")
        self.idcal_mpp_var.set(f"{float(np.max(m)-np.min(m)):.4f}")

        delta0 = float(dref - cmax)

        def f(delta):
            try:
                r = self._idcal_fit_diameter(theta, c, m, delta)
                return float(r["diam"] - dref), r
            except Exception:
                return float("nan"), None

        lo, hi = delta0 - 5.0, delta0 + 5.0
        flo, _ = f(lo)
        fhi, _ = f(hi)
        expand = 0
        while (not math.isfinite(flo) or not math.isfinite(fhi) or (flo * fhi > 0.0)) and expand < 6:
            lo -= 5.0
            hi += 5.0
            flo, _ = f(lo)
            fhi, _ = f(hi)
            expand += 1

        best_delta = delta0
        best = None
        if math.isfinite(flo) and math.isfinite(fhi) and (flo * fhi <= 0.0):
            a, b = lo, hi
            fa, fb = flo, fhi
            ra = None
            rb = None
            for _ in range(28):
                mid = 0.5 * (a + b)
                fm, rm = f(mid)
                if (not math.isfinite(fm)) or (rm is None):
                    break
                if fa * fm <= 0.0:
                    b, fb, rb = mid, fm, rm
                else:
                    a, fa, ra = mid, fm, rm
            best_delta = 0.5 * (a + b)
            _, best = f(best_delta)
        else:
            _, best = f(best_delta)

        if best is None:
            best_delta = delta0
            self.idcal_msg_var.set("拟合失败，退回 c_max 标定")
        self._idcal_delta_candidate = float(best_delta)
        self.idcal_delta_candidate_var.set(f"{float(best_delta):.4f}")
        if best is not None:
            self.idcal_fit_diam_var.set(f"{float(best['diam']):.3f}")
            self.idcal_fit_e_var.set(f"{float(best['e']):.4f}")
            self.idcal_fit_y0_var.set(f"{float(best['y0']):.4f}")
            self.idcal_fit_rmse_var.set(f"{float(best['rmse_R2']):.6f}")
            self.idcal_msg_var.set("计算完成（拟合+δc）")
        self.idcal_state_var.set("READY")

    def _idcal_apply(self) -> None:
        if self._idcal_delta_candidate is None:
            self.idcal_state_var.set("ERR")
            self.idcal_msg_var.set("请先计算")
            return
        try:
            dref = float(self._parse_float(self.idcal_dref_var.get(), 150.0))
        except Exception:
            dref = 150.0
        data = {"delta_c_mm": float(self._idcal_delta_candidate), "D_ref": float(dref), "ts": time.time()}
        self._idcal_save_active(data)
        self.idcal_delta_active_var.set(f"{float(self._idcal_delta_candidate):.4f}")
        self.idcal_state_var.set("APPLIED")
        self.idcal_msg_var.set("已应用并保存")

    def _idcal_export_raw(self) -> None:
        if not self._idcal_points:
            self.idcal_state_var.set("ERR")
            self.idcal_msg_var.set("无数据")
            return
        try:
            out_dir = self._app_root_dir() / "calibration"
            out_dir.mkdir(parents=True, exist_ok=True)
            p = out_dir / f"id_calib_raw_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            import csv
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ts", "theta_deg", "x1_mm", "x2_mm", "c_mm", "m_mm", "cnt_out4", "cnt_out5"])
                for it in self._idcal_points:
                    cnt = it.get("cnt") or {}
                    w.writerow([it.get("ts"), it.get("theta_deg"), it.get("x1_mm"), it.get("x2_mm"), it.get("c_mm"), it.get("m_mm"), cnt.get("out4"), cnt.get("out5")])
            self.idcal_msg_var.set(f"已导出: {p.name}")
        except Exception as e:
            self.idcal_state_var.set("ERR")
            self.idcal_msg_var.set(f"导出失败: {e}")


    def _idcal_verify(self) -> None:
        """复核：用“已应用”的 δc_active 采一圈数据，计算 D_fit 与 D_ref 的偏差。
        - 不修改 δc_candidate / δc_active
        - 自动旋转一圈（同“one_rev”）
        """
        if self._idcal_capturing:
            return

        # Get active delta (prefer file, fallback to UI var)
        delta = None
        dref = None
        try:
            p = self._idcal_file()
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                if data.get("delta_c_mm", None) is not None:
                    delta = float(data["delta_c_mm"])
                if data.get("D_ref", None) is not None:
                    dref = float(data["D_ref"])
        except Exception:
            pass
        if delta is None:
            try:
                delta = float(self.idcal_delta_active_var.get())
            except Exception:
                delta = None
        if dref is None:
            try:
                dref = float(self._parse_float(self.idcal_dref_var.get(), 150.0))
            except Exception:
                dref = 150.0

        if delta is None or (not math.isfinite(float(delta))):
            self.idcal_state_var.set("ERR")
            self.idcal_msg_var.set("复核失败：未找到 δc_active（请先“应用”）")
            return

        self._idcal_verify_pending = True
        self._idcal_verify_delta = float(delta)
        self._idcal_verify_dref = float(dref)

        # clear check display
        self.idcal_chk_err_var.set("--")
        self.idcal_chk_cov_var.set("--")
        self.idcal_chk_n_var.set("--")
        self.idcal_chk_dtheta_var.set("--")

        # Force one_rev for verify run (do not change UI selection)
        try:
            self._idcal_force_one_rev = True
        except Exception:
            pass

        self.idcal_state_var.set("CHK")
        self.idcal_msg_var.set("复核采集中...")
        self._idcal_start_capture()

    def _idcal_verify_compute(self) -> None:
        """Compute verify metrics based on the last captured points."""
        delta = getattr(self, "_idcal_verify_delta", None)
        dref = getattr(self, "_idcal_verify_dref", None)
        if delta is None or dref is None:
            raise RuntimeError("verify参数缺失")

        pts = [p for p in self._idcal_points if (
            p.get("theta_deg") is not None and math.isfinite(float(p.get("theta_deg"))) and
            p.get("c_mm") is not None and p.get("m_mm") is not None
        )]
        if len(pts) < 30:
            self.idcal_state_var.set("ERR")
            self.idcal_msg_var.set(f"复核样本不足: N={len(pts)}")
            return

        theta = np.array([float(p["theta_deg"]) for p in pts], dtype=float)
        c = np.array([float(p["c_mm"]) for p in pts], dtype=float)
        m = np.array([float(p["m_mm"]) for p in pts], dtype=float)

        # theta coverage / step
        th_rad = np.deg2rad(theta)
        th_unw = np.unwrap(th_rad)
        th_deg_unw = np.rad2deg(th_unw)
        # span (abs to support reverse rotation)
        span = float(abs(th_deg_unw[-1] - th_deg_unw[0]))
        dth = np.abs(np.diff(th_deg_unw))
        dth_max = float(np.max(dth)) if len(dth) else 0.0
        cov_pct = 100.0 * min(1.0, span / 360.0)

        # fit using active delta
        r = self._idcal_fit_diameter(theta, c, m, float(delta))
        diam = float(r["diam"])
        err = float(diam - float(dref))

        self.idcal_chk_err_var.set(f"{err:+.4f}")
        self.idcal_chk_cov_var.set(f"{cov_pct:.2f}%")
        self.idcal_chk_n_var.set(str(len(pts)))
        self.idcal_chk_dtheta_var.set(f"{dth_max:.3f}")

        # verdict (very light): coverage + diameter error
        ok = (cov_pct >= 95.0) and (abs(err) <= 0.020)
        self.idcal_state_var.set("CHK_OK" if ok else "CHK_NG")
        self.idcal_msg_var.set(f"复核{'OK' if ok else 'NG'}: ΔD={err:+.4f}mm  N={len(pts)}  cover={cov_pct:.2f}%")



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

        # update main-screen run info
        try:
            self.meas_seq_var.set(str(serial).split('-')[-1])
        except Exception:
            pass
        try:
            import datetime as _dt
            self.meas_start_var.set(_dt.datetime.fromtimestamp(float(self._run_start_ts)).strftime('%H:%M:%S'))
            self.meas_elapsed_var.set('00:00:00')
        except Exception:
            pass
        self._reset_summary_extrema()
        self._last_straight_od = None
        self._last_straight_id = None
        self._last_axis_dist = None
        self._run_summary = {}

        # auto length result cache (per-run)
        self._run_len_result = None


    def _ensure_run_identity(self) -> None:
        """Ensure run_serial/run_id/run_start_ts exist before export.

        Some UI events (e.g. AutoFlow 'auto_clear') should only clear result tables; however,
        to make the system robust, exporting will best-effort allocate missing identity fields.
        """
        # start_ts
        try:
            if not getattr(self, "_run_start_ts", None):
                self._run_start_ts = float(time.time())
        except Exception:
            self._run_start_ts = float(time.time())

        # run_id
        try:
            if not getattr(self, "_run_id", None):
                self._run_id = str(uuid.uuid4())
        except Exception:
            self._run_id = str(uuid.uuid4())

        # serial
        try:
            if not getattr(self, "_run_serial", None):
                try:
                    recipe_name = str(getattr(self.recipe, "name", "默认配方") or "默认配方")
                except Exception:
                    recipe_name = "默认配方"
                self._run_serial = self._next_serial(recipe_name)
                try:
                    self.pipe_sn_var.set(self._run_serial)
                except Exception:
                    pass
                try:
                    self.meas_seq_var.set(str(self._run_serial).split('-')[-1])
                except Exception:
                    pass
        except Exception:
            pass

        # main-screen start time (best effort)
        try:
            if getattr(self, "_run_start_ts", None) and hasattr(self, "meas_start_var"):
                import datetime as _dt
                self.meas_start_var.set(_dt.datetime.fromtimestamp(float(self._run_start_ts)).strftime('%H:%M:%S'))
        except Exception:
            pass

    def _export_current_run(self) -> tuple[bool, str]:
        """Export current run to exports directory. Returns (ok, msg)."""
        # Allow export as long as we have a DONE run (or at least computed rows). Ensure identity fields exist.
        try:
            self._ensure_run_identity()
        except Exception:
            pass
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
                    "od_avg_mm", "od_dev_mm", "od_runout_mm", "od_round_mm", "od_e_mm", "od_phi_deg",
                    "id_avg_mm", "id_dev_mm", "id_runout_mm", "id_round_mm", "id_e_mm", "id_phi_deg",
                    "concentricity_mm", "od_ecc_mm", "id_ecc_mm",
                    "cov_pct", "miss_bin", "max_gap_deg", "revs", "cov_elapsed_s", "cov_reason",
                    "raw",
                ])
                for r in rows:
                    cov_info = self._section_cov_info.get(int(getattr(r, "idx", 0) or 0), {})
                    cov_cols = self._format_cov_cols(cov_info)
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
                        "" if getattr(r, "od_e", None) is None else float(getattr(r, "od_e", 0.0)),
                        "" if getattr(r, "od_phi_deg", None) is None else float(getattr(r, "od_phi_deg", 0.0)),
                        float(getattr(r, "id_avg", 0.0)),
                        float(getattr(r, "id_dev", 0.0)),
                        float(getattr(r, "id_runout", 0.0)),
                        float(getattr(r, "id_round", 0.0)),
                        "" if getattr(r, "id_e", None) is None else float(getattr(r, "id_e", 0.0)),
                        "" if getattr(r, "id_phi_deg", None) is None else float(getattr(r, "id_phi_deg", 0.0)),
                        float(getattr(r, "concentricity", 0.0)),
                        "" if getattr(r, "od_ecc", None) is None else float(getattr(r, "od_ecc", 0.0)),
                        "" if getattr(r, "id_ecc", None) is None else float(getattr(r, "id_ecc", 0.0)),
                        *cov_cols,
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

            # Daily summary CSV (append/upsert one row per run)
            try:
                self._export_daily_summary_csv(day_dir=day_dir, start_ts=start_ts, end_ts=end_ts)
            except Exception:
                pass

            return True, f"已导出：{run_dir}"
        except Exception as e:
            return False, f"导出失败：{e}"

    def _export_daily_summary_csv(
        self,
        day_dir: Optional[Path] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        status: str = "DONE",
    ) -> None:
        """Write/Update a single summary row into exports/<day>/summary.csv.

        Notes:
            - One run_id corresponds to one row (upsert).
            - Called after DONE export, and may be called again when late postcalc arrives.
        """
        try:
            self._ensure_run_identity()
        except Exception:
            pass
        if not self._run_serial or not self._run_id:
            return

        try:
            _start = float(start_ts if start_ts is not None else self._run_start_ts)
        except Exception:
            return
        try:
            _end = float(end_ts if end_ts is not None else (self._run_end_ts or time.time()))
        except Exception:
            _end = float(time.time())

        if day_dir is None:
            try:
                day_dir = self._exports_root_dir() / datetime.date.fromtimestamp(_start).strftime("%Y-%m-%d")
                day_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                return

        summary_path = Path(day_dir) / "summary.csv"

        # recipe snapshot for key nominal values
        try:
            rcp = self.get_recipe_copy()
            recipe_name = str(getattr(rcp, "name", "") or "")
            od_std = getattr(rcp, "od_std_mm", None)
            id_std = getattr(rcp, "id_std_mm", None)
            od_tol = getattr(rcp, "od_tol_mm", None)
        except Exception:
            recipe_name = ""
            od_std, id_std, od_tol = None, None, None

        # run-level computed summary (maxima + straightness/axis_dist cache)
        try:
            s = self._calc_run_summary()
        except Exception as e:
            s = {"ok": False, "reason": f"异常: {e}"}

        def _num(x, fmt: str = "{:.3f}") -> str:
            try:
                if x is None:
                    return ""
                return fmt.format(float(x))
            except Exception:
                return ""

        header = [
            "date",
            "start_time",
            "end_time",
            "duration_s",
            "serial",
            "run_id",
            "recipe_name",
            "device_code",
            "od_std_mm",
            "id_std_mm",
            "od_tol_mm",
            "len_enabled",
            "len_skipped",
            "len_ok",
            "len_mm",
            "len_z_low",
            "len_z_high",
            "len_reason",
            "len_t_s",
            "straight_od_mm",
            "straight_id_mm",
            "axis_dist_mm",
            "conc_max_mm",
            "axis_span_max_mm",
            "od_tilt_deg",
            "od_end_off_mm",
            "od_slope_mm_per_mm",
            "id_tilt_deg",
            "id_end_off_mm",
            "id_slope_mm_per_mm",
            "max_od_dev_abs_mm",
            "max_id_dev_abs_mm",
            "max_od_round_mm",
            "max_id_round_mm",
            "od_mean_mm",
            "od_d_pp_mm",
            "od_e_mm",
            "id_mean_mm",
            "id_d_pp_mm",
            "summary_ok",
            "summary_reason",
            "status",
            "software_version",
        ]

        row = [
            datetime.date.fromtimestamp(_start).strftime("%Y-%m-%d"),
            datetime.datetime.fromtimestamp(_start).strftime("%H:%M:%S"),
            datetime.datetime.fromtimestamp(_end).strftime("%H:%M:%S"),
            f"{max(0.0, _end - _start):.3f}",
            str(self._run_serial),
            str(self._run_id),
            recipe_name,
            str(self._get_device_code() or ""),
            _num(od_std),
            _num(id_std),
            _num(od_tol),
            ('1' if bool((getattr(self, '_run_len_result', None) or {}).get('enabled', False)) else '0') if isinstance(getattr(self,'_run_len_result',None), dict) else '',
            ('1' if bool((getattr(self, '_run_len_result', None) or {}).get('skipped', False)) else '0') if isinstance(getattr(self,'_run_len_result',None), dict) else '',
            ('1' if bool((getattr(self, '_run_len_result', None) or {}).get('ok', False)) else '0') if isinstance(getattr(self,'_run_len_result',None), dict) else '',
            _num((getattr(self, '_run_len_result', None) or {}).get('length_mm', None)) if isinstance(getattr(self,'_run_len_result',None), dict) else '',
            _num((getattr(self, '_run_len_result', None) or {}).get('z_low', None)) if isinstance(getattr(self,'_run_len_result',None), dict) else '',
            _num((getattr(self, '_run_len_result', None) or {}).get('z_high', None)) if isinstance(getattr(self,'_run_len_result',None), dict) else '',
            str((getattr(self, '_run_len_result', None) or {}).get('reason', '') or '') if isinstance(getattr(self,'_run_len_result',None), dict) else '',
            _num((getattr(self, '_run_len_result', None) or {}).get('t_s', None), fmt='{:.3f}') if isinstance(getattr(self,'_run_len_result',None), dict) else '',
            _num(s.get("straight_od")),
            _num(s.get("straight_id")),
            _num(s.get("axis_dist")),
            _num(s.get("conc_max")),
            _num(s.get("axis_span_max")),
            _num(s.get("od_tilt_deg"), fmt='{:.4f}'),
            _num(s.get("od_end_off_mm")),
            _num(s.get("od_slope"), fmt='{:.6f}'),
            _num(s.get("id_tilt_deg"), fmt='{:.4f}'),
            _num(s.get("id_end_off_mm")),
            _num(s.get("id_slope"), fmt='{:.6f}'),
            _num(s.get("max_od_dev_abs")),
            _num(s.get("max_id_dev_abs")),
            _num(s.get("max_od_round")),
            _num(s.get("max_id_round")),
            _num(s.get("od_mean")),
            _num(s.get("od_d_pp")),
            _num(s.get("od_e")),
            _num(s.get("id_mean")),
            _num(s.get("id_d_pp")),
            "1" if bool(s.get("ok", False)) else "0",
            str(s.get("reason", "") or ""),
            str(status or ""),
            str(SOFTWARE_VERSION),
        ]

        # Upsert by run_id
        try:
            existing_rows: list[list[str]] = []
            old_header: list[str] = []
            if summary_path.exists():
                with open(summary_path, "r", newline="", encoding="utf-8-sig") as f:
                    r = csv.reader(f)
                    existing_rows = [list(x) for x in r]
            if existing_rows:
                old_header = list(existing_rows[0])

            # Convert existing rows to current header if needed (do NOT drop history)
            converted_rows: list[list[str]] = []
            if existing_rows and old_header and (old_header != header):
                old_map = {c: i for i, c in enumerate(old_header)}

                def _convert_one(rr: list[str]) -> list[str]:
                    out = ["" for _ in range(len(header))]
                    for j, col in enumerate(header):
                        i0 = old_map.get(col, None)
                        if i0 is None:
                            continue
                        try:
                            if i0 < len(rr):
                                out[j] = rr[i0]
                        except Exception:
                            pass
                    return out

                for rr in existing_rows[1:]:
                    try:
                        converted_rows.append(_convert_one(list(rr)))
                    except Exception:
                        # keep a blank row on conversion failure
                        converted_rows.append(["" for _ in range(len(header))])
            elif existing_rows and old_header and (old_header == header):
                converted_rows = [list(rr) for rr in existing_rows[1:]]

            # Compose output (upsert by run_id)
            out_rows: list[list[str]] = [header]
            run_id_col = None
            try:
                run_id_col = header.index("run_id")
            except Exception:
                run_id_col = 5

            replaced = False
            for rr in converted_rows:
                try:
                    if len(rr) > run_id_col and str(rr[run_id_col]) == str(self._run_id):
                        out_rows.append(row)
                        replaced = True
                    else:
                        out_rows.append(rr)
                except Exception:
                    out_rows.append(rr)

            if not existing_rows:
                out_rows = [header, row]
            elif not replaced:
                out_rows.append(row)


            tmp = summary_path.with_suffix(".tmp")
            with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerows(out_rows)
            tmp.replace(summary_path)
        except Exception:
            # Best-effort: if upsert fails, try append
            try:
                new_file = not summary_path.exists()
                with open(summary_path, "a", newline="", encoding="utf-8-sig") as f:
                    w = csv.writer(f)
                    if new_file:
                        w.writerow(header)
                    w.writerow(row)
            except Exception:
                pass

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

    def _cov_reason_text(self, reason: str) -> str:
        """Human readable text for coverage stop reason."""
        r = str(reason or "").strip()
        if not r:
            return ""
        mapping = {
            "COV": "覆盖率达标",
            "TIMEOUT": "超时退出",
            "REV": "圈数到达",
        }
        return mapping.get(r.upper(), r)

    def _format_cov_cols(self, info: dict) -> tuple[str, str, str, str, str, str]:
        """Format per-section coverage stats for table/export columns."""
        cov = info.get("cov", None)
        if cov is None:
            return ("--", "--", "--", "--", "--", "")

        try:
            cov_pct = f"{float(cov) * 100:.1f}"
        except Exception:
            cov_pct = "--"

        miss = info.get("miss", None)
        try:
            miss_bin = "--" if miss is None else str(int(miss))
        except Exception:
            miss_bin = "--"

        max_gap = info.get("max_gap_deg", None)
        try:
            max_gap_deg = "--" if max_gap is None else f"{float(max_gap):.1f}"
        except Exception:
            max_gap_deg = "--"

        revs = info.get("revs", None)
        try:
            revs_txt = "--" if revs is None else f"{float(revs):.2f}"
        except Exception:
            revs_txt = "--"

        elapsed = info.get("elapsed", None)
        try:
            elapsed_s = "--" if elapsed is None else f"{float(elapsed):.2f}"
        except Exception:
            elapsed_s = "--"

        reason_txt = self._cov_reason_text(info.get("reason", ""))
        return (cov_pct, miss_bin, max_gap_deg, revs_txt, elapsed_s, reason_txt)

    def _update_result_row_cov(self, sec_idx: int, info: dict) -> None:
        """Update cov columns in the results table for an existing section row."""
        try:
            iid = self._sec_iid_map.get(int(sec_idx))
        except Exception:
            iid = None
        if not iid:
            return
        try:
            vals = list(self.result_tree.item(iid, "values") or [])
        except Exception:
            return

        # base measurement columns count (keep in sync with table definition)
        base_n = 11
        if len(vals) < base_n:
            return
        cov_cols = list(self._format_cov_cols(info))
        new_vals = tuple(vals[:base_n] + cov_cols)
        try:
            self.result_tree.item(iid, values=new_vals)
        except Exception:
            pass

    def _set_straight_label(self, straight_od, straight_id, axis_dist, conc_max=None, axis_span_max=None) -> None:
        """Update straightness/concentricity labels.

        Notes:
            - axis_dist: overall OD/ID axis distance (方案3: overall)
            - conc_max: max per-section concentricity (方案3)
            - axis_span_max: max distance between OD axis and ID axis over span (方案3)
        """
        if straight_od is None and straight_id is None and axis_dist is None and conc_max is None and axis_span_max is None:
            self.straight_var.set("直线度   --（外圆） | --（内圆）")
            try:
                self.conc_var.set("整体同心度   --")
            except Exception:
                pass

            # split vars
            try:
                self.straight_od_var.set("--")
                self.straight_id_var.set("--")
                self.axis_dist_var.set("--")
                self.conc_max_var.set("--")
                self.axis_span_max_var.set("--")
            except Exception:
                pass
            return

        od_txt = "--" if straight_od is None else f"{float(straight_od):.3f}"
        id_txt = "--" if straight_id is None else f"{float(straight_id):.3f}"
        ax_txt = "--" if axis_dist is None else f"{float(axis_dist):.3f}"
        cmax_txt = "--" if conc_max is None else f"{float(conc_max):.3f}"
        span_txt = "--" if axis_span_max is None else f"{float(axis_span_max):.3f}"
        self.straight_var.set(f"直线度   {od_txt}（外圆） | {id_txt}（内圆）")
        try:
            # keep legacy one-line text, but include scheme-3 extras when available
            if conc_max is None and axis_span_max is None:
                self.conc_var.set(f"整体同心度   {ax_txt}")
            else:
                self.conc_var.set(f"整体同心度   {ax_txt} | 截面同心度max {cmax_txt} | 轴线间距max {span_txt}")
        except Exception:
            pass

        # split vars
        try:
            self.straight_od_var.set("--" if straight_od is None else f"{float(straight_od):.3f} mm")
            self.straight_id_var.set("--" if straight_id is None else f"{float(straight_id):.3f} mm")
            self.axis_dist_var.set("--" if axis_dist is None else f"{float(axis_dist):.3f} mm")
            self.conc_max_var.set("--" if conc_max is None else f"{float(conc_max):.3f} mm")
            self.axis_span_max_var.set("--" if axis_span_max is None else f"{float(axis_span_max):.3f} mm")
        except Exception:
            pass

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
