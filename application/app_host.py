# ./application/app_host.py
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
import inspect
import logging

from utils.logger import init_log, log, log_exc
from utils.perf import PerfAggregator, ns_to_ms
from typing import Any, List, Mapping, Optional, Tuple, Iterable

import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkfont

from application.recipe_form_mapper import RecipeFormMapper
from application.results_service import ResultsService
from application.shell import AppDependencies, ApplicationShell
from application.state import (
    CalibrationSnapshot,
    FIXED_SECTION_PRIMARY_METRICS,
    RunContext,
    RunIdentity,
    RunSession,
    RuntimeState,
    VALIDATION_MOVE_CHANNELS,
    VALIDATION_MOVE_SCENARIOS,
    ValidationSession,
)
from application.ui_event_dispatcher import UiEventDispatcher
from application.ui_events import (
    AutoClearEvent,
    AutoCoverageEvent,
    AutoLenEvent,
    AutoPostcalcEvent,
    AutoProgressEvent,
    AutoRawPointsEvent,
    AutoRowEvent,
    AutoStateEvent,
    AutoStraightnessEvent,
    GaugeConnEvent,
    GaugeErrEvent,
    GaugeOkEvent,
    GaugeRawEvent,
    GaugeTxEvent,
    OpConfirmCloseEvent,
    OpConfirmShowEvent,
    PlcErrEvent,
    PlcGiveupEvent,
    PlcManualEvent,
    PlcOkEvent,
    PlcReadEvent,
)
from repositories.calibration_repository import CalibrationRepository
from repositories.validation_repository import ValidationRepository
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
from domain.planning import (
    build_recipe_section_plan,
    format_recipe_section_name,
    plan_section_positions,
)
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
from application.app_adapters import (
    AppDeviceGateway,
    ScreenController,
    ScreenPresenter,
    ScreenUiContext,
)
from application.ui_queue_adapters import WorkflowUiEventAdapter
from application.axis_presenter import AxisScreenPresenter
from application.calibration_controller import CalibrationController
from application.gauge_presenter import GaugeScreenPresenter
from application.calibration_service import CalibrationService
from application.contracts import ValidationActionCancelled
from application.measurement_controller import MeasurementController
from application.recipe_presenter import RecipeScreenPresenter
from modes.calibration_mode import CalibrationMode
from modes.mode_machine import ModeMachine
from modes.production_mode import ProductionMode
from modes.validation_mode import ValidationMode
from repositories.run_repository import RunRepository
from frp_workflow.autoflow_orchestrator import AutoFlowOrchestrator
from frp_workflow.validation_workflow import (
    FixedSectionRepeatabilityRequest,
    ValidationWorkflow,
)

from ui.screens.axis_screen import build_axis_screen
from ui.screens.axis_cal_screen import build_axis_cal_screen
from ui.screens.recipe_screen import build_recipe_screen
from ui.screens.gauge_screen import build_gauge_screen
from ui.screens.validation_screen import build_validation_screen
from ui.screens.main_screen import build_main_screen
from ui.screens.key_test_screen import build_key_test_screen

logger = logging.getLogger("frp.app")
recipe_logger = logging.getLogger("frp.recipe")
modbus_logger = logging.getLogger("frp.modbus")
ax3_trace_logger = logging.getLogger("frp.autoflow")
plc_perf_logger = logging.getLogger("frp.modbus.perf")


SOFTWARE_VERSION = "ipc_nn_f60"
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


class AppHost(tk.Tk):
    _shell: ApplicationShell | None
    _dependencies: AppDependencies

    ui_q: queue.Queue[Any]
    cmd_q: queue.Queue[Any]
    worker: PlcWorker
    gauge_worker: GaugeWorker | None
    calibration_repository: CalibrationRepository
    recipe_store: Any

    results_service: ResultsService
    calibration_service: CalibrationService
    calibration_mode: CalibrationMode
    validation_mode: ValidationMode
    production_mode: ProductionMode
    mode_machine: ModeMachine
    calibration_controller: CalibrationController
    measurement_controller: MeasurementController
    _screen_controller: ScreenController
    _screen_presenter: ScreenPresenter
    _recipe_screen_presenter: RecipeScreenPresenter
    _axis_screen_presenter: AxisScreenPresenter
    _gauge_screen_presenter: GaugeScreenPresenter
    _screen_ui_context: ScreenUiContext

    axis_idx: tk.IntVar
    plc_status_var: tk.StringVar
    err_banner_var: tk.StringVar
    ip_var: tk.StringVar
    port_var: tk.StringVar
    gauge_conn_var: tk.StringVar
    gauge_last_var: tk.StringVar
    gauge_err_var: tk.StringVar

    recipe_name_var: tk.StringVar
    center_pos_var: tk.StringVar
    len_enable_var: tk.BooleanVar
    len_z_low_approach_var: tk.StringVar
    len_info_var: tk.StringVar
    len_status_var: tk.StringVar
    len_edge_state_var: tk.StringVar
    len_edge_low_var: tk.StringVar
    len_edge_high_var: tk.StringVar
    len_edge_len_var: tk.StringVar
    teach_axes_mode_var: tk.IntVar
    teach_rel_dist_var: tk.StringVar
    teach_abs_var: tk.StringVar
    teach_z_var: tk.StringVar
    teach_align_var: tk.StringVar
    teach_mode_var: tk.StringVar
    teach_axes_var: tk.StringVar
    start_info_var: tk.StringVar
    standby_info_var: tk.StringVar
    standby_state_var: tk.StringVar

    zero_abs_var: tk.StringVar
    sign_var: tk.StringVar

    sim_gauge_var: tk.IntVar
    sim_disp_var: tk.IntVar
    baud_var: tk.StringVar
    req_cmd_var: tk.StringVar

    validation_status_var: tk.StringVar
    validation_phase_var: tk.StringVar
    validation_wait_phase_var: tk.StringVar
    validation_wait_remaining_s_var: tk.StringVar
    validation_current_repeat_var: tk.StringVar
    validation_result_var: tk.StringVar
    validation_error_var: tk.StringVar
    validation_export_path_var: tk.StringVar
    validation_move_target_pos_var: tk.StringVar
    validation_move_actual_pos_var: tk.StringVar
    validation_current_metric_value_var: tk.StringVar
    validation_current_section_var: tk.StringVar
    validation_current_z_pos_var: tk.StringVar
    validation_current_concentricity_var: tk.StringVar
    validation_summary_count_var: tk.StringVar
    validation_summary_mean_var: tk.StringVar
    validation_summary_std_var: tk.StringVar
    validation_summary_min_var: tk.StringVar
    validation_summary_max_var: tk.StringVar
    validation_summary_range_var: tk.StringVar

    def __init__(
        self,
        dependencies: AppDependencies | None = None,
        shell: ApplicationShell | None = None,
    ):
        super().__init__()
        self._shell = shell
        if dependencies is None:
            if self._shell is None:
                self._shell = ApplicationShell()
            dependencies = self._shell.assemble_dependencies()
        self._dependencies = dependencies
        try:
            init_log(log_dir=str(self._app_root_dir() / "logs"), overwrite=False)
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

        self.ui_q = dependencies.ui_q
        self.cmd_q = dependencies.cmd_q
        self.worker = dependencies.worker
        self.gauge_worker = dependencies.gauge_worker
        self.calibration_repository = dependencies.calibration_repository

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
        self._perf_sync_read = PerfAggregator()
        self._perf_ui_queue = PerfAggregator()

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
        self._validation_thread: Optional[threading.Thread] = None
        self._validation_running: bool = False
        self._validation_cancel_event = threading.Event()
        self._validation_cancel_requested: bool = False


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
        self.recipe_store = dependencies.recipe_store

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

        # 凹陷/缺陷屏蔽（外径标定专用）
        # - TEMPLATE：使用“凹陷表(模板)”并对齐本次采样的相位后屏蔽角度段
        # - DYNAMIC：未学习模板时，按本次残差自动屏蔽最深的一段（可关闭）
        self.odcal_defect_mode_var = tk.StringVar(value="OFF")  # OFF | DYNAMIC | TEMPLATE
        self.odcal_defect_shift_var = tk.StringVar(value="--")  # 本次对齐 shift (deg)
        self.odcal_defects_var = tk.StringVar(value="--")       # 凹陷段(模板坐标，显示用)
        self.odcal_defect_dyn_enable_var = tk.IntVar(value=1)   # 未学习模板时，是否允许动态屏蔽
        self._odcal_defect_template_mask: list[int] = [0] * 360  # 0/1, template coordinate
        # NOTE: do NOT use the same name as a method (Tk Button command binding will
        # grab the instance attribute first, which would mask the method).
        self._odcal_defect_learn_A_data: Optional[dict] = None

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

        # ------------------------------
        # ID Single-probe Calibration (OUT2/L2)
        # ------------------------------
        self.id_single_cal_state_var = tk.StringVar(value="IDLE")
        self.id_single_cal_msg_var = tk.StringVar(value="-")
        self.id_single_cal_dref_var = tk.StringVar(value="150.000")
        self.id_single_cal_mean_var = tk.StringVar(value="--")
        self.id_single_cal_B_var = tk.StringVar(value="--")
        self.id_single_cal_ecc_amp_var = tk.StringVar(value="--")
        self.id_single_cal_ecc_ang_var = tk.StringVar(value="--")
        self.id_single_cal_cov_var = tk.StringVar(value="--")
        self.id_single_cal_warn_var = tk.StringVar(value="")

        # capture buffer/state
        self._id_single_cal_capturing: bool = False
        self._id_single_cal_points: list[dict] = []
        self._id_single_cal_after_id: Optional[str] = None
        self._id_single_cal_start_ts: Optional[float] = None
        self._id_single_cal_stop_reason: str = ""
        self._id_single_cal_prev_poll_profile: Optional[str] = None
        self._id_single_cal_last_out2_cnt: Optional[int] = None
        self._id_single_cal_theta_start: Optional[float] = None
        self._id_single_cal_theta_last: Optional[float] = None
        self._id_single_cal_theta_unwrap: float = 0.0
        self._id_single_cal_rev_progress_deg: float = 0.0
        self._id_single_cal_rev_target_deg: float = 360.0
        self._id_single_cal_one_rev_timeout_ts: Optional[float] = None
        self._id_single_cal_ax3_rotating: bool = False
        self._cal_id_single_last: Optional[dict] = None



        # Auto
        self._auto_thread: Optional[AutoFlowOrchestrator] = None
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
        self.ui_meas_mode_var = tk.StringVar(value="检测模式：--")  # 主界面显示：SYNC/SPLIT/OD_ONLY

        # Summary (main screen)
        self.max_od_dev_var = tk.StringVar(value="--")
        self.max_id_dev_var = tk.StringVar(value="--")
        self.max_od_round_var = tk.StringVar(value="--")
        self.max_id_round_var = tk.StringVar(value="--")
        # f2 main-screen OD diagnostics
        # - max_od_pp_var: strict peak-to-peak of OD diameter series (displayed as "外径峰峰")
        # - max_od_pp_rob_var: robust peak-to-peak/span
        # - max_od_fit_res_var: circle-fit residual span (robust)
        self.max_od_pp_var = tk.StringVar(value="--")
        self.max_od_pp_rob_var = tk.StringVar(value="--")
        self.max_od_fit_res_var = tk.StringVar(value="--")
        self.od_mean_var = tk.StringVar(value="--")
        self.od_dpp_var = tk.StringVar(value="--")
        self.od_e_var = tk.StringVar(value="--")
        self.id_mean_var = tk.StringVar(value="--")
        self.id_dpp_var = tk.StringVar(value="--")
        # New summary fields (main screen)
        self.od_range_var = tk.StringVar(value="--")  # 外径极差
        self.id_range_var = tk.StringVar(value="--")  # 内径极差
        self.od_slope_var = tk.StringVar(value="--")  # 外圆轴线斜率 (mm/m)
        self.id_slope_var = tk.StringVar(value="--")  # 内圆轴线斜率 (mm/m)
        # Optional: length measurement summary (main screen)
        self.len_meas_var = tk.StringVar(value="--")

        self._max_od_dev = None
        self._max_id_dev = None
        self._max_od_round = None
        self._max_id_round = None
        self._run_session = RunSession()
        self.validation_session = ValidationSession()
        self.runtime_state = RuntimeState.from_run_session(self._run_session)
        self._auto_export_done: bool = False
        self._last_run_export_path: Optional[str] = None
        self._validation_cancel_event = threading.Event()
        self._validation_cancel_requested: bool = False

        # Summary extrema caches (computed from per-section results)
        self._max_od_dev_abs: Optional[float] = None
        self._max_id_dev_abs: Optional[float] = None
        self._max_od_round: Optional[float] = None
        self._max_id_round: Optional[float] = None
        self._max_od_pp: Optional[float] = None
        self._max_od_pp_rob: Optional[float] = None
        self._max_od_fit_res: Optional[float] = None

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

        # Auto length result produced by AutoFlow (optional)
        self._run_len_result: Optional[dict] = None

        self._device_ui_event_dispatcher = self._build_device_ui_event_dispatcher()
        self._measurement_ui_event_dispatcher = self._build_measurement_ui_event_dispatcher()
        self.results_service = ResultsService()
        self.calibration_service = CalibrationService()
        self.calibration_mode = CalibrationMode()
        self.validation_mode = ValidationMode(
            stop_impl=self.stop_validation_run,
            runner_getter=lambda: self._validation_thread,
        )
        self.production_mode = ProductionMode(
            start_impl=self._start_measurement_impl,
            stop_impl=self._stop_measurement_impl,
            runner_getter=lambda: self._auto_thread,
            already_running_handler=lambda: messagebox.showwarning("Measurement", "Measurement is already running"),
        )
        self.mode_machine = ModeMachine(
            production_mode=self.production_mode,
            calibration_mode=self.calibration_mode,
            validation_mode=self.validation_mode,
            runtime_state=self.runtime_state,
        )
        self.calibration_controller = CalibrationController(
            host=self,
            service=self.calibration_service,
            mode_machine=self.mode_machine,
        )
        self.measurement_controller = MeasurementController(
            mode_machine=self.mode_machine,
        )
        self._screen_controller = ScreenController(self)
        self._screen_presenter = ScreenPresenter(self)
        self._recipe_screen_presenter = RecipeScreenPresenter(self)
        self._axis_screen_presenter = AxisScreenPresenter(self, self._screen_controller)
        self._gauge_screen_presenter = GaugeScreenPresenter(self, self._screen_controller)
        self._screen_ui_context = ScreenUiContext(self)

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

    @property
    def _run_serial(self) -> Optional[str]:
        return self._run_session.serial

    @_run_serial.setter
    def _run_serial(self, value: Optional[str]) -> None:
        self._run_session.serial = value

    @property
    def _run_id(self) -> Optional[str]:
        return self._run_session.run_id

    @_run_id.setter
    def _run_id(self, value: Optional[str]) -> None:
        self._run_session.run_id = value

    @property
    def _run_start_ts(self) -> Optional[float]:
        return self._run_session.start_ts

    @_run_start_ts.setter
    def _run_start_ts(self, value: Optional[float]) -> None:
        self._run_session.start_ts = value

    @property
    def _run_end_ts(self) -> Optional[float]:
        return self._run_session.end_ts

    @_run_end_ts.setter
    def _run_end_ts(self, value: Optional[float]) -> None:
        self._run_session.end_ts = value

    @property
    def _auto_rows(self) -> list[MeasureRow]:
        return self._run_session.rows

    @_auto_rows.setter
    def _auto_rows(self, value: list[MeasureRow]) -> None:
        self._run_session.rows = list(value or [])

    @property
    def _auto_raw_points(self) -> list[dict]:
        return self._run_session.raw_points

    @_auto_raw_points.setter
    def _auto_raw_points(self, value: list[dict]) -> None:
        self._run_session.raw_points = list(value or [])

    @property
    def _run_summary(self) -> dict:
        return self._run_session.summary_cache

    @_run_summary.setter
    def _run_summary(self, value: dict) -> None:
        self._run_session.summary_cache = dict(value or {})

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
                    else "115200"
                )
                or "115200"
            )
        except Exception:
            baud = 115200

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
                    baud=115200,
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
        shell = getattr(self, "_shell", None)
        if shell is not None:
            try:
                shell.close_app(self)
                return
            except Exception:
                pass
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
        tab_validation = ttk.Frame(nb)
        tab_gauge = ttk.Frame(nb)
        tab_keytest = ttk.Frame(nb)

        # Main operation tab first (left-most) and selected by default.
        nb.add(tab_main, text="主操作/自动测量")
        nb.add(tab_axis_cal, text="轴位标定")
        nb.add(tab_axis, text="轴参数/调试")
        nb.add(tab_recipe, text="配方/示教")
        nb.add(tab_gauge, text="外设通信")
        nb.add(tab_keytest, text="按键测试")

        build_main_screen(tab_main, presenter=self._screen_presenter, controller=self._screen_controller, ui=self._screen_ui_context)
        build_axis_cal_screen(tab_axis_cal, presenter=self._screen_presenter, controller=self._screen_controller, ui=self._screen_ui_context)
        build_axis_screen(tab_axis, presenter=self._axis_screen_presenter, controller=self._screen_controller, ui=self._screen_ui_context)
        build_recipe_screen(tab_recipe, presenter=self._recipe_screen_presenter, controller=self._screen_controller, ui=self._screen_ui_context)
        build_validation_screen(tab_validation, presenter=self._gauge_screen_presenter, controller=self._screen_controller, ui=self._screen_ui_context)
        build_gauge_screen(tab_gauge, presenter=self._gauge_screen_presenter, controller=self._screen_controller, ui=self._screen_ui_context)
        build_key_test_screen(tab_keytest, presenter=self._screen_presenter, controller=self._screen_controller, ui=self._screen_ui_context)
        nb.insert(tab_gauge, tab_validation, text="Validation")
        self._tab_validation = tab_validation

        try:
            nb.select(tab_main)
        except Exception:
            pass

        # init recipe store UI (dropdown, last recipe)
        try:
            self._recipe_store_init()
        except Exception:
            pass

    def apply_plc_connection(self):
        return self._apply_conn()

    def refresh_gauge_ports(self):
        return self._refresh_ports()

    def toggle_sim_gauge(self):
        return self._on_sim_gauge_toggle()

    def connect_gauge(self):
        return self._gauge_connect()

    def disconnect_gauge(self):
        return self._gauge_disconnect()

    def request_gauge_once(self):
        return self._gauge_request_once()

    def open_validation_screen(self) -> None:
        notebook = getattr(self, "_notebook", None)
        tab = getattr(self, "_tab_validation", None)
        if notebook is None or tab is None:
            return None
        try:
            notebook.select(tab)
        except Exception:
            return None
        return None

    def learn_odcal_defect_a(self):
        return self._odcal_defect_learn_A()

    def learn_odcal_defect_b(self):
        return self._odcal_defect_learn_B()

    def clear_odcal_defect_template(self):
        return self._odcal_defect_clear_template()

    def start_measurement(self):
        return self.measurement_controller.start_measurement()

    def stop_measurement(self):
        return self.measurement_controller.stop_measurement()

    def clear_measurement_results(self):
        return self._auto_clear_ui()

    def _set_validation_feedback(
        self,
        *,
        status: str = "",
        phase: str | None = None,
        wait_phase: str | None = None,
        wait_remaining_s: float | str | None = None,
        current_repeat: str | None = None,
        result: str = "",
        error: str = "",
        export_path: str = "",
        move_target_pos: object | None = None,
        move_actual_pos: object | None = None,
    ) -> None:
        try:
            self.validation_status_var.set(str(status or ""))
        except Exception:
            pass
        if phase is not None:
            self._set_validation_phase(phase)
        if wait_phase is not None:
            self._set_validation_wait_phase(wait_phase)
        if wait_remaining_s is not None:
            self._set_validation_wait_remaining(wait_remaining_s)
        if current_repeat is not None:
            self._set_validation_current_repeat(current_repeat)
        try:
            self.validation_result_var.set(str(result or ""))
        except Exception:
            pass
        try:
            self.validation_error_var.set(str(error or ""))
        except Exception:
            pass
        try:
            self.validation_export_path_var.set(str(export_path or ""))
        except Exception:
            pass
        if move_target_pos is not None or move_actual_pos is not None:
            self._set_validation_move_position(
                target_pos=move_target_pos,
                actual_pos=move_actual_pos,
            )

    def _set_validation_phase(self, phase: str) -> None:
        try:
            self.validation_phase_var.set(self._format_validation_phase(phase))
        except Exception:
            pass

    def _set_validation_move_position(
        self,
        *,
        target_pos: object | None = None,
        actual_pos: object | None = None,
    ) -> None:
        if target_pos is not None:
            try:
                self.validation_move_target_pos_var.set(
                    self._format_validation_position(target_pos)
                )
            except Exception:
                pass
        if actual_pos is not None:
            try:
                self.validation_move_actual_pos_var.set(
                    self._format_validation_position(actual_pos)
                )
            except Exception:
                pass

    def _set_validation_wait_phase(self, phase: str) -> None:
        try:
            self.validation_wait_phase_var.set(self._format_validation_phase(phase) if str(phase or "").strip() else "")
        except Exception:
            pass

    def _set_validation_wait_remaining(self, remaining_s: float | str) -> None:
        try:
            text = ""
            if remaining_s not in (None, ""):
                text = f"{float(remaining_s):.3f}s"
            self.validation_wait_remaining_s_var.set(text)
        except Exception:
            pass

    def _set_validation_current_repeat(self, current_repeat: str) -> None:
        try:
            self.validation_current_repeat_var.set(str(current_repeat or ""))
        except Exception:
            pass

    def _reset_validation_summary_panel(self) -> None:
        defaults = {
            "validation_current_metric_value_var": "",
            "validation_current_section_var": "",
            "validation_current_z_pos_var": "",
            "validation_current_concentricity_var": "",
            "validation_summary_count_var": "0",
            "validation_summary_mean_var": "",
            "validation_summary_std_var": "",
            "validation_summary_min_var": "",
            "validation_summary_max_var": "",
            "validation_summary_range_var": "",
        }
        for attr_name, value in defaults.items():
            try:
                getattr(self, attr_name).set(value)
            except Exception:
                pass

    def _set_validation_current_repeat_result(self, capture: object | None = None) -> None:
        if capture is None:
            return
        try:
            measured_value_mm = getattr(capture, "measured_value_mm", None)
            self.validation_current_metric_value_var.set(
                self._format_validation_numeric(measured_value_mm, digits=6)
            )
        except Exception:
            pass
        try:
            section_name = getattr(capture, "measure_section_name", "") or getattr(capture, "section_name", "")
            self.validation_current_section_var.set(str(section_name or ""))
        except Exception:
            pass
        try:
            measured_z_pos_mm = getattr(capture, "measured_z_pos_mm", None)
            self.validation_current_z_pos_var.set(
                self._format_validation_numeric(measured_z_pos_mm, digits=3)
            )
        except Exception:
            pass
        try:
            fit_result = getattr(capture, "fit_result", None)
            concentricity_mm = None if fit_result is None else getattr(fit_result, "concentricity_mm", None)
            self.validation_current_concentricity_var.set(
                self._format_validation_numeric(concentricity_mm, digits=6)
            )
        except Exception:
            pass

    def _set_validation_summary_values(self, summary: Mapping[str, Any] | None = None) -> None:
        payload = dict(summary or {})
        try:
            count_value = int(payload.get("count", 0) or 0)
        except Exception:
            count_value = 0
        try:
            self.validation_summary_count_var.set(str(count_value))
        except Exception:
            pass
        for attr_name, field_name in (
            ("validation_summary_mean_var", "mean"),
            ("validation_summary_std_var", "std"),
            ("validation_summary_min_var", "min"),
            ("validation_summary_max_var", "max"),
            ("validation_summary_range_var", "range"),
        ):
            try:
                getattr(self, attr_name).set(
                    self._format_validation_numeric(payload.get(field_name), digits=6)
                )
            except Exception:
                pass

    @staticmethod
    def _format_validation_phase(phase: str) -> str:
        raw = str(phase or "IDLE").strip()
        if not raw:
            raw = "IDLE"
        return raw.upper()

    @staticmethod
    def _format_validation_numeric(value: object, *, digits: int = 6) -> str:
        if value in (None, ""):
            return ""
        try:
            numeric = float(value)
        except Exception:
            return str(value)
        if not math.isfinite(numeric):
            return ""
        return f"{numeric:.{int(digits)}f}"

    @staticmethod
    def _format_validation_position(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, Mapping):
            parts: list[str] = []
            for key in sorted(value.keys(), key=lambda item: str(item)):
                try:
                    parts.append(f"{key}={float(value[key]):.3f}")
                except Exception:
                    parts.append(f"{key}={value[key]}")
            return " ".join(parts)
        text = str(value).strip()
        if not text:
            return ""
        try:
            return f"{float(text):.3f}"
        except Exception:
            return text

    def _validation_recipe_snapshot_from_ui(self) -> Recipe:
        try:
            return self._recipe_apply_from_ui()
        except Exception:
            return self.get_recipe_copy()

    def list_validation_section_choices(self) -> list[str]:
        try:
            recipe = self._validation_recipe_snapshot_from_ui()
            positions = list(plan_section_positions(recipe).positions_z)
        except Exception:
            positions = []
        if not positions:
            return ["1"]
        return [
            format_recipe_section_name(index, z_pos)
            for index, z_pos in enumerate(positions, start=1)
        ]

    def _set_validation_start_button_state(self, enabled: bool) -> None:
        for widget_name in ('validation_screen_start_btn',):
            start_btn = self._gauge_ui_widget(widget_name)
            if start_btn is None:
                continue
            try:
                start_btn.configure(state='normal' if enabled else 'disabled')
            except Exception:
                pass

    def _set_validation_stop_button_state(self, enabled: bool) -> None:
        for widget_name in ('validation_screen_stop_btn',):
            stop_btn = self._gauge_ui_widget(widget_name)
            if stop_btn is None:
                continue
            try:
                stop_btn.configure(state='normal' if enabled else 'disabled')
            except Exception:
                pass

    def _sync_validation_mode(self, workflow_state: str, message: str = "") -> None:
        try:
            sync_mode_state = getattr(self.mode_machine, "sync_validation_workflow_state", None)
            if callable(sync_mode_state):
                sync_mode_state(str(workflow_state or ""), str(message or ""))
        except Exception:
            pass

    def is_validation_cancel_requested(self) -> bool:
        cancel_event = getattr(self, "_validation_cancel_event", None)
        try:
            if cancel_event is not None and bool(cancel_event.is_set()):
                return True
        except Exception:
            pass
        try:
            return bool(getattr(self, "_validation_cancel_requested", False))
        except Exception:
            return False

    def _current_mode_kind_name(self) -> str:
        try:
            mode_machine = getattr(self, "mode_machine", None)
            mode_kind = getattr(mode_machine, "current_mode_kind", None)
            value = getattr(mode_kind, "value", None)
            if value is not None:
                return str(value)
            return str(mode_kind or "")
        except Exception:
            return ""

    def _is_auto_thread_alive(self) -> bool:
        try:
            runner = getattr(self, "_auto_thread", None)
            return bool(runner is not None and runner.is_alive())
        except Exception:
            return False

    def _prepare_validation_run(self, *, move_scenario: str) -> None:
        cancel_event = getattr(self, "_validation_cancel_event", None)
        if cancel_event is None:
            cancel_event = threading.Event()
            self._validation_cancel_event = cancel_event
        try:
            cancel_event.clear()
        except Exception:
            pass
        self._validation_cancel_requested = False
        current_profile = str(getattr(self, "_plc_poll_profile_req", "normal") or "normal")
        log(
            "VALIDATION_ENTER",
            current_poll_profile=current_profile,
            move_scenario=str(move_scenario or ""),
            mode_kind=self._current_mode_kind_name(),
            validation_running=bool(getattr(self, "_validation_running", False)),
            auto_thread_alive=self._is_auto_thread_alive(),
        )
        self.set_plc_poll_profile("normal", caller="validation_enter")

    def _cleanup_validation_run(
        self,
        *,
        status: str,
        phase: str | None = None,
        error: str = "",
    ) -> None:
        self.set_plc_poll_profile("normal", caller="validation_exit")
        cancel_event = getattr(self, "_validation_cancel_event", None)
        try:
            if cancel_event is not None:
                cancel_event.clear()
        except Exception:
            pass
        self._validation_cancel_requested = False
        self._validation_running = False
        self._validation_thread = None
        workflow_state = "ERR"
        normalized_status = str(status or "").strip().upper()
        if normalized_status == "DONE":
            workflow_state = "DONE"
        elif normalized_status == "STOP":
            workflow_state = "STOP"
        elif normalized_status == "STOPPING":
            workflow_state = "STOPPING"
        self._sync_validation_mode(workflow_state, error)
        log(
            "VALIDATION_EXIT",
            status=str(status or ""),
            phase=str(phase or ""),
            error=str(error or ""),
            poll_profile_after_cleanup=str(getattr(self, "_plc_poll_profile_req", "")),
        )

    def _finish_validation_run_ui(
        self,
        *,
        status: str,
        phase: str | None = None,
        result: str = "",
        error: str = "",
        export_path: str = "",
    ) -> None:
        self._cleanup_validation_run(status=status, phase=phase, error=error)
        self._set_validation_feedback(
            status=status,
            phase=phase,
            wait_phase="",
            wait_remaining_s="",
            result=result,
            error=error,
            export_path=export_path,
        )
        try:
            sync_mode_state = getattr(self.mode_machine, 'sync_current_mode_state', None)
            if callable(sync_mode_state):
                sync_mode_state()
        except Exception:
            pass
        self._set_validation_start_button_state(True)
        self._set_validation_stop_button_state(False)
        try:
            self.update_idletasks()
        except Exception:
            pass

    def stop_validation_run(self) -> None:
        if not bool(getattr(self, "_validation_running", False)):
            return None
        cancel_event = getattr(self, "_validation_cancel_event", None)
        if cancel_event is None:
            cancel_event = threading.Event()
            self._validation_cancel_event = cancel_event
        self._validation_cancel_requested = True
        try:
            cancel_event.set()
        except Exception:
            pass
        log(
            "VALIDATION_STOP_REQUEST",
            current_poll_profile=str(getattr(self, "_plc_poll_profile_req", "") or ""),
            mode_kind=self._current_mode_kind_name(),
            auto_thread_alive=self._is_auto_thread_alive(),
        )
        self._sync_validation_mode("STOPPING")
        self._set_validation_feedback(
            status="STOPPING",
            result="",
            error="",
        )
        self._set_validation_stop_button_state(False)
        try:
            self.abort_motion()
        except Exception:
            pass
        try:
            self.update_idletasks()
        except Exception:
            pass
        return None

    def start_validation_run(
        self,
        *,
        section_name: str,
        metric_name: str,
        repeat_count: int,
        reclamp_between_repeats: bool = False,
        reclamp_enabled: bool = False,
        rotation_stop_before_measure: bool = False,
        release_settle_s: float = 0.0,
        clamp_settle_s: float = 0.0,
        position_settle_s: float = 0.0,
        sample_delay_s: float = 0.0,
        validation_ax3_speed_dps: float = 60.0,
        move_enabled: bool = False,
        move_channel: str = "od_channel",
        move_away_delta_mm: float = 0.0,
        move_scenario: str = "distance_round_trip",
        move_from_section_index: int = 1,
        move_target_section_index: int = 1,
        move_return_section_index: int = 1,
    ) -> Optional[str]:
        try:
            def _bool_param(value) -> bool:
                if isinstance(value, bool):
                    return value
                text = str(value or "").strip().lower()
                if text in {"1", "true", "yes", "y", "on"}:
                    return True
                if text in {"", "0", "false", "no", "n", "off"}:
                    return False
                return bool(value)

            def _settle_param(value, field_name: str) -> float:
                text = str(value or "").strip()
                if not text:
                    return 0.0
                try:
                    numeric = float(text)
                except Exception as exc:
                    raise ValueError(f"{field_name} must be a number") from exc
                if numeric < 0.0:
                    raise ValueError(f"{field_name} must be >= 0")
                return numeric

            def _positive_param(value, field_name: str) -> float:
                text = str(value or "").strip()
                if not text:
                    raise ValueError(f"{field_name} must be > 0")
                try:
                    numeric = float(text)
                except Exception as exc:
                    raise ValueError(f"{field_name} must be a number") from exc
                if numeric <= 0.0:
                    raise ValueError(f"{field_name} must be > 0")
                return numeric

            def _choice_param(value, field_name: str, choices) -> str:
                text = str(value or "").strip()
                if text not in choices:
                    raise ValueError(f"{field_name} must be one of: " + ", ".join(choices))
                return text

            def _section_index_param(value, field_name: str) -> int:
                text = str(value or "").strip()
                if ":" in text:
                    text = text.split(":", 1)[0].strip()
                if not text:
                    raise ValueError(f"{field_name} must be >= 1")
                try:
                    numeric = int(float(text))
                except Exception as exc:
                    raise ValueError(f"{field_name} must be an integer") from exc
                if numeric < 1:
                    raise ValueError(f"{field_name} must be >= 1")
                return numeric

            metric = str(metric_name or "").strip()
            if metric not in FIXED_SECTION_PRIMARY_METRICS:
                raise ValueError(
                    "metric_name must be one of: " + ", ".join(FIXED_SECTION_PRIMARY_METRICS)
                )
            repeat = int(repeat_count)
            if repeat < 1:
                raise ValueError("repeat_count must be >= 1")
            request = FixedSectionRepeatabilityRequest(
                section_name=str(section_name or "").strip(),
                metric_name=metric,
                repeat_count=repeat,
                reclamp_between_repeats=_bool_param(reclamp_between_repeats),
                reclamp_enabled=_bool_param(reclamp_enabled),
                rotation_stop_before_measure=_bool_param(rotation_stop_before_measure),
                release_settle_s=_settle_param(release_settle_s, "release_settle_s"),
                clamp_settle_s=_settle_param(clamp_settle_s, "clamp_settle_s"),
                position_settle_s=_settle_param(position_settle_s, "position_settle_s"),
                sample_delay_s=_settle_param(sample_delay_s, "sample_delay_s"),
                validation_ax3_speed_dps=_positive_param(validation_ax3_speed_dps, "validation_ax3_speed_dps"),
                move_enabled=_bool_param(move_enabled),
                move_channel=_choice_param(
                    move_channel,
                    "move_channel",
                    VALIDATION_MOVE_CHANNELS,
                ),
                move_away_delta_mm=_settle_param(move_away_delta_mm, "move_away_delta_mm"),
                move_scenario=_choice_param(
                    move_scenario,
                    "move_scenario",
                    VALIDATION_MOVE_SCENARIOS,
                ),
                move_from_section_index=_section_index_param(
                    move_from_section_index,
                    "move_from_section_index",
                ),
                move_target_section_index=_section_index_param(
                    move_target_section_index,
                    "move_target_section_index",
                ),
                move_return_section_index=_section_index_param(
                    move_return_section_index,
                    "move_return_section_index",
                ),
            )

            if bool(getattr(self, '_validation_running', False)):
                raise RuntimeError("固定截面重复性验证正在运行")

            try:
                enter_validation = getattr(self.mode_machine, 'enter_validation', None)
                if callable(enter_validation):
                    enter_validation()
            except Exception:
                pass
            self._prepare_validation_run(move_scenario=request.move_scenario)

            validation_session = ValidationSession()
            self.validation_session = validation_session
            validation_runtime_state = RuntimeState.from_validation_session(validation_session)
            recipe_snapshot = self._validation_recipe_snapshot_from_ui()
            workflow = ValidationWorkflow(
                recipe=recipe_snapshot,
                calibration=self.get_calibration_snapshot(),
                runtime_state=validation_runtime_state,
                gateway=AppDeviceGateway(self),
                run_repository=self._make_run_repository(),
                validation_session=validation_session,
            )
            validation_repository = self._make_validation_repository()

            self._validation_running = True
            self._sync_validation_mode("RUN")
            self._set_validation_start_button_state(False)
            self._set_validation_stop_button_state(True)
            self._reset_validation_summary_panel()
            self._set_validation_feedback(
                status=f"RUNNING 0/{repeat}",
                phase="PREPARE",
                wait_phase="",
                wait_remaining_s="",
                current_repeat=f"0/{repeat}",
                result="",
                error="",
                export_path="",
                move_target_pos="",
                move_actual_pos="",
            )
            try:
                self.update_idletasks()
            except Exception:
                pass

            def _worker() -> None:
                try:
                    def _running_summary_snapshot() -> tuple[object | None, dict[str, Any]]:
                        latest_capture = None
                        summary_payload: dict[str, Any] = {}
                        try:
                            captures = tuple(getattr(workflow, "fixed_section_repeat_captures", ()) or ())
                            if captures:
                                latest_capture = captures[-1]
                        except Exception:
                            latest_capture = None
                        try:
                            build_summary = getattr(workflow, "build_fixed_section_repeatability_summary", None)
                            if callable(build_summary):
                                summary_payload = dict(build_summary() or {})
                        except Exception:
                            summary_payload = {}
                        return latest_capture, summary_payload

                    def _progress_update(index: int, total_count: int) -> None:
                        latest_capture, summary_payload = _running_summary_snapshot()
                        def _apply_progress() -> None:
                            self._set_validation_feedback(
                                status=f"RUNNING {int(index)}/{int(total_count)}",
                                result="",
                                error="",
                                export_path="",
                            )
                            self._set_validation_current_repeat_result(latest_capture)
                            self._set_validation_summary_values(summary_payload)
                        self.after(0, _apply_progress)

                    def _status_update(status_text: str) -> None:
                        def _apply_status() -> None:
                            self._set_validation_feedback(
                                status=str(status_text),
                                result="",
                                error="",
                                export_path="",
                            )
                        self.after(0, _apply_status)

                    def _phase_update(phase_event) -> None:
                        phase_name = str(getattr(phase_event, 'phase', '') or '')
                        payload = getattr(phase_event, 'payload', {}) or {}
                        try:
                            target_pos = payload.get(
                                'target_positions_mm',
                                payload.get('target_position_mm'),
                            )
                        except Exception:
                            target_pos = None
                        try:
                            actual_pos = payload.get(
                                'actual_positions_mm',
                                payload.get('actual_position_mm'),
                            )
                        except Exception:
                            actual_pos = None
                        repeat_index = int(getattr(phase_event, 'repeat_index', 0) or 0)
                        total_count = int(getattr(phase_event, 'total', 0) or 0)
                        def _apply_phase() -> None:
                            wait_phase = phase_name if phase_name.startswith('wait_') else ""
                            self._set_validation_feedback(
                                phase=phase_name,
                                wait_phase=wait_phase,
                                wait_remaining_s=(0.0 if wait_phase else ""),
                                current_repeat=(f"{repeat_index}/{total_count}" if total_count > 0 else ""),
                            )
                            self._set_validation_move_position(
                                target_pos=target_pos,
                                actual_pos=actual_pos,
                            )
                        self.after(0, _apply_phase)

                    def _wait_update(phase_name: str, repeat_index: int, total_count: int, remaining_s: float) -> None:
                        def _apply_wait() -> None:
                            self._set_validation_feedback(
                                wait_phase=phase_name,
                                wait_remaining_s=float(remaining_s),
                                current_repeat=(f"{int(repeat_index)}/{int(total_count)}" if int(total_count) > 0 else ""),
                            )
                        self.after(0, _apply_wait)

                    rows, summary = workflow.run_fixed_section_repeatability(
                        request,
                        progress_callback=_progress_update,
                        status_callback=_status_update,
                        phase_callback=_phase_update,
                        wait_callback=_wait_update,
                    )
                    export_dir = validation_repository.export_fixed_section_repeatability(
                        context=workflow.build_export_context(),
                        request=request,
                        rows=rows,
                        summary=summary,
                        captures=workflow.fixed_section_repeat_captures,
                    )
                    result_text = (
                        f"{metric} "
                        f"count={int(summary.get('count', 0))} "
                        f"mean={float(summary.get('mean', 0.0)):.6f} "
                        f"std={float(summary.get('std', 0.0)):.6f} "
                        f"min={float(summary.get('min', 0.0)):.6f} "
                        f"max={float(summary.get('max', 0.0)):.6f} "
                        f"range={float(summary.get('range', 0.0)):.6f}"
                    )
                    latest_capture, summary_payload = _running_summary_snapshot()
                    def _on_success() -> None:
                        self.validation_session = workflow.validation_session or validation_session
                        self._set_validation_current_repeat_result(latest_capture)
                        self._set_validation_summary_values(summary if summary else summary_payload)
                        self._finish_validation_run_ui(
                            status="DONE",
                            result=result_text,
                            error="",
                            export_path=export_dir,
                        )
                    self.after(0, _on_success)
                except ValidationActionCancelled:
                    def _on_cancel() -> None:
                        self.validation_session = workflow.validation_session or validation_session
                        self._finish_validation_run_ui(
                            status="STOP",
                            phase="STOP",
                            result="",
                            error="",
                            export_path="",
                        )
                    try:
                        self.after(0, _on_cancel)
                    except Exception:
                        self.validation_session = workflow.validation_session or validation_session
                        self._cleanup_validation_run(
                            status="STOP",
                            phase="STOP",
                            error="",
                        )
                except Exception as exc:
                    error_text = str(exc)
                    def _on_error() -> None:
                        self.validation_session = workflow.validation_session or validation_session
                        self._finish_validation_run_ui(
                            status="ERR",
                            result="",
                            error=error_text,
                            export_path="",
                        )
                    try:
                        self.after(0, _on_error)
                    except Exception:
                        self.validation_session = workflow.validation_session or validation_session
                        self._cleanup_validation_run(
                            status="ERR",
                            error=error_text,
                        )

            log(
                "VALIDATION_THREAD_STARTING",
                current_poll_profile=str(getattr(self, "_plc_poll_profile_req", "") or ""),
                move_scenario=str(request.move_scenario or ""),
            )
            worker = threading.Thread(
                target=_worker,
                name="validation-fixed-section-repeatability",
                daemon=True,
            )
            self._validation_thread = worker
            worker.start()
            return None
        except Exception as exc:
            self._finish_validation_run_ui(
                status="ERR",
                phase="IDLE",
                result="",
                error=str(exc),
                export_path="",
            )
            return None

    _set_validation_debug_feedback = _set_validation_feedback
    _set_validation_debug_phase = _set_validation_phase
    _set_validation_debug_move_position = _set_validation_move_position
    _set_validation_debug_wait_phase = _set_validation_wait_phase
    _set_validation_debug_wait_remaining = _set_validation_wait_remaining
    _set_validation_debug_current_repeat = _set_validation_current_repeat
    _reset_validation_debug_summary_panel = _reset_validation_summary_panel
    _set_validation_debug_current_repeat_result = _set_validation_current_repeat_result
    _set_validation_debug_summary_values = _set_validation_summary_values
    _format_validation_debug_phase = _format_validation_phase
    _format_validation_debug_numeric = _format_validation_numeric
    _format_validation_debug_position = _format_validation_position
    _set_validation_debug_start_button_state = _set_validation_start_button_state
    _set_validation_debug_stop_button_state = _set_validation_stop_button_state
    _sync_validation_debug_mode = _sync_validation_mode
    _prepare_validation_debug_run = _prepare_validation_run
    _cleanup_validation_debug_run = _cleanup_validation_run
    _finish_validation_debug_run_ui = _finish_validation_run_ui
    start_fixed_section_repeatability_debug = start_validation_run
    stop_fixed_section_repeatability_debug = stop_validation_run

    def handle_main_result_selection(self, event=None):
        return self._on_result_select(event)

    def refresh_main_summary_panel(self):
        return self._refresh_auto_std_panel()

    def write_keytest_y(self, y_point: int, value: int) -> None:
        self._keytest_write_y(y_point, value)

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

    def write_coil(self, coil_addr: int, value: int | bool) -> None:
        """Thread-safe one-shot write to a raw Modbus coil address."""
        try:
            coil = int(coil_addr)
            vv = 1 if bool(value) else 0
            self.cmd_q.put(CmdWriteCoil(coil_addr=coil, value=vv))
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
                    self.measurement_controller.start_measurement()
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
                    self.measurement_controller.stop_measurement()
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

    def _kv_row(self, parent: ttk.Frame, label: str, var: tk.StringVar, row: int):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="e", padx=6, pady=4
        )
        ttk.Entry(parent, width=18, textvariable=var).grid(
            row=row, column=1, sticky="w", padx=6, pady=4
        )

    def _recipe_ui_widget(self, name: str) -> Any:
        try:
            presenter = getattr(self, '_recipe_screen_presenter', None)
            widget_getter = getattr(presenter, 'widget', None)
            if callable(widget_getter):
                return widget_getter(name)
        except Exception:
            pass
        return None

    def _main_ui_widget(self, name: str) -> Any:
        try:
            presenter = getattr(self, '_screen_presenter', None)
            widget_getter = getattr(presenter, 'widget', None)
            if callable(widget_getter):
                return widget_getter(name)
        except Exception:
            pass
        return None

    def _main_view_state(self, name: str, default: Any = None) -> Any:
        try:
            presenter = getattr(self, '_screen_presenter', None)
            getter = getattr(presenter, 'view_state', None)
            if callable(getter):
                return getter(name, default)
        except Exception:
            pass
        return default

    def _gauge_ui_widget(self, name: str) -> Any:
        try:
            presenter = getattr(self, '_gauge_screen_presenter', None)
            widget_getter = getattr(presenter, 'widget', None)
            if callable(widget_getter):
                return widget_getter(name)
        except Exception:
            pass
        return None

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
          managed by dedicated "length/rotate position" and keepout logic.
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
        return RecipeFormMapper(self._recipe_screen_presenter).ui_vars_to_recipe()

    def _recipe_compute(self):
        try:
            r = self._recipe_apply_from_ui()
            rebuilt_positions = list(r.compute_default_positions_z())
            r.section_pos_z = list(rebuilt_positions)
            self.recipe.section_pos_z = list(rebuilt_positions)
            self.recipe.section_pos_ui = list(rebuilt_positions)
            self._refresh_recipe_table()
            self._refresh_auto_std_panel()

            try:
                r = self.get_recipe_copy()
                log("AUTO_START", section_count=getattr(r,'section_count',None), points_per_rev=getattr(r,'points_per_rev',None), min_bin_coverage=getattr(r,'min_bin_coverage',None), timeout_s=getattr(r,'sample_timeout_s',None), max_revolutions=getattr(r,'max_revolutions',None))
            except Exception:
                log("AUTO_START", section_count="unknown")
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
        combo = self._recipe_ui_widget('recipe_name_combo')
        try:
            if combo is not None:
                combo["values"] = names
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
        return RecipeFormMapper(self._recipe_screen_presenter).recipe_to_dict(r)

    def _recipe_apply_data_to_ui(self, data: dict) -> None:
        """Apply recipe dict to UI vars and internal recipe object (no dialogs)."""
        RecipeFormMapper(self._recipe_screen_presenter).apply_data_to_ui(data)

    def _recipe_load_from_store(self, name: str, *, show_msg: bool = False) -> None:
        data = self.recipe_store.load(name)
        self._recipe_apply_data_to_ui(data)
        try:
            recipe_logger.info("RECIPE_LOAD name=%s", name)
        except Exception:
            pass
        self._log_ax3_speed_trace("recipe_load_complete")
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
            try:
                recipe_logger.info("RECIPE_SAVE name=%s path=%s", safe, save_path)
            except Exception:
                pass
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

    def _build_recipe_section_plan(self, recipe: Optional[Recipe] = None):
        recipe_obj = self.recipe if recipe is None else recipe
        ax2_abs = float(self._get_ax2_keepout_ref_abs(prefer_rot=True))
        soft_limits = {
            0: (float(self.get_axis_copy(0).softlim_pos), float(self.get_axis_copy(0).softlim_neg)),
            1: (float(self.get_axis_copy(1).softlim_pos), float(self.get_axis_copy(1).softlim_neg)),
            4: (float(self.get_axis_copy(4).softlim_pos), float(self.get_axis_copy(4).softlim_neg)),
        }
        return build_recipe_section_plan(
            recipe_obj,
            self.axis_cal,
            ax2_abs=ax2_abs,
            soft_limits_abs=soft_limits,
        )

    def _refresh_recipe_table(self):
        tree = self._recipe_ui_widget('recipe_tree')
        try:
            if tree is None:
                return
            tree.delete(*tree.get_children())
        except Exception:
            return

        r = self.recipe

        # Ensure positions length (Z_Pos)
        if len(getattr(r, 'section_pos_z', [])) != int(r.section_count):
            r.section_pos_z = list(r.compute_default_positions_z())

        # Keep legacy aligned (deprecated)
        try:
            r.section_pos_ui = list(r.section_pos_z)
        except Exception:
            pass

        planned_positions = tuple(float(z) for z in plan_section_positions(r).positions_z)
        r.section_pos_z = list(planned_positions)
        try:
            r.section_pos_ui = list(planned_positions)
        except Exception:
            pass

        try:
            section_plan = self._build_recipe_section_plan(r)
            planned_rows = {
                int(row.section_index) - 1: row
                for row in getattr(section_plan, 'sections', ())
            }
        except Exception:
            planned_rows = {}

        for i, z_od_disp in enumerate(planned_positions):
            z_od_disp = float(z_od_disp)
            # 由 OD 截面位置推导：AX0/AX1/AX4 目标 abs 以及 ID 位置
            row = planned_rows.get(i)
            if row is not None:
                ax0_abs = float(row.ax0_abs)
                ax1_abs = float(row.ax1_abs)
                ax4_abs = float(row.ax4_abs)
                z_id_disp = float(row.z_id_disp)
            else:
                ax0_abs, ax1_abs, ax4_abs, z_id_disp = 0.0, 0.0, 0.0, z_od_disp + float(getattr(self.axis_cal, 'b14', 0.0))

            src = (
                "示教/保留"
                if hasattr(self, "_taught_mark")
                and getattr(self, "_taught_mark", {}).get(i, False)
                else "计算"
            )
            tree.insert(
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
        tree = self._recipe_ui_widget('recipe_tree')
        if tree is None:
            return None
        sel = tree.selection()
        if not sel:
            return None
        item = sel[0]
        vals = tree.item(item, "values")
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

            # Center clamp AX2: directly move AX2 by Z_disp (section position)
            if mode == 3:
                self.movea_abs(2, float(self.axis_cal.z_disp_to_abs(2, z_od_disp)), context='SectionMove')
                return
            section_plan = self._build_recipe_section_plan(r)
            selected_row = section_plan.section_for_recipe_index(idx)

            # Move selected teach axes
            if mode in (0, 2):
                self.movea_abs(0, float(selected_row.ax0_abs), context='SectionMove')
            if mode in (1, 2):
                self.movea_abs(1, float(selected_row.ax1_abs), context='SectionMove')
                self.movea_abs(4, float(selected_row.ax4_abs), context='SectionMove')
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

            # Center clamp AX2: save section position by current AX2 Z_disp
            if mode == 3:
                ac2 = self.get_axis_copy(2)
                z2_disp = float(self.axis_cal.abs_to_z_disp(2, ac2.act_pos))
                r.section_pos_z[idx] = float(z2_disp)
                self.recipe.section_pos_z = list(r.section_pos_z)
                self.recipe.section_pos_ui = list(self.recipe.section_pos_z)  # legacy
                if not hasattr(self, "_taught_mark"):
                    self._taught_mark = {}
                self._taught_mark[idx] = True
                self._refresh_recipe_table()
                self._refresh_teach_pos()
                return

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
                        log("teach: OD/ID not aligned; saving section using OD")
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
            ac1 = self.get_axis_copy(1)
            ac2 = self.get_axis_copy(2)
            ac4 = self.get_axis_copy(4)
            cal = self.axis_cal

            # ---- debug snapshot (entry) ----
            # log("DBG align_by_od", entry_ax0_act_abs=float(ac0.act_pos))
            # log("DBG align_by_od", entry_ax1_act_abs=float(ac1.act_pos))
            # log("DBG align_by_od", entry_ax2_act_abs=float(ac2.act_pos))
            # log("DBG align_by_od", entry_ax4_act_abs=float(ac4.act_pos))
            # log("DBG align_by_od", recipe_ax2_rot_abs=float(getattr(self.recipe, 'ax2_rot_abs', 0.0)))
            # log("DBG align_by_od", recipe_ax2_rot_valid=bool(getattr(self.recipe, 'ax2_rot_valid', False)))
            # log("DBG align_by_od", axis_cal_b14=float(getattr(cal, 'b14', 0.0)))
            # log("DBG align_by_od", axis_cal_b2=float(getattr(cal, 'b2', 0.0)))
            # log("DBG align_by_od", axis_cal_keepout_w=float(getattr(cal, 'keepout_w', 0.0)))
            # log("DBG align_by_od", axis_cal_off_ax0=float(getattr(cal, 'off_ax0', 0.0)))
            # log("DBG align_by_od", axis_cal_off_ax1=float(getattr(cal, 'off_ax1', 0.0)))
            # log("DBG align_by_od", axis_cal_off_ax2=float(getattr(cal, 'off_ax2', 0.0)))
            # log("DBG align_by_od", axis_cal_off_ax4=float(getattr(cal, 'off_ax4', 0.0)))
            try:
                # log("DBG align_by_od", axis_cal_sign_eff_0=int(cal.sign_eff(0)))
                # log("DBG align_by_od", axis_cal_sign_eff_1=int(cal.sign_eff(1)))
                # log("DBG align_by_od", axis_cal_sign_eff_2=int(cal.sign_eff(2)))
                # log("DBG align_by_od", axis_cal_sign_eff_4=int(cal.sign_eff(4)))
                pass
            except Exception as e_sign:
                # log("DBG align_by_od", axis_cal_sign_eff_err=e_sign)
                pass

            z0_raw = cal.abs_to_z_raw(0, ac0.act_pos)
            z_id_raw_tgt = float(z0_raw) + float(cal.b14)
            ax2_ref_abs = float(self._get_ax2_keepout_ref_abs(prefer_rot=True))
            z2_raw_ref = float(cal.abs_to_z_raw(2, ax2_ref_abs))
            handoff_raw = float(z2_raw_ref) + float(cal.b2) + float(cal.keepout_w)

            # log("DBG align_by_od", z0_raw=float(z0_raw))
            # log("DBG align_by_od", z_id_raw_tgt=float(z_id_raw_tgt))
            # log("DBG align_by_od", ax2_ref_abs=float(ax2_ref_abs))
            # log("DBG align_by_od", z2_raw_ref=float(z2_raw_ref))
            # log("DBG align_by_od", handoff_raw=float(handoff_raw))
            # log("DBG align_by_od", align_raw_delta=float(z_id_raw_tgt - z0_raw))
            # log("DBG align_by_od", align_raw_delta_minus_b14=float((z_id_raw_tgt - z0_raw) - float(cal.b14)))

            # split into AX1/AX4 raw by handoff + AX1 MoveA low-abs bound
            lim_lo_abs = -float('inf')
            try:
                p = float(getattr(ac1, 'softlim_pos', float('nan')))
                n = float(getattr(ac1, 'softlim_neg', float('nan')))
                if (p == p) and (n == n) and (abs(p) + abs(n) >= 1e-6):
                    lo_abs = min(p, n)
                    hi_abs = max(p, n)
                    if (hi_abs - lo_abs) >= 1e-9:
                        lim_lo_abs = max(lim_lo_abs, lo_abs)
            except Exception:
                pass

            try:
                # Match apply_soft_limits_abs(..., context='MoveA') keepout reference path.
                ax2_abs_movea = float(
                    self._get_ax2_keepout_ref_abs(prefer_rot=self._ctx_use_ax2_rot_ref('MoveA'))
                )
                z2_raw_movea = float(cal.abs_to_z_raw(2, ax2_abs_movea))
                zc_movea = float(z2_raw_movea + cal.b2)
                w_movea = float(cal.keepout_w)
                if abs(w_movea) >= 1e-6:
                    abs_min_keepout = float(cal.z_raw_to_abs(1, zc_movea + w_movea))
                    lim_lo_abs = max(lim_lo_abs, abs_min_keepout)
            except Exception:
                pass

            if math.isfinite(lim_lo_abs):
                z1_raw_lo = float(cal.abs_to_z_raw(1, lim_lo_abs))
            else:
                z1_raw_lo = float('nan')

            sign1 = int(cal.sign_eff(1))
            if sign1 < 0:
                if math.isfinite(z1_raw_lo):
                    z1_raw_tgt = min(float(z_id_raw_tgt), float(handoff_raw), float(z1_raw_lo))
                else:
                    z1_raw_tgt = min(float(z_id_raw_tgt), float(handoff_raw))
            else:
                z1_raw_tgt = min(float(z_id_raw_tgt), float(handoff_raw))
                if math.isfinite(z1_raw_lo):
                    z1_raw_tgt = max(float(z1_raw_tgt), float(z1_raw_lo))
                    # Keep keepout handoff strategy as an upper raw cap for AX1.
                    z1_raw_tgt = min(float(z1_raw_tgt), float(handoff_raw))

            z4_raw_tgt = float(z_id_raw_tgt) - float(z1_raw_tgt)
            if z4_raw_tgt < 0.0:
                # log("DBG align_by_od", z4_raw_negative_guard=float(z4_raw_tgt))
                z4_raw_tgt = 0.0

            # log("DBG align_by_od", lim_lo_abs=(float(lim_lo_abs) if math.isfinite(lim_lo_abs) else None))
            # log("DBG align_by_od", z1_raw_lo=(float(z1_raw_lo) if math.isfinite(z1_raw_lo) else None))
            # log("DBG align_by_od", z1_raw_tgt_final=float(z1_raw_tgt))
            # log("DBG align_by_od", z4_raw_tgt_final=float(z4_raw_tgt))

            z1_abs_req = float(cal.z_raw_to_abs(1, z1_raw_tgt))
            z4_abs_req = float(cal.z_raw_to_abs(4, z4_raw_tgt))
            # log("DBG align_by_od", z1_abs_req=float(z1_abs_req))
            # log("DBG align_by_od", z4_abs_req=float(z4_abs_req))

            ax1_abs_final = float(self.apply_soft_limits_abs(1, float(z1_abs_req), strict=False, context='MoveA'))
            ax4_abs_final = float(self.apply_soft_limits_abs(4, float(z4_abs_req), strict=False, context='MoveA'))
            # log("DBG align_by_od", ax1_abs_req=float(z1_abs_req))
            # log("DBG align_by_od", ax1_abs_final=float(ax1_abs_final))
            # log("DBG align_by_od", ax4_abs_req=float(z4_abs_req))
            # log("DBG align_by_od", ax4_abs_final=float(ax4_abs_final))

            self.movea_abs(1, float(z1_abs_req))
            self.movea_abs(4, float(z4_abs_req))
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
                # display only
                self._refresh_start_pos()
                return
            a0 = float(getattr(r, 'start_ax0_abs', 0.0))
            z_raw = float(self.axis_cal.abs_to_z_raw(0, a0))
            self.axis_cal.z_pos = z_raw
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
            messagebox.showinfo('Start', '已保存测量区间起始位(Start)：Z_Pos=0')
        except Exception as e:
            messagebox.showerror('Start保存失败', str(e))

    def _teach_start_from_standby(self) -> None:
        """Convenience: set Start from already-saved standby pose (AX0 only)."""
        try:
            if not bool(getattr(self.recipe, 'standby_valid', False)):
                messagebox.showwarning('Start', '待定点尚未设置：请先保存待定点。')
                return
            self.recipe.start_valid = True
            self.recipe.start_ax0_abs = float(getattr(self.recipe, 'standby_ax0_abs', 0.0))
            self._apply_start_anchor_from_recipe()
            self._refresh_recipe_table()
            self._refresh_teach_pos()
            messagebox.showinfo('Start', '已从待定点同步设置Start：Z_Pos=0')
        except Exception as e:
            messagebox.showerror('Start设置失败', str(e))

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
                messagebox.showwarning('Start', 'Start尚未设置：请先点击“保存为测量区间起始位(Start)”。')
                return

            z_od_disp = 0.0
            ax2_abs = float(self._get_ax2_keepout_ref_abs(prefer_rot=True))
            softlims = {
                0: (float(self.get_axis_copy(0).softlim_pos), float(self.get_axis_copy(0).softlim_neg)),
                1: (float(self.get_axis_copy(1).softlim_pos), float(self.get_axis_copy(1).softlim_neg)),
                4: (float(self.get_axis_copy(4).softlim_pos), float(self.get_axis_copy(4).softlim_neg)),
            }
            t = self.axis_cal.od_z_disp_to_targets(z_od_disp, ax2_abs=ax2_abs, softlims_abs=softlims)
            if mode in (0, 2):
                self.movea_abs(0, float(t['ax0_abs']), context='GotoStart')
            if mode in (1, 2):
                self.movea_abs(1, float(t['ax1_abs']), context='GotoStart')
                self.movea_abs(4, float(t['ax4_abs']), context='GotoStart')
        except Exception as e:
            messagebox.showerror('移动Start失败', str(e))

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
                messagebox.showwarning('End', 'Start尚未设置：请先保存Start，再移动到End。')
                return

            total = float(getattr(self.recipe, 'meas_total_len_mm', 0.0) or 0.0)
            if total <= 1e-6:
                total = float(getattr(self.recipe, 'pipe_len_mm', 0.0) or 0.0) - float(getattr(self.recipe, 'clamp_occupy_mm', 0.0) or 0.0)
            total = max(0.0, float(total))

            z_od_disp = float(total)
            ax2_abs = float(self._get_ax2_keepout_ref_abs(prefer_rot=True))
            softlims = {
                0: (float(self.get_axis_copy(0).softlim_pos), float(self.get_axis_copy(0).softlim_neg)),
                1: (float(self.get_axis_copy(1).softlim_pos), float(self.get_axis_copy(1).softlim_neg)),
                4: (float(self.get_axis_copy(4).softlim_pos), float(self.get_axis_copy(4).softlim_neg)),
            }
            t = self.axis_cal.od_z_disp_to_targets(z_od_disp, ax2_abs=ax2_abs, softlims_abs=softlims)
            if mode in (0, 2):
                self.movea_abs(0, float(t['ax0_abs']), context='GotoEnd')
            if mode in (1, 2):
                self.movea_abs(1, float(t['ax1_abs']), context='GotoEnd')
                self.movea_abs(4, float(t['ax4_abs']), context='GotoEnd')
        except Exception as e:
            messagebox.showerror('移动End失败', str(e))

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
        """Pick current AX0 position as bottom-approach (AX0 abs) for length measurement."""
        try:
            if not hasattr(self, "len_z_low_approach_var"):
                return
            a0 = float(self.get_axis_copy(0).act_pos)
            # store absolute act_pos to decouple from Start(Z_Pos)
            self.len_z_low_approach_var.set(f"{a0:.3f}")
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

            abs_low_appr = _f(getattr(self, "len_z_low_approach_var", tk.StringVar(value="0")).get(), 0.0)
            z_low_appr = float(self.axis_cal.abs_to_z_disp(0, abs_low_appr))
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
        """Toggle bottom-edge search thread (GO -> HI)."""
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
                    btn = self._recipe_ui_widget('btn_len_search_low')
                    if btn is not None:
                        btn.configure(text='尝试搜索底边(GO→HI)')
                except Exception:
                    pass
                return

            # start new
            stop_evt = threading.Event()
            self._len_edge_search_stop_evt = stop_evt
            th = threading.Thread(target=self._teach_len_search_low_worker, args=(stop_evt,), daemon=True)
            self._len_edge_search_thread = th

            try:
                btn = self._recipe_ui_widget('btn_len_search_low')
                if btn is not None:
                    btn.configure(text='停止搜索底边')
                if hasattr(self, 'len_edge_state_var'):
                    self.len_edge_state_var.set('底边搜索：准备...')
            except Exception:
                pass

            th.start()
        except Exception as e:
            messagebox.showerror('底边搜索', str(e))

    def _teach_len_search_high_toggle(self) -> None:
        """Toggle top-edge search thread (GO -> HI)."""
        try:
            th = getattr(self, '_len_edge_search_high_thread', None)
            if th is not None and getattr(th, 'is_alive', lambda: False)():
                evt = getattr(self, '_len_edge_search_high_stop_evt', None)
                if evt is not None:
                    evt.set()
                try:
                    if hasattr(self, 'len_edge_state_var'):
                        self.len_edge_state_var.set('顶边搜索：停止中...')
                    btn = self._recipe_ui_widget('btn_len_search_high')
                    if btn is not None:
                        btn.configure(text='尝试搜索顶边(GO→HI)')
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
                btn = self._recipe_ui_widget('btn_len_search_high')
                if btn is not None:
                    btn.configure(text='停止搜索顶边')
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
        """Start VelMove for a given axis with explicit setpoints (without relying on axis debug UI).

        Note: In this project, some commands are level-type bits; if a previous STOP/HALT/MOVEA/MOVER/JOG bit
        is still latched by mistake (or not yet cleared by SEQ/ACK), VelMove may be blocked and axis will
        appear "not moving". Here we proactively clear potentially conflicting bits before requesting VelMove.
        """
        ax = max(0, min(AXIS_COUNT - 1, int(axis)))
        base = self._base(ax)
        # write FP64 setpoints
        self._write_regs(base + OFF_VEL_VELMOVE, encode_float64_to_4regs(float(vel_velmove), FLOAT64_WORD_ORDER))
        self._write_regs(base + OFF_ACC, encode_float64_to_4regs(float(acc), FLOAT64_WORD_ORDER))
        self._write_regs(base + OFF_DEC, encode_float64_to_4regs(float(dec), FLOAT64_WORD_ORDER))
        self._write_regs(base + OFF_JERK, encode_float64_to_4regs(float(jerk), FLOAT64_WORD_ORDER))

        # clear other command bits that could block velmove, then set velmove
        clr = 0
        try:
            clr |= (CMD_JOG_F_REQ | CMD_JOG_B_REQ)
        except Exception:
            pass
        for _n in ('CMD_STOP_REQ', 'CMD_HALT_REQ', 'CMD_MOVEA_REQ', 'CMD_MOVER_REQ'):
            try:
                clr |= int(globals().get(_n, 0) or 0)
            except Exception:
                pass
        try:
            if clr:
                self.set_cmd_bits(ax, set_mask=0, clr_mask=clr)
        except Exception:
            pass

        # request velmove (level)
        if int(ax) == 3:
            self._log_ax3_speed_trace("velmove_start_axis_ax3_pre")
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

    def velmove(self, axis: int, velocity: float, *, acc: float = 80.0, dec: float = 80.0, jerk: float = 300.0) -> None:
        """Public wrapper for axis VelMove start."""
        self._velmove_start_axis(axis, velocity, acc=acc, dec=dec, jerk=jerk)

    def stop(self, axis: int) -> None:
        """Public wrapper for axis stop."""
        self._velmove_stop_axis(axis)

    def halt(self, axis: int) -> None:
        """Public wrapper for axis halt."""
        self._pulse_cmd_bits(axis, CMD_HALT_REQ)

    def reset(self, axis: int) -> None:
        """Public wrapper for axis reset."""
        self._pulse_cmd_bits(axis, CMD_RESET_REQ)

    def enable(self, axis: int) -> None:
        """Public wrapper for axis enable."""
        self.set_cmd_bits(axis, set_mask=CMD_EN_REQ, clr_mask=0)


    def _wait_axis_stop_settled(
        self,
        axis: int,
        *,
        timeout_s: float = 1.5,
        stable_cycles: int = 8,
        eps_abs: float = 0.02,
        stop_evt: 'threading.Event|None' = None,
    ) -> bool:
        """Wait until the axis is no longer in 'stopping' transient (warn=1003) and position becomes stable.

        This is used to avoid BMC_A_VelMove 1003 (axis is stopping) when issuing back-to-back velmove commands
        in bidirectional edge searches.

        Returns True if settled within timeout, else False.
        """
        try:
            ax = max(0, min(AXIS_COUNT - 1, int(axis)))
        except Exception:
            ax = int(axis)
        t0 = time.time()
        last = None
        stable = 0
        # clamp
        stable_cycles = max(3, int(stable_cycles))
        eps_abs = max(1e-6, float(eps_abs))
        while (time.time() - t0) < float(timeout_s):
            try:
                if stop_evt is not None and stop_evt.is_set():
                    return False
            except Exception:
                pass
            ac = None
            try:
                ac = self.get_axis_copy(ax)
            except Exception:
                ac = None
            if ac is None:
                time.sleep(0.05)
                continue
            try:
                err = int(getattr(ac, 'err', 0) or 0)
            except Exception:
                err = 0
            if err != 0:
                return False
            try:
                warn = int(getattr(ac, 'warn', 0) or 0)
            except Exception:
                warn = 0
            try:
                pos = float(getattr(ac, 'act_pos', 0.0) or 0.0)
            except Exception:
                pos = 0.0

            if last is not None and abs(pos - float(last)) <= eps_abs and warn != 1003:
                stable += 1
                if stable >= stable_cycles:
                    return True
            else:
                stable = 0
            last = pos
            time.sleep(0.05)
        return False

    def _teach_len_search_low_worker(self, stop_evt: threading.Event) -> None:
        """Worker thread: bottom-edge bidirectional search (GO->HI then HI->GO) and lock AX0 Z_disp.

        机制说明：
        - 需要测径仪返回比较器判定字段(judge)，例如 GO/HI/LO。
        - 第1段：从接近位开始，沿 +Z_disp 方向慢速运动，检测 GO→HI，并锁定“最后一次GO”的 Z_disp 作为 edge1。
        - 第2段：反向沿 -Z_disp 方向慢速运动，检测 HI→GO，并锁定“最后一次HI”的 Z_disp 作为 edge2。
        - 最终边沿 = (edge1 + edge2) / 2。
        """

        # local UI helpers
        def ui_msg(msg: str) -> None:
            try:
                if hasattr(self, 'len_edge_state_var'):
                    self._ui_set(self.len_edge_state_var, msg)
            except Exception:
                pass

        def ui_done_btn() -> None:
            try:
                btn = self._recipe_ui_widget('btn_len_search_low')
                if btn is not None:
                    self._ui_btn_text(btn, '尝试搜索底边(GO→HI)')
            except Exception:
                pass

        def _wait_new_judge(ts0: float, tmax: float = 0.8):
            """Return (ts, judge) for a new gauge sample; None on timeout."""
            t0 = time.time()
            last_ts = float(ts0)
            while (not stop_evt.is_set()) and ((time.time() - t0) < float(tmax)):
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
                if ts <= last_ts:
                    continue
                j = str(getattr(s, 'judge', 'UNK') or 'UNK').strip().upper()
                return ts, j
            return None

        def _axis_not_moving_guard(z_cur: float):
            """Detect 'not moving' and bail early to avoid waiting until timeout."""
            nonlocal last_move_z, last_move_ts
            if last_move_z is None:
                last_move_z = float(z_cur)
                last_move_ts = time.time()
                return False
            if abs(float(z_cur) - float(last_move_z)) >= 0.15:
                last_move_z = float(z_cur)
                last_move_ts = time.time()
                return False
            if (time.time() - float(last_move_ts)) >= 1.0:
                return True
            return False

        found = False
        edge_avg = None

        try:
            # --- validations ---
            if bool(getattr(self, 'sim_gauge_enabled', False)) or (hasattr(self, 'sim_gauge_var') and int(self.sim_gauge_var.get() or 0) == 1):
                ui_msg('底边搜索：模拟测径仪不支持比较器(GO)')
                return

            gw = getattr(self, 'gauge_worker', None)
            if gw is None or (not getattr(gw, 'enabled', False)):
                ui_msg('底边搜索：请先连接测径仪(串口)')
                return

            # Require comparator mode (M1,1 / M0,1) to get judge
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

            abs_appr = _f(getattr(self, 'len_z_low_approach_var', 0.0), 0.0)  # AX0 abs
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
            abs_tgt = float(abs_appr)
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
                if abs(float(ac0.act_pos) - abs_tgt) <= max(0.5, tol_z):
                    break
                time.sleep(0.05)

            if stop_evt.is_set():
                ui_msg('底边搜索：已停止')
                return

            if abs(float(self.get_axis_copy(0).act_pos) - abs_tgt) > max(0.8, tol_z * 2.0):
                ui_msg('底边搜索：到达接近位超时')
                return

            # Pre-check: ensure judge exists and is GO at approach
            ui_msg('底边搜索：确认比较器(GO/HI)...')
            last_ts = 0.0
            r = _wait_new_judge(last_ts, 1.5)
            if r is None:
                ui_msg('底边搜索：未收到测径仪数据(请检查串口/请求指令)')
                return
            last_ts, j0 = r
            if j0 == 'UNK':
                ui_msg('底边搜索：未收到比较器(judge)字段，请确认请求为 M1,1')
                return
            if j0 != 'GO':
                ui_msg(f'底边搜索：起点不是GO({j0})，请调整接近位/比较器阈值')
                return

            # ---------- Pass 1: GO -> HI (move +Z_disp) ----------
            ui_msg('底边搜索：第1段(GO→HI)慢速搜索中...')
            z_start = float(self.axis_cal.abs_to_z_disp(0, self.get_axis_copy(0).act_pos))
            vel_abs = float(v_z) * float(self.axis_cal.sign_eff(0))
            self._velmove_start_axis(0, vel_abs, acc=80.0, dec=80.0, jerk=300.0)

            t_search0 = time.time()
            hi_cnt = 0
            unk_cnt = 0
            edge1 = None
            last_go_z = float(z_start)
            last_move_z = None
            last_move_ts = time.time()

            while not stop_evt.is_set():
                ac0 = self.get_axis_copy(0)
                z_cur = float(self.axis_cal.abs_to_z_disp(0, ac0.act_pos))

                if d_max > 0.0 and (z_cur - z_start) >= (d_max - 1e-6):
                    ui_msg('底边搜索：第1段未找到(到达最大距离)')
                    break
                if (time.time() - t_search0) >= timeout_s:
                    ui_msg('底边搜索：第1段未找到(超时)')
                    break
                if int(getattr(ac0, 'err', 0) or 0) != 0:
                    ui_msg(f"底边搜索：AX0错误({int(getattr(ac0,'err',0) or 0)})")
                    break
                if _axis_not_moving_guard(z_cur):
                    ui_msg('底边搜索：AX0未运动(可能到达软限位/未进入速度模式)')
                    break

                r = _wait_new_judge(last_ts, 0.5)
                if r is None:
                    continue
                last_ts, j = r
                if j == 'UNK':
                    unk_cnt += 1
                    if unk_cnt >= 8:
                        ui_msg('底边搜索：未收到比较器(judge)字段，请确认请求为 M1,1')
                        break
                    continue
                unk_cnt = 0

                if j == 'GO':
                    last_go_z = float(z_cur)
                    hi_cnt = 0
                    continue

                if j in ('HI', 'HH'):
                    hi_cnt += 1
                    if hi_cnt >= max(1, int(deb_k)):
                        edge1 = float(last_go_z)
                        ui_msg(f"底边搜索：第1段锁定 {edge1:.3f} (GO→HI)")
                        break
                else:
                    hi_cnt = 0

            # stop motion always
            self._velmove_stop_axis(0)
            # wait axis to settle to avoid 1003 (axis stopping) on next velmove
            try:
                self._wait_axis_stop_settled(0, timeout_s=1.5, stable_cycles=8, eps_abs=0.02, stop_evt=stop_evt)
            except Exception:
                pass
            time.sleep(0.05)

            if stop_evt.is_set():
                ui_msg('底边搜索：已停止')
                return

            if edge1 is None:
                # message already set
                return

            # ---------- Pass 2: HI -> GO (move -Z_disp) ----------
            ui_msg('底边搜索：第2段(HI→GO)回扫中...')
            z_start2 = float(self.axis_cal.abs_to_z_disp(0, self.get_axis_copy(0).act_pos))
            vel_abs2 = -float(v_z) * float(self.axis_cal.sign_eff(0))
            self._velmove_start_axis(0, vel_abs2, acc=80.0, dec=80.0, jerk=300.0)

            t_search1 = time.time()
            go_cnt = 0
            unk_cnt = 0
            edge2 = None
            seen_hi = False
            last_hi_z = float(z_start2)
            last_move_z = None
            last_move_ts = time.time()

            while not stop_evt.is_set():
                ac0 = self.get_axis_copy(0)
                z_cur = float(self.axis_cal.abs_to_z_disp(0, ac0.act_pos))

                if d_max > 0.0 and (z_start2 - z_cur) >= (d_max - 1e-6):
                    ui_msg('底边搜索：第2段未找到(到达最大距离)')
                    break
                if (time.time() - t_search1) >= timeout_s:
                    ui_msg('底边搜索：第2段未找到(超时)')
                    break
                if int(getattr(ac0, 'err', 0) or 0) != 0:
                    ui_msg(f"底边搜索：AX0错误({int(getattr(ac0,'err',0) or 0)})")
                    break
                if _axis_not_moving_guard(z_cur):
                    ui_msg('底边搜索：AX0未运动(可能到达软限位/未进入速度模式)')
                    break

                r = _wait_new_judge(last_ts, 0.5)
                if r is None:
                    continue
                last_ts, j = r
                if j == 'UNK':
                    unk_cnt += 1
                    if unk_cnt >= 8:
                        ui_msg('底边搜索：未收到比较器(judge)字段，请确认请求为 M1,1')
                        break
                    continue
                unk_cnt = 0

                if not seen_hi:
                    if j in ('HI', 'HH'):
                        seen_hi = True
                        last_hi_z = float(z_cur)
                    continue

                if j in ('HI', 'HH'):
                    last_hi_z = float(z_cur)
                    go_cnt = 0
                    continue

                if j == 'GO':
                    go_cnt += 1
                    if go_cnt >= max(1, int(deb_k)):
                        edge2 = float(last_hi_z)
                        ui_msg(f"底边搜索：第2段锁定 {edge2:.3f} (HI→GO)")
                        break
                else:
                    go_cnt = 0

            # stop motion always
            self._velmove_stop_axis(0)
            # wait axis to settle to avoid 1003 (axis stopping) on next velmove
            try:
                self._wait_axis_stop_settled(0, timeout_s=1.5, stable_cycles=8, eps_abs=0.02, stop_evt=stop_evt)
            except Exception:
                pass
            time.sleep(0.05)

            if stop_evt.is_set():
                ui_msg('底边搜索：已停止')
                return

            if edge2 is None:
                return

            # Average
            edge_avg = 0.5 * (float(edge1) + float(edge2))
            found = True
            ui_msg(f"底边搜索：锁定 {edge_avg:.3f} (双向均值)")

            try:
                if hasattr(self, 'len_edge_low_var'):
                    self._ui_set(self.len_edge_low_var, f"{float(edge_avg):.3f}")
                try:
                    self.after(0, self._len_try_update_measured_length)
                except Exception:
                    pass
            except Exception:
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
        """Worker thread: top-edge bidirectional search (GO->HI then HI->GO) and lock AX0 Z_disp.

        机制说明：
        - 需要测径仪返回比较器判定字段(judge)，例如 GO/HI/LO。
        - 顶边判定采用双向扫描：
          1) 沿 -Z_disp 方向检测 GO→HI，锁定“最后一次GO”的 Z_disp 为 edge1；
          2) 反向沿 +Z_disp 方向检测 HI→GO，锁定“最后一次HI”的 Z_disp 为 edge2；
          3) 顶边 = (edge1 + edge2)/2。
        """

        def ui_msg(msg: str) -> None:
            try:
                if hasattr(self, 'len_edge_state_var'):
                    self._ui_set(self.len_edge_state_var, msg)
            except Exception:
                pass

        def ui_done_btn() -> None:
            try:
                btn = self._recipe_ui_widget('btn_len_search_high')
                if btn is not None:
                    self._ui_btn_text(btn, '尝试搜索顶边(GO→HI)')
            except Exception:
                pass

        def _wait_new_judge(ts0: float, tmax: float = 0.8):
            t0 = time.time()
            last_ts = float(ts0)
            while (not stop_evt.is_set()) and ((time.time() - t0) < float(tmax)):
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
                if ts <= last_ts:
                    continue
                j = str(getattr(s, 'judge', 'UNK') or 'UNK').strip().upper()
                return ts, j
            return None

        def _axis_not_moving_guard(z_cur: float):
            nonlocal last_move_z, last_move_ts
            if last_move_z is None:
                last_move_z = float(z_cur)
                last_move_ts = time.time()
                return False
            if abs(float(z_cur) - float(last_move_z)) >= 0.15:
                last_move_z = float(z_cur)
                last_move_ts = time.time()
                return False
            if (time.time() - float(last_move_ts)) >= 1.0:
                return True
            return False

        found = False
        edge_avg = None

        try:
            # --- validations ---
            if bool(getattr(self, 'sim_gauge_enabled', False)) or (hasattr(self, 'sim_gauge_var') and int(self.sim_gauge_var.get() or 0) == 1):
                ui_msg('顶边搜索：模拟测径仪不支持')
                return

            gw = getattr(self, 'gauge_worker', None)
            if gw is None or (not getattr(gw, 'enabled', False)):
                ui_msg('顶边搜索：请先连接测径仪(串口)')
                return

            # Require comparator mode (M1,1 / M0,1) so we can use judge(GO/HI) for edge detection
            try:
                req_cmd = str(getattr(gw, 'request_cmd', '') or '').strip().upper()
            except Exception:
                req_cmd = ''
            if (',1' not in req_cmd) and ('M1,1' not in req_cmd) and ('M0,1' not in req_cmd):
                ui_msg('顶边搜索：请将测径仪请求设为 M1,1 或 M0,1 (需包含比较器字段)')
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

            # Compute approach point for top edge (in Z_disp)
            if pipe_len <= 1e-6:
                ui_msg('顶边搜索：管长(配方)为0')
                return
            z_appr = float(z_low_edge - pipe_len + hi_margin)

            # Clamp to travel limits
            z_min, z_max, _travel = self._get_ax0_z_disp_limits()
            z_appr_clamped = max(float(z_min), min(float(z_max), float(z_appr)))
            if abs(float(z_appr_clamped) - float(z_appr)) > 1e-6:
                # if clamped to limit, we might not have space to scan further
                ui_msg('顶边搜索：接近位被行程限制裁剪，可能导致搜索失败(到限位后超时)')
            z_appr = float(z_appr_clamped)

            # Move to approach
            ui_msg('顶边搜索：移动到接近位...')
            # NOTE: top-edge approach is computed in Z_disp; convert to AX0 absolute position for MoveA.
            abs_tgt = float(self.axis_cal.z_disp_to_abs(0, float(z_appr)))
            self.movea_abs(0, abs_tgt, context='LenEdgeHighAppr')

            t0 = time.time()
            z_now = float(self.axis_cal.abs_to_z_disp(0, self.get_axis_copy(0).act_pos))
            while (not stop_evt.is_set()) and (time.time() - t0 < 15.0):
                ac0 = self.get_axis_copy(0)
                if int(getattr(ac0, 'err', 0) or 0) != 0:
                    ui_msg(f"顶边搜索：AX0错误({int(getattr(ac0,'err',0) or 0)})")
                    return
                z_now = float(self.axis_cal.abs_to_z_disp(0, ac0.act_pos))
                if abs(float(ac0.act_pos) - abs_tgt) <= max(0.5, tol_z):
                    break
                time.sleep(0.05)

            if stop_evt.is_set():
                ui_msg('顶边搜索：已停止')
                return

            if abs(float(self.get_axis_copy(0).act_pos) - abs_tgt) > max(0.8, tol_z * 2.0):
                ui_msg('顶边搜索：到达接近位超时')
                return

            # Pre-check: ensure we can get judge at approach and start in GO
            ui_msg('顶边搜索：确认比较器(GO/HI)...')
            last_ts = 0.0
            r = _wait_new_judge(last_ts, 1.5)
            if r is None:
                ui_msg('顶边搜索：未收到测径仪数据(请检查串口/请求指令)')
                return
            last_ts, j0 = r
            if j0 == 'UNK':
                ui_msg('顶边搜索：未收到比较器(judge)字段，请确认请求为 M1,1')
                return
            if j0 != 'GO':
                ui_msg(f'顶边搜索：起点不是GO({j0})，请调整接近位/比较器阈值')
                return

            # ---------- Pass 1: GO -> HI (move -Z_disp) ----------
            ui_msg('顶边搜索：第1段(GO→HI)慢速搜索中...')
            z_start = float(self.axis_cal.abs_to_z_disp(0, self.get_axis_copy(0).act_pos))
            vel_abs = -float(v_z) * float(self.axis_cal.sign_eff(0))
            self._velmove_start_axis(0, vel_abs, acc=80.0, dec=80.0, jerk=300.0)

            t_search0 = time.time()
            unk_cnt = 0
            hi_cnt = 0
            edge1 = None
            last_go_z = float(z_start)
            last_move_z = None
            last_move_ts = time.time()

            while not stop_evt.is_set():
                ac0 = self.get_axis_copy(0)
                z_cur = float(self.axis_cal.abs_to_z_disp(0, ac0.act_pos))

                if d_max > 0.0 and (z_start - z_cur) >= (d_max - 1e-6):
                    ui_msg('顶边搜索：第1段未找到(到达最大距离)')
                    break
                if (time.time() - t_search0) >= timeout_s:
                    ui_msg('顶边搜索：第1段未找到(超时)')
                    break
                if int(getattr(ac0, 'err', 0) or 0) != 0:
                    ui_msg(f"顶边搜索：AX0错误({int(getattr(ac0,'err',0) or 0)})")
                    break
                if _axis_not_moving_guard(z_cur):
                    ui_msg('顶边搜索：AX0未运动(可能到达软限位/未进入速度模式)')
                    break

                r = _wait_new_judge(last_ts, 0.5)
                if r is None:
                    continue
                last_ts, j = r
                if j == 'UNK':
                    unk_cnt += 1
                    if unk_cnt >= 8:
                        ui_msg('顶边搜索：未收到比较器(judge)字段，请确认请求为 M1,1')
                        break
                    continue
                unk_cnt = 0

                if j == 'GO':
                    last_go_z = float(z_cur)
                    hi_cnt = 0
                    continue

                if j in ('HI', 'HH'):
                    hi_cnt += 1
                    if hi_cnt >= max(1, int(deb_k)):
                        edge1 = float(last_go_z)
                        ui_msg(f"顶边搜索：第1段锁定 {edge1:.3f} (GO→HI)")
                        break
                else:
                    hi_cnt = 0

            self._velmove_stop_axis(0)
            # wait axis to settle to avoid 1003 (axis stopping) on next velmove
            try:
                self._wait_axis_stop_settled(0, timeout_s=1.5, stable_cycles=8, eps_abs=0.02, stop_evt=stop_evt)
            except Exception:
                pass
            time.sleep(0.05)

            if stop_evt.is_set():
                ui_msg('顶边搜索：已停止')
                return

            if edge1 is None:
                return

            # ---------- Pass 2: HI -> GO (move +Z_disp) ----------
            ui_msg('顶边搜索：第2段(HI→GO)回扫中...')
            z_start2 = float(self.axis_cal.abs_to_z_disp(0, self.get_axis_copy(0).act_pos))
            vel_abs2 = float(v_z) * float(self.axis_cal.sign_eff(0))
            self._velmove_start_axis(0, vel_abs2, acc=80.0, dec=80.0, jerk=300.0)

            t_search1 = time.time()
            unk_cnt = 0
            go_cnt = 0
            edge2 = None
            seen_hi = False
            last_hi_z = float(z_start2)
            last_move_z = None
            last_move_ts = time.time()

            while not stop_evt.is_set():
                ac0 = self.get_axis_copy(0)
                z_cur = float(self.axis_cal.abs_to_z_disp(0, ac0.act_pos))

                if d_max > 0.0 and (z_cur - z_start2) >= (d_max - 1e-6):
                    ui_msg('顶边搜索：第2段未找到(到达最大距离)')
                    break
                if (time.time() - t_search1) >= timeout_s:
                    ui_msg('顶边搜索：第2段未找到(超时)')
                    break
                if int(getattr(ac0, 'err', 0) or 0) != 0:
                    ui_msg(f"顶边搜索：AX0错误({int(getattr(ac0,'err',0) or 0)})")
                    break
                if _axis_not_moving_guard(z_cur):
                    ui_msg('顶边搜索：AX0未运动(可能到达软限位/未进入速度模式)')
                    break

                r = _wait_new_judge(last_ts, 0.5)
                if r is None:
                    continue
                last_ts, j = r
                if j == 'UNK':
                    unk_cnt += 1
                    if unk_cnt >= 8:
                        ui_msg('顶边搜索：未收到比较器(judge)字段，请确认请求为 M1,1')
                        break
                    continue
                unk_cnt = 0

                if not seen_hi:
                    if j in ('HI', 'HH'):
                        seen_hi = True
                        last_hi_z = float(z_cur)
                    continue

                if j in ('HI', 'HH'):
                    last_hi_z = float(z_cur)
                    go_cnt = 0
                    continue

                if j == 'GO':
                    go_cnt += 1
                    if go_cnt >= max(1, int(deb_k)):
                        edge2 = float(last_hi_z)
                        ui_msg(f"顶边搜索：第2段锁定 {edge2:.3f} (HI→GO)")
                        break
                else:
                    go_cnt = 0

            self._velmove_stop_axis(0)
            # wait axis to settle to avoid 1003 (axis stopping) on next velmove
            try:
                self._wait_axis_stop_settled(0, timeout_s=1.5, stable_cycles=8, eps_abs=0.02, stop_evt=stop_evt)
            except Exception:
                pass
            time.sleep(0.05)

            if stop_evt.is_set():
                ui_msg('顶边搜索：已停止')
                return

            if edge2 is None:
                return

            edge_avg = 0.5 * (float(edge1) + float(edge2))
            found = True
            ui_msg(f"顶边搜索：锁定 {edge_avg:.3f} (双向均值)")

            try:
                if hasattr(self, 'len_edge_high_var'):
                    self._ui_set(self.len_edge_high_var, f"{float(edge_avg):.3f}")
            except Exception:
                pass

            # Optional backoff to stay inside the tube (towards +Z_disp)
            if backoff_mm > 1e-6:
                try:
                    z_back = max(float(z_min), min(float(z_max), float(edge_avg) + float(backoff_mm)))
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

        # standby/start display
        try:
            self._refresh_standby_pos()
        except Exception:
            pass
        try:
            self._refresh_start_pos()
        except Exception:
            pass

    # =========================
    # Auto tab
    # =========================

    def _refresh_auto_std_panel(self):
        r = self.recipe
        # UI: keep the main-screen standard value concise; tolerance is shown/edited in recipe screen.
        lbl_od_std = self._main_ui_widget('lbl_od_std')
        lbl_id_std = self._main_ui_widget('lbl_id_std')
        if lbl_od_std is not None:
            lbl_od_std.config(text=f"{r.od_std_mm:.3f} mm")
        if lbl_id_std is not None:
            lbl_id_std.config(text=f"{r.id_std_mm:.3f} mm")

        # Apply main-screen UI mode (SYNC/SPLIT/OD_ONLY)
        try:
            self._apply_main_ui_mode()
        except Exception:
            pass

    def _ui_get_meas_mode(self) -> str:
        """Return one of: 'SYNC', 'SPLIT', 'OD_ONLY', 'ID_SINGLE', 'SPLIT_SINGLE'."""
        r = getattr(self, 'recipe', None)
        try:
            sm = str(
                getattr(
                    r,
                    'section_sampling_mode',
                    getattr(r, 'scan_mode', 'sync'),
                )
                or 'sync'
            ).strip().lower()
        except Exception:
            sm = 'sync'
        try:
            if bool(getattr(r, 'id_single_enable', False)):
                return 'SPLIT_SINGLE' if sm.startswith('split') else 'ID_SINGLE'
        except Exception:
            pass
        if sm.startswith('split'):
            return 'SPLIT'
        # OD-only / speedtest: disable ID Modbus reads (recipe)
        try:
            if bool(getattr(r, 'disable_id_modbus', False)):
                return 'OD_ONLY'
        except Exception:
            pass
        return 'SYNC'

    def _resize_result_tree_columns(self, tree: Any, columns: tuple[str, ...]) -> None:
        if tree is None or not columns:
            return
        try:
            preferred = dict(self._main_view_state('tree_column_widths', {}) or {})
            minimums = dict(self._main_view_state('tree_column_min_widths', {}) or {})
        except Exception:
            preferred, minimums = {}, {}

        pref: dict[str, int] = {}
        mins: dict[str, int] = {}
        for col in columns:
            try:
                current = int(tree.column(col, 'width') or 0)
            except Exception:
                current = 0
            pref[col] = max(1, int(preferred.get(col, current or 100) or 100))
            mins[col] = max(1, int(minimums.get(col, min(pref[col], 70)) or 70))
            if pref[col] < mins[col]:
                pref[col] = mins[col]

        try:
            available = max(0, int(tree.winfo_width() or 0) - 4)
        except Exception:
            available = 0

        total_pref = sum(pref.values())
        total_min = sum(mins.values())
        widths = dict(pref)

        if available > 0 and available < total_pref and total_pref > total_min:
            shrink_total = min(total_pref - available, total_pref - total_min)
            shrinkable = {col: max(0, pref[col] - mins[col]) for col in columns}
            shrink_base = sum(shrinkable.values())
            used = 0
            for col in columns:
                if shrink_base <= 0:
                    break
                shrink = int(round(shrink_total * shrinkable[col] / shrink_base))
                shrink = min(shrink, shrinkable[col])
                widths[col] = pref[col] - shrink
                used += shrink
            remainder = shrink_total - used
            for col in reversed(columns):
                if remainder <= 0:
                    break
                room = max(0, widths[col] - mins[col])
                if room <= 0:
                    continue
                delta = min(room, remainder)
                widths[col] -= delta
                remainder -= delta
        elif available > total_pref:
            extra = available - total_pref
            weights = {
                'x_ui': 2,
                'od_fit_res': 1,
                'od_pp_rob': 1,
                'id_round': 1,
                'concentricity': 1,
                'cov_reason': 2,
            }
            active_weights = {col: weights.get(col, 0) for col in columns}
            weight_sum = sum(active_weights.values())
            if weight_sum <= 0:
                active_weights = {col: 1 for col in columns}
                weight_sum = len(columns)
            used = 0
            for col in columns:
                add = int(extra * active_weights[col] / weight_sum)
                widths[col] = pref[col] + add
                used += add
            for col in columns:
                if used >= extra:
                    break
                widths[col] += 1
                used += 1

        for col in columns:
            try:
                tree.column(col, width=max(mins[col], int(widths[col])), minwidth=mins[col], stretch=False)
            except Exception:
                pass

    def _schedule_result_tree_column_resize(self, tree: Any, columns: tuple[str, ...]) -> None:
        try:
            self.after(0, lambda: self._resize_result_tree_columns(tree, tuple(columns)))
        except Exception:
            self._resize_result_tree_columns(tree, tuple(columns))

    def _apply_main_ui_mode(self) -> None:
        """Adjust main-screen widgets by measurement mode."""
        mode = self._ui_get_meas_mode()
        # Status line
        try:
            if mode == 'OD_ONLY':
                self.ui_meas_mode_var.set('检测模式：仅外径（OD Only）')
            elif mode in ('SPLIT', 'SPLIT_SINGLE'):
                if mode == 'SPLIT_SINGLE':
                    self.ui_meas_mode_var.set('检测模式：分圈（ID单探头）')
                else:
                    self.ui_meas_mode_var.set('检测模式：分圈（OD→ID）')
            elif mode == 'ID_SINGLE':
                self.ui_meas_mode_var.set('检测模式：同步（ID单探头）')
            else:
                self.ui_meas_mode_var.set('检测模式：同步（OD+ID）')
        except Exception:
            pass

        # Treeview displaycolumns presets (stored by main_screen.build)
        try:
            tree = self._main_ui_widget('result_tree')
            sync_cols = self._main_view_state('tree_displaycols_sync')
            split_cols = self._main_view_state('tree_displaycols_split')
            od_only_cols = self._main_view_state('tree_displaycols_od_only')
            if tree is not None and sync_cols:
                if mode == 'OD_ONLY':
                    active_cols = tuple(od_only_cols or ())
                elif mode in ('SPLIT', 'SPLIT_SINGLE'):
                    active_cols = tuple(split_cols or ())
                else:
                    active_cols = tuple(sync_cols or ())
                tree.configure(displaycolumns=active_cols)
                self._schedule_result_tree_column_resize(tree, active_cols)
        except Exception:
            pass

        # Summary panel placeholders: OD_ONLY disables ID/cross fields
        if mode == 'OD_ONLY':
            try:
                self.max_id_dev_var.set('--（OD Only）')
                self.max_id_round_var.set('--（OD Only）')
                self.id_mean_var.set('--（OD Only）')
                self.id_dpp_var.set('--（OD Only）')
                self.id_range_var.set('--（OD Only）')
                self.id_slope_var.set('--（OD Only）')
                self.id_tilt_var.set('--（OD Only）')
                self.id_endoff_var.set('--（OD Only）')
            except Exception:
                pass
            try:
                self.axis_dist_var.set('--（需要ID）')
                self.conc_max_var.set('--（需要ID）')
                self.axis_span_max_var.set('--（需要ID）')
            except Exception:
                pass

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
        combo = self._gauge_ui_widget('port_combo')
        if combo is None:
            return
        combo.configure(values=ports)

        cur = (combo.get() or "").strip()

        if not ports:
            combo.set(DEFAULT_GAUGE_PORT)
            return

        if cur and (cur in ports):
            combo.set(cur)
            return

        if DEFAULT_GAUGE_PORT in ports:
            combo.set(DEFAULT_GAUGE_PORT)
            return

        combo.set(ports[0])

    def _gauge_connect(self):
        """连接测径仪（只在需要时打开串口）。
        说明：
        - 重复点击“连接”不会重复 open 串口，只会更新参数（避免 Windows 下 PermissionError(13)）。
        - 会自动关闭“模拟测径仪”开关。
        """
        try:
            # if serial is None:
            #    raise RuntimeError("pyserial 未安装。")

            combo = self._gauge_ui_widget('port_combo')
            port = (combo.get().strip() if combo is not None else '') or DEFAULT_GAUGE_PORT
            baud = int(self.baud_var.get().strip() or "115200")
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
                baud=115200,
                timeout_s=0.5,
                eol="\r",
                request_cmd="",
            )
            self.gauge_conn_var.set("串口: 未连接")
            self.gauge_err_var.set("已断开")
        except Exception:
            self.gauge_conn_var.set("串口: 未连接")

    def set_gauge_request_command(self, cmd: str) -> str:
        norm = str(cmd or 'M1,1').strip() or 'M1,1'
        try:
            if getattr(self, 'gauge_worker', None) is not None:
                self.gauge_worker.request_cmd = norm
        except Exception:
            pass
        return norm

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
                self.set_gauge_request_command(cmd)
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




    def _odcal_deg_from_point(self, pt: dict) -> Optional[int]:
        """Return degree bin index [0..359] for a sample point.

        Priority:
        - theta_rel (one_rev 进度) -> 0..360+
        - theta (AX3 实际角度)   -> 任意，取 mod 360
        """
        try:
            th_rel = pt.get("theta_rel", None)
            if th_rel is not None:
                return int(math.floor(float(th_rel))) % 360
        except Exception:
            pass
        try:
            th = pt.get("theta", None)
            if th is not None:
                return int(math.floor(float(th))) % 360
        except Exception:
            pass
        return None

    def _odcal_bins_median(self, degs: list[int], vals: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Median per 1° bin. Returns (bin_vals[360], has_data[360])."""
        bins = [[] for _ in range(360)]
        for d, v in zip(degs, vals):
            try:
                bins[int(d) % 360].append(float(v))
            except Exception:
                pass
        bin_vals = np.full((360,), np.nan, dtype=float)
        has = np.zeros((360,), dtype=bool)
        for i in range(360):
            if bins[i]:
                arr = np.array(bins[i], dtype=float)
                bin_vals[i] = float(np.median(arr))
                has[i] = True
        return bin_vals, has

    def _odcal_fit_harmonics(self, bin_vals: np.ndarray, has: np.ndarray, order: int = 3) -> np.ndarray:
        """Fit low-order harmonic model to bin_vals on bins with data.

        Model: y = a0 + Σ (ak cos(kθ) + bk sin(kθ)), k=1..order
        """
        try:
            idx = np.where(has)[0]
            if idx.size < max(8, 2 * order + 3):
                # too few points, fallback flat
                m = float(np.nanmedian(bin_vals)) if np.isfinite(np.nanmedian(bin_vals)) else 0.0
                return np.full((360,), m, dtype=float)
            th = np.deg2rad(idx.astype(float))
            cols = [np.ones_like(th)]
            for k in range(1, int(order) + 1):
                cols.append(np.cos(k * th))
                cols.append(np.sin(k * th))
            A = np.stack(cols, axis=1)  # (n, m)
            y = bin_vals[idx]
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            # predict all 360
            th_all = np.deg2rad(np.arange(360, dtype=float))
            cols_all = [np.ones_like(th_all)]
            for k in range(1, int(order) + 1):
                cols_all.append(np.cos(k * th_all))
                cols_all.append(np.sin(k * th_all))
            A_all = np.stack(cols_all, axis=1)
            yhat = A_all @ coef
            return yhat.astype(float)
        except Exception:
            m = float(np.nanmedian(bin_vals)) if np.isfinite(np.nanmedian(bin_vals)) else 0.0
            return np.full((360,), m, dtype=float)

    def _odcal_residual_bins(self, degs: list[int], sums: list[float], order: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (s_bin, r_bin, has)."""
        s_bin, has = self._odcal_bins_median(degs, sums)
        yhat = self._odcal_fit_harmonics(s_bin, has, order=order)
        r_bin = np.full((360,), np.nan, dtype=float)
        try:
            idx = np.where(has)[0]
            r_bin[idx] = s_bin[idx] - yhat[idx]
        except Exception:
            pass
        return s_bin, r_bin, has

    def _odcal_mask_to_ranges(self, mask: list[int]) -> list[tuple[int, int]]:
        """Convert 0/1 mask[360] to circular ranges (inclusive)."""
        if not mask or len(mask) != 360:
            return []
        m = [1 if int(x) else 0 for x in mask]
        if sum(m) <= 0:
            return []

        # find runs on [0..359]
        runs: list[tuple[int, int]] = []
        i = 0
        while i < 360:
            if m[i]:
                j = i
                while j + 1 < 360 and m[j + 1]:
                    j += 1
                runs.append((i, j))
                i = j + 1
            else:
                i += 1

        # merge wrap if needed (end + start)
        if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == 359:
            a0, b0 = runs[0]
            a1, b1 = runs[-1]
            runs = [(a1, b0)] + runs[1:-1]

        # normalize to [0..359], keep inclusive
        out = []
        for a, b in runs:
            out.append((int(a) % 360, int(b) % 360))
        return out

    def _odcal_ranges_to_mask(self, ranges: list[tuple[int, int]]) -> list[int]:
        m = [0] * 360
        for a, b in ranges or []:
            try:
                a = int(a) % 360
                b = int(b) % 360
            except Exception:
                continue
            if a <= b:
                for d in range(a, b + 1):
                    m[d] = 1
            else:
                for d in range(a, 360):
                    m[d] = 1
                for d in range(0, b + 1):
                    m[d] = 1
        return m

    def _odcal_ranges_str(self, ranges: list[tuple[int, int]]) -> str:
        if not ranges:
            return "--"
        parts = []
        for a, b in ranges:
            if a == b:
                parts.append(f"{a}°")
            else:
                parts.append(f"{a}~{b}°")
        return ", ".join(parts)

    def _odcal_shift_mask(self, mask: list[int], shift: int) -> list[int]:
        """Rotate mask into current-run coordinate: out[j] = mask[(j-shift) mod 360]."""
        if not mask or len(mask) != 360:
            return [0] * 360
        s = int(shift) % 360
        out = [0] * 360
        for j in range(360):
            out[j] = 1 if int(mask[(j - s) % 360]) else 0
        return out

    def _odcal_detect_defect_mask(
        self,
        r_bin: np.ndarray,
        has: np.ndarray,
        abs_thr: float = 0.010,
        k_sigma: float = 5.5,
        gap1_close: bool = True,
        min_len: int = 2,
        pad: int = 1,
        top_n: Optional[int] = None,
    ) -> tuple[list[int], dict]:
        """Detect negative dents in residual curve and return mask[360].

        Returns (mask, debug).
        """
        dbg = {"abs_thr": float(abs_thr), "k_sigma": float(k_sigma)}
        try:
            idx = np.where(has & np.isfinite(r_bin))[0]
            if idx.size < 16:
                return [0] * 360, {"reason": "insufficient_bins", **dbg}

            rv = r_bin[idx].astype(float)
            med = float(np.median(rv))
            mad = float(np.median(np.abs(rv - med)))
            sigma = 1.4826 * mad
            thr = max(float(abs_thr), float(k_sigma) * float(sigma))
            dbg.update({"sigma_mad": float(sigma), "thr": float(thr)})

            cand = np.zeros((360,), dtype=bool)
            cand[idx] = (rv < (-thr))

            # close single-bin gaps: 1 0 1 -> 1 1 1
            if gap1_close:
                filled = cand.copy()
                for i in range(360):
                    if (not cand[i]) and cand[(i - 1) % 360] and cand[(i + 1) % 360]:
                        filled[i] = True
                cand = filled

            # find segments
            segs: list[tuple[int, int]] = []
            i = 0
            while i < 360:
                if cand[i]:
                    j = i
                    while j + 1 < 360 and cand[j + 1]:
                        j += 1
                    segs.append((i, j))
                    i = j + 1
                else:
                    i += 1
            # wrap merge
            if len(segs) >= 2 and segs[0][0] == 0 and segs[-1][1] == 359:
                a0, b0 = segs[0]
                a1, b1 = segs[-1]
                segs = [(a1, b0)] + segs[1:-1]

            # filter by min_len and score
            scored = []
            for a, b in segs:
                # length inclusive
                if a <= b:
                    degs = list(range(a, b + 1))
                else:
                    degs = list(range(a, 360)) + list(range(0, b + 1))
                if len(degs) < int(min_len):
                    continue
                # score: sum of negative depth beyond -thr
                sc = 0.0
                for d in degs:
                    if has[d] and np.isfinite(r_bin[d]):
                        sc += max(0.0, (-float(r_bin[d]) - thr))
                scored.append((sc, a, b, len(degs)))

            scored.sort(reverse=True, key=lambda x: x[0])
            if top_n is not None and top_n > 0:
                scored = scored[: int(top_n)]

            mask = np.zeros((360,), dtype=bool)
            kept_segs = []
            for sc, a, b, ln in scored:
                kept_segs.append((int(a), int(b), float(sc), int(ln)))
                # apply with padding
                if a <= b:
                    for d in range(a - pad, b + pad + 1):
                        mask[d % 360] = True
                else:
                    for d in list(range(a - pad, 360 + pad)) + list(range(0, b + pad + 1)):
                        mask[d % 360] = True

            dbg["segments"] = kept_segs
            return [1 if x else 0 for x in mask.tolist()], dbg
        except Exception as e:
            return [0] * 360, {"reason": f"exception:{e}", **dbg}

    def _odcal_best_shift_template(self, template_mask: list[int], r_bin: np.ndarray, has: np.ndarray) -> tuple[Optional[int], dict]:
        """Find circular shift that best aligns template dent positions to current residual."""
        dbg = {}
        if (not template_mask) or (len(template_mask) != 360) or (sum(int(x) for x in template_mask) <= 0):
            return None, {"reason": "no_template"}
        try:
            tpl_idx = [i for i, x in enumerate(template_mask) if int(x)]
            if len(tpl_idx) < 2:
                return None, {"reason": "template_too_small"}
            best_s = 0
            best_score = -1e30
            best_n = 0
            for s in range(360):
                sc = 0.0
                n = 0
                for i in tpl_idx:
                    j = (i + s) % 360
                    if bool(has[j]) and np.isfinite(r_bin[j]):
                        sc += (-float(r_bin[j]))  # prefer more negative
                        n += 1
                if n > 0:
                    sc = sc / n
                if sc > best_score:
                    best_score = sc
                    best_s = s
                    best_n = n
            dbg.update({"score": float(best_score), "n": int(best_n)})
            return int(best_s), dbg
        except Exception as e:
            return None, {"reason": f"exception:{e}"}

    def _odcal_best_shift_by_overlap(self, mask_a: list[int], mask_b: list[int]) -> tuple[Optional[int], dict]:
        """Find shift s maximizing overlap Σ a[i]*b[i+s]."""
        if (not mask_a) or (not mask_b) or (len(mask_a) != 360) or (len(mask_b) != 360):
            return None, {"reason": "bad_mask"}
        if sum(int(x) for x in mask_a) <= 0 or sum(int(x) for x in mask_b) <= 0:
            return None, {"reason": "empty_mask"}
        best_s = 0
        best_ov = -1
        for s in range(360):
            ov = 0
            for i in range(360):
                if int(mask_a[i]) and int(mask_b[(i + s) % 360]):
                    ov += 1
            if ov > best_ov:
                best_ov = ov
                best_s = s
        return int(best_s), {"overlap_bins": int(best_ov)}

    def _odcal_prepare_sums(self) -> tuple[list[float], dict]:
        """Prepare lL+lR list for B computation.

        Pipeline:
        1) raw sum series from points (optionally mapped OUT1/OUT2)
        2) optional dent masking (TEMPLATE aligned, or DYNAMIC if enabled and no template)
        3) optional median filter on time series
        4) optional outlier removal by sigma
        """
        meta: dict = {
            "defect_mode": "OFF",
            "defect_shift": None,
            "defect_masked": 0,
            "defect_debug": {},
        }

        pts = list(getattr(self, "_odcal_points", []) or [])
        if not pts:
            return [], meta

        # map OUT1/OUT2 to L/R
        out1_map = (self.odcal_map_out1_var.get() or "L").strip().upper()
        map_swap = (out1_map == "R")

        sums_raw: list[float] = []
        degs_raw: list[int] = []
        deg_missing = 0

        for pt in pts:
            try:
                v1 = float(pt.get("v1", 0.0))
                v2 = float(pt.get("v2", 0.0))
            except Exception:
                continue

            # swap mapping if OUT1->R
            if map_swap:
                v1, v2 = v2, v1

            sums_raw.append(v1 + v2)

            d = self._odcal_deg_from_point(pt)
            if d is None:
                deg_missing += 1
                degs_raw.append(0)
            else:
                degs_raw.append(int(d) % 360)

        # ------------------------------
        # 2) Dent masking
        # ------------------------------
        have_angle = (deg_missing == 0) and (len(degs_raw) >= 16)
        template_loaded = bool(getattr(self, "_odcal_defect_template_mask", None)) and (sum(int(x) for x in self._odcal_defect_template_mask) > 0)

        # default UI: show template ranges when idle
        try:
            if template_loaded and (self.odcal_defects_var.get() in ("--", "", None)):
                self.odcal_defects_var.set("模板: " + self._odcal_ranges_str(self._odcal_mask_to_ranges(self._odcal_defect_template_mask)))
        except Exception:
            pass

        masked_idx = set()

        if have_angle and template_loaded:
            try:
                _, r_bin, has = self._odcal_residual_bins(degs_raw, sums_raw, order=3)
                shift, sdbg = self._odcal_best_shift_template(self._odcal_defect_template_mask, r_bin, has)
                meta["defect_debug"] = {"align": sdbg}
                if shift is not None:
                    run_mask = self._odcal_shift_mask(self._odcal_defect_template_mask, shift)
                    for i, d in enumerate(degs_raw):
                        if int(run_mask[int(d) % 360]):
                            masked_idx.add(i)
                    meta.update({"defect_mode": "TEMPLATE", "defect_shift": int(shift), "defect_masked": int(len(masked_idx))})
                    # UI hints
                    try:
                        self.odcal_defect_mode_var.set("TEMPLATE")
                        self.odcal_defect_shift_var.set(f"{int(shift)}°")
                        self.odcal_defects_var.set("屏蔽: " + self._odcal_ranges_str(self._odcal_mask_to_ranges(run_mask)))
                    except Exception:
                        pass
            except Exception as e:
                meta["defect_debug"] = {"reason": f"template_exception:{e}"}

        # dynamic fallback (only when no template)
        if have_angle and (not template_loaded):
            try:
                dyn_en = int(getattr(self, "odcal_defect_dyn_enable_var", tk.IntVar(value=0)).get() or 0)
            except Exception:
                dyn_en = 0
            if dyn_en:
                try:
                    _, r_bin, has = self._odcal_residual_bins(degs_raw, sums_raw, order=3)
                    dyn_mask, ddbg = self._odcal_detect_defect_mask(r_bin, has, top_n=1)
                    meta["defect_debug"] = {"dynamic": ddbg}
                    if sum(int(x) for x in dyn_mask) > 0:
                        for i, d in enumerate(degs_raw):
                            if int(dyn_mask[int(d) % 360]):
                                masked_idx.add(i)
                        meta.update({"defect_mode": "DYNAMIC", "defect_shift": None, "defect_masked": int(len(masked_idx))})
                        try:
                            self.odcal_defect_mode_var.set("DYNAMIC")
                            self.odcal_defect_shift_var.set("--")
                            self.odcal_defects_var.set("屏蔽: " + self._odcal_ranges_str(self._odcal_mask_to_ranges(dyn_mask)))
                        except Exception:
                            pass
                except Exception as e:
                    meta["defect_debug"] = {"reason": f"dynamic_exception:{e}"}

        # apply masking to time series
        if masked_idx:
            sums1 = [s for i, s in enumerate(sums_raw) if i not in masked_idx]
        else:
            sums1 = list(sums_raw)

        # ------------------------------
        # 3) optional median filter on time series
        # ------------------------------
        try:
            mode = str(getattr(self, "odcal_filter_var", None).get() if hasattr(self, "odcal_filter_var") else "无")
        except Exception:
            mode = "无"

        if mode.startswith("中值"):
            try:
                win = 3 if "3" in mode else 5
                if win >= 3 and len(sums1) >= win:
                    arr = np.array(sums1, dtype=float)
                    out = []
                    half = win // 2
                    for i in range(len(arr)):
                        a = max(0, i - half)
                        b = min(len(arr), i + half + 1)
                        out.append(float(np.median(arr[a:b])))
                    sums1 = out
            except Exception:
                pass

        # ------------------------------
        # 4) outlier removal (sigma)
        # ------------------------------
        try:
            sig = float(getattr(self, "odcal_outlier_sigma_var", None).get() if hasattr(self, "odcal_outlier_sigma_var") else 0.0)
        except Exception:
            sig = 0.0

        sums2 = sums1
        if sig and sig > 0 and len(sums1) >= 8:
            try:
                arr = np.array(sums1, dtype=float)
                m = float(np.mean(arr))
                sd = float(np.std(arr))
                if sd > 1e-12:
                    keep = np.abs(arr - m) <= (sig * sd)
                    sums2 = [float(v) for v, k in zip(arr.tolist(), keep.tolist()) if k]
            except Exception:
                sums2 = sums1

        # finalize OFF mode UI if no masking
        if not masked_idx:
            try:
                if template_loaded:
                    self.odcal_defect_mode_var.set("TEMPLATE")
                    self.odcal_defects_var.set("模板: " + self._odcal_ranges_str(self._odcal_mask_to_ranges(self._odcal_defect_template_mask)))
                else:
                    self.odcal_defect_mode_var.set("OFF")
                    self.odcal_defect_shift_var.set("--")
                    # keep last shown, but if empty show --
                    if (self.odcal_defects_var.get() or "").strip() == "":
                        self.odcal_defects_var.set("--")
            except Exception:
                pass

        return sums2, meta





    def _odcal_defect_learn_A(self):
        """Record run-A residual/mask as a learning baseline."""
        try:
            if not getattr(self, "_odcal_points", None):
                self.odcal_msg_var.set("学习A：无采样数据")
                return
            # require angle
            degs = []
            sums = []
            miss = 0
            out1_map = (self.odcal_map_out1_var.get() or "L").strip().upper()
            swap = (out1_map == "R")
            for pt in self._odcal_points:
                d = self._odcal_deg_from_point(pt)
                if d is None:
                    miss += 1
                    continue
                try:
                    v1 = float(pt.get("v1", 0.0))
                    v2 = float(pt.get("v2", 0.0))
                except Exception:
                    continue
                if swap:
                    v1, v2 = v2, v1
                degs.append(int(d) % 360)
                sums.append(v1 + v2)

            if miss > 0 or len(degs) < 64:
                self.odcal_msg_var.set("学习A：需要一圈角度数据（建议 one_rev + 角度=AX3）")
                return

            _, r_bin, has = self._odcal_residual_bins(degs, sums, order=3)
            mask, dbg = self._odcal_detect_defect_mask(r_bin, has, top_n=None)
            if sum(int(x) for x in mask) <= 0:
                self.odcal_msg_var.set("学习A：未检测到明显凹陷（阈值过严或数据不足）")
                return

            self._odcal_defect_learn_A_data = {
                "r_bin": r_bin.tolist(),
                "has": has.tolist(),
                "mask": mask,
                "dbg": dbg,
                "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            self.odcal_defect_mode_var.set("LEARN_A")
            self.odcal_defect_shift_var.set("--")
            self.odcal_defects_var.set("A: " + self._odcal_ranges_str(self._odcal_mask_to_ranges(mask)))
            self.odcal_msg_var.set("学习A：已记录。请转动/翻转环规后再采集，点击“学习B(生成表)”")
        except Exception as e:
            self.odcal_msg_var.set(f"学习A失败: {e}")

    def _odcal_defect_learn_B(self):
        """Use current run as B, align to A, and generate a stable template mask."""
        try:
            A = getattr(self, "_odcal_defect_learn_A_data", None)
            if not A:
                self.odcal_msg_var.set("学习B：请先完成学习A")
                return
            if not getattr(self, "_odcal_points", None):
                self.odcal_msg_var.set("学习B：无采样数据")
                return

            # current (B)
            degs = []
            sums = []
            miss = 0
            out1_map = (self.odcal_map_out1_var.get() or "L").strip().upper()
            swap = (out1_map == "R")
            for pt in self._odcal_points:
                d = self._odcal_deg_from_point(pt)
                if d is None:
                    miss += 1
                    continue
                try:
                    v1 = float(pt.get("v1", 0.0))
                    v2 = float(pt.get("v2", 0.0))
                except Exception:
                    continue
                if swap:
                    v1, v2 = v2, v1
                degs.append(int(d) % 360)
                sums.append(v1 + v2)

            if miss > 0 or len(degs) < 64:
                self.odcal_msg_var.set("学习B：需要一圈角度数据（建议 one_rev + 角度=AX3）")
                return

            _, rB, hasB = self._odcal_residual_bins(degs, sums, order=3)
            maskB, dbgB = self._odcal_detect_defect_mask(rB, hasB, top_n=None)
            if sum(int(x) for x in maskB) <= 0:
                self.odcal_msg_var.set("学习B：未检测到明显凹陷（阈值过严或数据不足）")
                return

            maskA = list(A.get("mask") or [0] * 360)
            if len(maskA) != 360:
                self.odcal_msg_var.set("学习B：A 数据异常")
                return

            # align B to A by overlap
            shift_ab, sdbg = self._odcal_best_shift_by_overlap(maskA, maskB)
            if shift_ab is None:
                self.odcal_msg_var.set("学习B：对齐失败（A/B 凹陷段过少）")
                return

            # build mean residual in A-frame and re-detect on mean (more robust than intersection)
            rA = np.array(A.get("r_bin") or [np.nan] * 360, dtype=float)
            hasA = np.array(A.get("has") or [False] * 360, dtype=bool)

            rB_shift = np.full((360,), np.nan, dtype=float)
            hasB_shift = np.zeros((360,), dtype=bool)
            for i in range(360):
                j = (i + int(shift_ab)) % 360
                rB_shift[i] = rB[j] if np.isfinite(rB[j]) else np.nan
                hasB_shift[i] = bool(hasB[j])

            # mean
            rM = np.full((360,), np.nan, dtype=float)
            hasM = hasA | hasB_shift
            for i in range(360):
                if not hasM[i]:
                    continue
                vals = []
                if bool(hasA[i]) and np.isfinite(rA[i]):
                    vals.append(float(rA[i]))
                if bool(hasB_shift[i]) and np.isfinite(rB_shift[i]):
                    vals.append(float(rB_shift[i]))
                if vals:
                    rM[i] = float(sum(vals) / len(vals))

            maskT, dbgT = self._odcal_detect_defect_mask(rM, hasM, top_n=None)
            if sum(int(x) for x in maskT) <= 0:
                # fallback: intersection after alignment
                maskB_inA = [int(maskB[(i + int(shift_ab)) % 360]) for i in range(360)]
                maskT = [1 if (int(maskA[i]) and int(maskB_inA[i])) else 0 for i in range(360)]
                dbgT = {"fallback": "intersection"}

            # persist template
            self._odcal_defect_template_mask = [1 if int(x) else 0 for x in maskT]
            ranges = self._odcal_mask_to_ranges(self._odcal_defect_template_mask)

            # update calibration json (preserve existing fields)
            data = self.calibration_repository.load_od_active()
            data["defects"] = {
                "template_mask": list(self._odcal_defect_template_mask),
                "template_ranges": [[a, b] for a, b in ranges],
                "learn_shift_ab_deg": int(shift_ab),
                "learned_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "learn_dbg": {"A": A.get("dbg", {}), "B": dbgB, "align": sdbg, "template": dbgT},
            }
            self._odcal_save_active(data)

            # clear A buffer
            self._odcal_defect_learn_A_data = None

            self.odcal_defect_mode_var.set("TEMPLATE")
            self.odcal_defect_shift_var.set("--")
            self.odcal_defects_var.set("模板: " + self._odcal_ranges_str(ranges))
            self.odcal_msg_var.set(f"凹陷表已生成：{self._odcal_ranges_str(ranges)}（A<-B shift={int(shift_ab)}°）")
        except Exception as e:
            self.odcal_msg_var.set(f"学习B失败: {e}")

    def _odcal_defect_clear_template(self):
        """Remove persisted defect template."""
        try:
            self._odcal_defect_template_mask = [0] * 360
            self._odcal_defect_learn_A_data = None

            data = self.calibration_repository.load_od_active()
            if "defects" in data:
                data.pop("defects", None)
            self._odcal_save_active(data)

            self.odcal_defect_mode_var.set("OFF")
            self.odcal_defect_shift_var.set("--")
            self.odcal_defects_var.set("--")
            self.odcal_msg_var.set("已清除凹陷表")
        except Exception as e:
            self.odcal_msg_var.set(f"清除失败: {e}")




    def _odcal_on_gauge_sample(self, payload: dict):
        return self.calibration_service.on_od_gauge_sample(self, payload)

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
    def _make_auto_runner(self) -> AutoFlowOrchestrator:
        self.runtime_state.sync_from_run_session(self._run_session)
        return AutoFlowOrchestrator(
            gateway=AppDeviceGateway(self),
            recipe=self.get_recipe_copy(),
            calibration=self.get_calibration_snapshot(),
            run_session=self._run_session,
            event_sink=WorkflowUiEventAdapter(self.ui_q),
            runtime_state=self.runtime_state,
            run_repository=self._make_run_repository(),
        )

    def _start_measurement_impl(self):
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
            self._auto_thread = self._make_auto_runner()
            self._log_ax3_speed_trace("auto_start_before_autoflow_start")
            self._auto_thread.start()
        except Exception as e:
            messagebox.showerror("启动失败", str(e))


    def _stop_measurement_impl(self):
        try:
            log("AUTO_STOP")
            if self._auto_thread and self._auto_thread.is_alive():
                self._auto_thread.stop()
                self.set_plc_poll_profile("normal", caller="stop_measurement")
                # Immediately stop axis motions on PLC side to avoid "in-position timeout" -> ERR.
                self.abort_motion()
        except Exception:
            pass


    def _auto_clear_ui(self, preserve_run: bool = False):
        tree = self._main_ui_widget('result_tree')
        if tree is not None:
            tree.delete(*tree.get_children())
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

        # Apply UI mode after clearing (affects columns/placeholder text)
        try:
            self._apply_main_ui_mode()
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
        # f2 OD diagnostics
        self._max_od_pp = None
        self._max_od_pp_rob = None
        self._max_od_fit_res = None
        try:
            self.max_od_dev_var.set("--")
            self.max_id_dev_var.set("--")
            self.max_od_round_var.set("--")
            self.max_id_round_var.set("--")
            self.max_od_pp_var.set("--")
            self.max_od_pp_rob_var.set("--")
            self.max_od_fit_res_var.set("--")
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
            # f2: OD panel uses peak-to-peak + robust PP + fit-residual (as alternative)
            od_pp = _to_float(getattr(row, 'od_pp_mm', None))
            if od_pp is None:
                od_pp = _to_float(getattr(row, 'od_round', None))
            od_pp_rob = _to_float(getattr(row, 'od_pp_rob_mm', None))
            if od_pp_rob is None:
                od_pp_rob = _to_float(getattr(row, 'od_round', None))
            od_fit_res = _to_float(getattr(row, 'od_round_fit_rob_mm', None))
            if od_fit_res is None:
                od_fit_res = _to_float(getattr(row, 'od_round_fit_mm', None))

            # Keep legacy max_od_round (not shown on main f2 UI) for compatibility
            od_round = _to_float(getattr(row, 'od_round_fit_rob_mm', None))
            if od_round is None:
                od_round = _to_float(getattr(row, 'od_round', None))
            id_round = _to_float(getattr(row, 'id_round_fit_rob_mm', None))
            if id_round is None:
                id_round = _to_float(getattr(row, 'id_round', None))

            if od_dev is not None:
                self._max_od_dev_abs = _upd_max(self._max_od_dev_abs, abs(od_dev))
            if id_dev is not None:
                self._max_id_dev_abs = _upd_max(self._max_id_dev_abs, abs(id_dev))
            if od_round is not None:
                self._max_od_round = _upd_max(self._max_od_round, od_round)
            if id_round is not None:
                self._max_id_round = _upd_max(self._max_id_round, id_round)

            if od_pp is not None:
                self._max_od_pp = _upd_max(self._max_od_pp, od_pp)
            if od_pp_rob is not None:
                self._max_od_pp_rob = _upd_max(self._max_od_pp_rob, od_pp_rob)
            if od_fit_res is not None:
                self._max_od_fit_res = _upd_max(self._max_od_fit_res, od_fit_res)

            if self._max_od_dev_abs is not None:
                self.max_od_dev_var.set(f"{self._max_od_dev_abs:.3f} mm")
            if self._max_id_dev_abs is not None:
                self.max_id_dev_var.set(f"{self._max_id_dev_abs:.3f} mm")
            if self._max_od_round is not None:
                self.max_od_round_var.set(f"{self._max_od_round:.3f} mm")
            if self._max_id_round is not None:
                self.max_id_round_var.set(f"{self._max_id_round:.3f} mm")

            if self._max_od_pp is not None:
                self.max_od_pp_var.set(f"{self._max_od_pp:.3f} mm")
            if self._max_od_pp_rob is not None:
                self.max_od_pp_rob_var.set(f"{self._max_od_pp_rob:.3f} mm")
            if self._max_od_fit_res is not None:
                self.max_od_fit_res_var.set(f"{self._max_od_fit_res:.3f} mm")
        except Exception:
            pass

    def _calc_run_summary(self) -> dict:
        return self.results_service.compute_run_summary(
            recipe=self.recipe,
            rows=list(getattr(self, '_auto_rows', []) or []),
            raw_points=list(getattr(self, '_auto_raw_points', []) or []),
            summary_cache=dict(self._run_summary or {}),
        )

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
                self.max_od_pp_var.set('--')
                self.max_od_pp_rob_var.set('--')
                self.max_od_fit_res_var.set('--')
                self.od_mean_var.set('--')
                self.od_dpp_var.set('--')
                self.od_e_var.set('--')
                self.id_mean_var.set('--')
                self.id_dpp_var.set('--')
                self.od_range_var.set('--')
                self.id_range_var.set('--')
                self.od_slope_var.set('--')
                self.id_slope_var.set('--')
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
        _set_var(self.max_od_pp_var, summary.get('max_od_pp'))
        _set_var(self.max_od_pp_rob_var, summary.get('max_od_pp_rob'))
        _set_var(self.max_od_fit_res_var, summary.get('max_od_fit_res'))

        _set_var(self.od_mean_var, summary.get('od_mean'), unit=' mm')
        _set_var(self.od_dpp_var, summary.get('od_d_pp'), unit=' mm')
        _set_var(self.od_e_var, summary.get('od_e'), unit=' mm')
        _set_var(self.od_range_var, summary.get('od_range'), unit=' mm')

        _set_var(self.id_mean_var, summary.get('id_mean'), unit=' mm')
        _set_var(self.id_dpp_var, summary.get('id_d_pp'), unit=' mm')
        _set_var(self.id_range_var, summary.get('id_range'), unit=' mm')

        # axis-line orientation
        # NOTE: tilt angles are typically very small (<<0.1°). Show 3 decimals to avoid displaying 0.00°.
        try:
            self.od_tilt_var.set("--" if summary.get('od_tilt_deg') is None else f"{float(summary.get('od_tilt_deg')):.3f}°")
            self.od_endoff_var.set("--" if summary.get('od_end_off_mm') is None else f"{float(summary.get('od_end_off_mm')):.3f} mm")
            self.id_tilt_var.set("--" if summary.get('id_tilt_deg') is None else f"{float(summary.get('id_tilt_deg')):.3f}°")
            self.id_endoff_var.set("--" if summary.get('id_end_off_mm') is None else f"{float(summary.get('id_end_off_mm')):.3f} mm")
            self.od_slope_var.set("--" if summary.get('od_slope') is None else f"{float(summary.get('od_slope'))*1000:.3f} mm/m")
            self.id_slope_var.set("--" if summary.get('id_slope') is None else f"{float(summary.get('id_slope'))*1000:.3f} mm/m")
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

    def _ax3_trace_float_or_none(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _ax3_trace_recipe_speed(self, recipe_obj: Any = None) -> Optional[float]:
        r = recipe_obj if recipe_obj is not None else getattr(self, "recipe", None)
        for name in ("ax3_rot_speed", "rot_vel_velmove", "rot_speed"):
            try:
                v = getattr(r, name)
            except Exception:
                continue
            fv = self._ax3_trace_float_or_none(v)
            if fv is not None:
                return fv
        return None

    def _ax3_trace_axis_ui_speed(self) -> Optional[float]:
        def _from_var(obj: Any) -> Optional[float]:
            try:
                if obj is None:
                    return None
                if hasattr(obj, "get"):
                    return self._ax3_trace_float_or_none(obj.get())
                return self._ax3_trace_float_or_none(obj)
            except Exception:
                return None

        try:
            fv = _from_var(getattr(self, "ax3_vel_var", None))
            if fv is not None:
                return fv
        except Exception:
            pass

        try:
            fv = _from_var(self._axis_ui_widget('ent_vel_velmove', 3))
            if fv is not None:
                return fv
        except Exception:
            pass

        try:
            fv = _from_var(getattr(self, "ent_vel_velmove", None))
            if fv is not None:
                return fv
        except Exception:
            pass

        try:
            fv = _from_var(getattr(self, "rot_vel_velmove_var", None))
            if fv is not None:
                return fv
        except Exception:
            pass

        return None

    def _ax3_trace_runtime_speed(self) -> Optional[float]:
        try:
            runtime = getattr(self, "runtime", None)
            axis_params = getattr(runtime, "axis_params", None)
            ax3 = getattr(axis_params, "ax3", None)
            fv = self._ax3_trace_float_or_none(getattr(ax3, "vel", None))
            if fv is not None:
                return fv
        except Exception:
            pass

        try:
            ac3 = self.get_axis_copy(3)
            fv = self._ax3_trace_float_or_none(getattr(ac3, "vel_velmove", None))
            if fv is not None:
                return fv
            fv = self._ax3_trace_float_or_none(getattr(ac3, "vel", None))
            if fv is not None:
                return fv
        except Exception:
            pass
        return None

    def _ax3_trace_fmt(self, value: Optional[float]) -> str:
        if value is None:
            return "None"
        try:
            return f"{float(value):.6f}"
        except Exception:
            return "None"

    def _log_ax3_speed_trace(
        self,
        location_name: str,
        *,
        recipe_obj: Any = None,
        caller_name: Optional[str] = None,
    ) -> None:
        if caller_name is None:
            try:
                caller = inspect.stack()[1].function
            except Exception:
                caller = "unknown"
        else:
            caller = str(caller_name)

        recipe_speed = self._ax3_trace_recipe_speed(recipe_obj=recipe_obj)
        axis_ui_speed = self._ax3_trace_axis_ui_speed()
        runtime_speed = self._ax3_trace_runtime_speed()

        ax3_trace_logger.debug(
            "[AX3_SPEED_TRACE] location=%s | recipe=%s | axis_ui=%s | runtime=%s",
            f"{location_name}:{caller}",
            self._ax3_trace_fmt(recipe_speed),
            self._ax3_trace_fmt(axis_ui_speed),
            self._ax3_trace_fmt(runtime_speed),
        )

    # =========================
    # Low-level write helpers
    # =========================
    def _base(self, axis: int) -> int:
        return axis_base(axis)

    def _write_regs(self, d_addr: int, values: List[int]):
        self.cmd_q.put(CmdWriteRegs(d_addr=d_addr, values=values))

    def _sync_read_perf_group(self, d_addr: int, count: int) -> str:
        try:
            addr = int(d_addr)
            cnt = int(count)
        except Exception:
            return "other"
        try:
            ax3_addr = int(self._base(3) + OFF_ACT_POS)
            if addr == ax3_addr and cnt == 4:
                return "ax3"
        except Exception:
            pass
        try:
            cl145_addrs = {
                int(CL_IN_BASE_D + CL_OUT1_WORD_OFF),
                int(CL_IN_BASE_D + CL_OUT1_UPD_WORD_OFF),
            }
            if addr in cl145_addrs:
                return "cl145"
        except Exception:
            pass
        try:
            cl3_addrs = {
                int(CL_IN_BASE_D + CL_ID_WORD_OFF),
                int(CL_IN_BASE_D + CL_ID_UPD_WORD_OFF),
            }
            if addr in cl3_addrs:
                return "cl3"
        except Exception:
            pass
        return "other"

    def _flush_sync_read_perf_if_due(self) -> None:
        snap = self._perf_sync_read.drain_if_due(every_s=1.0)
        if snap is None:
            return
        c = snap.counts
        t = snap.times
        for cat in ("ax3", "cl145", "cl3", "other"):
            n = int(c.get(f"{cat}.n", 0))
            to_cnt = int(c.get(f"{cat}.timeout", 0))
            if n <= 0 and to_cnt <= 0:
                continue
            st_total = t.get(f"{cat}.total")
            st_put = t.get(f"{cat}.put_cmd")
            st_wait = t.get(f"{cat}.wait_evt")
            st_evt = t.get(f"{cat}.evt_delay")
            total_avg_ms = (ns_to_ms(int(st_total.sum_ns)) / float(st_total.n)) if (st_total and st_total.n > 0) else 0.0
            total_max_ms = ns_to_ms(int(st_total.max_ns)) if (st_total and st_total.n > 0) else 0.0
            put_avg_ms = (ns_to_ms(int(st_put.sum_ns)) / float(st_put.n)) if (st_put and st_put.n > 0) else 0.0
            wait_avg_ms = (ns_to_ms(int(st_wait.sum_ns)) / float(st_wait.n)) if (st_wait and st_wait.n > 0) else 0.0
            evt_avg_ms = (ns_to_ms(int(st_evt.sum_ns)) / float(st_evt.n)) if (st_evt and st_evt.n > 0) else 0.0
            evt_max_ms = ns_to_ms(int(st_evt.max_ns)) if (st_evt and st_evt.n > 0) else 0.0
            try:
                plc_perf_logger.info(
                    "[PLC_PERF] sync_%s n=%d avg_ms=%.3f max_ms=%.3f timeout=%d put_avg_ms=%.3f wait_avg_ms=%.3f evt_delay_avg_ms=%.3f evt_delay_max_ms=%.3f",
                    cat,
                    n,
                    float(total_avg_ms),
                    float(total_max_ms),
                    to_cnt,
                    float(put_avg_ms),
                    float(wait_avg_ms),
                    float(evt_avg_ms),
                    float(evt_max_ms),
                )
            except Exception:
                pass

    def _flush_uiq_perf_if_due(self) -> None:
        snap = self._perf_ui_queue.drain_if_due(every_s=1.0)
        if snap is None:
            return
        c = snap.counts
        v = snap.values
        t = snap.times
        auto_alive = bool(getattr(self, "_auto_thread", None) and self._auto_thread.is_alive())
        plc_read_n = int(c.get("plc_read", 0))
        if (not auto_alive) and plc_read_n <= 0:
            return
        st_loop = t.get("loop")
        st_evt = t.get("evt_delay")
        st_evtlog = t.get("event_log")
        st_refresh = t.get("run_time_refresh")
        loop_avg_ms = (ns_to_ms(int(st_loop.sum_ns)) / float(st_loop.n)) if (st_loop and st_loop.n > 0) else 0.0
        loop_max_ms = ns_to_ms(int(st_loop.max_ns)) if (st_loop and st_loop.n > 0) else 0.0
        evt_avg_ms = (ns_to_ms(int(st_evt.sum_ns)) / float(st_evt.n)) if (st_evt and st_evt.n > 0) else 0.0
        evt_max_ms = ns_to_ms(int(st_evt.max_ns)) if (st_evt and st_evt.n > 0) else 0.0
        evtlog_avg_ms = (ns_to_ms(int(st_evtlog.sum_ns)) / float(st_evtlog.n)) if (st_evtlog and st_evtlog.n > 0) else 0.0
        evtlog_max_ms = ns_to_ms(int(st_evtlog.max_ns)) if (st_evtlog and st_evtlog.n > 0) else 0.0
        refresh_avg_ms = (ns_to_ms(int(st_refresh.sum_ns)) / float(st_refresh.n)) if (st_refresh and st_refresh.n > 0) else 0.0
        refresh_max_ms = ns_to_ms(int(st_refresh.max_ns)) if (st_refresh and st_refresh.n > 0) else 0.0
        bs = v.get("batch_size")
        batch_avg = (float(bs.sum_v) / float(bs.n)) if (bs and bs.n > 0) else 0.0
        batch_max = float(bs.max_v) if (bs and bs.n > 0) else 0.0
        try:
            plc_perf_logger.info(
                "[UIQ_PERF] plc_read=%d calls=%d evt_delay_avg_ms=%.3f evt_delay_max_ms=%.3f "
                "batch_avg=%.2f batch_max=%.0f loop_avg_ms=%.3f loop_max_ms=%.3f "
                "ui_log_avg_ms=%.3f ui_log_max_ms=%.3f ui_refresh_avg_ms=%.3f ui_refresh_max_ms=%.3f",
                plc_read_n,
                int(c.get("calls", 0)),
                float(evt_avg_ms),
                float(evt_max_ms),
                float(batch_avg),
                float(batch_max),
                float(loop_avg_ms),
                float(loop_max_ms),
                float(evtlog_avg_ms),
                float(evtlog_max_ms),
                float(refresh_avg_ms),
                float(refresh_max_ms),
            )
        except Exception:
            pass

    def _read_regs_sync(self, d_addr: int, count: int, timeout_s: float = 0.35) -> Optional[List[int]]:
        """Synchronous Modbus holding-register read via PlcWorker.

        This is used by AutoFlow to obtain a tighter snapshot for binding samples:
        (theta from AX3 act_pos, ID from CL OUT3) at the moment an OD sample arrives.
        """
        t_total0_ns = time.perf_counter_ns()
        perf_cat = self._sync_read_perf_group(d_addr, count)
        self._perf_sync_read.add_count(f"{perf_cat}.n", 1)
        tag = f"sync:{time.time_ns()}"
        evt = threading.Event()
        with self._sync_reads_lock:
            self._sync_reads[tag] = {
                "evt": evt,
                "regs": None,
                "perf_cat": perf_cat,
            }
        try:
            t_put0_ns = time.perf_counter_ns()
            self.cmd_q.put(CmdReadRegs(d_addr, int(count), tag))
            self._perf_sync_read.add_time_ns(f"{perf_cat}.put_cmd", time.perf_counter_ns() - t_put0_ns)
        except Exception:
            with self._sync_reads_lock:
                self._sync_reads.pop(tag, None)
            self._perf_sync_read.add_count(f"{perf_cat}.timeout", 1)
            self._perf_sync_read.add_time_ns(f"{perf_cat}.total", time.perf_counter_ns() - t_total0_ns)
            self._flush_sync_read_perf_if_due()
            return None

        t_wait0_ns = time.perf_counter_ns()
        wait_ok = bool(evt.wait(float(timeout_s)))
        self._perf_sync_read.add_time_ns(f"{perf_cat}.wait_evt", time.perf_counter_ns() - t_wait0_ns)
        if not wait_ok:
            try:
                modbus_logger.debug(
                    "SYNC_READ_TIMEOUT d_addr=%s count=%s timeout_s=%.3f",
                    d_addr,
                    count,
                    float(timeout_s),
                )
            except Exception:
                pass
            with self._sync_reads_lock:
                self._sync_reads.pop(tag, None)
            self._perf_sync_read.add_count(f"{perf_cat}.timeout", 1)
            self._perf_sync_read.add_time_ns(f"{perf_cat}.total", time.perf_counter_ns() - t_total0_ns)
            self._flush_sync_read_perf_if_due()
            return None

        with self._sync_reads_lock:
            slot = self._sync_reads.pop(tag, None)
        if not slot:
            self._perf_sync_read.add_count(f"{perf_cat}.timeout", 1)
            self._perf_sync_read.add_time_ns(f"{perf_cat}.total", time.perf_counter_ns() - t_total0_ns)
            self._flush_sync_read_perf_if_due()
            return None
        regs = slot.get("regs", None)
        try:
            if regs is not None:
                modbus_logger.debug("SYNC_READ_OK d_addr=%s count=%s", d_addr, count)
        except Exception:
            pass
        if regs is None:
            self._perf_sync_read.add_count(f"{perf_cat}.timeout", 1)
            self._perf_sync_read.add_time_ns(f"{perf_cat}.total", time.perf_counter_ns() - t_total0_ns)
            self._flush_sync_read_perf_if_due()
            return None
        try:
            return list(regs)
        except Exception:
            return None
        finally:
            self._perf_sync_read.add_time_ns(f"{perf_cat}.total", time.perf_counter_ns() - t_total0_ns)
            self._flush_sync_read_perf_if_due()

    def read_regs_sync(self, d_addr: int, count: int, timeout_s: float = 0.35) -> Optional[List[int]]:
        """Public wrapper for synchronous holding-register reads."""
        return self._read_regs_sync(d_addr, count, timeout_s=timeout_s)

    def _decode_fp64_4regs(self, regs: List[int]) -> float:
        try:
            return float(decode_float64_from_4regs(list(regs[:4]), FLOAT64_WORD_ORDER))
        except Exception:
            return 0.0

    def _get_latest_ax3_angle_deg(self):
        plc = getattr(self, "plc", None)
        if plc is None:
            plc = getattr(self, "worker", None)
        if plc is None:
            return None
        angle = getattr(plc, "latest_angle_deg", None)
        if angle is None:
            return None
        try:
            return float(angle) % 360.0
        except Exception:
            return None

    def _get_latest_cl145(self):
        plc = getattr(self, "plc", None)
        if plc is None:
            plc = getattr(self, "worker", None)
        if plc is None:
            return None
        ts_ns = getattr(plc, "latest_cl145_ts_ns", None)
        if ts_ns is not None:
            try:
                if (time.perf_counter_ns() - int(ts_ns)) > 500_000_000:
                    return None
            except Exception:
                pass
        data = getattr(plc, "latest_cl145", None)
        if data is None:
            return None
        try:
            x1_mm, x2_mm, c_mm, m_mm, raw_dict, cnt_dict = data
            if not isinstance(raw_dict, dict) or not isinstance(cnt_dict, dict):
                return None
            return (x1_mm, x2_mm, c_mm, m_mm, raw_dict, cnt_dict)
        except Exception:
            return None

    def _get_latest_cl3(self):
        plc = getattr(self, "plc", None)
        if plc is None:
            plc = getattr(self, "worker", None)
        if plc is None:
            return None
        ts_ns = getattr(plc, "latest_cl3_ts_ns", None)
        if ts_ns is not None:
            try:
                if (time.perf_counter_ns() - int(ts_ns)) > 500_000_000:
                    return None
            except Exception:
                pass
        data = getattr(plc, "latest_cl3", None)
        if data is None:
            return None
        try:
            id_mm, raw, cnt = data
        except Exception:
            return None
        try:
            raw_i = None if raw is None else int(raw)
        except Exception:
            raw_i = None
        try:
            cnt_i = None if cnt is None else int(cnt)
        except Exception:
            cnt_i = None
        try:
            id_v = None if id_mm is None else float(id_mm)
        except Exception:
            id_v = None
        if raw_i is not None and raw_i in {CL_OUT_INVALID, CL_OUT_STANDBY, CL_OUT_POS_OVER, CL_OUT_NEG_OVER}:
            id_v = None
        if id_v is None and raw_i is not None:
            try:
                id_v = float(raw_i) * float(CL_ID_SCALE_MM)
            except Exception:
                id_v = None
        if id_v is not None:
            try:
                id_v += float(self.idcal_delta_active_var.get())
            except Exception:
                pass
        return (id_v, raw_i, cnt_i)

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

    def read_cl_sync(self, channel: str, *, timeout_s: float = 0.5):
        """Public CL sync-read wrapper used by the device gateway adapter."""
        ch = str(channel or "out145").strip().lower()
        if ch == "out145":
            return self.read_cl_out145_sync(timeout_s=timeout_s)
        if ch == "out3":
            return self.read_cl_out3_sync(timeout_s=timeout_s)
        raise ValueError(f"unsupported CL channel: {channel}")


    def set_cmd_bits(self, axis: int, set_mask: int = 0, clr_mask: int = 0):
        try:
            ax = int(axis)
            mask_all = int(set_mask) | int(clr_mask)
            if ax == 3 and (mask_all & int(CMD_VELMOVE_REQ)):
                try:
                    caller = inspect.stack()[1].function
                except Exception:
                    caller = "unknown"
                self._log_ax3_speed_trace(
                    "ax3_cmd_pre_velmove",
                    caller_name=caller,
                )
        except Exception:
            pass
        self.cmd_q.put(CmdSetCmdMask(axis=axis, set_mask=set_mask, clr_mask=clr_mask))

    def set_plc_poll_profile(self, profile: str = "normal", *, caller: str | None = None) -> None:
        # Set PLC worker background polling profile.
        # profile:
        #   - 'normal': poll all axes + CL + keytest
        #   - 'sampling': poll only AX3 and disable CL/keytest background polling
        try:
            prof = str(profile or 'normal').strip().lower()
            if prof not in ('normal', 'sampling'):
                prof = 'normal'
            previous_profile = str(getattr(self, "_plc_poll_profile_req", "normal") or "normal")
            caller_name = str(caller or "").strip()
            if not caller_name:
                try:
                    caller_name = inspect.stack()[1].function
                except Exception:
                    caller_name = "unknown"
            mode_kind = self._current_mode_kind_name()
            validation_running = bool(getattr(self, "_validation_running", False))
            auto_thread_alive = self._is_auto_thread_alive()
            log(
                "PLC_POLL_PROFILE_SET",
                requested_profile=prof,
                previous_profile=previous_profile,
                mode_kind=mode_kind,
                validation_running=validation_running,
                auto_thread_alive=auto_thread_alive,
                caller=caller_name,
            )
            self._plc_poll_profile_req = prof
            self.cmd_q.put(CmdSetPollProfile(profile=prof))
        except Exception:
            pass


    def _pulse_cmd_bits(self, axis: int, pulse_mask: int, pulse_ms: int = 120):
        try:
            ax = int(axis)
            mask = int(pulse_mask)
            try:
                caller = inspect.stack()[1].function
            except Exception:
                caller = "unknown"
            if ax == 3 and (mask & int(CMD_MOVEA_REQ)):
                self._log_ax3_speed_trace(
                    "ax3_cmd_pre_movea",
                    caller_name=caller,
                )
            if ax == 3 and (mask & int(CMD_MOVER_REQ)):
                self._log_ax3_speed_trace(
                    "ax3_cmd_pre_mover",
                    caller_name=caller,
                )
        except Exception:
            pass
        self.cmd_q.put(
            CmdPulseCmdMask(axis=axis, pulse_mask=pulse_mask, pulse_ms=pulse_ms)
        )

    def pulse_cmd_mask(self, axis: int, pulse_mask: int, pulse_ms: int = 120) -> None:
        """Public wrapper for PLC pulse command masks."""
        self._pulse_cmd_bits(axis, pulse_mask, pulse_ms=pulse_ms)


    def _parse_float(self, s: str, default: float) -> float:
        try:
            return float(str(s).strip())
        except Exception:
            return float(default)

    def _axis_ui_widget(self, name: str, axis: Optional[int] = None) -> Any:
        presenter = getattr(self, '_axis_screen_presenter', None)
        if presenter is not None:
            try:
                if axis is None:
                    return presenter.current_widget(name)
                return presenter.widget_for(axis, name)
            except Exception:
                pass
        return getattr(self, name, None)

    def _axis_ui_power_var(self, axis: Optional[int] = None) -> Any:
        presenter = getattr(self, '_axis_screen_presenter', None)
        if presenter is not None:
            try:
                if axis is None:
                    return presenter.power_var_for()
                return presenter.power_var_for(axis)
            except Exception:
                pass
        return getattr(self, 'power_var', None)

    def _read_axis_params_from_ui(self) -> tuple[float, float, float, float, int, float, float, float]:
        """Read per-axis motion parameters from UI entries.

        Returns:
            (vel_movea, vel_mover, vel_jog, vel_velmove, dir_mover, acc, dec, jerk)
        """
        # New UI (recommended)
        ent_vel_movea = self._axis_ui_widget('ent_vel_movea')
        if ent_vel_movea is not None:
            vel_movea = self._parse_float(ent_vel_movea.get(), 100.0)
            vel_mover = self._parse_float(self._axis_ui_widget('ent_vel_mover').get(), vel_movea)
            vel_jog = self._parse_float(self._axis_ui_widget('ent_vel_jog').get(), 80.0)
            vel_velmove = self._parse_float(self._axis_ui_widget('ent_vel_velmove').get(), 200.0)
            acc = self._parse_float(self._axis_ui_widget('ent_acc').get(), 200.0)
            dec = self._parse_float(self._axis_ui_widget('ent_dec').get(), 200.0)
            jerk = self._parse_float(self._axis_ui_widget('ent_jerk').get(), 500.0)

            dir_mover = DIR_NONE
            dir_mover_var = self._axis_ui_widget('dir_mover_var')
            cmb_dir_mover = self._axis_ui_widget('cmb_dir_mover')
            if dir_mover_var is not None:
                try:
                    dir_mover = int(dir_mover_var.get())
                except Exception:
                    dir_mover = DIR_NONE
            elif cmb_dir_mover is not None:
                try:
                    txt = str(cmb_dir_mover.get())
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

        # Compatibility UI fallback: one vel + acc/dec/jerk
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
        # Note: keepout can block length edge search because approach moves are clamped.
        # We bypass keepout during LenEdge* contexts; soft limits remain enforced.
        if context and ('LenEdge' in str(context)):
            pass
        else:
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
        if int(axis) == 3:
            self._log_ax3_speed_trace("movea_abs_ax3_pre_pulse")
        self._pulse_cmd_bits(axis, CMD_MOVEA_REQ)

    # =========================
    # UI actions (manual)
    # =========================
    def _axis(self) -> int:
        i = int(self.axis_idx.get())
        return max(0, min(AXIS_COUNT - 1, i))

    def _noop_ui_event_handler(self, _payload: Any) -> None:
        pass

    def _build_device_ui_event_dispatcher(self) -> UiEventDispatcher:
        return UiEventDispatcher(
            {
                PlcOkEvent: self._handle_plc_ok_event,
                PlcErrEvent: self._handle_plc_err_event,
                PlcGiveupEvent: self._handle_plc_giveup_event,
                PlcManualEvent: self._handle_plc_manual_event,
                PlcReadEvent: self._handle_plc_read_event,
                GaugeConnEvent: self._handle_gauge_conn_event,
                GaugeTxEvent: self._handle_gauge_tx_event,
                GaugeOkEvent: self._handle_gauge_ok_event,
                GaugeRawEvent: self._handle_gauge_raw_event,
                GaugeErrEvent: self._handle_gauge_err_event,
            }
        )

    def _build_measurement_ui_event_dispatcher(self) -> UiEventDispatcher:
        return UiEventDispatcher(
            {
                OpConfirmShowEvent: self._handle_op_confirm_show_event,
                OpConfirmCloseEvent: self._handle_op_confirm_close_event,
                AutoClearEvent: self._handle_auto_clear_event,
                AutoLenEvent: self._handle_auto_len_event,
                AutoProgressEvent: self._handle_auto_progress_event,
                AutoCoverageEvent: self._handle_auto_coverage_event,
                AutoStraightnessEvent: self._handle_auto_straightness_event,
                AutoPostcalcEvent: self._handle_auto_postcalc_event,
                AutoRawPointsEvent: self._handle_auto_raw_points_event,
                AutoRowEvent: self._handle_auto_row_event,
                AutoStateEvent: self._handle_auto_state_event,
            }
        )

    def _handle_plc_ok_event(self, event: PlcOkEvent) -> None:
        payload = event.to_payload()
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

    def _handle_plc_err_event(self, event: PlcErrEvent) -> None:
        payload = event.to_payload()
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

    def _handle_plc_giveup_event(self, event: PlcGiveupEvent) -> None:
        payload = event.to_payload()
        retry = payload.get("retry", 0)
        mx = payload.get("max", 0)
        self.plc_status_var.set(
            f"PLC: GIVE UP after {retry}/{mx}. Click Apply to reconnect."
        )

    def _handle_plc_manual_event(self, event: PlcManualEvent) -> None:
        payload = event.to_payload()
        ip = payload.get("ip", "")
        port = payload.get("port", "")
        self.plc_status_var.set(f"PLC: MANUAL CONNECT... ip={ip}:{port}")

    def _handle_plc_read_event(self, event: PlcReadEvent) -> None:
        payload = event.to_payload()
        tag = payload.get("tag", "")
        d_addr = payload.get("d_addr", None)
        count = payload.get("count", None)
        regs = payload.get("regs", [])

        # sync reads (AutoFlow sampling)
        if isinstance(tag, str) and tag.startswith("sync:"):
            now_ns = time.perf_counter_ns()
            try:
                t_uiq_put_ns = int(payload.get("t_uiq_put_ns", 0) or 0)
                if t_uiq_put_ns > 0:
                    self._perf_ui_queue.add_time_ns("evt_delay", now_ns - t_uiq_put_ns)
            except Exception:
                pass
            try:
                with self._sync_reads_lock:
                    slot = self._sync_reads.get(tag, None)
                    if slot is not None:
                        slot["regs"] = list(regs)
                        try:
                            perf_cat = str(slot.get("perf_cat", "other") or "other")
                            t_uiq_put_ns = int(payload.get("t_uiq_put_ns", 0) or 0)
                            if t_uiq_put_ns > 0:
                                self._perf_sync_read.add_time_ns(
                                    f"{perf_cat}.evt_delay",
                                    now_ns - t_uiq_put_ns,
                                )
                        except Exception:
                            pass
                        try:
                            slot["evt"].set()
                        except Exception:
                            pass
            except Exception:
                pass
            # Do not fall through to axis_cal parsing
            return

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

    def _handle_gauge_conn_event(self, event: GaugeConnEvent) -> None:
        payload = event.to_payload()
        if payload.get("connected"):
            port = payload.get("port", "")
            baud = payload.get("baud", "")
            self.gauge_conn_var.set(f"串口: 已连接 ({port}@{baud})")
        else:
            self.gauge_conn_var.set("串口: 未连接")

    def _handle_gauge_tx_event(self, event: GaugeTxEvent) -> None:
        payload = event.to_payload()
        # 可选：显示最近一次发送的请求（避免刷屏，只做轻提示）
        cmd = payload.get("cmd", "")
        if cmd:
            self.gauge_err_var.set(f"已发送: {cmd}")

    def _handle_gauge_ok_event(self, event: GaugeOkEvent) -> None:
        payload = event.to_payload()
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

    def _handle_gauge_raw_event(self, event: GaugeRawEvent) -> None:
        payload = event.to_payload()
        # only update if no parsed value is flowing
        pass

    def _handle_gauge_err_event(self, event: GaugeErrEvent) -> None:
        payload = event.to_payload()
        self.gauge_err_var.set(f"Gauge ERROR: {payload.get('err')}")

    def _handle_op_confirm_show_event(self, event: OpConfirmShowEvent) -> None:
        payload = event.to_payload()
        try:
            self._show_op_confirm_popup(
                token=str(payload.get('token', '')),
                title=str(payload.get('title', '操作员确认')),
                message=str(payload.get('message', '')),
                allow_stop=bool(payload.get('allow_stop', True)),
            )
        except Exception:
            pass

    def _handle_op_confirm_close_event(self, event: OpConfirmCloseEvent) -> None:
        payload = event.to_payload()
        try:
            self._close_op_confirm_popup(str(payload.get('token', '')))
        except Exception:
            pass

    def _handle_auto_clear_event(self, event: AutoClearEvent) -> None:
        payload = event.to_payload()
        # AutoFlow sends auto_clear at the beginning of a run; do NOT wipe run identity/timestamps.
        self._auto_clear_ui(preserve_run=True)

    def _handle_auto_len_event(self, event: AutoLenEvent) -> None:
        payload = event.to_payload()
        # Published by AutoFlow after S30 (length measurement)
        p = self._cache_auto_len_result(payload)

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
                                    # UI: hide tolerance text here; keep result predictable and compact.
                                    self.len_meas_var.set(f"{l:.3f} mm  (Δ {dev:+.3f})  {judge_txt}")
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

    def _handle_auto_progress_event(self, event: AutoProgressEvent) -> None:
        payload = event.to_payload()
        idx = int(payload.get("idx", 0))
        total = int(payload.get("total", 0))
        # UI uses 1-based section index
        self._auto_cur_sec_idx = idx + 1
        self.auto_progress_var.set(f"当前截面: {idx + 1} / 总截面: {total}")
        self.auto_done_var.set("测量完成: 否")

    def _handle_auto_coverage_event(self, event: AutoCoverageEvent) -> None:
        payload = event.to_payload()
        # Coverage info may optionally carry a 1-based section idx.
        sec_idx_int, info = self._cache_section_cov_info(payload)
        txt = self._format_cov_info(info)
        # If user selected a section row, keep showing that row's info
        # unless the update corresponds to the same section.
        if (self._selected_sec_idx is None) or (sec_idx_int is None) or (int(self._selected_sec_idx) == int(sec_idx_int)):
            self.cov_var.set(txt)

    def _handle_auto_straightness_event(self, event: AutoStraightnessEvent) -> None:
        payload = event.to_payload()
        self._apply_run_summary_payload(payload)
        self._refresh_done_run_summary_and_export()

    def _handle_auto_postcalc_event(self, event: AutoPostcalcEvent) -> None:
        payload = event.to_payload()
        self._apply_run_summary_payload(payload)
        self._refresh_done_run_summary_and_export()
        self._apply_postcalc_eccentricity(payload)

    def _handle_auto_raw_points_event(self, event: AutoRawPointsEvent) -> None:
        payload = event.to_payload()
        self._cache_auto_raw_points(payload)

    def _handle_auto_row_event(self, event: AutoRowEvent) -> None:
        payload = event.to_payload()
        row: MeasureRow = payload["row"]
        self._append_result_row(row)

    def _handle_auto_state_event(self, event: AutoStateEvent) -> None:
        payload = event.to_payload()
        st = payload.get("state", "IDLE")
        msg = payload.get("msg", "-")
        try:
            self.mode_machine.sync_production_workflow_state(str(st), str(msg))
        except Exception:
            pass
        self.auto_state_var.set(str(st))
        self.auto_msg_var.set(str(msg))
        if st == "DONE":
            self.auto_done_var.set("\u6d4b\u91cf\u5b8c\u6210: \u662f")
            self._trigger_run_export()
        elif st in ("ERR", "STOP"):
            self.auto_done_var.set("\u6d4b\u91cf\u5b8c\u6210: \u5426")
            self._freeze_run_end_ts_if_missing()

    def _poll_ui_queue(self):
        t_poll0_ns = time.perf_counter_ns()
        batch_size = 0
        plc_read_n = 0
        try:
            while True:
                k, payload = self.ui_q.get_nowait()
                batch_size += 1

                # lightweight workflow logging (avoid high-frequency spam)
                t_evtlog0_ns = time.perf_counter_ns()
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
                self._perf_ui_queue.add_time_ns("event_log", time.perf_counter_ns() - t_evtlog0_ns)


                if k == "plc_read":
                    plc_read_n += 1

                handled = self._device_ui_event_dispatcher.dispatch(k, payload)
                if not handled:
                    self._measurement_ui_event_dispatcher.dispatch(k, payload)

        except queue.Empty:
            pass
        try:
            self._perf_ui_queue.add_count("calls", 1)
            self._perf_ui_queue.add_count("plc_read", int(plc_read_n))
            self._perf_ui_queue.add_value("batch_size", float(batch_size))
            self._perf_ui_queue.add_time_ns("loop", time.perf_counter_ns() - t_poll0_ns)
            self._flush_uiq_perf_if_due()
            self._flush_sync_read_perf_if_due()
        except Exception:
            pass
        t_refresh0_ns = time.perf_counter_ns()
        try:
            self._refresh_run_time_ui()
        except Exception:
            pass
        self._perf_ui_queue.add_time_ns("run_time_refresh", time.perf_counter_ns() - t_refresh0_ns)
        self.after(60, self._poll_ui_queue)

    def _cache_auto_len_result(self, payload: Any) -> dict:
        p = payload if isinstance(payload, dict) else {}
        try:
            self._run_len_result = dict(p)
        except Exception:
            self._run_len_result = None
        return p

    def _cache_section_cov_info(self, payload: Any) -> tuple[Optional[int], dict]:
        p = payload if isinstance(payload, dict) else {}
        sec_idx = p.get("idx", None)
        try:
            sec_idx_int = int(sec_idx) if sec_idx is not None else (int(self._auto_cur_sec_idx) if self._auto_cur_sec_idx is not None else None)
        except Exception:
            sec_idx_int = int(self._auto_cur_sec_idx) if self._auto_cur_sec_idx is not None else None

        info = {
            "cov": p.get("cov", None),
            "miss": p.get("miss", None),
            "max_gap_deg": p.get("max_gap_deg", None),
            "reason": str(p.get("reason", "") or ""),
            "revs": p.get("revs", None),
            "elapsed": p.get("elapsed", None),
        }

        if sec_idx_int is not None:
            self._section_cov_info[int(sec_idx_int)] = info
            try:
                self._update_result_row_cov(int(sec_idx_int), info)
            except Exception:
                pass
        return sec_idx_int, info

    def _apply_run_summary_payload(self, payload: Any) -> None:
        snapshot = self.results_service.summary_snapshot_from_payload(payload)
        self._run_summary = self.results_service.merge_summary_snapshot(self._run_summary, snapshot)

        def _assign_if_provided(field_name: str, host_attr: str) -> None:
            if field_name not in snapshot.provided_fields:
                return
            setattr(self, host_attr, getattr(snapshot, field_name))

        _assign_if_provided('axis_dist', '_axis_dist')
        _assign_if_provided('conc_max', '_conc_max')
        _assign_if_provided('axis_span_max', '_axis_span_max')

        self._set_straight_label(
            snapshot.straight_od,
            snapshot.straight_id,
            self._axis_dist,
            self._conc_max,
            self._axis_span_max,
        )

        self._last_straight_od = snapshot.straight_od if 'straight_od' in snapshot.provided_fields else self._last_straight_od
        self._last_straight_id = snapshot.straight_id if 'straight_id' in snapshot.provided_fields else self._last_straight_id
        self._last_axis_dist = self._axis_dist if 'axis_dist' in snapshot.provided_fields else self._last_axis_dist
        self._last_conc_max = self._conc_max if 'conc_max' in snapshot.provided_fields else self._last_conc_max
        self._last_axis_span_max = self._axis_span_max if 'axis_span_max' in snapshot.provided_fields else self._last_axis_span_max
        self._last_od_tilt_deg = snapshot.od_tilt_deg if 'od_tilt_deg' in snapshot.provided_fields else self._last_od_tilt_deg
        self._last_od_end_off_mm = snapshot.od_end_off_mm if 'od_end_off_mm' in snapshot.provided_fields else self._last_od_end_off_mm
        self._last_od_slope = snapshot.od_slope if 'od_slope' in snapshot.provided_fields else self._last_od_slope
        self._last_id_tilt_deg = snapshot.id_tilt_deg if 'id_tilt_deg' in snapshot.provided_fields else self._last_id_tilt_deg
        self._last_id_end_off_mm = snapshot.id_end_off_mm if 'id_end_off_mm' in snapshot.provided_fields else self._last_id_end_off_mm
        self._last_id_slope = snapshot.id_slope if 'id_slope' in snapshot.provided_fields else self._last_id_slope

        try:
            self.od_tilt_var.set("--" if self._last_od_tilt_deg is None else f"{float(self._last_od_tilt_deg):.3f}°")
            self.od_endoff_var.set("--" if self._last_od_end_off_mm is None else f"{float(self._last_od_end_off_mm):.3f} mm")
            self.od_slope_var.set("--" if self._last_od_slope is None else f"{float(self._last_od_slope)*1000:.3f} mm/m")
            self.id_tilt_var.set("--" if self._last_id_tilt_deg is None else f"{float(self._last_id_tilt_deg):.3f}°")
            self.id_endoff_var.set("--" if self._last_id_end_off_mm is None else f"{float(self._last_id_end_off_mm):.3f} mm")
            self.id_slope_var.set("--" if self._last_id_slope is None else f"{float(self._last_id_slope)*1000:.3f} mm/m")
        except Exception:
            pass
    def _refresh_done_run_summary_and_export(self) -> None:
        try:
            if str(self.auto_state_var.get() or '') == 'DONE':
                self._compute_and_apply_run_summary()
                try:
                    ctx = self._build_run_context_for_export(status='DONE')
                    self._make_run_repository().export_daily_summary(ctx)
                except Exception:
                    pass
        except Exception:
            pass

    def _apply_postcalc_eccentricity(self, payload: Any) -> None:
        updates = self.results_service.build_eccentricity_updates(payload)
        try:
            for update in updates:
                if int(update.row_index) >= len(self._result_iids):
                    break
                iid = self._result_iids[int(update.row_index)]
                tree = self._main_ui_widget('result_tree')
                if tree is None:
                    break
                tree.set(iid, "od_ecc", "--" if update.od_ecc is None else f"{float(update.od_ecc):.3f}")
                tree.set(iid, "id_ecc", "--" if update.id_ecc is None else f"{float(update.id_ecc):.3f}")
        except Exception:
            pass
        try:
            self._auto_rows = self.results_service.apply_eccentricity_updates(self._auto_rows, updates)
        except Exception:
            pass
    def _cache_auto_raw_points(self, payload: Any) -> None:
        p = payload if isinstance(payload, dict) else {}
        pts = p.get("points", []) or []
        try:
            if isinstance(pts, list):
                self._auto_raw_points.extend([point for point in pts if isinstance(point, dict)])
        except Exception:
            pass

    def _freeze_run_end_ts_if_missing(self) -> None:
        if getattr(self, '_run_end_ts', None) is None and getattr(self, '_run_start_ts', None):
            try:
                self._run_end_ts = float(time.time())
            except Exception:
                pass

    def _compact_status_path(self, path: Any, *, keep_parts: int = 3) -> str:
        text = str(path or "").strip()
        if not text:
            return ""
        try:
            p = Path(text)
            parts = list(p.parts)
            if len(parts) <= keep_parts + 1:
                return text
            anchor = p.drive or p.anchor.rstrip("\\/")
            sep = "\\" if "\\" in text else os.sep
            prefix = f"{anchor}{sep}..." if anchor else "..."
            return sep.join([prefix, *parts[-keep_parts:]])
        except Exception:
            if len(text) <= 80:
                return text
            return "..." + text[-77:]

    def _trigger_run_export(self) -> None:
        if getattr(self, "_auto_export_done", False):
            return
        try:
            self._run_end_ts = float(time.time())
        except Exception:
            self._run_end_ts = None
        try:
            ctx = self._build_run_context_for_export(status='DONE')
            run_dir = self._make_run_repository().export_run(ctx)
            self._last_run_export_path = str(run_dir)
            ok, emsg = True, f"导出完成: {self._compact_status_path(run_dir)}"
        except Exception as e:
            self._last_run_export_path = None
            ok, emsg = False, f"export failed: {e}"
        self._auto_export_done = True if ok else False
        try:
            self.auto_msg_var.set(str(emsg))
        except Exception:
            pass
        try:
            self._compute_and_apply_run_summary()
        except Exception:
            pass

    def _append_result_row(self, row: MeasureRow):
        od_ecc_txt = "--" if getattr(row, "od_ecc", None) is None else f"{float(row.od_ecc):.3f}"
        id_ecc_txt = "--" if getattr(row, "id_ecc", None) is None else f"{float(row.id_ecc):.3f}"

        od_e_txt = "--" if getattr(row, "od_e", None) is None else f"{float(getattr(row, 'od_e', 0.0)):.3f}"
        od_phi_txt = "--" if getattr(row, "od_phi_deg", None) is None else f"{float(getattr(row, 'od_phi_deg', 0.0)):+.1f}"

        id_e_txt = "--" if getattr(row, "id_e", None) is None else f"{float(getattr(row, 'id_e', 0.0)):.3f}"
        id_phi_txt = "--" if getattr(row, "id_phi_deg", None) is None else f"{float(getattr(row, 'id_phi_deg', 0.0)):+.1f}"

        # Main-screen UI (f2):
        # - od_round column is displayed as "外径峰峰" => strict peak-to-peak of diameter series
        # - od_pp_rob: robust peak-to-peak
        # - od_fit_res: fit-residual (robust span) as an alternative roundness metric
        try:
            od_pp_ui = getattr(row, 'od_pp_mm', None)
            if od_pp_ui is None:
                od_pp_ui = getattr(row, 'od_round', None)
        except Exception:
            od_pp_ui = getattr(row, 'od_round', None)

        try:
            od_pp_rob_ui = getattr(row, 'od_pp_rob_mm', None)
            if od_pp_rob_ui is None:
                od_pp_rob_ui = getattr(row, 'od_round', None)
        except Exception:
            od_pp_rob_ui = getattr(row, 'od_round', None)

        try:
            od_fit_res_ui = getattr(row, 'od_round_fit_rob_mm', None)
            if od_fit_res_ui is None:
                od_fit_res_ui = getattr(row, 'od_round_fit_mm', None)
        except Exception:
            od_fit_res_ui = getattr(row, 'od_round_fit_mm', None)
        try:
            id_round_ui = getattr(row, 'id_round_fit_rob_mm', None)
            if id_round_ui is None:
                id_round_ui = getattr(row, 'id_round', None)
        except Exception:
            id_round_ui = getattr(row, 'id_round', None)

        # fill cov columns if available (auto_cov message may arrive before/after auto_row)
        cov_info = self._section_cov_info.get(int(getattr(row, "idx", 0) or 0), {})
        cov_cols = self._format_cov_cols(cov_info)

        def _fmt_float(v, nd: int = 3) -> str:
            try:
                if v is None:
                    return "--"
                return f"{float(v):.{nd}f}"
            except Exception:
                return "--"

        def _fmt_signed(v, nd: int = 3) -> str:
            try:
                if v is None:
                    return "--"
                return f"{float(v):+.{nd}f}"
            except Exception:
                return "--"

        def _fmt_shift_deg(v) -> str:
            try:
                if v is None:
                    return "--"
                return f"{float(v):.1f}"
            except Exception:
                return "--"

        def _fmt_unreliable(v) -> str:
            # v can be bool/0/1/None
            if v is None:
                return "--"
            try:
                return "否" if bool(v) else "是"
            except Exception:
                return "--"


        id_dev_txt = _fmt_signed(getattr(row, "id_dev", None), nd=3)
        try:
            if str(getattr(row, "id_mode", "") or "").strip().lower() == "single":
                if id_dev_txt == "--":
                    id_dev_txt = "S"
                else:
                    id_dev_txt = f"{id_dev_txt} (S)"
        except Exception:
            pass

        tree = self._main_ui_widget('result_tree')
        if tree is None:
            return

        iid = tree.insert(
            "",
            "end",
            values=(
                row.idx,
                _fmt_float(getattr(row, 'x_ui', None), nd=3),

                # OD
                _fmt_signed(getattr(row, 'od_dev', None), nd=3),
                _fmt_float(getattr(row, 'od_runout', None), nd=3),
                _fmt_float(od_pp_ui, nd=3),
                _fmt_float(od_pp_rob_ui, nd=3),
                _fmt_float(od_fit_res_ui, nd=3),
                od_e_txt,
                od_phi_txt,
                od_ecc_txt,

                # ID
                id_dev_txt,
                _fmt_float(getattr(row, 'id_runout', None), nd=3),
                _fmt_float(id_round_ui, nd=3),
                id_e_txt,
                id_phi_txt,
                id_ecc_txt,

                # cross
                _fmt_float(getattr(row, 'concentricity', None), nd=3),

                # split diagnostics
                _fmt_shift_deg(getattr(row, 'split_shift_deg', None)),
                _fmt_unreliable(getattr(row, 'coax_unreliable', None)),

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



    def _odcal_save_active(self, data: dict) -> None:
        self.calibration_repository.save_od_active(data or {})

    def _odcal_load_active(self) -> None:
        data = self.calibration_repository.load_od_prefill()
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
            out1 = str(data.get("out1_map", "L") or "L").upper()
            if out1 in ("L", "R"):
                self.odcal_map_out1_var.set(out1)
        except Exception:
            pass

        # Prefill advanced params if present
        try:
            ang = str(data.get("angle_src_ui", "") or "").strip()
            if ang:
                self.odcal_angle_src_var.set(ang)
            flt = str(data.get("filter", "") or "").strip()
            if flt:
                self.odcal_filter_var.set(flt)
            sig = data.get("outlier_sigma", None)
            if sig is not None:
                self.odcal_outlier_sigma_var.set(str(sig))
        except Exception:
            pass
        # load defect template (if any)
        try:
            self._odcal_defect_template_mask = list(data.get("defect_template_mask", [0] * 360) or [0] * 360)

            if sum(int(x) for x in (self._odcal_defect_template_mask or [])) > 0:
                self.odcal_defect_mode_var.set("TEMPLATE")
                self.odcal_defect_shift_var.set("--")
                self.odcal_defects_var.set("模板: " + self._odcal_ranges_str(self._odcal_mask_to_ranges(self._odcal_defect_template_mask)))
            else:
                self.odcal_defect_mode_var.set("OFF")
                self.odcal_defect_shift_var.set("--")
                self.odcal_defects_var.set("--")
        except Exception:
            pass




    # ------------------------------
    # ID Calibration helpers (Chord OUT4 + m OUT5)
    # ------------------------------



    def _idcal_load_active(self) -> None:
        data = self.calibration_repository.load_id_prefill()
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

    # ------------------------------
    # ID single-probe calibration (OUT2/L2)
    # ------------------------------
    def _id_single_cal_start_ax3_rotation(self, speed_degps: float) -> None:
        try:
            try:
                self.set_cmd_bits(3, set_mask=CMD_EN_REQ, clr_mask=0)
            except Exception:
                pass
            self._velmove_start_axis(3, float(speed_degps))
            self._id_single_cal_ax3_rotating = True
        except Exception:
            self._id_single_cal_ax3_rotating = False
            raise

    def _id_single_cal_stop_ax3_rotation(self) -> None:
        try:
            if not bool(self._id_single_cal_ax3_rotating):
                return
            self._velmove_stop_axis(3)
        finally:
            self._id_single_cal_ax3_rotating = False

    def _id_single_cal_update_rev_progress(self, theta_deg: float) -> None:
        if self._id_single_cal_theta_start is None:
            self._id_single_cal_theta_start = float(theta_deg)
            self._id_single_cal_theta_last = float(theta_deg)
            self._id_single_cal_theta_unwrap = 0.0
            self._id_single_cal_rev_progress_deg = 0.0
            return
        last = float(self._id_single_cal_theta_last if self._id_single_cal_theta_last is not None else theta_deg)
        cur = float(theta_deg)
        d = cur - last
        if d < -180.0:
            d += 360.0
        elif d > 180.0:
            d -= 360.0
        self._id_single_cal_theta_unwrap += d
        self._id_single_cal_theta_last = cur
        self._id_single_cal_rev_progress_deg = abs(self._id_single_cal_theta_unwrap)

    def _id_single_cal_rev_done(self) -> bool:
        return bool(self._id_single_cal_rev_progress_deg >= float(self._id_single_cal_rev_target_deg))

    def _id_single_cal_clear(self) -> None:
        return self.calibration_controller.clear_id_single_capture()

    def _id_single_cal_stop_capture(self, reason: str = "") -> None:
        return self.calibration_controller.stop_id_single_capture(reason)

    def _id_single_cal_start_capture(self) -> None:
        return self.calibration_controller.start_id_single_capture()

    def _id_single_cal_tick(self) -> None:
        if not self._id_single_cal_capturing:
            return
        now = time.time()
        if self._id_single_cal_one_rev_timeout_ts is not None:
            try:
                if now >= float(self._id_single_cal_one_rev_timeout_ts):
                    self._id_single_cal_stop_capture("一圈超时")
                    return
            except Exception:
                pass

        # theta from snapshot
        theta_deg = float("nan")
        try:
            with self._snapshot_lock:
                theta_deg = float(self._axis_snapshot[3].act_pos)
        except Exception:
            pass

        if math.isfinite(theta_deg):
            self._id_single_cal_update_rev_progress(float(theta_deg))
            if self._id_single_cal_rev_done():
                self._id_single_cal_stop_capture("已采满一圈")
                return

        # cached OUT2
        x1_mm, x2_mm, _c_mm, _m_mm, raw, cnt = self.get_cl_out145_cached()
        out2_cnt = None
        try:
            out2_cnt = cnt.get("out2", None) if isinstance(cnt, dict) else None
        except Exception:
            out2_cnt = None

        accept = False
        if x2_mm is not None and math.isfinite(float(x2_mm)):
            if out2_cnt is None:
                accept = True
            else:
                last = getattr(self, "_id_single_cal_last_out2_cnt", None)
                accept = (last is None) or (int(out2_cnt) != int(last))
            if accept and out2_cnt is not None:
                self._id_single_cal_last_out2_cnt = int(out2_cnt)

        if accept:
            self._id_single_cal_points.append({
                "ts": now,
                "theta_deg": float(theta_deg),
                "out2_mm": float(x2_mm),
                "raw": raw,
                "cnt": cnt,
            })

        # schedule next
        try:
            hz = float(self._parse_float(self.idcal_hz_var.get(), 20.0))
            hz = max(1.0, min(100.0, hz))
        except Exception:
            hz = 20.0
        period_ms = int(max(5, round(1000.0 / hz)))
        self._id_single_cal_after_id = self.after(period_ms, self._id_single_cal_tick)

    def _id_single_cal_compute_apply(self) -> None:
        return self.calibration_controller.compute_and_write_id_single_calibration()

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





    def _lsq_fit_cos_sin(theta_rad: np.ndarray, y: np.ndarray):
        X = np.column_stack([np.ones_like(theta_rad), np.cos(theta_rad), np.sin(theta_rad)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        x0, A, B = float(beta[0]), float(beta[1]), float(beta[2])
        return x0, A, B

    def calc_id_single_from_out2(self, theta_deg: Iterable[float], out2_mm: Iterable[float], recipe: Recipe) -> dict:
        return self.calibration_service.calc_id_single_from_out2(theta_deg, out2_mm, recipe)

    def _idcal_fit_diameter(self, theta_deg: np.ndarray, c_mm: np.ndarray, m_mm: np.ndarray, delta_c: float):
        return self.calibration_service.fit_id_diameter(theta_deg, c_mm, m_mm, delta_c)






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

    def _make_run_repository(self) -> RunRepository:
        """Build a repository with current runtime/device metadata."""
        return RunRepository(
            app_root_dir=self._app_root_dir(),
            software_version=str(SOFTWARE_VERSION),
            plc_info={
                "ip": getattr(self.worker, "ip", ""),
                "port": getattr(self.worker, "port", ""),
                "unit": getattr(self.worker, "unit_id", ""),
            },
            gauge_info={
                "enabled": bool(getattr(self, "sim_gauge_enabled", False)) is False,
                "port": getattr(self.gauge_worker, "port", None) if getattr(self, "gauge_worker", None) is not None else None,
            },
        )

    def _make_validation_repository(self) -> ValidationRepository:
        return ValidationRepository(
            app_root_dir=self._app_root_dir(),
            software_version=str(SOFTWARE_VERSION),
        )

    def get_calibration_snapshot(self) -> CalibrationSnapshot:
        """Build a workflow-facing calibration snapshot from current app state.

        The repository provides the persisted base values, while current UI/recipe
        state may override them so measurement flow can consume one explicit
        snapshot object instead of reading scattered Tk variables.
        """
        try:
            base = self.calibration_repository.load_snapshot()
        except Exception:
            base = CalibrationSnapshot()

        def _read_float_var(name: str, fallback: float | None) -> float | None:
            try:
                var = getattr(self, name, None)
                if var is None:
                    return fallback
                raw = var.get() if hasattr(var, "get") else var
                text = str(raw).strip()
                if text in ("", "--", "None", "nan", "NaN"):
                    return fallback
                return float(text)
            except Exception:
                return fallback

        def _read_text_var(name: str, fallback: str) -> str:
            try:
                var = getattr(self, name, None)
                if var is None:
                    return str(fallback or "")
                raw = var.get() if hasattr(var, "get") else var
                text = str(raw).strip()
                return text if text else str(fallback or "")
            except Exception:
                return str(fallback or "")

        def _read_bool_var(name: str, fallback: bool) -> bool:
            try:
                var = getattr(self, name, None)
                if var is None:
                    return bool(fallback)
                raw = var.get() if hasattr(var, "get") else var
                if isinstance(raw, str):
                    text = raw.strip().lower()
                    if text in ("1", "true", "yes", "y", "on"):
                        return True
                    if text in ("0", "false", "no", "n", "off", ""):
                        return False
                return bool(raw)
            except Exception:
                return bool(fallback)

        try:
            recipe = self.get_recipe_copy()
        except Exception:
            recipe = getattr(self, "recipe", Recipe())

        try:
            recipe_id_single_k = float(getattr(recipe, "id_single_k", base.id_single_k) or base.id_single_k)
        except Exception:
            recipe_id_single_k = float(base.id_single_k)
        try:
            recipe_id_single_b = float(getattr(recipe, "id_single_b", base.id_single_b_mm) or base.id_single_b_mm)
        except Exception:
            recipe_id_single_b = float(base.id_single_b_mm)

        od_out1_map = _read_text_var("odcal_map_out1_var", base.od_out1_map or "L").upper()
        if od_out1_map not in ("L", "R"):
            od_out1_map = "L"

        od_request_cmd = _read_text_var("odcal_cmd_var", base.od_request_cmd or "")

        return CalibrationSnapshot(
            od_b_active_mm=float(_read_float_var("odcal_B_active_var", base.od_b_active_mm) or 0.0),
            od_out1_map=od_out1_map,
            od_d_ref_mm=_read_float_var("odcal_dref_var", base.od_d_ref_mm),
            od_request_cmd=od_request_cmd,
            id_delta_c_mm=float(_read_float_var("idcal_delta_active_var", base.id_delta_c_mm) or 0.0),
            id_d_ref_mm=_read_float_var("idcal_dref_var", base.id_d_ref_mm),
            id_single_enabled=_read_bool_var("id_single_enable_var", bool(getattr(recipe, "id_single_enable", base.id_single_enabled))),
            id_single_k=float(_read_float_var("id_single_k_var", recipe_id_single_k) or 1.0),
            id_single_b_mm=float(_read_float_var("id_single_b_var", recipe_id_single_b) or 0.0),
            id_single_d_ref_mm=_read_float_var("id_single_cal_dref_var", base.id_single_d_ref_mm),
        )

    def _build_run_context_for_export(
        self,
        *,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        status: str = "DONE",
    ) -> RunContext:
        """Build the current run context used by repository-backed exports."""
        self._ensure_run_identity()
        if not self._run_serial or not self._run_id or not self._run_start_ts:
            raise ValueError("未生成流水号/RunId，无法导出。")

        try:
            recipe = self.get_recipe_copy()
        except Exception:
            recipe = Recipe()

        try:
            summary = self._calc_run_summary()
        except Exception as e:
            summary = {"ok": False, "reason": f"异常: {e}"}

        try:
            length_result = dict(self._run_len_result or {}) if isinstance(self._run_len_result, dict) else None
        except Exception:
            length_result = None

        _start = float(start_ts if start_ts is not None else self._run_start_ts)
        _end = float(end_ts if end_ts is not None else (self._run_end_ts or time.time()))

        return RunContext(
            identity=RunIdentity(
                serial=str(self._run_serial),
                run_id=str(self._run_id),
                started_at_ts=_start,
            ),
            recipe=recipe,
            calibration=self.get_calibration_snapshot(),
            rows=list(self._auto_rows or []),
            raw_points=list(self._auto_raw_points or []),
            section_coverage=dict(self._section_cov_info or {}),
            length_result=length_result,
            summary=dict(summary or {}),
            finished_at_ts=_end,
            status=str(status or ""),
        )

    def _prepare_new_run(self) -> None:
        """Allocate a new Serial/RunId for the next Auto measurement."""
        try:
            recipe_name = str(getattr(self.recipe, "name", "默认配方") or "默认配方")
        except Exception:
            recipe_name = "默认配方"
        session = self._run_session
        serial = self._next_serial(recipe_name)
        session.serial = serial
        session.run_id = str(uuid.uuid4())
        session.start_ts = float(time.time())
        session.end_ts = None
        self._auto_export_done = False
        self._last_run_export_path = None
        # reset caches for this run
        session.rows.clear()
        session.raw_points.clear()
        session.summary_cache.clear()
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
            self.meas_start_var.set(_dt.datetime.fromtimestamp(float(session.start_ts)).strftime('%H:%M:%S'))
            self.meas_elapsed_var.set('00:00:00')
        except Exception:
            pass
        self._reset_summary_extrema()
        self._last_straight_od = None
        self._last_straight_id = None
        self._last_axis_dist = None

        # auto length result cache (per-run)
        self._run_len_result = None


    def _ensure_run_identity(self) -> None:
        """Ensure run_serial/run_id/run_start_ts exist before export.

        Some UI events (e.g. AutoFlow 'auto_clear') should only clear result tables; however,
        to make the system robust, exporting will best-effort allocate missing identity fields.
        """
        session = self._run_session
        if not session.start_ts:
            session.start_ts = float(time.time())
        if not session.run_id:
            session.run_id = str(uuid.uuid4())
        if not session.serial:
            try:
                recipe_name = str(getattr(self.recipe, "name", "默认配方") or "默认配方")
            except Exception:
                recipe_name = "默认配方"
            session.serial = self._next_serial(recipe_name)
            try:
                self.pipe_sn_var.set(session.serial)
            except Exception:
                pass
            try:
                self.meas_seq_var.set(str(session.serial).split('-')[-1])
            except Exception:
                pass

        # main-screen start time (best effort)
        try:
            if session.start_ts and hasattr(self, "meas_start_var"):
                import datetime as _dt
                self.meas_start_var.set(_dt.datetime.fromtimestamp(float(session.start_ts)).strftime('%H:%M:%S'))
        except Exception:
            pass



    def _format_cov_info(self, info: dict) -> str:
        cov = info.get("cov", None)
        miss = info.get("miss", None)
        reason = str(info.get("reason", "") or "")
        revs = info.get("revs", None)
        elapsed = info.get("elapsed", None)

        cov_od = info.get("cov_od", None)
        cov_id = info.get("cov_id", None)
        n_od = info.get("n_od", None)
        n_id = info.get("n_id", None)

        reason_txt = ""
        if reason:
            mapping = {
                "COV": "覆盖率达标",
                "TIMEOUT": "超时退出",
                "REV": "圈数到达",
            }
            reason_txt = mapping.get(reason.upper(), reason)

        if cov is None and (cov_od is None and cov_id is None):
            return "采样覆盖率：--"

        # Split-aware formatting: show OD/ID separately when available
        if (cov_od is not None) or (cov_id is not None):
            parts = ["采样覆盖率："]
            if cov_od is not None:
                try:
                    od_txt = f"OD {float(cov_od) * 100:.1f}%"
                except Exception:
                    od_txt = f"OD {cov_od}"
                if n_od is not None:
                    try:
                        od_txt += f"(n={int(n_od)})"
                    except Exception:
                        pass
                parts.append(od_txt)
            if cov_id is not None:
                try:
                    id_txt = f"ID {float(cov_id) * 100:.1f}%"
                except Exception:
                    id_txt = f"ID {cov_id}"
                if n_id is not None:
                    try:
                        id_txt += f"(n={int(n_id)})"
                    except Exception:
                        pass
                parts.append(id_txt)
            # join OD/ID parts with separator
            parts = [" | ".join(parts)]
        else:
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
            tree = self._main_ui_widget('result_tree')
            if tree is None:
                return
            vals = list(tree.item(iid, "values") or [])
        except Exception:
            return

        # base measurement columns count (keep in sync with table definition)
        base_n = 11
        if len(vals) < base_n:
            return
        cov_cols = list(self._format_cov_cols(info))
        new_vals = tuple(vals[:base_n] + cov_cols)
        try:
            tree.item(iid, values=new_vals)
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
            tree = self._main_ui_widget('result_tree')
            if tree is None:
                return
            sel = tree.selection()
            if not sel:
                self._selected_sec_idx = None
                # fallback to current section (or keep last shown)
                if self._auto_cur_sec_idx is not None:
                    self._show_cov_for_section(int(self._auto_cur_sec_idx))
                return

            iid = sel[0]
            vals = tree.item(iid, "values")
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
        lbl_actpos = self._axis_ui_widget('lbl_actpos', ax)
        if lbl_actpos is not None:
            lbl_actpos.config(text=f"Act_Pos(abs): {act_pos:.6f}")
        lbl_uipos = self._axis_ui_widget('lbl_uipos', ax)
        if lbl_uipos is not None:
            lbl_uipos.config(
                text=f"UI_Pos(相对): {ui_pos:.3f}    (ZeroAbs={self.ui_coord.zero_abs:.3f}, sign={self.ui_coord.sign:+d})"
            )

        err = int(getattr(ac, 'err', 0) or 0)
        warn = int(getattr(ac, 'warn', 0) or 0)
        sts = int(getattr(ac, 'sts', 0) or 0)
        st_id = int(getattr(ac, 'st_id', 0) or 0)
        seq = int(getattr(ac, 'seq', 0) or 0)
        seq_ack = int(getattr(ac, 'seq_ack', 0) or 0)

        lbl_err = self._axis_ui_widget('lbl_err', ax)
        if lbl_err is not None:
            lbl_err.config(text=f"ErrCode: {err}    Warn: {warn}")
        lbl_sts = self._axis_ui_widget('lbl_sts', ax)
        if lbl_sts is not None:
            lbl_sts.config(text=f"Sts(raw_state): {sts}    (0..8)")
        lbl_stid = self._axis_ui_widget('lbl_stid', ax)
        if lbl_stid is not None:
            lbl_stid.config(text=f"St_ID: {st_id}    Seq/Ack: {seq}/{seq_ack}")
        lbl_cmd = self._axis_ui_widget('lbl_cmd', ax)
        if lbl_cmd is not None:
            lbl_cmd.config(text=f"Cmd: 0x{int(getattr(ac, 'cmd', 0) or 0):04X}")
        lbl_flags = self._axis_ui_widget('lbl_flags', ax)
        if lbl_flags is not None:
            lbl_flags.config(text="")

        # UI显示使能：Sts==0 视为未使能，其余视为已使能（含错误态）
        # 为避免用户点击 Enable 后在反馈尚未更新前被刷新逻辑立即“打回”，
        # 在短暂的 pending 窗口内不强制覆盖 power_var。
        pend_t = 0.0
        try:
            pend_t = float(self._power_cmd_pending[ax])
        except Exception:
            pend_t = 0.0
        if (time.time() - pend_t) > 0.6:
            power_var = self._axis_ui_power_var(ax)
            if power_var is not None:
                power_var.set(1 if sts != 0 else 0)

        # keep teach panel synced
        self._refresh_teach_pos()

    def _on_power_toggle(self):
        ax = self._axis()
        power_var = self._axis_ui_power_var(ax)
        if power_var is None:
            return
        want_en = 1 if int(power_var.get() or 0) else 0

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
        ent_pos = self._axis_ui_widget('ent_pos', ax)
        try:
            pos = float(ent_pos.get().strip())
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return
        if int(ax) == 3:
            self._log_ax3_speed_trace("manual_ax3_movea_pre")
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
        if int(ax) == 3:
            self._log_ax3_speed_trace("manual_ax3_mover_pre")
        self._pulse_cmd_bits(ax, CMD_MOVER_REQ)

    def _do_vel_start(self):
        ax = self._axis()
        # write params (Vel_VelMove etc.)
        self._write_axis_params(ax)
        # VelMove is LEVEL command
        if int(ax) == 3:
            self._log_ax3_speed_trace("manual_ax3_velmove_pre")
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


__all__ = ["AppHost", "SOFTWARE_VERSION"]
