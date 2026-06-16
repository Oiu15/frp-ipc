# ./application/app_host.py
from __future__ import annotations

import numpy as np
"""FRP 管检测 IPC 应用（Tkinter）。

本文件保留“应用层/编排层”的职责：
- Tk 主线程：UI、事件回调、状态刷新
- 驱动层：PLC(Modbus TCP) 与测径仪(Serial) 的后台线程
- 工作流层：AutoFlowOrchestrator 与 AutoFlow 后台执行器

解耦原则：
- 协议常量与地址：./config/addresses.py
- 数据模型：./core/models.py
- IO 驱动：./drivers/*
- 自动测量流程：./frp_workflow/autoflow_orchestrator.py 与 ./frp_workflow/autoflow_executor.py
- UI 构建：./ui/screens/*
"""

import queue
import threading
import time
import os
from pathlib import Path
import math
import inspect
import logging

from utils.logger import init_log, log
from utils.perf import PerfAggregator, ns_to_ms
from typing import Any, List, Optional, Tuple, Iterable

import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkfont

from application.host.confirm import HostConfirmMixin
from application.host.keytest import HostKeytestMixin
from application.host.export import HostExportMixin
from application.host.identity import HostIdentityMixin
from application.host.ui import HostUIMixin
from application.host.validation import HostValidationMixin
from application.host.calibration.gauge_connection import HostGaugeConnectionMixin
from application.host.measurement.length import HostLengthMeasurementMixin
from application.host.recipe import HostRecipeMixin
from application.host.teach import HostTeachMixin
from application.host.main_view import HostMainViewMixin
from application.host.calibration.axis import HostAxisCalibrationMixin
from application.host.calibration.od import HostOdCalibrationMixin
from application.host.ui_state_compat import UiStateCompatMixin
from application.host.calibration.state import AxisCalibrationState
from application.handlers.actions import (
    CallbackAxisViewActions,
    CallbackDeviceStateActions,
    CallbackExportActions,
    CallbackRunStateActions,
    CallbackRunViewActions,
    CallbackWorkflowStatusActions,
)
from application.handlers.device import GaugeErrEventHandler, PlcErrEventHandler, PlcOkEventHandler
from application.handlers.measurement import (
    AutoCoverageEventHandler,
    AutoLenEventHandler,
    AutoPostcalcEventHandler,
    AutoProgressEventHandler,
    AutoRowEventHandler,
    AutoStateEventHandler,
)
from application.sync_reader import PlcSyncReader
from services.results_service import ResultsService
from services.history_export_coordinator import HistoryExportCoordinator
from services.run_export_coordinator import ExportKind, ExportResult, ExportStatus, RunExportCoordinator
from application.shell import AppDependencies, ApplicationShell
from events.pump import UiQueuePump
from domain.state import (
    CalibrationSnapshot,
    RunContext,
    RunIdentity,
    RunSession,
    RuntimeState,
    ValidationSession,
)
from events.dispatcher import UiEventDispatcher
from events.types import (
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
    CL_OUT1_WORD_OFF,
    CL_OUT1_UPD_WORD_OFF,
    CL_ID_WORD_OFF,
    CL_ID_UPD_WORD_OFF,
    CL_OUT_SCALE_MM,
    CL_OUT1_SCALE_MM,
    CL_OUT2_SCALE_MM,
    CL_OUT4_SCALE_MM,
    CL_OUT5_SCALE_MM,
    CL_ID_SCALE_MM,
    CL_OUT_INVALID,
    CL_OUT_STANDBY,
    CL_OUT_POS_OVER,
    CL_OUT_NEG_OVER,
    # legacy aliases (still referenced by some code paths)
    AXISCAL_MB_BASE,
    AXISCAL_WORDS,
    LINEAR_AXES,
    KEYTEST_X_POINTS,
    KEYTEST_Y_POINTS,
)

from core.models import AxisComm, UiCoord, Recipe, MeasureRow, AxisCal
from drivers.plc_client import (
    PlcWorker,
    CmdWriteRegs,
    CmdReadRegs,
    CmdSetPollProfile,
    CmdSetCmdMask,
    CmdPulseCmdMask,
    encode_float64_to_4regs,
    decode_float64_from_4regs,
)
from drivers.gauge_driver import GaugeWorker
from application.adapters.device_gateway import AppDeviceGateway
from application.adapters.ui_queue import WorkflowUiEventAdapter
from services.calibration_controller import CalibrationController
from services.calibration_service import CalibrationService
from services.id_single_calibration import IdSingleCalibrationService
from services.od_calibration import OdCalibrationService
from services.id_calibration import IdCalibrationService
from services.measurement_service import MeasurementController
from _version import SOFTWARE_VERSION
from modes.calibration_mode import CalibrationMode
from modes.mode_machine import ModeMachine
from modes.production_mode import ProductionMode
from modes.validation_mode import ValidationMode
from repositories.run_repository import RunRepository
from services.history_result_export_service import HistoryResultExportService
from repositories.settings_repository import SettingsRepository
from core.serial_service import (
    CUSTOM_KEYS,
    build_preview_serial,
    default_serial_template,
    normalize_serial_template,
    validate_serial_template,
)
from frp_workflow.autoflow_orchestrator import AutoFlowOrchestrator
from ui.state import UiState, UiStateDefaults

logger = logging.getLogger("frp.app")
recipe_logger = logging.getLogger("frp.recipe")
modbus_logger = logging.getLogger("frp.modbus")
ax3_trace_logger = logging.getLogger("frp.autoflow")
plc_perf_logger = logging.getLogger("frp.modbus.perf")



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


class AppHost(UiStateCompatMixin, HostIdentityMixin, HostUIMixin, HostGaugeConnectionMixin, HostLengthMeasurementMixin, HostRecipeMixin, HostTeachMixin, HostMainViewMixin, HostValidationMixin, HostAxisCalibrationMixin, HostOdCalibrationMixin, HostConfirmMixin, HostKeytestMixin, HostExportMixin, tk.Tk):
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
    ui: UiState

    axis_idx: tk.IntVar
    plc_status_var: tk.StringVar
    err_banner_var: tk.StringVar
    ip_var: tk.StringVar
    port_var: tk.StringVar
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

    sim_disp_var: tk.IntVar

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

    def _make_ui_state_defaults(self) -> UiStateDefaults:
        defaults = UiStateDefaults()
        recipe = getattr(self, "recipe", None)
        if recipe is None:
            return defaults

        try:
            defaults.pipe_len = str(getattr(recipe, "pipe_len_mm", defaults.pipe_len))
        except Exception:
            pass
        try:
            defaults.len_enable = bool(getattr(recipe, "len_enable", defaults.len_enable))
        except Exception:
            pass
        try:
            legacy_z = float(getattr(recipe, "len_z_low_approach", 1300.0))
            abs_appr = float(getattr(recipe, "len_low_approach_abs", 0.0) or 0.0)
            if abs_appr == 0.0:
                abs_appr = float(self.axis_cal.z_disp_to_abs(0, legacy_z))
            defaults.len_z_low_approach = str(abs_appr)
        except Exception:
            defaults.len_z_low_approach = "0.0"
        length_recipe_fields = {
            "len_low_search_dist": "len_low_search_dist",
            "len_high_search_dist": "len_high_search_dist",
            "len_search_vel": "len_search_vel",
            "len_search_timeout": "len_search_timeout_s",
            "len_tol": "len_tol_mm",
            "len_high_margin": "len_high_margin",
            "len_debounce_k": "len_debounce_k",
            "len_max_stale_ms": "len_max_stale_ms",
            "len_backoff": "len_backoff_mm",
        }
        for default_name, recipe_name in length_recipe_fields.items():
            try:
                setattr(defaults, default_name, str(getattr(recipe, recipe_name, getattr(defaults, default_name))))
            except Exception:
                pass
        try:
            defaults.teach_axes_mode = int(getattr(recipe, "teach_axes_mode", defaults.teach_axes_mode))
        except Exception:
            pass
        return defaults

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
        self._plc_sync_reader = PlcSyncReader(
            cmd_q=self.cmd_q,
            sync_reads=self._sync_reads,
            sync_reads_lock=self._sync_reads_lock,
            perf_sync_read=self._perf_sync_read,
            perf_ui_queue=self._perf_ui_queue,
            perf_group=self._sync_read_perf_group,
            flush_perf=self._flush_sync_read_perf_if_due,
        )

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
        self._keytest_y_points_has_read = False
        self._keytest_y_last_command_state = [0 for _ in range(len(KEYTEST_Y_POINTS))]

        # Axis calibration block (stored in PLC HD area)
        # Note: z_pos is IPC-only temporary shift, not written to PLC.
        self.axis_cal = AxisCal()  # sign defaults to -1
        self._axis_cal_state = AxisCalibrationState(self.axis_cal)
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
        self.ui = UiState.create(root=self, defaults=self._make_ui_state_defaults())

        # ------------------------------
        # OD Calibration (B) UI state
        # ------------------------------
        # 说明：
        # - B 值属于“工装/安装状态”的参数，不应散落在配方中。
        # - f2_0 主要落地 UI 布局与接口；采集/计算做最小可用实现（按定时采样）。
        # AX3 rotation speed for one-rev capture (deg/s)

        # Advanced sampling parameters (folded UI)
        # - 角度来源：AX3 编码器 / 无角度
        # - 去抖/滤波：用于降低抖动噪声（先对 sum=lL+lR 处理，后续可扩展到 v1/v2）
        # - 异常剔除阈值：基于 sigma 的离群点剔除
        # 凹陷/缺陷屏蔽（外径标定专用）
        # - TEMPLATE：使用“凹陷表(模板)”并对齐本次采样的相位后屏蔽角度段
        # - DYNAMIC：未学习模板时，按本次残差自动屏蔽最深的一段（可关闭）
        self._odcal_defect_template_mask: list[int] = [0] * 360  # 0/1, template coordinate
        # NOTE: do NOT use the same name as a method (Tk Button command binding will
        # grab the instance attribute first, which would mask the method).
        self._odcal_defect_learn_A_data: Optional[dict] = None

        # capture-time snapshot
        self._odcal_angle_enabled: bool = True
        self._odcal_filter_mode: str = "无"
        self._odcal_outlier_sigma: float = 3.0

        # Results
        # Quality stats (sum = lL+lR)

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

        # Summary split vars (main screen)
        # scheme-3 overall concentricity metrics
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
        self._flow_confirm_lock = threading.Lock()
        self._flow_confirm_token = None
        self._flow_confirm_evt = None
        self._flow_confirm_result = None
        self._flow_confirm_popup = None
        self._flow_confirm_confirm_cb = None
        self._flow_confirm_cancel_cb = None
        self._stack_light_state = None
        self._stack_light_buzzer_after_id = None

        # ------------------------------
        # Run/Export (MSA)
        # ------------------------------

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
        self._history_export_coordinator = HistoryExportCoordinator()
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

        self._device_ui_event_dispatcher = self._build_device_ui_event_dispatcher()
        self._measurement_ui_event_dispatcher = self._build_measurement_ui_event_dispatcher()
        self._ui_queue_pump = UiQueuePump(
            ui_q=self.ui_q,
            device_dispatcher=self._device_ui_event_dispatcher,
            measurement_dispatcher=self._measurement_ui_event_dispatcher,
            perf_ui_queue=self._perf_ui_queue,
            log_filter=LOG_UI_EVENT_FILTER,
        )
        self.results_service = ResultsService()
        self._run_export_coordinator = RunExportCoordinator(
            repository=self._make_run_repository,
            results_service=self.results_service,
            recipe_provider=self.get_recipe_copy,
            calibration_provider=self.get_calibration_snapshot,
            coverage_provider=lambda: dict(self._section_cov_info or {}),
        )
        self.calibration_service = CalibrationService()
        self.calibration_gateway = AppDeviceGateway(self)
        self.od_calibration_svc = OdCalibrationService(
            rotation=self.calibration_gateway,
            sensors=self.calibration_gateway,
            scheduler=self.calibration_gateway,
            state_sink=self.calibration_gateway,
            poll_profile=self.calibration_gateway,
            repository=self.calibration_repository,
        )
        self.id_calibration_svc = IdCalibrationService(
            rotation=self.calibration_gateway,
            sensors=self.calibration_gateway,
            scheduler=self.calibration_gateway,
            state_sink=self.calibration_gateway,
            poll_profile=self.calibration_gateway,
            repository=self.calibration_repository,
        )
        self.id_single_calibration_svc = IdSingleCalibrationService(
            rotation=self.calibration_gateway,
            sensors=self.calibration_gateway,
            scheduler=self.calibration_gateway,
            state_sink=self.calibration_gateway,
            poll_profile=self.calibration_gateway,
            repository=self.calibration_repository,
        )
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
            od_service=self.od_calibration_svc,
            id_service=self.id_calibration_svc,
            id_single_service=self.id_single_calibration_svc,
        )
        self.measurement_controller = MeasurementController(
            mode_machine=self.mode_machine,
        )
        self._init_presenters()
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

    def apply_plc_connection(self):
        return self._apply_conn()

    def start_measurement(self):
        return self.measurement_controller.start_measurement()

    def stop_measurement(self):
        return self.measurement_controller.stop_measurement()

    def clear_measurement_results(self):
        return self._refresh_measurement_display()

    def open_serial_template_settings(self) -> None:
        repo = SettingsRepository(app_root_dir=self._app_root_dir())
        try:
            template = repo.load_serial_template()
        except Exception:
            template = default_serial_template()
        fields: list[dict[str, Any]] = [dict(field) for field in template.get("fields", [])]
        custom_values: dict[str, str] = dict(template.get("custom_values", {}))

        win = tk.Toplevel(self)
        win.title("流水号模板设置")
        win.transient(self)
        win.geometry("720x520")

        field_labels = {
            "date": "日期(date)",
            "time": "时间(time)",
            "recipe": "配方名(recipe)",
            "seq": "序号(seq)",
            "customer": "客户名称(customer)",
            "batch": "批次(batch)",
            "work_order": "工单号(work_order)",
            "team": "班组(team)",
            "remark": "备注(remark)",
            "text": "自定义文本(text)",
        }
        field_choices = [
            ("system", "date"),
            ("system", "time"),
            ("system", "recipe"),
            ("system", "seq"),
            ("custom", "customer"),
            ("custom", "batch"),
            ("custom", "work_order"),
            ("custom", "team"),
            ("custom", "remark"),
            ("text", "text"),
        ]
        choice_labels = [field_labels[key] for _type, key in field_choices]
        choice_by_label = {field_labels[key]: (_type, key) for _type, key in field_choices}

        sep_value = str(template.get("separator", "-") or "-")
        sep_mode_var = tk.StringVar(value=(sep_value if sep_value in {"-", "_", " "} else "自定义"))
        custom_sep_var = tk.StringVar(value=("" if sep_value in {"-", "_", " "} else sep_value))
        add_field_var = tk.StringVar(value=choice_labels[0])
        value_var = tk.StringVar(value="")
        preview_var = tk.StringVar(value="")
        warning_var = tk.StringVar(value="")

        outer = ttk.Frame(win)
        outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(1, weight=1)

        sep_frame = ttk.Frame(outer)
        sep_frame.grid(row=0, column=0, sticky="we", pady=(0, 8))
        ttk.Label(sep_frame, text="分隔符").pack(side=tk.LEFT)
        sep_combo = ttk.Combobox(sep_frame, textvariable=sep_mode_var, values=["-", "_", " ", "自定义"], width=10, state="readonly")
        sep_combo.pack(side=tk.LEFT, padx=(8, 6))
        ttk.Entry(sep_frame, textvariable=custom_sep_var, width=12).pack(side=tk.LEFT)

        body = ttk.Frame(outer)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        tree = ttk.Treeview(body, columns=("enabled", "field", "value"), show="headings", height=12)
        tree.heading("enabled", text="启用")
        tree.heading("field", text="字段")
        tree.heading("value", text="值")
        tree.column("enabled", width=70, stretch=False)
        tree.column("field", width=220, stretch=True)
        tree.column("value", width=260, stretch=True)
        tree.grid(row=0, column=0, sticky="nsew")

        ysb = ttk.Scrollbar(body, orient=tk.VERTICAL, command=tree.yview)
        ysb.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=ysb.set)

        side = ttk.Frame(body)
        side.grid(row=0, column=2, sticky="ns", padx=(10, 0))
        ttk.Combobox(side, textvariable=add_field_var, values=choice_labels, state="readonly", width=22).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(side, text="添加字段", command=lambda: add_field()).pack(fill=tk.X, pady=3)
        ttk.Button(side, text="删除字段", command=lambda: delete_selected()).pack(fill=tk.X, pady=3)
        ttk.Button(side, text="上移", command=lambda: move_selected(-1)).pack(fill=tk.X, pady=3)
        ttk.Button(side, text="下移", command=lambda: move_selected(1)).pack(fill=tk.X, pady=3)
        ttk.Button(side, text="启用/禁用", command=lambda: toggle_selected()).pack(fill=tk.X, pady=3)

        edit = ttk.Frame(outer)
        edit.grid(row=2, column=0, sticky="we", pady=(8, 0))
        edit.grid_columnconfigure(1, weight=1)
        ttk.Label(edit, text="字段值").grid(row=0, column=0, sticky="w")
        ttk.Entry(edit, textvariable=value_var).grid(row=0, column=1, sticky="we", padx=(8, 8))
        ttk.Button(edit, text="应用到选中字段", command=lambda: apply_value()).grid(row=0, column=2, sticky="e")

        preview = ttk.LabelFrame(outer, text="预览")
        preview.grid(row=3, column=0, sticky="we", pady=(10, 0))
        preview.grid_columnconfigure(0, weight=1)
        ttk.Label(preview, textvariable=preview_var, font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=8, pady=(6, 2))
        ttk.Label(preview, textvariable=warning_var, foreground="#a15c00").grid(row=1, column=0, sticky="w", padx=8, pady=(0, 6))

        actions = ttk.Frame(outer)
        actions.grid(row=4, column=0, sticky="e", pady=(10, 0))
        ttk.Button(actions, text="恢复默认", command=lambda: reset_default()).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(actions, text="保存", command=lambda: save()).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(actions, text="关闭", command=win.destroy).pack(side=tk.LEFT)

        def selected_index() -> int | None:
            selected = tree.selection()
            if not selected:
                return None
            try:
                return int(tree.index(selected[0]))
            except Exception:
                return None

        def current_separator() -> str:
            mode = sep_mode_var.get()
            if mode == "自定义":
                return custom_sep_var.get()
            return mode

        def current_template() -> dict[str, Any]:
            return {
                "separator": current_separator(),
                "custom_values": dict(custom_values),
                "fields": [dict(field) for field in fields],
            }

        def refresh_tree(select_idx: int | None = None) -> None:
            for iid in tree.get_children():
                tree.delete(iid)
            for field in fields:
                key = str(field.get("key", ""))
                field_type = str(field.get("type", ""))
                value = ""
                if field_type == "custom":
                    value = str(custom_values.get(key, ""))
                elif field_type == "text":
                    value = str(field.get("value", ""))
                tree.insert("", tk.END, values=("是" if bool(field.get("enabled", True)) else "否", field_labels.get(key, key), value))
            if select_idx is not None and 0 <= select_idx < len(fields):
                iid = tree.get_children()[select_idx]
                tree.selection_set(iid)
                tree.focus(iid)
                update_value_entry()
            refresh_preview()

        def refresh_preview(*_args) -> None:
            try:
                recipe_name = str(getattr(self.recipe, "name", "default_recipe") or "default_recipe")
            except Exception:
                recipe_name = "default_recipe"
            try:
                result = build_preview_serial(current_template(), recipe_name=recipe_name)
                preview_var.set(result.serial)
                warning_var.set(result.warning)
            except Exception as exc:
                preview_var.set("")
                warning_var.set(f"模板无效: {exc}")

        def update_value_entry(_event=None) -> None:
            idx = selected_index()
            if idx is None:
                value_var.set("")
                return
            field = fields[idx]
            field_type = str(field.get("type", ""))
            key = str(field.get("key", ""))
            if field_type == "custom":
                value_var.set(str(custom_values.get(key, "")))
            elif field_type == "text":
                value_var.set(str(field.get("value", "")))
            else:
                value_var.set("")

        def add_field() -> None:
            field_type, key = choice_by_label.get(add_field_var.get(), ("system", "date"))
            item: dict[str, Any] = {"type": field_type, "key": key, "enabled": True}
            if field_type == "text":
                item["value"] = "TEXT"
            fields.append(item)
            refresh_tree(len(fields) - 1)

        def delete_selected() -> None:
            idx = selected_index()
            if idx is None:
                return
            del fields[idx]
            refresh_tree(min(idx, len(fields) - 1))

        def move_selected(delta: int) -> None:
            idx = selected_index()
            if idx is None:
                return
            new_idx = idx + int(delta)
            if new_idx < 0 or new_idx >= len(fields):
                return
            fields[idx], fields[new_idx] = fields[new_idx], fields[idx]
            refresh_tree(new_idx)

        def toggle_selected() -> None:
            idx = selected_index()
            if idx is None:
                return
            fields[idx]["enabled"] = not bool(fields[idx].get("enabled", True))
            refresh_tree(idx)

        def apply_value() -> None:
            idx = selected_index()
            if idx is None:
                return
            field = fields[idx]
            field_type = str(field.get("type", ""))
            key = str(field.get("key", ""))
            if field_type == "custom" and key in CUSTOM_KEYS:
                custom_values[key] = value_var.get()
            elif field_type == "text":
                field["value"] = value_var.get()
            refresh_tree(idx)

        def reset_default() -> None:
            default_template = default_serial_template()
            fields[:] = [dict(field) for field in default_template["fields"]]
            custom_values.clear()
            custom_values.update(default_template["custom_values"])
            sep_mode_var.set("-")
            custom_sep_var.set("")
            refresh_tree(0)

        def save() -> None:
            try:
                normalized = normalize_serial_template(current_template())
                validate_serial_template(normalized)
                repo.save_serial_template(normalized)
            except Exception as exc:
                messagebox.showerror("保存失败", str(exc), parent=win)
                return
            messagebox.showinfo("保存完成", "流水号模板设置已保存。", parent=win)
            win.destroy()

        try:
            sep_mode_var.trace_add("write", refresh_preview)
            custom_sep_var.trace_add("write", refresh_preview)
        except Exception:
            pass
        try:
            tree.bind("<<TreeviewSelect>>", update_value_entry)
        except Exception:
            pass
        refresh_tree(0)

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

        # Axis hard errors. Warnings remain visible on the axis debug page only.
        try:
            with self._snapshot_lock:
                axes = list(self._axis_snapshot)
            for i, ax in enumerate(axes):
                e = int(getattr(ax, "err", 0) or 0)
                if e:
                    msgs.append(f"AX{i} ERR={e}")
        except Exception:
            pass

        # Gauge connection error (if any)
        try:
            gauge_err_var = getattr(self, "gauge_err_var", None)
            gerr = str(gauge_err_var.get() if gauge_err_var is not None else "").strip()
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

    # Host state helpers shared by validation, control, and PLC polling.
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

    # =========================
    # Auto tab
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
        if not preserve_run:
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
        run_start_ts = self._run_start_ts
        if run_start_ts is None:
            try:
                self.meas_start_var.set("--")
                self.meas_elapsed_var.set("--")
            except Exception:
                pass
            return

        try:
            start_ts = float(run_start_ts)
        except Exception:
            return

        try:
            import datetime as _dt
            self.meas_start_var.set(_dt.datetime.fromtimestamp(start_ts).strftime("%H:%M:%S"))
        except Exception:
            pass

        try:
            run_end_ts = self._run_end_ts
            end_ts = float(run_end_ts) if run_end_ts is not None else float(time.time())
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
                    cur = str(self._run_session.message or '')
                    if cur in ('-', '', 'None'):
                        new_msg = f'汇总失败: {reason}'
                        self.auto_msg_var.set(new_msg)
                        self._run_session.message = new_msg
                    elif '汇总失败' not in cur:
                        new_msg = f'{cur} | 汇总失败: {reason}'
                        self.auto_msg_var.set(new_msg)
                        self._run_session.message = new_msg
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
            def _summary_text(value: object, *, scale: float = 1.0, suffix: str = "") -> str:
                if value is None:
                    return "--"
                if not isinstance(value, (str, int, float, np.number)):
                    return str(value)
                return f"{float(value) * scale:.3f}{suffix}"
            self.od_tilt_var.set(_summary_text(summary.get('od_tilt_deg'), suffix="\u00b0"))
            self.od_endoff_var.set(_summary_text(summary.get('od_end_off_mm'), suffix=" mm"))
            self.id_tilt_var.set(_summary_text(summary.get('id_tilt_deg'), suffix="\u00b0"))
            self.id_endoff_var.set(_summary_text(summary.get('id_end_off_mm'), suffix=" mm"))
            self.od_slope_var.set(_summary_text(summary.get('od_slope'), scale=1000.0, suffix=" mm/m"))
            self.id_slope_var.set(_summary_text(summary.get('id_slope'), scale=1000.0, suffix=" mm/m"))
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
        auto_thread = self._auto_thread
        auto_alive = bool(auto_thread is not None and auto_thread.is_alive())
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

    def _get_plc_sync_reader(self) -> PlcSyncReader:
        reader = self.__dict__.get("_plc_sync_reader", None)
        if not isinstance(reader, PlcSyncReader):
            reader = PlcSyncReader(
                cmd_q=self.cmd_q,
                sync_reads=self._sync_reads,
                sync_reads_lock=self._sync_reads_lock,
                perf_sync_read=self._perf_sync_read,
                perf_ui_queue=self._perf_ui_queue,
                perf_group=self._sync_read_perf_group,
                flush_perf=self._flush_sync_read_perf_if_due,
            )
            self._plc_sync_reader = reader
        return reader

    def _read_regs_sync(self, d_addr: int, count: int, timeout_s: float = 0.35) -> Optional[List[int]]:
        """Synchronous Modbus holding-register read via PlcWorker."""
        return self._get_plc_sync_reader().read_regs_sync(d_addr, count, timeout_s=timeout_s)

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
            if id_mm is not None:
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

    def _require_axis_ui_widget(self, name: str, axis: Optional[int] = None) -> Any:
        widget = self._axis_ui_widget(name, axis)
        if widget is None:
            raise RuntimeError(f"{self.__class__.__name__}: missing axis UI field '{name}'")
        return widget

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
            vel_mover = self._parse_float(self._require_axis_ui_widget('ent_vel_mover').get(), vel_movea)
            vel_jog = self._parse_float(self._require_axis_ui_widget('ent_vel_jog').get(), 80.0)
            vel_velmove = self._parse_float(self._require_axis_ui_widget('ent_vel_velmove').get(), 200.0)
            acc = self._parse_float(self._require_axis_ui_widget('ent_acc').get(), 200.0)
            dec = self._parse_float(self._require_axis_ui_widget('ent_dec').get(), 200.0)
            jerk = self._parse_float(self._require_axis_ui_widget('ent_jerk').get(), 500.0)

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

    def _write_axis_params(self, axis: int, dir_mover_override: int | None = None):
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
        if dir_mover_override is not None:
            dir_mover = int(dir_mover_override)

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

    def _get_run_view_actions(self) -> CallbackRunViewActions:
        actions = self.__dict__.get("_run_view_actions", None)
        if not isinstance(actions, CallbackRunViewActions):
            actions = CallbackRunViewActions(
                set_auto_progress_cb=self._set_auto_progress_view,
                set_auto_done_cb=self._set_auto_done_view,
                project_auto_len_result_cb=self._project_auto_len_result,
                show_section_coverage_cb=self._show_section_coverage,
                set_auto_state_cb=self._set_auto_state_view,
                refresh_done_run_summary_cb=self._refresh_done_run_summary_and_export,
            )
            self._run_view_actions = actions
        return actions

    def _get_run_state_actions(self) -> CallbackRunStateActions:
        actions = self.__dict__.get("_run_state_actions", None)
        if not isinstance(actions, CallbackRunStateActions):
            actions = CallbackRunStateActions(
                set_current_section_index_cb=self._set_auto_current_section_index,
                cache_auto_len_result_cb=self._cache_auto_len_result,
                cache_section_coverage_cb=self._cache_section_cov_info,
                should_show_section_coverage_cb=self._should_show_section_coverage,
                update_run_status_cb=self._update_run_status_message,
                freeze_run_end_ts_if_missing_cb=self._freeze_run_end_ts_if_missing,
                append_result_row_cb=self._append_result_row,
                apply_postcalc_result_cb=self._apply_postcalc_result_payload,
            )
            self._run_state_actions = actions
        return actions

    def _get_workflow_status_actions(self) -> CallbackWorkflowStatusActions:
        actions = self.__dict__.get("_workflow_status_actions", None)
        if not isinstance(actions, CallbackWorkflowStatusActions):
            actions = CallbackWorkflowStatusActions(
                sync_production_workflow_state_cb=self._sync_production_workflow_state,
                refresh_stack_light_for_state_cb=self._refresh_stack_light_for_state,
            )
            self._workflow_status_actions = actions
        return actions

    def _get_export_actions(self) -> CallbackExportActions:
        actions = self.__dict__.get("_export_actions", None)
        if not isinstance(actions, CallbackExportActions):
            actions = CallbackExportActions(
                trigger_terminal_export_cb=self._trigger_terminal_export,
                maybe_retry_terminal_export_cb=self._maybe_retry_terminal_export,
            )
            self._export_actions = actions
        return actions

    def _get_device_state_actions(self) -> CallbackDeviceStateActions:
        actions = self.__dict__.get("_device_state_actions", None)
        if not isinstance(actions, CallbackDeviceStateActions):
            actions = CallbackDeviceStateActions(
                set_plc_ok_status_cb=self._set_plc_ok_status,
                set_plc_error_status_cb=self._set_plc_error_status,
                update_axis_snapshot_from_plc_cb=self._update_axis_snapshot_from_plc,
                update_cl_cache_and_ui_cb=self._update_cl_cache_and_ui,
                set_gauge_error_cb=self._set_gauge_error,
            )
            self._device_state_actions = actions
        return actions

    def _get_axis_view_actions(self) -> CallbackAxisViewActions:
        actions = self.__dict__.get("_axis_view_actions", None)
        if not isinstance(actions, CallbackAxisViewActions):
            actions = CallbackAxisViewActions(
                update_keytest_from_plc_cb=self._update_keytest_from_plc,
                refresh_axis_panel_from_snapshot_cb=self._refresh_axis_panel_from_snapshot,
                handle_axis_cal_one_shot_read_cb=self._handle_axis_cal_one_shot_read,
                refresh_axis_cal_status_cb=self.axis_cal_refresh_status,
            )
            self._axis_view_actions = actions
        return actions

    def _get_auto_progress_event_handler(self) -> AutoProgressEventHandler:
        handler = self.__dict__.get("_auto_progress_event_handler", None)
        if not isinstance(handler, AutoProgressEventHandler):
            handler = AutoProgressEventHandler(self._get_run_state_actions(), self._get_run_view_actions())
            self._auto_progress_event_handler = handler
        return handler

    def _get_auto_coverage_event_handler(self) -> AutoCoverageEventHandler:
        handler = self.__dict__.get("_auto_coverage_event_handler", None)
        if not isinstance(handler, AutoCoverageEventHandler):
            handler = AutoCoverageEventHandler(self._get_run_state_actions(), self._get_run_view_actions())
            self._auto_coverage_event_handler = handler
        return handler

    def _get_auto_len_event_handler(self) -> AutoLenEventHandler:
        handler = self.__dict__.get("_auto_len_event_handler", None)
        if not isinstance(handler, AutoLenEventHandler):
            handler = AutoLenEventHandler(self._get_run_state_actions(), self._get_run_view_actions())
            self._auto_len_event_handler = handler
        return handler

    def _get_auto_state_event_handler(self) -> AutoStateEventHandler:
        handler = self.__dict__.get("_auto_state_event_handler", None)
        if not isinstance(handler, AutoStateEventHandler):
            handler = AutoStateEventHandler(
                self._get_run_state_actions(),
                self._get_run_view_actions(),
                self._get_workflow_status_actions(),
                self._get_export_actions(),
            )
            self._auto_state_event_handler = handler
        return handler

    def _get_auto_row_event_handler(self) -> AutoRowEventHandler:
        handler = self.__dict__.get("_auto_row_event_handler", None)
        if not isinstance(handler, AutoRowEventHandler):
            handler = AutoRowEventHandler(self._get_run_state_actions())
            self._auto_row_event_handler = handler
        return handler

    def _get_auto_postcalc_event_handler(self) -> AutoPostcalcEventHandler:
        handler = self.__dict__.get("_auto_postcalc_event_handler", None)
        if not isinstance(handler, AutoPostcalcEventHandler):
            handler = AutoPostcalcEventHandler(
                self._get_run_state_actions(),
                self._get_run_view_actions(),
                self._get_export_actions(),
            )
            self._auto_postcalc_event_handler = handler
        return handler

    def _get_gauge_err_event_handler(self) -> GaugeErrEventHandler:
        handler = self.__dict__.get("_gauge_err_event_handler", None)
        if not isinstance(handler, GaugeErrEventHandler):
            handler = GaugeErrEventHandler(self._get_device_state_actions())
            self._gauge_err_event_handler = handler
        return handler

    def _get_plc_err_event_handler(self) -> PlcErrEventHandler:
        handler = self.__dict__.get("_plc_err_event_handler", None)
        if not isinstance(handler, PlcErrEventHandler):
            handler = PlcErrEventHandler(self._get_device_state_actions())
            self._plc_err_event_handler = handler
        return handler

    def _get_plc_ok_event_handler(self) -> PlcOkEventHandler:
        handler = self.__dict__.get("_plc_ok_event_handler", None)
        if not isinstance(handler, PlcOkEventHandler):
            handler = PlcOkEventHandler(
                self._get_device_state_actions(),
                self._get_axis_view_actions(),
                self._get_workflow_status_actions(),
            )
            self._plc_ok_event_handler = handler
        return handler

    def _build_device_ui_event_dispatcher(self) -> UiEventDispatcher:
        return UiEventDispatcher(
            {
                PlcOkEvent: self._get_plc_ok_event_handler().handle,
                PlcErrEvent: self._get_plc_err_event_handler().handle,
                PlcGiveupEvent: self._handle_plc_giveup_event,
                PlcManualEvent: self._handle_plc_manual_event,
                PlcReadEvent: self._handle_plc_read_event,
                GaugeConnEvent: self._handle_gauge_conn_event,
                GaugeTxEvent: self._handle_gauge_tx_event,
                GaugeOkEvent: self._handle_gauge_ok_event,
                GaugeRawEvent: self._handle_gauge_raw_event,
                GaugeErrEvent: self._get_gauge_err_event_handler().handle,
            }
        )

    def _build_measurement_ui_event_dispatcher(self) -> UiEventDispatcher:
        return UiEventDispatcher(
            {
                OpConfirmShowEvent: self._handle_op_confirm_show_event,
                OpConfirmCloseEvent: self._handle_op_confirm_close_event,
                "flow_confirm_show": getattr(self, "_handle_flow_confirm_show_event", lambda _payload: None),
                "flow_confirm_close": getattr(self, "_handle_flow_confirm_close_event", lambda _payload: None),
                AutoClearEvent: self._handle_auto_clear_event,
                AutoLenEvent: self._get_auto_len_event_handler().handle,
                AutoProgressEvent: self._get_auto_progress_event_handler().handle,
                AutoCoverageEvent: self._get_auto_coverage_event_handler().handle,
                AutoStraightnessEvent: self._handle_auto_straightness_event,
                AutoPostcalcEvent: self._get_auto_postcalc_event_handler().handle,
                AutoRawPointsEvent: self._handle_auto_raw_points_event,
                AutoRowEvent: self._get_auto_row_event_handler().handle,
                AutoStateEvent: self._get_auto_state_event_handler().handle,
            }
        )

    def _set_auto_current_section_index(self, section_index: int) -> None:
        self._auto_cur_sec_idx = int(section_index)

    def _set_auto_progress_view(self, idx: int, total: int) -> None:
        self.auto_progress_var.set(f"当前截面: {int(idx) + 1} / 总截面: {int(total)}")

    def _set_auto_done_view(self, completed: bool) -> None:
        self.auto_done_var.set("测量完成: 是" if completed else "测量完成: 否")

    def _should_show_section_coverage(self, section_index: int | None) -> bool:
        return (
            self._selected_sec_idx is None
            or section_index is None
            or int(self._selected_sec_idx) == int(section_index)
        )

    def _show_section_coverage(self, info: dict[str, Any]) -> None:
        self.cov_var.set(self._format_cov_info(info))

    def _set_auto_state_view(self, state: str, message: str) -> None:
        self.auto_state_var.set(str(state))
        self.auto_msg_var.set(str(message))

    def _update_run_status_message(self, state: str, message: str) -> None:
        self._run_session.status = str(state)
        self._run_session.message = str(message)

    def _sync_production_workflow_state(self, state: str, message: str) -> None:
        try:
            self.mode_machine.sync_production_workflow_state(str(state), str(message))
        except Exception:
            pass

    def _trigger_terminal_export(self, status: str, completed: bool) -> None:
        try:
            self._trigger_run_export(status=str(status), completed=bool(completed))
        except TypeError:
            self._trigger_run_export()

    def _maybe_retry_terminal_export(self) -> None:
        status = str(getattr(self._run_session, "status", "") or "").upper()
        if status not in {"DONE", "ERR", "STOP", "ABORTED"}:
            return
        last_result = self.__dict__.get("_last_run_export_result", None)
        if not isinstance(last_result, ExportResult):
            return
        if last_result.status not in {ExportStatus.PENDING, ExportStatus.FAILED}:
            return
        self._maybe_trigger_completed_export()

    def _apply_postcalc_result_payload(self, payload: Any) -> None:
        self._apply_run_summary_payload(payload)
        self._apply_postcalc_eccentricity(payload)

    def _set_gauge_error(self, message: str) -> None:
        self.gauge_err_var.set(str(message))

    def _set_plc_ok_status(self) -> None:
        self.plc_status_var.set(
            f"PLC: OK   {time.strftime('%H:%M:%S')}   ip={self.worker.ip}:{self.worker.port}   unit={self.worker.unit_id}"
        )

    def _set_plc_error_status(
        self, err: str, retry: int | None, max_attempts: int | None, backoff_s: float | None
    ) -> None:
        if retry is not None and max_attempts is not None and backoff_s is not None:
            self.plc_status_var.set(
                f"PLC: ERROR  {err}   (retry {retry}/{max_attempts}, next in {backoff_s}s)"
            )
        else:
            self.plc_status_var.set(f"PLC: ERROR   {err}")

    def _update_axis_snapshot_from_plc(self, event: PlcOkEvent) -> None:
        with self._snapshot_lock:
            self._axis_snapshot = list(event.axes)

    def _update_cl_cache_and_ui(self, event: PlcOkEvent) -> None:
        payload = event.to_payload()
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

            self.cl_out1_var.set(_fmt(out1_mm, out1_raw, 4))
            self.cl_out2_var.set(_fmt(out2_mm, out2_raw, 4))
            self.cl_out3_var.set(_fmt(out3_mm, out3_raw, 3))
            self.cl_out4_var.set(_fmt(out4_mm, out4_raw, 3))
            self.cl_out5_var.set(_fmt(out5_mm, out5_raw, 4))

            self.cl_out1_cnt_var.set("--" if out1_cnt is None else str(int(out1_cnt)))
            self.cl_out2_cnt_var.set("--" if out2_cnt is None else str(int(out2_cnt)))
            self.cl_out3_cnt_var.set("--" if out3_cnt is None else str(int(out3_cnt)))
            self.cl_out4_cnt_var.set("--" if out4_cnt is None else str(int(out4_cnt)))
            self.cl_out5_cnt_var.set("--" if out5_cnt is None else str(int(out5_cnt)))

            self.cl_id_var.set(self.cl_out4_var.get())
            self.cl_cnt_var.set("--" if out4_cnt is None else str(int(out4_cnt)))

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

            if out4_cnt is not None and out4_mm is not None:
                if self._last_cl_cnt is None or int(out4_cnt) != int(self._last_cl_cnt):
                    self._last_cl_cnt = int(out4_cnt)
                    self._id_samples.append(float(out4_mm))
                    self._refresh_id_stats()
        except Exception:
            pass

    def _update_keytest_from_plc(self, event: PlcOkEvent) -> None:
        payload = event.to_payload()
        try:
            self._keytest_apply_bits(
                payload.get("keytest_x_bits", None),
                payload.get("keytest_y_bits", None),
            )
        except Exception:
            pass

    def _refresh_axis_panel_from_snapshot(self) -> None:
        self._refresh_axis_panel()

    def _handle_axis_cal_one_shot_read(self) -> None:
        if getattr(self, "_dbg_axis_cal_sent", False):
            return
        try:
            self.cmd_q.put(CmdReadRegs(AXISCAL_MB_BASE, AXISCAL_WORDS, "axis_cal"))
            self._dbg_axis_cal_sent = True
            print(f"[axis_cal] request read(after plc_ok): addr={AXISCAL_MB_BASE} count={AXISCAL_WORDS}")
        except Exception as e:
            print(f"[axis_cal] enqueue read failed(after plc_ok): {e}")

    def _handle_plc_ok_event(self, event: PlcOkEvent) -> None:
        self._get_plc_ok_event_handler().handle(event)
        payload = event.to_payload()
        return
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
            self._refresh_stack_light_for_state()
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
        self._get_plc_err_event_handler().handle(event)

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

        if self._get_plc_sync_reader().handle_plc_read_payload(payload):
            return

        # f2/f3/f4: parse axis calibration block if requested
        if tag == "axis_cal" or tag == "axis_cal_verify":
            try:
                cal = AxisCal.from_regs(regs)

                if tag == "axis_cal_verify":
                    exp = getattr(self, "_axis_cal_write_expect_regs", None)
                    ok = self._get_axis_calibration_state().matches_expected_regs(regs) or (
                        exp is not None and list(exp) == list(regs)
                    )

                    if ok:
                        # success: accept PLC readback and refresh UI
                        self._set_axis_cal(cal)
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
                            "写入成功",
                        )
                        print(
                            "[axis_cal] verify OK; readback matches written regs. "
                            f"sign={cal.sign} off_ax0={cal.off_ax0:.6f} off_ax1={cal.off_ax1:.6f} "
                            f"off_ax2={cal.off_ax2:.6f} off_ax4={cal.off_ax4:.6f} "
                            f"b14={cal.b14:.6f} b2={cal.b2:.6f} keepout_w={cal.keepout_w:.6f}"
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
                                "keepout_w",
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
                    self._set_axis_cal_write_expect_regs(None)

                else:
                    # Normal read: keep in-memory copy and refresh calibration UI
                    self._set_axis_cal(cal)
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
                        f"b14={cal.b14:.6f} b2={cal.b2:.6f} keepout_w={cal.keepout_w:.6f}"
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
        od1 = event.od
        od2 = event.od2
        j1 = str(event.judge or "").strip()
        j2 = str(event.judge2 or "").strip()

        raw = str(event.raw or "").strip()
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
        event.to_payload()
        # only update if no parsed value is flowing
        pass

    def _handle_gauge_err_event(self, event: GaugeErrEvent) -> None:
        self._get_gauge_err_event_handler().handle(event)

    def _handle_auto_clear_event(self, event: AutoClearEvent) -> None:
        event.to_payload()
        # AutoFlow sends auto_clear at the beginning of a run; do NOT wipe run identity/timestamps.
        self._auto_clear_ui(preserve_run=True)

    def _project_auto_len_result(self, payload: dict[str, Any]) -> None:
        p = payload if isinstance(payload, dict) else {}
        ok = bool(p.get("ok", False))
        skipped = bool(p.get("skipped", False))
        reason = str(p.get("reason", "") or "")
        z_low = p.get("z_low", None)
        z_high = p.get("z_high", None)
        length_mm = p.get("length_mm", None)

        try:
            if hasattr(self, 'len_meas_var'):
                enabled = bool(p.get('enabled', False))
                if not enabled:
                    self.len_meas_var.set("未启用")
                elif skipped:
                    self.len_meas_var.set(f"跳过（{reason}）" if reason else "跳过")
                elif ok and length_mm is not None:
                    try:
                        exp = float(getattr(self.recipe, 'pipe_len_mm', 0.0) or 0.0)
                    except Exception:
                        exp = 0.0
                    try:
                        tol = float(getattr(self.recipe, 'len_tol_mm', 0.0) or 0.0)
                    except Exception:
                        tol = 0.0
                    try:
                        length_value = float(length_mm)
                    except Exception:
                        length_value = None
                    if length_value is None:
                        self.len_meas_var.set("--")
                    elif exp > 1e-6:
                        dev = length_value - exp
                        if tol > 1e-6:
                            judge_txt = "OK" if abs(dev) <= tol else "NG"
                            self.len_meas_var.set(f"{length_value:.3f} mm  (Δ {dev:+.3f})  {judge_txt}")
                        else:
                            self.len_meas_var.set(f"{length_value:.3f} mm  (Δ {dev:+.3f})")
                    else:
                        self.len_meas_var.set(f"{length_value:.3f} mm")
                else:
                    self.len_meas_var.set(f"失败（{reason}）" if reason else "失败")
        except Exception:
            pass

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

    def _handle_auto_len_event(self, event: AutoLenEvent) -> None:
        self._get_auto_len_event_handler().handle(event)

    def _handle_auto_progress_event(self, event: AutoProgressEvent) -> None:
        self._get_auto_progress_event_handler().handle(event)

    def _handle_auto_coverage_event(self, event: AutoCoverageEvent) -> None:
        self._get_auto_coverage_event_handler().handle(event)

    def _handle_auto_straightness_event(self, event: AutoStraightnessEvent) -> None:
        payload = event.to_payload()
        self._apply_run_summary_payload(payload)
        self._refresh_done_run_summary_and_export()

    def _handle_auto_postcalc_event(self, event: AutoPostcalcEvent) -> None:
        self._get_auto_postcalc_event_handler().handle(event)

    def _handle_auto_raw_points_event(self, event: AutoRawPointsEvent) -> None:
        payload = event.to_payload()
        self._cache_auto_raw_points(payload)

    def _handle_auto_row_event(self, event: AutoRowEvent) -> None:
        self._get_auto_row_event_handler().handle(event)

    def _handle_auto_state_event(self, event: AutoStateEvent) -> None:
        self._get_auto_state_event_handler().handle(event)

    def _get_ui_queue_pump(self) -> UiQueuePump:
        pump = self.__dict__.get("_ui_queue_pump", None)
        if not isinstance(pump, UiQueuePump):
            pump = UiQueuePump(
                ui_q=self.ui_q,
                device_dispatcher=self._device_ui_event_dispatcher,
                measurement_dispatcher=self._measurement_ui_event_dispatcher,
                perf_ui_queue=self._perf_ui_queue,
                log_filter=LOG_UI_EVENT_FILTER,
            )
            self._ui_queue_pump = pump
        return pump

    def _poll_ui_queue(self):
        t_poll0_ns = time.perf_counter_ns()
        pump_result = self._get_ui_queue_pump().drain()
        try:
            self._perf_ui_queue.add_count("calls", 1)
            self._perf_ui_queue.add_count("plc_read", int(pump_result.plc_read_n))
            self._perf_ui_queue.add_value("batch_size", float(pump_result.batch_size))
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
            if str(self._run_session.status or '') == 'DONE':
                self._compute_and_apply_run_summary()
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

    def _get_run_export_coordinator(self) -> RunExportCoordinator:
        coordinator = self.__dict__.get("_run_export_coordinator", None)
        if not isinstance(coordinator, RunExportCoordinator):
            results_service = self.__dict__.get("results_service", None)
            if not isinstance(results_service, ResultsService):
                results_service = ResultsService()
                self.results_service = results_service
            coordinator = RunExportCoordinator(
                repository=self._make_run_repository,
                results_service=results_service,
                recipe_provider=self.get_recipe_copy,
                calibration_provider=self.get_calibration_snapshot,
                coverage_provider=lambda: dict(self._section_cov_info or {}),
            )
            self._run_export_coordinator = coordinator
        return coordinator

    def _current_run_result_for_export(self) -> Any | None:
        try:
            return getattr(getattr(self, "_auto_thread", None), "run_result", None)
        except Exception:
            return None

    def _maybe_trigger_completed_export(self) -> None:
        result = self._get_run_export_coordinator().try_export_terminal_run(
            self._run_session,
            self._current_run_result_for_export(),
        )
        self._apply_export_result_to_ui(result)

    def _trigger_run_export(
        self,
        status: str = "DONE",
        abort_reason: str | None = None,
        completed: bool | None = None,
    ) -> None:
        st = str(status or "DONE").upper()
        session = self.__dict__.get("_run_session", None)
        try:
            if isinstance(session, RunSession):
                session.end_ts = float(time.time())
        except Exception:
            pass
        try:
            self._run_session.status = st
        except Exception:
            pass
        result = self._get_run_export_coordinator().try_export_terminal_run(
            self._run_session,
            self._current_run_result_for_export(),
        )
        self._apply_export_result_to_ui(result)
        try:
            if result.kind is ExportKind.COMPLETED and result.status is not ExportStatus.PENDING:
                self._compute_and_apply_run_summary()
        except Exception:
            pass

    def _apply_export_result_to_ui(self, result: ExportResult) -> None:
        self._last_run_export_result = result
        if result.status is ExportStatus.EXPORTED and result.path is not None:
            self._last_run_export_path = str(result.path)
            self._auto_export_done = result.kind is not ExportKind.MANUAL
            message = f"exported: {self._compact_status_path(result.path)}"
        elif result.status is ExportStatus.FAILED:
            self._last_run_export_path = None
            message = result.message
        else:
            message = result.message

        try:
            current_msg = str(self._run_session.message or "").strip()
            should_merge = (
                result.kind is ExportKind.PARTIAL
                and result.status in {ExportStatus.EXPORTED, ExportStatus.FAILED, ExportStatus.SKIPPED}
                and current_msg
                and current_msg not in {"-", "None"}
                and str(message) not in current_msg
            )
            final_message = f"{current_msg} | {message}" if should_merge else str(message)
            self.auto_msg_var.set(final_message)
            self._run_session.message = final_message
        except Exception:
            pass

    def _append_result_row(self, row: MeasureRow):
        od_ecc = row.od_ecc
        id_ecc = row.id_ecc
        od_ecc_txt = "--" if od_ecc is None else f"{float(od_ecc):.3f}"
        id_ecc_txt = "--" if id_ecc is None else f"{float(id_ecc):.3f}"

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

    def _exports_root_dir(self) -> Path:
        return self._app_root_dir() / "exports"

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

        if accept and x2_mm is not None:
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





    @staticmethod
    def _lsq_fit_cos_sin(theta_rad: np.ndarray, y: np.ndarray):
        X = np.column_stack([np.ones_like(theta_rad), np.cos(theta_rad), np.sin(theta_rad)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        x0, A, B = float(beta[0]), float(beta[1]), float(beta[2])
        return x0, A, B

    def calc_id_single_from_out2(self, theta_deg: Iterable[float], out2_mm: Iterable[float], recipe: Recipe) -> dict:
        return self.calibration_service.calc_id_single_from_out2(theta_deg, out2_mm, recipe)

    def _idcal_fit_diameter(self, theta_deg: np.ndarray, c_mm: np.ndarray, m_mm: np.ndarray, delta_c: float):
        return self.calibration_service.fit_id_diameter(theta_deg, c_mm, m_mm, delta_c)






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
            on_export_index=self._make_history_export_service().upsert_history_index_entry,
        )

    def _make_history_export_service(self) -> HistoryResultExportService:
        return HistoryResultExportService(app_root_dir=self._app_root_dir())

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
        completed: bool | None = None,
        abort_reason: str | None = None,
    ) -> RunContext:
        """Build the current run context used by repository-backed exports."""
        raise RuntimeError("RunExportCoordinator owns production run export context construction")
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
            session = self._run_session
            length_result = dict(session.length_result or {}) if isinstance(session.length_result, dict) else None
        except Exception:
            length_result = None

        _start = float(start_ts if start_ts is not None else self._run_start_ts)
        _end = float(end_ts if end_ts is not None else (self._run_end_ts or time.time()))
        completed_sections = len(list(self._auto_rows or []))
        try:
            expected_sections = int(getattr(recipe, "section_count", 0) or 0)
        except Exception:
            expected_sections = 0
        if completed is None:
            completed = str(status or "").upper() == "DONE"
        summary = dict(summary or {})
        summary["completed"] = bool(completed)
        summary["abort_reason"] = abort_reason
        summary["completed_sections"] = int(completed_sections)
        summary["expected_sections"] = int(expected_sections)

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
            summary=summary,
            finished_at_ts=_end,
            status=str(status or ""),
            completed=bool(completed),
            abort_reason=abort_reason,
            completed_sections=int(completed_sections),
            expected_sections=int(expected_sections),
        )

    def _prepare_new_run(self) -> None:
        """Allocate a new Serial/RunId for the next Auto measurement."""
        try:
            recipe_name = str(getattr(self.recipe, "name", "默认配方") or "默认配方")
        except Exception:
            recipe_name = "默认配方"
        session = self._run_session
        identity = self._make_run_repository().prepare_run(recipe_name)
        serial = str(identity.serial)
        session.serial = serial
        session.run_id = str(identity.run_id)
        session.start_ts = float(identity.started_at_ts)
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
            self.meas_seq_var.set(self._display_seq_text(serial))
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
        if not (session.serial and session.run_id and session.start_ts):
            try:
                recipe_name = str(getattr(self.recipe, "name", "默认配方") or "默认配方")
            except Exception:
                recipe_name = "默认配方"
            identity = self._make_run_repository().prepare_run(recipe_name)
            session.serial = str(identity.serial)
            session.run_id = str(identity.run_id)
            session.start_ts = float(identity.started_at_ts)
            try:
                self.pipe_sn_var.set(session.serial)
            except Exception:
                pass
            try:
                self.meas_seq_var.set(self._display_seq_text(session.serial))
            except Exception:
                pass

        # main-screen start time (best effort)
        try:
            if session.start_ts and hasattr(self, "meas_start_var"):
                import datetime as _dt
                self.meas_start_var.set(_dt.datetime.fromtimestamp(float(session.start_ts)).strftime('%H:%M:%S'))
        except Exception:
            pass

    def _display_seq_text(self, serial: str | None) -> str:
        text = str(serial or "")
        if "__" in text:
            return "--"
        try:
            tail = text.split("-")[-1]
            return tail if tail.isdigit() else "--"
        except Exception:
            return "--"

    def sync_run_session_to_ui(self) -> None:
        """Project RunSession fields onto Tk display variables.

        Call this after any code mutates RunSession fields outside the
        normal event-handler path.  The normal path (_handle_auto_state_event,
        _prepare_new_run) already sets Tk vars inline, so this method is
        primarily useful for test setup and batch updates.

        Business logic MUST read from RunSession; Tk variables are
        display-only mirrors.
        """
        session = self._run_session
        try:
            self.pipe_sn_var.set(str(session.serial or "--"))
        except Exception:
            pass
        try:
            self.meas_seq_var.set(self._display_seq_text(session.serial))
        except Exception:
            pass
        try:
            self.auto_state_var.set(session.status)
        except Exception:
            pass
        try:
            self.auto_msg_var.set(session.message)
        except Exception:
            pass
        try:
            if session.start_ts:
                import datetime as _dt
                self.meas_start_var.set(
                    _dt.datetime.fromtimestamp(float(session.start_ts)).strftime('%H:%M:%S')
                )
        except Exception:
            pass

    # -- _run_len_result property (delegates to RunSession) ---------------

    @property
    def _run_len_result(self) -> dict | None:
        return self._run_session.length_result

    @_run_len_result.setter
    def _run_len_result(self, value: dict | None) -> None:
        self._run_session.length_result = value

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
        try:
            ent_pos = self._require_axis_ui_widget('ent_pos', ax)
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
            ent_pos_r = self._axis_ui_widget('ent_pos_r', ax)
            if ent_pos_r is None:
                ent_pos_r = self._axis_ui_widget('ent_pos2', ax)
            if ent_pos_r is None:
                ent_pos_r = self._require_axis_ui_widget('ent_pos_r', ax)
            dis = float(ent_pos_r.get().strip())
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return

        dir_mover = DIR_POS if dis > 0 else DIR_NEG if dis < 0 else DIR_NONE

        base = self._base(ax)
        # Pos_MoveR (relative displacement)
        self._write_regs(
            base + OFF_POS_MOVER,
            encode_float64_to_4regs(float(dis), FLOAT64_WORD_ORDER),
        )
        # Dir_MoveR follows the sign entered in the relative displacement field.
        self._write_axis_params(ax, dir_mover_override=dir_mover)
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
                self._read_common_params()
            except Exception:
                pass
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
