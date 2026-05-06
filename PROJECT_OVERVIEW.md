# FRP 管尺寸检测 IPC 项目概览（PROJECT_OVERVIEW）

## 目标与当前边界

这是一个基于 Tkinter 的 FRP 管尺寸检测 IPC 桌面应用，负责在上位机侧完成以下工作：

- PLC 通信与轴状态监控
- 测径仪通信与采样请求
- 正式测量流程编排
- 标定流程与标定数据管理
- 配方管理、运行数据落盘与导出
- 结果展示、事件分发与操作界面

当前仓库已经从“`app.py` 大一统”逐步迁移到“入口 / shell / app host / mode / workflow / repository / presenter”结构。
其中，workflow 作为架构概念仍保留该名称，但对应的实际 Python 包目录已经从 `workflow/` 重命名为 `frp_workflow/`，以避免与 PyInstaller 同名 hook 冲突。

需要特别说明：

- `app.py` 现在是薄入口，不再承载主体业务实现。
- 正式测量主链已经默认走 `AutoFlowOrchestrator`。
- `services/autoflow_service.py` 仍保留一部分稳定算法/辅助逻辑，但不再是旧的主启动入口。
- 文档中如果再出现 `legacy_app_host.py`、`legacy_app_adapter.py`、`screen_api.py`、`AutoFlow(self)` 作为主路径，均视为过时描述。

---

## 当前主运行链路

启动方式：

```bash
python app.py
```

实际链路如下：

1. `app.py`
   - 导出 `App` 工厂与 `main()`
   - 调用 `ApplicationShell().run(App)`

2. `application/shell.py`
   - 创建 `ui_q` / `cmd_q`
   - 启动 `PlcWorker` 与 `GaugeWorker`
   - 组装 `CalibrationRepository`、`RecipeStore`、默认的 UI dispatcher
   - 创建并运行 `AppHost`

3. `application/app_host.py`
   - 作为当前 Tk 根对象与应用宿主
   - 构建 presenter / controller / mode machine / `frp_workflow` 装配点
   - 挂载各个 screen
   - 消费 `ui_q`，将事件转发给 typed dispatcher 与 presenter

4. 正式测量主链
   - `main_screen` -> `MeasurementController`
   - `MeasurementController` -> `ModeMachine.enter_production()`
   - `AppHost` 创建 `AppDeviceGateway + RunRepository + AutoFlowOrchestrator`
   - `AutoFlowOrchestrator` 运行 section loop、采样、后处理、事件发射
   - 结果经 `WorkflowUiEventAdapter` 写入 `ui_q`
   - `AppHost` 用 typed event dispatcher 更新 presenter / run cache / 导出

5. 标定主链
   - `gauge_screen` -> `CalibrationController`
   - `CalibrationController` -> `ModeMachine.enter_calibration()`
   - `CalibrationService` 负责采集、拟合、写仓储
   - 标定算法在 `domain/calibration.py`
   - 标定 JSON / history / raw export 全部通过 `CalibrationRepository`

6. 验证主链
   - `validation_screen` -> `ScreenController.start_validation_run()` -> `AppHost.start_validation_run()`
   - `AppHost` 创建 `ValidationWorkflow + ValidationRepository` 并驱动独立 Validation 页面状态
   - Validation 导出独立落到 `validation_exports/`，与正式测量导出 schema 分离

---

## 当前目录结构

```text
frp-ipc/
  app.py                         # 薄入口；只负责 App 工厂与 shell handoff
  PROJECT_OVERVIEW.md
  requirements.txt

  application/
    app_host.py                  # 当前 Tk 宿主；UI 主循环、装配、事件消费
    app_adapters.py              # AppDeviceGateway / ScreenPresenter / ScreenController / ScreenUiContext
    shell.py                     # Tk root 生命周期、worker 启停、依赖装配
    state.py                     # RunSession / RuntimeState / ValidationSession / CalibrationSnapshot
    contracts.py                 # 应用层协议边界
    measurement_controller.py    # 正式测量入口
    calibration_controller.py    # 标定入口
    calibration_service.py       # 标定流程编排
    recipe_form_mapper.py        # Recipe <-> UI vars <-> dict
    recipe_presenter.py          # 配方 screen 状态与控件引用
    axis_presenter.py            # 轴调试 screen 状态与控件引用
    gauge_presenter.py           # 外设/标定 screen 状态与控件引用
    results_service.py           # 结果逻辑薄包装（底层已委托 domain）
    ui_events.py                 # typed UI event dataclass
    ui_event_dispatcher.py       # 按事件类型分发
    ui_queue_adapters.py         # worker/workflow -> ui_q 兼容适配层

  config/
    addresses.py                 # PLC / CL 地址、位定义、偏移、默认参数
    default.yaml                 # 预留配置文件

  core/
    models.py                    # AxisComm / AxisCal / Recipe / MeasureRow 等
    modbus_codec.py
    recipe_store.py              # 当前仍在使用的配方持久化实现

  domain/
    planning.py                  # section 规划、Start/Standby/AX2 规则、合法性判断
    summaries.py                 # 直线度/同心度/run summary/post-calc 纯函数
    calibration.py               # OD/ID/单探头标定计算纯函数

  drivers/
    plc_client.py                # PLC worker
    gauge_driver.py              # 测径仪 worker

  machine/
    device_gateway.py            # 正式测量链最小机器接口
    plc_gateway.py               # 预留 / 逐步收口中
    gauge_gateway.py             # 预留 / 逐步收口中

  modes/
    mode_machine.py              # Production / Calibration / Validation 统一切换
    production_mode.py
    calibration_mode.py
    validation_mode.py

  repositories/
    run_repository.py            # run_id / serial / exports / summary.csv
    calibration_repository.py    # 标定 JSON / history / raw export
    validation_repository.py     # validation_exports 独立 schema
    recipe_repository.py         # 配方仓储包装层（已存在，尚未完全接为主依赖）

  services/
    autoflow_service.py          # 旧 AutoFlow helper 与稳定算法复用点

  ui/
    screens/
      main_screen.py
      axis_screen.py
      axis_cal_screen.py
      recipe_screen.py
      gauge_screen.py
      validation_screen.py
      key_test_screen.py

  utils/
    logger.py
    perf.py

  frp_workflow/
    autoflow_orchestrator.py     # 正式测量 orchestrator
    production_workflow.py       # 正式测量 typed event / result / summary 边界
    validation_workflow.py       # 验证模式 typed event / result / export context 边界

  tests/
    ...                          # 纯逻辑、frp_workflow、repository、event routing 测试
```

说明：

- `build/`、`dist/`、`demo/`、`*.spec` 不属于主运行链路。
- 当前真实运行主链集中在 `application/ + modes/ + frp_workflow/ + repositories/ + drivers/ + ui/`。

---

## 分层职责

### 1. 入口与应用壳

- `app.py`
  - 薄入口
  - 不持有业务状态

- `application/shell.py`
  - worker 生命周期
  - 依赖装配
  - 创建 `AppHost`

- `application/app_host.py`
  - Tk root 与 UI 主循环
  - screen mounting
  - presenter / controller / mode machine 装配
  - `ui_q` 事件消费与 UI 更新协调

### 2. 模式层

- `modes/mode_machine.py`
  - 统一管理当前 mode
  - 提供显式 transition：
    - `enter_production()`
    - `enter_calibration()`
    - `enter_validation()`
    - `stop_current()`
    - `recover_error()`
  - 将 mode 状态同步到 `RuntimeState`

- `production_mode.py`
  - 状态：`idle / preparing / running / stopping / error / completed`

- `calibration_mode.py`
  - 状态：`idle / acquiring / fitting / saving / error`

- `validation_mode.py`
  - 当前为骨架，承载验证模式状态与后续扩展点

### 3. `frp_workflow` 包（Workflow 层）

- `frp_workflow/autoflow_orchestrator.py`
  - 正式测量编排壳
  - 负责 start/stop、section loop、运动控制顺序、事件发射
  - 复用 `services/autoflow_service.py` 中已验证的辅助算法

- `frp_workflow/production_workflow.py`
  - 正式测量 workflow 的纯边界对象
  - 输入只保留：
    - `Recipe`
    - `CalibrationSnapshot`
    - `RuntimeState`
    - `MachineGateway`
    - `RunRepositoryProtocol`
  - 输出只保留：
    - typed event
    - `RunResult`
    - `RawPoints`
    - `summary`

- `frp_workflow/validation_workflow.py`
  - 验证模式边界
  - 管理 `ValidationSession`、typed event、result、export context

### 4. Repository 层

- `RunRepository`
  - 分配 `serial / run_id`
  - 写 `section_results.csv`
  - 写 `raw_points.csv`
  - 写 `meta.json`
  - 更新 `summary.csv`

- `CalibrationRepository`
  - 管理 `od_calibration.json`
  - 管理 `id_calibration.json`
  - 管理单探头标定 JSON / history
  - 管理标定原始数据导出路径

- `ValidationRepository`
  - 单独写 `validation_exports/<day>/<serial>/...`
  - 不污染正式测量 `exports/` schema

- `RecipeStore` / `recipe_repository.py`
  - 当前实际主链仍主要通过 `core.recipe_store.RecipeStore`
  - `recipe_repository.py` 是已经预留好的包装层，但尚未完全替换主依赖

### 5. Domain 层

- `domain/planning.py`
  - section 位置规划
  - Start / Standby / AX2 长度位/旋转位规则
  - keepout 与 target 合法性处理

- `domain/summaries.py`
  - 直线度汇总
  - 同心度汇总
  - run summary
  - post-calc
  - 全部以纯函数形式存在

- `domain/calibration.py`
  - OD B 标定
  - 单探头补救标定
  - ID 直径拟合
  - delta 求解与复核判定
  - 与 repository / UI / workflow 解耦，可独立单测

### 6. Screen / Presenter / Controller 层

- `ui/screens/*`
  - 只保留：控件创建、布局、绑定
  - 不再直接持有业务状态
  - 不再直接访问 worker
  - 不再直接写回 `app.xxx = widget`
  - `validation_screen.py` 是 Validation 正式入口页
  - `gauge_screen.py` 中的 Validation 区已收口为跳转提示 + 只读状态

- `application/*_presenter.py`
  - 持有 screen 所需的 `StringVar/BooleanVar/IntVar`
  - 维护少量必要的 widget/view-state registry

- `application/*_controller.py`
  - 将 UI 事件翻译成 mode / workflow / service intent

---

## Mode Machine 与共享状态

当前 mode 状态不再散落在 controller / presenter / workflow 各处，而是统一归集到 `RuntimeState`：

- `mode_kind`
- `mode_state`
- `mode_error`

`ModeMachine` 是唯一的 mode 切换入口，controller 不再直接 new workflow 或自管 mode 状态。

这样做的目的：

- 避免“一个 controller 一份状态、一个 presenter 一份状态”
- 让 Production / Calibration / Validation 三套模式共享统一 transition 语义
- 为后续更明确的 mode UI、错误恢复、跨模式切换打基础

---

## Workflow / Repository / Event Flow

### 1. Worker 与 workflow 发事件

- PLC worker / Gauge worker
  - 通过 `WorkerUiEventAdapter` 发事件
  - 仍保持旧的 `ui_q.put((event_name, payload))` tuple 兼容协议

- 正式测量 workflow
  - 通过 `WorkflowUiEventAdapter` 发事件
  - 同样保持旧 payload 结构兼容

### 2. typed event 定义

`application/ui_events.py` 定义了 typed UI event dataclass，例如：

- `PlcOkEvent`
- `PlcErrEvent`
- `GaugeOkEvent`
- `AutoStateEvent`
- `AutoRowEvent`
- 以及 `auto_progress / auto_cov / auto_postcalc / auto_raw_points` 等同类事件

### 3. UI 事件分发

- `AppHost._poll_ui_queue()` 从 `ui_q` 取出 tuple
- `UiEventDispatcher` 先将 tuple 解析为 typed event
- 再按事件类型分发到 handler
- handler 更新 presenter、run cache、mode state、导出触发等

### 4. 正式测量结果链

正式测量主链的结果路径大致为：

1. `AutoFlowOrchestrator`
2. `ProductionWorkflow`
3. `WorkflowUiEventAdapter`
4. `ui_q`
5. `UiEventDispatcher`
6. `AppHost` handler
7. presenter / view-state 更新
8. `RunRepository` 导出

### 5. 验证模式结果链

验证模式单独走：

1. `ValidationWorkflow`
2. `ValidationExportContext`
3. `ValidationRepository`
4. `validation_exports/<day>/<serial>/...`

正式测量与验证模式的导出 schema 已分离。

---

## 持久化与导出

默认数据根目录：

```text
C:\Users\<user>\FRP_IPC
```

当前主要内容：

- `recipes/`
  - 配方 JSON
  - `index.json`

- `logs/`
  - 运行日志

- `calibration/`
  - `od_calibration.json`
  - `id_calibration.json`
  - 单探头标定文件
  - history 与 raw export

- `exports/`
  - 正式测量导出
  - `section_results.csv`
  - `raw_points.csv`
  - `meta.json`
  - `summary.csv`

- `validation_exports/`
  - 验证模式导出
  - `validation_result.json`
  - `validation_meta.json`
  - `validation_events.json`
  - `repeat_results.csv`
  - `repeat_raw_points.csv`
  - `repeat_fit_results.csv`
  - legacy compatibility: `repeat_rows.csv` / `repeat_section_results.csv` / `repeat_windows.csv` / `repeat_summary.json`

---

## 已删除的兼容路径

以下兼容路径已从主链或仓库中移除，不应再视为当前架构的一部分：

- `application/legacy_app_host.py`
  - 已替换为 `application/app_host.py`

- `application/legacy_app_adapter.py`
  - 已替换为 `application/app_adapters.py`

- `ui/screens/screen_api.py`
  - screen 不再通过 bundled app-like facade 访问 presenter/controller/ui

- 旧 `AutoFlow(self)` 启动路径
  - 正式测量现在只从 orchestrator 主链启动

- 一批已迁移完成的 `_auto_* / _export_* / 旧兼容 wrapper`
  - 已按 usage audit 删除

- screen 内部的 `app.xxx = widget` 写回方式
  - 已删除
  - widget / variable 所有权转移到 presenter / ui context

说明：

- `services/autoflow_service.py` 仍存在，但角色已变化。
- 它现在主要作为稳定 helper/算法复用点，而不是旧式主入口。

---

## 当前实现状态摘要

1. 正式测量模式
   - 已有 `ProductionMode + ModeMachine + AutoFlowOrchestrator + ProductionWorkflow`
   - 是当前主运行链路

2. 标定模式
   - 已有 `CalibrationMode + CalibrationController + CalibrationService + CalibrationRepository`
   - 标定算法已抽到 `domain/calibration.py`

3. 验证模式
   - 已有 `ValidationMode + ValidationWorkflow + ValidationSession + ValidationRepository`
   - 已有独立 `validation_screen` 入口、运行中 summary panel、canonical export 结构

4. UI 事件系统
   - 已从字符串 if/elif 分发迁到 typed event + dispatcher
   - payload 协议在迁移期仍保持兼容

5. 测试
   - 已覆盖：
     - planning 纯逻辑
     - summaries 纯逻辑
     - calibration 纯逻辑
     - typed event routing
     - workflow smoke test
     - validation repeat runs
     - mode machine transition matrix
   - 当前项目级 `pyright` 已清零：`0 errors, 0 warnings`
   - 当前源码目录 `compileall` 通过，测试套件为 `168 passed`

---

## 文档使用说明

在后续讨论和重构分析中，请优先以本文档描述的结构为准。

涉及模块路径时，请使用 `frp_workflow/...` 或 `from frp_workflow...`；仅在讨论职责分层时，才使用泛化的 “workflow” 概念称呼。

如果发现以下表述，请视为旧版本信息：

- `app.py` 仍然是 God Object
- `legacy_app_host.py` / `legacy_app_adapter.py` 仍然存在
- `screen_api.py` 仍然是 screen 主入口
- `AutoFlow(self)` 仍然是正式测量默认启动路径
- screen 仍然直接持有业务状态或 worker 访问
---

## 旧架构到新骨架替代关系

下表用于说明“旧 App 架构中的职责”在当前新骨架中的对应落点。

| Old | New |
| --- | --- |
| `app.py` God Object | `app.py` 薄入口 + `application/shell.py` + `application/app_host.py` |
| `LegacyAppHost` / `legacy_app_host.py` | `AppHost` / `application/app_host.py` |
| `legacy_app_adapter.py` | `application/app_adapters.py` |
| `LegacyAppDeviceGateway` | `AppDeviceGateway` |
| `LegacyScreenPresenter` | `ScreenPresenter` |
| `LegacyScreenController` | `ScreenController` |
| `LegacyScreenUiContext` | `ScreenUiContext` |
| screen 直接拿整包 `app` | screen 只接 `presenter / controller / ui` |
| `screen_api.py` 过渡 facade | 已删除；screen 直接绑定 presenter/controller/ui |
| `AutoFlow(self)` 作为正式测量入口 | `MeasurementController -> ModeMachine.enter_production() -> AutoFlowOrchestrator` |
| `App` 内部散落的 mode flag | `ModeMachine + RuntimeState` |
| `app.py / App` 里的 run 身份与缓存字段 | `RunSession` / `RunContext` |
| `App` 里 `_export_*` 导出逻辑 | `RunRepository` |
| `App` 里标定 JSON 路径、读写、history | `CalibrationRepository` |
| `App` 里 validation 导出混在正式测量导出中 | `ValidationRepository`，单独写 `validation_exports/` |
| `_poll_ui_queue()` 里的字符串 `if/elif` 路由 | `ui_events.py` + `UiEventDispatcher` typed 分发 |
| worker / workflow 直接 `ui_q.put((name, payload))` | `WorkerUiEventAdapter` / `WorkflowUiEventAdapter` |
| `App` / workflow 里 section 规划与 AX2/Start/Standby 规则 | `domain/planning.py` |
| presenter / host 里的直线度、同心度、post-calc、run summary | `domain/summaries.py` |
| `CalibrationService` / UI 回调内嵌标定算法 | `domain/calibration.py` |
| `app` 持有 recipe screen 的 `StringVar` | `RecipeScreenPresenter` |
| `app.xxx = widget` 写回 host | presenter widget registry / ui context |
| 配方 JSON 直接由 UI 层拼装 | `RecipeFormMapper` |

补充说明：

- 当前正式测量仍会在 `AutoFlowOrchestrator` 内部复用 `services/autoflow_service.py` 中的一部分稳定 helper；这表示“启动主链已切换”，不表示“旧 helper 已完全消失”。
- `recipe_repository.py` 已存在，但配方持久化主链当前仍主要使用 `core.recipe_store.RecipeStore`。

---

## Rollback 注意事项

如果需要从当前新骨架回退到旧提交，请优先做“整段提交级回退”，不要只回退单个文件或单个类名。当前几个模块是成组收口的：

1. `app.py`、`application/app_host.py`、`application/app_adapters.py`
   - 这三者现在是配套关系。
   - 如果只回退其中一个，导入路径和类名会立刻错位。

2. `ui/screens/*` 与 presenter/controller
   - screen 已经不再接整包 `app`。
   - 如果回退 screen，但不回退 presenter/controller 接线，按钮和变量绑定会断。

3. `UiEventDispatcher`、`ui_events.py`、`ui_queue_adapters.py`
   - 现在消费者已经按 typed event 注册 handler。
   - 但生产者 payload 仍保持旧 tuple 兼容，因此这一组可以整体回退，也可以整体保留。
   - 不建议只回退 dispatcher 而保留 typed handler 注册。

4. 正式测量主链
   - 当前默认入口是 orchestrator。
   - 旧 `AutoFlow(self)` 启动路径已删除，不能靠改一个 flag 回切。
   - 如果要回到旧实现，只能从历史提交恢复整套旧入口。

5. 用户数据目录
   - 回退代码时不要删除 `C:\Users\<user>\FRP_IPC`。
   - 当前仓储层仍刻意保持配方、标定、正式测量导出 schema 的兼容性，整仓回退时应继续复用这批文件。

6. 回退验证模式相关改动时
   - `validation_exports/` 是新增目录。
   - 旧版本通常不会读取它，但也不会影响正式测量导出；可以保留，不需要清理。

推荐 rollback 顺序：

1. 先确认目标提交点是否仍包含完整的旧入口链。
2. 整体回退 `app.py + application/ + ui/screens/ + frp_workflow/ + services/autoflow_service.py` 的对应提交。
3. 保留 `FRP_IPC` 用户数据目录不动。
4. 启动后优先检查：配方加载、PLC 连接、测径仪连接、正式测量启动、导出落盘。

---

## 兼容文件格式说明

当前重构过程中，文件格式兼容策略如下。

### 1. 配方文件

- 路径：`FRP_IPC/recipes/<name>.json`
- 索引：`FRP_IPC/recipes/index.json`
- 当前主链仍由 `RecipeStore` 管理
- 兼容要求：
  - 旧配方 JSON 继续可读
  - `index.json` 仍保存最近使用配方
  - 回退旧版本时不需要迁移配方目录结构

### 2. 标定文件

- 路径：
  - `FRP_IPC/calibration/od_calibration.json`
  - `FRP_IPC/calibration/id_calibration.json`
  - `FRP_IPC/calibration/id_single_calibration.json`
- history：
  - `od_calibration_history.jsonl`
  - `id_calibration_history.jsonl`
  - `id_single_calibration_history.jsonl`
- 兼容要求：
  - `od_calibration.json` / `id_calibration.json` 保持历史 schema 可读
  - `CalibrationRepository` 已兼容 `utf-8-sig` 旧文件
  - 单探头标定是新增独立文件，不会覆盖 OD/ID 历史 active 文件

### 3. 标定原始数据导出

- OD raw：`FRP_IPC/exports/od_calib/od_calib_raw_<ts>.csv`
- ID raw：`FRP_IPC/calibration/id_calib_raw_<ts>.csv`
- 兼容要求：
  - 目录风格与原链路保持一致
  - CSV 表头不随这轮重构改变

### 4. 正式测量导出

- 路径：`FRP_IPC/exports/<day>/<serial>/`
- 文件：
  - `section_results.csv`
  - `raw_points.csv`
  - `meta.json`
- 日汇总：`FRP_IPC/exports/<day>/summary.csv`
- 计数器：`FRP_IPC/run_counter.json`
- 兼容要求：
  - 目录结构不变
  - CSV schema 不变
  - `serial / run_id` 仍由 `RunRepository` 按旧规则生成与落盘

### 5. 验证模式导出

- 路径：`FRP_IPC/validation_exports/<day>/<serial>/`
- 文件：
  - `validation_result.json`
  - `validation_events.json`
- 日汇总：`FRP_IPC/validation_exports/<day>/summary.csv`
- 兼容说明：
  - 这是新增 schema
  - 与正式测量 `exports/` 完全分离
  - 不影响旧正式测量导出读取逻辑

### 6. UI 事件 payload

- queue 仍然使用 `(event_name, payload)` tuple 兼容协议
- worker/workflow 现在通过 adapter 发事件
- consumer 侧已经迁到 typed event 分发
- 兼容含义：
  - 旧 payload 结构尽量不变
  - 新 dispatcher/typed event 只是消费方式升级，不要求生产者立刻改 schema
