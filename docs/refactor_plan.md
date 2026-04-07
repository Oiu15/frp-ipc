# 重构计划

## 1. 本次重构的定位

本次重构不是功能重写，也不是整仓库全面整理。

本次重构的目标是：

- 以自动测量主链为中心，先把运行时边界梳理清楚。
- 降低 `app.py` 对流程、UI、线程事件、导出的混合耦合。
- 在不改变现有测量行为的前提下，为后续拆分 `App` / `AutoFlow` / UI 事件处理建立稳定边界。

本次文档中提到的“主链”，特指从主界面点击“开始测量”开始，到自动流程执行、结果回传、UI落表、汇总与导出结束的这一条运行链路。

---

## 2. 本次重构范围

### 2.1 In Scope

本次重构只覆盖自动测量主链及其直接依赖边界：

- `app.py`
  - `App._auto_start()`
  - `App._auto_stop()`
  - `App._prepare_new_run()`
  - `App._poll_ui_queue()`
  - `App._append_result_row()`
  - 自动测量相关的 UI 状态变量、结果表、汇总与导出触发逻辑
- `services/autoflow_service.py`
  - `AutoFlow` 线程本体
  - 自动测量流程状态推进
  - 流程结果事件发布：`auto_state` / `auto_row` / `auto_postcalc` / `auto_cov` / `auto_len`
- `drivers/plc_client.py`
  - 仅限主链相关的命令投递与轮询交互边界
  - 仅限 `Cmd*` 命令对象、`PlcWorker` 队列消费、采样档位切换等主链所依赖部分
- `drivers/gauge_driver.py`
  - 仅限主链测量所依赖的请求/响应、最新样本读取与 UI 事件回传
- `ui/screens/main_screen.py`
  - 仅限主界面自动测量入口、状态展示、结果表展示

### 2.2 Out Of Scope

本次重构默认不处理以下内容，除非为了打通主链必须做最小改动：

- PLC 地址映射、寄存器定义、底层协议规则本身
- 测径仪协议格式本身
- OD / ID 算法公式、拟合细节、标定数学逻辑
- 配方数据结构重设计
- 各校准页、轴页、按键测试页、非主界面 UI 的全面整理
- 打包、构建、发布流程
- 历史日志格式与已有导出文件格式的大改

### 2.3 范围控制原则

- 先梳理边界，再考虑模块下沉。
- 先保证主链可读、可维护，再考虑广度扩展。
- 不以“模式化改造”为目标，不为了拆而拆。
- 若某项改动不直接改善主链耦合或主链可测试性，则不进入本轮。

---

## 3. 当前主链

### 3.1 主链总览

当前自动测量主链可以概括为：

`主界面按钮`
-> `App._auto_start()`
-> `AutoFlow(self).start()`
-> `AutoFlow.run()`
-> `通过 App 方法/属性驱动 PLC、读取测径仪、采样并计算`
-> `AutoFlow 通过 ui_q 发布事件`
-> `App._poll_ui_queue()`
-> `主界面状态更新 / 结果表落表 / 汇总 / 导出`

### 3.2 当前主链分段

#### A. 入口层：主界面启动自动测量

- `ui/screens/main_screen.py`
  - `build_main_screen()` 中“开始测量”按钮直接绑定 `app._auto_start()`
- `app.py`
  - `App._auto_start()` 负责准备新一轮运行上下文，并创建 `AutoFlow(self)` 线程

这一段说明：

- 主链入口不在 `services/`，而是先从 UI 直接进入 `App`
- `App` 不是单纯启动器，而是主链宿主

#### B. 流程层：AutoFlow 驱动自动测量

- `services/autoflow_service.py`
  - `AutoFlow.run()` 是当前自动测量的核心流程入口
  - 负责：
    - 读取配方快照
    - 轴使能、夹爪准备、长度测量
    - 截面循环
    - 采样、拟合、生成 `MeasureRow`
    - 完成、停止、异常状态发布

这一段说明：

- `AutoFlow` 已经承担了“流程状态机”角色
- 但它并不是独立服务，它仍然强依赖 `app.App` 作为运行时宿主

#### C. 设备交互支链：PLC / Gauge

主链中真正的设备 IO 并不直接由 UI 完成，而是经过两个后台 worker：

- PLC 支链
  - `App` 通过 `cmd_q` 投递 `CmdWriteRegs`、`CmdReadRegs`、`CmdSetPollProfile` 等命令
  - `drivers/plc_client.py` 中 `PlcWorker` 在线程内执行这些命令并轮询 PLC
- Gauge 支链
  - `drivers/gauge_driver.py` 中 `GaugeWorker` 负责串口请求、解析、缓存最新样本
  - 自动流程和 UI 都会间接依赖其输出

这一段说明：

- 当前主链不是单线程直连硬件，而是“流程线程 + worker线程 + UI线程”的组合
- 因此主链重构必须保留线程边界与队列边界，不能把硬件 IO 重新拉回 UI

#### D. 回传层：AutoFlow 通过 UI 队列回推结果

- `services/autoflow_service.py`
  - 通过 `self.app.ui_q.put(...)` 发布：
    - `auto_state`
    - `auto_row`
    - `auto_postcalc`
    - `auto_cov`
    - `auto_len`
- `app.py`
  - `App._poll_ui_queue()` 周期性消费这些事件

这一段说明：

- 当前主链的“状态更新”和“结果落地”并不是 `AutoFlow` 直接改 UI
- 当前系统已经具备事件回传主干，但事件消费仍然集中在 `app.py`

#### E. 终点层：UI落表、汇总、导出

- `app.py`
  - `App._append_result_row()` 负责将 `MeasureRow` 落到 `result_tree`
  - `App._poll_ui_queue()` 同时负责：
    - 更新 `auto_state_var` / `auto_msg_var`
    - DONE 时触发汇总
    - DONE 时触发导出

这一段说明：

- 当前主链终点不只是“显示结果”，还包括运行级汇总和导出触发
- 因此导出虽然看起来像后处理，实际上仍处在当前主链尾部

---

## 4. 当前主链对应的核心文件

当前主链横跨以下文件：

- `ui/screens/main_screen.py`
- `app.py`
- `services/autoflow_service.py`
- `drivers/plc_client.py`
- `drivers/gauge_driver.py`

其中：

- `app.py` 是当前主链的运行时宿主与汇合点
- `services/autoflow_service.py` 是当前主链的流程核心
- `drivers/*` 是当前主链的 IO 支链
- `ui/screens/main_screen.py` 是当前主链的用户入口与显示终点

---

## 5. 本次重构的落点

本轮重构优先做的是“把主链讲清楚并收边界”，而不是立即大规模拆文件。

优先级如下：

1. 先明确主链边界，避免继续把非主链内容掺进本轮改造
2. 先明确 `App`、`AutoFlow`、`PlcWorker`、`GaugeWorker`、主界面之间的职责
3. 再决定哪些逻辑适合从 `app.py` 抽离

简化理解：

- 本次重构围绕“自动测量主链”
- 不围绕“整个 Tk 应用”
- 不围绕“所有页面”
- 不围绕“所有算法”
