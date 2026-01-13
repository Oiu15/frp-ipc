# FRP 管尺寸检测 IPC 项目概览（PROJECT_OVERVIEW）

## 目标与边界

该 IPC（工控机）软件用于配合 PLC 完成 FRP 管的尺寸检测与自动测量流程。当前版本聚焦于：

- 五轴状态监控与手动调试（AX0/1/2/3/4）
- 测量配方（Recipe）参数管理与示教（截面位置 UI 坐标）
- 测径仪（外径 OD）串口通信（真实串口/模拟）
- 主操作界面：自动测量启动/停止、结果表格显示、直线度与采样覆盖率统计

该版本不直接修改 PLC 的运动学/控制算法，而是通过既定的 Axis_CommD 通信区对 PLC 下发“命令位/目标值/速度参数”，并周期读取 PLC 反馈状态。

---

## 目录结构

```
frp_app/
  app.py                         # 应用层：Tk 主线程 + 回调 + 编排
  PROJECT_OVERVIEW.md            # 本文件
  config/
    addresses.py                 # 协议常量、地址、位定义、偏移
  core/
    models.py                    # 纯数据模型（AxisComm/Recipe/MeasureRow/...）
  drivers/
    plc_client.py                # Modbus TCP 轮询 + 指令队列（PlcWorker）
    gauge_driver.py              # 串口测径仪驱动（GaugeWorker）
  services/
    autoflow_service.py          # 自动测量流程线程（AutoFlow）
  ui/
    screens/
      axis_screen.py             # 页1：轴参数与轴调试 UI 构建
      recipe_screen.py           # 页2：配方/示教 UI 构建
      gauge_screen.py            # 页3：测径仪通信 UI 构建
      main_screen.py             # 页4：主操作/自动测量 UI 构建
```

---

## 四个界面与功能划分

1. **轴参数/调试**
   - 选择轴、状态显示（Pos/Vel/Err/Sts/Flags）
   - 通用参数（vel/acc/dec/jerk）写入 PLC
   - 点动 Jog / Inch / 速度模式 VelMove / MoveA / Stop/Halt/Reset
   - **界面零点坐标（UI ZeroAbs + sign）**：仅影响 UI 相对坐标显示与“示教”位置换算，不写入 PLC

2. **配方/示教**
   - 配方参数：管长、夹持占用、边距、截面数、扫描轴、OD/ID 标准值、容差等
   - 自动计算默认截面位置（UI 坐标）
   - 示教：将“当前 UI 位置”写入指定截面位置表
   - 配方保存/加载（JSON）

3. **测径仪通信**
   - 串口扫描、连接/断开
   - 请求指令配置（默认 `M1,0`）
   - 收到的 OD 值显示与错误提示
   - “模拟测径仪”：在无真实串口时，为 AutoFlow 提供近似数据源

4. **主操作/自动测量**
   - 自动测量启动/停止
   - 结果表格（截面号、X(UI)、OD平均/最大/最小、偏差、真圆度、合格判定）
   - 采样覆盖率统计（等角 bin 覆盖率、缺失 bin、圈数/用时、结束原因）
   - 直线度（基于截面圆心）显示

---

## 关键数据流与线程模型

### 线程

- **UI 线程（Tk 主线程）**
  - 渲染界面
  - 将用户操作转成命令写入 `cmd_q`
  - 周期从 `ui_q` 取消息更新 UI（`_poll_ui_queue`）

- **PlcWorker（drivers/plc_client.py）**
  - 维护 Modbus TCP 连接
  - 周期读取每轴通信区并解析为 `AxisComm`
  - 消费 `cmd_q` 并执行写寄存器/置位/脉冲/模式写入
  - 将结果通过 `ui_q` 回传（plc_ok / plc_err）

- **GaugeWorker（drivers/gauge_driver.py）**
  - 维护串口连接
  - 接收数据并解析 OD
  - 将结果通过 `ui_q` 回传（gauge_ok / gauge_err / gauge_conn）

- **AutoFlow（services/autoflow_service.py）**
  - 自动流程状态机线程
  - 控制轴运动（MoveA/VelMove/Stop/Halt）与采样
  - 计算截面 OD 统计、圆拟合、直线度、覆盖率
  - 将进度/结果通过 `ui_q` 回传（auto_*）

### 队列消息约定（ui_q）

- `plc_ok / plc_err`
- `gauge_conn / gauge_ok / gauge_err / gauge_tx`
- `auto_clear / auto_progress / auto_row / auto_straightness / auto_cov / auto_state`

---

## 地址/协议的维护方式

- 所有 Axis_CommD 偏移与位定义统一在 `config/addresses.py`。
- 若 PLC 侧协议变更（比如通信区 base 或 offset 变化），只修改该文件即可。

---

## 运行方式

在项目根目录（包含 `frp_app/` 文件夹）执行：

```bash
python -m frp_app.app
```

或直接：

```bash
python frp_app/app.py
```

---

## 后续建议（解耦与迭代）

- 将“轴参数（vel/acc/dec/jerk + UI_POS）”写入**配方**并做“轴别绑定”，可在 `core/models.py` 增加 `AxisProfile`，并在 `recipe_screen.py` 做导入/保存。
- 将 AutoFlow 对 PLC 的命令下发再封装一层 `MotionFacade`（services 层），减少 AutoFlow 对 App 的依赖，便于单元测试。
