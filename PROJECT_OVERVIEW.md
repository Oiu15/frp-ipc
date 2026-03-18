# FRP 管尺寸检测 IPC 项目概览（PROJECT_OVERVIEW）

## 目标与边界

该项目是一个基于 Tkinter 的 IPC 应用，用于配合 PLC 完成 FRP 管尺寸检测、自动测量、结果展示与运行数据导出。当前实现的重点是：

- 五轴设备联动与状态监控：AX0/AX1/AX2/AX3/AX4
- 统一 Z 坐标标定与示教
- 外径测径仪串口通信
- PLC 内径/位移计数据读取与标定
- 自动流程：长度测量、截面扫描、圆拟合、直线度/同心度/覆盖率统计
- 配方、标定、运行结果、原始点与汇总报表的落盘

轴职责在当前实现中大致为：

- AX0：OD 方向位置轴
- AX1 + AX4：ID 方向位置轴组合
- AX2：中心架/夹持相关轴
- AX3：旋转采样轴

IPC 不直接实现 PLC 的运动学或底层控制算法，而是通过 PLC 的 `AXIS_Ctrl` 通信区写入命令位、目标值和速度参数，并周期读取反馈状态。

---

## 当前目录结构

```text
frp-ipc/
  app.py                         # 应用入口；UI 主线程、回调编排、导出逻辑
  PROJECT_OVERVIEW.md            # 本文件
  requirements.txt
  config/
    addresses.py                 # PLC/CL 地址、位定义、偏移、默认参数
    default.yaml                 # 预留配置文件
  core/
    models.py                    # AxisComm / AxisCal / Recipe / MeasureRow 等
    modbus_codec.py              # 编解码辅助
    recipe_store.py              # 配方持久化
  drivers/
    plc_client.py                # Modbus TCP 后台轮询与命令执行
    gauge_driver.py              # 外径测径仪串口线程
  services/
    autoflow_service.py          # 自动测量流程线程与后处理
  ui/
    screens/
      main_screen.py             # 主操作/自动测量
      axis_cal_screen.py         # 轴位标定
      axis_screen.py             # 轴参数/调试
      recipe_screen.py           # 配方/示教
      gauge_screen.py            # 外设通信/标定
      key_test_screen.py         # X/Y 点按键测试
  utils/
    logger.py                    # 日志初始化与封装
    perf.py                      # 性能统计辅助
```

仓库内还包含 `build/`、`dist/`、`demo/`、`*.spec` 等打包或现场辅助文件，但不属于主运行链路。

---

## 运行方式

在仓库根目录执行：

```bash
python app.py
```

依赖来自 `requirements.txt`，主要包括 `numpy`、`scipy`、`pymodbus`、`pyserial`。

---

## 核心数据模型

- `AxisComm`
  - 当前对应 PLC 的 `AXIS_Ctrl` 布局
  - 反馈值和运动设定值均以当前协议字段为主
  - 保留了部分旧字段别名以兼容历史 UI/服务代码

- `AxisCal`
  - 统一 Z 坐标映射模型
  - 维护 `sign`、`off_ax0/off_ax1/off_ax2/off_ax4`、`b14`、`b2`、`keepout_w`、`z_pos`
  - 约定 `Z` 正方向向下，`z_pos` 为 IPC 侧临时显示偏移，不直接写入 PLC

- `Recipe`
  - 除基础管长、边距、截面数、标准值与容差外，还包含：
  - `section_pos_z` 截面 Z 位置
  - `start_ax0_abs` 起点锚点
  - `standby_ax0_abs/ax1_abs/ax4_abs` 待定点
  - `ax2_len_abs / ax2_rot_abs` 中心架长度位和旋转位
  - 长度测量参数
  - 采样覆盖率与旋转速度参数
  - OD/ID 算法开关、单探头补救参数、`scan_mode`

- `MeasureRow`
  - 每个截面的最终结果行
  - 包含 OD/ID 偏差、径向跳动、真圆度、偏心量、同心度、后处理结果和覆盖率统计

---

## 六个界面与职责

1. **主操作/自动测量**
   - 启动/停止自动流程
   - 显示运行状态、流水号、开始时间、耗时、提示信息
   - 展示 OD/ID 汇总、整体同心度、轴线指标、长度测量值
   - 显示截面结果表和采样覆盖率信息

2. **轴位标定**
   - 读取/写入 PLC 侧轴位标定块
   - 管理 `sign`、各轴 offset、`B14`、避让区中心与宽度、`z_pos`
   - 支持“采集各轴 offset”“标定 B14”“标定避让区”“设置 Z_Pos 零点”

3. **轴参数/调试**
   - 查看单轴状态与反馈
   - 写入运动参数
   - 执行 Jog、MoveA、MoveR、VelMove、Stop、Halt、Reset、Enable 等调试动作

4. **配方/示教**
   - 编辑测量基础参数、截面规划参数、标准值和容差
   - 配置采样与后处理参数
   - 配置长度测量参数
   - 配置 OD/ID 算法开关和 split scan 选项
   - 示教截面位置、保存待定点、设置 Start、保存 AX2 长度位/旋转位
   - 保存、加载、导入、导出配方

5. **外设通信**
   - PLC Modbus TCP 连接/重连
   - 外径测径仪串口扫描、连接、断开、单次请求、模拟模式
   - 外径标定（B 标定）与原始数据导出
   - 内径/位移计相关实时显示、标定与原始数据导出

6. **按键测试**
   - 读取 PLC 的 X 点状态
   - 对 Y 点执行单次写 0/1
   - 用于现场联调线圈映射，不做持续强制写入

---

## 线程模型与数据流

### UI 主线程

- 创建并持有全部页面与状态变量
- 将用户动作转成 `cmd_q` 命令或直接调用应用层方法
- 周期消费 `ui_q` 更新界面、缓存结果、触发导出

### PlcWorker

- 维护 Modbus TCP 连接、重连与退避
- 解析每轴 `AXIS_Ctrl` 数据块为 `AxisComm`
- 轮询 PLC 内部 CL 输入块，读取内径/位移计相关数据
- 轮询按键测试 X/Y 线圈
- 支持按需读取寄存器块，例如轴位标定区
- 支持普通轮询与采样轮询两种 profile

典型 UI 事件：

- `plc_ok`
- `plc_err`
- `plc_manual`
- `plc_giveup`
- `plc_read`

### GaugeWorker

- 维护外径测径仪串口连接
- 使用请求/响应方式发送 `M0/M1/M2` 指令
- 严格解析返回帧，支持 OUT1、OUT2 和判定结果
- 在无真实串口时可由 UI 切换到模拟模式

典型 UI 事件：

- `gauge_conn`
- `gauge_ok`
- `gauge_err`
- `gauge_tx`

### AutoFlow

- 运行自动测量状态机
- 控制夹爪、中心架、位置轴、旋转轴
- 可选执行长度测量
- 对每个截面做旋转采样与后处理
- 产出 OD/ID、同心度、直线度、覆盖率、原始点和最终表格结果

典型 UI 事件：

- `auto_state`
- `auto_progress`
- `auto_cov`
- `auto_row`
- `auto_len`
- `auto_raw_points`
- `auto_straightness`
- `auto_postcalc`
- `auto_clear`

---

## 协议与地址维护

- PLC、CL、X/Y 点和轴位标定相关地址集中定义在 `config/addresses.py`
- 当前 IPC 代码基于 `AXIS_Ctrl` 通信布局
- 若 PLC 协议调整，优先修改 `config/addresses.py`，其次检查 `core/models.py` 与 `drivers/plc_client.py`

---

## 持久化与导出

应用运行时的用户数据根目录为：

```text
C:\Users\<user>\FRP_IPC
```

主要内容包括：

- `recipes/`
  - 配方 JSON
  - `index.json` 保存最近使用记录

- `logs/`
  - 运行日志

- `calibration/`
  - `od_calibration.json` 与历史记录
  - `id_calibration.json` 与历史记录
  - 标定原始数据导出

- `exports/`
  - 每次自动测量按日期落盘
  - 典型输出：
    - `section_results.csv`
    - `raw_points.csv`
    - `meta.json`
    - `summary.csv`

说明：

- 配方默认保存在用户目录 `FRP_IPC/recipes`
- 如果用户目录不可用，代码会退回到仓库内的 `./data/recipes`

---

## 当前实现状态摘要

- 项目当前不是 `frp_app` 包结构，而是仓库根目录下直接运行 `app.py`
- 文档中若再出现 `Axis_CommD`、`frp_app/`、或“仅四个界面”等描述，均视为过时
- 当前 UI 与数据模型已围绕 `AxisCal + section_pos_z + AXIS_Ctrl + 自动导出` 这一套实现展开
