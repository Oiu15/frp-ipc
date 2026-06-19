# Stabilization Regression Checklist

本清单用于 Phase 6-9 结构重构后的人工回归。执行前确认使用测试工件、允许的设备和安全运动范围；未获授权时不得写 PLC 参数、启动轴运动或覆盖生产配方。

每项记录：执行日期、应用版本/提交、执行人、环境、结果（PASS/FAIL/BLOCKED）和证据路径。

## 1. 启动冒烟

- [ ] 使用项目虚拟环境执行 `python app.py`，主窗口可创建且无启动期 traceback。
- [ ] Main、Axis、Axis Calibration、Recipe、Gauge、Validation、Key Test 页面均可打开。
- [ ] 页面切换、窗口缩放和关闭流程正常，无 Tk callback exception。
- [ ] 关闭应用后 PLC/Gauge worker 正常退出，无残留后台线程阻止进程结束。

## 2. 配方加载与保存

- [ ] 使用测试配方加载全部字段，确认 section 表格、长度参数和算法参数正确回填。
- [ ] 修改一个低风险字段并另存为测试配方，重新加载后值保持一致。
- [ ] 删除测试配方时只删除目标文件，不影响现有生产配方。
- [ ] 非法或缺失字段使用既有兼容/default 行为，不导致 UI 崩溃。
- [ ] 测试结束后恢复原配方选择，不覆盖生产配方。

## 3. PLC 离线/在线

- [ ] PLC 不可达时应用仍可启动，连接状态和错误提示可见且 UI 不冻结。
- [ ] PLC 恢复后可按既有方式重新连接，轴状态轮询恢复。
- [ ] 在线状态下只读检查轴位置、输入点和状态字更新。
- [ ] 如获运动授权，按安全范围验证一次显式轴命令；停止命令始终可用。
- [ ] 断线/重连过程中不产生重复 worker 或持续异常弹窗。

## 4. Gauge 离线/在线

- [ ] Gauge 未连接时页面可打开，串口列表和离线状态正常显示。
- [ ] 模拟 Gauge 模式可切换，采样显示按既有行为更新。
- [ ] 真实 Gauge 获授权后可连接、读取一次并断开；请求命令保持当前配置。
- [ ] 串口异常或断开后 UI 不冻结，错误状态可恢复。

## 5. 正式测量模拟路径

- [ ] 使用模拟设备和专用测试配方启动正式测量，不连接生产工件。
- [ ] prepare、section plan 和 section 执行顺序与配方一致。
- [ ] SPLIT/SYNC 选择按配方生效，采样点、coverage 和 row 均产生。
- [ ] raw publish、coverage publish、row build、record row、row publish 顺序无异常。
- [ ] postcalc summary 和 finalize 完成，运行状态回到预期终态。
- [ ] 在安全模拟路径验证一次 stop/cancel，确认现有停止语义未改变。

## 6. Validation 路径

- [ ] Validation 页面可加载 section 选项和当前状态。
- [ ] 使用模拟设备执行一次最小 repeat run，进度、结果和 summary 更新。
- [ ] 非法 repeat count/参数显示既有校验反馈，不启动 workflow。
- [ ] 停止 Validation 后状态和按钮恢复，无残留运行线程。
- [ ] Validation export 写入独立目录，未混入正式测量 schema。

## 7. 标定页面基础操作

- [ ] Axis Calibration、OD Calibration、ID Calibration 和单探头区域可正常显示。
- [ ] 输入变量、模式切换、高级参数展开/收起和按钮 enable 状态正常。
- [ ] 在未授权写入时只检查 UI 和只读状态，不执行标定写入。
- [ ] 使用模拟数据时可完成一次低风险计算预览，错误输入反馈保持既有行为。
- [ ] Gauge/Validation 共享状态在页面切换后保持一致，不出现动态属性错误。

## 8. 导出路径检查

- [ ] 正式测量 run 目录包含预期 row、raw、coverage 和 summary 文件。
- [ ] 历史结果导出列顺序、单位和记录选择与当前 schema 一致。
- [ ] Validation 导出位于 `validation_exports/`，字段不与 production export 混用。
- [ ] 标定 raw/history 导出使用测试目标目录，不覆盖既有标定文件。
- [ ] 取消导出、无写权限路径和重复文件名按既有错误处理工作。
- [ ] 打开导出文件抽查序列号、section index、OD/ID、coverage 和 summary 值。

## 验收记录

| 项目 | 结果 | 证据/备注 |
| --- | --- | --- |
| 启动冒烟 |  |  |
| 配方加载/保存 |  |  |
| PLC 离线/在线 |  |  |
| Gauge 离线/在线 |  |  |
| 正式测量模拟路径 |  |  |
| Validation 路径 |  |  |
| 标定页面基础操作 |  |  |
| 导出路径检查 |  |  |

