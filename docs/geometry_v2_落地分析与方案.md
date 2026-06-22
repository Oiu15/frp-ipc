# Geometry-V2 标定/补偿算法并行落地 — 现状分析与实施方案

> 状态:**设计确认稿**(第 1、2 步产出)。代码尚未实现。
> 输入材料:《标定流程与补偿算法设计.md》+ `frp_metrology_core.py`(纯算法骨架,含自检)。
> 总目标:把几何重建算法**与旧算法并行、可切换、可回退**地落地,默认关闭,不动现有产线。

---

## 0. 核心原则

1. **默认 legacy**:新增 `Recipe.algo_version`,默认 `"legacy"` → 现有产线零行为变化。
2. **纯函数 + 新模块**:骨架逻辑移植进 `domain/`,只依赖 numpy/scipy,不改旧函数。
3. **只加不改旧**:新标定文件、新导出列/文件均为追加;`od/id_calibration.json` 与 `section_results.csv` / `raw_points.csv` 现有 schema 一字不改。
4. **可回退**:关掉开关即回 legacy;新文件旧版本不读取、保留无害。

---

## 第 1 步:现状摸底

### 1.1 旧标定算法位置与 I/O

| 模块 | 函数 | 输入 | 输出 |
|---|---|---|---|
| `domain/calibration.py` | `compute_od_b_candidate(sums, d_ref)` | OD 两边沿和 `a+b`、参考径 | `B_candidate = d_ref + mean(sum)`(标量零位) |
| | `fit_id_diameter(theta_deg, c_mm, m_mm, delta_c)` | θ、弦 c(OUT4)、不对称 m(OUT5)、基线补偿 | `IdDiameterFitResult{radius,diam,e,phi_rad,x0,y0,rmse_r2}`,**弦+1阶偏心模型,非点云圆拟合** |
| | `solve_id_delta_candidate(...)` | 同上 + d_ref | 二分求 `delta_c` 使 `diam==d_ref` |
| | `verify_id_calibration(...)` | 同上 | 覆盖率/误差校核 |
| | `fit_id_single_from_out2(theta_deg, out2_mm, recipe)` | 单探头 OUT2 序列 | 按 bin 拟合去偏心残差,出 id_est/ecc/pp |
| `services/calibration_service.py` / `services/calibration_controller.py` / `services/od_calibration.py` / `services/id_calibration.py` | — | 采集编排 → 调 domain → 写 repo | — |
| `repositories/calibration_repository.py` | `save/load_od_active`、`save/load_id_active`、`load_snapshot` | JSON dict | `od/id/id_single_calibration.json` + history |

**旧标定模型本质**:OD 只标标量零位 `B_eff`(`B_active`);ID 只标标量基线补偿 `delta_c_mm`。无探头位姿(s/q/γ)、轴方位 ψ、远心度等几何参数概念。

### 1.2 旧补偿/几何计算位置与 I/O

**每截面**(`frp_workflow/row_math.py::_compute_measure_row_result`):
- 输入:`coords_od`/`coords_id`(已是 `(N,2)` 笛卡尔点)、`raw_points`、recipe、`LegacyFitPort`。
- **关键**:`coords_od/coords_id` 在采样阶段就用 `r=0.5·读数, x=r·cosθ, y=r·sinθ` 生成(`frp_workflow/executor/_executor_sampling.py:829-839`)——即旧链假设"读数=直径、θ=方位",直接极坐标转点再 `fit_circle` 求圆心(偏心)。这与新方案的"支撑函数重建 / 线-圆命中点"是两种几何。
- 输出:`MeasureRowComputationResult`(od/id 的 avg/dev/round/runout/e/phi/center/concentricity 等)。

**跨截面**(`domain/summaries.py::_compute_postcalc_geometry`):
- 输入:`centers_xyz`、`centers_xyz_id`、`concentricity_list`。
- 输出:`straight_od/straight_id`(PCA 主轴拟合后径向 max-min)、`axis_dist`、`conc_max`、`axis_span_max`、`od/id_tilt_deg/end_off/slope`。
- 旧的 `tilt_deg/slope` 已从圆心串拟合轴线倾斜(与新方案 τ 同源数据),但**未用它纠正截面圆度的椭圆假象**。

**主链**(`frp_workflow/autoflow_orchestrator.py`):
- 每截面 `_build_measure_row_from_sampling` 累积 `centers_xyz`;收尾 `_run_postcalc_impl`(L1521)调 `compute_postcalc_result` → `publish_straightness/postcalc` 事件 → `RunRepository` 导出。
- postcalc step 边界:`frp_workflow/steps/postcalc_summary.py`(纯委托 `_run_postcalc_impl`)。

### 1.3 数据结构与持久化 schema

- **CalibrationSnapshot**(`domain/state.py:61`):`od_b_active_mm / od_d_ref_mm / id_delta_c_mm / id_d_ref_mm / id_single_*`,全是标量,无几何位姿。
- **od_calibration.json**:`{B_active, D_ref, cmd_used, out_map{OUT1}, params{...}, defects{...}}`。
- **id_calibration.json**:`{delta_c_mm, D_ref}`。
- **exports/section_results.csv**:固定 39 列(serial…raw)。
- **exports/raw_points.csv**:仅 14 列 `serial,run_id,section_idx,z_pos_mm,sample_idx,ts,theta_deg,bin,phase,od_mm,id_mm,cl_cnt,raw_od,raw_id`。
- **meta.json**:含 `_recipe_dump_dict` 的完整 recipe 白名单 dump。

### 1.4 每截面原始量在哪、字段、单位 ⭐(最关键发现)

**内存 `raw_points` dict 远比落盘 CSV 丰富**(`_executor_sampling.py:785-810`),已含新算法所需几乎全部原始量:

| 新算法需要 | 内存字段 | 单位 | 落盘 raw_points.csv? |
|---|---|---|---|
| 编码器角 θ | `theta_deg` | 度 | ✅ |
| ID 探头 A 读数 L1 | `id_x1_mm` | mm | ❌ 未落盘 |
| ID 探头 B 读数 L2 | `id_x2_mm` | mm | ❌ 未落盘 |
| ID 弦 c(OUT4) | `id_c_mm` | mm | ❌ 未落盘 |
| ID 不对称 m(OUT5) | `id_m_mm` | mm | ❌ 未落盘 |
| OD 左/右边沿 a,b | `od_out1`,`od_out2` | mm | ❌ 未落盘 |
| OD 单边 delta | `od_delta` | mm | ❌ 未落盘 |
| OD 合成直径 | `od_mm` | mm | ✅ |
| ID 合成直径/弦 | `id_mm` | mm | ✅ |

**结论**:
- **在线(实时跑)**:内存 raw_points 已足够喂新算法,postcalc 可直接消费。
- **离线(重跑历史 run)**:现有 CSV 不够(缺 L1/L2/边沿)→ 需 sidecar 扩展(见方案)。
- 还缺新算法独有的 **ID 探头位姿(D/s/q/γ/k)** 与 **OD ψ/远心度** —— 旧链无此概念,需新建 `tooling_calibration.json`。

### 1.5 (a) 新补偿值 → 现有模块落点映射

| 新补偿值 | 骨架函数 | 落点 | 与旧关系 |
|---|---|---|---|
| OD 零位 `B_eff` / 远心度 `k_o0,k_o1` | `od_calibrate_scale`/`od_apply_scale` | 新增 `domain/geometry_calibration.py` | 扩展 `compute_od_b_candidate` |
| ID 基线 `D_eff`、位姿 `s/q/γ/k` | `calibrate_id_tooling`、`id_tooling_from_simple`、`ProbePose` | 新增 `domain/geometry_calibration.py` | 替换/扩展 `solve_id_delta_candidate` |
| OD 轴方位 ψ(+β) | (设计 §Phase2,估计器待补) | geometry_calibration | 全新 |
| 回转轴直线 `axis(z)` | `straightness(..., axis_straightness=)` 入参 | 新增 `domain/geometry_fit.py` + tooling json | 扩展直线度(可传基准) |
| 卡盘误差 E 定界 | `separate_spindle_multistep`(本机仅定界,不分离) | `domain/geometry_fit.py`(保留,不入主链) | 全新(Phase0) |
| 同心度残差 `Δ_reg` / `ref_coaxiality` | `CrossReg`/`concentricity`/`concentricity_uncertainty` | `domain/geometry_fit.py` + tooling json | 替换 row_math 裸相减 |
| 装夹斜率 τ | `centerline_tilt`、`correct_clamping_tilt`、`roundness_corrected_for_tilt` | `domain/geometry_fit.py`,postcalc 调用 | 全新(旧有 tilt_deg 但不纠正圆度) |
| 单截面补偿(去旋转→拟合圆) | `support_to_boundary`、`id_points_from_readings`、`fit_circle_geometric`、`roundness_from_points` | `domain/geometry_fit.py`,row_math 分流 | 替换极坐标转点+fit_circle |

### 1.6 (b) 新旧输入接口差异

| | Legacy | geometry_v2 |
|---|---|---|
| OD 直径 | `od_mm`(=对射宽度 `B−(a+b)`)直接当直径 | `od_out1/out2`(边沿)→ `od_apply_scale` → 支撑函数 → `support_to_boundary` 重建边界点 → 去旋转 → 圆拟合 |
| ID 直径 | `id_mm`(≈`D+L1+L2`)直接当直径,或 `fit_id_diameter` 弦+偏心 | `id_x1/x2`(L1/L2)+ ProbePose → `id_hit_point` → 去旋转 → 圆拟合(消 h≠0 偏置) |
| 必备入参 | 标量 `B_active`/`delta_c_mm` | 每截面整圈 `(θ,L1,L2,边沿a/b)` 序列 + ProbePose/ψ(tooling) |
| raw_points 是否够 | ✅ | 在线够;离线不够(缺 L1/L2/边沿) |

### 1.7 (c) 风险点

1. **raw_points.csv schema**(中):离线对比需 L1/L2/边沿 → 采用 **sidecar `raw_points_ext.csv`**(零侵入,已选定),不动现有 14 列。
2. **section_results/summary schema**:新结果走 **独立 `section_results_v2.csv`**(已选定),旧列不改。
3. **scipy 依赖**:`requirements.txt` 已含 `scipy==1.16.3`、`numpy==2.4.0`;骨架对无 scipy 有退回。
4. **import-linter**:新模块放 `domain/`,仅依赖 core/config/utils + 外部库;骨架只 import numpy/scipy → 合规(`domain-independence` 契约不禁外部库)。
5. **主视图改动**:结果表加 v2 列触及 `application/host/main_view.py` —— 做成纯追加列 + `algo_version` 守卫。
6. **回退**:`algo_version="legacy"` 即回旧链;`tooling_calibration.json`、`*_v2.csv`、`raw_points_ext.csv` 旧版本不读取、保留无害(对齐 PROJECT_OVERVIEW Rollback §5/§6)。

---

## 第 2 步:并行落地方案(已确认切分)

### 2.1 已确认决策

| 决策点 | 选定 |
|---|---|
| 原始量扩展导出 | 旁路 sidecar `raw_points_ext.csv` |
| 新结果并列导出 | 独立文件 `section_results_v2.csv` |
| 新纯函数模块拆分 | 拆两个:`domain/geometry_fit.py` + `domain/geometry_calibration.py` |
| 几何标定 UI 位置 | 新增独立「几何标定 V2」tab |
| 第一版标定 UI 范围 | 高价值子集:ID位姿(Phase3)+ OD方位(Phase2)+ 自检;Phase0/4 只读占位 |
| 结果展示 | 主屏结果表加 v2 列/切换 + 显示 τ |

### 2.2 策略开关与状态
- `Recipe.algo_version: str = "legacy"`(`"legacy"|"geometry_v2"`),`_recipe_dump_dict` 追加 + `RecipeFormMapper` 映射。
- `CalibrationSnapshot.tooling: ToolingCalibration | None = None`(末尾附加,旧读取不变)。

### 2.3 新模块(纯函数,仅 numpy/scipy)
- `domain/geometry_fit.py`:去旋转、Kåsa+几何LM 圆拟合、Fourier 圆度、`support_to_boundary`、`id_points_from_readings`、`concentricity`、`centerline_tilt`、`correct_clamping_tilt`、`straightness`、`separate_spindle_multistep`(保留不入主链)。
- `domain/geometry_calibration.py`:`od_calibrate_scale/od_apply_scale`、`ProbePose/IdTooling`、`calibrate_id_tooling`、`CrossReg`、`ToolingCalibration`。
- 旧 `calibration.py`/`summaries.py`/`row_math.py` 现有函数不改。

### 2.4 标定持久化(只加不动旧文件)
- `tooling_calibration.json`(设计 §7 schema),`CalibrationRepository` 增 `tooling_calibration_file()` + `load/save_tooling_active()` + history;`load_snapshot()` 末尾附加读取(失败→None)。

### 2.5 补偿链分流 + 并列导出
- row_math / `_run_postcalc_impl` 按 `recipe.algo_version` 分流;legacy 路径字节级不变。
- geometry_v2 额外写 `section_results_v2.csv` + sidecar `raw_points_ext.csv`(θ, id_x1/x2/c/m, od_out1/out2/delta,以 `section_idx+sample_idx` 对齐主表)。

### 2.6 回退
- `algo_version="legacy"`(默认)→ 完全回旧链;新增文件保留无害。

---

## 第 2 步附:UI 设计

### A. Recipe 屏 — 新旧算法开关
在现有「算法参数 ▾」折叠组顶部加「几何算法版本」下拉(`algo_version_var`,默认 legacy)+ 灰色回退提示(复用 `recipe_screen.py:624` 提示样式)。未标定(tooling 缺失)运行时自动回退 legacy 并告警。

### B. 标定屏 — 新增「几何标定 V2」tab(gauge_screen notebook 第三页)
沿用现有 `采集→计算(候选 `*_candidate`)→应用(落盘 `*_active`)` 模式,变量前缀 `tcal_*`,回调挂 `calibration_controller`,落盘进 `tooling_calibration.json`(不碰 od/id json)。第一版做:
- **Box0 工装状态**(只读汇总 + `重新加载`/`清除全部`)。
- **Box1 ID 探头位姿(Phase3,多重夹联合 LM)**:入参 r_known/D_init;`采集本次装夹(1圈)`/`加入数据集`/`清空`;`联合拟合(LM)` → 候选 s/axis/q/cost;`应用`。复用现有 OUT4/OUT5 读取路径。
- **Box2 OD 方位 ψ / 远心度(Phase2)**:旋转参考圆柱取单边支撑序列 → 候选 ψ;`采集→计算→应用`。
- **Box5 合成自检**:`运行合成自检` 跑骨架 6 用例,显示通过/恢复误差(纯软件)。
- **Phase0/4 留只读占位**(可手填),后续补全。

### C. 装夹斜率 τ
postcalc 每根自动算(6 截面圆心斜率),无标定按钮;主结果区只读显示 `τx/τy(deg)` + 「已纠正」标记(geometry_v2)。

### D. 主结果 UI
`application/host/main_view.py` 结果表追加 geometry_v2 列 + 新旧切换 + τ,受 `algo_version` 守卫(legacy 隐藏/空),纯追加。

### E. 运行时回退 UX
geometry_v2 启动时 tooling 缺关键段 → 主屏状态条告警「工装未标定,本根回退 legacy」(同 `od_use_edges` 未配 B 的回退提示款式)。

---

## 第 3 步:实现批次(待执行)

按依赖顺序分批,每批跑通测试再进下一批:
1. **纯算法 domain + pytest 自检**(骨架 6 合成用例 → 断言直径/圆度/τ 恢复精度)。
2. **开关/状态/tooling 持久化**。
3. **postcalc 分流 + 并列导出**(`section_results_v2.csv` + `raw_points_ext.csv`)+ 新旧对比测试。
4. **Recipe UI 开关**。
5. **几何标定 V2 tab**(Box0/1/2/5)。
6. **主结果 v2 列/τ 显示**。

**验收**:现有 `pytest -q` 全绿、legacy 行为不变、`lint-imports` 通过、pyright 零错误。

**约束**:每步先读后写;不改 `od/id_calibration.json` 与 exports 现有 schema;新算法默认关闭;所有新逻辑可单测、与 UI 解耦。
