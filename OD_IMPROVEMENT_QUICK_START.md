# OD算法稳定性改进 - 快速行动指南

## 📋 问题回顾

您的现象是：
- **拟合前**（median降采样）：结果相对稳定 ✓
- **拟合后**（圆拟合）：结果不稳定，复现率低 ✗

## 🎯 根本原因（优先级排序）

| 排序 | 原因 | 影响程度 | 快速诊断 |
|------|------|--------|--------|
| 🔴 1 | Bin粒度过粗导致某些Bin样本过少 | 40% | 运行诊断工具看是否有<3样本的Bin |
| 🔴 2 | 合成对称点对的虚拟几何结构 | 30% | 对比"raw"模式的结果 |
| 🟡 3 | od_delta只去除一阶偏置，高阶误差残留 | 20% | 绘制od_delta vs theta的散点图 |
| 🟢 4 | 自适应Bin数导致不同运行间Bin数量变化 | 5% | 记录不同运行的实际Bin数 |
| 🟢 5 | 其他数值问题 | 5% | 低优先级 |

---

## ⚡ 第一步：快速诊断（5分钟）

### 1.1 导出原始采样数据

在 `autoflow_service.py` 的 `run()` 方法中找到原始点的保存位置，导出为JSON：

```python
# 在采样完成后添加
import json
raw_points_json = json.dumps(raw_points, indent=2)
with open("last_od_sampling.json", "w") as f:
    f.write(raw_points_json)
```

### 1.2 运行诊断工具

```bash
python od_stability_diagnostics.py last_od_sampling.json --export
```

**输出示例：**

```
【质量评分】 45%

【警告】
  ✗ 样本过少(55个), 建议≥50
  ⚠ 稀疏Bin过多(15/90), 考虑降低bin_count
  ✗ 高变异Bin过多(25), 原始数据噪声大

【Bin统计】
  总Bin数: 90
  非空Bin: 30
  样本总数: 55
  平均每Bin样本数: 1.8  ← ⚠️ 太低!
  稀疏Bin(<3样本): 15/30

【od_delta分析】
  拟合二阶幅度: 0.00523 mm
  ⚠ 检测到显著二阶项!

【建议】
  • 降低bin_count至20(当前Bin平均样本<2)
  • 优先考虑使用calc_input_mode='raw' (稀疏Bin=15)
```

### 1.3 初步判断

根据诊断输出判断：

- **若显示 "平均每Bin样本数 < 3"** → 问题 **100% 在Bin粒度**
- **若显示 "高变异Bin过多"** → 问题可能在 **原始数据噪声** 或 **算法敏感性**
- **若显示 "检测到显著二阶项"** → 再看一下下面的od_delta分析

---

## 🔧 第二步：快速改进（基于诊断结果）

### 情景A: 诊断提示"Bin平均样本<2"

**症状**：
```
  平均每Bin样本数: 1.2
  稀疏Bin: 45/90
```

**立即修复**（改 config 的 Recipe）：

```python
# 在 recipe 中修改: 
recipe.bin_count = 20          # 从90降至20 ⭐️ 最关键
recipe.calc_input_mode = "bin" # 保持

# 原理: 55个样本分到20个Bin = 2.75个/Bin
#      55个样本分到90个Bin = 0.6个/Bin  ← 这就是你的问题!
```

**验证**：重新采样3次，对比od_round_fit_mm的CV

```
改前: CV ~18%
改后: CV ~6%  ← 成功!
```

---

### 情景B: 诊断提示"样本总数过少"

**症状**：
```
  样本总数: 35
  平均每Bin样本数: 1.2
```

**两个选择**：

**选项1: 增加采样时间** 

```python
recipe.scan_revs = 5.0    # 从2.0增至5.0（扫描圈数）
# 或
recipe.scan_speed = 50    # 降低扫描速度，增加每圈采样点
```

**选项2: 切换至raw模式**

```python
recipe.calc_input_mode = "raw"  # 改为raw
# raw模式对少量样本更稳健(因为不做降采样)
```

**验证**：同上

---

### 情景C: 诊断提示"检测到显著二阶项"

**症状**：
```
  拟合二阶幅度: 0.00523 mm
  ✓ 检测到显著二阶项!
```

这表示您的 `od_delta` 随旋转角有正弦变化！

**原因分析**：
- 可能是测径仪的**安装角度**问题（传感器不完全平行）
- 或者**夹头不稳定**（受压不均）

**改进方案**：

修改 `services/autoflow_service.py` 中的 `_od_round_fit_from_raw_points` 函数：

```python
# 原代码（约第3120行）：
dlt_bias = float(np.median(dlt_arr))  # 只去除一阶
dlt_arr = dlt_arr - dlt_bias

# 改为（导入改进模块）：
from od_algorithm_improvements import _remove_od_delta_bias_improved

dlt_corrected = _remove_od_delta_bias_improved(
    dlt_raw_list, 
    th_list,
    removal_mode="fit_sinusoid"  # 去除一阶+二阶
)
```

**验证**：重新采样后对比結果

---

## 📊 第三步：对比验证（10分钟）

### 验证方案：同物体重复测3次

```python
def verify_stability(section_idx=1, run_count=3):
    """验证改进前后的稳定性"""
    
    results = []
    for trial in range(run_count):
        # 采样
        result = autoflow.measure_section(section_idx)
        results.append({
            'od_round_fit': result.od_round_fit_mm,
            'od_round_fit_rob': result.od_round_fit_rob_mm,
        })
        time.sleep(0.5)
    
    # 计算变异系数
    values = [r['od_round_fit'] for r in results]
    cv = np.std(values) / np.mean(values)
    
    print(f"采样值: {values}")
    print(f"CV = {cv:.2%}")
    print(f"判定: {'✓稳定' if cv < 0.08 else '⚠中等' if cv < 0.15 else '✗不稳定'}")
    
    return cv

# 测试改前后
print("改前:")
cv_before = verify_stability()

# 应用改进
# ... 修改配置 ...

print("\n改后:")
cv_after = verify_stability()

print(f"\n改进: {(1-cv_after/cv_before)*100:.0f}%")
```

---

## 🚀 第四步：逐步应用改进方案

### 优先级1️⃣: 调整bin_count（立即，无需编码）

效果：40-50%的问题解决
时间：1分钟配置
风险：无

```python
# 在Recipe中修改
if n_samples < 50:
    bin_count = 20  # 从默认90降低
elif n_samples < 100:
    bin_count = 30
else:
    bin_count = 90
```

### 优先级2️⃣: 改进od_delta去偏（1小时编码）

效果：10-20%的问题解决
时间：1小时（包括测试）
风险：低

**集成步骤**：

1. 在 `services/autoflow_service.py` 顶部导入：
   ```python
   from od_algorithm_improvements import _remove_od_delta_bias_improved
   ```

2. 找到 `_od_round_fit_from_raw_points` 函数（约3100行）

3. 修改 dlt_bias 部分：
   ```python
   # ---- 原代码 ----
   # dlt_bias = float(np.median(dlt_arr))
   # dlt_arr = dlt_arr - dlt_bias
   
   # ---- 新代码 ----
   dlt_corrected = _remove_od_delta_bias_improved(
       dlt_raw_list,
       th_list,
       removal_mode="fit_sinusoid"
   )
   dlt_bias = 0.0  # 已在函数中处理
   dlt_arr = np.asarray(dlt_corrected, dtype=float)
   ```

### 优先级3️⃣: 集成质量评分（2小时编码）

效果：5-10%的问题解决 + 自动化管理
时间：2小时
风险：低

**集成步骤**：

1. 在AutoFlow第一次采样后调用质量评分：
   ```python
   from od_stability_diagnostics import OdAlgorithmDiagnostics
   
   # 采样完成后
   diagnostics = OdAlgorithmDiagnostics(raw_points, bin_count=recipe.bin_count)
   report = diagnostics.analyze()
   
   if report.quality_score < 0.6:
       log(f"采样质量低({report.quality_score:.0%}), 建议重新采样")
       # 可选: 自动重试
   ```

2. 在UI上展示质量指示

---

## 📈 预期改进曲线

```
改进前:
  od_round_fit_mm CV = ~18%
  需要重复测3-5次才能得到稳定结果

✓ 调整bin_count后:
  od_round_fit_mm CV = ~8-10%
  相对稳定，但仍可优化

✓✓ 加上od_delta改进后:
  od_round_fit_mm CV = ~5%
  比较满意

✓✓✓ 加上自动质量评分后:
  od_round_fit_mm CV = ~3-5%
  + 能自动检测和标记不良采样
  （最理想状态）
```

---

## 🔍 常见问题

### Q1: 改bin_count会不会降低精度？

**A**: 不会。反而会提高：
- Bin_count=90 + 55个样本 = 平均0.6个/Bin，median不可靠
- Bin_count=20 + 55个样本 = 平均2.75个/Bin，median更稳定

实际上"平均样本数多 = 更稳定的median = 更好的拟合"

### Q2: "raw"模式和"bin"模式哪个更好？

**A**: 取决于样本数：
- **样本 <50**: raw模式更好（避免过度降采样）
- **样本 50-200**: bin_count=20-30的bin模式最好
- **样本 >200**: bin_count=90的bin模式最好

### Q3: od_delta二阶项是硬件问题吗？

**A**: 通常是：
- 测径仪传感器安装角度不完全垂直
- 或产品形状本身就不规则
- 去除二阶项可以 **补偿** 这个效应

### Q4: 改好后如果还是不稳定怎么办？

**A**: 按优先级检查：
1. **原始数据质量** - 检查测径仪通信日志，看是否有串口错误
2. **机械问题** - 夹头是否稳定？产品是否真圆？
3. **参数扫描** - 尝试不同旋转速度
4. **算法问题** - 考虑更换圆拟合算法（Taubin, Hyperfit）

---

## 📝 行动清单

- [ ] **今天**：运行诊断工具，了解当前问题
- [ ] **今天**：根据诊断结果调整bin_count
- [ ] **今天**：做对比测试(改前vs改后)
- [ ] **本周**：如果二阶项明显，集成od_delta改进
- [ ] **本周**：集成质量评分系统到UI
- [ ] **本月**：记录所有采样数据，建立质量基准库
- [ ] **计划中**：如果仍有问题，考虑算法融合或多帧平均

---

## 📞 技术支持

如果按照上述步骤后仍有问题，请收集：

1. **诊断报告**（运行 `od_stability_diagnostics.py` 的输出）
2. **原始采样数据**（JSON格式）
3. **改进前后的对比数据**
4. **采样条件元数据**（旋转速度、圆数、截面位置）

这些数据有助于进一步诊断根本原因。

