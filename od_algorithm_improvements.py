"""OD拟合算法改进方案代码

这个模块提供了多个改进方案，可以集成到 services/autoflow_service.py 中
"""

import numpy as np
import math
from typing import List, Optional, Tuple


# ============================================================================
# 方案A: 增强Bin鲁棒性 (推荐优先级: 最高)
# ============================================================================

def _adaptive_bin_strategy_improved(
    n_samples: int,
    requested_bin_count: int = 90,
    min_samples_per_bin: int = 5,
) -> Tuple[int, str]:
    """
    改进的自适应bin选择策略.
    
    返回: (adaptive_bin_count, strategy_description)
    
    原理:
    - 样本少时: 自动降低bin数，确保每bin≥min_samples_per_bin个样本
    - 样本多时: 使用完整的bin_count
    
    示例:
      n_samples=50, requested=90
      → bin_count=10, "Conservative(样本尽够)"
      
      n_samples=500, requested=90
      → bin_count=90, "Full resolution"
    """
    requested = max(3, int(requested_bin_count))
    ns = int(n_samples)
    
    if ns < 12:
        # 样本太少，切换到raw模式
        return 0, "FORCE_RAW_MODE (样本<12)"
    
    # 计算能容纳的最大bin数
    max_safe_bins = max(3, ns // min_samples_per_bin)
    
    if max_safe_bins < 10:
        # 样本有限，用保守策略
        adaptive = max(3, max_safe_bins)
        return adaptive, f"Conservative (每bin≥{min_samples_per_bin})"
    
    if max_safe_bins >= requested:
        # 样本充足，用完整分辨率
        return requested, "Full resolution"
    
    # 中等样本，折中
    return max_safe_bins, f"Balanced (每bin~{ns//max_safe_bins})"


def _detect_sparse_bins_and_filter(
    th_list: List[float],
    rr_list: List[float],
    rl_list: List[float],
    bin_count: int,
    sparse_threshold: int = 3,
) -> Tuple[List[float], List[float], List[float], List[int]]:
    """
    检测稀疏bin，并对其进行IQR异常值过滤.
    
    返回:
      (filtered_th, filtered_rr, filtered_rl, affected_bin_indices)
    
    策略:
    - bin样本<sparse_threshold: 标记为"稀疏"
    - 对稀疏bin应用IQR异常值检测，去除极值
    - 对正常bin保持原样
    """
    n = bin_count
    
    # 分bin
    bin_th = [[] for _ in range(n)]
    bin_rr = [[] for _ in range(n)]
    bin_rl = [[] for _ in range(n)]
    
    for th, rr, rl in zip(th_list, rr_list, rl_list):
        bin_idx = int((float(th) / 360.0) * n)
        if bin_idx >= n:
            bin_idx = 0
        bin_th[bin_idx].append(th)
        bin_rr[bin_idx].append(rr)
        bin_rl[bin_idx].append(rl)
    
    # 过滤稀疏bin中的异常值
    affected_bins = []
    
    for i in range(n):
        if not bin_rr[i]:
            continue
        
        bin_size = len(bin_rr[i])
        
        if bin_size < sparse_threshold and bin_size >= 2:
            # 稀疏bin: 应用IQR过滤
            affected_bins.append(i)
            
            for data_list in [bin_rr[i], bin_rl[i]]:
                arr = np.asarray(data_list, dtype=float)
                q1, q3 = np.percentile(arr, [25, 75])
                iqr = q3 - q1
                
                # 删除超过1.5*IQR的异常值
                mask = (arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)
                new_list = arr[mask].tolist()
                
                # 防止过滤后为空
                if new_list:
                    data_list.clear()
                    data_list.extend(new_list)
    
    # 重新整理为flat列表
    filtered_th = []
    filtered_rr = []
    filtered_rl = []
    
    for i in range(n):
        filtered_th.extend(bin_th[i])
        filtered_rr.extend(bin_rr[i])
        filtered_rl.extend(bin_rl[i])
    
    return filtered_th, filtered_rr, filtered_rl, affected_bins


# ============================================================================
# 方案B: 改进od_delta处理 (优先级: 中)
# ============================================================================

def _remove_od_delta_bias_improved(
    dlt_raw_list: List[float],
    th_list: List[float],
    removal_mode: str = "offset_only",
) -> List[float]:
    """
    改进的od_delta偏置去除.
    
    removal_mode:
      - "offset_only": 仅去除median偏置 (当前方式)
      - "fit_sinusoid": 去除offset + 一阶正弦项 (推荐新方式)
      - "fit_quad": 去除offset + 一阶 + 二阶项 (激进)
    
    返回: 修正后的od_delta列表
    """
    dlt_arr = np.asarray(dlt_raw_list, dtype=float)
    th_arr = np.asarray(th_list, dtype=float)
    
    if dlt_arr.size < 3:
        return dlt_raw_list
    
    if removal_mode == "offset_only":
        # 原始方式
        dlt_bias = float(np.median(dlt_arr[np.isfinite(dlt_arr)]))
        return (dlt_arr - dlt_bias).tolist()
    
    if removal_mode == "fit_sinusoid":
        # 拟合 dlt = a0 + A*cos(θ) + B*sin(θ)
        th_rad = np.deg2rad(th_arr)
        
        A_fit = np.column_stack([
            np.ones_like(th_rad),
            np.cos(th_rad),
            np.sin(th_rad),
        ])
        
        # 加权最小二乘(可选)
        try:
            coef, _, _, _ = np.linalg.lstsq(A_fit, dlt_arr, rcond=None)
            a0, a, b = coef
            
            # 去除offset和一阶正弦
            fitted_sin = a * np.cos(th_rad) + b * np.sin(th_rad)
            dlt_corrected = dlt_arr - a0 - fitted_sin
            
            return dlt_corrected.tolist()
        except Exception:
            # 回退到offset_only
            dlt_bias = float(np.median(dlt_arr[np.isfinite(dlt_arr)]))
            return (dlt_arr - dlt_bias).tolist()
    
    if removal_mode == "fit_quad":
        # 更激进: 拟合二阶项
        th_rad = np.deg2rad(th_arr)
        
        A_fit = np.column_stack([
            np.ones_like(th_rad),
            np.cos(th_rad),
            np.sin(th_rad),
            np.cos(2 * th_rad),
            np.sin(2 * th_rad),
        ])
        
        try:
            coef, _, _, _ = np.linalg.lstsq(A_fit, dlt_arr, rcond=None)
            
            fitted = A_fit @ coef
            dlt_corrected = dlt_arr - fitted + float(np.median(fitted))
            
            return dlt_corrected.tolist()
        except Exception:
            dlt_bias = float(np.median(dlt_arr[np.isfinite(dlt_arr)]))
            return (dlt_arr - dlt_bias).tolist()
    
    return dlt_raw_list


# ============================================================================
# 方案C: 加权圆拟合
# ============================================================================

def _compute_point_weights_from_bins(
    pts: List[Tuple[float, float]],
    theta_deg_list: List[float],
    bin_sizes: Optional[List[int]] = None,
    weight_method: str = "log",
) -> np.ndarray:
    """
    根据点来自的bin大小，计算每个点的拟合权重.
    
    weight_method:
      - "log": weight = log1p(bin_size)
      - "sqrt": weight = sqrt(bin_size)
      - "linear": weight = bin_size
      - "inv": weight = 1 / bin_size
    
    理由:
    - 来自大bin的点更可靠 → 权重高
    - 来自稀疏bin的点易被噪声主导 → 权重低
    """
    if bin_sizes is None or len(bin_sizes) != len(theta_deg_list):
        # 默认权重相同
        return np.ones(len(pts), dtype=float)
    
    weights = []
    
    for i, (th_deg, bin_size) in enumerate(zip(theta_deg_list, bin_sizes)):
        if bin_size <= 0:
            w = 1.0
        elif weight_method == "log":
            w = np.log1p(bin_size)
        elif weight_method == "sqrt":
            w = np.sqrt(bin_size)
        elif weight_method == "linear":
            w = float(bin_size)
        elif weight_method == "inv":
            w = 1.0 / float(bin_size)
        else:
            w = 1.0
        
        # 每个点pair(右+左)都用同样权重
        weights.append(w)
        weights.append(w)
    
    return np.asarray(weights, dtype=float)


# ============================================================================
# 方案D: 分阶段质量评估
# ============================================================================

class OdFittingQualityAssessment:
    """OD拟合质量评分系统"""
    
    @staticmethod
    def assess_sample_quality(
        th_list: List[float],
        rr_list: List[float],
        rl_list: List[float],
        bin_count: int = 90,
    ) -> dict:
        """
        对采样数据进行多维评分.
        
        返回评分dict:
        {
            'score': 0.0-1.0,
            'coverage': 0.0-1.0,         # 角度覆盖率
            'uniformity': 0.0-1.0,       # 样本分布均匀度
            'stability': 0.0-1.0,        # 数据稳定性(低CV)
            'recommendation': str,
        }
        """
        n = len(th_list)
        if n < 8:
            return {
                'score': 0.0,
                'coverage': 0.0,
                'uniformity': 0.0,
                'stability': 0.0,
                'recommendation': '样本过少(<8), 必须重新采样',
            }
        
        # 1. 覆盖率
        th_arr = np.asarray(th_list, dtype=float)
        th_sorted = np.sort(th_arr)
        th_gaps = np.diff(np.concatenate([[th_sorted[0] - 360], th_sorted, [th_sorted[-1] + 360]]))
        max_gap = float(np.max(th_gaps))
        coverage = 1.0 - min(1.0, max_gap / 90.0)  # 期望最大间隔<90°
        
        # 2. 均匀度 (std of bins per sample)
        bin_counts = [0] * bin_count
        for th in th_list:
            b = int((float(th) / 360.0) * bin_count)
            if b >= bin_count:
                b = 0
            bin_counts[b] += 1
        
        bin_counts_nonzero = [c for c in bin_counts if c > 0]
        if bin_counts_nonzero:
            samples_per_bin_mean = np.mean(bin_counts_nonzero)
            samples_per_bin_std = np.std(bin_counts_nonzero)
            uniformity = 1.0 - min(1.0, samples_per_bin_std / (samples_per_bin_mean + 1e-9))
        else:
            uniformity = 0.0
        
        # 3. 稳定性 (CoeffVar)
        cv_rr = np.std(rr_list) / (np.mean(rr_list) + 1e-9)
        cv_rl = np.std(rl_list) / (np.mean(rl_list) + 1e-9)
        cv_mean = (cv_rr + cv_rl) / 2.0
        stability = max(0.0, 1.0 - cv_mean)
        
        # 综合评分
        score = 0.4 * coverage + 0.3 * uniformity + 0.3 * stability
        
        # 推荐
        if score >= 0.8:
            rec = "✓优秀, 可使用bin模式"
        elif score >= 0.6:
            rec = "⚠中等, 建议bin_count=30"
        elif score >= 0.4:
            rec = "⚠较差, 建议bin_count=20或切换raw"
        else:
            rec = "✗很差, 需要重新采样"
        
        return {
            'score': float(score),
            'coverage': float(coverage),
            'uniformity': float(uniformity),
            'stability': float(stability),
            'recommendation': rec,
        }


# ============================================================================
# 使用示例 (可集成到AutoFlow._od_round_fit_from_raw_points)
# ============================================================================

def _od_round_fit_from_raw_points_improved(
    raw_points: List[dict],
    calc_input_mode: str = "auto",  # "auto" / "bin" / "raw"
    bin_count: int = 90,
    bin_method: str = "median",
    pp_mode: str = "p99_p1",
    od_delta_removal_mode: str = "fit_sinusoid",  # 改进: 使用更好的去偏方法
) -> Tuple[Optional[float], Optional[float]]:
    """
    改进版的OD真圆度计算 (集成多个改进方案).
    
    主要改变:
    1. 自适应模式选择 (calc_input_mode="auto")
    2. 智能Bin数调整
    3. 稀疏Bin异常值过滤
    4. 改进的od_delta去偏
    
    使用方式:
      od_round_fit_mm, od_round_fit_rob_mm = _od_round_fit_from_raw_points_improved(
          raw_points,
          calc_input_mode="auto",
          bin_count=90,
      )
    """
    # 步骤1: 数据提取(与原始相同)
    th_list = []
    ts_list = []
    r_list = []
    dlt_raw_list = []
    
    for pnt in raw_points or []:
        if not isinstance(pnt, dict):
            continue
        
        th = pnt.get("theta_deg")
        ts = pnt.get("ts")
        od_mm = pnt.get("od_mm")
        od_delta = pnt.get("od_delta", 0.0)
        
        if th is None or ts is None or od_mm is None:
            continue
        
        try:
            thf = float(th) % 360.0
            tsf = float(ts)
            df = float(od_mm)
            dlt = float(od_delta or 0.0)
        except Exception:
            continue
        
        if (not math.isfinite(thf)) or (not math.isfinite(tsf)) or \
           (not math.isfinite(df)) or df <= 0.0:
            continue
        
        r = 0.5 * df
        th_list.append(thf)
        ts_list.append(tsf)
        r_list.append(r)
        if not math.isfinite(dlt):
            dlt = 0.0
        dlt_raw_list.append(dlt)
    
    if len(th_list) < 3:
        return None, None
    
    # 步骤2: 自适应模式和bin选择
    if calc_input_mode == "auto":
        # 自动判断使用raw还是bin
        n_samples = len(th_list)
        if n_samples < 20:
            mode = "raw"
        elif n_samples < 50:
            mode = "bin"
            bin_count = min(20, max(3, n_samples // 5))
        else:
            mode = "bin"
            adaptive_bin, _ = _adaptive_bin_strategy_improved(
                n_samples, bin_count, min_samples_per_bin=5
            )
            bin_count = adaptive_bin
    else:
        mode = calc_input_mode
    
    # 步骤3: 改进的od_delta去偏
    dlt_list = _remove_od_delta_bias_improved(
        dlt_raw_list, th_list, removal_mode=od_delta_removal_mode
    )
    
    # 计算右左半径
    rr_list = [r + d for r, d in zip(r_list, dlt_list)]
    rl_list = [r - d for r, d in zip(r_list, dlt_list)]
    
    # 步骤4: 稀疏Bin过滤
    if mode == "bin":
        th_list, rr_list, rl_list, affected_bins = _detect_sparse_bins_and_filter(
            th_list, rr_list, rl_list, bin_count, sparse_threshold=3
        )
        
        # 如果过滤后样本过少，降级为raw
        if len(th_list) < 6:
            mode = "raw"
    
    # 步骤5: 生成点集并拟合 (与原始逻辑相同)
    pts = []
    if mode == "raw":
        for th_deg, rr, rl in zip(th_list, rr_list, rl_list):
            th = math.radians(th_deg)
            c = math.cos(th)
            s = math.sin(th)
            pts.append((rr * c, rr * s))
            pts.append((-rl * c, -rl * s))
    else:
        # bin模式(与原始相同)
        n = bin_count
        rr_bins = [[] for _ in range(n)]
        rl_bins = [[] for _ in range(n)]
        
        for th_deg, rr, rl in zip(th_list, rr_list, rl_list):
            b = int((th_deg / 360.0) * n)
            if b >= n:
                b = 0
            rr_bins[b].append(rr)
            rl_bins[b].append(rl)
        
        for i in range(n):
            if (not rr_bins[i]) or (not rl_bins[i]):
                continue
            
            th = math.radians((i + 0.5) * (360.0 / n))
            c = math.cos(th)
            s = math.sin(th)
            
            rr = float(np.median(rr_bins[i]))
            rl = float(np.median(rl_bins[i]))
            
            if math.isfinite(rr) and math.isfinite(rl):
                pts.append((rr * c, rr * s))
                pts.append((-rl * c, -rl * s))
    
    if len(pts) < 6:
        return None, None
    
    # 步骤6: 圆拟合和真圆度计算 (调用原始的_fit_circle)
    # 这里假设 _fit_circle 可通过self访问
    # xc, yc, r_fit, _sigma = self._fit_circle(np.asarray(pts, dtype=float))
    
    coords = np.asarray(pts, dtype=float)
    # 由于这是独立函数，圆拟合部分需要从原始代码中保留
    # 或通过dependency injection传入
    
    # ... 后续与原始相同 ...
    
    return None, None  # 占位符
