#!/usr/bin/env python3
"""OD圆拟合稳定性诊断工具

用法:
  python od_stability_diagnostics.py <raw_points_json> [--export-plots]
  
或在AutoFlow中调用:
  diagnostics = OdAlgorithmDiagnostics(raw_points, recipe)
  report = diagnostics.analyze()
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class BinStatistics:
    """单个bin的统计信息"""
    bin_idx: int
    theta_range: Tuple[float, float]
    rr_samples: List[float]
    rl_samples: List[float]
    rr_median: float
    rl_median: float
    rr_cv: float  # CoeffVar
    rl_cv: float
    sample_count: int
    is_sparse: bool  # <3个样本


@dataclass
class DiagnosticReport:
    """诊断报告"""
    quality_score: float  # 0-1
    bin_statistics: List[BinStatistics]
    od_delta_analysis: Dict
    recommendation: str
    warnings: List[str]


class OdAlgorithmDiagnostics:
    """OD算法诊断器"""

    def __init__(
        self,
        raw_points: List[Dict],
        bin_count: int = 90,
        bin_method: str = "median",
    ):
        """
        Args:
            raw_points: 原始采样点列表，每个点包含:
                - theta_deg
                - ts
                - od_mm
                - od_delta (可选)
            bin_count: bin数量
            bin_method: 降采样方法 ("median" / "mean")
        """
        self.raw_points = raw_points
        self.bin_count = bin_count
        self.bin_method = bin_method

        # 提取数据
        self.th_list = []
        self.ts_list = []
        self.r_list = []
        self.dlt_raw_list = []

        self._extract_data()

    def _extract_data(self):
        """从raw_points提取有效数据"""
        for pnt in self.raw_points or []:
            if not isinstance(pnt, dict):
                continue

            th = pnt.get("theta_deg", None)
            ts = pnt.get("ts", None)
            od_mm = pnt.get("od_mm", None)
            od_delta = pnt.get("od_delta", 0.0)

            if th is None or ts is None or od_mm is None:
                continue

            try:
                thf = float(th)
                tsf = float(ts)
                df = float(od_mm)
                dlt = float(od_delta or 0.0)
            except Exception:
                continue

            if (
                not math.isfinite(thf)
                or not math.isfinite(tsf)
                or not math.isfinite(df)
                or df <= 0.0
            ):
                continue

            r = 0.5 * df
            self.th_list.append(float(thf) % 360.0)
            self.ts_list.append(float(tsf))
            self.r_list.append(float(r))
            if not math.isfinite(dlt):
                dlt = 0.0
            self.dlt_raw_list.append(float(dlt))

    def analyze(self) -> DiagnosticReport:
        """执行完整诊断"""
        warnings = []

        # 1. 样本数检查
        n_samples = len(self.th_list)
        if n_samples < 8:
            warnings.append(f"✗ 样本过少({n_samples}个), 建议≥50")

        # 2. 分bin分析
        bin_stats = self._analyze_bins()

        # 3. od_delta分析
        od_delta_analysis = self._analyze_od_delta()

        # 4. 综合评分
        quality_score = self._compute_quality_score(bin_stats, od_delta_analysis, n_samples)

        # 5. 推荐
        recommendation = self._get_recommendation(quality_score, bin_stats, n_samples)

        if quality_score < 0.6:
            warnings.append(
                f"⚠ 拟合质量低({quality_score:.1%}), 考虑重新采样或调整bin参数"
            )

        sparse_bins = sum(1 for bs in bin_stats if bs.is_sparse)
        if sparse_bins > len(bin_stats) * 0.1:
            warnings.append(
                f"⚠ 稀疏bin过多({sparse_bins}/{len(bin_stats)}), 建议降低bin_count"
            )

        high_cv_bins = sum(
            1 for bs in bin_stats if bs.rr_cv > 0.15 or bs.rl_cv > 0.15
        )
        if high_cv_bins > len(bin_stats) * 0.2:
            warnings.append(
                f"✗ 高变异bin过多({high_cv_bins}), 原始数据噪声大"
            )

        return DiagnosticReport(
            quality_score=quality_score,
            bin_statistics=bin_stats,
            od_delta_analysis=od_delta_analysis,
            recommendation=recommendation,
            warnings=warnings,
        )

    def _analyze_bins(self) -> List[BinStatistics]:
        """分析各bin的样本分布"""
        stats_list = []
        n = self.bin_count

        # 初始化bin
        rr_bins = [[] for _ in range(n)]
        rl_bins = [[] for _ in range(n)]

        # 计算od_delta偏置
        dlt_arr = np.asarray(self.dlt_raw_list, dtype=float)
        dlt_arr_clean = dlt_arr[np.isfinite(dlt_arr)]
        dlt_bias = float(np.median(dlt_arr_clean)) if dlt_arr_clean.size else 0.0

        # 分bin
        for th_deg, r, dlt_raw in zip(self.th_list, self.r_list, self.dlt_raw_list):
            d = float(dlt_raw) - float(dlt_bias)
            rr = float(r) + d
            rl = float(r) - d

            bin_idx = int((float(th_deg) / 360.0) * n)
            if bin_idx >= n:
                bin_idx = 0

            rr_bins[bin_idx].append(float(rr))
            rl_bins[bin_idx].append(float(rl))

        # 统计
        for i in range(n):
            rr_list = rr_bins[i]
            rl_list = rl_bins[i]

            if not rr_list or not rl_list:
                continue

            rr_med = float(np.median(rr_list))
            rl_med = float(np.median(rl_list))
            rr_cv = float(np.std(rr_list)) / (abs(rr_med) + 1e-9) if len(rr_list) > 1 else 0.0
            rl_cv = float(np.std(rl_list)) / (abs(rl_med) + 1e-9) if len(rl_list) > 1 else 0.0

            sample_count = len(rr_list)
            is_sparse = sample_count < 3

            theta_min = (i * 360.0) / n
            theta_max = ((i + 1) * 360.0) / n

            stats_list.append(
                BinStatistics(
                    bin_idx=i,
                    theta_range=(theta_min, theta_max),
                    rr_samples=rr_list,
                    rl_samples=rl_list,
                    rr_median=rr_med,
                    rl_median=rl_med,
                    rr_cv=rr_cv,
                    rl_cv=rl_cv,
                    sample_count=sample_count,
                    is_sparse=is_sparse,
                )
            )

        return stats_list

    def _analyze_od_delta(self) -> Dict:
        """分析od_delta的统计特性"""
        dlt_arr = np.asarray(self.dlt_raw_list, dtype=float)
        dlt_arr = dlt_arr[np.isfinite(dlt_arr)]

        if dlt_arr.size == 0:
            return {"status": "no_delta_data"}

        result = {
            "status": "ok",
            "count": int(dlt_arr.size),
            "mean": float(np.mean(dlt_arr)),
            "median": float(np.median(dlt_arr)),
            "std": float(np.std(dlt_arr)),
            "min": float(np.min(dlt_arr)),
            "max": float(np.max(dlt_arr)),
            "range": float(np.max(dlt_arr) - np.min(dlt_arr)),
        }

        # 检查是否存在二阶结构 (与theta的相关性)
        if len(self.th_list) >= 8:
            th_rad = np.deg2rad(np.asarray(self.th_list, dtype=float))
            
            # 尝试拟合: dlt = a0 + a*cos(th) + b*sin(th)
            A_fit = np.column_stack([
                np.ones_like(th_rad),
                np.cos(th_rad),
                np.sin(th_rad)
            ])
            try:
                coef, residuals, _, _ = np.linalg.lstsq(A_fit, dlt_arr, rcond=None)
                a0, a, b = coef
                
                # 二阶幅度
                amplitude = float(np.hypot(a, b))
                phase = float(np.rad2deg(np.arctan2(b, a)))
                
                # 拟合优度
                if residuals.size > 0:
                    rmse = float(np.sqrt(residuals[0] / len(dlt_arr)))
                else:
                    rmse = 0.0
                
                result["fit_offset"] = float(a0)
                result["fit_amplitude"] = amplitude
                result["fit_phase_deg"] = phase
                result["fit_rmse"] = rmse
                
                # 判断是否存在显著的二阶项
                amplitude_ratio = amplitude / (abs(a0) + 1e-9) if abs(a0) > 1e-6 else 0.0
                result["has_sinusoid"] = amplitude_ratio > 0.2  # 二阶项 > 20% offset
                
            except Exception as e:
                result["fit_status"] = f"error: {str(e)}"

        return result

    def _compute_quality_score(
        self, bin_stats: List[BinStatistics], od_delta_analysis: Dict, n_samples: int
    ) -> float:
        """计算综合质量评分 (0-1)"""
        scores = {}

        # 1. 样本充足度
        scores["sample_count"] = min(1.0, n_samples / 100.0)

        # 2. Bin填充度
        if bin_stats:
            filled_bins = len(bin_stats)
            total_bins = self.bin_count
            scores["bin_fullness"] = filled_bins / total_bins
        else:
            scores["bin_fullness"] = 0.0

        # 3. Bin丰富度 (每bin平均样本数)
        if bin_stats:
            avg_samples_per_bin = n_samples / len(bin_stats)
            scores["sample_per_bin"] = min(1.0, avg_samples_per_bin / 5.0)
        else:
            scores["sample_per_bin"] = 0.0

        # 4. 稀疏bin比例
        if bin_stats:
            sparse_count = sum(1 for bs in bin_stats if bs.is_sparse)
            scores["no_sparse_bins"] = 1.0 - (sparse_count / len(bin_stats))
        else:
            scores["no_sparse_bins"] = 0.0

        # 5. od_delta一致性
        if od_delta_analysis.get("status") == "ok":
            # 低std更好
            std = od_delta_analysis.get("std", 0.0)
            range_val = od_delta_analysis.get("range", 0.0)
            if range_val > 1e-6:
                cv = std / range_val
                scores["delta_consistency"] = max(0.0, 1.0 - cv)
            else:
                scores["delta_consistency"] = 1.0
            
            # 是否存在二阶项?
            if od_delta_analysis.get("has_sinusoid"):
                scores["delta_consistency"] *= 0.8  # 扣分
        else:
            scores["delta_consistency"] = 0.5  # 无法评估,中等评分

        # 6. Bin内的变异系数
        if bin_stats:
            high_cv_bins = sum(
                1 for bs in bin_stats if bs.rr_cv > 0.15 or bs.rl_cv > 0.15
            )
            scores["low_cv"] = 1.0 - (high_cv_bins / len(bin_stats))
        else:
            scores["low_cv"] = 0.0

        # 加权平均
        weights = {
            "sample_count": 0.15,
            "bin_fullness": 0.15,
            "sample_per_bin": 0.20,
            "no_sparse_bins": 0.15,
            "delta_consistency": 0.20,
            "low_cv": 0.15,
        }

        total_weight = sum(weights.values())
        quality = sum(scores.get(k, 0.0) * v for k, v in weights.items()) / total_weight

        return max(0.0, min(1.0, quality))

    def _get_recommendation(
        self, quality_score: float, bin_stats: List[BinStatistics], n_samples: int
    ) -> str:
        """生成改进建议"""
        if quality_score >= 0.85:
            return "✓ 质量优秀, 无需调整"

        recommendations = []

        # 样本不足
        if n_samples < 30:
            recommendations.append("• 增加采样速度或扫描圈数 (当前样本过少)")

        # Bin过密
        if bin_stats and n_samples / len(bin_stats) < 2:
            recommendations.append(f"• 降低bin_count至{max(10, self.bin_count // 3)}(当前Bin平均样本<2)")

        # 稀疏bin过多
        sparse_bins = sum(1 for bs in bin_stats if bs.is_sparse)
        if sparse_bins > len(bin_stats) * 0.2:
            recommendations.append(f"• 优先考虑使用calc_input_mode='raw' (稀疏Bin={sparse_bins})")

        # 噪声大
        high_cv = sum(1 for bs in bin_stats if bs.rr_cv > 0.2 or bs.rl_cv > 0.2)
        if high_cv > len(bin_stats) * 0.3:
            recommendations.append("• 检查测径仪硬件或通信质量(高噪声信号)")

        if not recommendations:
            recommendations.append("• 建议使用'raw'模式替代'bin'模式尝试")

        return "\n".join(recommendations)

    def print_report(self, report: DiagnosticReport):
        """打印诊断报告"""
        print("=" * 70)
        print("OD算法诊断报告")
        print("=" * 70)

        print(f"\n【质量评分】 {report.quality_score:.1%}")

        if report.warnings:
            print("\n【警告】")
            for w in report.warnings:
                print(f"  {w}")

        print("\n【Bin统计】")
        print(f"  总Bin数: {self.bin_count}")
        print(f"  非空Bin: {len(report.bin_statistics)}")
        print(f"  样本总数: {len(self.th_list)}")

        if report.bin_statistics:
            samples_per_bin = len(self.th_list) / len(report.bin_statistics)
            sparse_count = sum(1 for bs in report.bin_statistics if bs.is_sparse)
            print(f"  平均每Bin样本数: {samples_per_bin:.1f}")
            print(f"  稀疏Bin(<3样本): {sparse_count}/{len(report.bin_statistics)}")

            # 找出最稀疏和最繁忙的bin
            if report.bin_statistics:
                min_bin = min(report.bin_statistics, key=lambda x: x.sample_count)
                max_bin = max(report.bin_statistics, key=lambda x: x.sample_count)
                print(f"  最稀疏Bin: {min_bin.bin_idx} (θ={min_bin.theta_range[0]:.0f}°, n={min_bin.sample_count})")
                print(f"  最繁忙Bin: {max_bin.bin_idx} (θ={max_bin.theta_range[0]:.0f}°, n={max_bin.sample_count})")

        print("\n【od_delta分析】")
        od_delta = report.od_delta_analysis
        if od_delta.get("status") == "ok":
            print(f"  均值: {od_delta['mean']:.5f} mm")
            print(f"  中位数: {od_delta['median']:.5f} mm")
            print(f"  标准差: {od_delta['std']:.5f} mm")
            print(f"  范围: [{od_delta['min']:.5f}, {od_delta['max']:.5f}] mm")

            if "fit_amplitude" in od_delta:
                print(f"  拟合二阶幅度: {od_delta['fit_amplitude']:.5f} mm")
                print(f"  拟合二阶相位: {od_delta['fit_phase_deg']:.1f}°")
                if od_delta.get("has_sinusoid"):
                    print(f"  ⚠ 检测到显著二阶项!")
        else:
            print(f"  无od_delta数据")

        print("\n【建议】")
        print(report.recommendation)

        print("\n" + "=" * 70)

    def export_diagnostics_json(self, output_file: Path):
        """导出诊断数据为JSON"""
        report = self.analyze()

        bin_stats_data = [
            {
                "bin_idx": bs.bin_idx,
                "theta_range": bs.theta_range,
                "sample_count": bs.sample_count,
                "rr_median": bs.rr_median,
                "rl_median": bs.rl_median,
                "rr_cv": bs.rr_cv,
                "rl_cv": bs.rl_cv,
                "is_sparse": bs.is_sparse,
            }
            for bs in report.bin_statistics
        ]

        data = {
            "quality_score": report.quality_score,
            "bin_count": self.bin_count,
            "sample_count": len(self.th_list),
            "bin_statistics": bin_stats_data,
            "od_delta_analysis": report.od_delta_analysis,
            "warnings": report.warnings,
            "recommendation": report.recommendation,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"诊断数据已导出到: {output_file}")


# 示例使用
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # 读取raw_points JSON文件
    json_file = Path(sys.argv[1])
    if not json_file.exists():
        print(f"错误: 文件不存在 {json_file}")
        sys.exit(1)

    with open(json_file, "r", encoding="utf-8") as f:
        raw_points = json.load(f)

    # 运行诊断
    print(f"分析{len(raw_points)}个采样点...")
    diagnostics = OdAlgorithmDiagnostics(raw_points, bin_count=90, bin_method="median")
    report = diagnostics.analyze()
    diagnostics.print_report(report)

    # 导出诊断数据
    if "--export" in sys.argv:
        output_json = json_file.parent / (json_file.stem + "_diagnostics.json")
        diagnostics.export_diagnostics_json(output_json)
