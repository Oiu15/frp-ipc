from __future__ import annotations

"""几何拟合与补偿纯算法(geometry_v2)。

来源:`docs/frp_metrology_core.py` 骨架的"基础几何 / 圆拟合 / 支撑函数重建 /
装夹斜率 τ 纠正 / 主轴误差分离"部分。本模块只放纯几何/拟合,仅依赖
numpy / scipy,不依赖 UI / driver / AppHost / repositories,可单测。

与 `domain/geometry_calibration.py` 的分工:
  - 本模块:与传感器无关的纯几何(旋转、圆拟合、Fourier 圆度、支撑函数重建、
    τ 椭圆假象纠正、直线度、主轴多步分离),不 import 任何 domain 兄弟模块。
  - geometry_calibration:传感器正/逆向模型(OD 支撑、ID 线-圆)、工装位姿、
    联合 LM 标定、互配准/同心度,import 本模块。

关键几何约定(与设计文档一致):
  * 机床系 2D:原点 O = 回转轴 ∩ 截面平面;编码器角 θ;旋转 R(θ)。
  * 工件固连点 p 在角度 θ 时位于 R(θ) p;去旋转 = 左乘 R(-θ)。
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:  # pragma: no cover - scipy 始终在 requirements 中,这里仅做无 scipy 退回
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover
    least_squares = None


# =============================================================================
# 0. 基础几何
# =============================================================================

def rot(theta: float) -> np.ndarray:
    """2x2 旋转矩阵 R(theta)。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def derotate(points: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """把机床系边界点按各自编码器角去旋转到工件系:p_part = R(-θ) p_machine。

    points: (N,2),thetas: (N,)。返回 (N,2)。
    """
    points = np.asarray(points, float)
    thetas = np.asarray(thetas, float)
    c, s = np.cos(thetas), np.sin(thetas)
    x, y = points[:, 0], points[:, 1]
    # R(-θ) = [[c, s], [-s, c]]
    xp = c * x + s * y
    yp = -s * x + c * y
    return np.column_stack([xp, yp])


# =============================================================================
# 1. 圆拟合 + Fourier 圆度
# =============================================================================

@dataclass(slots=True)
class CircleFit:
    cx: float
    cy: float
    r: float
    residual_rms: float  # 径向残差 RMS


def fit_circle_kasa(x: np.ndarray, y: np.ndarray) -> CircleFit:
    """Kåsa 代数圆拟合(闭式,作为初值)。最小化 Σ(x²+y²+Dx+Ey+F)²。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    D, E, F = float(sol[0]), float(sol[1]), float(sol[2])
    cx, cy = -D / 2.0, -E / 2.0
    r = float(np.sqrt(max(cx ** 2 + cy ** 2 - F, 0.0)))
    rr = np.hypot(x - cx, y - cy)
    return CircleFit(float(cx), float(cy), r, float(np.sqrt(np.mean((rr - r) ** 2))))


def fit_circle_geometric(x: np.ndarray, y: np.ndarray,
                         init: Optional[CircleFit] = None) -> CircleFit:
    """几何圆拟合(最小化径向残差,LM 精修)。无 scipy 时退回 Kåsa。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if init is None:
        init = fit_circle_kasa(x, y)
    if least_squares is None:  # pragma: no cover
        return init

    def resid(p: np.ndarray) -> np.ndarray:
        cx, cy, r = p
        return np.hypot(x - cx, y - cy) - r

    sol = least_squares(resid, [init.cx, init.cy, init.r], method="lm")
    cx, cy, r = float(sol.x[0]), float(sol.x[1]), float(sol.x[2])
    rr = np.hypot(x - cx, y - cy)
    return CircleFit(cx, cy, r, float(np.sqrt(np.mean((rr - r) ** 2))))


@dataclass(slots=True)
class Roundness:
    r_mean: float
    roundness_lsc: float                 # 最小二乘圆基准下 max-min 径向偏差
    upr: dict[int, float]                # 各阶谐波幅值(undulations per rev)
    phi: np.ndarray = field(repr=False)  # 角度
    dr: np.ndarray = field(repr=False)   # 径向偏差曲线


def roundness_from_points(x: np.ndarray, y: np.ndarray,
                          cx: float, cy: float,
                          max_harmonic: int = 15) -> Roundness:
    """以 LSC(最小二乘圆心)为基准的圆度与谐波分解。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    phi = np.arctan2(y - cy, x - cx)
    rr = np.hypot(x - cx, y - cy)
    r_mean = float(np.mean(rr))
    dr = rr - r_mean
    upr: dict[int, float] = {}
    for k in range(1, max_harmonic + 1):
        ak = 2.0 * np.mean(dr * np.cos(k * phi))
        bk = 2.0 * np.mean(dr * np.sin(k * phi))
        upr[k] = float(np.hypot(ak, bk))
    return Roundness(r_mean, float(dr.max() - dr.min()), upr, phi, dr)


# =============================================================================
# 2. OD:投影式测径仪支撑函数 → 边界点
# =============================================================================

def support_to_boundary(theta: np.ndarray, h: np.ndarray,
                        n_grid: int = 720) -> np.ndarray:
    """由整圈支撑函数读数重建工件系边界点。

    测径仪测量轴方向 u 固定;工件转过 θ 时,机床方向 u 对应工件系方向 α=-θ,
    故 h_part(α=-θ) = 读数(θ)。重建公式(凸/近圆):
        x = h cosα - h' sinα,  y = h sinα + h' cosα
    h' 用 FFT 在均匀 α 栅格上求导。

    theta: (N,) 整圈编码器角(rad);h: (N,) 已标定的物理支撑距离。
    返回工件系边界点 (n_grid,2)。
    """
    theta = np.asarray(theta, float)
    h = np.asarray(h, float)
    alpha = np.mod(-theta, 2 * np.pi)
    order = np.argsort(alpha)
    a_s, h_s = alpha[order], h[order]
    # 周期插值到均匀栅格
    grid = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    a_ext = np.concatenate([a_s, a_s[:1] + 2 * np.pi])
    h_ext = np.concatenate([h_s, h_s[:1]])
    hg = np.interp(grid, a_ext, h_ext)
    # FFT 求导
    H = np.fft.fft(hg)
    kfreq = np.fft.fftfreq(n_grid, d=(grid[1] - grid[0])) * 2 * np.pi
    hp = np.real(np.fft.ifft(1j * kfreq * H))
    x = hg * np.cos(grid) - hp * np.sin(grid)
    y = hg * np.sin(grid) + hp * np.cos(grid)
    return np.column_stack([x, y])


# =============================================================================
# 3. 装夹斜率 τ:中心线斜率 + 椭圆假象纠正
# =============================================================================

def centerline_tilt(centers: np.ndarray, z: np.ndarray,
                    axis_straightness: Optional[np.ndarray] = None) -> np.ndarray:
    """从各截面圆心随 z 的斜率提取装夹斜率 τ=(τx,τy)(rad,小角近似)。

    centers:(M,2)  z:(M,)。同时是直线度所用数据,τ 免费得到。
    """
    centers = np.asarray(centers, float)
    z = np.asarray(z, float)
    if axis_straightness is not None:
        centers = centers - np.asarray(axis_straightness, float)
    A = np.column_stack([z, np.ones_like(z)])
    tau = np.empty(2)
    for j in range(2):
        sol = np.linalg.lstsq(A, centers[:, j], rcond=None)[0]
        tau[j] = float(sol[0])  # d(offset)/dz ≈ tan τ ≈ τ
    return tau


def correct_clamping_tilt(phi: np.ndarray, dr: np.ndarray, r_mean: float,
                          tau: np.ndarray) -> tuple[np.ndarray, float]:
    """扣除装夹斜率 τ 产生的椭圆假象(2 次谐波 + 直径偏置)。

    椭圆假象(工件系,方位 ψ): r_mean·(τ²/4)·[1 + cos 2(ψ-ψτ)]。
    用独立测得的 τ 预测并只减这一份;残余 2 次谐波为工件真实椭圆度。

    返回 (dr_corr, r_true)。dr 为相对 r_mean 的径向偏差。
    """
    tau = np.asarray(tau, float)
    tmag = float(np.hypot(tau[0], tau[1]))
    psi_t = float(np.arctan2(tau[1], tau[0]))
    amp = r_mean * tmag ** 2 / 4.0
    artifact = amp * (1.0 + np.cos(2.0 * (phi - psi_t)))
    dr_corr = np.asarray(dr, float) - artifact
    r_true = r_mean - amp  # 平均半径偏置 +Rτ²/4 → 扣回
    return dr_corr, r_true


def roundness_corrected_for_tilt(x: np.ndarray, y: np.ndarray,
                                 cx: float, cy: float, tau: np.ndarray,
                                 max_harmonic: int = 15) -> Roundness:
    """先按 LSC 求径向偏差,再扣除 τ 椭圆假象,得真实圆度与谐波。"""
    base = roundness_from_points(x, y, cx, cy, max_harmonic)
    dr_corr, r_true = correct_clamping_tilt(base.phi, base.dr, base.r_mean, tau)
    upr: dict[int, float] = {}
    for k in range(1, max_harmonic + 1):
        ak = 2.0 * np.mean(dr_corr * np.cos(k * base.phi))
        bk = 2.0 * np.mean(dr_corr * np.sin(k * base.phi))
        upr[k] = float(np.hypot(ak, bk))
    return Roundness(r_true, float(dr_corr.max() - dr_corr.min()),
                     upr, base.phi, dr_corr)


def straightness(centers: np.ndarray, z: np.ndarray,
                 axis_straightness: Optional[np.ndarray] = None) -> float:
    """整管直线度:各截面圆心(去回转轴直线度后)拟合直线的最大偏移。

    注:装夹斜率 τ 是直线的整体倾斜,被最佳拟合直线吸收,不计入直线度。

    centers: (M,2);z: (M,);axis_straightness: (M,2) 回转轴直线度基准(可选)。
    """
    centers = np.asarray(centers, float)
    z = np.asarray(z, float)
    if axis_straightness is not None:
        centers = centers - np.asarray(axis_straightness, float)
    devs = []
    for j in range(2):  # x、y 两个方向各拟合直线
        A = np.column_stack([z, np.ones_like(z)])
        sol = np.linalg.lstsq(A, centers[:, j], rcond=None)[0]
        k, b = float(sol[0]), float(sol[1])
        devs.append(centers[:, j] - (k * z + b))
    devs = np.array(devs)
    radial = np.hypot(devs[0], devs[1])
    return float(radial.max())


# =============================================================================
# 4. 主轴误差多步分离(重夹索引法,简版;本机暂不入主链,留作将来)
# =============================================================================

def separate_spindle_multistep(profiles: list[np.ndarray] | tuple[np.ndarray, ...],
                               index_angles: list[float] | tuple[float, ...],
                               n_grid: int = 360) -> tuple[np.ndarray, np.ndarray]:
    """多次重夹(已知索引角)分离"工件形状(工件系固定)"与"主轴误差(机床系固定)"。

    profiles[m]: 第 m 次装夹、在【机床系角度栅格】上的径向偏差 dr(机床角)。
    index_angles[m]: 该次装夹工件相对主轴的索引角(rad)。

    返回 (part_form(工件系), spindle_err(机床系)),均在 n_grid 栅格上。
    原理:工件形状随索引角在机床系平移;主轴误差不随索引角变。
    """
    grid = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    rows: list[np.ndarray] = []
    for dr in profiles:
        dr = np.asarray(dr, float)
        ang = np.linspace(0, 2 * np.pi, len(dr), endpoint=False)
        rows.append(np.interp(grid, ang, dr, period=2 * np.pi))
    P = np.array(rows)
    idxs = np.asarray(index_angles, float)

    # 工件形状:把各次按 -索引角对齐到工件系后平均(主轴误差被抹匀)
    part_aligned = []
    for prof, idx in zip(P, idxs):
        shift = int(round(idx / (2 * np.pi) * n_grid))
        part_aligned.append(np.roll(prof, -shift))
    part_form = np.mean(part_aligned, axis=0)

    # 主轴误差:从各次扣掉旋回机床系的工件形状后平均
    spindle = []
    for prof, idx in zip(P, idxs):
        shift = int(round(idx / (2 * np.pi) * n_grid))
        spindle.append(prof - np.roll(part_form, shift))
    spindle_err = np.mean(spindle, axis=0)
    return part_form, spindle_err


__all__ = [
    "CircleFit",
    "Roundness",
    "centerline_tilt",
    "correct_clamping_tilt",
    "derotate",
    "fit_circle_geometric",
    "fit_circle_kasa",
    "rot",
    "roundness_corrected_for_tilt",
    "roundness_from_points",
    "separate_spindle_multistep",
    "straightness",
    "support_to_boundary",
]
