"""
frp_metrology_core.py — FRP 管检测：标定与补偿核心纯算法骨架

设计依据：见《标定流程与补偿算法设计.md》。本文件只放纯算法，
仅依赖 numpy / scipy，不依赖 UI / driver / AppHost / Tk，可单测。

PROJECT_OVERVIEW 落点建议：
  - 圆拟合 / Fourier 圆度 / 支撑函数重建 / 去旋转 / 主轴分离
        -> domain/summaries.py（或新增 domain/geometry_fit.py）
  - 传感器正/逆向模型（OD 支撑函数、ID 线-圆）、联合 LM 标定、互配准
        -> domain/calibration.py
  - 生产补偿管线（截面 + 整管）
        -> frp_workflow/row_math.py 调用上面纯函数

关键几何约定（与设计文档一致）：
  * 机床系 2D：原点 O = 回转轴 ∩ 截面平面；编码器角 θ；旋转 R(θ)。
  * 工件固连点 p 在角度 θ 时位于 R(θ) p；去旋转 = 左乘 R(-θ)。
  * OD 投影式测径仪给“支撑函数”h(α)（卡尺值），不是半径 → 先重建边界点。
  * ID 位移计给真边界点：hit = f + (k·L)·n。
  * 偏心(约束3)、探头不过心(约束4)由“去旋转 + 拟合圆”自然吸收。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:
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
    """把机床系边界点按各自编码器角去旋转到工件系：p_part = R(-θ) p_machine。

    points: (N,2)，thetas: (N,)。返回 (N,2)。
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

@dataclass
class CircleFit:
    cx: float
    cy: float
    r: float
    residual_rms: float  # 径向残差 RMS


def fit_circle_kasa(x: np.ndarray, y: np.ndarray) -> CircleFit:
    """Kåsa 代数圆拟合（闭式，作为初值）。最小化 Σ(x²+y²+Dx+Ey+F)²。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = -D / 2.0, -E / 2.0
    r = np.sqrt(max(cx ** 2 + cy ** 2 - F, 0.0))
    rr = np.hypot(x - cx, y - cy)
    return CircleFit(cx, cy, r, float(np.sqrt(np.mean((rr - r) ** 2))))


def fit_circle_geometric(x: np.ndarray, y: np.ndarray,
                         init: Optional[CircleFit] = None) -> CircleFit:
    """几何圆拟合（最小化径向残差，LM 精修）。无 scipy 时退回 Kåsa。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if init is None:
        init = fit_circle_kasa(x, y)
    if least_squares is None:
        return init

    def resid(p):
        cx, cy, r = p
        return np.hypot(x - cx, y - cy) - r

    sol = least_squares(resid, [init.cx, init.cy, init.r], method="lm")
    cx, cy, r = sol.x
    rr = np.hypot(x - cx, y - cy)
    return CircleFit(float(cx), float(cy), float(r),
                     float(np.sqrt(np.mean((rr - r) ** 2))))


@dataclass
class Roundness:
    r_mean: float
    roundness_lsc: float                 # 最小二乘圆基准下 max-min 径向偏差
    upr: dict[int, float]                # 各阶谐波幅值（undulations per rev）
    phi: np.ndarray = field(repr=False)  # 角度
    dr: np.ndarray = field(repr=False)   # 径向偏差曲线


def roundness_from_points(x: np.ndarray, y: np.ndarray,
                          cx: float, cy: float,
                          max_harmonic: int = 15) -> Roundness:
    """以 LSC（最小二乘圆心）为基准的圆度与谐波分解。"""
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
# 2. OD：投影式测径仪（支撑函数 → 边界点）
# =============================================================================

def support_to_boundary(theta: np.ndarray, h: np.ndarray,
                        n_grid: int = 720) -> np.ndarray:
    """由整圈支撑函数读数重建工件系边界点。

    测径仪测量轴方向 u 固定；工件转过 θ 时，机床方向 u 对应工件系方向 α=-θ，
    故 h_part(α=-θ) = 读数(θ)。重建公式（凸/近圆）：
        x = h cosα - h' sinα,  y = h sinα + h' cosα
    h' 用 FFT 在均匀 α 栅格上求导。

    theta: (N,) 整圈编码器角(rad)；h: (N,) 已标定的物理支撑距离。
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


def od_calibrate_scale(known_diam: Sequence[float],
                       raw_width: Sequence[float],
                       positions: Optional[Sequence[float]] = None):
    """OD 标度/零位标定。两点以上线性拟合：diam = k·raw + b。

    若给 positions（柱规在量隙的横向位置），追加远心度项 diam = (k0 + k1·x)·raw + b。
    返回 dict。
    """
    known = np.asarray(known_diam, float)
    raw = np.asarray(raw_width, float)
    if positions is None:
        A = np.column_stack([raw, np.ones_like(raw)])
        k, b = np.linalg.lstsq(A, known, rcond=None)[0]
        return {"k0": float(k), "k1": 0.0, "b": float(b)}
    pos = np.asarray(positions, float)
    A = np.column_stack([raw, raw * pos, np.ones_like(raw)])
    k0, k1, b = np.linalg.lstsq(A, known, rcond=None)[0]
    return {"k0": float(k0), "k1": float(k1), "b": float(b)}


def od_apply_scale(raw_width: np.ndarray, cal: dict,
                   position: float = 0.0) -> np.ndarray:
    """把 OD 原始宽度/支撑读数转成物理量（含远心度修正）。"""
    raw = np.asarray(raw_width, float)
    return (cal["k0"] + cal["k1"] * position) * raw + cal["b"]


# =============================================================================
# 3. ID：背向位移计（线-圆 正/逆向模型）
# =============================================================================

@dataclass
class ProbePose:
    """单个位移计在机床系(2D)的位姿。hit = f + (k·L·cosγ)·n。"""
    fx: float
    fy: float
    nx: float
    ny: float
    k: float = 1.0          # 标度
    gamma_deg: float = 0.0  # 出平面倾角 → 弦过读，作为标度修正

    @property
    def f(self) -> np.ndarray:
        return np.array([self.fx, self.fy])

    @property
    def n(self) -> np.ndarray:
        v = np.array([self.nx, self.ny])
        return v / np.linalg.norm(v)


@dataclass
class IdTooling:
    probe_a: ProbePose
    probe_b: ProbePose


def id_tooling_from_simple(D: float, s: float = 0.0,
                           axis_deg: float = 0.0,
                           q: Sequence[float] = (0.0, 0.0),
                           k_a: float = 1.0, k_b: float = 1.0,
                           gamma_deg: float = 0.0) -> IdTooling:
    """由“理想背向共线 + 工装偏差”构造位姿。

    D     : 两感测面间距（背向，朝外）
    s     : 两探头线横向偏移（不共线，约束2）
    axis_deg: 探头线方位角
    q     : 探头线相对 O 的横向位置（约束4，随截面不同）
    """
    ax = np.deg2rad(axis_deg)
    n = np.array([np.cos(ax), np.sin(ax)])       # B 朝 +n
    perp = np.array([-np.sin(ax), np.cos(ax)])   # 横向
    q = np.asarray(q, float)
    # 背向：A 面在 q-(D/2)n 朝 -n；B 面在 q+(D/2)n 朝 +n；B 线横向偏移 s
    fa = q - (D / 2.0) * n
    fb = q + (D / 2.0) * n + s * perp
    a = ProbePose(*fa, *(-n), k=k_a, gamma_deg=gamma_deg)
    b = ProbePose(*fb, *(n), k=k_b, gamma_deg=gamma_deg)
    return IdTooling(a, b)


def _line_circle_outward_t(f: np.ndarray, n: np.ndarray,
                           m: np.ndarray, r: float) -> float:
    """从 f 沿 +n 到半径 r、圆心 m 的圆的“朝外壁”交点距离 t（取较大正根）。"""
    d = f - m
    B = float(n @ d)
    C = float(d @ d - r * r)
    disc = B * B - C
    if disc < 0:
        return np.nan
    return -B + np.sqrt(disc)


def id_predict_L(pose: ProbePose, e: np.ndarray, theta: float, r: float) -> float:
    """正向模型：给定内圆半径 r、θ=0 时内圆心 e（机床系），预测该探头读数 L。

    内圆心随旋转 m(θ)=R(θ)e 绕 O 公转（含约束3偏心）。
    """
    m = rot(theta) @ np.asarray(e, float)
    t = _line_circle_outward_t(pose.f, pose.n, m, r)
    cg = np.cos(np.deg2rad(pose.gamma_deg))
    return t / (pose.k * cg)


def id_hit_point(pose: ProbePose, L: float) -> np.ndarray:
    """逆向：把读数 L 还原成机床系壁面命中点。"""
    cg = np.cos(np.deg2rad(pose.gamma_deg))
    return pose.f + (pose.k * L * cg) * pose.n


def id_points_from_readings(tooling: IdTooling,
                            theta: np.ndarray,
                            L1: np.ndarray, L2: np.ndarray) -> np.ndarray:
    """整圈 (θ,L1,L2) → 去旋转后的工件系内壁点云 (2N,2)。"""
    theta = np.asarray(theta, float)
    L1 = np.asarray(L1, float)
    L2 = np.asarray(L2, float)
    ha = np.array([id_hit_point(tooling.probe_a, l) for l in L1])
    hb = np.array([id_hit_point(tooling.probe_b, l) for l in L2])
    pa = derotate(ha, theta)
    pb = derotate(hb, theta)
    return np.vstack([pa, pb])


# =============================================================================
# 4. 联合 LM 标定（ID 位姿：s / 方位 / q / 标度，约束 2&4）
# =============================================================================

@dataclass
class IdCalDataset:
    """一次装夹的参考件旋转数据。"""
    theta: np.ndarray
    L1: np.ndarray
    L2: np.ndarray


@dataclass
class IdCalResult:
    tooling: IdTooling
    eccentricities: list[np.ndarray]  # 每次装夹拟合出的 e
    cost: float
    success: bool


def calibrate_id_tooling(datasets: Sequence[IdCalDataset],
                         r_known: float,
                         D_init: float,
                         init_s: float = 0.0,
                         init_axis_deg: float = 0.0,
                         fit_axis: bool = True,
                         fit_scale: bool = False) -> IdCalResult:
    """以参考件已知内半径 r_known 为约束，联合拟合共享工装参数 + 每次装夹偏心。

    共享参数: [s, axis_deg, qx, qy, (k_a, k_b)]；每装夹 nuisance: e=(ex,ey)。
    残差: 各采样的 (预测 L - 实测 L)。多次装夹(不同 e)改善 s 与 q 的可分性。

    注意: 绝对 q（探头线相对 O）由“去旋转后点必须落在以 r_known 为半径的圆上”锁定——
    错误的 q 会让去旋转点云呈外摆线状、残差上升。
    """
    if least_squares is None:
        raise RuntimeError("需要 scipy.optimize.least_squares")

    nset = len(datasets)
    nshared = 4 + (2 if fit_scale else 0)

    def unpack(p):
        s, axis_deg, qx, qy = p[0], p[1], p[2], p[3]
        if fit_scale:
            k_a, k_b = p[4], p[5]
        else:
            k_a = k_b = 1.0
        es = [np.array([p[nshared + 2 * i], p[nshared + 2 * i + 1]])
              for i in range(nset)]
        return s, axis_deg, qx, qy, k_a, k_b, es

    def residuals(p):
        s, axis_deg, qx, qy, k_a, k_b, es = unpack(p)
        tl = id_tooling_from_simple(D_init, s=s, axis_deg=axis_deg,
                                    q=(qx, qy), k_a=k_a, k_b=k_b)
        out = []
        for ds, e in zip(datasets, es):
            for th, l in zip(ds.theta, ds.L1):
                out.append(id_predict_L(tl.probe_a, e, th, r_known) - l)
            for th, l in zip(ds.theta, ds.L2):
                out.append(id_predict_L(tl.probe_b, e, th, r_known) - l)
        return np.array(out)

    p0 = [init_s, init_axis_deg, 0.0, 0.0]
    if fit_scale:
        p0 += [1.0, 1.0]
    for _ in range(nset):
        p0 += [0.0, 0.0]

    sol = least_squares(residuals, p0, method="lm")
    s, axis_deg, qx, qy, k_a, k_b, es = unpack(sol.x)
    tooling = id_tooling_from_simple(D_init, s=s, axis_deg=axis_deg,
                                     q=(qx, qy), k_a=k_a, k_b=k_b)
    return IdCalResult(tooling, es, float(sol.cost), bool(sol.success))


# =============================================================================
# 5. 互配准 + 同心度（含预留 ref_coaxiality_observed）
# =============================================================================

@dataclass
class CrossReg:
    delta_reg: np.ndarray                      # OD↔ID 传感器零位横向偏移
    ref_coaxiality_observed: Optional[np.ndarray] = None  # 预留：未知=None
    ref_coaxiality_uncertainty: float = 0.05   # mm；未知时整体计入不确定度

    def coax(self) -> np.ndarray:
        if self.ref_coaxiality_observed is None:
            return np.zeros(2)
        return np.asarray(self.ref_coaxiality_observed, float)


def concentricity(c_o: np.ndarray, c_i: np.ndarray, reg: CrossReg) -> float:
    """同心度 = |(c_o - c_i) - Δ_reg - ref_coaxiality_observed|（设计 §6）。"""
    v = np.asarray(c_o, float) - np.asarray(c_i, float) - reg.delta_reg - reg.coax()
    return float(np.linalg.norm(v))


def concentricity_uncertainty(fit_noise: float, reg: CrossReg) -> float:
    """同心度不确定度：拟合噪声 ⊕（同轴度未知时其上界）。"""
    extra = 0.0 if reg.ref_coaxiality_observed is not None \
        else reg.ref_coaxiality_uncertainty
    return float(np.hypot(fit_noise, extra))


# =============================================================================
# 6. 主轴误差多步分离（重夹索引法，简版）
# =============================================================================

def separate_spindle_multistep(profiles: Sequence[np.ndarray],
                               index_angles: Sequence[float],
                               n_grid: int = 360):
    """多次重夹（已知索引角）分离“工件形状(工件系固定)”与“主轴误差(机床系固定)”。

    profiles[m]: 第 m 次装夹、在【机床系角度栅格】上的径向偏差 dr(机床角)。
    index_angles[m]: 该次装夹工件相对主轴的索引角(rad)。

    返回 (part_form(工件系), spindle_err(机床系))，均在 n_grid 栅格上。
    原理：工件形状随索引角在机床系平移；主轴误差不随索引角变。
    """
    grid = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    P = []
    for dr, idx in zip(profiles, index_angles):
        dr = np.asarray(dr, float)
        ang = np.linspace(0, 2 * np.pi, len(dr), endpoint=False)
        P.append(np.interp(grid, ang, dr, period=2 * np.pi))
    P = np.array(P)
    idxs = np.asarray(index_angles, float)

    # 工件形状：把各次按 -索引角对齐到工件系后平均（主轴误差被抹匀）
    part_aligned = []
    for prof, idx in zip(P, idxs):
        shift = int(round(idx / (2 * np.pi) * n_grid))
        part_aligned.append(np.roll(prof, -shift))
    part_form = np.mean(part_aligned, axis=0)

    # 主轴误差：从各次扣掉旋回机床系的工件形状后平均
    spindle = []
    for prof, idx in zip(P, idxs):
        shift = int(round(idx / (2 * np.pi) * n_grid))
        spindle.append(prof - np.roll(part_form, shift))
    spindle_err = np.mean(spindle, axis=0)
    return part_form, spindle_err


# =============================================================================
# 7. 生产补偿管线（截面 + 整管）
# =============================================================================

@dataclass
class SectionResult:
    z: float
    od_diameter: float
    od_center: np.ndarray
    od_roundness: float
    id_diameter: float
    id_center: np.ndarray
    id_roundness: float
    concentricity: float


def compensate_section(z: float,
                       theta: np.ndarray,
                       od_support_raw: np.ndarray,
                       id_L1: np.ndarray, id_L2: np.ndarray,
                       od_cal: dict,
                       id_tooling: IdTooling,
                       reg: CrossReg,
                       od_position: float = 0.0) -> SectionResult:
    """单截面补偿：原始读数 → 直径/圆心/圆度/同心度。"""
    # --- OD：标定 → 支撑函数 → 边界点 → 圆拟合 ---
    h = od_apply_scale(od_support_raw, od_cal, position=od_position) / 1.0
    od_pts = support_to_boundary(theta, h)
    cf_o = fit_circle_geometric(od_pts[:, 0], od_pts[:, 1])
    rnd_o = roundness_from_points(od_pts[:, 0], od_pts[:, 1], cf_o.cx, cf_o.cy)

    # --- ID：读数 → 命中点 → 去旋转 → 圆拟合 ---
    id_pts = id_points_from_readings(id_tooling, theta, id_L1, id_L2)
    cf_i = fit_circle_geometric(id_pts[:, 0], id_pts[:, 1])
    rnd_i = roundness_from_points(id_pts[:, 0], id_pts[:, 1], cf_i.cx, cf_i.cy)

    c_o = np.array([cf_o.cx, cf_o.cy])
    c_i = np.array([cf_i.cx, cf_i.cy])
    conc = concentricity(c_o, c_i, reg)
    return SectionResult(z, 2 * cf_o.r, c_o, rnd_o.roundness_lsc,
                         2 * cf_i.r, c_i, rnd_i.roundness_lsc, conc)


def centerline_tilt(centers: np.ndarray, z: np.ndarray,
                    axis_straightness: Optional[np.ndarray] = None) -> np.ndarray:
    """从各截面圆心随 z 的斜率提取装夹斜率 τ=(τx,τy)（rad，小角近似）。

    centers:(M,2)  z:(M,)。同时是直线度所用数据，τ 免费得到。
    """
    centers = np.asarray(centers, float)
    z = np.asarray(z, float)
    if axis_straightness is not None:
        centers = centers - np.asarray(axis_straightness, float)
    A = np.column_stack([z, np.ones_like(z)])
    tau = np.empty(2)
    for j in range(2):
        k, _ = np.linalg.lstsq(A, centers[:, j], rcond=None)[0]
        tau[j] = k  # d(offset)/dz ≈ tan τ ≈ τ
    return tau


def correct_clamping_tilt(phi: np.ndarray, dr: np.ndarray, r_mean: float,
                          tau: np.ndarray):
    """扣除装夹斜率 τ 产生的椭圆假象（2 次谐波 + 直径偏置）。

    椭圆假象（工件系，方位 ψ）: r_mean·(τ²/4)·[1 + cos 2(ψ-ψτ)]。
    用独立测得的 τ 预测并只减这一份；残余 2 次谐波为工件真实椭圆度。

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
    """先按 LSC 求径向偏差，再扣除 τ 椭圆假象，得真实圆度与谐波。"""
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
    """整管直线度：各截面圆心(去回转轴直线度后)拟合直线的最大偏移。

    注：装夹斜率 τ 是直线的整体倾斜，被最佳拟合直线吸收，不计入直线度。

    centers: (M,2)；z: (M,)；axis_straightness: (M,2) 回转轴直线度基准(可选)。
    """
    centers = np.asarray(centers, float)
    z = np.asarray(z, float)
    if axis_straightness is not None:
        centers = centers - np.asarray(axis_straightness, float)
    devs = []
    for j in range(2):  # x、y 两个方向各拟合直线
        A = np.column_stack([z, np.ones_like(z)])
        k, b = np.linalg.lstsq(A, centers[:, j], rcond=None)[0]
        devs.append(centers[:, j] - (k * z + b))
    devs = np.array(devs)
    radial = np.hypot(devs[0], devs[1])
    return float(radial.max())


# =============================================================================
# 8. 自检（合成数据验证恢复精度）
# =============================================================================

def _make_inner_circle_readings(tooling, r, e, lobes=None, n=360, noise=0.0,
                                rng=None):
    """合成 ID 读数：内圆半径 r、θ=0 圆心 e，可加多瓣形状误差与噪声。"""
    rng = rng or np.random.default_rng(0)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    L1 = np.empty(n)
    L2 = np.empty(n)
    for i, th in enumerate(theta):
        rr = r
        if lobes:
            for k, amp in lobes.items():
                rr += amp * np.cos(k * th)  # 工件系形状随 θ 一起转
        L1[i] = id_predict_L(tooling.probe_a, e, th, rr)
        L2[i] = id_predict_L(tooling.probe_b, e, th, rr)
    if noise:
        L1 = L1 + rng.normal(0, noise, n)
        L2 = L2 + rng.normal(0, noise, n)
    return theta, L1, L2


def _make_od_support(R, center=(0, 0), lobes=None, n=720):
    """合成 OD 支撑函数读数：用真实边界点取各方向最大投影（含奇数瓣）。"""
    cx, cy = center
    phi = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    rr = np.full_like(phi, float(R))
    if lobes:
        for k, amp in lobes.items():
            rr += amp * np.cos(k * phi)
    bx = cx + rr * np.cos(phi)
    by = cy + rr * np.sin(phi)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    h = np.empty(n)
    for i, th in enumerate(theta):
        alpha = -th  # 机床方向 u 对应工件系方向 α=-θ
        u = np.array([np.cos(alpha), np.sin(alpha)])
        h[i] = np.max(bx * u[0] + by * u[1])
    return theta, h


def _selftest():
    rng = np.random.default_rng(42)
    print("=" * 64)
    print("自检 1 — ID 生产补偿（已知工装，含偏心+不共线+3瓣，恢复直径/圆度）")
    true_tool = id_tooling_from_simple(D=140.0, s=0.3, axis_deg=12.0, q=(2.0, -1.5))
    r_true = 76.35  # 6" 内半径 (ID 152.7)
    e_true = np.array([1.2, -0.8])           # 偏心(约束3)
    lobes = {3: 0.012, 2: 0.006}             # 真圆度形状（含奇瓣 3）
    theta, L1, L2 = _make_inner_circle_readings(
        true_tool, r_true, e_true, lobes=lobes, n=360, noise=0.0005, rng=rng)
    pts = id_points_from_readings(true_tool, theta, L1, L2)
    cf = fit_circle_geometric(pts[:, 0], pts[:, 1])
    rnd = roundness_from_points(pts[:, 0], pts[:, 1], cf.cx, cf.cy)
    print(f"  内径恢复 {2*cf.r:.4f} mm (真 {2*r_true:.4f}), 误差 {2*(cf.r-r_true)*1000:+.2f} µm")
    print(f"  圆心恢复 ({cf.cx:+.4f},{cf.cy:+.4f}) (真偏心 {e_true})")
    print(f"  圆度 LSC {rnd.roundness_lsc*1000:.2f} µm (真峰峰约36µm), 拟合RMS {cf.residual_rms*1000:.2f} µm")
    print(f"  3阶谐波 {rnd.upr[3]*1000:.2f} µm（>0，对射宽度法恒为0；注：ID两探头只扫到内圆"
          f"部分弧段，\n          逐阶谐波归属偏弱，可靠的是总圆度与直径/圆心；逐阶圆度建议以OD为准）")
    assert abs(2 * cf.r - 2 * r_true) < 0.005, "内径恢复超差"

    print("=" * 64)
    print("自检 2 — OD 支撑函数重建（含3瓣奇数瓣，验证不被卡尺盲区抹掉）")
    R_true = 95.0  # OD 190 外半径
    od_lobes = {3: 0.010, 2: 0.005}
    th_o, h_o = _make_od_support(R_true, center=(0.9, -0.6), lobes=od_lobes, n=720)
    bnd = support_to_boundary(th_o, h_o)
    cfo = fit_circle_geometric(bnd[:, 0], bnd[:, 1])
    rndo = roundness_from_points(bnd[:, 0], bnd[:, 1], cfo.cx, cfo.cy)
    print(f"  外半径恢复 {cfo.r:.4f} mm (真 {R_true:.4f}), 误差 {(cfo.r-R_true)*1000:+.2f} µm")
    print(f"  圆心恢复 ({cfo.cx:+.4f},{cfo.cy:+.4f}) (真 0.9,-0.6)")
    print(f"  圆度 LSC {rndo.roundness_lsc*1000:.2f} µm；3阶谐波 {rndo.upr[3]*1000:.2f} µm（奇瓣被测到 ✓）")
    assert abs(cfo.r - R_true) < 0.01, "外径恢复超差"
    assert rndo.upr[3] * 1000 > 5, "未测到3阶奇瓣"

    print("=" * 64)
    print("自检 3 — ID 联合 LM 位姿标定（未知 s/q，3次装夹联合恢复）")
    truth = id_tooling_from_simple(D=140.0, s=0.35, axis_deg=8.0, q=(1.5, 0.7))
    ecc = [np.array([1.0, 0.5]), np.array([-1.3, 0.9]), np.array([0.4, -1.6])]
    datasets = []
    for e in ecc:
        th, l1, l2 = _make_inner_circle_readings(truth, 76.35, e, n=240,
                                                 noise=0.0003, rng=rng)
        datasets.append(IdCalDataset(th, l1, l2))
    res = calibrate_id_tooling(datasets, r_known=76.35, D_init=140.0,
                               init_s=0.0, init_axis_deg=0.0)
    # 反推恢复的 s：A、B 线横向间距
    fa, fb = res.tooling.probe_a.f, res.tooling.probe_b.f
    n = res.tooling.probe_b.n
    perp = np.array([-n[1], n[0]])
    s_rec = float((fb - fa) @ perp)
    print(f"  success={res.success}, cost={res.cost:.2e}")
    q_rec = (res.tooling.probe_a.f + res.tooling.probe_b.f) / 2
    print(f"  恢复 s = {s_rec:+.4f} mm (真 0.35)")
    print(f"  恢复 q ≈ ({q_rec[0]:+.3f}, {q_rec[1]:+.3f}) (真 1.5, 0.7；注:q与每装夹偏心部分耦合，多装夹/已知偏心夹具可改善)")
    print(f"  恢复偏心 e[0] = {res.eccentricities[0]} (真 {ecc[0]})")
    assert abs(s_rec - 0.35) < 0.02, "s 标定超差"

    print("=" * 64)
    print("自检 4 — 同心度（预留同轴度参数：未知→带不确定度；填值→收紧）")
    reg_unknown = CrossReg(delta_reg=np.array([0.05, -0.02]),
                           ref_coaxiality_observed=None,
                           ref_coaxiality_uncertainty=0.05)
    c_o, c_i = np.array([0.20, 0.10]), np.array([0.16, 0.09])
    conc0 = concentricity(c_o, c_i, reg_unknown)
    u0 = concentricity_uncertainty(0.008, reg_unknown)
    reg_known = CrossReg(delta_reg=np.array([0.05, -0.02]),
                         ref_coaxiality_observed=np.array([-0.01, 0.005]))
    conc1 = concentricity(c_o, c_i, reg_known)
    u1 = concentricity_uncertainty(0.008, reg_known)
    print(f"  未知同轴度: 同心度={conc0*1000:.1f} µm, 不确定度={u0*1000:.1f} µm")
    print(f"  填入观测值: 同心度={conc1*1000:.1f} µm, 不确定度={u1*1000:.1f} µm（收紧 ✓）")
    assert u1 < u0, "填入同轴度后不确定度未收紧"

    print("=" * 64)
    print("自检 5 — 主轴误差多步分离（3次重夹索引）")
    grid = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    part_true = 0.004 * np.cos(3 * grid) + 0.002 * np.cos(2 * grid)  # 工件形状
    spindle_true = 0.003 * np.cos(grid + 0.5) + 0.001 * np.cos(2 * grid)  # 主轴误差
    idxs = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    profs = []
    for idx in idxs:
        shift = int(round(idx / (2 * np.pi) * 360))
        profs.append(np.roll(part_true, shift) + spindle_true
                     + rng.normal(0, 0.0002, 360))
    pf, se = separate_spindle_multistep(profs, idxs)
    err_part = np.std(pf - part_true)
    print(f"  分离后工件形状残差 std = {err_part*1000:.2f} µm（应远小于主轴误差幅值 3µm）")
    assert err_part * 1000 < 1.5, "主轴分离效果不足"

    print("=" * 64)
    print("自检 6 — 装夹斜率 τ 椭圆假象纠正（用中心线斜率，保真实椭圆度）")
    R = 95.0  # OD 190 外半径，装夹斜率假象随 R 增大
    tau_true = np.array([np.deg2rad(2.0), np.deg2rad(0.0)])  # 2° 夹斜
    tmag = np.hypot(*tau_true)
    psi_t = np.arctan2(tau_true[1], tau_true[0])
    # 合成：水平截面切斜圆柱=椭圆(长半轴 R/cosτ) + 真实小椭圆(2阶,方位错开45°)
    phi = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    a, b = R / np.cos(tmag), R
    r_ellipse = a * b / np.hypot(b * np.cos(phi - psi_t), a * np.sin(phi - psi_t))
    real_ovality = 0.004 * np.cos(2 * (phi - np.pi / 4))   # 真实椭圆度 ±4µm
    rr = r_ellipse + real_ovality
    xs, ys = rr * np.cos(phi), rr * np.sin(phi)
    cf = fit_circle_geometric(xs, ys)
    before = roundness_from_points(xs, ys, cf.cx, cf.cy)
    # τ 来自中心线斜率：此处直接用真值模拟“已从多截面圆心拟合得到”
    after = roundness_corrected_for_tilt(xs, ys, cf.cx, cf.cy, tau_true)
    print(f"  纠正前: 直径 {2*before.r_mean:.4f}, 圆度 {before.roundness_lsc*1000:.1f} µm, "
          f"2阶 {before.upr[2]*1000:.1f} µm（含夹斜椭圆假象）")
    print(f"  纠正后: 直径 {2*after.r_mean:.4f} (真 {2*R:.4f}), 圆度 {after.roundness_lsc*1000:.1f} µm, "
          f"2阶 {after.upr[2]*1000:.1f} µm（真实椭圆度≈8µm峰峰 ✓）")
    assert abs(2 * after.r_mean - 2 * R) < 0.003, "直径偏置未扣净"
    assert after.upr[2] < before.upr[2] * 0.5, "夹斜椭圆假象未去除"
    assert after.upr[2] * 1000 > 2, "真实椭圆度被误删"

    print("=" * 64)
    print("全部自检通过 ✓")


if __name__ == "__main__":
    _selftest()
