from __future__ import annotations

"""传感器正/逆向模型、工装位姿与标定反演(geometry_v2)。

来源:`docs/frp_metrology_core.py` 骨架的"OD 投影式测径仪 / ID 背向位移计线-圆模型 /
联合 LM 标定 / 互配准 + 同心度 / 生产补偿管线"部分。仅依赖 numpy / scipy 与
`domain.geometry_fit`,不依赖 UI / driver / AppHost / repositories,可单测。

`ToolingCalibration` 是 `tooling_calibration.json`(设计 §7)的内存表示,提供
to_dict/from_dict;repository 层负责落盘,本模块不做 IO。
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from domain.geometry_fit import (
    derotate,
    fit_circle_geometric,
    rot,
    roundness_from_points,
)

try:  # pragma: no cover - scipy 始终在 requirements 中
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover
    least_squares = None


# =============================================================================
# 1. OD:投影式测径仪标度/零位(含远心度)
# =============================================================================

def od_calibrate_scale(known_diam: Sequence[float],
                       raw_width: Sequence[float],
                       positions: Optional[Sequence[float]] = None) -> dict[str, float]:
    """OD 标度/零位标定。两点以上线性拟合:diam = k·raw + b。

    若给 positions(柱规在量隙的横向位置),追加远心度项 diam = (k0 + k1·x)·raw + b。
    返回 dict {k0, k1, b}。
    """
    known = np.asarray(known_diam, float)
    raw = np.asarray(raw_width, float)
    if positions is None:
        A = np.column_stack([raw, np.ones_like(raw)])
        sol = np.linalg.lstsq(A, known, rcond=None)[0]
        return {"k0": float(sol[0]), "k1": 0.0, "b": float(sol[1])}
    pos = np.asarray(positions, float)
    A = np.column_stack([raw, raw * pos, np.ones_like(raw)])
    sol = np.linalg.lstsq(A, known, rcond=None)[0]
    return {"k0": float(sol[0]), "k1": float(sol[1]), "b": float(sol[2])}


def od_apply_scale(raw_width: np.ndarray, cal: Mapping[str, float],
                   position: float = 0.0) -> np.ndarray:
    """把 OD 原始宽度/支撑读数转成物理量(含远心度修正)。"""
    raw = np.asarray(raw_width, float)
    return (float(cal["k0"]) + float(cal["k1"]) * position) * raw + float(cal["b"])


# =============================================================================
# 2. ID:背向位移计(线-圆 正/逆向模型)
# =============================================================================

@dataclass(slots=True)
class ProbePose:
    """单个位移计在机床系(2D)的位姿。hit = f + (k·L·cosγ)·n。"""
    fx: float
    fy: float
    nx: float
    ny: float
    k: float = 1.0          # 标度
    gamma_deg: float = 0.0  # 出平面倾角 → 弦过读,作为标度修正

    @property
    def f(self) -> np.ndarray:
        return np.array([self.fx, self.fy])

    @property
    def n(self) -> np.ndarray:
        v = np.array([self.nx, self.ny])
        return v / np.linalg.norm(v)


@dataclass(slots=True)
class IdTooling:
    probe_a: ProbePose
    probe_b: ProbePose


def id_tooling_from_simple(D: float, s: float = 0.0,
                           axis_deg: float = 0.0,
                           q: Sequence[float] = (0.0, 0.0),
                           k_a: float = 1.0, k_b: float = 1.0,
                           gamma_deg: float = 0.0) -> IdTooling:
    """由"理想背向共线 + 工装偏差"构造位姿。

    D     : 两感测面间距(背向,朝外)
    s     : 两探头线横向偏移(不共线,约束2)
    axis_deg: 探头线方位角
    q     : 探头线相对 O 的横向位置(约束4,随截面不同)
    """
    ax = np.deg2rad(axis_deg)
    n = np.array([np.cos(ax), np.sin(ax)])       # B 朝 +n
    perp = np.array([-np.sin(ax), np.cos(ax)])   # 横向
    qv = np.asarray(q, float)
    # 背向:A 面在 q-(D/2)n 朝 -n;B 面在 q+(D/2)n 朝 +n;B 线横向偏移 s
    fa = qv - (D / 2.0) * n
    fb = qv + (D / 2.0) * n + s * perp
    a = ProbePose(float(fa[0]), float(fa[1]), float(-n[0]), float(-n[1]),
                  k=k_a, gamma_deg=gamma_deg)
    b = ProbePose(float(fb[0]), float(fb[1]), float(n[0]), float(n[1]),
                  k=k_b, gamma_deg=gamma_deg)
    return IdTooling(a, b)


def _line_circle_outward_t(f: np.ndarray, n: np.ndarray,
                           m: np.ndarray, r: float) -> float:
    """从 f 沿 +n 到半径 r、圆心 m 的圆的"朝外壁"交点距离 t(取较大正根)。"""
    d = f - m
    B = float(n @ d)
    C = float(d @ d - r * r)
    disc = B * B - C
    if disc < 0:
        return float("nan")
    return float(-B + np.sqrt(disc))


def id_predict_L(pose: ProbePose, e: np.ndarray, theta: float, r: float) -> float:
    """正向模型:给定内圆半径 r、θ=0 时内圆心 e(机床系),预测该探头读数 L。

    内圆心随旋转 m(θ)=R(θ)e 绕 O 公转(含约束3偏心)。
    """
    m = rot(theta) @ np.asarray(e, float)
    t = _line_circle_outward_t(pose.f, pose.n, m, r)
    cg = np.cos(np.deg2rad(pose.gamma_deg))
    return float(t / (pose.k * cg))


def id_hit_point(pose: ProbePose, L: float) -> np.ndarray:
    """逆向:把读数 L 还原成机床系壁面命中点。"""
    cg = np.cos(np.deg2rad(pose.gamma_deg))
    return pose.f + (pose.k * L * cg) * pose.n


def id_points_from_readings(tooling: IdTooling,
                            theta: np.ndarray,
                            L1: np.ndarray, L2: np.ndarray) -> np.ndarray:
    """整圈 (θ,L1,L2) → 去旋转后的工件系内壁点云 (2N,2)。"""
    theta = np.asarray(theta, float)
    L1 = np.asarray(L1, float)
    L2 = np.asarray(L2, float)
    ha = np.array([id_hit_point(tooling.probe_a, float(lv)) for lv in L1])
    hb = np.array([id_hit_point(tooling.probe_b, float(lv)) for lv in L2])
    pa = derotate(ha, theta)
    pb = derotate(hb, theta)
    return np.vstack([pa, pb])


# =============================================================================
# 3. 联合 LM 标定(ID 位姿:s / 方位 / q / 标度,约束 2&4)
# =============================================================================

@dataclass(slots=True)
class IdCalDataset:
    """一次装夹的参考件旋转数据。"""
    theta: np.ndarray
    L1: np.ndarray
    L2: np.ndarray


@dataclass(slots=True)
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
    """以参考件已知内半径 r_known 为约束,联合拟合共享工装参数 + 每次装夹偏心。

    共享参数: [s, axis_deg, qx, qy, (k_a, k_b)];每装夹 nuisance: e=(ex,ey)。
    残差: 各采样的 (预测 L - 实测 L)。多次装夹(不同 e)改善 s 与 q 的可分性。

    注意: 绝对 q(探头线相对 O)由"去旋转后点必须落在以 r_known 为半径的圆上"锁定——
    错误的 q 会让去旋转点云呈外摆线状、残差上升。
    """
    if least_squares is None:  # pragma: no cover
        raise RuntimeError("需要 scipy.optimize.least_squares")

    nset = len(datasets)
    nshared = 4 + (2 if fit_scale else 0)

    def unpack(p: np.ndarray):
        s, axis_deg, qx, qy = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        if fit_scale:
            k_a, k_b = float(p[4]), float(p[5])
        else:
            k_a = k_b = 1.0
        es = [np.array([p[nshared + 2 * i], p[nshared + 2 * i + 1]])
              for i in range(nset)]
        return s, axis_deg, qx, qy, k_a, k_b, es

    def residuals(p: np.ndarray) -> np.ndarray:
        s, axis_deg, qx, qy, k_a, k_b, es = unpack(p)
        tl = id_tooling_from_simple(D_init, s=s, axis_deg=axis_deg,
                                    q=(qx, qy), k_a=k_a, k_b=k_b)
        out: list[float] = []
        for ds, e in zip(datasets, es):
            for th, lv in zip(ds.theta, ds.L1):
                out.append(id_predict_L(tl.probe_a, e, float(th), r_known) - float(lv))
            for th, lv in zip(ds.theta, ds.L2):
                out.append(id_predict_L(tl.probe_b, e, float(th), r_known) - float(lv))
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
# 4. 互配准 + 同心度(含预留 ref_coaxiality_observed)
# =============================================================================

@dataclass(slots=True)
class CrossReg:
    delta_reg: np.ndarray                      # OD↔ID 传感器零位横向偏移
    ref_coaxiality_observed: Optional[np.ndarray] = None  # 预留:未知=None
    ref_coaxiality_uncertainty: float = 0.05   # mm;未知时整体计入不确定度

    def coax(self) -> np.ndarray:
        if self.ref_coaxiality_observed is None:
            return np.zeros(2)
        return np.asarray(self.ref_coaxiality_observed, float)


def concentricity(c_o: np.ndarray, c_i: np.ndarray, reg: CrossReg) -> float:
    """同心度 = |(c_o - c_i) - Δ_reg - ref_coaxiality_observed|(设计 §6)。"""
    v = np.asarray(c_o, float) - np.asarray(c_i, float) - reg.delta_reg - reg.coax()
    return float(np.linalg.norm(v))


def concentricity_uncertainty(fit_noise: float, reg: CrossReg) -> float:
    """同心度不确定度:拟合噪声 ⊕(同轴度未知时其上界)。"""
    extra = 0.0 if reg.ref_coaxiality_observed is not None \
        else reg.ref_coaxiality_uncertainty
    return float(np.hypot(fit_noise, extra))


# =============================================================================
# 5. 生产补偿管线(单截面)
# =============================================================================

@dataclass(slots=True)
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
                       od_cal: Mapping[str, float],
                       id_tooling: IdTooling,
                       reg: CrossReg,
                       od_position: float = 0.0) -> SectionResult:
    """单截面补偿:原始读数 → 直径/圆心/圆度/同心度。"""
    # --- OD:标定 → 支撑函数 → 边界点 → 圆拟合 ---
    from domain.geometry_fit import support_to_boundary

    h = od_apply_scale(od_support_raw, od_cal, position=od_position)
    od_pts = support_to_boundary(theta, h)
    cf_o = fit_circle_geometric(od_pts[:, 0], od_pts[:, 1])
    rnd_o = roundness_from_points(od_pts[:, 0], od_pts[:, 1], cf_o.cx, cf_o.cy)

    # --- ID:读数 → 命中点 → 去旋转 → 圆拟合 ---
    id_pts = id_points_from_readings(id_tooling, theta, id_L1, id_L2)
    cf_i = fit_circle_geometric(id_pts[:, 0], id_pts[:, 1])
    rnd_i = roundness_from_points(id_pts[:, 0], id_pts[:, 1], cf_i.cx, cf_i.cy)

    c_o = np.array([cf_o.cx, cf_o.cy])
    c_i = np.array([cf_i.cx, cf_i.cy])
    conc = concentricity(c_o, c_i, reg)
    return SectionResult(float(z), 2 * cf_o.r, c_o, rnd_o.roundness_lsc,
                         2 * cf_i.r, c_i, rnd_i.roundness_lsc, conc)


# =============================================================================
# 6. 工装标定参数(tooling_calibration.json 内存表示,设计 §7)
# =============================================================================

def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass(slots=True)
class ToolingCalibration:
    """`tooling_calibration.json` 的内存表示(设计 §7)。

    纯数据 + 序列化,不做 IO。repository 负责读写文件。
    """
    # OD 标度/零位/轴方位
    od_k0: float = 1.0
    od_k1: float = 0.0
    od_b: float = 0.0
    od_psi_deg: float = 0.0
    od_beta_deg: float = 0.0
    # ID 基线/位姿
    id_D_eff: float = 0.0
    id_s_lateral: float = 0.0
    id_axis_deg: float = 0.0
    id_qx: float = 0.0
    id_qy: float = 0.0
    id_k_a: float = 1.0
    id_k_b: float = 1.0
    id_gamma_deg: float = 0.0
    # 互配准
    delta_reg: tuple[float, float] = (0.0, 0.0)
    ref_coaxiality_observed: Optional[tuple[float, float]] = None
    ref_coaxiality_uncertainty: float = 0.05
    # 回转轴直线度(线性:斜率 + 截距,机床 x/y 随 z)
    axis_slope_x: float = 0.0
    axis_slope_y: float = 0.0
    # 参考件
    ref_od: Optional[float] = None
    ref_id: Optional[float] = None
    ref_roundness: Optional[float] = None
    # 卡盘误差定界(Phase 0 实测幅值上界)
    chuck_error_bound: Optional[float] = None
    # 元数据
    meta: dict[str, Any] = field(default_factory=dict)

    def id_calibrated(self) -> bool:
        """ID 工装是否已标定(D_eff 有效)。未标定时 geometry_v2 跳过 ID 重建。"""
        return float(self.id_D_eff) > 0.0

    def od_calibrated(self) -> bool:
        """OD 工装是否已标定(标度/零位/方位之一被设过)。默认未标定。"""
        return (self.od_k0 != 1.0) or (self.od_b != 0.0) or (self.od_psi_deg != 0.0)

    def id_tooling(self) -> IdTooling:
        """构造 ID 工装位姿对象。"""
        return id_tooling_from_simple(
            self.id_D_eff, s=self.id_s_lateral, axis_deg=self.id_axis_deg,
            q=(self.id_qx, self.id_qy), k_a=self.id_k_a, k_b=self.id_k_b,
            gamma_deg=self.id_gamma_deg,
        )

    def od_cal(self) -> dict[str, float]:
        """OD 标度 dict(供 od_apply_scale)。"""
        return {"k0": self.od_k0, "k1": self.od_k1, "b": self.od_b}

    def cross_reg(self) -> CrossReg:
        observed = (
            None if self.ref_coaxiality_observed is None
            else np.asarray(self.ref_coaxiality_observed, float)
        )
        return CrossReg(
            delta_reg=np.asarray(self.delta_reg, float),
            ref_coaxiality_observed=observed,
            ref_coaxiality_uncertainty=self.ref_coaxiality_uncertainty,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "od": {
                "k_o0": self.od_k0, "k_o1": self.od_k1, "B_eff": self.od_b,
                "psi_deg": self.od_psi_deg, "beta_deg": self.od_beta_deg,
            },
            "id": {
                "D_eff": self.id_D_eff, "s_lateral": self.id_s_lateral,
                "axis_deg": self.id_axis_deg, "qx": self.id_qx, "qy": self.id_qy,
                "k_A": self.id_k_a, "k_B": self.id_k_b, "gamma_deg": self.id_gamma_deg,
            },
            "cross_registration": {
                "delta_reg": [self.delta_reg[0], self.delta_reg[1]],
                "ref_coaxiality_observed": (
                    None if self.ref_coaxiality_observed is None
                    else [self.ref_coaxiality_observed[0], self.ref_coaxiality_observed[1]]
                ),
                "ref_coaxiality_uncertainty": self.ref_coaxiality_uncertainty,
            },
            "axis": {
                "axis_slope_x": self.axis_slope_x,
                "axis_slope_y": self.axis_slope_y,
            },
            "reference_part": {
                "ref_OD": self.ref_od, "ref_ID": self.ref_id,
                "ref_roundness": self.ref_roundness,
                "chuck_error_bound": self.chuck_error_bound,
            },
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ToolingCalibration":
        d = data if isinstance(data, Mapping) else {}
        od = d.get("od", {}) if isinstance(d.get("od"), Mapping) else {}
        idp = d.get("id", {}) if isinstance(d.get("id"), Mapping) else {}
        cr = d.get("cross_registration", {}) if isinstance(d.get("cross_registration"), Mapping) else {}
        ax = d.get("axis", {}) if isinstance(d.get("axis"), Mapping) else {}
        rp = d.get("reference_part", {}) if isinstance(d.get("reference_part"), Mapping) else {}

        def _pair(value: Any) -> Optional[tuple[float, float]]:
            if value is None:
                return None
            try:
                return (float(value[0]), float(value[1]))
            except Exception:
                return None

        delta = _pair(cr.get("delta_reg")) or (0.0, 0.0)
        observed = _pair(cr.get("ref_coaxiality_observed"))
        meta = d.get("meta", {})
        return cls(
            od_k0=_f(od.get("k_o0"), 1.0),
            od_k1=_f(od.get("k_o1"), 0.0),
            od_b=_f(od.get("B_eff"), 0.0),
            od_psi_deg=_f(od.get("psi_deg"), 0.0),
            od_beta_deg=_f(od.get("beta_deg"), 0.0),
            id_D_eff=_f(idp.get("D_eff"), 0.0),
            id_s_lateral=_f(idp.get("s_lateral"), 0.0),
            id_axis_deg=_f(idp.get("axis_deg"), 0.0),
            id_qx=_f(idp.get("qx"), 0.0),
            id_qy=_f(idp.get("qy"), 0.0),
            id_k_a=_f(idp.get("k_A"), 1.0),
            id_k_b=_f(idp.get("k_B"), 1.0),
            id_gamma_deg=_f(idp.get("gamma_deg"), 0.0),
            delta_reg=delta,
            ref_coaxiality_observed=observed,
            ref_coaxiality_uncertainty=_f(cr.get("ref_coaxiality_uncertainty"), 0.05),
            axis_slope_x=_f(ax.get("axis_slope_x"), 0.0),
            axis_slope_y=_f(ax.get("axis_slope_y"), 0.0),
            ref_od=None if rp.get("ref_OD") is None else _f(rp.get("ref_OD")),
            ref_id=None if rp.get("ref_ID") is None else _f(rp.get("ref_ID")),
            ref_roundness=None if rp.get("ref_roundness") is None else _f(rp.get("ref_roundness")),
            chuck_error_bound=None if rp.get("chuck_error_bound") is None else _f(rp.get("chuck_error_bound")),
            meta=dict(meta) if isinstance(meta, Mapping) else {},
        )


def estimate_od_axis_psi(theta: np.ndarray, h: np.ndarray,
                         ref_phi: Optional[np.ndarray] = None,
                         ref_dr: Optional[np.ndarray] = None,
                         n_grid: int = 720) -> float:
    """OD 测量轴方位角 ψ 估计(设计 §Phase2)。

    单边支撑序列重建工件系边界 → 径向偏差廓线,与参考件**已知径向偏差廓线**
    (证书,ref_phi[rad]/ref_dr[mm])做圆周互相关,返回使两者对齐的角位移 ψ(deg)。
    缺少参考廓线时返回 0(圆对称件无角向基准,ψ 不可观测)。
    """
    from domain.geometry_fit import (
        fit_circle_geometric,
        roundness_from_points,
        support_to_boundary,
    )

    pts = support_to_boundary(np.asarray(theta, float), np.asarray(h, float), n_grid)
    cf = fit_circle_geometric(pts[:, 0], pts[:, 1])
    rnd = roundness_from_points(pts[:, 0], pts[:, 1], cf.cx, cf.cy)
    if ref_phi is None or ref_dr is None:
        return 0.0

    grid = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)

    def _resample(phi: np.ndarray, dr: np.ndarray) -> np.ndarray:
        phi = np.mod(np.asarray(phi, float), 2 * np.pi)
        dr = np.asarray(dr, float)
        order = np.argsort(phi)
        ps = np.concatenate([phi[order], phi[order][:1] + 2 * np.pi])
        ds = np.concatenate([dr[order], dr[order][:1]])
        return np.interp(grid, ps, ds)

    meas = _resample(rnd.phi, rnd.dr)
    ref = _resample(ref_phi, ref_dr)
    # circular cross-correlation via FFT; shift maximizing alignment
    corr = np.fft.irfft(np.fft.rfft(meas) * np.conj(np.fft.rfft(ref)), n=n_grid)
    k = int(np.argmax(corr))
    psi = k / float(n_grid) * 360.0
    if psi > 180.0:
        psi -= 360.0
    return float(psi)


def run_synthetic_selftest() -> dict[str, Any]:
    """合成数据自检(供「几何标定 V2」页 Box5 与单测共用)。

    用已知偏心+不共线+多瓣+装夹斜率的合成数据,验证算法链的恢复精度。
    返回 {"ok": bool, "checks": [{"name","passed","detail"}, ...]}。纯函数,
    不依赖硬件/UI。需要 scipy 的项在缺 scipy 时标记 skipped(不算失败)。
    """
    from domain.geometry_fit import (
        fit_circle_geometric,
        roundness_corrected_for_tilt,
        roundness_from_points,
        support_to_boundary,
    )

    checks: list[dict[str, Any]] = []

    def _add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    rng = np.random.default_rng(42)

    # 1. ID 生产补偿(偏心+不共线+3瓣)
    try:
        tool = id_tooling_from_simple(D=140.0, s=0.3, axis_deg=12.0, q=(2.0, -1.5))
        r_true = 76.35
        e_true = np.array([1.2, -0.8])
        theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        lobes = {3: 0.012, 2: 0.006}
        L1 = np.empty(360)
        L2 = np.empty(360)
        for i, th in enumerate(theta):
            rr = r_true + sum(a * np.cos(k * th) for k, a in lobes.items())
            L1[i] = id_predict_L(tool.probe_a, e_true, float(th), rr) + rng.normal(0, 0.0005)
            L2[i] = id_predict_L(tool.probe_b, e_true, float(th), rr) + rng.normal(0, 0.0005)
        pts = id_points_from_readings(tool, theta, L1, L2)
        cf = fit_circle_geometric(pts[:, 0], pts[:, 1])
        err_um = (2 * cf.r - 2 * r_true) * 1000.0
        _add("ID 内径恢复", abs(err_um) < 5.0, f"内径误差 {err_um:+.2f} µm (<5)")
    except Exception as exc:  # pragma: no cover
        _add("ID 内径恢复", False, f"异常: {exc}")

    # 2. OD 支撑函数重建(含奇瓣)
    try:
        R_true = 95.0
        phi = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        rr = R_true + 0.010 * np.cos(3 * phi) + 0.005 * np.cos(2 * phi)
        bx = 0.9 + rr * np.cos(phi)
        by = -0.6 + rr * np.sin(phi)
        th_o = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        h = np.array([np.max(bx * np.cos(-t) + by * np.sin(-t)) for t in th_o])
        bnd = support_to_boundary(th_o, h)
        cfo = fit_circle_geometric(bnd[:, 0], bnd[:, 1])
        rndo = roundness_from_points(bnd[:, 0], bnd[:, 1], cfo.cx, cfo.cy)
        ok = abs(cfo.r - R_true) < 0.01 and rndo.upr[3] * 1000 > 5
        _add("OD 支撑重建", ok, f"外半径误差 {(cfo.r - R_true) * 1000:+.2f} µm, 3阶 {rndo.upr[3] * 1000:.1f} µm")
    except Exception as exc:  # pragma: no cover
        _add("OD 支撑重建", False, f"异常: {exc}")

    # 3. ID 联合 LM 位姿标定(需 scipy)
    if least_squares is None:  # pragma: no cover
        _add("ID 联合LM标定", True, "skipped (无 scipy)")
    else:
        try:
            truth = id_tooling_from_simple(D=140.0, s=0.35, axis_deg=8.0, q=(1.5, 0.7))
            datasets = []
            for e in (np.array([1.0, 0.5]), np.array([-1.3, 0.9]), np.array([0.4, -1.6])):
                th = np.linspace(0, 2 * np.pi, 240, endpoint=False)
                l1 = np.array([id_predict_L(truth.probe_a, e, float(t), 76.35) + rng.normal(0, 0.0003) for t in th])
                l2 = np.array([id_predict_L(truth.probe_b, e, float(t), 76.35) + rng.normal(0, 0.0003) for t in th])
                datasets.append(IdCalDataset(th, l1, l2))
            res = calibrate_id_tooling(datasets, r_known=76.35, D_init=140.0)
            n = res.tooling.probe_b.n
            perp = np.array([-n[1], n[0]])
            s_rec = float((res.tooling.probe_b.f - res.tooling.probe_a.f) @ perp)
            _add("ID 联合LM标定", abs(s_rec - 0.35) < 0.02, f"恢复 s={s_rec:+.4f} mm (真 0.35)")
        except Exception as exc:  # pragma: no cover
            _add("ID 联合LM标定", False, f"异常: {exc}")

    # 4. 装夹斜率 τ 椭圆假象纠正
    try:
        R = 95.0
        tau_true = np.array([np.deg2rad(2.0), 0.0])
        tmag = float(np.hypot(*tau_true))
        phi = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        a, b = R / np.cos(tmag), R
        r_ell = a * b / np.hypot(b * np.cos(phi), a * np.sin(phi))
        rr = r_ell + 0.004 * np.cos(2 * (phi - np.pi / 4))
        xs, ys = rr * np.cos(phi), rr * np.sin(phi)
        cf = fit_circle_geometric(xs, ys)
        before = roundness_from_points(xs, ys, cf.cx, cf.cy)
        after = roundness_corrected_for_tilt(xs, ys, cf.cx, cf.cy, tau_true)
        ok = abs(2 * after.r_mean - 2 * R) < 0.003 and after.upr[2] < before.upr[2] * 0.5 and after.upr[2] * 1000 > 2
        _add("τ 椭圆假象纠正", ok, f"纠正后直径误差 {(2 * after.r_mean - 2 * R) * 1000:+.2f} µm, 2阶 {after.upr[2] * 1000:.1f} µm")
    except Exception as exc:  # pragma: no cover
        _add("τ 椭圆假象纠正", False, f"异常: {exc}")

    return {"ok": all(c["passed"] for c in checks), "checks": checks}


__all__ = [
    "CrossReg",
    "IdCalDataset",
    "IdCalResult",
    "IdTooling",
    "ProbePose",
    "SectionResult",
    "ToolingCalibration",
    "calibrate_id_tooling",
    "compensate_section",
    "concentricity",
    "concentricity_uncertainty",
    "estimate_od_axis_psi",
    "id_hit_point",
    "id_points_from_readings",
    "id_predict_L",
    "id_tooling_from_simple",
    "od_apply_scale",
    "od_calibrate_scale",
    "run_synthetic_selftest",
]
