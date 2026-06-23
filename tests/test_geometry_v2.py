from __future__ import annotations

"""geometry_v2 纯算法自检(合成数据,断言恢复精度)。

移植自 `docs/frp_metrology_core.py` 的 `_selftest()`:已知偏心 + 不共线 +
多瓣 + 装夹斜率 τ 的合成数据,验证直径 / 圆度 / s 标定 / τ 椭圆假象纠正 /
同心度不确定度 / 主轴多步分离的恢复精度。
"""

import numpy as np
import pytest

from domain.geometry_calibration import (
    CrossReg,
    IdCalDataset,
    ProbePose,
    ToolingCalibration,
    calibrate_id_tooling,
    concentricity,
    concentricity_uncertainty,
    id_points_from_readings,
    id_predict_L,
    id_tooling_from_simple,
)
from domain.geometry_fit import (
    fit_circle_geometric,
    roundness_corrected_for_tilt,
    roundness_from_points,
    separate_spindle_multistep,
    support_to_boundary,
)


# ---------------------------------------------------------------------------
# 合成数据 helper(与骨架一致)
# ---------------------------------------------------------------------------

def _make_inner_circle_readings(tooling, r, e, lobes=None, n=360, noise=0.0, rng=None):
    """合成 ID 读数:内圆半径 r、θ=0 圆心 e,可加多瓣形状误差与噪声。"""
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
    """合成 OD 支撑函数读数:用真实边界点取各方向最大投影(含奇数瓣)。"""
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


# ---------------------------------------------------------------------------
# 自检 1 — ID 生产补偿(已知工装,含偏心+不共线+3瓣,恢复直径/圆度)
# ---------------------------------------------------------------------------

def test_id_section_recovers_diameter_with_eccentricity_and_lobes():
    rng = np.random.default_rng(42)
    true_tool = id_tooling_from_simple(D=140.0, s=0.3, axis_deg=12.0, q=(2.0, -1.5))
    r_true = 76.35  # 6" 内半径 (ID 152.7)
    e_true = np.array([1.2, -0.8])           # 偏心(约束3)
    lobes = {3: 0.012, 2: 0.006}             # 真圆度形状(含奇瓣 3)
    theta, L1, L2 = _make_inner_circle_readings(
        true_tool, r_true, e_true, lobes=lobes, n=360, noise=0.0005, rng=rng)
    pts = id_points_from_readings(true_tool, theta, L1, L2)
    cf = fit_circle_geometric(pts[:, 0], pts[:, 1])
    # 内径恢复到 5µm 内
    assert abs(2 * cf.r - 2 * r_true) < 0.005
    # 圆心恢复出真实偏心
    assert np.allclose([cf.cx, cf.cy], e_true, atol=0.02)


# ---------------------------------------------------------------------------
# 自检 2 — OD 支撑函数重建(含3瓣奇数瓣,验证不被卡尺盲区抹掉)
# ---------------------------------------------------------------------------

def test_od_support_reconstruction_captures_odd_lobe():
    R_true = 95.0  # OD 190 外半径
    od_lobes = {3: 0.010, 2: 0.005}
    th_o, h_o = _make_od_support(R_true, center=(0.9, -0.6), lobes=od_lobes, n=720)
    bnd = support_to_boundary(th_o, h_o)
    cfo = fit_circle_geometric(bnd[:, 0], bnd[:, 1])
    rndo = roundness_from_points(bnd[:, 0], bnd[:, 1], cfo.cx, cfo.cy)
    assert abs(cfo.r - R_true) < 0.01
    assert np.allclose([cfo.cx, cfo.cy], [0.9, -0.6], atol=0.02)
    # 3 阶奇瓣被测到(对射宽度法恒为 0)
    assert rndo.upr[3] * 1000 > 5


# ---------------------------------------------------------------------------
# 自检 3 — ID 联合 LM 位姿标定(未知 s/q,3次装夹联合恢复)
# ---------------------------------------------------------------------------

def test_id_joint_lm_calibration_recovers_lateral_offset():
    pytest.importorskip("scipy")  # 联合 LM 标定强依赖 scipy.optimize.least_squares
    rng = np.random.default_rng(7)
    truth = id_tooling_from_simple(D=140.0, s=0.35, axis_deg=8.0, q=(1.5, 0.7))
    ecc = [np.array([1.0, 0.5]), np.array([-1.3, 0.9]), np.array([0.4, -1.6])]
    datasets = []
    for e in ecc:
        th, l1, l2 = _make_inner_circle_readings(truth, 76.35, e, n=240,
                                                 noise=0.0003, rng=rng)
        datasets.append(IdCalDataset(th, l1, l2))
    res = calibrate_id_tooling(datasets, r_known=76.35, D_init=140.0,
                               init_s=0.0, init_axis_deg=0.0)
    assert res.success
    # 反推恢复的 s:A、B 线横向间距
    fa, fb = res.tooling.probe_a.f, res.tooling.probe_b.f
    n = res.tooling.probe_b.n
    perp = np.array([-n[1], n[0]])
    s_rec = float((fb - fa) @ perp)
    assert abs(s_rec - 0.35) < 0.02


# ---------------------------------------------------------------------------
# 自检 4 — 同心度(预留同轴度参数:未知→带不确定度;填值→收紧)
# ---------------------------------------------------------------------------

def test_concentricity_uncertainty_tightens_with_observed_coaxiality():
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
    assert conc0 >= 0.0 and conc1 >= 0.0
    # 填入观测同轴度后不确定度收紧
    assert u1 < u0


# ---------------------------------------------------------------------------
# 自检 5 — 主轴误差多步分离(3次重夹索引)
# ---------------------------------------------------------------------------

def test_spindle_multistep_separation_recovers_part_form():
    rng = np.random.default_rng(123)
    grid = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    part_true = 0.004 * np.cos(3 * grid) + 0.002 * np.cos(2 * grid)       # 工件形状
    spindle_true = 0.003 * np.cos(grid + 0.5) + 0.001 * np.cos(2 * grid)  # 主轴误差
    idxs = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    profs = []
    for idx in idxs:
        shift = int(round(idx / (2 * np.pi) * 360))
        profs.append(np.roll(part_true, shift) + spindle_true
                     + rng.normal(0, 0.0002, 360))
    pf, _se = separate_spindle_multistep(profs, idxs)
    err_part = float(np.std(pf - part_true))
    assert err_part * 1000 < 1.5  # 残差 < 1.5µm,远小于主轴误差幅值 3µm


# ---------------------------------------------------------------------------
# 自检 6 — 装夹斜率 τ 椭圆假象纠正(用中心线斜率,保真实椭圆度)
# ---------------------------------------------------------------------------

def test_clamping_tilt_correction_removes_ellipse_artifact_keeps_real_ovality():
    R = 95.0  # OD 190 外半径,装夹斜率假象随 R 增大
    tau_true = np.array([np.deg2rad(2.0), np.deg2rad(0.0)])  # 2° 夹斜
    tmag = float(np.hypot(*tau_true))
    psi_t = float(np.arctan2(tau_true[1], tau_true[0]))
    # 水平截面切斜圆柱=椭圆(长半轴 R/cosτ) + 真实小椭圆(2阶,方位错开45°)
    phi = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    a, b = R / np.cos(tmag), R
    r_ellipse = a * b / np.hypot(b * np.cos(phi - psi_t), a * np.sin(phi - psi_t))
    real_ovality = 0.004 * np.cos(2 * (phi - np.pi / 4))   # 真实椭圆度 ±4µm
    rr = r_ellipse + real_ovality
    xs, ys = rr * np.cos(phi), rr * np.sin(phi)
    cf = fit_circle_geometric(xs, ys)
    before = roundness_from_points(xs, ys, cf.cx, cf.cy)
    after = roundness_corrected_for_tilt(xs, ys, cf.cx, cf.cy, tau_true)
    # 直径偏置扣净
    assert abs(2 * after.r_mean - 2 * R) < 0.003
    # 夹斜椭圆假象去除(2 阶大幅下降)
    assert after.upr[2] < before.upr[2] * 0.5
    # 真实椭圆度未被误删
    assert after.upr[2] * 1000 > 2


# ---------------------------------------------------------------------------
# ToolingCalibration 序列化往返 + 派生对象
# ---------------------------------------------------------------------------

def test_tooling_calibration_roundtrip_and_derived_objects():
    tc = ToolingCalibration(
        od_k0=1.001, od_b=0.5, od_psi_deg=1.2,
        id_D_eff=140.0, id_s_lateral=0.35, id_axis_deg=8.0, id_qx=1.5, id_qy=0.7,
        delta_reg=(0.05, -0.02), ref_coaxiality_observed=(-0.01, 0.005),
        axis_slope_x=1e-5, ref_od=190.0, ref_id=152.7, chuck_error_bound=0.02,
    )
    restored = ToolingCalibration.from_dict(tc.to_dict())
    assert restored.id_D_eff == pytest.approx(140.0)
    assert restored.id_s_lateral == pytest.approx(0.35)
    assert restored.delta_reg == pytest.approx((0.05, -0.02))
    assert restored.ref_coaxiality_observed == pytest.approx((-0.01, 0.005))
    assert restored.ref_od == pytest.approx(190.0)
    assert restored.chuck_error_bound == pytest.approx(0.02)

    # 派生对象类型正确
    tooling = restored.id_tooling()
    assert isinstance(tooling.probe_a, ProbePose)
    reg = restored.cross_reg()
    assert reg.ref_coaxiality_observed is not None

    # observed=None 时序列化往返保持 None
    tc2 = ToolingCalibration(ref_coaxiality_observed=None)
    assert ToolingCalibration.from_dict(tc2.to_dict()).ref_coaxiality_observed is None


def test_run_synthetic_selftest_all_pass():
    pytest.importorskip("scipy")
    from domain.geometry_calibration import run_synthetic_selftest

    report = run_synthetic_selftest()
    assert report["ok"] is True, report
    names = {c["name"] for c in report["checks"]}
    assert "ID 内径恢复" in names and "τ 椭圆假象纠正" in names
