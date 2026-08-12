"""ts_inverse.py — Round 3 T3: 可微 Tanabe-Sugano 反向设计.

给定目标 λ_em (e.g. 800nm NIR-II), 用 PyTorch autograd 反推满足该目标的
晶场参数 (Dq, B, C, S, ℏω) — 即"想烧 800nm Cr3+ 荧光粉, 我应该找什么样的 host?"

核心: TS layer 整条链全微分 (predict_engine/ts_torch.py 已实现 torch.linalg.eigh).
反向设计 loss = MSE(predicted_lambda_em, target_lambda) + 物理可行域 penalty.

用法:
    from predict_engine.ts_inverse import inverse_design
    res = inverse_design(target_lambda_nm=800.0)
    # -> {Dq, B, C, S, hbar_omega, predicted_lambda, n_iter, converged}

Web 入口: dashboard.py /api/ts_inverse?target=800
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from predict_engine.ts_torch import d3_ts_torch


# 物理可行域 (Cr3+ 在 oxide host 八面体中的常见值, 参考 Sugano-Tanabe-Kamimura)
# 这些 box 用于约束优化 + soft-clip
PHYS_BOUNDS = {
    "Dq_cm1":      (1200.0, 2400.0),  # 弱→强场, 一般在 1400-2200
    "B_cm1":       (550.0, 850.0),    # Racah B
    "C_cm1":       (2500.0, 4500.0),  # Racah C
    "S_huang_rhys":(2.0, 12.0),       # Huang-Rhys (强电声耦合 > 弱)
    "hbar_omega_cm1":(250.0, 600.0),  # 主声子频率
}


def _soft_box(x, lo, hi, k=0.005):
    """Soft penalty pushing x into [lo, hi]."""
    return k * (torch.relu(lo - x).pow(2) + torch.relu(x - hi).pow(2))


class _OptimVars(nn.Module):
    """5 个可学习的晶场参数, 初始化为可行域中心."""
    def __init__(self):
        super().__init__()
        # 用 sigmoid scaling 避免 NaN
        # 初始化 raw param 都为 0 -> sigmoid -> 0.5 -> 取值域中点
        self.raw_Dq = nn.Parameter(torch.tensor(0.0))
        self.raw_B = nn.Parameter(torch.tensor(0.0))
        self.raw_C = nn.Parameter(torch.tensor(0.0))
        self.raw_S = nn.Parameter(torch.tensor(0.0))
        self.raw_hw = nn.Parameter(torch.tensor(0.0))

    def values(self):
        def _scale(raw, lo, hi):
            return lo + torch.sigmoid(raw) * (hi - lo)
        Dq = _scale(self.raw_Dq, *PHYS_BOUNDS["Dq_cm1"])
        B = _scale(self.raw_B, *PHYS_BOUNDS["B_cm1"])
        C = _scale(self.raw_C, *PHYS_BOUNDS["C_cm1"])
        S = _scale(self.raw_S, *PHYS_BOUNDS["S_huang_rhys"])
        hw = _scale(self.raw_hw, *PHYS_BOUNDS["hbar_omega_cm1"])
        return Dq, B, C, S, hw


def inverse_design(
    target_lambda_nm: float,
    target_fwhm_nm: Optional[float] = None,
    n_steps: int = 1500,
    lr: float = 0.05,
    seed: int = 42,
    verbose: bool = False,
) -> Dict:
    """通过梯度下降反推 (Dq, B, C, S, ℏω) 使 λ_em ≈ target.

    Args:
        target_lambda_nm: 目标发射波长 (nm). 例 800 (NIR-I), 1100 (NIR-II window).
        target_fwhm_nm: 可选 目标 FWHM (nm). 若给, loss 也包含 FWHM 项.
        n_steps: Adam 梯度下降步数.
        lr: 学习率.
        seed: 随机种子.

    Returns:
        {
          Dq_cm1, B_cm1, C_cm1, S_huang_rhys, hbar_omega_cm1,  # 反推参数
          predicted_lambda_em_nm, predicted_fwhm_nm,           # 末步预测
          target_lambda_nm, lambda_error_nm,                   # 误差
          n_iter, converged, loss_history,                     # 收敛诊断
          host_recommendation                                   # 启发式 host 建议
        }
    """
    torch.manual_seed(seed)
    model = _OptimVars()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    target = torch.tensor(target_lambda_nm)
    target_fwhm = torch.tensor(target_fwhm_nm) if target_fwhm_nm else None

    loss_hist = []
    converged = False
    for step in range(n_steps):
        optim.zero_grad()
        Dq, B, C, S, hw = model.values()
        out = d3_ts_torch(Dq, B, C)
        # 4T2 -> 4A2 emission with Stokes shift S * ℏω
        E_em = out["E_4T2_cm1"] - S * hw
        # safe inversion to nm
        lam = 1.0e7 / torch.clamp(E_em, min=200.0)
        # FWHM Mostafa formula 2.355 * sqrt(S * hw^2 * coth(hw/2kT)) at T=300K
        kT = 208.5  # cm-1 at 300K
        coth = torch.cosh(hw / (2 * kT)) / torch.sinh(hw / (2 * kT))
        fwhm_cm1 = 2.355 * torch.sqrt(S * hw * hw * coth)
        fwhm_nm = fwhm_cm1 * (lam * lam) / 1.0e7

        loss = (lam - target).pow(2) / 100.0
        if target_fwhm is not None:
            loss = loss + 0.1 * (fwhm_nm - target_fwhm).pow(2) / 100.0
        # gentle penalty to keep params near physical mid-range
        loss = loss + 0.005 * (model.raw_Dq.pow(2) + model.raw_B.pow(2) + model.raw_C.pow(2)
                               + model.raw_S.pow(2) + model.raw_hw.pow(2))

        loss.backward()
        optim.step()
        loss_hist.append(loss.item())
        if verbose and step % 100 == 0:
            print(f"step {step:4d} | loss={loss.item():.4f} | λ={lam.item():.1f}nm fwhm={fwhm_nm.item():.1f}nm")
        if abs((lam - target).item()) < 0.5:
            converged = True
            break

    Dq, B, C, S, hw = model.values()
    out = d3_ts_torch(Dq, B, C)
    E_em = out["E_4T2_cm1"] - S * hw
    lam_final = (1.0e7 / E_em).item()
    fwhm_cm1 = 2.355 * torch.sqrt(S * hw * hw * (torch.cosh(hw / (2 * 208.5)) / torch.sinh(hw / (2 * 208.5))))
    fwhm_nm_final = (fwhm_cm1 * (lam_final ** 2) / 1.0e7).item()

    # 启发式 host 建议: 根据 Dq/B 比定 host_family
    Dq_v, B_v = Dq.item(), B.item()
    DqB = Dq_v / B_v
    if DqB > 2.6:
        host_family = "强场 (尖晶石/钙钛矿, e.g. ZnGa2O4, MgGa2O4, BaSnO3)"
    elif DqB > 2.3:
        host_family = "中场 (石榴石/橄榄石, e.g. Y3Al5O12, Gd3Ga5O12, Mg2SiO4)"
    elif DqB > 2.0:
        host_family = "弱中场 (镓酸盐/铟酸盐, e.g. LaGaO3, La3Ga5SiO14)"
    else:
        host_family = "弱场 (氟化物/卤化物, e.g. LiYF4, K2NaScF6)"

    return {
        "target_lambda_nm": float(target_lambda_nm),
        "predicted_lambda_em_nm": round(lam_final, 1),
        "predicted_fwhm_nm": round(fwhm_nm_final, 1),
        "lambda_error_nm": round(abs(lam_final - target_lambda_nm), 2),
        "Dq_cm1": round(Dq_v, 1),
        "B_cm1": round(B_v, 1),
        "C_cm1": round(C.item(), 1),
        "S_huang_rhys": round(S.item(), 2),
        "hbar_omega_cm1": round(hw.item(), 1),
        "Dq_over_B": round(DqB, 3),
        "n_iter": len(loss_hist),
        "converged": converged,
        "loss_final": round(loss_hist[-1], 4) if loss_hist else None,
        "host_family_hint": host_family,
        "method": "differentiable_TS_inverse_via_torch.autograd",
        "physical_bounds_used": PHYS_BOUNDS,
    }


if __name__ == "__main__":
    # 测试 NIR 三档
    for target in (700, 800, 900, 1100):
        res = inverse_design(target, n_steps=1500, verbose=False)
        print(f"target {target}nm -> 预测 {res['predicted_lambda_em_nm']}nm "
              f"(误差 {res['lambda_error_nm']}nm, {res['n_iter']} iter, "
              f"converged={res['converged']}) | Dq={res['Dq_cm1']} B={res['B_cm1']} "
              f"S={res['S_huang_rhys']} ℏω={res['hbar_omega_cm1']}")
        print(f"  -> {res['host_family_hint']}")
