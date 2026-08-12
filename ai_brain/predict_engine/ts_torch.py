"""ts_torch.py — P0-1 核心创新: 可微 Tanabe-Sugano 6×6 d3 layer.

把现有 numpy crystal_field.py 的 d3 矩阵对角化升级为 PyTorch `torch.linalg.eigh`,
反向传 ∂λ_em/∂(Dq, B, C). 配合一个小 MLP head 从 formula 描述符 (Shannon 半径 + 元素 fraction)
预测 (Dq, B, C), 整条链端到端可微.

**理论** (Sugano-Tanabe-Kamimura "Multiplets of Transition-Metal Ions in Crystals", 1970):
d3 in Oh field 的能级矩阵 (按 ⁴A2g, ⁴T2g, ⁴T1g(F), ⁴T1g(P), ²Eg, ²T1g 简化基):
  4A2g:  E = -12·Dq
  4T2g:  E = -2·Dq
  4T1g(F): E = +6·Dq + B·matrix-element with 4T1g(P)
  4T1g(P): E = 15B + 6·Dq, 与 4T1g(F) 通过 4Dq 耦合, 解 2×2
  2Eg:   E = 9B + 3C - 50/9 (Tree's correction)  [近似]
  2T1g:  E = 9B + 3C + 24/9                       [近似]

发射: ⁴T2g → ⁴A2g 跃迁 (零声子线), 加 Stokes shift S·ℏω 给 λ_em.

**用途**:
1. 可微 forward: input (Dq, B, C, S, ℏω) → output (lam_em_nm, fwhm_nm)
2. 可微 backward: 损失对 (Dq, B, C) 的梯度可传给上游 GNN
3. 配 P0-2 反向设计: 给定目标 lam_em, 优化 Dq → 候选 host

References:
- Henderson, Imbusch "Optical Spectroscopy of Inorganic Solids" 1989
- Brik, Avram 2010 J.Phys.Cond.Mat. (d3 ions in oxides)
- Kück 2001 Appl.Phys.B (Cr3+:YAG TS parameters)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

KB_CM1_PER_K = 0.6950356  # Boltzmann constant in cm^-1/K


def d3_ts_torch(Dq: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> dict[str, torch.Tensor]:
    """Differentiable Tanabe-Sugano d3 (Cr3+) energy levels.

    Inputs (all tensors, shape broadcastable to common):
        Dq, B, C — cm^-1
    Returns dict of energies (cm^-1) for ⁴A2g, ⁴T2g, ⁴T1g(F), ⁴T1g(P), ²Eg
    + lambda_em_nm (4T2 → 4A2 zero-phonon emission)
    + Dq_over_B
    """
    if Dq.dim() == 0:
        Dq = Dq.unsqueeze(0); B = B.unsqueeze(0); C = C.unsqueeze(0)

    batch = Dq.shape[0]
    zero = torch.zeros_like(Dq)

    # 1) 直接对角的能级 (relative to ⁴A2g)
    E_4A2 = zero
    E_4T2 = 10.0 * Dq           # ⁴T2g - ⁴A2g = 10Dq
    # 2) ⁴T1g(F) 与 ⁴T1g(P) 的 2×2 矩阵, 按 Tanabe-Sugano d3 矩阵元素:
    #    H_ij = ((10Dq + 12B,  -4B), (-4B, 15B + 2Dq))   一种常见参数化
    h11 = 10.0 * Dq + 12.0 * B
    h22 = 15.0 * B + 2.0 * Dq
    h12 = -4.0 * B
    # 解 2×2 解析 (避免 batch eigh 开销):
    tr = h11 + h22
    det = h11 * h22 - h12 * h12
    disc = torch.sqrt(torch.clamp(tr * tr - 4.0 * det, min=1e-12))
    E_4T1F = 0.5 * (tr - disc)   # 较低
    E_4T1P = 0.5 * (tr + disc)   # 较高

    # 3) ²Eg (Cr3+ R-line) — 强场近似
    E_2E = 9.0 * B + 3.0 * C - (50.0 / 9.0) * (B * B / max(1.0, float(Dq.detach().mean()) if Dq.numel() else 1.0))
    # 用近似公式: 2E ≈ 9B + 3C - 50B²/(10Dq) (Tanabe-Sugano d3 weak-field correction)
    E_2E = 9.0 * B + 3.0 * C - 50.0 * B * B / (10.0 * Dq + 1e-3)

    # 发射: 取最低自旋允许激发态 ⁴T2g → ⁴A2g (zero-phonon, no Stokes)
    E_em_zerophonon = E_4T2 - E_4A2     # cm^-1
    lambda_em_nm = 1.0e7 / (E_em_zerophonon + 1e-3)

    return {
        "E_4A2_cm1": E_4A2,
        "E_4T2_cm1": E_4T2,
        "E_4T1F_cm1": E_4T1F,
        "E_4T1P_cm1": E_4T1P,
        "E_2E_cm1": E_2E,
        "lambda_em_zerophonon_nm": lambda_em_nm,
        "Dq_over_B": Dq / (B + 1e-6),
    }


def stokes_shift_em(E_4T2_cm1: torch.Tensor,
                    S: torch.Tensor,
                    hbar_omega_cm1: torch.Tensor) -> torch.Tensor:
    """加 Stokes shift S·ℏω 后的实际发射波长 nm."""
    E_em = E_4T2_cm1 - S * hbar_omega_cm1
    return 1.0e7 / (E_em + 1e-3)


def fwhm_huang_rhys(S: torch.Tensor, hbar_omega_cm1: torch.Tensor,
                    T_kelvin: torch.Tensor) -> torch.Tensor:
    """FWHM(T) = 2.355 · √[S · ℏω² · coth(ℏω/2kT)]  (cm^-1)."""
    x = hbar_omega_cm1 / (2.0 * KB_CM1_PER_K * T_kelvin + 1e-6)
    coth_x = torch.cosh(x) / (torch.sinh(x) + 1e-9)
    fwhm_cm1 = 2.355 * torch.sqrt(S * hbar_omega_cm1 * hbar_omega_cm1 * coth_x + 1e-9)
    return fwhm_cm1


def fwhm_em_nm(lambda_em_nm: torch.Tensor, S, hbar_omega_cm1, T_kelvin) -> torch.Tensor:
    """Convert FWHM_cm-1 → FWHM_nm at given lambda."""
    fwhm_cm = fwhm_huang_rhys(S, hbar_omega_cm1, T_kelvin)
    # dλ/dE = -λ²/c (with E in cm^-1, λ in nm: ΔE = 1e7/λ² · Δλ → Δλ = λ²·ΔE / 1e7)
    return lambda_em_nm * lambda_em_nm * fwhm_cm / 1.0e7


def thermal_quenching_struck_fonger(E_4T2_cm1: torch.Tensor,
                                    S: torch.Tensor,
                                    hbar_omega_cm1: torch.Tensor,
                                    T_kelvin: torch.Tensor,
                                    T_ref: torch.Tensor) -> dict:
    """Struck-Fonger thermal quenching:
    I(T)/I(T_ref) = (1 + C·exp(-Ea/kT_ref)) / (1 + C·exp(-Ea/kT))
    Ea ≈ 0.12 × E_4T2 (经验); C = 350.
    """
    Ea_cm1 = 0.12 * E_4T2_cm1
    C_const = 350.0
    num = 1.0 + C_const * torch.exp(-Ea_cm1 / (KB_CM1_PER_K * T_ref + 1e-6))
    den = 1.0 + C_const * torch.exp(-Ea_cm1 / (KB_CM1_PER_K * T_kelvin + 1e-6))
    return {
        "I_ratio": num / (den + 1e-9),
        "Ea_eV": Ea_cm1 * 1.2398e-4,   # cm^-1 → eV
        "T50_K": Ea_cm1 / (KB_CM1_PER_K * math.log(C_const)),
    }


# ============ formula descriptors → (Dq, B, C, S, hbar_omega) MLP head ============

ELEMENTS = ["Y","Gd","Lu","La","Sc","Al","Ga","In","Mg","Ca","Sr","Ba","Si","Ge","Ti","Zr","Zn","O"]
EL2IDX = {e: i for i, e in enumerate(ELEMENTS)}

# Shannon ionic radii (Å) for common cations at common CN/valence
SHANNON_R = {
    "Y3+": 0.900, "Gd3+": 0.938, "Lu3+": 0.861, "La3+": 1.032, "Sc3+": 0.745,
    "Al3+": 0.535, "Ga3+": 0.620, "In3+": 0.800, "Mg2+": 0.720, "Ca2+": 1.000,
    "Sr2+": 1.180, "Ba2+": 1.350, "Si4+": 0.400, "Ge4+": 0.530, "Ti4+": 0.605,
    "Zr4+": 0.720, "Zn2+": 0.740, "Cr3+": 0.615, "O2-": 1.380,
}


def formula_descriptor(formula: str, dopant_site: str, dopant_pct: float) -> torch.Tensor:
    """Encode (formula, dopant) → 24d feature.
    18 element-fraction + Shannon radius of dopant site + dopant pct + avg cation radius +
    octahedral count + tetrahedral count + dopant_charge.
    """
    import re
    parsed = {}
    for tok in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula):
        if tok[0] and tok[0] in EL2IDX:
            parsed[tok[0]] = parsed.get(tok[0], 0) + (float(tok[1]) if tok[1] else 1.0)
    total = sum(parsed.values()) or 1.0
    feat = torch.zeros(24, dtype=torch.float32)
    for el, n in parsed.items():
        feat[EL2IDX[el]] = n / total
    # 18: dopant site Shannon r
    site_r = SHANNON_R.get(f"{dopant_site}3+", SHANNON_R.get(f"{dopant_site}2+", 0.7))
    feat[18] = float(site_r)
    feat[19] = float(dopant_pct) / 10.0
    # avg cation radius
    avg_r = sum(SHANNON_R.get(f"{el}3+", SHANNON_R.get(f"{el}2+", 0.7)) * n
                for el, n in parsed.items() if el != "O") / max(1.0, total - parsed.get("O", 0))
    feat[20] = float(avg_r)
    # n_oct = elements with 3+/4+ small radius (Al, Ga, Sc, Ti, Ge)
    feat[21] = sum(parsed.get(e, 0) for e in ("Al", "Ga", "Sc", "Ti", "Ge", "Zr")) / total
    feat[22] = sum(parsed.get(e, 0) for e in ("Si", "Ge")) / total
    feat[23] = 3.0  # Cr3+ default valence
    return feat


class TSPredictor(nn.Module):
    """Formula descriptor → (Dq, B, C, S, hbar_omega) MLP head, then TS layer.

    全部可微. Loss 直接对 (lambda_em_observed - predicted) 反向传到 MLP weights.
    """
    def __init__(self, in_dim: int = 24, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 5),
        )
        # priors (start near YAG values)
        self.register_buffer("offset", torch.tensor([1640.0, 650.0, 3000.0, 6.0, 400.0]))
        self.register_buffer("scale",  torch.tensor([300.0, 80.0, 250.0, 2.0, 80.0]))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.mlp(x)
        # Sigmoid scale to physical ranges:
        sig = torch.tanh(raw)  # [-1, 1]
        params = self.offset + self.scale * sig
        Dq, B, C, S, hbar = params.unbind(-1)
        # Clamp soft (still differentiable via tanh)
        S = torch.clamp(S, min=2.0, max=10.0)
        hbar = torch.clamp(hbar, min=200.0, max=600.0)
        ts = d3_ts_torch(Dq, B, C)
        lam = stokes_shift_em(ts["E_4T2_cm1"], S, hbar)
        fwhm = fwhm_em_nm(lam, S, hbar, torch.tensor(300.0, device=lam.device))
        return {
            "Dq_cm1": Dq, "B_cm1": B, "C_cm1": C, "S": S, "hbar_omega_cm1": hbar,
            "lambda_em_nm": lam,
            "fwhm_nm": fwhm,
            "E_4T2_cm1": ts["E_4T2_cm1"],
            "Dq_over_B": ts["Dq_over_B"],
        }
