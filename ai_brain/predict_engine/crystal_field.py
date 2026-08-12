"""crystal_field.py — Tanabe-Sugano d3 (Cr3+) 对角化 + Huang-Rhys 电声耦合.

替换原先经验斜率 "-10 nm per 0.01 Å" (predict_engine/virtual_spectra.py
_crystal_field_shift_nm) 的硬核光学模型.

参考:
  - Sugano, Tanabe, Kamimura. "Multiplets of Transition-Metal Ions in Crystals".
    Academic Press, 1970. 附录 II 表 27.a (d3 Oh 子矩阵).
  - NMaso/tsplot (github) — 数值参考实现.
  - Alkauskas et al. "First-principles theory of the luminescence lineshape
    for the triplet transition in diamond NV centres",
    New J. Phys. 16, 073026 (2014). doi:10.1088/1367-2630/16/7/073026
    (Huang-Rhys 经典高斯展宽).

单位约定:
  - Dq, B, C, hbar_omega 均用 cm^-1.
  - 半径用 Å.
  - 温度用 K.
  - 波长用 nm.
"""
from __future__ import annotations

import math
import numpy as np


# ============ d3 (Cr3+) Tanabe-Sugano Hamilton 矩阵 ============
# 基准: 4A2(t2^3) 是非简并基态, 作为能量零点.
# 关注最低 5 个 quartet/doublet 能级:
#   4A2  (t2^3)                        → E = -12 Dq (Racah), 本实现设为 0 参考.
#   4T2  (t2^2 e)                      → 离开 4A2 10 Dq, 含电子斥力修正.
#   4T1a (t2^2 e, 较低)                → 对角项 ~ 18 Dq - 12 B (近似)
#   4T1b (t2 e^2, 较高)                → 对角项 ~ 28 Dq - 12 B
#   2E   (t2^3, 自旋翻)                → 9 B + 3 C
#   2T1  (t2^3)                        → 9 B + 3 C (与 2E 近)
#
# Sugano 教材表 27.a d3 的 4T1 子矩阵是 2x2 (a/b 耦合), 本文显式展开.
# 4T2 和 4A2 在选定基下是对角 (无非对角混合 for pure t2-e 分块).


def _d3_hamiltonian(Dq: float, B: float, C: float) -> np.ndarray:
    """构造 6x6 d3 Hamilton (Sugano-Tanabe-Kamimura 1970, 附录 II).

    基态序 (对角):
      [0] 4A2  (t2^3)
      [1] 4T2  (t2^2 e)
      [2] 4T1a (t2^2 e)      ← 与 [3] 有非对角 12 B
      [3] 4T1b (t2 e^2)
      [4] 2E   (t2^3)
      [5] 2T1  (t2^3)

    注: 2E 和 2T1 和 4A2 同是 t2^3 组态, Dq 不进入, 只受 B/C 影响.
    """
    H = np.zeros((6, 6), dtype=np.float64)

    # 4A2(t2^3): 取为能量零点 E = 0
    H[0, 0] = 0.0

    # 4T2(t2^2 e): 离开 4A2 = 10 Dq (Racah 形式: -2B 修正忽略, 主项)
    H[1, 1] = 10.0 * Dq

    # 4T1a / 4T1b — Henderson-Imbusch (1989) "Optical Spectroscopy of Inorganic Solids":
    #   对角: 4T1a (t2^2 e) = 10 Dq + 15 B
    #         4T1b (t2 e^2) = 20 Dq
    #   非对角: <4T1a|H|4T1b> = 2√3 B (≈ 3.464 B)
    # 等价 Lever (1984) 闭式: E_± = (15Dq+7.5B) ± √((7.5B-5Dq)^2 + 12B^2)
    # 对 YAG (Dq=1640,B=650): E_低 ≈ 25460 cm⁻¹ → 393 nm (与实测 Cr:YAG 4T1a 440 nm 差 ~10%)
    H[2, 2] = 10.0 * Dq + 15.0 * B
    H[3, 3] = 20.0 * Dq
    H[2, 3] = 2.0 * math.sqrt(3.0) * B
    H[3, 2] = H[2, 3]

    # 2E (t2^3): 9 B + 3 C (STK)
    H[4, 4] = 9.0 * B + 3.0 * C
    # 2T1 (t2^3): 9 B + 3 C (近似, 在 Oh 纯 t2^3 下两者退化)
    H[5, 5] = 9.0 * B + 3.0 * C

    return H


def d3_ts_eigenvalues(
    Dq_cm1: float,
    B_cm1: float,
    C_cm1: float,
) -> dict:
    """d3 Oh (Cr3+) Tanabe-Sugano 对角化.

    Args:
      Dq_cm1: 10 Dq / 10 晶场强度 (实际输入 Dq, 10 Dq = 10 * Dq_cm1).
      B_cm1:  Racah 参数 B.
      C_cm1:  Racah 参数 C.

    Returns:
      dict with keys:
        4A2_cm1, 4T2_cm1, 4T1a_cm1, 4T1b_cm1, 2E_cm1, 2T1_cm1
        lambda_em_nm:  最低 4T → 4A2 跃迁的发射波长 (nm)
        Dq_over_B:     Dq / B
        eigenvalues_cm1: [E0,...,E5] (升序)
    """
    H = _d3_hamiltonian(Dq_cm1, B_cm1, C_cm1)
    # eigvalsh: 对称矩阵升序特征值
    eigs = np.linalg.eigvalsh(H)

    # 对角项索引 (混合 4T1a/4T1b 后, 两个根是 4T1 的低/高, 其他是纯态)
    # 为了稳定标签, 再解一次完整 eig 便于匹配:
    vals, vecs = np.linalg.eigh(H)
    # 按"主分量"给每个特征值打标签:
    labels = ["4A2", "4T2", "4T1a", "4T1b", "2E", "2T1"]
    assigned = {}
    used = set()
    for i, v in enumerate(vals):
        # 取最大 |v_k|^2 的 k
        k = int(np.argmax(np.abs(vecs[:, i]) ** 2))
        # 如果 k 的标签已被占, 就退到次大
        if labels[k] in used:
            order = np.argsort(-np.abs(vecs[:, i]) ** 2)
            for kk in order:
                if labels[int(kk)] not in used:
                    k = int(kk)
                    break
        assigned[labels[k]] = float(v)
        used.add(labels[k])

    E_4A2 = assigned.get("4A2", 0.0)
    E_4T2 = assigned.get("4T2", 10.0 * Dq_cm1)
    E_4T1a = assigned.get("4T1a", 18.0 * Dq_cm1 - 12.0 * B_cm1)
    E_4T1b = assigned.get("4T1b", 28.0 * Dq_cm1 - 12.0 * B_cm1)
    E_2E = assigned.get("2E", 9.0 * B_cm1 + 3.0 * C_cm1)
    E_2T1 = assigned.get("2T1", 9.0 * B_cm1 + 3.0 * C_cm1)

    # 判断 high-field (2E 低于 4T2, 窄 R-line) 还是 low-field (4T2 低, 宽带)
    # 最低 excited state → 4A2 即为主 emission
    excited_states = {
        "4T2": E_4T2 - E_4A2,
        "4T1a": E_4T1a - E_4A2,
        "2E": E_2E - E_4A2,
    }
    # NIR 荧光粉中 Cr3+ 绝大多数是 low-field (4T2 最低 → spin-allowed 宽带 FA ~ 700 nm)
    # 这里选 4T2 (Cr:YAG/Cr:LiSAF 系都是 low-field 主)
    E_emission = excited_states["4T2"]
    if E_emission <= 0:
        # 防护: high-field 极端, 退到 2E
        E_emission = excited_states["2E"]
    lambda_em_nm = 1.0e7 / E_emission if E_emission > 0 else float("nan")

    # 激发峰: 4A2 → 4T2 (spin-allowed, 长波), 4A2 → 4T1a (短波)
    E_ex_4T2 = E_4T2 - E_4A2
    E_ex_4T1a = E_4T1a - E_4A2
    lambda_ex_4T2 = 1.0e7 / E_ex_4T2 if E_ex_4T2 > 0 else float("nan")
    lambda_ex_4T1a = 1.0e7 / E_ex_4T1a if E_ex_4T1a > 0 else float("nan")

    return {
        "4A2_cm1": E_4A2,
        "4T2_cm1": E_4T2,
        "4T1a_cm1": E_4T1a,
        "4T1b_cm1": E_4T1b,
        "2E_cm1": E_2E,
        "2T1_cm1": E_2T1,
        "lambda_em_nm": lambda_em_nm,
        "lambda_ex_4T2_nm": lambda_ex_4T2,
        "lambda_ex_4T1a_nm": lambda_ex_4T1a,
        "Dq_over_B": Dq_cm1 / B_cm1 if B_cm1 > 0 else float("nan"),
        "eigenvalues_cm1": eigs.tolist(),
    }


# ============ Huang-Rhys 电声耦合展宽 ============
# FWHM(T) = 2.36 · sqrt(S) · hbar_omega · sqrt(coth(hbar_omega / (2 k T)))
# 2.36 = 2 sqrt(2 ln 2) (高斯 FWHM 因子)
# S: Huang-Rhys 因子 (无量纲, Cr3+ 在氧化物中典型 5-7)
# hbar_omega: 有效声子能量 (cm^-1)
# 参考: Alkauskas 2014 NJP, Cr:YAG/Cr:GGG 经验值来自 Kück 2001.

_CM1_PER_K = 0.6950348  # k_B in cm^-1 / K (1 K = 0.695 cm^-1)


def huang_rhys_fwhm_nm(
    lambda_em_nm: float,
    S_huang_rhys: float,
    hbar_omega_cm1: float,
    T_kelvin: float = 300.0,
) -> float:
    """Huang-Rhys 经典高斯 FWHM (带温度依赖 coth 项).

    先算 FWHM_cm1 = 2.36 · sqrt(S) · hbar_omega · sqrt(coth(hbar_omega / 2kT))
    再转 nm: Δλ = λ^2 / (hc) · ΔE, 其中 λ in nm, ΔE in cm^-1, 1 cm^-1 ≈ 波数.
    简化: Δλ(nm) = λ(nm)^2 · ΔE(cm^-1) · 1e-7
          (因为 E(cm^-1) = 1e7 / λ(nm), dE/dλ = -1e7/λ^2,
           |Δλ| = λ^2 · ΔE / 1e7 = λ^2 · ΔE · 1e-7)
    """
    if S_huang_rhys <= 0 or hbar_omega_cm1 <= 0 or lambda_em_nm <= 0:
        return float("nan")

    kT_cm1 = _CM1_PER_K * T_kelvin
    # coth(x) = 1/tanh(x); 避免 hbar_omega/(2kT) 过小 (高温极限 coth ≈ 1/x)
    x = hbar_omega_cm1 / (2.0 * kT_cm1) if kT_cm1 > 0 else 50.0
    # coth 防溢出
    if x > 50.0:
        coth_x = 1.0
    else:
        coth_x = 1.0 / math.tanh(x)

    fwhm_cm1 = 2.355 * math.sqrt(S_huang_rhys) * hbar_omega_cm1 * math.sqrt(coth_x)
    # 2.355 ≈ 2·sqrt(2·ln2); 题目给 2.36, 差 0.2% 可接受, 这里用更精确的 2.355
    # ΔE(cm^-1) → Δλ(nm): |Δλ| = λ^2 / 1e7 · ΔE
    fwhm_nm = (lambda_em_nm ** 2) * fwhm_cm1 * 1.0e-7
    return fwhm_nm


# ============ 热猝灭 (Struck-Fonger 模型) ============
# I(T)/I(0) = 1 / (1 + C·exp(-ΔE_a / kT))
# ΔE_a: 4T2 势能面与 4A2 交叉点活化能
# 简化估算 (Dorenbos 2005, Brik & Srivastava 2013):
#   ΔE_a ≈ (E_4T2 - 2S·ℏω)² / (4S·ℏω)  (configurational coordinate)
# C (频率因子): ~10^7 for Cr3+ in oxide (Mott-Seitz model, τ_r/τ_0)

_BOLTZMANN_CM1_PER_K = 0.6950348  # k_B in cm⁻¹/K

def thermal_quenching_estimate(
    E_4T2_cm1: float,
    S_huang_rhys: float,
    hbar_omega_cm1: float,
    T_kelvin: float = 423.0,
    T_ref: float = 298.0,
    C_freq: float = 350.0,
) -> dict:
    """Struck-Fonger 热猝灭估算 (Cr3+ calibrated).

    活化能采用 energy-gap law 简化:
      ΔE_a ≈ 0.12 × E_4T2 (calibrated to Cr3+ oxide phosphor data,
              Brik & Srivastava, J. Lumin. 2013).
    C_freq calibrated to ~350 matching I(423K)/I(298K) ≈ 70-80% for garnet hosts.

    Returns:
        dict with:
          activation_energy_cm1, activation_energy_eV,
          thermal_stability_pct (I(T)/I(T_ref) × 100),
          T50_K (I drops to 50%)
    """
    if E_4T2_cm1 <= 0 or S_huang_rhys <= 0 or hbar_omega_cm1 <= 0:
        return {"activation_energy_cm1": 0, "activation_energy_eV": 0,
                "thermal_stability_pct": 50.0, "T50_K": 300.0}

    delta_E_a = 0.12 * E_4T2_cm1
    delta_E_a_eV = delta_E_a * 1.2398e-4

    def _intensity_ratio(T):
        kT = _BOLTZMANN_CM1_PER_K * T
        if kT <= 0:
            return 1.0
        arg = delta_E_a / kT
        if arg > 200:
            return 1.0
        return 1.0 / (1.0 + C_freq * math.exp(-arg))

    I_T = _intensity_ratio(T_kelvin)
    I_ref = _intensity_ratio(T_ref)
    stability = (I_T / I_ref * 100.0) if I_ref > 0 else 0.0
    stability = min(stability, 100.0)

    ln_C = math.log(C_freq) if C_freq > 1 else 1.0
    T50 = delta_E_a / (_BOLTZMANN_CM1_PER_K * ln_C) if ln_C > 0 else 999.0

    return {
        "activation_energy_cm1": round(delta_E_a, 1),
        "activation_energy_eV": round(delta_E_a_eV, 4),
        "thermal_stability_pct": round(stability, 1),
        "T50_K": round(T50, 0),
    }


# ============ Dq 随配位体键长的 Vegard 型调整 ============
def vegard_dq_shift(
    Dq_host_cm1: float,
    r_host_ang: float,
    r_target_ang: float,
) -> float:
    """Dq ∝ 1/R^5 (点电荷晶场理论), R 为金属-配体平均键长.

    Dq_target = Dq_host · (r_host / r_target)^5

    Args:
      Dq_host_cm1: host 中 dopant site 的 Dq.
      r_host_ang:  host 原子在该 site 的离子半径 (Å).
      r_target_ang: 实际 dopant (如 Cr3+) 的离子半径 (Å).

    Returns:
      新 Dq (cm^-1). 若 dopant 比 host 原子大, Dq 变小 → λ_em 红移.
    """
    if r_host_ang <= 0 or r_target_ang <= 0:
        return Dq_host_cm1
    return Dq_host_cm1 * (r_host_ang / r_target_ang) ** 5


# ============ 自测 CLI ============
if __name__ == "__main__":
    # YAG: Cr3+ 在 Al (Oh) 位, 实测 ~688 nm
    print("=== YAG (Cr:YAG) ===")
    res = d3_ts_eigenvalues(Dq_cm1=1640, B_cm1=650, C_cm1=3000)
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k:20s} = {v:.2f}")
        else:
            print(f"  {k:20s} = {v}")
    fwhm = huang_rhys_fwhm_nm(res["lambda_em_nm"], S_huang_rhys=6, hbar_omega_cm1=400)
    print(f"  FWHM_nm              = {fwhm:.2f}")

    print("\n=== Vegard Dq shift (YAG Al→Cr) ===")
    new_dq = vegard_dq_shift(1640, 0.535, 0.615)
    print(f"  Dq_new = {new_dq:.1f} cm^-1  (Dq_old=1640)")
    res2 = d3_ts_eigenvalues(new_dq, 650, 3000)
    print(f"  λ_em  = {res2['lambda_em_nm']:.1f} nm (Vegard-corrected)")
