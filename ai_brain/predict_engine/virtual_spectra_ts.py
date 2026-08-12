"""virtual_spectra_ts.py — Tanabe-Sugano + Huang-Rhys 驱动的虚拟 PL 光谱.

替换 virtual_spectra.py::_crystal_field_shift_nm 的经验斜率 "-10 nm per 0.01 Å".
本文件**不修改** virtual_spectra.py; 供后续集成时调用.

Pipeline:
  1. 查 cft_params.json 拿 host 的 (Dq, B, C, hbar_omega, S, r_host).
  2. Vegard scaling: Dq_target = Dq_host · (r_host / r_Cr3+)^5  (1/R^5 point-charge).
     **注**: cft_params.json 里的 Dq 已是 Cr3+-in-host 实测/拟合值, 对 tabulated host
     我们 **默认不再套 Vegard** (否则会 double count 导致过度红移 > 1500 nm).
     `apply_vegard=True` 时才启用, 用于未来跨 host 外推 (e.g. 从 YAG 推 LuAG).
  3. d3_ts_eigenvalues(Dq, B, C) → E_4T2 (cm^-1).
  4. Stokes shift: E_emission = E_4T2 - S · hbar_omega  (经验单 S 位移, 匹配 Cr:YAG
     实测 688 nm; 严格 Franck-Condon 2S 会过红移).
  5. huang_rhys_fwhm_nm(lambda_em, S, hbar_omega, T) → FWHM.

仅支持 Cr3+ (d3). 其他掺杂返回 applied=False.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .crystal_field import (
    d3_ts_eigenvalues,
    huang_rhys_fwhm_nm,
    thermal_quenching_estimate,
    vegard_dq_shift,
)

_CFT_PATH = Path(__file__).parent / "cft_params.json"
with open(_CFT_PATH, encoding="utf-8") as f:
    _CFT_DB = json.load(f)

# Cr3+ octahedral Shannon radius (Å)
CR3_CN6_RADIUS_ANG = 0.615


def _normalize_dopant(dopant_elem: str) -> str | None:
    """'Cr', 'Cr3+', 'cr3+' → 统一到 'Cr3+'; 非 Cr 返回 None."""
    s = dopant_elem.strip().lower().replace(" ", "")
    if s in ("cr", "cr3+", "cr3", "cr+3"):
        return "Cr3+"
    return None


def virtual_pl_ts(
    host_name: str,
    dopant_elem: str,
    dopant_site: str,
    dopant_pct: float,
    T_kelvin: float = 300.0,
    apply_vegard: bool = False,
) -> dict:
    """Tanabe-Sugano + Huang-Rhys 虚拟 PL.

    Args:
      host_name:  'YAG', 'GGG', 'GAGG', 'Y2O3', 'SYGO' (见 cft_params.json).
      dopant_elem: 'Cr' / 'Cr3+' (唯一支持).
      dopant_site: 替代位点标签 (如 'Al'), 当前仅记录, Vegard 已用 r_host_avg.
      dopant_pct:  摩尔百分比 (当前仅记录, 真实浓度依赖效应后续 Phase 1.3 加).
      T_kelvin:    温度 (默认 300 K).

    Returns:
      dict:
        applied (bool)
        reason (str, 失败时)
        method, lambda_em_nm, fwhm_nm, Dq_cm1_host, Dq_cm1_target,
        B_cm1, S, T_kelvin, hbar_omega_cm1, source_doi, host_formula,
        ts_eigenvalues (完整 d3_ts_eigenvalues 返回)
    """
    # 1. Host 存在性检查
    if host_name not in _CFT_DB or host_name.startswith("_"):
        return {
            "applied": False,
            "reason": "host_not_in_cft_params",
            "host_name": host_name,
            "available_hosts": [k for k in _CFT_DB.keys() if not k.startswith("_")],
        }

    # 2. Dopant 支持性检查
    dop = _normalize_dopant(dopant_elem)
    if dop is None:
        return {
            "applied": False,
            "reason": "only_Cr3+_supported",
            "dopant_elem_raw": dopant_elem,
        }

    p = _CFT_DB[host_name]
    Dq_host = float(p["Dq_cm1"])
    B = float(p["B_cm1"])
    C = float(p["C_cm1"])
    hbar_omega = float(p["hbar_omega_cm1"])
    S = float(p["S_huang_rhys"])
    r_host = float(p["r_host_avg_ang"])
    source_doi = p.get("source_doi", "")

    # 3. Vegard Dq shift: 默认关闭 (tabulated Dq 已是 Cr3+-in-host 拟合值).
    #    apply_vegard=True 用于跨 host 外推.
    if apply_vegard:
        Dq_target = vegard_dq_shift(Dq_host, r_host, CR3_CN6_RADIUS_ANG)
    else:
        Dq_target = Dq_host

    # 4. 对角化
    ts = d3_ts_eigenvalues(Dq_target, B, C)
    E_4T2 = ts["4T2_cm1"]

    # 5. Stokes shift (empirical S·ħω single-shift, matches Cr:YAG 688 nm)
    E_em_cm1 = E_4T2 - S * hbar_omega
    if E_em_cm1 <= 0:
        return {
            "applied": False,
            "reason": "negative_emission_energy_after_stokes",
            "E_4T2_cm1": E_4T2,
            "stokes_cm1": S * hbar_omega,
        }
    lambda_em_nm = 1.0e7 / E_em_cm1

    # 6. FWHM
    fwhm_nm = huang_rhys_fwhm_nm(lambda_em_nm, S, hbar_omega, T_kelvin)

    # 7. 激发峰 (from TS eigenvalues)
    excitation_peaks_nm = []
    for key in ("lambda_ex_4T2_nm", "lambda_ex_4T1a_nm"):
        v = ts.get(key)
        if v and v == v and v > 0:  # not NaN
            excitation_peaks_nm.append(round(v, 1))

    # 8. 热稳定性 (Struck-Fonger)
    tq = thermal_quenching_estimate(E_4T2, S, hbar_omega, T_kelvin=423.0, T_ref=298.0)

    return {
        "applied": True,
        "method": "tanabe_sugano_huang_rhys",
        "host_name": host_name,
        "host_formula": p.get("formula", ""),
        "dopant": dop,
        "dopant_site": dopant_site,
        "dopant_pct": dopant_pct,
        "lambda_em_nm": round(lambda_em_nm, 2),
        "fwhm_nm": round(fwhm_nm, 2),
        "excitation_peaks_nm": excitation_peaks_nm,
        "thermal_stability_pct_423K": tq["thermal_stability_pct"],
        "thermal_activation_energy_eV": tq["activation_energy_eV"],
        "T50_K": tq["T50_K"],
        "Dq_cm1_host": Dq_host,
        "Dq_cm1_target": round(Dq_target, 2),
        "B_cm1": B,
        "C_cm1": C,
        "hbar_omega_cm1": hbar_omega,
        "S": S,
        "stokes_shift_cm1": round(S * hbar_omega, 2),
        "r_host_ang": r_host,
        "r_target_ang": CR3_CN6_RADIUS_ANG,
        "T_kelvin": T_kelvin,
        "source_doi": source_doi,
        "ts_eigenvalues": ts,
    }


# ============ CLI ============
def _main():
    if len(sys.argv) < 2:
        # 默认 demo: 跑 5 host
        print("Usage: python -m predict_engine.virtual_spectra_ts <HOST> [dopant] [site] [pct] [T]")
        print("Demo across all hosts:\n")
        for host in ["YAG", "GGG", "GAGG", "Y2O3", "SYGO"]:
            r = virtual_pl_ts(host, "Cr3+", "Al", 1.0)
            if r.get("applied"):
                print(f"  {host:6s} | λ_em={r['lambda_em_nm']:.1f} nm | "
                      f"FWHM={r['fwhm_nm']:.1f} nm | "
                      f"Dq_host={r['Dq_cm1_host']}→Dq_tgt={r['Dq_cm1_target']:.0f} | "
                      f"source={r['source_doi']}")
            else:
                print(f"  {host:6s} | NOT APPLIED: {r.get('reason')}")
        return

    host = sys.argv[1]
    dop = sys.argv[2] if len(sys.argv) > 2 else "Cr3+"
    site = sys.argv[3] if len(sys.argv) > 3 else "Al"
    pct = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    T = float(sys.argv[5]) if len(sys.argv) > 5 else 300.0
    r = virtual_pl_ts(host, dop, site, pct, T)
    import json as _json
    print(_json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
