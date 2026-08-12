"""vegard.py — 一阶 Vegard 定律峰位修正.

给定:
  - 类比材料的理论峰 (reported_peaks, 2θ + intensity)
  - 类比材料的化学式 analog_comp
  - 拟合成的化学式 target_comp

估算:
  - 平均阳离子半径变化 Δr̄
  - 近似晶格常数变化 Δa/a ≈ Δr̄ / r̄
  - 布拉格方程一阶: Δ2θ = -2·tan(θ)·Δa/a (弧度)

输出 perturbed_peaks (2θ + intensity, intensity 不改; 浓度补正留 M2).

**适用性**: 只在 host family 一致 (Jaccard ≥ 0.4) 时可信. 跨 family 调用返回原值 + flag.
"""
from __future__ import annotations

import math
from .formula_parser import Composition, get_shannon_radius, _DEFAULT_VALENCE


def _cation_species(elem: str) -> str:
    """`"La"` → `"La3+"` 用默认价态."""
    val = _DEFAULT_VALENCE.get(elem, 0)
    if val > 0:
        return f"{elem}{val}+"
    if val < 0:
        return f"{elem}{abs(val)}-"
    return elem


def _avg_cation_radius(comp: Composition, exclude_anion: bool = True) -> float:
    """按元素原子数加权平均阳离子半径 (CN=6). 缺失半径的元素跳过."""
    total_atoms = 0.0
    total_r = 0.0
    for elem, count in comp.elements.items():
        val = _DEFAULT_VALENCE.get(elem, 0)
        if exclude_anion and val <= 0:
            continue
        sp = _cation_species(elem)
        r = get_shannon_radius(sp, cn=6)
        if r is None:
            continue
        total_atoms += count
        total_r += count * r
    return (total_r / total_atoms) if total_atoms > 0 else 0.0


def vegard_shift_peaks(
    reported_peaks: list[dict],
    analog_comp: Composition,
    target_comp: Composition,
    max_abs_delta_deg: float = 2.0,
) -> tuple[list[dict], dict]:
    """返回 (perturbed_peaks, meta).

    meta 含 {avg_r_analog, avg_r_target, delta_a_over_a, jaccard, applied}.
    applied=False 表示 host family 不匹配, 直接返回原峰.
    """
    if not reported_peaks:
        return [], {"applied": False, "reason": "no_reference_peaks"}

    jac = analog_comp.jaccard(target_comp)
    r_analog = _avg_cation_radius(analog_comp)
    r_target = _avg_cation_radius(target_comp)

    meta = {
        "jaccard": round(jac, 3),
        "avg_r_analog": round(r_analog, 4),
        "avg_r_target": round(r_target, 4),
    }

    if jac < 0.4 or r_analog <= 0 or r_target <= 0:
        meta.update({"applied": False, "reason": f"host_family_mismatch (J={jac:.2f})"})
        return list(reported_peaks), meta

    da_over_a = (r_target - r_analog) / r_analog
    meta["delta_a_over_a"] = round(da_over_a, 5)

    perturbed = []
    for p in reported_peaks:
        two_theta = float(p.get("two_theta", p.get("position", 0)))
        if two_theta <= 0:
            continue
        theta_rad = math.radians(two_theta / 2.0)
        delta_2theta = math.degrees(-2.0 * math.tan(theta_rad) * da_over_a)
        # 限幅 (防 extreme substitution 导致非物理位移)
        delta_2theta = max(-max_abs_delta_deg, min(max_abs_delta_deg, delta_2theta))
        new_2theta = two_theta + delta_2theta
        perturbed.append({
            "two_theta": round(new_2theta, 3),
            "intensity": float(p.get("intensity", 1.0)),
            "original_2theta": two_theta,
            "delta": round(delta_2theta, 3),
            "hkl": p.get("hkl"),
        })
    meta["applied"] = True
    meta["n_peaks"] = len(perturbed)
    return perturbed, meta
