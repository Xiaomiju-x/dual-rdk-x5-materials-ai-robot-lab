"""failure_flags.py — 拒绝经验规则 / 警告生成.

输入 (target_comp, analog_comp, dopant_info) → 返回 [{level, code, message, recommendation}]
level ∈ {"info", "warn", "error"}. error 级别会强制 verdict ≤ REVISE.
"""
from __future__ import annotations

from .formula_parser import Composition, get_shannon_radius, _DEFAULT_VALENCE


def _cation_species(elem: str) -> str:
    val = _DEFAULT_VALENCE.get(elem, 0)
    if val > 0:
        return f"{elem}{val}+"
    return elem


def compute_failure_flags(
    target: Composition,
    analog: Composition | None,
    dopant: dict,
    radii_tolerance_pct: float = 20.0,
    charge_tolerance: float = 0.5,
) -> list[dict]:
    flags: list[dict] = []

    # 1. 电荷不平衡
    if target.charge_balance is not None and abs(target.charge_balance) > charge_tolerance:
        flags.append({
            "level": "error",
            "code": "charge_imbalance",
            "message": f"目标化学式电荷不平衡: net={target.charge_balance:+.2f} "
                       f"(容差 ±{charge_tolerance})",
            "recommendation": "检查下标是否正确, 或考虑多价态/空位补偿",
        })

    # 2. Jaccard 过低 → 没有可靠类比
    if analog is not None:
        jac = target.jaccard(analog)
        if jac < 0.4:
            flags.append({
                "level": "warn",
                "code": "low_analog_similarity",
                "message": f"最近类比 {analog.formula} 元素重合度 Jaccard={jac:.2f} < 0.4",
                "recommendation": "考虑先做一个小样本的探索性合成, 预测置信度不会高",
            })

    # 3. 掺杂半径失配 (相对于拟替代位)
    dop_elem = dopant.get("element")
    dop_site = dopant.get("site")
    dop_val = dopant.get("valence") or 3
    if dop_elem and dop_site:
        r_dop = get_shannon_radius(_cation_species(dop_elem), cn=6)
        r_site = get_shannon_radius(_cation_species(dop_site), cn=6)
        if r_dop and r_site:
            mismatch_pct = (r_dop - r_site) / r_site * 100
            if abs(mismatch_pct) > radii_tolerance_pct:
                flags.append({
                    "level": "error",
                    "code": "dopant_radius_mismatch",
                    "message": f"{dop_elem}3+→{dop_site} 半径失配 {mismatch_pct:+.1f}% "
                               f"(r={r_dop:.3f} vs {r_site:.3f} Å, 容差 ±{radii_tolerance_pct}%)",
                    "recommendation": f"考虑换位点 (如 {dop_site} 以外) 或换更接近半径的掺杂离子",
                })
            elif abs(mismatch_pct) > radii_tolerance_pct * 0.6:
                flags.append({
                    "level": "warn",
                    "code": "dopant_radius_warning",
                    "message": f"{dop_elem}3+→{dop_site} 半径失配 {mismatch_pct:+.1f}% "
                               f"(接近阈值), 可能伴随 minor 杂相",
                    "recommendation": "烧结温度可能需要提 +50°C 延长 2h",
                })

    # 3b. 价态失配 (Cr3+ 放 Zn2+/Si4+ 位需电荷补偿)
    if dop_elem and dop_site:
        site_val = _DEFAULT_VALENCE.get(dop_site, 0)
        if site_val > 0 and dop_val:
            delta_v = dop_val - site_val
            if abs(delta_v) >= 2:
                flags.append({
                    "level": "error",
                    "code": "dopant_valence_mismatch",
                    "message": f"{dop_elem}{dop_val}+→{dop_site}{site_val}+ "
                               f"价态差 {delta_v:+d}, 电荷极度失衡 (>1)",
                    "recommendation": "必须共掺补偿离子 (如 Na+/Ba2+/Ca2+), 或换同价位点",
                })
            elif abs(delta_v) == 1:
                flags.append({
                    "level": "warn",
                    "code": "dopant_valence_warning",
                    "message": f"{dop_elem}{dop_val}+→{dop_site}{site_val}+ "
                               f"价态差 {delta_v:+d}, 需 V_O 或阳离子空位补偿",
                    "recommendation": "实测常伴随 minor 杂相 (固溶度低), "
                                     "建议优先选同价位点 (例如 Cr3+ 选 Al/Ga/Sc/Fe)",
                })

    # 4. 掺杂浓度经验范围
    dop_pct = dopant.get("pct", 0) or 0
    try:
        dop_pct = float(dop_pct)
    except (TypeError, ValueError):
        dop_pct = 0
    if dop_pct > 0:
        if dop_pct > 5.0:
            flags.append({
                "level": "error",
                "code": "dopant_too_high",
                "message": f"掺杂浓度 {dop_pct}% 超出常规 NIR 荧光粉范围 (≤5%)",
                "recommendation": "浓度淬灭导致发射强度崩塌, 建议降到 0.5-2%",
            })
        elif dop_pct > 2.0:
            flags.append({
                "level": "warn",
                "code": "dopant_quenching_risk",
                "message": f"掺杂浓度 {dop_pct}% 可能进入浓度淬灭区间",
                "recommendation": "同样配方做 0.5%/1%/2% 三档浓度梯度, 取最佳",
            })

    # 5. 价态错误 (Cr2+ 罕见)
    dop_val = dopant.get("valence")
    if dop_elem == "Cr" and dop_val and dop_val != 3:
        flags.append({
            "level": "warn",
            "code": "unusual_valence",
            "message": f"Cr{dop_val}+ 在氧化物中罕见, 通常为 Cr3+",
            "recommendation": "确认原料是 Cr2O3 (提供 Cr3+)",
        })

    return flags


def severity(flags: list[dict]) -> str:
    """Return highest severity level among flags."""
    levels = [f.get("level") for f in flags]
    if "error" in levels:
        return "error"
    if "warn" in levels:
        return "warn"
    if "info" in levels:
        return "info"
    return "ok"
