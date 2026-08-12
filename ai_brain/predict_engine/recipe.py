"""recipe.py — 自动生成称量单 (Phase 3.1)

输入: 化学式 + dopant + 目标质量 (e.g. 2g)
输出: 原料清单 (mass_g, vendor, sku, safety) + 烧结 step + 安全提示

依赖: predict_engine/raw_materials.json + formula_parser
不依赖: pymatgen (X5 上不装) — 全 Python 计算分子量

## 用法
```python
from predict_engine.recipe import generate_recipe
r = generate_recipe("Y3Al5O12", {"element":"Cr3+","site":"Al","pct":1.0}, target_mass_g=2.0)
print(r["raw_materials"])  # [{name, mass_g, ...}, ...]
print(r["sinter_steps"])    # [{action, temp_C, hours, ...}, ...]
print(r["safety_notes"])
```

## CSV 导出
recipe_to_csv(recipe) → 每行: 原料 / 克重 / 纯度 / 安全
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from .formula_parser import parse_formula

_RAW_PATH = Path(__file__).parent / "raw_materials.json"
with open(_RAW_PATH, encoding="utf-8") as f:
    _RAW_DB = json.load(f)

_SINTER_PROFILES_PATH = Path(__file__).parent / "sintering_profiles.json"
try:
    with open(_SINTER_PROFILES_PATH, encoding="utf-8") as f:
        _SINTER_PROFILES = json.load(f)
except FileNotFoundError:
    _SINTER_PROFILES = {}


def _pick_profile(formula: str, host_family: str | None) -> tuple[str, dict]:
    """Return (family_key, profile_dict). Fall back 顺序: host_family hit → formula
    suffix 启发 → default_unknown."""
    fam = (host_family or "").lower()
    if "garnet" in fam and ("Ga" in formula):
        key = "garnet_gallium"
    elif "garnet" in fam:
        key = "garnet"
    elif "perovskite" in fam:
        key = "perovskite"
    elif "spinel" in fam:
        key = "spinel"
    elif "orthosilicate" in fam or "silicate" in fam:
        key = "orthosilicate"
    elif "pyrochlore" in fam:
        key = "pyrochlore"
    elif "aluminate" in fam or "SYGO" in fam.upper():
        key = "aluminate"
    elif fam in ("oxide", "oxide_simple", "sesquioxide"):
        key = "oxide_simple"
    elif "borate" in fam:
        key = "borate"
    elif "phosphate" in fam:
        key = "phosphate"
    elif "germanate" in fam:
        key = "germanate"
    elif "fluoride" in fam:
        key = "fluoride"
    else:
        # fallback by formula suffix
        if formula.endswith("O12") and "Ga" in formula:
            key = "garnet_gallium"
        elif formula.endswith("O12"):
            key = "garnet"
        elif formula.endswith("O3") and not formula.endswith("CO3"):
            key = "perovskite"
        elif formula.endswith("O4") and "P" not in formula:
            key = "spinel"
        elif formula.endswith("O4") and "Si" in formula:
            key = "orthosilicate"
        elif formula.endswith("O3") and ("B" in formula or "B2O3" in formula):
            key = "borate"
        else:
            key = "default_unknown"
    return key, _SINTER_PROFILES.get(key, {})


# 元素 → 默认原料 (优先氧化物, 大部分合成路径都用)
ELEM_TO_RAW = {
    "Y": "Y2O3", "Gd": "Gd2O3", "Lu": "Lu2O3", "Sc": "Sc2O3", "La": "La2O3",
    "Al": "Al2O3", "Ga": "Ga2O3",
    "Mg": "MgO", "Ca": "CaCO3", "Sr": "SrCO3", "Ba": "BaCO3",
    "Zn": "ZnO", "Ge": "GeO2", "Si": "SiO2", "Ti": "TiO2", "Nb": "Nb2O5",
    "Cr": "Cr2O3", "Fe": "Fe2O3", "Ni": "NiO", "Mn": "MnO2", "In": "In2O3",
    "O": None,  # 不算
}

# 简单原子量 (X5 不装 pymatgen, 用本地表)
ATOMIC_WEIGHTS = {
    "H": 1.008, "Li": 6.941, "Be": 9.012, "B": 10.811, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086,
    "P": 30.974, "S": 32.065, "Cl": 35.453, "K": 39.098, "Ca": 40.078, "Sc": 44.956,
    "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933,
    "Ni": 58.693, "Cu": 63.546, "Zn": 65.380, "Ga": 69.723, "Ge": 72.640, "As": 74.922,
    "Br": 79.904, "Sr": 87.620, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.940,
    "In": 114.818, "Sn": 118.710, "Cs": 132.905, "Ba": 137.327, "La": 138.905,
    "Ce": 140.116, "Pr": 140.908, "Nd": 144.242, "Sm": 150.360, "Eu": 151.964,
    "Gd": 157.250, "Tb": 158.925, "Dy": 162.500, "Ho": 164.930, "Er": 167.259,
    "Tm": 168.934, "Yb": 173.054, "Lu": 174.967, "Hf": 178.490, "Ta": 180.948,
    "W": 183.840, "Bi": 208.980,
}


def _formula_molar_mass(formula: str) -> float:
    """简单计算化学式分子量, 不支持复杂括号嵌套."""
    comp = parse_formula(formula)
    return sum(count * ATOMIC_WEIGHTS.get(elem, 0) for elem, count in comp.elements.items())


def _moles_of_target(formula: str, target_mass_g: float) -> tuple[float, dict]:
    """返回 (n_mol_target, elements dict {elem: total atoms in 1 unit cell})."""
    comp = parse_formula(formula)
    M = _formula_molar_mass(formula)
    n_mol = target_mass_g / M
    return n_mol, comp.elements


def generate_recipe(formula: str, dopant: dict, target_mass_g: float = 2.0,
                     atmosphere: str = "air", host_family: str | None = None) -> dict:
    """主入口: 根据化学式 + 掺杂量 + 目标质量, 反推原料 + 烧结条件.

    Args:
      formula: 化学式 e.g. "Y3Al5O12"
      dopant: {"element": "Cr3+"|"Cr", "site": "Al", "pct": 1.0}
      target_mass_g: 想烧多少克 (默认 2g, 实验室常见)
      atmosphere: "air" | "H2" | "H2/N2 (5%/95%)" — 影响烧结建议

    Returns:
      {raw_materials: [...], sinter_steps: [...], safety_notes: [...],
       total_mass_g: float, total_cost_yuan: float, host_family: str}
    """
    n_mol, elements = _moles_of_target(formula, target_mass_g)

    # 1. 主原料按化学计量
    raws = []
    for elem, count_per_unit in elements.items():
        if elem == "O":
            continue
        raw_name = ELEM_TO_RAW.get(elem)
        if not raw_name:
            raws.append({"name": f"?[{elem}]", "mass_g": 0.0,
                          "warning": f"未在原料库中找到 {elem} 的默认源"})
            continue
        raw = _RAW_DB.get(raw_name)
        if not raw:
            continue
        # n_mol_raw: 每个 raw 含元素的化学式系数 (e.g. Y2O3 含 2 Y, 所以 n_Y / 2)
        n_atoms_in_raw = parse_formula(raw_name).elements.get(elem, 1)
        n_mol_raw = n_mol * count_per_unit / n_atoms_in_raw
        mass_pure = n_mol_raw * raw["molar_mass"]
        # 按纯度修正实称量
        mass_actual = mass_pure / (raw["purity_pct"] / 100.0)
        raws.append({
            "name": raw_name,
            "elem": elem,
            "molar_mass": raw["molar_mass"],
            "n_mol": round(n_mol_raw, 6),
            "mass_g": round(mass_actual, 4),
            "purity_pct": raw["purity_pct"],
            "vendor": raw["vendor"],
            "sku": raw["sku"],
            "safety": raw["safety"],
            "cost_yuan": round(mass_actual * raw["unit_price_yuan_per_g"], 2),
        })

    # 2. 掺杂原料
    dop_elem_raw = dopant.get("element", "").rstrip("0123456789+-")
    dop_pct = float(dopant.get("pct") or 0)
    dop_site = dopant.get("site")
    if dop_elem_raw and dop_pct > 0 and dop_site and dop_site in elements:
        # 替代后真实摩尔: 原 site 的 n × pct/100 转给 dopant
        n_dop_mol = n_mol * elements[dop_site] * (dop_pct / 100.0)
        dop_raw_name = ELEM_TO_RAW.get(dop_elem_raw)
        if dop_raw_name:
            dop_raw = _RAW_DB[dop_raw_name]
            n_atoms_in_dop_raw = parse_formula(dop_raw_name).elements.get(dop_elem_raw, 1)
            n_mol_dop_raw = n_dop_mol / n_atoms_in_dop_raw
            mass_pure = n_mol_dop_raw * dop_raw["molar_mass"]
            mass_actual = mass_pure / (dop_raw["purity_pct"] / 100.0)
            raws.append({
                "name": dop_raw_name,
                "elem": dop_elem_raw,
                "molar_mass": dop_raw["molar_mass"],
                "n_mol": round(n_mol_dop_raw, 6),
                "mass_g": round(mass_actual, 4),
                "purity_pct": dop_raw["purity_pct"],
                "vendor": dop_raw["vendor"],
                "sku": dop_raw["sku"],
                "safety": dop_raw["safety"],
                "cost_yuan": round(mass_actual * dop_raw["unit_price_yuan_per_g"], 2),
                "is_dopant": True,
                "dopant_pct": dop_pct,
                "dopant_site": dop_site,
            })
            # 主原料对应 site 的克重要扣除 dopant 占的部分
            for r in raws[:-1]:
                if r.get("elem") == dop_site:
                    r["mass_g"] = round(r["mass_g"] * (1 - dop_pct / 100), 4)
                    r["cost_yuan"] = round(r["mass_g"] * (_RAW_DB.get(r["name"], {}).get("unit_price_yuan_per_g") or 0), 2)
                    r["dopant_corrected"] = True

    total_mass = sum(r.get("mass_g", 0) for r in raws)
    total_cost = sum(r.get("cost_yuan", 0) for r in raws)

    # 3. 烧结 Step — 从 sintering_profiles.json 查表 (含文献源)
    family_key, prof = _pick_profile(formula, host_family)
    sinter_steps = []
    sinter_sources: list = []
    if prof:
        cal = prof.get("calcine") or {}
        sin = prof.get("sinter") or {}
        atm = sin.get("atmosphere", atmosphere)
        ramp = sin.get("ramp_C_per_min", 5)
        crucible = prof.get("crucible", "Al2O3")
        cycles = prof.get("grinding_cycles", 2)
        cool = prof.get("cooling", "随炉冷")
        sinter_steps.append({"action": "湿磨混料", "duration_min": 30,
                             "tool": "玛瑙研钵 + 无水乙醇", "notes": "至均匀膏状, 风干"})
        if cal:
            sinter_steps.append({"action": "预烧", "temp_C": cal.get("temp_C"),
                                 "hours": cal.get("hours"),
                                 "ramp_C_per_min": cal.get("ramp_C_per_min", 5),
                                 "atmosphere": cal.get("atmosphere", "air"),
                                 "crucible": crucible,
                                 "notes": "分解碳酸盐/硝酸盐, 出杂质后再研磨"})
        sinter_steps.append({"action": "中间研磨", "duration_min": 20,
                             "notes": f"共 {cycles} 轮研磨"})
        sinter_steps.append({"action": "压片", "tool": "12mm 模具", "pressure_MPa": 200})
        sinter_steps.append({"action": "烧结升温", "temp_C": sin.get("temp_C"),
                             "ramp_C_per_min": ramp, "atmosphere": atm, "crucible": crucible})
        sinter_steps.append({"action": "烧结保温", "temp_C": sin.get("temp_C"),
                             "hours": sin.get("hours"), "atmosphere": atm, "crucible": crucible})
        sinter_steps.append({"action": "降温", "ramp_C_per_min": 3, "atmosphere": atm,
                             "notes": cool})
        sinter_sources = prof.get("sources", [])
    else:
        # 兜底 (profile 未加载)
        sinter_steps = [
            {"action": "湿磨混料", "duration_min": 30, "tool": "玛瑙研钵"},
            {"action": "压片", "pressure_MPa": 200},
            {"action": "烧结保温", "temp_C": 1450, "hours": 6, "atmosphere": atmosphere},
            {"action": "降温", "ramp_C_per_min": 3, "notes": "profile 未加载, 兜底默认"},
        ]

    # 4. 安全提示
    safety_notes = []
    for r in raws:
        if r.get("safety") and ("毒" in r.get("safety", "") or "腐" in r.get("safety", "") or "致敏" in r.get("safety", "")):
            safety_notes.append(f"⚠ {r['name']}: {r['safety']}")
    if dop_elem_raw == "Cr":
        safety_notes.append("⚠ Cr2O3 通风橱内称量, 防 Cr6+ 残留")
    if atmosphere not in ("air", "Ar", "N2"):
        safety_notes.append(f"⚠ 烧结气氛 {atmosphere} — 还原气, 操作前查 SOP")
    safety_notes.append("✓ 烧后样品取出: 戴隔热手套, 用陶瓷夹子, 防止热震开裂")

    return {
        "formula": formula,
        "dopant": dopant,
        "target_mass_g": target_mass_g,
        "atmosphere": atmosphere,
        "host_family": host_family or family_key,
        "family_key_resolved": family_key,
        "raw_materials": raws,
        "sinter_steps": sinter_steps,
        "sinter_sources": sinter_sources,
        "safety_notes": safety_notes,
        "total_mass_g": round(total_mass, 4),
        "total_cost_yuan": round(total_cost, 2),
    }


def recipe_to_csv(recipe: dict) -> str:
    """转为 CSV 字符串用于 dashboard 下载."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["项目", "原料", "纯度%", "称量(g)", "供应商", "SKU", "成本(¥)", "安全"])
    for r in recipe["raw_materials"]:
        w.writerow([
            "掺杂" if r.get("is_dopant") else "主料",
            r.get("name", "?"),
            r.get("purity_pct", ""),
            r.get("mass_g", ""),
            r.get("vendor", ""),
            r.get("sku", ""),
            r.get("cost_yuan", ""),
            r.get("safety", ""),
        ])
    w.writerow([])
    w.writerow(["总质量(g)", "", "", recipe["total_mass_g"], "", "", recipe["total_cost_yuan"], ""])
    w.writerow([])
    w.writerow(["烧结步骤", "动作", "温度(°C)", "时长", "气氛", "升降温", "工具", "备注"])
    for s in recipe["sinter_steps"]:
        w.writerow([
            "", s.get("action", ""), s.get("temp_C", ""),
            f"{s.get('hours', s.get('duration_min', '?'))} {'h' if s.get('hours') else 'min'}",
            s.get("atmosphere", ""),
            f"{s.get('ramp_C_per_min', '?')} °C/min",
            s.get("tool", ""), s.get("notes", ""),
        ])
    w.writerow([])
    w.writerow(["安全提示"])
    for n in recipe["safety_notes"]:
        w.writerow([n])
    return buf.getvalue()
