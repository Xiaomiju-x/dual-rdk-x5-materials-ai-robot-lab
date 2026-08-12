"""formula_parser.py — 轻量化学式解析 (不依赖 pymatgen, X5 可跑).

只支持最常见的无机化学式, 足以覆盖本项目的 garnet/perovskite/尖晶石 族.
不支持: 水合物 / 括号嵌套 / 分数系数 / 价态标注混写.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
import json

_ELEM_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
_VALENCE_RE = re.compile(r"^([A-Z][a-z]?)(\d*)([+\-])$")

_SHANNON_PATH = Path(__file__).parent / "shannon_radii.json"
with open(_SHANNON_PATH, encoding="utf-8") as f:
    _SHANNON = json.load(f)["radii"]


@dataclass
class Composition:
    formula: str                                  # 原始字符串 "La3ZnGa3GeO12"
    elements: dict[str, float] = field(default_factory=dict)   # {"La": 3, "Zn": 1, ...}
    charge_balance: float | None = None           # sum(valence × count)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def element_set(self) -> set[str]:
        return set(self.elements.keys())

    def total_atoms(self) -> float:
        return sum(self.elements.values())

    def jaccard(self, other: "Composition") -> float:
        a, b = self.element_set(), other.element_set()
        if not a and not b:
            return 0.0
        return len(a & b) / len(a | b)

    def is_valid(self) -> bool:
        return not self.errors and len(self.elements) >= 2

    def to_dict(self) -> dict:
        return {
            "formula": self.formula,
            "elements": self.elements,
            "total_atoms": self.total_atoms(),
            "errors": self.errors,
            "warnings": self.warnings,
        }


# 常见阳离子的默认氧化态 (用于电荷平衡估算; 多价态元素给出首选)
_DEFAULT_VALENCE = {
    "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1,
    "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Zn": 2,
    "Al": 3, "Ga": 3, "In": 3, "Sc": 3, "Y": 3,
    "La": 3, "Gd": 3, "Lu": 3, "Cr": 3, "Fe": 3,
    "Si": 4, "Ge": 4, "Sn": 4, "Ti": 4, "Zr": 4, "Hf": 4,
    "Nb": 5, "Ta": 5, "W": 6, "Mo": 6, "Sb": 5, "P": 5,
    "O": -2, "N": -3, "S": -2, "F": -1, "Cl": -1,
}


def parse_formula(formula: str) -> Composition:
    """`"La3ZnGa3GeO12"` → Composition(elements={La:3, Zn:1, Ga:3, Ge:1, O:12}).

    注意: 小写首字母会被解读为前一元素延续 (e.g. "Co" = 钴, "CO" = C + O).
    输入前请规范化大小写.
    """
    comp = Composition(formula=formula.strip())
    if not formula:
        comp.errors.append("空字符串")
        return comp

    matches = _ELEM_RE.findall(formula)
    if not matches:
        comp.errors.append(f"无法解析: {formula!r}")
        return comp

    for elem, count_str in matches:
        if not elem:
            continue
        count = float(count_str) if count_str else 1.0
        comp.elements[elem] = comp.elements.get(elem, 0.0) + count

    # 电荷平衡检查
    total_charge = 0.0
    unknown = []
    for elem, count in comp.elements.items():
        val = _DEFAULT_VALENCE.get(elem)
        if val is None:
            unknown.append(elem)
            continue
        total_charge += val * count
    comp.charge_balance = round(total_charge, 3)
    if unknown:
        comp.warnings.append(f"未知元素价态: {unknown} (忽略)")
    if abs(total_charge) > 0.5:
        comp.warnings.append(
            f"电荷不平衡: net = {total_charge:+.2f} (使用默认价态估算, 可能多价态混合)"
        )
    return comp


def parse_dopant(dopant_str: str) -> dict:
    """`"Cr3+"` → {"element": "Cr", "valence": 3, "symbol": "Cr3+"}.

    接受的格式: "Cr3+", "Ni2+", "Cr+Ni" (共掺, 返回第一个).
    """
    if not dopant_str:
        return {"element": None, "valence": None, "symbol": "", "raw": ""}
    first = dopant_str.strip().split("+")[0] + dopant_str.strip().split("+")[1][0] \
            if "+" in dopant_str and not dopant_str.endswith("+") \
            else dopant_str.strip()
    # 简单 fallback: 抓头部元素 + 末尾 + 号
    m = _VALENCE_RE.match(dopant_str.strip())
    if m:
        return {
            "element": m.group(1),
            "valence": int(m.group(2)) if m.group(2) else None,
            "symbol": dopant_str.strip(),
            "raw": dopant_str,
        }
    # Fallback: 取首个元素符号
    em = _ELEM_RE.match(dopant_str.strip())
    if em and em.group(1):
        return {"element": em.group(1), "valence": None, "symbol": em.group(1), "raw": dopant_str}
    return {"element": None, "valence": None, "symbol": dopant_str, "raw": dopant_str}


def get_shannon_radius(species: str, cn: int = 6) -> float | None:
    """`"Al3+", 6 → 0.535`.  找不到就往低 cn 降级."""
    entry = _SHANNON.get(species)
    if not entry:
        return None
    # 优先精确 CN, 再退到附近 CN
    for candidate_cn in [cn, 8, 6, 4]:
        v = entry.get(str(candidate_cn))
        if v is not None:
            return float(v)
    return None


if __name__ == "__main__":
    for f in ["La3ZnGa3GeO12", "Y3Al5O12", "CaY2Al4SiO12",
              "NaY2Ga2InGe2O12", "Sr6Y2Al4O15"]:
        c = parse_formula(f)
        print(f"{f:25s} | {c.elements} | charge={c.charge_balance:+.1f}")
