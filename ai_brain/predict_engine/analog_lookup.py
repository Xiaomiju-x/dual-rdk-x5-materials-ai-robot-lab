"""analog_lookup.py — 根据目标化学式 + 掺杂查找最近类比.

来源:
  1. crystal_data_shared/candidate_pool.json  → theoretical_peaks (XRD 参考)
  2. exp_ground_truth/observed_pl.csv         → 实测 λ_em/FWHM/热稳定性 (PL 参考)

相似度 = 0.5 × host_family_match + 0.3 × dopant_match + 0.2 × element_jaccard
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from .formula_parser import Composition, parse_formula


# 项目内可能的路径 (X5 和 PC 都能找到)
_POOL_CANDIDATES = [
    Path(__file__).parent.parent / "crystal_data_shared" / "candidate_pool.json",
    Path("/home/rdk/crystal_data_shared/candidate_pool.json"),
    Path("/home/rdk/xrd_num/candidate_pool.json"),
    Path("/home/rdk/xrd_vision/candidate_pool.json"),
    Path("/home/rdk/spec_num/candidate_pool.json"),
    Path("/home/rdk/spec_vision/candidate_pool.json"),
    Path("/home/rdk/candidate_pool.json"),
    Path("./candidate_pool.json"),
]

_PL_CSV_CANDIDATES = [
    Path(__file__).parent.parent / "exp_ground_truth" / "observed_pl.csv",
    Path("/home/rdk/exp_ground_truth/observed_pl.csv"),
    Path("./exp_ground_truth/observed_pl.csv"),
]


def _find_first(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


@dataclass
class XRDAnalog:
    formula: str
    host_family: str
    mp_id: str | None
    spacegroup: str | None
    theoretical_peaks: list[dict]      # [{two_theta, intensity}, ...]
    source: str                        # "candidate_pool:garnet" 等
    similarity: float = 0.0            # 0-1, 综合打分

    def to_dict(self) -> dict:
        return {
            "formula": self.formula,
            "host_family": self.host_family,
            "mp_id": self.mp_id,
            "spacegroup": self.spacegroup,
            "source": self.source,
            "similarity": round(self.similarity, 3),
            "peak_count": len(self.theoretical_peaks),
        }


@dataclass
class PLAnalog:
    formula: str
    host_family: str
    dopant_element: str
    dopant_pct: float
    dopant_site: str | None
    sinter_temp_C: float | None
    sinter_hours: float | None
    xrd_result: str                    # pure / mixed / amorphous / unknown
    lambda_ex_nm: float | None
    lambda_em_nm: float | None
    fwhm_nm: float | None
    thermal_stability_pct: float | None
    quantum_yield_pct: float | None
    source: str
    similarity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "formula": self.formula,
            "host_family": self.host_family,
            "dopant": f"{self.dopant_element}-{self.dopant_pct}%",
            "dopant_site": self.dopant_site,
            "sinter": (f"{self.sinter_temp_C}°C/{self.sinter_hours}h"
                       if self.sinter_temp_C else None),
            "xrd_result": self.xrd_result,
            "lambda_ex_nm": self.lambda_ex_nm,
            "lambda_em_nm": self.lambda_em_nm,
            "fwhm_nm": self.fwhm_nm,
            "thermal_stability_pct": self.thermal_stability_pct,
            "quantum_yield_pct": self.quantum_yield_pct,
            "source": self.source,
            "similarity": round(self.similarity, 3),
        }


import threading

_POOL_CACHE = None
_PL_CACHE = None
_LOAD_LOCK = threading.Lock()


def _load_pool() -> dict:
    global _POOL_CACHE
    if _POOL_CACHE is not None:
        return _POOL_CACHE
    with _LOAD_LOCK:
        if _POOL_CACHE is not None:
            return _POOL_CACHE
        p = _find_first(_POOL_CANDIDATES)
        if p is None:
            _POOL_CACHE = {}
            return _POOL_CACHE
        with open(p, encoding="utf-8") as f:
            _POOL_CACHE = json.load(f)
    return _POOL_CACHE


def _load_pl_table() -> list[dict]:
    global _PL_CACHE
    if _PL_CACHE is not None:
        return _PL_CACHE
    with _LOAD_LOCK:
        if _PL_CACHE is not None:
            return _PL_CACHE
        p = _find_first(_PL_CSV_CANDIDATES)
        if p is None:
            _PL_CACHE = []
            return _PL_CACHE
        rows: list[dict] = []
        with open(p, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        _PL_CACHE = rows
    return _PL_CACHE


def _parse_float(s) -> float | None:
    if s is None or s == "":
        return None
    try:
        # 兼容 "1400-5h" 这种非纯数格式 (取头部数字)
        s = str(s).strip()
        # 尝试直接解析
        return float(s)
    except (ValueError, TypeError):
        # 提取首段数字
        import re
        m = re.match(r"[+-]?\d+\.?\d*", s)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                pass
        return None


def find_xrd_analog(target: Composition, host_hint: str | None = None) -> XRDAnalog | None:
    """遍历 candidate_pool, 按 element Jaccard 选最相似.

    host_hint 来自用户下拉选择 (如 "garnet"), 若指定则限定在该 family 内.
    """
    pool = _load_pool()
    best: XRDAnalog | None = None
    best_score = -1.0

    keys = [host_hint] if host_hint and host_hint in pool else list(pool.keys())
    for family in keys:
        for entry in pool.get(family, []):
            cand_formula = entry.get("formula", "")
            cand_comp = parse_formula(cand_formula)
            jac = target.jaccard(cand_comp)
            family_bonus = 0.2 if host_hint and family == host_hint else 0.0
            score = jac + family_bonus
            if score > best_score:
                best_score = score
                peaks = [{"two_theta": float(p[0]), "intensity": float(p[1])}
                         for p in entry.get("theoretical_peaks", [])]
                best = XRDAnalog(
                    formula=cand_formula,
                    host_family=family,
                    mp_id=entry.get("mp_id"),
                    spacegroup=entry.get("spacegroup_symbol"),
                    theoretical_peaks=peaks,
                    source=f"candidate_pool:{family}",
                    similarity=min(1.0, score),
                )
    return best


def find_pl_analogs(
    target: Composition,
    dopant: dict,
    top_k: int = 3,
) -> list[PLAnalog]:
    """查 PL 实测表. 返回按相似度排序的 top_k 条."""
    rows = _load_pl_table()
    if not rows:
        return []

    scored: list[tuple[float, PLAnalog]] = []
    target_elements = target.element_set()
    dop_el = (dopant.get("element") or "").lower()
    dop_pct = float(dopant.get("pct") or 0)
    dop_site = (dopant.get("site") or "").lower()

    for row in rows:
        cand_formula = (row.get("formula") or "").strip()
        if not cand_formula:
            continue
        cand_comp = parse_formula(cand_formula)
        jac = len(target_elements & cand_comp.element_set()) / max(
            len(target_elements | cand_comp.element_set()), 1)

        row_dop_el = (row.get("dopant_element") or "").strip().lower()
        row_dop_pct = _parse_float(row.get("dopant_pct")) or 0
        row_dop_site = (row.get("dopant_site") or "").strip().lower()

        dop_match = 1.0 if (row_dop_el and dop_el and row_dop_el == dop_el) else 0.0
        pct_match = 1.0 - min(abs(row_dop_pct - dop_pct) / max(dop_pct, 0.1), 1.0) \
                    if (row_dop_pct and dop_pct) else 0.0
        # 位点匹配 (避免 Cr3+@Zn 被 Cr3+@Ga 记录拉高相似度)
        site_match = 1.0 if (row_dop_site and dop_site and row_dop_site == dop_site) else 0.0

        # 位点权重 0.25, 元素 0.25, host 0.35, pct 0.15
        score = 0.35 * jac + 0.25 * dop_match + 0.25 * site_match + 0.15 * pct_match
        if score <= 0.05:
            continue

        analog = PLAnalog(
            formula=cand_formula,
            host_family=(row.get("host_family") or "").strip(),
            dopant_element=row.get("dopant_element", ""),
            dopant_pct=row_dop_pct,
            dopant_site=row.get("dopant_site") or None,
            sinter_temp_C=_parse_float(row.get("sinter_temp_C")),
            sinter_hours=_parse_float(row.get("sinter_hours")),
            xrd_result=(row.get("xrd_result") or "unknown").strip().lower(),
            lambda_ex_nm=_parse_float(row.get("lambda_ex_nm")),
            lambda_em_nm=_parse_float(row.get("lambda_em_nm")),
            fwhm_nm=_parse_float(row.get("fwhm_nm")),
            thermal_stability_pct=_parse_float(row.get("thermal_stability_pct_at_150C")),
            quantum_yield_pct=_parse_float(row.get("quantum_yield_pct")),
            source=(row.get("source") or "exp_ground_truth").strip(),
            similarity=min(1.0, score),
        )
        scored.append((score, analog))

    scored.sort(key=lambda t: -t[0])
    return [a for _, a in scored[:top_k]]


def get_preset_formulas() -> list[dict]:
    """dashboard datalist 用. 返回 [{formula, host_family}, ...]."""
    pool = _load_pool()
    out: list[dict] = []
    for family, entries in pool.items():
        for e in entries:
            f = e.get("formula")
            if f:
                out.append({"formula": f, "host_family": family})
    # 再并入 observed_pl.csv 里的 formula (去重)
    seen = {x["formula"] for x in out}
    for row in _load_pl_table():
        f = (row.get("formula") or "").strip()
        if f and f not in seen:
            out.append({"formula": f, "host_family": (row.get("host_family") or "").strip()})
            seen.add(f)
    return out
