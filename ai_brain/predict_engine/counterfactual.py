"""counterfactual.py — 反事实 reasoning (P0-5).

给定 {formula, dopant} → 产生邻域变体 (元素替换 / pct 变化 / dopant ion 替换 / 组合),
每个变体单独跑 predict(), 聚合 {verdict_variant, delta_lambda_em, visual_tag}.

用途: dashboard /counterfactual 页的 d3-sankey 图, 答辩时展示"只改一个元素就从 GO 变 DROP"的
因果直觉, 以及原方案在 neighborhood 中的鲁棒性.
"""
from __future__ import annotations

import re
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .formula_parser import parse_formula, parse_dopant


# 同族元素替换表 (化学 / 晶体化学上相近, 替换后仍可能形成同构相)
_HOMOGROUP = {
    # 稀土 / Y / Sc / La 系 (3+, 8-coord / 9-coord 常见)
    "Y":  ["La", "Gd", "Lu"],
    "La": ["Y",  "Gd", "Lu"],
    "Gd": ["Y",  "La", "Lu"],
    "Lu": ["Y",  "La", "Gd"],
    # 三价 B 族
    "Al": ["Ga", "In", "Sc"],
    "Ga": ["Al", "In", "Sc"],
    "In": ["Al", "Ga", "Sc"],
    "Sc": ["Al", "Ga", "In"],
    # 碱土 2+
    "Ca": ["Sr", "Ba", "Mg"],
    "Sr": ["Ca", "Ba", "Mg"],
    "Ba": ["Ca", "Sr"],
    "Mg": ["Ca", "Sr", "Zn"],
    # 4+ 四面体
    "Si": ["Ge", "Sn"],
    "Ge": ["Si", "Sn"],
    "Sn": ["Si", "Ge"],
}

# dopant ion 替换 pair (保持在同一 site 上做激活剂替换)
_DOPANT_SWAP = {
    "Cr": {"element": "Ni", "valence": 2, "symbol": "Ni2+"},
    "Ni": {"element": "Cr", "valence": 3, "symbol": "Cr3+"},
    # Round 6 可能扩的其他替换
    "Mn": {"element": "Fe", "valence": 3, "symbol": "Fe3+"},
    "Fe": {"element": "Mn", "valence": 3, "symbol": "Mn3+"},
}


def _rebuild_formula(original: str, sub_map: dict[str, str]) -> str:
    """把 original 里的元素按 sub_map 替换, 保留系数."""
    pat = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
    out = []
    for elem, cnt in pat.findall(original):
        if not elem:
            continue
        new_el = sub_map.get(elem, elem)
        out.append(f"{new_el}{cnt}")
    return "".join(out)


def _tag(delta_lambda: float | None, v_orig: str, v_var: str) -> str:
    """根据 verdict 是否翻转 / Δλ 幅度给 visual tag.

    - 'flip'     : verdict 换了 (GO→DROP / GO→REVISE / REVISE→GO ...)
    - 'bigshift' : verdict 相同, 但 Δλ_em > 30 nm
    - 'stable'   : verdict 相同且 Δλ 小
    """
    if v_orig and v_var and v_orig != v_var:
        return "flip"
    if delta_lambda is not None and abs(delta_lambda) > 30:
        return "bigshift"
    return "stable"


def enumerate_counterfactuals(
    formula: str,
    dopant: dict,
    max_variants: int = 12,
) -> list[dict]:
    """产生 ~8-12 邻域变体的 plan (还未执行 predict).

    返回 list of plan dict, 每 dict: {kind, variant_formula, variant_dopant, change_description}
    后续由 run_counterfactuals 把每 plan 跑一遍.
    """
    target = parse_formula(formula)
    if not target.is_valid():
        return []

    elements = list(target.elements.keys())
    cations = [e for e in elements if e != "O" and e != "N" and e != "F"]

    plans: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def _push(kind: str, new_formula: str, new_dopant: dict, desc: str):
        key = (new_formula, f"{new_dopant.get('element')}{new_dopant.get('valence','')}/{new_dopant.get('site','')}/{new_dopant.get('pct','')}")
        if key in seen:
            return
        if new_formula == formula and new_dopant == dopant:
            return
        seen.add(key)
        plans.append({
            "kind": kind,
            "variant_formula": new_formula,
            "variant_dopant": new_dopant,
            "change_description": desc,
        })

    # -------- 1) 元素替换 (3 条: 每个可替换阳离子 pick 1 同族候选) --------
    n_elem_sub = 0
    for cation in cations:
        if n_elem_sub >= 3:
            break
        candidates = _HOMOGROUP.get(cation)
        if not candidates:
            continue
        for repl in candidates[:1]:  # 每 cation 取首选
            new_f = _rebuild_formula(formula, {cation: repl})
            _push(
                "element_substitution",
                new_f, dict(dopant),
                f"阳离子同族替换: {cation} → {repl}",
            )
            n_elem_sub += 1
            break

    # -------- 2) 掺杂浓度变化 (3 条: 0.5x / 1.5x / 2.0x) --------
    orig_pct = float(dopant.get("pct", 1.0) or 1.0)
    for scale, label in [(0.5, "半量"), (1.5, "1.5 倍"), (2.0, "双倍")]:
        new_pct = round(orig_pct * scale, 3)
        if new_pct <= 0:
            continue
        nd = dict(dopant); nd["pct"] = new_pct
        _push(
            "pct_change",
            formula, nd,
            f"掺杂浓度 {orig_pct}% → {new_pct}% ({label})",
        )

    # -------- 3) dopant ion 替换 (2 条: Cr3+ ↔ Ni2+ 等) --------
    dop_el = (dopant or {}).get("element")
    if dop_el and dop_el in _DOPANT_SWAP:
        swap = _DOPANT_SWAP[dop_el]
        nd = dict(dopant)
        nd["element"] = swap["element"]
        nd["valence"] = swap["valence"]
        nd["symbol"] = swap["symbol"]
        _push(
            "dopant_swap",
            formula, nd,
            f"激活剂替换: {dop_el}{dopant.get('valence','')}+ → {swap['symbol']}",
        )
    # second dopant swap: 共掺 (把 pct 拆一半给新 ion)
    if dop_el and dop_el in _DOPANT_SWAP:
        nd = dict(dopant)
        nd["pct"] = round(orig_pct * 0.5, 3)
        nd["element"] = _DOPANT_SWAP[dop_el]["element"]
        nd["valence"] = _DOPANT_SWAP[dop_el]["valence"]
        nd["symbol"] = _DOPANT_SWAP[dop_el]["symbol"]
        _push(
            "dopant_swap_half",
            formula, nd,
            f"激活剂替换+半量: → {nd['symbol']} @ {nd['pct']}%",
        )

    # -------- 4) 组合扰动 (3 条: 元素替换 + pct 变化) --------
    if cations:
        primary = cations[0]
        repls = _HOMOGROUP.get(primary, [])
        for i, repl in enumerate(repls[:2]):
            new_f = _rebuild_formula(formula, {primary: repl})
            nd = dict(dopant)
            nd["pct"] = round(orig_pct * (1.5 if i == 0 else 0.5), 3)
            _push(
                "combo",
                new_f, nd,
                f"组合扰动: {primary}→{repl} + pct={nd['pct']}%",
            )
    # 组合第三条: dopant swap + pct 1.5x
    if dop_el and dop_el in _DOPANT_SWAP:
        swap = _DOPANT_SWAP[dop_el]
        nd = dict(dopant)
        nd["element"] = swap["element"]
        nd["valence"] = swap["valence"]
        nd["symbol"] = swap["symbol"]
        nd["pct"] = round(orig_pct * 1.5, 3)
        _push(
            "combo",
            formula, nd,
            f"组合扰动: {swap['symbol']} + pct={nd['pct']}%",
        )

    return plans[:max_variants]


def _extract_verdict_and_lambda(res: dict) -> tuple[str, float | None]:
    """从 predict() 返回 partial 抽 verdict + lambda_em."""
    vh = (res.get("heuristic_verdict") or {}).get("verdict") or "UNKNOWN"
    pl_meta = res.get("virtual_pl_meta") or {}
    lam = pl_meta.get("predicted_lambda_em_nm") or pl_meta.get("lambda_em_nm")
    try:
        lam = float(lam) if lam is not None else None
    except Exception:
        lam = None
    return vh, lam


def run_counterfactuals(
    formula: str,
    dopant: dict,
    sinter_temp_C: float | None = None,
    host_hint: str | None = None,
    max_variants: int = 12,
    max_workers: int = 3,
) -> dict:
    """主入口: 原 + 所有变体各跑 predict(), 聚合成 dashboard 可直接吃的结构.

    返回:
      {
        "trace_id": "cf:xxxx",
        "original": {"formula", "dopant", "verdict", "lambda_em"},
        "variants": [
           {"kind", "variant_formula", "variant_dopant", "change_description",
            "verdict_original", "verdict_variant", "lambda_em_variant",
            "delta_lambda_em", "visual_tag", "trace_id_variant"}, ...
        ],
        "n_variants": int,
        "elapsed_s": float,
      }
    """
    from .engine import predict  # 延迟 import 避免循环

    t0 = time.perf_counter()
    if isinstance(dopant, str):
        dopant = parse_dopant(dopant)
    dopant = dopant or {}
    plans = enumerate_counterfactuals(formula, dopant, max_variants=max_variants)

    cf_id = "cf:" + hashlib.sha1(
        f"{formula}|{time.time()}|{len(plans)}".encode()).hexdigest()[:10]

    # -------- 原始方案跑一次 --------
    try:
        orig_res = predict(formula, dopant, sinter_temp_C=sinter_temp_C, host_hint=host_hint)
    except Exception as e:
        return {
            "trace_id": cf_id, "ok": False,
            "error": f"原始 predict 失败: {e}",
            "variants": [],
        }
    v_orig, lam_orig = _extract_verdict_and_lambda(orig_res)

    # -------- 并发跑每个变体 --------
    def _one(plan: dict) -> dict:
        try:
            r = predict(
                plan["variant_formula"],
                plan["variant_dopant"],
                sinter_temp_C=sinter_temp_C,
                host_hint=host_hint,
            )
            v_var, lam_var = _extract_verdict_and_lambda(r)
            delta = None
            if lam_orig is not None and lam_var is not None:
                delta = round(lam_var - lam_orig, 1)
            return {
                **plan,
                "ok": True,
                "verdict_original": v_orig,
                "verdict_variant": v_var,
                "lambda_em_original": lam_orig,
                "lambda_em_variant": lam_var,
                "delta_lambda_em": delta,
                "visual_tag": _tag(delta, v_orig, v_var),
                "trace_id_variant": r.get("trace_id"),
                "heuristic_reason": (r.get("heuristic_verdict") or {}).get("reason", ""),
            }
        except Exception as e:
            return {
                **plan, "ok": False, "error": str(e)[:200],
                "verdict_original": v_orig, "verdict_variant": "UNKNOWN",
                "delta_lambda_em": None, "visual_tag": "stable",
            }

    variants: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cf") as exe:
        futs = [exe.submit(_one, p) for p in plans]
        for f in as_completed(futs):
            variants.append(f.result())

    # 排序: flip > bigshift > stable, 同类按 |Δλ| 降序
    _rank = {"flip": 0, "bigshift": 1, "stable": 2}
    variants.sort(key=lambda x: (
        _rank.get(x.get("visual_tag"), 3),
        -abs(x.get("delta_lambda_em") or 0),
    ))

    return {
        "ok": True,
        "trace_id": cf_id,
        "formula": formula,
        "dopant": dopant,
        "original": {
            "formula": formula,
            "dopant": dopant,
            "verdict": v_orig,
            "lambda_em": lam_orig,
            "trace_id": orig_res.get("trace_id"),
            "reason": (orig_res.get("heuristic_verdict") or {}).get("reason", ""),
        },
        "variants": variants,
        "n_variants": len(variants),
        "n_flip": sum(1 for v in variants if v.get("visual_tag") == "flip"),
        "n_bigshift": sum(1 for v in variants if v.get("visual_tag") == "bigshift"),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
