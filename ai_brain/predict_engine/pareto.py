"""多目标 Pareto 前沿 — λ_em 命中 × 热稳定性 × 原料成本.

第 2 期 #4 (2026-06-11): 对标 Citrine 多目标材料筛选.
- 数据源: predictions.jsonl partial 记录 (λ_em + thermal_stability_pct_423K)
  + r1_verdict* 记录 (trace_id → 终审 verdict) + actual 记录 (实测 λ 覆盖预测)
- 成本: recipe.generate_recipe (53 原料 Aladdin 2025 参考价) total_cost_yuan
- 三目标: |λ−target| min / thermal max / cost min → O(n²) 非支配扫描 (n~300 够快)
"""
from __future__ import annotations

from typing import Any, Optional


def _latest_partials(recs: list[dict]) -> dict[tuple, dict]:
    """按 (formula, symbol, site, pct) 去重, 保留最新 partial."""
    out: dict[tuple, dict] = {}
    for r in recs:
        if r.get("type") != "partial":
            continue
        pl = r.get("payload") or {}
        formula = (r.get("formula") or pl.get("formula") or "").strip()
        if not formula:
            continue
        dop = r.get("dopant") or pl.get("dopant") or {}
        key = (formula, dop.get("symbol", "Cr3+"), dop.get("site", "Al"),
               float(dop.get("pct") or 1.0))
        out[key] = r   # load_all 按时间序, 后者覆盖前者 = 最新
    return out


def _verdict_map(recs: list[dict]) -> dict[str, str]:
    """trace_id → 最终 verdict (r1_verdict > r1_verdict_sc > r1_verdict_local)."""
    prio = {"r1_verdict": 3, "r1_verdict_sc": 2, "r1_verdict_local": 1}
    best: dict[str, tuple[int, str]] = {}
    for r in recs:
        t = r.get("type", "")
        if t not in prio:
            continue
        tid = r.get("trace_id", "")
        v = r.get("verdict")
        if isinstance(v, dict):
            v = v.get("verdict")
        if not (tid and v):
            continue
        if tid not in best or prio[t] >= best[tid][0]:
            best[tid] = (prio[t], str(v))
    return {k: v[1] for k, v in best.items()}


def _actual_map(recs: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in recs:
        if r.get("type") != "actual":
            continue
        tid = r.get("trace_id", "")
        a = (r.get("actual") or {}).get("actual_lambda_em_nm")
        try:
            if tid and a not in (None, ""):
                out[tid] = float(a)
        except (TypeError, ValueError):
            pass
    return out


def _pareto_front(points: list[dict]) -> None:
    """三目标非支配扫描, 就地标 front=True. 目标全转最小化:
    (d_lambda, -thermal_pct, cost_yuan). 只有三目标齐全的点参与."""
    elig = [p for p in points
            if p["d_lambda"] is not None and p["thermal_pct"] is not None
            and p["cost_yuan"] is not None]
    for p in elig:
        p_obj = (p["d_lambda"], -p["thermal_pct"], p["cost_yuan"])
        dominated = False
        for q in elig:
            if q is p:
                continue
            q_obj = (q["d_lambda"], -q["thermal_pct"], q["cost_yuan"])
            if all(qo <= po for qo, po in zip(q_obj, p_obj)) and q_obj != p_obj:
                dominated = True
                break
        p["front"] = not dominated
    for p in points:
        p.setdefault("front", False)


def collect_points(target_nm: float = 900.0, mass_g: float = 2.0,
                   max_points: int = 500) -> dict[str, Any]:
    from predict_engine import persistence
    from predict_engine.recipe import generate_recipe

    recs = persistence.load_all()
    latest = _latest_partials(recs)
    vmap = _verdict_map(recs)
    amap = _actual_map(recs)

    points: list[dict] = []
    cost_cache: dict[tuple, Optional[float]] = {}
    for key, r in latest.items():
        formula, symbol, site, pct = key
        pl = r.get("payload") or {}
        pm = pl.get("virtual_pl_meta") or {}
        lam = pm.get("predicted_lambda_em_nm") or pm.get("lambda_em_nm")
        if lam is None:
            continue
        tid = r.get("trace_id", "")
        measured = tid in amap
        lam_eff = amap[tid] if measured else float(lam)
        thermal = pm.get("thermal_stability_pct_423K")
        # 成本 (同 formula+dopant 缓存; generate_recipe 解析失败 → None 不上前沿)
        ck = (formula, symbol, site, pct)
        if ck not in cost_cache:
            try:
                rec = generate_recipe(formula, {"symbol": symbol, "site": site,
                                                "pct": pct}, target_mass_g=mass_g)
                cost_cache[ck] = float(rec.get("total_cost_yuan"))
            except Exception:
                cost_cache[ck] = None
        cost = cost_cache[ck]
        verdict = vmap.get(tid) or ""
        if not verdict:
            hv = pl.get("heuristic_verdict")
            if isinstance(hv, dict):
                verdict = hv.get("verdict") or ""
            elif hv:
                verdict = str(hv)
        points.append({
            "formula": formula,
            "dopant": {"symbol": symbol, "site": site, "pct": pct},
            "lambda_nm": round(lam_eff, 1),
            "measured": measured,
            "d_lambda": round(abs(lam_eff - target_nm), 1),
            "thermal_pct": round(float(thermal), 1) if thermal is not None else None,
            "cost_yuan": round(cost, 2) if cost is not None else None,
            "verdict": verdict or "UNKNOWN",
            "trace_id": tid,
            "ts": r.get("timestamp", ""),
        })

    # 截断: 留信息最全 + 最新的
    points.sort(key=lambda p: (p["thermal_pct"] is None, p["cost_yuan"] is None,
                               p["ts"]), reverse=False)
    points = points[-max_points:] if len(points) > max_points else points
    _pareto_front(points)
    n_full = sum(1 for p in points if p["thermal_pct"] is not None
                 and p["cost_yuan"] is not None)
    return {
        "ok": True, "target_nm": target_nm, "mass_g": mass_g,
        "n_points": len(points), "n_full3d": n_full,
        "n_front": sum(1 for p in points if p["front"]),
        "n_measured": sum(1 for p in points if p["measured"]),
        "points": points,
    }
