"""engine.py — 合成预测主入口.

dashboard.py 调用流程:
    from predict_engine.engine import predict, run_r1_judge_stream

    partial = predict(formula, dopant, sinter_temp_C, host_hint)     # 同步, 5 秒内
    # partial 含 stages(4 BPU) + analog + flags + rag + virtual_spectra meta
    # 然后 dashboard 开一个 SSE 把 run_r1_judge_stream(partial) 流到前端
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

import threading
from .formula_parser import parse_formula, parse_dopant, Composition
from .analog_lookup import (
    find_xrd_analog, find_pl_analogs, XRDAnalog, PLAnalog, get_preset_formulas,
)
from .vegard import vegard_shift_peaks
from .ml_cache_lookup import lookup_mace_cache, cache_to_perturbed_peaks
from .failure_flags import compute_failure_flags, severity
from .virtual_spectra import (
    virtual_xrd_peaks, build_xrd_feature_190d, virtual_pl_spectrum,
    build_pl_feature_80d, render_xrd_png_b64, render_pl_png_b64,
)
from . import persistence


# 4 条 BPU 线的 HTTP 端点 (X5 内部调用 127.0.0.1, 可被 env override)
import os
_BPU_ENDPOINTS = {
    "xrd_num":    os.environ.get("PRED_BPU_XRD_NUM", "http://127.0.0.1:5000/api/bpu_infer_190d"),
    "xrd_vision": os.environ.get("PRED_BPU_XRD_VIS", "http://127.0.0.1:8080/api/bpu_detect_b64"),
    "pl_num":     os.environ.get("PRED_BPU_PL_NUM",  "http://127.0.0.1:5001/api/bpu_infer_80d"),
    "pl_vision":  os.environ.get("PRED_BPU_PL_VIS",  "http://127.0.0.1:8081/api/bpu_detect_b64"),
}


def _call_bpu(endpoint: str, payload: dict, timeout: float = 10.0) -> dict:
    """统一包装, 失败返回 {ok:False, error}."""
    try:
        r = requests.post(endpoint, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e), "endpoint": endpoint}


def _trace_id(formula: str, dopant: dict) -> str:
    m = hashlib.sha1(f"{formula}|{json.dumps(dopant, sort_keys=True)}|{int(time.time())}".encode()).hexdigest()
    return "sha1:" + m[:12]


# ============ RAG 懒加载 (thread-safe via lock) ============
_rag_xrd = None
_rag_pl = None
_RAG_LOCK = threading.Lock()


def _load_rags():
    global _rag_xrd, _rag_pl
    if _rag_xrd is not None or _rag_pl is not None:
        return _rag_xrd, _rag_pl
    with _RAG_LOCK:
        if _rag_xrd is not None or _rag_pl is not None:
            return _rag_xrd, _rag_pl

    # 尝试多路径
    import sys
    candidate_paths = [
        Path(__file__).parent.parent / "xrd_numerical",
        Path(__file__).parent.parent / "spectrum_numerical",
        Path("/home/rdk/xrd_num"),
        Path("/home/rdk/spec_num"),
    ]
    for p in candidate_paths:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    try:
        # xrd_numerical/rag_engine.py 和 spectrum_numerical/rag_engine.py 同名, 用路径加载区分
        import importlib.util
        for name, base_candidates in [
            ("_rag_xrd_mod", [Path("/home/rdk/xrd_num/rag_engine.py"),
                              Path(__file__).parent.parent / "xrd_numerical" / "rag_engine.py"]),
            ("_rag_pl_mod", [Path("/home/rdk/spec_num/rag_engine.py"),
                             Path(__file__).parent.parent / "spectrum_numerical" / "rag_engine.py"]),
        ]:
            for path in base_candidates:
                if path.exists():
                    spec = importlib.util.spec_from_file_location(name, str(path))
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        rag = mod.RAGEngine() if hasattr(mod, "RAGEngine") else None
                        if name == "_rag_xrd_mod":
                            _rag_xrd = rag
                        else:
                            _rag_pl = rag
                        break
    except Exception as e:
        print(f"[predict_engine] RAG 加载失败: {e}")
    return _rag_xrd, _rag_pl


def _retrieve_rag(rag, query: str, top_k: int = 3) -> list[dict]:
    if rag is None:
        return []
    try:
        if hasattr(rag, "retrieve_chunks"):
            return rag.retrieve_chunks(query, top_k=top_k)
        # Fallback: format 成 string 再分割
        text = rag.retrieve(query, top_k=top_k)
        return [{"text": text, "source": "formatted"}]
    except Exception as e:
        return [{"error": str(e)}]


# ============ 主入口 ============
def predict(
    formula: str,
    dopant: dict | str,
    sinter_temp_C: float | None = None,
    host_hint: str | None = None,
) -> dict:
    """同步调用. 返回 partial result (不含 R1 verdict). 耗时 ~1-5s (含 4 次 BPU HTTP).

    Args:
        formula: "La3ZnGa3GeO12"
        dopant: {"element": "Cr", "valence": 3, "site": "Ga", "pct": 0.75}
                或者字符串 "Cr3+" (自动解析, 缺 site/pct)
        sinter_temp_C: 用户指定烧结温度 (可选, 没填则由类比推荐)
        host_hint: "garnet" / "perovskite" 等, 前端下拉传入 (可选)
    """
    t0 = time.perf_counter()
    if isinstance(dopant, str):
        dopant = parse_dopant(dopant)

    target = parse_formula(formula)
    trace_id = _trace_id(formula, dopant)

    result: dict[str, Any] = {
        "trace_id": trace_id,
        "formula": formula,
        "dopant": dopant,
        "sinter_temp_C": sinter_temp_C,
        "host_hint": host_hint,
        "parse": target.to_dict(),
        "stages": {},
        "timing_ms": {},
    }

    if not target.is_valid():
        result["verdict_fallback"] = "UNKNOWN"
        result["error"] = "化学式解析失败"
        return result

    # ============ 1. XRD 类比查找 ============
    ts = time.perf_counter()
    xrd_analog = find_xrd_analog(target, host_hint)
    # 若 candidate_pool 的 Jaccard <0.3, 但 host_hint='garnet', 尝试取 garnet 任一条拿 theoretical_peaks
    # (下面 BPU 推理用它, 至少让链路能走完)
    if not xrd_analog or xrd_analog.similarity < 0.3:
        fallback = find_xrd_analog(target, host_hint="garnet")  # 强制 garnet pool
        if fallback and fallback.similarity > (xrd_analog.similarity if xrd_analog else 0):
            xrd_analog = fallback
    result["xrd_analog"] = xrd_analog.to_dict() if xrd_analog else None
    result["timing_ms"]["analog_xrd"] = round((time.perf_counter() - ts) * 1000, 1)

    # ============ 2. XRD 虚拟峰 (三级 fallback: MACE → Vegard → 无) ============
    ts = time.perf_counter()
    mace_record = None
    if isinstance(dopant, dict) and dopant.get("site") and dopant.get("element"):
        mace_record = lookup_mace_cache(
            formula, dopant["element"], dopant["site"], float(dopant.get("pct", 0))
        )
    if mace_record:
        virtual_peaks, vegard_meta = cache_to_perturbed_peaks(mace_record)
        analog_comp = parse_formula(xrd_analog.formula) if xrd_analog else None
    elif xrd_analog is not None:
        analog_comp = parse_formula(xrd_analog.formula)
        virtual_peaks, vegard_meta = virtual_xrd_peaks(xrd_analog, target, analog_comp)
    else:
        virtual_peaks, vegard_meta = [], {"applied": False, "reason": "no_xrd_analog"}
        analog_comp = None
    result["vegard"] = vegard_meta  # 字段名保留兼容; 现可能是 MACE 或 Vegard 元数据
    result["xrd_method"] = vegard_meta.get("method", "vegard_1st_order" if vegard_meta.get("applied") else "none")
    result["virtual_xrd_peaks"] = virtual_peaks[:15]   # UI 只显示前 15
    result["timing_ms"]["vegard"] = round((time.perf_counter() - ts) * 1000, 1)

    # ============ 3. BPU #1: xrd_num MLP (190D) ============
    ts = time.perf_counter()
    if virtual_peaks:
        feat_190d = build_xrd_feature_190d(virtual_peaks)
        bpu_xrd = _call_bpu(_BPU_ENDPOINTS["xrd_num"], {"feat": feat_190d.tolist()})
    else:
        bpu_xrd = {"ok": False, "error": "no_virtual_peaks"}
    result["stages"]["bpu_xrd_num"] = bpu_xrd
    result["timing_ms"]["bpu_xrd_num"] = round((time.perf_counter() - ts) * 1000, 1)

    # ============ 4. 渲染 + BPU #2: xrd_vision YOLO ============
    ts = time.perf_counter()
    if virtual_peaks:
        try:
            xrd_png = render_xrd_png_b64(virtual_peaks)
            bpu_xrd_vis = _call_bpu(_BPU_ENDPOINTS["xrd_vision"], {"image_b64": xrd_png}, timeout=15)
        except Exception as e:
            bpu_xrd_vis = {"ok": False, "error": f"render_or_yolo: {e}"}
    else:
        bpu_xrd_vis = {"ok": False, "error": "no_virtual_peaks"}
    result["stages"]["bpu_xrd_vision"] = bpu_xrd_vis
    result["timing_ms"]["bpu_xrd_vision"] = round((time.perf_counter() - ts) * 1000, 1)

    # ============ 5. PL 类比 Top-3 ============
    ts = time.perf_counter()
    pl_analogs = find_pl_analogs(target, dopant, top_k=3)
    result["pl_analogs"] = [a.to_dict() for a in pl_analogs]
    result["timing_ms"]["analog_pl"] = round((time.perf_counter() - ts) * 1000, 1)

    # ============ 6. 虚拟 PL + BPU #3: spec_num MLP ============
    ts = time.perf_counter()
    wavelength, counts, pl_meta = virtual_pl_spectrum(pl_analogs, dopant, target, formula=formula)

    # Phase INN-1: Conformal Prediction 对 λ_em 加 distribution-free 置信区间
    try:
        from .conformal import load_calibrator
        _cal = load_calibrator()
        lam_hat = pl_meta.get("predicted_lambda_em_nm")
        if _cal is not None and _cal.n >= 3 and lam_hat is not None:
            ci90 = _cal.predict_interval(float(lam_hat), alpha=0.10)
            pl_meta["conformal_ci90"] = ci90.to_dict()
            ci80 = _cal.predict_interval(float(lam_hat), alpha=0.20)
            pl_meta["conformal_ci80"] = ci80.to_dict()
    except Exception as _e:
        pl_meta["conformal_error"] = str(_e)[:120]

    result["virtual_pl_meta"] = pl_meta
    if pl_meta["applied"]:
        feat_80d = build_pl_feature_80d(wavelength, counts)
        bpu_pl = _call_bpu(_BPU_ENDPOINTS["pl_num"], {"feat": feat_80d.tolist()})
    else:
        bpu_pl = {"ok": False, "error": pl_meta.get("reason", "no_pl_feature")}
    result["stages"]["bpu_pl_num"] = bpu_pl
    result["timing_ms"]["bpu_pl_num"] = round((time.perf_counter() - ts) * 1000, 1)

    # ============ 7. 渲染 + BPU #4: spec_vision YOLO ============
    ts = time.perf_counter()
    if pl_meta["applied"]:
        try:
            pl_png = render_pl_png_b64(wavelength, counts, meta=pl_meta)
            bpu_pl_vis = _call_bpu(_BPU_ENDPOINTS["pl_vision"], {"image_b64": pl_png}, timeout=15)
        except Exception as e:
            bpu_pl_vis = {"ok": False, "error": f"render_or_yolo: {e}"}
    else:
        bpu_pl_vis = {"ok": False, "error": pl_meta.get("reason", "no_pl")}
    result["stages"]["bpu_pl_vision"] = bpu_pl_vis
    result["timing_ms"]["bpu_pl_vision"] = round((time.perf_counter() - ts) * 1000, 1)

    # ============ 8. RAG (Phase 2.0.c: HyDE+BM25 hybrid 优先, fallback bi-encoder) ============
    ts = time.perf_counter()
    rag_refs = []
    rag_method = "none"
    # 先尝试 hybrid (BM25 + HyDE + RRF) — 真 2462 篇 embedding 时最强
    try:
        from spectrum_knowledge_shared.hybrid_retriever import hybrid_search
        # 优化: 只在 cross-host (Jaccard<0.4) 或弱 PL 类比时触发 HyDE (省 R1 call), 否则 only BM25
        weak_pl = (not pl_analogs) or (pl_analogs and pl_analogs[0].similarity < 0.5)
        weak_xrd = (not xrd_analog) or (xrd_analog.similarity < 0.4)
        use_hyde = weak_pl or weak_xrd
        query = (f"{formula} {dopant.get('element','')} dopant {dopant.get('site','')} site "
                 f"NIR photoluminescence emission XRD phase synthesis")
        if use_hyde:
            hits = hybrid_search(query, top_n_final=4, allow_bm25_fail=True, allow_hyde_fail=True)
        else:
            from spectrum_knowledge_shared.bm25_index import bm25_search
            hits = bm25_search(query, top_k=4)
        rag_refs = [{"text": (h.get("text") or "")[:300], "source": h.get("source", "?"),
                      "score": h.get("rrf_score") or h.get("score")} for h in hits if h]
        rag_method = "hybrid_bm25_hyde" if use_hyde else "bm25_only"
    except Exception as e:
        # Fallback 老 bi-encoder cosine path
        rag_xrd, rag_pl = _load_rags()
        if rag_xrd:
            rag_refs.extend(_retrieve_rag(rag_xrd, f"{formula} {dopant.get('element','')} XRD phase", top_k=2))
        if rag_pl:
            rag_refs.extend(_retrieve_rag(rag_pl, f"{formula} Cr3+ Ni2+ photoluminescence emission NIR", top_k=2))
        rag_method = f"bi_encoder_fallback ({type(e).__name__})"
    result["rag"] = rag_refs[:4]
    result["rag_method"] = rag_method
    result["timing_ms"]["rag"] = round((time.perf_counter() - ts) * 1000, 1)

    # ============ 9. Failure flags ============
    flags = compute_failure_flags(target, analog_comp, dopant)

    # 光谱参数硬条件检查 (Phase 4.1: NIR 应用)
    _pl_meta = result.get("virtual_pl_meta", {}) or {}
    _t_stab = _pl_meta.get("thermal_stability_pct_423K")
    _lam_em = _pl_meta.get("predicted_lambda_em_nm") or _pl_meta.get("lambda_em_nm")
    if _t_stab is not None and _t_stab < 40:
        flags.append({
            "level": "warn",
            "code": "low_thermal_stability",
            "message": f"热稳定性 I(423K)/I(298K)={_t_stab:.1f}% < 40% (Struck-Fonger 估算)",
            "recommendation": "可考虑换更强晶场 host (如 Y3Al5O12/Lu3Al5O12) 或引入共掺 Sensitizer 改善热稳定",
        })
    if _lam_em and _lam_em < 650:
        flags.append({
            "level": "warn",
            "code": "em_below_nir",
            "message": f"预测 λ_em={_lam_em:.0f} nm 低于 NIR-I 窗 (650 nm), 偏可见光",
            "recommendation": "换弱晶场 host (如 LaGaO3/CaSc2O4) 使 Dq 降低, λ_em 红移入 NIR",
        })
    if _lam_em and _lam_em > 950:
        flags.append({
            "level": "warn",
            "code": "em_beyond_nir1",
            "message": f"预测 λ_em={_lam_em:.0f} nm 超 NIR-I 窗 (>950 nm), 探测器响应差",
            "recommendation": "换更强晶场 host 或降低 dopant 浓度减少 Cr-Cr 相互作用",
        })

    result["flags"] = flags
    result["flag_severity"] = severity(flags)

    # ============ 10. 粗判 (给 R1 做起点; R1 可以推翻) ============
    result["heuristic_verdict"] = _heuristic_verdict(result)

    ts = time.perf_counter()
    try:
        from .flybrain import flybrain_verdict
        result["flybrain"] = flybrain_verdict(result)
    except Exception as e:
        result["flybrain"] = {"ok": False, "error": str(e)[:200]}
    result["timing_ms"]["flybrain"] = round((time.perf_counter() - ts) * 1000, 1)

    result["timing_ms"]["total"] = round((time.perf_counter() - t0) * 1000, 1)

    # ============ 11. 持久化 (M2.1) ============
    try:
        # 只存关键字段, 别把全部 BPU + RAG 都塞进 jsonl (太大)
        slim = {k: result[k] for k in (
            "trace_id", "formula", "dopant", "sinter_temp_C", "host_hint",
            "parse",   # ← 必须含: r1_judge._format_user_input 会访问 p['parse']
            "xrd_analog", "vegard", "xrd_method",  # xrd_method: mace_mpa_0 | vegard_1st_order | none
            "stages", "pl_analogs",
            "virtual_pl_meta", "flags", "flag_severity", "heuristic_verdict",
            "flybrain",
            "rag",     # R1 引用文献需要
            "timing_ms",
        ) if k in result}
        persistence.append_jsonl({
            "type": "partial",
            "trace_id": result["trace_id"],
            "formula": result["formula"],
            "dopant": result.get("dopant"),
            "payload": slim,
        })
    except Exception as e:
        print(f"[engine] persistence.append_jsonl 失败 (继续, 非致命): {e}")

    return result


def predict_batch(items: list[dict], max_workers: int = 4) -> dict:
    """M2.2: 批量预测.

    items: [{"formula": str, "dopant": dict, "host_hint": str|None, "sinter_temp_C": float|None}, ...]
    返回 {"batch_id": str, "results": [{"index", "ok", "error", ...predict_partial}]}.

    R1 不在批量里跑 (太慢), 调用方各自 trace_id 走 SSE.
    """
    import concurrent.futures as cf
    batch_id = "batch:" + hashlib.sha1(
        f"{time.time()}{len(items)}".encode()).hexdigest()[:10]
    results: list[dict] = [None] * len(items)

    def _one(i: int, item: dict) -> tuple[int, dict]:
        try:
            r = predict(
                formula=item["formula"],
                dopant=item.get("dopant", {}),
                sinter_temp_C=item.get("sinter_temp_C"),
                host_hint=item.get("host_hint"),
            )
            r["batch_id"] = batch_id
            r["batch_index"] = i
            return i, r
        except Exception as e:
            import traceback
            return i, {"ok": False, "error": str(e), "batch_id": batch_id,
                       "batch_index": i, "formula": item.get("formula"),
                       "traceback": traceback.format_exc()[:500]}

    with cf.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="batch") as exe:
        futs = {exe.submit(_one, i, it): i for i, it in enumerate(items)}
        for fut in cf.as_completed(futs):
            i, r = fut.result()
            results[i] = r

    # 持久化批量索引
    try:
        idx_path = persistence.BATCHES_DIR / f"{batch_id.replace(':','_')}.json"
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump({
                "batch_id": batch_id,
                "timestamp": time.time(),
                "n_items": len(items),
                "trace_ids": [r.get("trace_id") for r in results if r],
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[engine.predict_batch] 写 batch 索引失败: {e}")

    return {"batch_id": batch_id, "n_items": len(items), "results": results}


def predict_matrix(
    formula: str,
    scan: dict,
    sinter_temp_C: float | None = None,
    host_hint: str | None = None,
    max_cells: int = 30,
    max_workers: int = 4,
) -> dict:
    """M3.1: 把 1 个 host + scan 维度 → N×M×K cells, 每 cell 跑一遍 predict.

    Args:
        formula: "Y3ZnGa3GeO12"
        scan: {"dopant_element": ["Cr3+", "Ni2+"],
               "dopant_site": ["Ga", "Al"],
               "dopant_pct": [0.5, 0.75, 1.0, 1.5]}
        max_cells: 上限 (avoid 跑爆 X5 BPU 队列), 超出截断
    """
    elements = scan.get("dopant_element") or ["Cr3+"]
    sites = scan.get("dopant_site") or ["Ga"]
    pcts = scan.get("dopant_pct") or [0.75]
    cells = []
    for el in elements:
        for site in sites:
            for pct in pcts:
                if len(cells) >= max_cells:
                    break
                # 解析 element/valence
                import re as _re
                m = _re.match(r"^([A-Z][a-z]?)(\d*)\+?$", str(el).strip())
                if m:
                    e_name = m.group(1)
                    val = int(m.group(2)) if m.group(2) else 3
                    sym = el
                else:
                    e_name = el; val = 3; sym = el
                cells.append({
                    "formula": formula,
                    "dopant": {"element": e_name, "valence": val, "site": site,
                              "pct": float(pct), "symbol": sym},
                    "sinter_temp_C": sinter_temp_C,
                    "host_hint": host_hint,
                })

    matrix_id = "matrix:" + hashlib.sha1(
        f"{formula}{time.time()}{len(cells)}".encode()).hexdigest()[:10]

    out = predict_batch(cells, max_workers=max_workers)

    # 写矩阵索引
    try:
        idx_path = persistence.BATCHES_DIR / f"{matrix_id.replace(':','_')}.json"
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump({
                "matrix_id": matrix_id,
                "formula": formula,
                "scan": scan,
                "cells": [{"trace_id": r.get("trace_id"),
                          "dopant": r.get("dopant"),
                          "verdict": (r.get("heuristic_verdict") or {}).get("verdict"),
                          "confidence": (r.get("heuristic_verdict") or {}).get("confidence")}
                         for r in out["results"] if r],
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[predict_matrix] 写索引失败: {e}")

    return {
        "matrix_id": matrix_id,
        "formula": formula,
        "scan": scan,
        "n_cells": len(cells),
        "results": out["results"],
        "batch_id": out["batch_id"],
    }


def prewarm() -> dict:
    """启动时调用, 提前加载 RAG + 类比池, 避免首次预测 + 并发批量的冷启动延迟."""
    t0 = time.perf_counter()
    info = {}
    try:
        from .analog_lookup import _load_pool, _load_pl_table
        pool = _load_pool()
        info["pool_keys"] = list(pool.keys())
        info["pool_total_entries"] = sum(len(v) for v in pool.values())
        rows = _load_pl_table()
        info["pl_table_rows"] = len(rows)
    except Exception as e:
        info["analog_load_error"] = str(e)
    try:
        rag_x, rag_p = _load_rags()
        info["rag_xrd_loaded"] = rag_x is not None
        info["rag_pl_loaded"] = rag_p is not None
    except Exception as e:
        info["rag_load_error"] = str(e)
    info["prewarm_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return info


def _heuristic_verdict(p: dict) -> dict:
    """本地启发式判决, 不调 R1 也能给结果 (离线/降级用).

    PL 类比 Top-1 相似度足够高 (≥0.8) 时信号比 candidate_pool 更可靠 —
    直接用实测 xrd_result 驱动判决. 新增: 热稳定性/λ_em 过阈值调低 confidence.
    """
    sev = p.get("flag_severity", "ok")
    if sev == "error":
        return {"verdict": "REVISE", "confidence": 0.4, "reason": "失败旗帜 (error 级)"}

    # TS 光谱参数过低检查 (竞赛 NIR 荧光粉应用硬条件)
    pl_meta = p.get("virtual_pl_meta", {}) or {}
    t_stab = pl_meta.get("thermal_stability_pct_423K")
    lam_em = pl_meta.get("predicted_lambda_em_nm") or pl_meta.get("lambda_em_nm")
    _spec_penalty = 0.0
    if t_stab is not None and t_stab < 50:
        _spec_penalty += 0.2
    if lam_em and (lam_em < 650 or lam_em > 900):
        _spec_penalty += 0.15  # 不在 NIR-I 窗内

    pl_top = (p.get("pl_analogs") or [{}])[0]
    pl_sim = pl_top.get("similarity", 0)
    pl_xrd = (pl_top.get("xrd_result") or "").lower()

    # 强 PL 类比 (有同配方或同 host+掺杂已测过) → 直接用实测判
    if pl_sim >= 0.8:
        if pl_xrd == "pure":
            return {"verdict": "GO", "confidence": 0.75,
                    "reason": f"Top-1 PL 实测 {pl_top.get('formula')} 已验证纯相 (相似度 {pl_sim:.2f})"}
        if pl_xrd == "mixed":
            return {"verdict": "REVISE", "confidence": 0.65,
                    "reason": f"Top-1 PL 实测 {pl_top.get('formula')} 呈杂相, 需调参"}
        # unknown/amorphous
        return {"verdict": "REVISE", "confidence": 0.5,
                "reason": f"Top-1 PL 类比 {pl_top.get('formula')} XRD 未定 ({pl_xrd or 'unknown'})"}

    # BPU 信号为主的判决
    bpu_x = p.get("stages", {}).get("bpu_xrd_num", {})
    xrd_a = p.get("xrd_analog") or {}

    if not xrd_a or xrd_a.get("similarity", 0) < 0.4:
        # XRD 类比弱 + PL 类比也弱 → UNKNOWN
        if pl_sim < 0.4:
            return {"verdict": "UNKNOWN", "confidence": 0.3,
                    "reason": "XRD + PL 类比都弱, 建议 PC 侧扩 candidate_pool 或做小样探索"}
        # PL 类比中等 (0.4-0.8), 没 BPU XRD 可证, 保守 REVISE
        return {"verdict": "REVISE", "confidence": 0.45,
                "reason": f"仅中等 PL 类比 (sim={pl_sim:.2f}), XRD 无可靠参考"}

    if bpu_x.get("ok") and bpu_x.get("prob", 0) < 0.5:
        return {"verdict": "DROP", "confidence": 0.6,
                "reason": f"BPU xrd_num 主相概率 {bpu_x.get('prob'):.2f} < 0.5"}
    if bpu_x.get("ok") and bpu_x.get("prob", 0) >= 0.7:
        verdict = "REVISE" if _spec_penalty >= 0.3 else "GO"
        conf = max(0.45, 0.7 - _spec_penalty)
        reason_parts = [f"BPU xrd_num {bpu_x.get('prob'):.2f} + XRD 类比 sim={xrd_a.get('similarity')}"]
        if _spec_penalty > 0:
            if t_stab is not None and t_stab < 50:
                reason_parts.append(f"T_stab={t_stab}% 偏低")
            if lam_em and (lam_em < 650 or lam_em > 900):
                reason_parts.append(f"λ_em={lam_em}nm 超 NIR 窗")
        return {"verdict": verdict, "confidence": conf,
                "reason": "; ".join(reason_parts)}
    return {"verdict": "REVISE", "confidence": 0.5,
            "reason": "BPU 置信度中等 + 类比数据不足"}
