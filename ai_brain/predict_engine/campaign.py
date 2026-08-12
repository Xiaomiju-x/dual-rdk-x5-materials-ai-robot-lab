"""Campaign 闭环工作台 — 目标 → GP/EI 推荐 → 预测 → 实测回填 → 下一轮.

第 2 期 #2+#3 (2026-06-11): 对标 Berkeley A-Lab 闭环.
- 一个 campaign = 一个优化目标 (target λ_em ± tol), 多轮 round 迭代
- 每轮: active_learning.fit_gp_and_recommend (GP + EI/UCB + diversity) 出 top-K 批次
- 回填实测走 persistence.append_actual → 下一轮 GP 训练集自动长大 = 真闭环
- #3 贝叶斯 EI: 对跑过完整引擎预测的 pick, 用 Conformal CI90 半宽算 σ
  (split conformal q_hat@90% ≈ 1.645σ 高斯等效) 重算 EI_conformal, 与 GP EI 并列

存储: predictions/campaigns.json (整文件读写 + threading.Lock, 单进程 Flask 够用)
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

# ---- repo root (同 active_learning.py 规则) ----
_HERE = Path(__file__).resolve().parent
_REPO_CANDS = [_HERE.parent, Path("/home/rdk"), Path.cwd()]
REPO_ROOT = next((p for p in _REPO_CANDS if (p / "exp_ground_truth").exists()), _HERE.parent)
CAMPAIGNS_PATH = REPO_ROOT / "predictions" / "campaigns.json"

_LOCK = threading.Lock()

# z_{0.95} = 1.645: 90% 双侧区间半宽 ≈ 1.645σ 高斯等效
# Source: standard normal quantile, e.g. Vovk et al. Algorithmic Learning in a Random World (2005)
_Z90 = 1.645


# ============================================================ 存取
def _load() -> dict:
    if CAMPAIGNS_PATH.exists():
        try:
            return json.load(CAMPAIGNS_PATH.open(encoding="utf-8"))
        except Exception:
            pass
    return {"campaigns": []}


def _save(data: dict) -> None:
    CAMPAIGNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CAMPAIGNS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(CAMPAIGNS_PATH)


def _find(data: dict, cid: str) -> Optional[dict]:
    return next((c for c in data["campaigns"] if c["cid"] == cid), None)


# ============================================================ CRUD
def list_campaigns() -> list[dict]:
    with _LOCK:
        data = _load()
    out = []
    for c in data["campaigns"]:
        prog = compute_progress(c)
        out.append({
            "cid": c["cid"], "name": c["name"], "goal": c["goal"],
            "status": c.get("status", "active"), "created_at": c["created_at"],
            "n_rounds": len(c.get("rounds", [])),
            "n_measured": prog["n_measured_total"],
            "n_hits": prog["n_hits_total"],
            "best_abs_err_nm": prog["best_abs_err_nm"],
        })
    return out


def create_campaign(name: str, target_nm: float, tol_nm: float = 20.0,
                    dopant_element: str = "Cr", dopant_pct: float = 1.0,
                    notes: str = "") -> dict:
    cid = f"cg{int(time.time())}"
    camp = {
        "cid": cid, "name": (name or "未命名 Campaign").strip()[:60],
        "goal": {"target_nm": float(target_nm), "tol_nm": float(tol_nm),
                 "dopant_element": dopant_element, "dopant_pct": float(dopant_pct)},
        "notes": notes[:500], "status": "active",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rounds": [],
    }
    with _LOCK:
        data = _load()
        data["campaigns"].append(camp)
        _save(data)
    return camp


def get_campaign(cid: str) -> Optional[dict]:
    with _LOCK:
        c = _find(_load(), cid)
    if c:
        c = dict(c)
        c["progress"] = compute_progress(c)
    return c


def set_status(cid: str, status: str) -> bool:
    with _LOCK:
        data = _load()
        c = _find(data, cid)
        if not c:
            return False
        c["status"] = status
        _save(data)
    return True


# ============================================================ 闭环核心
def run_round(cid: str, k: int = 5, kappa: float = 2.0) -> dict:
    """跑一轮: GP (labeled=observed_pl+actuals, 回填后自动长大) → EI top-K."""
    with _LOCK:
        data = _load()
        c = _find(data, cid)
    if not c:
        return {"ok": False, "error": f"campaign {cid} 不存在"}
    if c.get("status") != "active":
        return {"ok": False, "error": "campaign 已归档, 先恢复 active"}

    from predict_engine.active_learning import fit_gp_and_recommend
    target = c["goal"]["target_nm"]
    res = fit_gp_and_recommend(top_k=int(k), target_lambda_nm=float(target),
                               kappa=float(kappa))
    if not res.get("ok"):
        return res

    round_n = len(c.get("rounds", [])) + 1
    picks = []
    for p in res["top5"]:
        picks.append({
            "formula": p["formula"],
            "dopant_element": p.get("dopant_element", "Cr"),
            "dopant_site": p.get("dopant_site", "Al"),
            "dopant_pct": p.get("dopant_pct", 1.0),
            "gp_mu_nm": p["predicted_lambda_em_nm"],
            "gp_sigma_nm": p["uncertainty_nm"],
            "EI": p["EI"], "UCB": p["UCB"], "cluster": p["cluster"],
            "source": p.get("source"), "why": p.get("why"),
            # 后续动作填充:
            "engine_lambda_nm": None, "engine_verdict": None,
            "conformal_sigma_nm": None, "EI_conformal": None, "trace_id": None,
            "actual_nm": None, "hit": None, "measured_at": None,
        })
    gs = res["gp_summary"]
    rec = {
        "round_n": round_n, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {"k": int(k), "kappa": float(kappa)},
        "gp": {"n_labeled": gs["n_labeled"], "n_unlabeled": gs["n_unlabeled"],
               "kernel": gs["learned_kernel"], "lml": gs["log_marginal_likelihood"],
               "sigma_mean_nm": round(gs["pred_sigma_mean_nm"], 1)},
        "picks": picks,
    }
    with _LOCK:
        data = _load()
        c = _find(data, cid)
        if not c:
            return {"ok": False, "error": "campaign 消失 (并发删除?)"}
        c.setdefault("rounds", []).append(rec)
        _save(data)
    return {"ok": True, "cid": cid, "round": rec}


def attach_engine_prediction(cid: str, round_n: int, formula: str,
                             pred_result: dict, trace_id: str = "") -> dict:
    """把完整引擎预测 (含 conformal CI90) 挂到 pick 上, 重算 EI_conformal (#3).

    EI_conformal: 目标接近型 y=-|λ−target|, σ 取 conformal_ci90.half_width/1.645
    (split conformal 90% 半宽的高斯等效), 纯 numpy 同 active_learning.expected_improvement.
    """
    import numpy as np
    from predict_engine.active_learning import expected_improvement

    pm = (pred_result or {}).get("virtual_pl_meta", {}) or {}
    lam = pm.get("predicted_lambda_em_nm") or pm.get("lambda_em_nm")
    ci = pm.get("conformal_ci90") or {}
    half = ci.get("half_width")
    verdict = (pred_result or {}).get("heuristic_verdict") or \
              (pred_result or {}).get("verdict") or ""
    if isinstance(verdict, dict):   # heuristic_verdict 是 {verdict, confidence, ...}
        verdict = verdict.get("verdict") or verdict.get("label") or ""

    with _LOCK:
        data = _load()
        c = _find(data, cid)
        if not c:
            return {"ok": False, "error": "campaign 不存在"}
        rnd = next((r for r in c.get("rounds", []) if r["round_n"] == int(round_n)), None)
        if not rnd:
            return {"ok": False, "error": f"round {round_n} 不存在"}
        pick = next((p for p in rnd["picks"] if p["formula"] == formula), None)
        if not pick:
            return {"ok": False, "error": f"{formula} 不在该轮 picks"}

        pick["trace_id"] = trace_id or pick.get("trace_id")
        pick["engine_verdict"] = str(verdict)[:24]
        if lam is not None:
            pick["engine_lambda_nm"] = round(float(lam), 1)
        if lam is not None and half:
            sigma_c = float(half) / _Z90
            pick["conformal_sigma_nm"] = round(sigma_c, 1)
            target = c["goal"]["target_nm"]
            # best_y: 已实测里离 target 最近的 λ; 没有实测就用 target±tol 边界当门槛
            measured = [p["actual_nm"] for r in c["rounds"] for p in r["picks"]
                        if p.get("actual_nm") is not None]
            best_y = (min(measured, key=lambda v: abs(v - target))
                      if measured else target + c["goal"]["tol_nm"])
            ei = expected_improvement(np.array([float(lam)]), np.array([sigma_c]),
                                      best_y=float(best_y), target=float(target))
            pick["EI_conformal"] = round(float(ei[0]), 4)
        _save(data)
        return {"ok": True, "pick": pick}


def record_actual(cid: str, round_n: int, formula: str, actual_nm: float,
                  notes: str = "", measured_by: str = "") -> dict:
    """实测回填: 写 pick + persistence.append_actual → 下一轮 GP 自动学到 (闭环)."""
    with _LOCK:
        data = _load()
        c = _find(data, cid)
        if not c:
            return {"ok": False, "error": "campaign 不存在"}
        rnd = next((r for r in c.get("rounds", []) if r["round_n"] == int(round_n)), None)
        if not rnd:
            return {"ok": False, "error": f"round {round_n} 不存在"}
        pick = next((p for p in rnd["picks"] if p["formula"] == formula), None)
        if not pick:
            return {"ok": False, "error": f"{formula} 不在该轮 picks"}
        goal = c["goal"]
        pick["actual_nm"] = float(actual_nm)
        pick["hit"] = abs(float(actual_nm) - goal["target_nm"]) <= goal["tol_nm"]
        pick["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _save(data)

    # 落 actuals.csv (active_learning.load_labeled_pl 会读它 → 闭环长大)
    trace_id = pick.get("trace_id") or f"camp-{cid}-r{round_n}-{formula}"
    try:
        from predict_engine import persistence as _pers
        _pers.append_actual({
            "trace_id": trace_id, "formula": formula,
            "dopant": json.dumps({"element": pick.get("dopant_element", "Cr"),
                                  "site": pick.get("dopant_site", "Al"),
                                  "pct": pick.get("dopant_pct", 1.0)},
                                 ensure_ascii=False),
            "actual_lambda_em_nm": float(actual_nm),
            "measured_by": measured_by or "campaign",
            "measurement_date": time.strftime("%Y-%m-%d"),
            "notes": (notes or f"campaign {cid} round {round_n}")[:200],
        })
        _pers.append_jsonl({"type": "campaign_actual", "cid": cid,
                            "round_n": int(round_n), "trace_id": trace_id,
                            "formula": formula, "actual_lambda_em_nm": float(actual_nm)})
    except Exception as e:
        return {"ok": True, "pick": pick,
                "warn": f"pick 已记录但 actuals.csv 落盘失败: {e}"}
    return {"ok": True, "pick": pick, "trace_id": trace_id}


# ============================================================ 进度
def compute_progress(c: dict) -> dict:
    """收敛曲线数据: 每轮 best |λ−target| + 命中数."""
    target = c["goal"]["target_nm"]
    per_round, best_so_far = [], None
    n_meas_total = n_hits_total = 0
    for r in c.get("rounds", []):
        errs = [abs(p["actual_nm"] - target) for p in r["picks"]
                if p.get("actual_nm") is not None]
        hits = sum(1 for p in r["picks"] if p.get("hit"))
        n_meas_total += len(errs)
        n_hits_total += hits
        rb = min(errs) if errs else None
        if rb is not None:
            best_so_far = rb if best_so_far is None else min(best_so_far, rb)
        per_round.append({"round_n": r["round_n"], "n_picks": len(r["picks"]),
                          "n_measured": len(errs), "n_hits": hits,
                          "best_abs_err_nm": round(rb, 1) if rb is not None else None,
                          "best_so_far_nm": round(best_so_far, 1) if best_so_far is not None else None})
    return {"per_round": per_round, "n_measured_total": n_meas_total,
            "n_hits_total": n_hits_total,
            "best_abs_err_nm": round(best_so_far, 1) if best_so_far is not None else None}
