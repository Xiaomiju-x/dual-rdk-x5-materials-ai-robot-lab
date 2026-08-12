"""failure_library.py — 实测回填后挖掘失败案例 (Phase 3.2)

输入: predictions/predictions.jsonl (R1 verdict) + predictions/actuals.csv (实测回填)
输出:
  - predictions/failure_patterns.json (结构化失败案例库)
  - 在 R1 system prompt 注入 "最近 N 条失败案例" 段 (>= 20 条才启用)

判错条件:
  - GO + actual_xrd != "pure" → 误肯定 (会浪费实验)
  - DROP + actual_xrd == "pure" → 误否定 (错过好材料)
  - REVISE + actual 二选一 都视为 "需要重审"

R1 prompt 学习模式 (>= 20 case 启用):
  "<historical_failures>
   - Case 1: GO Y3InO12+Cr@In 1% but actual mixed (radius mismatch >12%)
   - Case 2: DROP Sr2ScNbO6+Cr@Sc 0.5% but actual pure (你低估了 perovskite Cr@Sc 兼容性)
   ...
   </historical_failures>"
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = REPO_ROOT / "predictions"
FAILURE_PATTERNS = PREDICTIONS_DIR / "failure_patterns.json"

# X5 端真实路径 (load 时备查)
X5_PREDICTIONS_DIR = Path("/home/rdk/predictions")


def _resolve_predictions_dir() -> Path:
    """X5 端用 /home/rdk/predictions, PC 端用 repo predictions/."""
    if X5_PREDICTIONS_DIR.exists():
        return X5_PREDICTIONS_DIR
    return PREDICTIONS_DIR


def _load_actuals(actuals_csv: Path) -> dict[str, dict]:
    """trace_id → actual record. 缺 trace_id 行跳过."""
    if not actuals_csv.exists():
        return {}
    actuals = {}
    with open(actuals_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("trace_id", "").strip()
            if tid:
                actuals[tid] = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
    return actuals


def _scan_predictions(jsonl: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """trace_id → (partial, r1_verdict). 只取最后一条 r1_verdict per trace."""
    partials = {}
    verdicts = {}
    if not jsonl.exists():
        return partials, verdicts
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            tid = rec.get("trace_id")
            if not tid:
                continue
            t = rec.get("type")
            if t == "partial":
                partials[tid] = rec.get("payload", rec)
            elif t in ("r1_verdict", "r1_verdict_sc"):
                verdicts[tid] = rec.get("verdict", rec)
    return partials, verdicts


def classify_failure(verdict_label: str, actual_xrd: str) -> str | None:
    """返回 failure type 或 None (不是失败)."""
    v = (verdict_label or "").upper()
    a = (actual_xrd or "").lower()
    if not v or not a:
        return None
    if v == "GO" and a in ("mixed", "amorphous"):
        return "false_positive"  # 推荐了不该做的
    if v == "DROP" and a == "pure":
        return "false_negative"  # 错过了好材料
    if v == "REVISE" and a == "pure":
        return "overcaution"  # 应该 GO 的, 但只让重审
    if v == "REVISE" and a in ("mixed", "amorphous"):
        return "correct_revise"  # 重审是对的, 不算失败
    return None


def build_failure_library(force: bool = False) -> dict:
    """主入口: 扫 jsonl + actuals, 归类失败 → 写 failure_patterns.json."""
    pred_dir = _resolve_predictions_dir()
    jsonl = pred_dir / "predictions.jsonl"
    actuals_csv = pred_dir / "actuals.csv"

    actuals = _load_actuals(actuals_csv)
    partials, verdicts = _scan_predictions(jsonl)

    failures = []
    for tid, actual in actuals.items():
        verdict = verdicts.get(tid)
        partial = partials.get(tid)
        if not verdict:
            continue
        ftype = classify_failure(verdict.get("verdict"), actual.get("actual_xrd_result"))
        if ftype is None or ftype == "correct_revise":
            continue
        failures.append({
            "trace_id": tid,
            "type": ftype,
            "formula": (partial or {}).get("formula") or actual.get("formula"),
            "dopant": (partial or {}).get("dopant"),
            "predicted_verdict": verdict.get("verdict"),
            "predicted_confidence": verdict.get("confidence"),
            "predicted_lambda_em": (partial or {}).get("virtual_pl_meta", {}).get("predicted_lambda_em_nm"),
            "actual_xrd": actual.get("actual_xrd_result"),
            "actual_lambda_em": actual.get("actual_lambda_em_nm"),
            "actual_fwhm": actual.get("actual_fwhm_nm"),
            "flags": [f.get("code") for f in (partial or {}).get("flags", []) if isinstance(f, dict)],
            "heuristic_verdict": (partial or {}).get("heuristic_verdict", {}).get("verdict"),
            "key_signal_summary": _extract_key_signals(partial),
        })

    failures.sort(key=lambda x: x.get("trace_id", ""), reverse=True)

    out = {
        "schema": "failure_patterns.v1",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "n_total_actuals": len(actuals),
        "n_failures": len(failures),
        "n_by_type": dict(Counter([f["type"] for f in failures])),
        "failures": failures[:50],  # 只存最近 50 条
    }
    FAILURE_PATTERNS.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILURE_PATTERNS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def _extract_key_signals(partial: dict | None) -> str:
    """从 partial 提取关键信号一句话总结 (供 R1 prompt 注入用)."""
    if not partial:
        return ""
    parts = []
    veg = partial.get("vegard", {})
    if veg.get("method") == "mace_mpa_0":
        parts.append(f"MACE form_E={veg.get('energy_per_atom_eV')} eV/at, K={veg.get('bulk_modulus_GPa')} GPa")
    pl = partial.get("virtual_pl_meta", {})
    if pl.get("method") == "tanabe_sugano_huang_rhys":
        parts.append(f"TS λ_em={pl.get('predicted_lambda_em_nm')} nm")
    bpu = partial.get("stages", {})
    xrd_prob = (bpu.get("bpu_xrd_num", {}) or {}).get("prob")
    if xrd_prob is not None:
        parts.append(f"BPU xrd_num prob={xrd_prob}")
    flags = partial.get("flags", [])
    if flags:
        parts.append(f"flags={[f.get('code') for f in flags[:3] if isinstance(f, dict)]}")
    return "; ".join(parts)


def render_for_r1_prompt(min_cases: int = 20, n_recent: int = 10) -> str:
    """为 R1 system prompt 生成 "<historical_failures>...</historical_failures>" 段.

    返回空串 (跳过注入) 若失败案例不足 min_cases.
    """
    if not FAILURE_PATTERNS.exists():
        return ""
    try:
        data = json.loads(FAILURE_PATTERNS.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if data.get("n_failures", 0) < min_cases:
        return ""
    fails = data.get("failures", [])[:n_recent]
    if not fails:
        return ""
    lines = ["<historical_failures>",
             f"# 历史失败案例库 (最近 {len(fails)}/{data['n_failures']} 条; 总实测 {data['n_total_actuals']})",
             "# Format: type | formula+dopant | predicted vs actual | key signals"]
    for f in fails:
        dop = f.get("dopant", {}) or {}
        dop_str = f"{dop.get('element','?')}@{dop.get('site','?')} {dop.get('pct','?')}%"
        lines.append(
            f"- {f['type']}: {f['formula']} + {dop_str} | "
            f"R1 said {f['predicted_verdict']} ({f['predicted_confidence']}) "
            f"but实测 xrd={f['actual_xrd']} λ_em={f['actual_lambda_em']}nm | "
            f"signals: {f.get('key_signal_summary','')}"
        )
    lines.append("</historical_failures>")
    return "\n".join(lines)


# 辅助 (Counter 内联避免顶部 import 干扰)
from collections import Counter  # noqa: E402


if __name__ == "__main__":
    out = build_failure_library()
    print(f"[failure_library] 写 {out['n_failures']} 失败案例 → {FAILURE_PATTERNS.relative_to(REPO_ROOT)}")
    print(f"  by type: {out['n_by_type']}")
    print(f"  total actuals: {out['n_total_actuals']}")
    if out["n_failures"] >= 20:
        print(f"\n[r1_prompt segment]:\n{render_for_r1_prompt()}")
    else:
        print(f"  <{20} cases, R1 注入未启用 (data['n_failures']={out['n_failures']})")
