"""cove.py — Chain-of-Verification + Self-Consistency aggregation (Phase 1.3).

Two public functions:
  - `majority_vote_verdict(verdicts)` — simple vote on verdict label, returns
    (winning_verdict, vote_distribution, confidence_avg)
  - `build_cove_followup_prompt(verdicts, original_user_msg)` — when disagreement,
    produce a follow-up message asking R1 to verify the minority claims

References:
  - Self-Consistency: Wang et al. 2022, arXiv:2203.11171
  - Chain-of-Verification: Dhuliawala et al. 2023 (Meta), arXiv:2309.11495
"""
from __future__ import annotations

from collections import Counter
from typing import Optional


VALID_VERDICTS = ("GO", "REVISE", "DROP", "UNKNOWN")


def _verdict_label(v: Optional[dict]) -> str:
    """Safely extract verdict label, normalize."""
    if not isinstance(v, dict):
        return "UNKNOWN"
    label = (v.get("verdict") or "").upper().strip()
    return label if label in VALID_VERDICTS else "UNKNOWN"


def majority_vote_verdict(verdicts: list[dict]) -> dict:
    """Self-Consistency 多数投票.

    Args:
        verdicts: list of verdict dicts (some may be None for failed calls)

    Returns:
        {
            "winner": "GO" | "REVISE" | "DROP" | "UNKNOWN",
            "vote_dist": {"GO": 3, "REVISE": 2, ...},
            "n_samples": int,
            "n_valid": int (non-None),
            "consensus_strength": 0-1 (winner_count / n_valid),
            "chosen_sample": dict — 代表性完整 verdict (在 winner 中挑 confidence 最高的一份),
            "min_conf": float, "max_conf": float, "avg_conf": float,
        }
    """
    valid = [v for v in verdicts if isinstance(v, dict) and _verdict_label(v) in VALID_VERDICTS]
    n_samples = len(verdicts)
    n_valid = len(valid)

    if n_valid == 0:
        return {
            "winner": "UNKNOWN", "vote_dist": {},
            "n_samples": n_samples, "n_valid": 0,
            "consensus_strength": 0.0,
            "chosen_sample": None,
            "min_conf": 0.0, "max_conf": 0.0, "avg_conf": 0.0,
        }

    labels = [_verdict_label(v) for v in valid]
    dist = dict(Counter(labels))
    # 按 (票数, 平均置信度) 决胜
    def score_key(label: str):
        count = dist.get(label, 0)
        confs = [float(v.get("confidence") or 0) for v in valid if _verdict_label(v) == label]
        avg = sum(confs) / len(confs) if confs else 0.0
        return (count, avg)
    winner = max(dist.keys(), key=score_key)

    winner_samples = [v for v in valid if _verdict_label(v) == winner]
    # 挑置信度最高的一份作为代表
    chosen = max(winner_samples, key=lambda v: float(v.get("confidence") or 0))
    confs = [float(v.get("confidence") or 0) for v in valid]

    return {
        "winner": winner,
        "vote_dist": dist,
        "n_samples": n_samples,
        "n_valid": n_valid,
        "consensus_strength": round(dist.get(winner, 0) / max(n_valid, 1), 3),
        "chosen_sample": chosen,
        "min_conf": round(min(confs), 3),
        "max_conf": round(max(confs), 3),
        "avg_conf": round(sum(confs) / len(confs), 3),
    }


def needs_cove(vote_result: dict, min_consensus: float = 0.6) -> bool:
    """是否需要触发 CoVe?

    条件:
      1. 多数派票数 < 60% (3/5 为阈值)
      2. 或置信度差异 > 0.3 (max-min)
      3. 或只有 1 个有效样本 (n_valid <= 1, 无法互验证)
    """
    if vote_result["n_valid"] <= 1:
        return False  # 只有 1 票, CoVe 单轮问无用
    if vote_result["consensus_strength"] < min_consensus:
        return True
    conf_spread = vote_result["max_conf"] - vote_result["min_conf"]
    if conf_spread > 0.3:
        return True
    return False


def build_cove_followup_prompt(verdicts: list[dict], original_user_msg: str) -> str:
    """生成 CoVe 第二轮 prompt.

    让 R1 先列出每个样本的关键 claim, 再生成验证问题 → 回答 → 修订 verdict.
    """
    lines = ["## Self-Consistency 5 投票出现分歧, 触发 Chain-of-Verification\n"]
    for i, v in enumerate(verdicts, 1):
        if not isinstance(v, dict):
            lines.append(f"**样本 {i}**: 调用失败 / 无效")
            continue
        label = _verdict_label(v)
        conf = v.get("confidence", "?")
        rzn = (v.get("reasoning") or "")[:180]
        lines.append(f"**样本 {i}**: verdict={label} conf={conf} reasoning=\"{rzn}...\"")

    lines.append("\n## 任务:")
    lines.append("1. 列出 5 个样本**分歧的关键 claim** (例如 'BPU 概率 0.99 足够可信' vs '类比 similarity 0.3 太弱')")
    lines.append("2. 对每个分歧 claim, 生成**可验证的子问题** (3-5 个)")
    lines.append("3. 结合 MACE-MPA-0 热力学证据 + PL 实测类比, 回答每个子问题")
    lines.append("4. **最终给出修正后的单一 verdict** (调用 submit_synthesis_verdict 工具)")
    lines.append("5. 在 reasoning 字段里**必须**明确说明 'CoVe 分歧分析' + '采信的样本号'")

    lines.append("\n## 原始输入 (用于对比):\n")
    lines.append(original_user_msg[:2000])
    return "\n".join(lines)


def summarize_self_consistency(vote_result: dict, cove_verdict: Optional[dict] = None) -> dict:
    """汇总最终结果 (给 persistence + UI).

    Args:
        vote_result: majority_vote_verdict 的返回
        cove_verdict: 若触发 CoVe, 为 CoVe 第二轮返回的 verdict; 否则 None

    Returns:
        Final verdict dict + self_consistency_meta sub-field
    """
    if cove_verdict is not None and isinstance(cove_verdict, dict):
        final = dict(cove_verdict)
        final["self_consistency_meta"] = {
            "n_samples": vote_result["n_samples"],
            "n_valid": vote_result["n_valid"],
            "vote_dist": vote_result["vote_dist"],
            "consensus_strength": vote_result["consensus_strength"],
            "cove_triggered": True,
            "pre_cove_winner": vote_result["winner"],
        }
        return final

    chosen = vote_result["chosen_sample"] or {}
    final = dict(chosen)
    final["self_consistency_meta"] = {
        "n_samples": vote_result["n_samples"],
        "n_valid": vote_result["n_valid"],
        "vote_dist": vote_result["vote_dist"],
        "consensus_strength": vote_result["consensus_strength"],
        "cove_triggered": False,
    }
    return final
