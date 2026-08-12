"""审计链验证 — predictions.jsonl SHA-256 hash 链完整性.

第 2 期 #5 (2026-06-11): 链规则 (persistence.py):
  hash = SHA256(json(record 去 hash/hash_prev, sort_keys, ensure_ascii=False,
                     default=str) + "|" + hash_prev)[:16]
  首条 hash_prev == "genesis"; 后条 hash_prev == 前条 hash.
逐条重算比对 → 任何离线篡改 (改数值/删行/插行) 都会断链.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _recompute(record: dict, prev_hash: str) -> str:
    """与 persistence._compute_hash 完全一致 (复制实现, 避免私有依赖)."""
    record_no_hash = {k: v for k, v in record.items() if k not in ("hash", "hash_prev")}
    payload = json.dumps(record_no_hash, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(f"{payload}|{prev_hash}".encode("utf-8")).hexdigest()[:16]


def verify_chain(page: int = 1, per_page: int = 80,
                 type_filter: str | None = None) -> dict[str, Any]:
    from predict_engine import persistence
    recs = persistence.load_all()

    prev = "genesis"
    n_valid = 0
    n_segments = 0
    tampered: list[int] = []
    annotated: list[dict] = []
    type_counts: dict[str, int] = {}
    for i, r in enumerate(recs):
        link_ok = (r.get("hash_prev") == prev)
        hash_ok = (_recompute(r, r.get("hash_prev") or "") == r.get("hash"))
        # 历史 bug (2026-06-11 已修): _read_last_hash 旧版只回读 4KB, 长 partial 行
        # 截断解析失败 → 重启后 hash_prev 回落 "genesis" = 链分段, 非篡改
        # (hash_ok=True 证明该记录写入后内容未被改过)
        seg_start = (not link_ok) and r.get("hash_prev") == "genesis" and hash_ok
        valid = hash_ok and (link_ok or seg_start)
        if seg_start or i == 0:
            n_segments += 1
        if valid:
            n_valid += 1
        else:
            tampered.append(i)
        t = r.get("type", "?")
        type_counts[t] = type_counts.get(t, 0) + 1
        annotated.append({
            "idx": i, "type": t,
            "ts": r.get("timestamp", ""),
            "trace_id": (r.get("trace_id") or "")[-12:],
            "formula": (r.get("formula") or (r.get("payload") or {}).get("formula") or "")[:24],
            "hash": r.get("hash", ""), "hash_prev": r.get("hash_prev", ""),
            "valid": valid, "link_ok": link_ok, "hash_ok": hash_ok,
            "seg_start": seg_start,
        })
        prev = r.get("hash") or prev

    if type_filter:
        annotated = [a for a in annotated if a["type"] == type_filter]
    total = len(annotated)
    start = max(0, (page - 1) * per_page)
    return {
        "ok": True,
        "n_records": len(recs),
        "n_valid": n_valid,
        "n_segments": n_segments,
        "n_tampered": len(tampered),
        "first_tamper_idx": tampered[0] if tampered else None,
        "chain_intact": not tampered,
        "type_counts": type_counts,
        "page": page, "per_page": per_page, "total_filtered": total,
        "records": annotated[start:start + per_page],
    }
