"""persistence.py — predictions.jsonl append-only 持久化 + SHA hash 链.

Schema 三种 record type:
  type='partial'     ← /api/predict 返回时写, 含 trace_id + 启发式 + 4 BPU + flags
  type='r1_verdict'  ← /api/predict_stream R1 完成时写, 含 trace_id + verdict
  type='actual'      ← /api/actual 用户回填实测时写, 含 trace_id + 实测字段

Hash 链: 每条 record 的 hash = SHA256(json(record_without_hash) + prev_hash).
启动时校验链完整性 (打印警告但不阻塞).

复用 xrd_vision/visual_line/deploy_xrd_system.py:1326-1328 的 SHA hash 链思路.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

# 持久化路径 (X5 + PC 都试)
_PRED_DIR_CANDIDATES = [
    Path("/home/rdk/predictions"),
    Path(__file__).parent.parent / "predictions",
    Path("./predictions"),
]


def _resolve_dir() -> Path:
    for p in _PRED_DIR_CANDIDATES:
        try:
            p.mkdir(parents=True, exist_ok=True)
            # 测试可写
            (p / ".write_test").touch()
            (p / ".write_test").unlink()
            return p
        except Exception:
            continue
    # 最后 fallback: 临时目录
    import tempfile
    p = Path(tempfile.gettempdir()) / "predictions"
    p.mkdir(parents=True, exist_ok=True)
    return p


PRED_DIR = _resolve_dir()
JSONL_PATH = PRED_DIR / "predictions.jsonl"
ACTUALS_CSV_PATH = PRED_DIR / "actuals.csv"
BATCHES_DIR = PRED_DIR / "batches"
BATCHES_DIR.mkdir(parents=True, exist_ok=True)

_WRITE_LOCK = threading.Lock()
_LAST_HASH_CACHE: str | None = None


def _compute_hash(record: dict, prev_hash: str) -> str:
    """SHA256(json(record_without_hash) + '|' + prev_hash) → 16 hex."""
    record_no_hash = {k: v for k, v in record.items() if k not in ("hash", "hash_prev")}
    payload = json.dumps(record_no_hash, sort_keys=True, ensure_ascii=False, default=str)
    h = hashlib.sha256(f"{payload}|{prev_hash}".encode("utf-8")).hexdigest()
    return h[:16]


def _read_last_hash() -> str:
    """读 jsonl 最后一行的 hash; 不存在返回 'genesis'."""
    global _LAST_HASH_CACHE
    if _LAST_HASH_CACHE is not None:
        return _LAST_HASH_CACHE
    if not JSONL_PATH.exists():
        _LAST_HASH_CACHE = "genesis"
        return _LAST_HASH_CACHE
    last_line = ""
    try:
        with open(JSONL_PATH, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                _LAST_HASH_CACHE = "genesis"
                return _LAST_HASH_CACHE
            # 反向递增窗口找完整最后一行 (partial payload 可超 4KB —
            # 固定 4KB 窗截断长行 → json 解析失败回落 genesis → 链分段, 2026-06-11 修)
            chunk = min(4096, size)
            while True:
                f.seek(-chunk, os.SEEK_END)
                data = f.read().decode("utf-8", errors="ignore")
                lines = [l for l in data.splitlines() if l.strip()]
                # 窗口已覆盖全文件, 或窗口含 ≥2 行 (说明最后一行的行首换行在窗内 = 完整)
                if chunk >= size or len(lines) >= 2:
                    if lines:
                        last_line = lines[-1]
                    break
                chunk = min(size, chunk * 4)
    except Exception:
        _LAST_HASH_CACHE = "genesis"
        return _LAST_HASH_CACHE
    try:
        rec = json.loads(last_line)
        _LAST_HASH_CACHE = rec.get("hash", "genesis")
    except json.JSONDecodeError:
        _LAST_HASH_CACHE = "genesis"
    return _LAST_HASH_CACHE


def append_jsonl(record: dict) -> dict:
    """Append 一条带 hash 链的 record. 返回带 hash + hash_prev 的新 record."""
    global _LAST_HASH_CACHE
    with _WRITE_LOCK:
        prev = _read_last_hash()
        record = dict(record)
        record.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
        record["hash_prev"] = prev
        record["hash"] = _compute_hash(record, prev)
        try:
            with open(JSONL_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            _LAST_HASH_CACHE = record["hash"]
        except Exception as e:
            print(f"[persistence] append_jsonl 失败: {e}")
    return record


def load_recent(n: int = 100) -> list[dict]:
    """读取最近 n 条 (任意 type). 用于启动时 warm 内存缓存."""
    if not JSONL_PATH.exists():
        return []
    out = []
    try:
        with open(JSONL_PATH, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[persistence] load_recent 失败: {e}")
    return out[-n:]


def load_all() -> list[dict]:
    """读全部 (M2.5 accuracy_report 用)."""
    if not JSONL_PATH.exists():
        return []
    out = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def query(verdict: str | None = None, since: str | None = None,
          until: str | None = None, formula_contains: str | None = None,
          page: int = 1, per_page: int = 50) -> dict:
    """分页 + 多维过滤. 返回 {items, total, page, per_page, total_pages}."""
    all_recs = load_all()

    # 把 partial / r1_verdict / actual 按 trace_id 聚合
    by_trace: dict[str, dict] = OrderedDict()
    for r in all_recs:
        tid = r.get("trace_id")
        if not tid:
            continue
        if tid not in by_trace:
            by_trace[tid] = {"trace_id": tid, "partial": None, "r1": None, "actual": None,
                              "ts_first": r.get("timestamp")}
        t = r.get("type", "")
        if t == "partial":
            by_trace[tid]["partial"] = r
            by_trace[tid]["ts_first"] = r.get("timestamp", by_trace[tid]["ts_first"])
        elif t == "r1_verdict":
            by_trace[tid]["r1"] = r
        elif t == "actual":
            by_trace[tid]["actual"] = r

    items = list(by_trace.values())

    # 过滤
    def _keep(it):
        if verdict:
            v = (it.get("r1") or {}).get("verdict") or \
                (it.get("partial") or {}).get("heuristic_verdict", {}).get("verdict")
            if (v or "").upper() != verdict.upper():
                return False
        ts = it.get("ts_first") or ""
        if since and ts < since:
            return False
        if until and ts > until:
            return False
        if formula_contains:
            f = (it.get("partial") or {}).get("formula", "")
            if formula_contains.lower() not in (f or "").lower():
                return False
        return True

    items = [it for it in items if _keep(it)]
    items.sort(key=lambda it: it.get("ts_first") or "", reverse=True)

    total = len(items)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page

    return {
        "items": items[start:end],
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
    }


def verify_chain() -> dict:
    """校验整个 jsonl hash 链完整性. 返回 {ok, broken_at, total_records}."""
    if not JSONL_PATH.exists():
        return {"ok": True, "broken_at": None, "total_records": 0}
    prev = "genesis"
    n = 0
    with open(JSONL_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                return {"ok": False, "broken_at": i, "total_records": n,
                        "reason": "invalid_json"}
            n += 1
            if rec.get("hash_prev") != prev:
                return {"ok": False, "broken_at": i, "total_records": n,
                        "reason": "hash_prev_mismatch"}
            expected = _compute_hash(rec, prev)
            if rec.get("hash") != expected:
                return {"ok": False, "broken_at": i, "total_records": n,
                        "reason": "hash_mismatch"}
            prev = rec["hash"]
    return {"ok": True, "broken_at": None, "total_records": n}


# ============ Actuals CSV ============
ACTUAL_FIELDS = [
    "trace_id", "formula", "dopant", "actual_xrd_result",
    "actual_lambda_em_nm", "actual_fwhm_nm",
    "actual_thermal_stability_pct", "actual_quantum_yield_pct",
    "measured_by", "measurement_date", "notes",
]


def append_actual(record: dict) -> None:
    """Append 一行实测到 actuals.csv (建表头自动写)."""
    import csv
    new_file = not ACTUALS_CSV_PATH.exists()
    with _WRITE_LOCK:
        with open(ACTUALS_CSV_PATH, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ACTUAL_FIELDS)
            if new_file:
                w.writeheader()
            row = {k: record.get(k, "") for k in ACTUAL_FIELDS}
            for k, v in row.items():
                if v is None:
                    row[k] = ""
            w.writerow(row)


def load_actuals() -> dict[str, dict]:
    """{trace_id: actual_record}."""
    if not ACTUALS_CSV_PATH.exists():
        return {}
    import csv
    out = {}
    with open(ACTUALS_CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = row.get("trace_id")
            if tid:
                out[tid] = row
    return out


# ============ 内存 LRU 缓存 (替代 dashboard 里的 _PRED_CACHE) ============
class PredCache:
    """OrderedDict + 文件 jsonl 双层缓存. 只保存 partial result, R1 verdict 单独存."""

    def __init__(self, max_items: int = 200):
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._max = max_items
        self._lock = threading.Lock()

    def put(self, trace_id: str, payload: dict) -> None:
        with self._lock:
            if trace_id in self._cache:
                self._cache.move_to_end(trace_id)
            self._cache[trace_id] = payload
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)

    def get(self, trace_id: str) -> dict | None:
        with self._lock:
            v = self._cache.get(trace_id)
            if v is not None:
                self._cache.move_to_end(trace_id)
            return v

    def warm_from_jsonl(self, n: int = 100) -> int:
        """启动时从 jsonl 拉最近 n 条 partial 进内存. 返回加载条数."""
        recs = load_recent(n * 3)  # 多读点, 因为 jsonl 含 r1_verdict / actual 混合
        n_loaded = 0
        for r in recs:
            if r.get("type") == "partial":
                tid = r.get("trace_id")
                payload = r.get("payload")
                if tid and payload:
                    self.put(tid, payload)
                    n_loaded += 1
        return n_loaded

    def __len__(self):
        with self._lock:
            return len(self._cache)
