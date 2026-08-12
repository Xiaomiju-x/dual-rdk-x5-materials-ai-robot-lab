"""Read-only summaries for AI-brain prediction, actual, and campaign records.

This module intentionally avoids importing ``predict_engine.persistence`` because
that module resolves writable paths at import time. Everything here takes paths
explicitly or resolves local files without creating them.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> str:
    return str(value)


def compute_record_hash(record: dict[str, Any], prev_hash: str) -> str:
    """Compute the same short SHA-256 hash used by persistence.append_jsonl."""
    record_no_hash = {k: v for k, v in record.items() if k not in ("hash", "hash_prev", "_line")}
    payload = json.dumps(record_no_hash, sort_keys=True, ensure_ascii=False, default=_json_default)
    return hashlib.sha256(f"{payload}|{prev_hash}".encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(records, errors)`` from a JSONL file without raising on bad lines."""
    if not path.exists():
        return [], [{"line": 0, "error": "missing_file", "path": str(path)}]
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"line": lineno, "error": "invalid_json", "detail": str(exc)})
                continue
            if not isinstance(rec, dict):
                errors.append({"line": lineno, "error": "not_object"})
                continue
            rec["_line"] = lineno
            records.append(rec)
    return records, errors


def verify_hash_chain(path: Path) -> dict[str, Any]:
    """Verify append-only hash chain. Missing file is reported, not fatal."""
    records, errors = read_jsonl(path)
    if errors and errors[0].get("error") == "missing_file":
        return {"ok": True, "missing": True, "total_records": 0, "errors": errors}

    prev = "genesis"
    segments = 0
    tampered: list[dict[str, Any]] = []
    for idx, rec in enumerate(records, 1):
        hash_prev = str(rec.get("hash_prev") or "")
        link_ok = hash_prev == prev
        expected = compute_record_hash(rec, hash_prev)
        hash_ok = rec.get("hash") == expected
        seg_start = (not link_ok) and hash_prev == "genesis" and hash_ok
        if idx == 1 or seg_start:
            segments += 1
        if not (hash_ok and (link_ok or seg_start)):
            tampered.append(
                {
                    "line": rec.get("_line", idx),
                    "reason": "hash_mismatch" if not hash_ok else "hash_prev_mismatch",
                    "expected_prev": prev,
                    "actual_prev": rec.get("hash_prev"),
                    "expected_hash": expected,
                    "actual_hash": rec.get("hash"),
                }
            )
        prev = str(rec.get("hash"))

    return {
        "ok": not errors and not tampered,
        "missing": False,
        "broken_at": tampered[0]["line"] if tampered else None,
        "total_records": len(records),
        "last_hash": prev,
        "segments": segments,
        "tampered_count": len(tampered),
        "tampered_sample": tampered[:20],
        "errors": errors,
    }


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], str | None]:
    if not path.exists():
        return [], "missing_file"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle)), None


def summarize_prediction_chain(root: Path) -> dict[str, Any]:
    """Summarize predictions.jsonl and actuals.csv without writing anything."""
    pred_dir = root / "predictions"
    jsonl_path = pred_dir / "predictions.jsonl"
    actuals_path = pred_dir / "actuals.csv"

    records, jsonl_errors = read_jsonl(jsonl_path)
    actual_rows, actuals_error = read_csv_rows(actuals_path)
    chain = verify_hash_chain(jsonl_path)

    type_counts = Counter(str(r.get("type") or "unknown") for r in records)
    trace_counts = Counter(str(r.get("trace_id") or "") for r in records if r.get("trace_id"))
    partial_trace_counts = Counter(
        str(r.get("trace_id") or "") for r in records if r.get("type") == "partial" and r.get("trace_id")
    )
    duplicate_partial_trace_ids = sorted(tid for tid, count in partial_trace_counts.items() if count > 1)
    record_multiplicity = [
        {"trace_id": tid, "records": count}
        for tid, count in trace_counts.most_common(20)
        if count > 1
    ]

    partials: dict[str, dict[str, Any]] = {}
    verdicts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    jsonl_actuals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_required: list[dict[str, Any]] = []

    for rec in records:
        rec_type = rec.get("type")
        tid = str(rec.get("trace_id") or "")
        if rec_type == "partial":
            payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else rec
            if tid:
                partials[tid] = payload
            for key in ("formula", "dopant"):
                if not payload.get(key):
                    missing_required.append({"line": rec.get("_line"), "trace_id": tid, "missing": key})
        elif str(rec_type).startswith("r1_verdict"):
            if tid:
                verdicts[tid].append(rec)
        elif rec_type in ("actual", "campaign_actual"):
            if tid:
                jsonl_actuals[tid].append(rec)

    actual_trace_ids = {str(row.get("trace_id") or "") for row in actual_rows if row.get("trace_id")}
    partial_trace_ids = set(partials)
    joined_actuals = sorted(tid for tid in actual_trace_ids if tid in partial_trace_ids)

    return {
        "schema": "prediction_chain_summary.v1",
        "root": str(root),
        "files": {
            "predictions_jsonl": str(jsonl_path),
            "actuals_csv": str(actuals_path),
        },
        "hash_chain": chain,
        "jsonl": {
            "total_records": len(records),
            "type_counts": dict(type_counts),
            "errors": jsonl_errors,
            "duplicate_partial_trace_id_count": len(duplicate_partial_trace_ids),
            "duplicate_partial_trace_ids_sample": duplicate_partial_trace_ids[:20],
            "record_multiplicity_sample": record_multiplicity,
            "missing_required_sample": missing_required[:30],
        },
        "partials": {
            "count": len(partials),
            "with_r1_verdict": sum(1 for tid in partial_trace_ids if verdicts.get(tid)),
            "with_jsonl_actual": sum(1 for tid in partial_trace_ids if jsonl_actuals.get(tid)),
        },
        "actuals_csv": {
            "count": len(actual_rows),
            "error": actuals_error,
            "trace_id_count": len(actual_trace_ids),
            "joined_to_partial_count": len(joined_actuals),
            "joined_trace_ids_sample": joined_actuals[:20],
        },
    }


def summarize_closed_loop(root: Path) -> dict[str, Any]:
    """Summarize observed PL rows, actual rows, campaigns, and calibration caches."""
    observed_path = root / "exp_ground_truth" / "observed_pl.csv"
    actuals_path = root / "predictions" / "actuals.csv"
    campaigns_path = root / "predictions" / "campaigns.json"
    conformal_path = root / "crystal_data_shared" / "conformal_cache.json"
    mc_conformal_path = root / "crystal_data_shared" / "mc_conformal_cache.json"
    ml_cache_dir = root / "crystal_data_shared" / "ml_cache"
    generated_dir = root / "crystal_data_shared" / "generated"

    observed_rows, observed_error = read_csv_rows(observed_path)
    actual_rows, actuals_error = read_csv_rows(actuals_path)

    campaign_count = 0
    campaign_rounds = 0
    campaign_measured = 0
    campaign_error = None
    if campaigns_path.exists():
        try:
            data = json.loads(campaigns_path.read_text(encoding="utf-8"))
            campaigns = data.get("campaigns", []) if isinstance(data, dict) else []
            campaign_count = len(campaigns)
            for campaign in campaigns:
                for round_rec in campaign.get("rounds", []):
                    campaign_rounds += 1
                    campaign_measured += sum(
                        1 for pick in round_rec.get("picks", []) if pick.get("actual_nm") is not None
                    )
        except Exception as exc:  # noqa: BLE001 - report-only tool
            campaign_error = str(exc)
    else:
        campaign_error = "missing_file"

    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"missing": True}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"error": "not_object"}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    conformal = _load_json(conformal_path)
    mc_conformal = _load_json(mc_conformal_path)

    return {
        "schema": "closed_loop_summary.v1",
        "root": str(root),
        "observed_pl": {
            "path": str(observed_path),
            "count": len(observed_rows),
            "error": observed_error,
        },
        "actuals": {
            "path": str(actuals_path),
            "count": len(actual_rows),
            "error": actuals_error,
        },
        "campaigns": {
            "path": str(campaigns_path),
            "count": campaign_count,
            "rounds": campaign_rounds,
            "measured_picks": campaign_measured,
            "error": campaign_error,
        },
        "calibration": {
            "conformal_n": conformal.get("n_calibration"),
            "conformal_quantile_90": conformal.get("quantile_90"),
            "mc_conformal_n": mc_conformal.get("n_calibration"),
            "mc_conformal_avg_half_width_90": mc_conformal.get("avg_half_width_90"),
        },
        "artifacts": {
            "ml_cache_json_count": len(list(ml_cache_dir.glob("*.json"))) if ml_cache_dir.exists() else 0,
            "generated_cif_count": len(list(generated_dir.rglob("*.cif"))) if generated_dir.exists() else 0,
        },
        "claim_boundary": (
            "Closed-loop scientific improvement is not established until real actual rows exist "
            "and campaign measurements are recorded."
        ),
    }
