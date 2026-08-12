"""Reusable public DTO helpers for redaction, status, and provenance.

The functions here are pure with respect to application state: importing this
module does not touch Flask, SQLite, threads, subprocesses, network, or files.
"""

from __future__ import annotations

import datetime
import json
import re


STATUS_TAXONOMY = {
    "operational": {"label": "Operational", "source": "live", "rank": 0},
    "degraded": {"label": "Degraded", "source": "stale", "rank": 1},
    "mirror": {"label": "Mirror", "source": "mirror", "rank": 2},
    "replay": {"label": "Replay", "source": "replay", "rank": 3},
    "planned": {"label": "Planned", "source": "planned", "rank": 3},
    "offline": {"label": "Offline", "source": "offline", "rank": 4},
    "unknown": {"label": "Unknown", "source": "unknown", "rank": 5},
}

_PUBLIC_STATES = frozenset({"live", "mirror", "replay", "mock", "stale", "offline", "unknown", "planned"})


def public_safe_text(value, limit: int = 220) -> str:
    text = str(value or "")
    text = re.sub(
        r"(?i)(api[_-]?key|token|secret|password|passwd|authorization|bearer)\s*[:=]\s*[^,\s;]+",
        r"\1=[redacted]",
        text,
    )
    text = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "sk-[redacted]", text)
    return text[:limit]


def public_asset_text(value, limit: int = 180) -> str:
    text = public_safe_text(value, limit)
    text = re.sub(r"\b(?:(?:\d{1,3}|[xX*])\.){3}(?:\d{1,3}|[xX*])\b", "[ip-redacted]", text)
    text = re.sub(r"(?<!\d)\.\d{1,3}\b", "[address-redacted]", text)
    text = re.sub(
        r"(?i)(?:桥接|中继|二跳|静态邻居表|cron\s*守护|(?:network\s+)?overlay|frp[c]?|tunnel)",
        "[network-topology]",
        text,
    )
    text = re.sub(r"(?i)(?:TIM\d+(?:_CH\d+)?|P[A-H]\d+|GPIO|PWM|SERVO|/dev/[A-Za-z0-9_./-]+)", "[hardware-detail]", text)
    text = re.sub(r"(?<![A-Za-z0-9]):\d{2,5}(?![A-Za-z0-9])", ":[port-redacted]", text)
    text = re.sub(r"\b\d{2,5}/(?:tcp|udp)\b", "[port-redacted]", text, flags=re.I)
    text = re.sub(r"(?i)\b(?:systemd|socket\s+activation|lb_policy|forward_auth)\b", "[service-detail]", text)
    return text[:limit]


def public_runbook_text(value, limit: int = 220) -> str:
    text = public_asset_text(value, limit)
    if re.search(r"(?i)\b(ssh|systemctl|journalctl|curl|ss\s+-|frpc?|caddy|rollback\.sh|start_llamas|bash\s+~)", text):
        return "operator_runbook_required"
    text = re.sub(r"(?i)\b(WorkCockpit|NavCockpit|Caddy|frp|frpc)\b", "protected-gateway", text)
    return text[:limit]


def public_redaction_scan(obj) -> dict:
    raw = json.dumps(obj, ensure_ascii=False)
    deny = [
        r"\b(?:10|172\.(?:1[6-9]|2\d|3[0-1])|192\.168|127\.0\.0\.1|43\.129)\.",
        r"(?<!\d)\.\d{1,3}\b",
        r"(?<![A-Za-z0-9]):\d{2,5}(?![A-Za-z0-9])",
        r"(?i)\b(?:ssh|systemctl|journalctl|ss\s+-|frpc?|rollback\.sh|deploy_staged\.sh)\b",
        r"(?i)\b(?:systemd|socket\s+activation|forward_auth|lb_policy)\b",
        r"(?i)(?:TIM\d+|P[A-H]\d+|GPIO|PWM|/dev/[A-Za-z0-9_./-]+)",
    ]
    hits = [pattern for pattern in deny if re.search(pattern, raw)]
    return {"scan_pass": not hits, "denylist_hits": len(hits)}


def public_asset_group(group) -> dict:
    keep = {"key", "icon", "name", "host", "ip", "children", "serving", "real_ms", "mirror_ms"}
    sanitized = {key: value for key, value in dict(group or {}).items() if key in keep}
    if "host" in sanitized:
        sanitized["host"] = public_asset_text(sanitized["host"], 160)
    if "ip" in sanitized:
        sanitized["ip"] = "public-safe redacted"
        sanitized["network_note"] = "private/public addresses are not exposed on the public evidence API"
    children = []
    for child in sanitized.get("children", []) or []:
        row = {key: child.get(key) for key in ("id", "name", "kind", "spec", "status", "maint_n") if key in child}
        row["id"] = re.sub(r"(?i)\bfrpc?\b", "gateway", public_asset_text(row.get("id"), 80))
        row["name"] = re.sub(r"(?i)\bfrpc?\b", "protected gateway", public_asset_text(row.get("name"), 120))
        row["kind"] = public_asset_text(row.get("kind"), 100)
        row["spec"] = public_asset_text(row.get("spec"), 190)
        row["status"] = public_asset_text(row.get("status"), 120)
        children.append(row)
    sanitized["children"] = children
    return sanitized


def mask_ip(ip) -> str:
    text = str(ip or "-").strip()
    if not text or text == "-":
        return "-"
    if ":" in text:
        parts = text.split(":")
        return ":".join(parts[:2]) + ":****"
    parts = text.split(".")
    if len(parts) == 4:
        return ".".join([parts[0], parts[1], "x", "x"])
    return text[:3] + "***"


def public_severity(status, *, test_mode: bool = False):
    del test_mode  # Test mode must preserve the production DTO shape.
    try:
        status_int = int(status)
    except Exception:
        return "info"
    if status_int >= 500:
        return "critical"
    if status_int >= 400:
        return "warning"
    return "info"


def route_service(route) -> str:
    text = str(route or "")
    if text.startswith("/api/twin") or text.startswith("/twin"):
        return "twin"
    if text.startswith("/api/workorders") or text.startswith("/tasks") or text.startswith("/api/tasks"):
        return "tasks"
    if text.startswith("/api/log") or text.startswith("/api/trace") or text.startswith("/logs") or text.startswith("/traces"):
        return "logs"
    if text.startswith("/api/fleet") or text.startswith("/fleet"):
        return "fleet"
    if text.startswith("/api/metrics") or text.startswith("/api/observability") or text.startswith("/observability"):
        return "observability"
    if text.startswith("/api/materials") or text.startswith("/api/predictions"):
        return "research"
    if text.startswith("/api/public_status") or text.startswith("/status"):
        return "status"
    return "portal"


def serving_source(serving, age_s=None) -> str:
    if age_s is not None and age_s > 300 and serving not in ("down", None):
        return "stale"
    return {"real": "live", "mirror": "mirror", "down": "offline"}.get(serving, "unknown")


def status_meta(key: str) -> dict:
    return dict(STATUS_TAXONOMY.get(key, STATUS_TAXONOMY["unknown"]))


def status_from_serving(serving, latency_ms=None) -> dict:
    if serving == "real":
        if latency_ms is not None and latency_ms > 3000:
            return status_meta("degraded")
        return status_meta("operational")
    if serving == "mirror":
        return status_meta("mirror")
    if serving == "down":
        return status_meta("offline")
    return status_meta("unknown")


def status_envelope(
    state,
    source=None,
    checked_at=None,
    ttl_s: int = 90,
    error=None,
    confidence=None,
    *,
    release: str,
    now: float | None = None,
) -> dict:
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).timestamp() if now is None else float(now)
    checked = float(checked_at) if checked_at else None
    age_s = max(0, round(timestamp - checked, 1)) if checked else None
    freshness = "unknown" if age_s is None else ("fresh" if age_s <= ttl_s else "stale")
    normalized = str(state or "unknown").lower()
    if freshness == "stale" and normalized not in {"offline", "planned"}:
        normalized = "stale"
    if normalized not in _PUBLIC_STATES:
        normalized = "unknown"
    confidence_map = {
        "high": "verified",
        "medium": "reported",
        "low": "unknown",
        "verified": "verified",
        "reported": "reported",
        "inferred": "inferred",
        "unknown": "unknown",
    }
    confidence_value = confidence_map.get(str(confidence or "").lower())
    if not confidence_value:
        confidence_value = "verified" if freshness == "fresh" else "unknown"
    checked_text = (
        datetime.datetime.fromtimestamp(checked, tz=datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        if checked else None
    )
    return {
        "state": normalized,
        "source": str(source or normalized),
        "checked_at": checked_text,
        "age_s": age_s,
        "freshness": freshness,
        "confidence": confidence_value,
        "error": str(error)[:240] if error else None,
        "release": release,
    }
