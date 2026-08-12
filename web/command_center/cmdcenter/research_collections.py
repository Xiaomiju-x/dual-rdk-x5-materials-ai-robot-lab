"""Pure public research collection projection for Site32.

The module has no Flask, file, database, thread, subprocess, or network side
effects. Callers inject already-public material and evidence records; this
module still projects them through an explicit allowlist before returning a
collection payload.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable, Mapping
from urllib.parse import quote


SCHEMA_VERSION = "site32.research_collections.v1"
PROJECTION_VERSION = "site32.public_collection_projection.v1"
MAX_QUERY_LENGTH = 160
MAX_LIMIT = 50

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_SLUG_RE = re.compile(r"[^a-z0-9._-]+")
_PROHIBITED_KEYS = frozenset({
    "work_order", "batch", "wo_log", "operator", "user", "email", "ip",
    "host_address", "port", "ssh", "route", "network", "device_path",
    "gpio", "pwm", "actuator", "cmd_vel", "control_url", "raw_command",
    "token", "secret", "private_prompt", "raw_log",
})
_MATERIAL_FIELDS = (
    "formula", "host", "dopant", "site", "verdict", "lambda_em",
    "confidence_interval", "band", "method", "uncertainty",
    "metadata_completeness_score",
)


COLLECTION_SPECS = (
    {
        "collection_id": "rc:xrd:materials-atlas",
        "title": "近红外荧光材料参考集",
        "title_en": "NIR Phosphor Reference Collection",
        "description": "按化学式、基质、掺杂、波段与来源浏览公开材料对象。",
        "description_en": "Browse public material objects by formula, host, dopant, band and source.",
        "scope": "ai_brain",
        "topics": ["nir-phosphor", "materials", "xrd", "pl"],
        "featured": True,
        "display_order": 10,
        "member_rules": ["page:atlas", "materials", "evidence:ev:xrd:materials"],
        "limitations": ["公开集合规模有限；策展或回放记录不能替代原始实验与来源许可复核。"],
        "limitations_en": ["The public collection is limited in scale; curated or replay records do not replace primary experiments or source-license review."],
    },
    {
        "collection_id": "rc:xrd:prediction-review",
        "title": "配方预测与不确定性复核",
        "title_en": "Prediction and Uncertainty Review",
        "description": "集中查看 AI 判决方法、系统卡、适用范围与实验验证边界。",
        "description_en": "Review AI verdict methods, system cards, intended use and experimental validation limits.",
        "scope": "ai_brain",
        "topics": ["prediction", "conformal", "fly-mb", "mlip"],
        "featured": True,
        "display_order": 20,
        "member_rules": ["page:brain", "evidence:ev:xrd:prediction_engine"],
        "limitations": ["预测建议不替代烧制、XRD 与 PL 实测。"],
        "limitations_en": ["Predictions do not replace synthesis, XRD or PL measurements."],
    },
    {
        "collection_id": "rc:xrd:public-evidence",
        "title": "公开证据与复现入口",
        "title_en": "Public Evidence and Reproduction",
        "description": "把科研护照、系统卡、来源、限制、引用与公开下载组织为可复核对象。",
        "description_en": "Organize the research passport, system cards, provenance, limits, citations and public downloads as reviewable objects.",
        "scope": "public_site",
        "topics": ["evidence", "citation", "provenance", "reproduction"],
        "featured": True,
        "display_order": 30,
        "member_rules": ["page:defense", "evidence:all"],
        "limitations": ["项目证据门禁不是第三方科研认证、安全认证或全球排名。"],
        "limitations_en": ["Project evidence gates are not third-party scientific certification, security certification or a global ranking."],
    },
    {
        "collection_id": "rc:xrd:embodied-replay",
        "title": "具身脑真机闭环与 Lab-FSD 影子证据",
        "title_en": "Embodied Real-hardware Loop and Lab-FSD Shadow Evidence",
        "description": "复核取瓶、升顶、0.50m 里程计闭环、放瓶复位与 SLAM 回放；Lab-FSD 保持 shadow/assist。",
        "description_en": "Review the bottle-handling, lift, 0.50 m odometry loop, release/reset and SLAM replay while Lab-FSD remains shadow/assist.",
        "scope": "embodied_brain",
        "topics": ["slam", "lab-fsd", "shadow", "replay"],
        "featured": False,
        "display_order": 40,
        "member_rules": ["page:fsd", "page:replay", "evidence:ev:xrd:slam_shadow"],
        "limitations": ["真机闭环由冻结执行链完成；Lab-FSD 保持 shadow/assist，公开面和算法均不持有底盘执行权。"],
        "limitations_en": ["The frozen execution chain completed the hardware loop; Lab-FSD remains shadow/assist and neither the public surface nor the algorithm holds chassis authority."],
    },
    {
        "collection_id": "rc:xrd:arm01-redundancy",
        "title": "双机械臂复赛协同与视觉门控",
        "title_en": "Finals Dual-arm Collaboration and Visual Gate",
        "description": "复核 arm01 单臂视觉冗余、投袋与 arm02 并发四周期研磨的真机证据。",
        "description_en": "Review real-hardware evidence for arm01 visual redundancy and bag drop with arm02 concurrent four-cycle grinding.",
        "scope": "arm01",
        "topics": ["arm01", "arm02", "dual-arm", "visual-gate", "grinding", "replay"],
        "featured": False,
        "display_order": 50,
        "member_rules": ["page:replay", "evidence:ev:xrd:arm01_redundancy"],
        "limitations": ["袋状态以 X5 CPU/OpenCV 判定为权威；BPU 仅作辅助语义与执行证据，公开面不下发动作。"],
        "limitations_en": ["X5 CPU/OpenCV is authoritative for bag state; BPU is supporting semantic and execution evidence only, and the public surface issues no motion commands."],
    },
)

_PAGES = {
    "atlas": ("材料图鉴", "Materials Atlas", "/atlas", "发现公开材料、配方、表征与来源"),
    "brain": ("AI 脑解释", "AI Brain Explain", "/brain", "复核预测方法、不确定性与实验边界"),
    "defense": ("答辩证据地图", "Evidence Map", "/defense", "核对主张、来源、限制和公开边界"),
    "fsd": ("FSD 世界模型", "FSD World Model", "/fsd", "查看 SLAM 与 shadow/assist 证据"),
    "replay": ("实验回放", "Experiment Replay", "/replay", "查看公开只读实验与具身回放"),
}


def _text(value: Any, limit: int = 220) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = _CONTROL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _same_origin(path: Any) -> str:
    value = _text(path, 320)
    if not value.startswith("/") or value.startswith("//") or "://" in value:
        return ""
    return value


def _slug(value: Any) -> str:
    return _SLUG_RE.sub("-", _text(value, 120).casefold()).strip("-._")


def _material_id(row: Mapping[str, Any]) -> str:
    existing = _slug(row.get("id") or row.get("trace_id"))
    if existing and not re.fullmatch(r"(?:atlas|observed|wo)-\d+", existing):
        return f"mat:xrd:{existing}"
    identity = "|".join(_text(row.get(key), 120).casefold() for key in (
        "formula", "dopant", "site", "lambda_em", "created", "source"
    ))
    return "mat:xrd:sha256-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _page_member(key: str, release: str, released_at: str) -> dict[str, Any]:
    title, title_en, href, purpose = _PAGES[key]
    return {
        "object_id": f"page:xrd:{key}",
        "kind": "page",
        "title": title,
        "title_en": title_en,
        "subtitle": purpose,
        "canonical_url": href,
        "source_label": "site-navigation",
        "state": "live",
        "release": release,
        "properties": {"page_key": key, "href": href, "purpose": purpose},
        "provenance": _provenance("site-navigation", f"page:xrd:{key}", href, release, released_at),
        "limitations": ["页面只展示公开只读信息。"],
        "limitations_en": ["The page presents public read-only information only."],
        "relations": [],
    }


def _provenance(mode: str, source_id: str, endpoint: str, release: str, released_at: str) -> dict[str, Any]:
    return {
        "origin_mode": mode,
        "source_label": mode,
        "source_object_id": source_id,
        "source_endpoint": _same_origin(endpoint),
        "as_of": released_at,
        "release": release,
        "projection": PROJECTION_VERSION,
        "snapshot_semantics": "as-released",
    }


def _project_material(row: Mapping[str, Any], release: str, released_at: str) -> dict[str, Any]:
    object_id = _material_id(row)
    source = _text(row.get("source"), 48).casefold() or "unknown"
    state = _text(row.get("state"), 48).casefold() or "unknown"
    properties = {key: row.get(key) for key in _MATERIAL_FIELDS if row.get(key) not in (None, "")}
    formula = _text(row.get("formula"), 120) or "Unknown material"
    canonical = _same_origin(row.get("detail_url")) or "/materials/" + quote(_text(row.get("id"), 160), safe="")
    return {
        "object_id": object_id,
        "kind": "material",
        "title": formula,
        "title_en": formula,
        "subtitle": " · ".join(filter(None, [_text(row.get("dopant"), 48), _text(row.get("band"), 48), _text(row.get("verdict"), 48)])),
        "canonical_url": canonical,
        "source_label": source,
        "state": state,
        "release": release,
        "properties": properties,
        "provenance": _provenance(source, object_id, canonical, release, released_at),
        "limitations": [_text(row.get("uncertainty"), 220) or "公开记录未提供定量不确定性。"],
        "limitations_en": ["See the source label and object detail; the public row may not provide quantified uncertainty."],
        "relations": [],
    }


def _project_evidence(row: Mapping[str, Any], release: str, released_at: str) -> dict[str, Any]:
    evidence_id = _text(row.get("evidence_id"), 160)
    source = _text(row.get("source_label"), 48).casefold() or "unknown"
    canonical = _same_origin(row.get("canonical_url")) or "/evidence/" + quote(evidence_id, safe="")
    uncertainty = row.get("uncertainty") if isinstance(row.get("uncertainty"), Mapping) else {}
    rights = row.get("rights") if isinstance(row.get("rights"), Mapping) else {}
    properties = {
        "evidence_id": evidence_id,
        "evidence_kind": _text(row.get("kind"), 64),
        "scope": _text(row.get("scope"), 64),
        "claim_status": _text(row.get("claim_status"), 64),
        "validation_status": _text(row.get("validation_status"), 64),
        "origin": [_text(value, 64) for value in row.get("origin", []) if _text(value, 64)],
        "uncertainty": _text(uncertainty.get("statement"), 240),
        "rights": {"license": _text(rights.get("license"), 120), "access": _text(rights.get("access"), 64)},
    }
    limitations = [_text(value, 240) for value in row.get("limitations", []) if _text(value, 240)]
    limitations_en = [_text(value, 240) for value in row.get("limitations_en", []) if _text(value, 240)]
    return {
        "object_id": evidence_id,
        "kind": "evidence",
        "title": _text(row.get("title"), 180),
        "title_en": _text(row.get("title_en"), 180),
        "subtitle": _text(row.get("description"), 220),
        "canonical_url": canonical,
        "source_label": source,
        "state": _text((row.get("freshness") or {}).get("state"), 48).casefold() or source,
        "release": release,
        "properties": properties,
        "provenance": _provenance(source, evidence_id, canonical, release, released_at),
        "limitations": limitations or ["见证据对象详情中的适用范围与限制。"],
        "limitations_en": limitations_en or ["See intended use and limitations in the evidence object detail."],
        "relations": [
            {"relation_type": _text(rel.get("relation_type"), 64), "target": _same_origin(rel.get("target"))}
            for rel in row.get("relations", []) if isinstance(rel, Mapping) and _same_origin(rel.get("target"))
        ],
    }


def _assert_public_projection(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _text(key, 80).casefold()
            if normalized in _PROHIBITED_KEYS:
                raise ValueError(f"prohibited public field: {path}.{normalized}")
            _assert_public_projection(child, f"{path}.{normalized}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_public_projection(child, f"{path}[{index}]")


def build_research_collections(
    *,
    materials: Iterable[Mapping[str, Any]],
    evidence_objects: Iterable[Mapping[str, Any]],
    release: str,
    released_at: str,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    query = params or {}
    q = _text(query.get("q"), MAX_QUERY_LENGTH).casefold()
    scope = _text(query.get("scope"), 64).casefold()
    topic = _text(query.get("topic"), 64).casefold()
    has_kind = _text(query.get("has_kind"), 32).casefold()
    try:
        limit = min(max(int(query.get("limit", 20)), 1), MAX_LIMIT)
    except (TypeError, ValueError):
        limit = 20

    material_members = [_project_material(row, release, released_at) for row in materials]
    evidence_members = [_project_evidence(row, release, released_at) for row in evidence_objects]
    evidence_by_id = {item["object_id"]: item for item in evidence_members}
    collections: list[dict[str, Any]] = []
    for spec in COLLECTION_SPECS:
        members: list[dict[str, Any]] = []
        for rule in spec["member_rules"]:
            family, _, key = rule.partition(":")
            if family == "page" and key in _PAGES:
                members.append(_page_member(key, release, released_at))
            elif rule == "materials":
                members.extend(material_members)
            elif family == "evidence" and key == "all":
                members.extend(evidence_members)
            elif family == "evidence" and key in evidence_by_id:
                members.append(evidence_by_id[key])
        unique = {member["object_id"]: member for member in members}
        members = sorted(unique.values(), key=lambda item: (item["kind"], item["title"].casefold(), item["object_id"]))
        counts = dict(sorted(Counter(item["kind"] for item in members).items()))
        item = {
            key: spec[key] for key in (
                "collection_id", "title", "title_en", "description", "description_en",
                "scope", "topics", "featured", "display_order", "limitations", "limitations_en",
            )
        }
        item.update({
            "collection_kind": "curated_public_view",
            "visibility": "public-read-only",
            "member_count": len(members),
            "counts_by_kind": counts,
            "updated_at": released_at,
            "canonical_url": "/api/research_collections/" + quote(spec["collection_id"], safe=""),
            "browse_url": "/?collection=" + quote(spec["collection_id"], safe=""),
            "provenance": {
                "source_endpoints": ["/api/materials/explorer", "/api/evidence_objects"],
                "release": release,
                "projection": PROJECTION_VERSION,
                "snapshot_semantics": "as-released",
            },
            "members": members,
        })
        haystack = " ".join([item["collection_id"], item["title"], item["title_en"], item["description"], item["description_en"], *item["topics"]]).casefold()
        if q and q not in haystack:
            continue
        if scope and item["scope"].casefold() != scope:
            continue
        if topic and topic not in [value.casefold() for value in item["topics"]]:
            continue
        if has_kind and not item["counts_by_kind"].get(has_kind):
            continue
        collections.append(item)

    collections.sort(key=lambda item: (not item["featured"], item["display_order"], item["collection_id"]))
    total = len(collections)
    collections = collections[:limit]
    facets = {
        "scope": dict(sorted(Counter(item["scope"] for item in collections).items())),
        "topic": dict(sorted(Counter(topic for item in collections for topic in item["topics"]).items())),
        "kind": dict(sorted(Counter(kind for item in collections for kind in item["counts_by_kind"]).items())),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "release": release,
        "as_of": released_at,
        "query": {"q": q, "scope": scope, "topic": topic, "has_kind": has_kind},
        "total": total,
        "count": len(collections),
        "limit": limit,
        "next_cursor": None,
        "partial": False,
        "warnings": [],
        "facets": facets,
        "items": collections,
        "empty": None if collections else {"reason": "no_match", "message": "No public research collection matched the current filters."},
    }
    _assert_public_projection(payload)
    return payload


def collection_detail(payload: Mapping[str, Any], collection_id: str) -> dict[str, Any] | None:
    wanted = _text(collection_id, 160)
    for item in payload.get("items", []):
        if item.get("collection_id") == wanted:
            return {
                "schema_version": SCHEMA_VERSION,
                "release": payload.get("release"),
                "as_of": payload.get("as_of"),
                "item": item,
            }
    return None


__all__ = [
    "COLLECTION_SPECS",
    "PROJECTION_VERSION",
    "SCHEMA_VERSION",
    "build_research_collections",
    "collection_detail",
]
