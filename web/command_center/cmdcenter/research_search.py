"""Pure Site32 federated research search kernel.

The functions in this module are intentionally independent from Flask, files,
SQLite, subprocesses, threads, and network state.  The application layer may pass
in a richer corpus later; the bundled corpus is a small deterministic public
seed used for default behavior and regression tests.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlencode, urlsplit


SCHEMA_VERSION = "site32.research_search.v2"
DEFAULT_LANGUAGE = "zh-CN"
DEFAULT_LIMIT = 12
MAX_LIMIT = 50

KIND_LABELS = {
    "material": "材料",
    "prediction": "预测",
    "evidence": "证据对象",
    "page": "页面",
    "work_order": "工单",
}

STATUS_LABELS = {
    "live": "实时",
    "mirror": "镜像",
    "replay": "回放",
    "mock": "模拟",
    "stale": "过期",
    "offline": "离线",
    "unknown": "未知",
    "planned": "计划",
}

SOURCE_LABELS = {
    "curated": "人工整理",
    "mirror": "镜像",
    "replay": "回放",
    "site-navigation": "站内导航",
    "release-evidence": "发布证据",
    "history": "历史记录",
    "unknown": "未知来源",
}

DEFAULT_SUGGESTIONS = (
    "YAG:Cr3+",
    "GGG:Ni2+",
    "ev:xrd:materials",
    "材料图鉴",
    "public NIR phosphor materials dataset",
)

DEFAULT_CORPUS: tuple[dict[str, Any], ...] = (
    {
        "kind": "page",
        "id": "atlas",
        "title": "材料图鉴",
        "title_en": "Materials Atlas",
        "subtitle": "按化学式、host:dopant、波段、判决与来源检索",
        "href": "/atlas",
        "status": "live",
        "source": "site-navigation",
        "preview": "材料、配方、掺杂、预测 trace 与公开来源的只读入口。",
        "search_fields": {
            "keywords": "材料 图鉴 atlas phosphor formula host dopant prediction trace",
            "claim": "按化学式、掺杂和公开来源定位材料对象",
        },
    },
    {
        "kind": "material",
        "id": "seed-yag-cr3",
        "title": "Y3Al5O12:Cr3+",
        "formula": "Y3Al5O12:Cr3+",
        "host": "YAG",
        "dopant": "Cr3+",
        "site": "Al",
        "verdict": "REFERENCE",
        "lambda_em": 714.0,
        "band": "nir_i",
        "subtitle": "YAG:Cr3+ · REFERENCE · 714 nm",
        "href": "/materials/seed-yag-cr3",
        "status": "replay",
        "source": "curated",
        "preview": "Project TS seed; compare observed history when available.",
        "search_fields": {
            "host_dopant": "YAG:Cr3+",
            "aliases": "YAG Cr3+ chromium-doped YAG 近红外 荧光 粉体",
            "method": "Project TS seed; compare observed history when available",
        },
    },
    {
        "kind": "material",
        "id": "seed-ggg-ni2",
        "title": "Gd3Ga5O12:Ni2+",
        "formula": "Gd3Ga5O12:Ni2+",
        "host": "GGG",
        "dopant": "Ni2+",
        "site": "Ga",
        "verdict": "REFERENCE",
        "lambda_em": None,
        "band": "unknown",
        "subtitle": "GGG:Ni2+ · REFERENCE",
        "href": "/materials/seed-ggg-ni2",
        "status": "replay",
        "source": "curated",
        "preview": "Public example chip; no live claim.",
        "search_fields": {
            "host_dopant": "GGG:Ni2+",
            "aliases": "GGG Ni2+ nickel-doped gadolinium gallium garnet 镍 掺杂 石榴石",
            "method": "Public example chip; no live claim",
        },
    },
    {
        "kind": "evidence",
        "id": "ev:xrd:passport",
        "title": "基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人公开科研护照",
        "title_en": "Material-Synthesis AI Prediction and Multi-Robot Embodied Laboratory Assistant Based on Dual-RDK X5 Heterogeneous Collaboration Public Research Passport",
        "subtitle": "公开只读科研证据门户",
        "href": "/api/evidence_objects/ev%3Axrd%3Apassport",
        "status": "mirror",
        "source": "mirror",
        "preview": "说明受众、引用、证据、限制与 trust posture。",
        "search_fields": {
            "claim": "本站是面向全球材料科研工作者的公开只读科研证据门户。",
            "claim_en": "A public, read-only research evidence portal for materials researchers worldwide.",
            "kind": "document",
            "scope": "public_site",
        },
    },
    {
        "kind": "evidence",
        "id": "ev:xrd:materials",
        "title": "近红外荧光材料公开数据对象",
        "title_en": "Public NIR Phosphor Materials Dataset",
        "subtitle": "材料图鉴、CI、来源和导出路径组织为可引用对象",
        "href": "/api/evidence_objects/ev%3Axrd%3Amaterials",
        "status": "mirror",
        "source": "mirror",
        "preview": "公开材料记录连接预测、不确定性、provenance 和导出。",
        "search_fields": {
            "claim": "材料图鉴把公开材料、预测、CI、来源和导出路径组织为可引用对象。",
            "claim_en": "Public material records connect predictions, uncertainty, provenance and exports as citable objects.",
            "kind": "material_dataset",
            "scope": "ai_brain",
            "aliases": "NIR phosphor materials dataset fluorescence atlas public data",
        },
    },
    {
        "kind": "evidence",
        "id": "ev:xrd:prediction_engine",
        "title": "AI 脑预测引擎系统卡",
        "title_en": "AI Brain Prediction Engine System Card",
        "subtitle": "TS、MLIP、Conformal 与 Fly-MB 的公开边界",
        "href": "/api/evidence_objects/ev%3Axrd%3Aprediction_engine",
        "status": "mirror",
        "source": "mirror",
        "preview": "AI 预测不替代烧制、XRD、PL 实测。",
        "search_fields": {
            "claim": "AI 脑将 TS/MLIP/Conformal/Fly-MB 与本地 LLM/BPU 证据合并为配方判决。",
            "claim_en": "The AI brain combines TS, MLIP, conformal and Fly-MB evidence while preserving experimental limits.",
            "kind": "model_system_card",
            "scope": "ai_brain",
        },
    },
)


_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_SPACES_RE = re.compile(r"\s+")
_FILTER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,48}$")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[:+._-][a-z0-9]+)*|[\u4e00-\u9fff]{2,}", re.I)

_FIELD_WEIGHTS = {
    "id": 120,
    "evidence_id": 120,
    "trace_id": 110,
    "work_order": 100,
    "formula": 100,
    "host_dopant": 105,
    "host": 70,
    "dopant": 70,
    "title": 80,
    "title_en": 75,
    "subtitle": 50,
    "claim": 45,
    "claim_en": 45,
    "aliases": 40,
    "keywords": 35,
    "preview": 25,
    "method": 25,
}
_ID_FIELDS = frozenset({"id", "evidence_id", "trace_id", "work_order"})
_CHEMISTRY_FIELDS = frozenset({"formula", "host_dopant", "host", "dopant", "aliases"})
_SORT_RANK = {"material": 0, "evidence": 1, "prediction": 2, "work_order": 3, "page": 4}


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else ""
    return value


def _clean_text(value: Any, *, limit: int = 180) -> str:
    text = "" if value is None else str(_first(value))
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_RE.sub(" ", text)
    text = text.replace("\u2028", " ").replace("\u2029", " ")
    text = re.sub(r"[<>`]", "", text)
    text = _SPACES_RE.sub(" ", text).strip()
    return text[:limit]


def normalize_text(value: Any) -> str:
    return _clean_text(value, limit=500).casefold()


def _compact(value: Any) -> str:
    text = normalize_text(value)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff:+._-]+", "", text)


def _tokens(value: Any) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(normalize_text(value))]


def _join_search_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        parts: list[str] = []
        for key, inner in sorted(value.items()):
            parts.append(_join_search_value(key))
            parts.append(_join_search_value(inner))
        return " ".join(part for part in parts if part)
    if isinstance(value, (list, tuple, set, frozenset)):
        return " ".join(_join_search_value(item) for item in value)
    return _clean_text(value, limit=500)


def _clean_filter(value: Any, *, allowed: Mapping[str, str] | None = None) -> str:
    text = _clean_text(value, limit=64).casefold()
    if not text or not _FILTER_RE.fullmatch(text):
        return ""
    if allowed is not None and text not in allowed:
        return ""
    return text


def _clean_limit(value: Any) -> int:
    try:
        parsed = int(str(_first(value)).strip())
    except Exception:
        return DEFAULT_LIMIT
    return min(max(parsed, 1), MAX_LIMIT)


def normalize_search_params(params: Mapping[str, Any] | str | None = None) -> dict[str, Any]:
    """Return bounded, display-safe search parameters.

    ``params`` may be a Flask-like args mapping, a parsed query-string mapping,
    a plain mapping, or a raw query string.
    """

    if isinstance(params, str):
        raw: Mapping[str, Any] = {"q": params}
    else:
        raw = params or {}
    sort = _clean_filter(raw.get("sort"))
    if sort not in {"relevance", "title", "kind", "status", "source"}:
        sort = "relevance"
    return {
        "q": _clean_text(raw.get("q"), limit=160),
        "kind": _clean_filter(raw.get("kind"), allowed=KIND_LABELS),
        "status": _clean_filter(raw.get("status"), allowed=STATUS_LABELS),
        "source": _clean_filter(raw.get("source")),
        "limit": _clean_limit(raw.get("limit", DEFAULT_LIMIT)),
        "sort": sort,
    }


def is_same_origin_href(href: Any) -> bool:
    """Return True only for root-relative, same-origin hrefs."""

    text = _clean_text(href, limit=500)
    if not text or not text.startswith("/") or text.startswith("//") or "\\" in text:
        return False
    parsed = urlsplit(text)
    if parsed.scheme or parsed.netloc:
        return False
    decoded_path = unquote(parsed.path)
    if _CONTROL_RE.search(decoded_path):
        return False
    lowered = text.casefold()
    if "%2f" in lowered or "%5c" in lowered:
        return False
    segments = [segment for segment in decoded_path.split("/") if segment]
    if any(segment in {".", ".."} for segment in segments):
        return False
    return True


def safe_same_origin_href(href: Any, *, fallback: str = "/") -> str:
    return _clean_text(href, limit=500) if is_same_origin_href(href) else fallback


def _record_search_fields(record: Mapping[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key in (
        "id",
        "evidence_id",
        "trace_id",
        "work_order",
        "title",
        "title_en",
        "subtitle",
        "formula",
        "host",
        "dopant",
        "site",
        "verdict",
        "band",
        "preview",
    ):
        value = record.get(key)
        if value not in (None, ""):
            fields[key] = _join_search_value(value)
    nested = record.get("search_fields")
    if isinstance(nested, Mapping):
        for key, value in nested.items():
            safe_key = re.sub(r"[^a-z0-9_]+", "_", str(key).casefold()).strip("_")
            if safe_key:
                fields[safe_key] = _join_search_value(value)
    host = fields.get("host")
    dopant = fields.get("dopant")
    if host and dopant:
        fields.setdefault("host_dopant", f"{host}:{dopant}")
    return fields


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = _record_search_fields(record)
    item_id = _clean_text(record.get("id") or fields.get("id"), limit=120)
    kind = _clean_filter(record.get("kind"), allowed=KIND_LABELS) or "page"
    status = _clean_filter(record.get("status"), allowed=STATUS_LABELS) or "unknown"
    source = _clean_filter(record.get("source")) or "unknown"
    normalized = {
        "kind": kind,
        "kind_label": KIND_LABELS.get(kind, kind),
        "id": item_id,
        "title": _clean_text(record.get("title") or item_id, limit=160),
        "subtitle": _clean_text(record.get("subtitle"), limit=180),
        "href": safe_same_origin_href(record.get("href"), fallback="/"),
        "status": status,
        "source": source,
        "preview": _clean_text(record.get("preview"), limit=260),
        "search_fields": fields,
    }
    for key in ("formula", "host", "dopant", "site", "verdict", "band", "lambda_em"):
        if key in record:
            normalized[key] = record.get(key)
    return normalized


def _score_record(query: str, record: Mapping[str, Any]) -> tuple[int, list[str]]:
    q_norm = normalize_text(query)
    if not q_norm:
        return 0, []
    q_compact = _compact(query)
    tokens = _tokens(query)
    matched: list[str] = []
    score = 0
    fields = record.get("search_fields") or {}
    for field, value in fields.items():
        text_norm = normalize_text(value)
        if not text_norm:
            continue
        text_compact = _compact(value)
        weight = _FIELD_WEIGHTS.get(field, 20)
        field_score = 0
        if field in _ID_FIELDS and q_norm == text_norm:
            field_score += 10000
        elif field in _ID_FIELDS and q_compact and q_compact == text_compact:
            field_score += 9500
        elif q_compact and field in _CHEMISTRY_FIELDS and q_compact == text_compact:
            field_score += weight * 7
        elif q_norm == text_norm:
            field_score += weight * 4
        elif q_compact and field in _CHEMISTRY_FIELDS and q_compact in text_compact:
            field_score += weight * 3
        elif q_norm in text_norm:
            field_score += weight * 2

        token_hits = 0
        for token in tokens:
            token_compact = _compact(token)
            if token and (token in text_norm or (token_compact and token_compact in text_compact)):
                token_hits += 1
        if token_hits:
            field_score += token_hits * max(8, weight // 3)

        if field_score:
            score += field_score
            if field not in matched:
                matched.append(field)
    return score, matched


def _apply_filters(rows: Iterable[tuple[dict[str, Any], int, list[str]]], params: Mapping[str, Any]):
    for record, score, matched in rows:
        if params.get("kind") and record.get("kind") != params["kind"]:
            continue
        if params.get("status") and record.get("status") != params["status"]:
            continue
        if params.get("source") and record.get("source") != params["source"]:
            continue
        yield record, score, matched


def _sort_rows(rows: list[tuple[dict[str, Any], int, list[str]]], sort: str) -> None:
    if sort == "title":
        rows.sort(key=lambda row: (normalize_text(row[0].get("title")), -row[1], row[0].get("id", "")))
    elif sort in {"kind", "status", "source"}:
        rows.sort(key=lambda row: (row[0].get(sort, ""), -row[1], normalize_text(row[0].get("title"))))
    else:
        rows.sort(
            key=lambda row: (
                -row[1],
                _SORT_RANK.get(row[0].get("kind"), 99),
                normalize_text(row[0].get("title")),
                row[0].get("id", ""),
            )
        )


def _public_item(record: Mapping[str, Any], score: int, matched: list[str]) -> dict[str, Any]:
    item = {
        "kind": record.get("kind"),
        "kind_label": record.get("kind_label"),
        "id": record.get("id"),
        "title": record.get("title"),
        "subtitle": record.get("subtitle"),
        "href": record.get("href"),
        "status": record.get("status"),
        "source": record.get("source"),
        "matched_fields": matched,
        "preview": record.get("preview"),
        "score": score,
    }
    for key in ("formula", "host", "dopant", "site", "verdict", "band", "lambda_em"):
        if key in record:
            item[key] = record.get(key)
    return item


def build_share_params(params: Mapping[str, Any] | str | None = None) -> dict[str, str]:
    normalized = normalize_search_params(params)
    share: dict[str, str] = {}
    if normalized["q"]:
        share["q"] = normalized["q"]
    for key in ("kind", "status", "source"):
        if normalized[key]:
            share[key] = normalized[key]
    if normalized["sort"] != "relevance":
        share["sort"] = normalized["sort"]
    if normalized["limit"] != DEFAULT_LIMIT:
        share["limit"] = str(normalized["limit"])
    return share


def build_share_query(params: Mapping[str, Any] | str | None = None) -> str:
    return urlencode(list(build_share_params(params).items()))


def build_share_href(path: Any, params: Mapping[str, Any] | str | None = None) -> str:
    base = safe_same_origin_href(path, fallback="/search")
    query = build_share_query(params)
    return f"{base}?{query}" if query else base


def _facet_groups(records: list[dict[str, Any]], params: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    labels = {
        "kind": ("类型", KIND_LABELS),
        "status": ("状态", STATUS_LABELS),
        "source": ("来源", SOURCE_LABELS),
    }
    for key, (label, option_labels) in labels.items():
        counts = Counter(str(record.get(key) or "unknown") for record in records)
        options = []
        for value, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
            next_params = dict(params)
            next_params[key] = value
            options.append({
                "value": value,
                "label": option_labels.get(value, value),
                "count": count,
                "selected": params.get(key) == value,
                "query": build_share_query(next_params),
            })
        groups.append({"key": key, "label": label, "options": options})
    return groups


def search_research(
    params: Mapping[str, Any] | str | None = None,
    *,
    corpus: Iterable[Mapping[str, Any]] | None = None,
    release: str | None = None,
) -> dict[str, Any]:
    """Search a Site32-compatible public research corpus.

    A non-empty query must match at least one indexed field.  Materials are not
    assigned a fallback relevance score merely because they exist in the corpus.
    """

    normalized = normalize_search_params(params)
    records = [_normalize_record(record) for record in (corpus or DEFAULT_CORPUS)]

    scored: list[tuple[dict[str, Any], int, list[str]]] = []
    if normalized["q"]:
        for record in records:
            score, matched = _score_record(normalized["q"], record)
            if score > 0:
                scored.append((record, score, matched))

    facet_base = [row[0] for row in scored] if normalized["q"] else records
    filtered = list(_apply_filters(scored, normalized))
    _sort_rows(filtered, normalized["sort"])
    total = len(filtered)
    items = [_public_item(record, score, matched) for record, score, matched in filtered[: normalized["limit"]]]
    filters = {key: normalized[key] for key in ("kind", "status", "source") if normalized[key]}
    return {
        "schema_version": SCHEMA_VERSION,
        "release": release or "",
        "default_language": DEFAULT_LANGUAGE,
        "schema": {
            "default_language": DEFAULT_LANGUAGE,
            "item_fields": [
                "kind",
                "kind_label",
                "id",
                "title",
                "subtitle",
                "href",
                "status",
                "source",
                "matched_fields",
                "preview",
                "score",
            ],
            "score_type": "relevance",
            "filters": ["kind", "status", "source"],
            "facet_group_fields": ["kind", "status", "source"],
            "safe_href": "root-relative same-origin only",
        },
        "query": {
            "raw": normalized["q"],
            "normalized": normalize_text(normalized["q"]),
            "filters": filters,
            "limit": normalized["limit"],
            "sort": normalized["sort"],
            "share_query": build_share_query(normalized),
            "share_href": build_share_href("/", normalized),
        },
        "total": total,
        "limit": normalized["limit"],
        "facet_groups": _facet_groups(facet_base, normalized),
        "default_suggestions": list(DEFAULT_SUGGESTIONS),
        "suggestions": list(DEFAULT_SUGGESTIONS),
        "items": items,
    }


federated_research_search = search_research
site32_research_search = search_research
is_safe_same_origin_href = is_same_origin_href
sanitize_href = safe_same_origin_href


__all__ = [
    "DEFAULT_CORPUS",
    "DEFAULT_LANGUAGE",
    "DEFAULT_SUGGESTIONS",
    "SCHEMA_VERSION",
    "build_share_href",
    "build_share_params",
    "build_share_query",
    "federated_research_search",
    "is_same_origin_href",
    "is_safe_same_origin_href",
    "normalize_search_params",
    "normalize_text",
    "sanitize_href",
    "safe_same_origin_href",
    "search_research",
    "site32_research_search",
]
