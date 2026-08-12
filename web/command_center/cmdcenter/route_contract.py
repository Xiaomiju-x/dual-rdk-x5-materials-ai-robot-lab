"""Authoritative Flask route, access and documentation inventory."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from .access import classify_request


_ROUTE_VARIABLE = re.compile(r"<(?:(?:[^:<>]+):)?([^<>]+)>")
_METHOD_SPLIT = re.compile(r"[/|]")
_AUTOMATIC_METHODS = frozenset({"HEAD", "OPTIONS"})


def normalize_route_path(path: str) -> str:
    """Convert Flask route variables and legacy angle placeholders to OpenAPI form."""

    return _ROUTE_VARIABLE.sub(lambda match: "{" + match.group(1) + "}", path or "/")


def split_methods(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        methods = _METHOD_SPLIT.split(value)
    else:
        methods = list(value)
    return sorted({str(method).strip().upper() for method in methods if str(method).strip()})


def route_inventory(app) -> list[dict]:
    """Return every registered method surface with its enforced access decision."""

    inventory: list[dict] = []
    for rule in sorted(app.url_map.iter_rules(), key=lambda item: (item.rule, item.endpoint)):
        for method in sorted(set(rule.methods or ()) - _AUTOMATIC_METHODS):
            decision_path = "/asset.js" if rule.endpoint == "static" else rule.rule
            decision = classify_request(decision_path, method)
            inventory.append({
                "route": rule.rule,
                "documented_path": normalize_route_path(rule.rule),
                "endpoint": rule.endpoint,
                "method": method,
                "scope": decision.scope,
                "source": decision.source,
                "policy_pattern": decision.pattern,
                "data_origin": decision.data_origin,
                "runtime_source": decision.runtime_source,
                "freshness_policy": decision.freshness_policy,
                "mutates": decision.mutates,
            })
    return inventory


def route_inventory_summary(inventory: Sequence[dict]) -> dict:
    return {
        "routes": len({item["route"] for item in inventory}),
        "method_surfaces": len(inventory),
        "unclassified": sum(not item.get("scope") or not item.get("source") for item in inventory),
        "protected_default": sum(item.get("source") == "protected-api" for item in inventory),
        "scopes": {
            scope: sum(item.get("scope") == scope for item in inventory)
            for scope in ("public", "reviewer", "internal", "admin")
        },
    }


def _legacy_doc_index(docs: Sequence[tuple]) -> tuple[dict, dict]:
    exact: dict[tuple[str, str], dict] = {}
    by_path: dict[str, dict] = {}
    for group, endpoints in docs:
        for method_value, path, params, _legacy_role, description in endpoints:
            normalized_path = normalize_route_path(path)
            metadata = {
                "group": group,
                "params": params,
                "description": description,
            }
            by_path.setdefault(normalized_path, metadata)
            for method in split_methods(method_value):
                exact[(normalized_path, method)] = metadata
    return exact, by_path


def reconciled_api_docs(app, docs: Sequence[tuple]) -> list[dict]:
    """Build API docs from the real Flask map and use legacy text only as metadata.

    A stale legacy row can no longer create a fictional OpenAPI operation. Conversely,
    every registered API surface is represented even if descriptive copy has not yet
    been curated.
    """

    exact, by_path = _legacy_doc_index(docs)
    entries: list[dict] = []
    for item in route_inventory(app):
        path = item["documented_path"]
        if not (path.startswith("/api/") or path == "/metrics"):
            continue
        method = item["method"]
        metadata = exact.get((path, method)) or by_path.get(path) or {}
        endpoint_name = item["endpoint"].rsplit(".", 1)[-1].replace("_", " ")
        group = metadata.get("group") or "Registered Site32 surface"
        description = metadata.get("description") or (
            f"Registered {item['scope']} API surface: {endpoint_name}."
        )
        entries.append({
            "group": group,
            "method": method,
            "methods": [method],
            "path": path,
            "params": metadata.get("params") or "—",
            "role": item["scope"],
            "description": description,
            "endpoint": item["endpoint"],
            "source": item["source"],
            "data_origin": item["data_origin"],
            "runtime_source": item["runtime_source"],
            "freshness_policy": item["freshness_policy"],
            "mutates": item["mutates"],
        })
    return entries


def group_doc_entries(entries: Sequence[dict]) -> list[tuple[str, list[dict]]]:
    grouped: dict[str, list[dict]] = {}
    for entry in entries:
        grouped.setdefault(entry["group"], []).append(entry)
    return list(grouped.items())
