"""Machine-readable Public, Reviewer and Internal access contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fnmatch import fnmatchcase


SAFE_PUBLIC_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
ROLE_LEVEL = {"public": 0, "reviewer": 1, "internal": 2, "admin": 3}


@dataclass(frozen=True, slots=True)
class AccessRule:
    pattern: str
    scope: str
    methods: tuple[str, ...] = ("GET", "HEAD", "OPTIONS")
    source: str = "curated-public-dto"
    note: str = ""
    data_origin: str = "application"
    runtime_source: str = "release-bound"
    freshness_policy: str = "endpoint-defined"
    mutates: bool = False


ACCESS_RULES = (
    AccessRule("/", "public", source="versioned-static-shell"),
    AccessRule("/status", "public", source="public-status-dto"),
    AccessRule("/atlas", "public", source="public-material-shell"),
    AccessRule("/brain", "public", source="public-capability-shell"),
    AccessRule("/models", "public", source="public-model-card-shell"),
    AccessRule("/assets", "public", source="redacted-public-asset-shell"),
    AccessRule("/twin", "public", source="read-only-twin-shell"),
    AccessRule("/materials/*", "public", source="public-material-dto"),
    AccessRule("/predictions/*", "public", source="public-prediction-dto"),
    AccessRule("/evidence/*", "public", source="evidence-object-v3-shell"),
    AccessRule("/robots.txt", "public", source="static-policy"),
    AccessRule("/sitemap.xml", "public", source="static-policy"),
    AccessRule("/healthz", "public", source="release-health"),
    AccessRule("/api/public_status", "public", source="public-status-dto"),
    AccessRule("/api/me", "public", source="current-session-dto", freshness_policy="no-store"),
    AccessRule("/api/search", "public", source="federated-public-index"),
    AccessRule("/api/search/*", "public", source="federated-public-index"),
    AccessRule("/api/materials/*", "public", source="public-material-dto"),
    AccessRule("/api/predictions/*", "public", source="public-prediction-dto"),
    AccessRule("/api/evidence_objects", "public", source="evidence-object-v3"),
    AccessRule("/api/evidence_objects/*", "public", source="evidence-object-v3"),
    AccessRule("/api/research_portal", "public", source="public-research-contract"),
    AccessRule("/api/research_collections", "public", source="public-research-collections"),
    AccessRule("/api/research_collections/*", "public", source="public-research-collections"),
    AccessRule("/api/research_passport", "public", source="public-research-passport"),
    AccessRule("/api/evidence_bundle.json", "public", source="public-evidence-export"),
    AccessRule("/api/evidence_bundle.txt", "public", source="public-evidence-export"),
    AccessRule("/api/trust_center", "public", source="public-trust-summary"),
    AccessRule("/api/public_manifest", "public", source="safe-field-manifest"),
    AccessRule("/api/openapi.json", "public", source="public-api-catalog"),
    AccessRule("/api/models", "public", source="public-model-cards"),
    AccessRule("/api/atlas", "public", source="curated-material-atlas"),
    AccessRule("/api/ai_brain/explain", "public", source="public-capability-card"),
    AccessRule("/api/rb_voe/explain", "public", source="release-bound-rb-voe-evidence", freshness_policy="no-store"),
    AccessRule("/api/assets", "public", source="redacted-public-assets"),
    AccessRule("/api/fleet", "public", source="redacted-runtime-status", freshness_policy="ttl-30s"),
    AccessRule("/api/kpi", "public", source="redacted-research-kpi", freshness_policy="ttl-30s"),
    AccessRule("/api/ops", "public", source="redacted-runtime-status", freshness_policy="ttl-30s"),
    AccessRule("/api/systems", "public", source="redacted-system-catalog"),
    AccessRule("/api/twin", "public", source="read-only-twin-snapshot", freshness_policy="ttl-5s"),
    AccessRule("/api/uptime", "public", source="public-availability-summary"),
    AccessRule("/api/hardening", "public", source="public-trust-summary"),
    AccessRule("/api/global_benchmark", "public", source="public-benchmark-summary"),
    AccessRule("/api/site32/contract", "public", source="site32-product-contract"),
    AccessRule("/api/site32/access-matrix", "reviewer", source="site32-access-contract"),
    AccessRule("/api/admin/*", "admin", methods=("GET", "HEAD", "OPTIONS"), source="internal-db"),
    AccessRule("/api/config", "internal", methods=("GET", "HEAD", "OPTIONS", "POST"), source="internal-config"),
    AccessRule("/api/releases", "internal", source="internal-release-ledger"),
    AccessRule("/api/logs", "internal", source="internal-observability"),
    AccessRule("/api/logs/*", "internal", source="internal-observability"),
    AccessRule("/api/stream", "reviewer", source="reviewer-event-stream"),
    AccessRule("/api/site31_gate_evidence", "reviewer", source="release-gate-evidence"),
    AccessRule("/api/site31_scorecard", "reviewer", source="internal-readiness-score"),
    AccessRule("/api/site31_portal", "reviewer", source="reviewer-portal"),
    AccessRule("/api/*", "reviewer", source="protected-api", note="explicit public and internal rules take precedence"),
)


def _specific_rules(path: str) -> list[AccessRule]:
    matches = [rule for rule in ACCESS_RULES if fnmatchcase(path, rule.pattern)]
    return sorted(matches, key=lambda rule: len(rule.pattern), reverse=True)


def classify_request(path: str, method: str = "GET") -> AccessRule:
    """Return the most specific access rule for a request surface."""

    clean_path = (path or "/").split("?", 1)[0]
    verb = (method or "GET").upper()
    matched = _specific_rules(clean_path)
    if matched:
        rule = matched[0]
        if verb in rule.methods:
            return rule
        return AccessRule(
            rule.pattern, "internal", (verb,), "write-operation",
            "write methods are never public", mutates=True,
        )

    if clean_path.startswith("/api/"):
        scope = "internal" if verb not in SAFE_PUBLIC_METHODS else "reviewer"
        return AccessRule("/api/*", scope, (verb,), "protected-api", mutates=verb not in SAFE_PUBLIC_METHODS)
    if clean_path.endswith((".css", ".js", ".mjs", ".svg", ".png", ".jpg", ".jpeg", ".webp", ".avif", ".ico", ".woff", ".woff2", ".webmanifest")):
        return AccessRule("static-asset", "public", (verb,), "versioned-static-asset")
    return AccessRule("spa-reviewer-route", "reviewer", (verb,), "reviewer-shell")


def role_from_headers(user: str | None, role: str | None) -> str:
    normalized = (role or "").strip().lower()
    if not (user or "").strip():
        return "public"
    if normalized == "judge":
        return "reviewer"
    if normalized == "member":
        return "internal"
    if normalized == "admin":
        return "admin"
    return "public"


def role_allows(role: str, scope: str) -> bool:
    return ROLE_LEVEL.get(role, 0) >= ROLE_LEVEL.get(scope, 3)


def public_matrix_payload() -> dict:
    return {
        "schema_version": "site32.access_matrix.v1",
        "roles": [
            {"role": "public", "level": 0, "description": "anonymous curated read-only surface"},
            {"role": "reviewer", "level": 1, "description": "SSO protected, read-only review evidence"},
            {"role": "internal", "level": 2, "description": "authenticated operations surface"},
            {"role": "admin", "level": 3, "description": "explicit administrative authority"},
        ],
        "safe_public_methods": sorted(SAFE_PUBLIC_METHODS),
        "default_api_scope": "reviewer for reads; internal for writes",
        "rules": [asdict(rule) for rule in ACCESS_RULES],
        "physical_control_publicly_available": False,
    }
