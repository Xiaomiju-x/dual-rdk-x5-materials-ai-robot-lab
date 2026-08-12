#!/usr/bin/env python3
"""Generate repeatable Site31/Site32 security and accessibility evidence.

The report is intentionally an internal release gate, not a penetration-test,
WCAG certification, or Cloudflare configuration assertion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


REQUIRED_BROWSER_CHECKS = {
    "desktop_viewport_matrix", "route_and_aria_current", "theater_01_04",
    "keyboard_and_skip_link", "localization", "zoom_equivalent", "console",
    "same_origin_offline_probe", "research_search_v2", "search_url_roundtrip",
    "evidence_object_detail",
}
REQUIRED_ORIGIN_CHECKS = {
    "ufw_default_deny", "loopback_origins", "caddy_forward_auth", "spoofed_identity_rejected",
}
BASE_QUERY_ASSETS = ("style.css", "i18n.js", "app.js", "twin.js")
R4_QUERY_ASSETS = ("r4.css", "r4.js", "r4-performance.js", "r4-accessibility.js")
SITE32_QUERY_ASSETS = ("site32.css", "site32.js")
SITE32_R0_CONTRACT_MIN_VERSION = (1, 5)
SITE32_R0_EVIDENCE_PATHS = {
    "browser": "static/quality/site31_browser_evidence.json",
    "origin": "static/quality/site31_origin_evidence.json",
    "style": "static/quality/site32_style_audit.json",
    "baseline": "static/quality/site32_r0_baseline.json",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
RELEASE_TOKEN_RE = re.compile(
    r"(?:site31-global-commercial-r(?:\d+(?:\.\d+)?)|"
    r"site32-global-commercial-v(?:\d+(?:\.\d+)?))-\d{8}"
)


class _DocumentAudit(HTMLParser):
    INTERACTIVE_TAGS = {"button", "input", "select", "textarea"}
    INTERACTIVE_ROLES = {"button", "link", "checkbox", "radio", "switch", "tab", "menuitem"}
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.html_lang = ""
        self.ids: list[str] = []
        self.labels_for: set[str] = set()
        self.landmarks: list[tuple[str, str]] = []
        self.skip_targets: list[str] = []
        self.live_regions: list[str] = []
        self.controls: list[dict] = []
        self._active_controls: list[int] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attr = dict(attrs)
        if tag == "html":
            self.html_lang = attr.get("lang", "")
        if attr.get("id"):
            self.ids.append(attr["id"])
        if tag == "label" and attr.get("for"):
            self.labels_for.add(attr["for"])
        if tag in {"main", "nav", "header", "footer", "aside"} or attr.get("role") in {
            "main", "navigation", "banner", "contentinfo", "complementary"
        }:
            self.landmarks.append((tag, attr.get("role", "")))
        if tag == "a" and (attr.get("class") or "").find("skip-link") >= 0:
            self.skip_targets.append((attr.get("href") or "").lstrip("#"))
        if attr.get("aria-live") or attr.get("role") in {"status", "alert"}:
            self.live_regions.append(attr.get("id", "anonymous"))

        role = attr.get("role", "")
        interactive = (tag in self.INTERACTIVE_TAGS or (tag == "a" and "href" in attr)
                       or role in self.INTERACTIVE_ROLES or bool(attr.get("onclick")))
        if interactive:
            record = {
                "tag": tag,
                "id": attr.get("id", ""),
                "type": attr.get("type", ""),
                "role": role,
                "aria_label": (attr.get("aria-label") or "").strip(),
                "aria_labelledby": (attr.get("aria-labelledby") or "").strip(),
                "title": (attr.get("title") or "").strip(),
                "value": (attr.get("value") or "").strip(),
                "alt": (attr.get("alt") or "").strip(),
                "onclick": (attr.get("onclick") or "").strip(),
                "tabindex": (attr.get("tabindex") or "").strip(),
                "text": "",
            }
            self.controls.append(record)
            if tag not in self.VOID_TAGS:
                self._active_controls.append(len(self.controls) - 1)

        if tag == "img" and self._active_controls:
            alt = (attr.get("alt") or "").strip()
            if alt:
                for idx in self._active_controls:
                    self.controls[idx]["text"] += " " + alt

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.handle_starttag(tag, attrs)
        if tag in self.INTERACTIVE_TAGS:
            self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        value = " ".join(data.split())
        if value:
            for idx in self._active_controls:
                self.controls[idx]["text"] += " " + value

    def handle_endtag(self, tag: str) -> None:
        for pos in range(len(self._active_controls) - 1, -1, -1):
            idx = self._active_controls[pos]
            if self.controls[idx]["tag"] == tag:
                del self._active_controls[pos]
                break

    def unnamed_controls(self) -> list[dict]:
        unnamed = []
        for control in self.controls:
            input_type = control["type"].lower()
            if control["tag"] == "input" and input_type in {"hidden"}:
                continue
            if "event.target===this" in control.get("onclick", "") and not control.get("role"):
                continue
            text_name = control["text"] if control["tag"] in {"button", "a"} or control["role"] else ""
            has_name = any(
                value
                for value in (control["aria_label"], control["aria_labelledby"], control["title"],
                              control["alt"], text_name)
            )
            if control["tag"] == "input" and input_type in {"button", "submit", "reset"}:
                has_name = has_name or bool(control["value"])
            if control["id"] and control["id"] in self.labels_for:
                has_name = True
            if not has_name:
                unnamed.append(control)
        return unnamed


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _report_hash(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _timestamp_seconds(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc).timestamp()


def _release_major(release: str) -> int | None:
    if re.fullmatch(r"site32-global-commercial-v\d+(?:\.\d+)?-\d{8}", release):
        return 32
    match = re.fullmatch(r"site31-global-commercial-r(\d+)(?:\.\d+)?-(\d{8})", release)
    return int(match.group(1)) if match else None


def _is_site32_release(release: str) -> bool:
    return bool(re.fullmatch(r"site32-global-commercial-v\d+(?:\.\d+)?-\d{8}", release))


def _site32_release_version(release: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"site32-global-commercial-v(\d+)(?:\.(\d+))?-\d{8}", release)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def _site32_r0_contract_required(release: str) -> bool:
    version = _site32_release_version(release)
    return version is not None and version >= SITE32_R0_CONTRACT_MIN_VERSION


def _release_not_before(release: str, released_at=None) -> float | None:
    explicit = _timestamp_seconds(released_at)
    if explicit is not None:
        return explicit
    match = re.fullmatch(
        r"(?:site31-global-commercial-r|site32-global-commercial-v)"
        r"(?:\d+(?:\.\d+)?)-(\d{8})",
        release,
    )
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=timezone.utc).timestamp()


def _evidence_time_detail(
    evidence: dict,
    *,
    now: float,
    max_age_s: int,
    release_not_before_s: float | None,
) -> tuple[bool, dict]:
    completed_at_s = _timestamp_seconds(evidence.get("completed_at"))
    age_s = None if completed_at_s is None else max(0.0, now - completed_at_s)
    future_skew_s = None if completed_at_s is None else completed_at_s - now
    newer_than_release = (
        completed_at_s is not None
        and (release_not_before_s is None or completed_at_s >= release_not_before_s)
    )
    valid = (
        completed_at_s is not None
        and future_skew_s is not None
        and future_skew_s <= 300
        and age_s is not None
        and age_s <= max_age_s
        and newer_than_release
    )
    return valid, {
        "completed_at": evidence.get("completed_at"),
        "age_s": None if age_s is None else round(age_s, 1),
        "max_age_s": max_age_s,
        "future_skew_s": None if future_skew_s is None else round(future_skew_s, 1),
        "release_not_before": (
            None if release_not_before_s is None
            else datetime.fromtimestamp(release_not_before_s, timezone.utc).isoformat()
        ),
        "newer_than_release": newer_than_release,
    }


def _validate_browser_evidence(
    evidence: dict,
    *,
    release: str,
    manifest_digest: str,
    now: float,
    max_age_s: int,
    release_not_before_s: float | None = None,
) -> tuple[bool, dict]:
    time_valid, time_detail = _evidence_time_detail(
        evidence, now=now, max_age_s=max_age_s, release_not_before_s=release_not_before_s,
    )
    checks = evidence.get("checks", [])
    observed = {
        item.get("key") for item in checks
        if isinstance(item, dict) and item.get("state") == "pass"
    } if isinstance(checks, list) else set()
    base_url = evidence.get("base_url")
    valid = (
        evidence.get("release") == release
        and bool(manifest_digest)
        and evidence.get("manifest_digest") == manifest_digest
        and isinstance(base_url, str)
        and base_url.startswith(("https://", "http://"))
        and time_valid
        and REQUIRED_BROWSER_CHECKS.issubset(observed)
    )
    return valid, {
        **time_detail,
        "base_url": base_url,
        "observed_checks": sorted(observed),
        "missing_checks": sorted(REQUIRED_BROWSER_CHECKS - observed),
        "release_matches": evidence.get("release") == release,
        "manifest_matches": bool(manifest_digest) and evidence.get("manifest_digest") == manifest_digest,
    }


def _validate_origin_evidence(
    evidence: dict,
    *,
    release: str,
    manifest_digest: str,
    now: float,
    max_age_s: int,
    release_not_before_s: float | None = None,
    require_manifest: bool = True,
) -> tuple[bool, dict]:
    time_valid, time_detail = _evidence_time_detail(
        evidence, now=now, max_age_s=max_age_s, release_not_before_s=release_not_before_s,
    )
    checks = evidence.get("checks", [])
    observed = {
        item.get("key") for item in checks
        if isinstance(item, dict) and item.get("state") == "pass"
    } if isinstance(checks, list) else set()
    manifest_matches = bool(manifest_digest) and evidence.get("manifest_digest") == manifest_digest
    valid = (
        evidence.get("release") == release
        and time_valid
        and REQUIRED_ORIGIN_CHECKS.issubset(observed)
        and (manifest_matches or not require_manifest)
    )
    return valid, {
        **time_detail,
        "observed_checks": sorted(observed),
        "missing_checks": sorted(REQUIRED_ORIGIN_CHECKS - observed),
        "release_matches": evidence.get("release") == release,
        "manifest_required": require_manifest,
        "manifest_matches": manifest_matches,
    }


def _read_evidence_object(path: Path) -> tuple[dict | None, dict]:
    detail = {
        "path": str(path),
        "exists": False,
        "readable": False,
        "sha256": None,
    }
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        detail["error"] = "file not found"
        return None, detail
    except OSError as exc:
        detail["error"] = f"{type(exc).__name__}: {exc}"
        return None, detail

    detail["exists"] = True
    detail["sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        detail["error"] = f"{type(exc).__name__}: {exc}"
        return None, detail
    if not isinstance(payload, dict):
        detail["error"] = "JSON root must be an object"
        return None, detail
    detail["readable"] = True
    return payload, detail


def _evidence_identity_detail(
    evidence: dict,
    *,
    release: str,
    manifest_digest: str,
) -> tuple[bool, dict]:
    candidate_digest_valid = bool(SHA256_RE.fullmatch(manifest_digest))
    release_matches = evidence.get("release") == release
    manifest_matches = (
        candidate_digest_valid and evidence.get("manifest_digest") == manifest_digest
    )
    return release_matches and manifest_matches, {
        "release": evidence.get("release"),
        "manifest_digest": evidence.get("manifest_digest"),
        "release_matches": release_matches,
        "manifest_matches": manifest_matches,
    }


def _validate_site32_gate_evidence_contract(
    root: Path,
    *,
    release: str,
    manifest_digest: str,
    phase: str,
    now: float,
    browser_max_age_s: int,
    origin_max_age_s: int,
    release_not_before_s: float | None = None,
) -> tuple[bool, dict]:
    """Validate candidate-bound R0 artifacts without reading downstream matrices."""
    required = _site32_r0_contract_required(release)
    result = {
        "schema_version": "site32.gate_evidence_contract.v1",
        "required": required,
        "phase": phase,
        "release": release,
        "manifest_digest": manifest_digest or None,
        "artifacts": {},
        "failures": [],
    }
    if not required:
        result["valid"] = True
        return True, result

    failures: list[str] = result["failures"]
    artifacts: dict[str, dict] = result["artifacts"]
    for name, relative in SITE32_R0_EVIDENCE_PATHS.items():
        payload, file_detail = _read_evidence_object(root / relative)
        if payload is None:
            failure = f"{name}.missing" if not file_detail["exists"] else f"{name}.malformed"
            failures.append(failure)
            artifacts[name] = {**file_detail, "valid": False}
            continue

        identity_valid, identity_detail = _evidence_identity_detail(
            payload,
            release=release,
            manifest_digest=manifest_digest,
        )
        if not identity_detail["release_matches"]:
            failures.append(f"{name}.release_mismatch")
        if not identity_detail["manifest_matches"]:
            failures.append(f"{name}.manifest_mismatch")

        if name == "browser":
            policy_valid, policy_detail = _validate_browser_evidence(
                payload,
                release=release,
                manifest_digest=manifest_digest,
                now=now,
                max_age_s=browser_max_age_s,
                release_not_before_s=release_not_before_s,
            )
            if identity_valid and not policy_valid:
                failures.append("browser.validation_failed")
        elif name == "origin":
            policy_valid, policy_detail = _validate_origin_evidence(
                payload,
                release=release,
                manifest_digest=manifest_digest,
                now=now,
                max_age_s=origin_max_age_s,
                release_not_before_s=release_not_before_s,
                require_manifest=True,
            )
            if identity_valid and not policy_valid:
                failures.append("origin.validation_failed")
        elif name == "style":
            style_ok = payload.get("ok") is True
            policy_valid = style_ok
            policy_detail = {
                "ok": payload.get("ok"),
                "exit_code": payload.get("exit_code"),
                "style_ok": style_ok,
            }
            if not style_ok:
                failures.append("style.not_ok")
        else:
            policy_valid = True
            policy_detail = {"identity_only": True}

        artifact_valid = identity_valid and policy_valid
        artifacts[name] = {
            **file_detail,
            **identity_detail,
            "valid": artifact_valid,
            "validation": policy_detail,
        }

    valid = not failures and all(
        artifacts.get(name, {}).get("valid") is True
        for name in SITE32_R0_EVIDENCE_PATHS
    )
    result["valid"] = valid
    return valid, result


def _validate_release_bindings(
    *,
    release: str,
    index_text: str,
    app_text: str,
    i18n_text: str,
    sw_text: str,
    manifest_payload: dict,
) -> tuple[bool, dict]:
    major = _release_major(release)
    query_assets = BASE_QUERY_ASSETS + (R4_QUERY_ASSETS if major is not None and major >= 4 else ())
    if _is_site32_release(release):
        query_assets += SITE32_QUERY_ASSETS
    references = re.findall(r"(?:src|href)\s*=\s*['\"]([^'\"]+)['\"]", index_text, flags=re.I)
    versioned_references: dict[str, list[str]] = {}
    for reference in references:
        parsed = urlsplit(reference)
        if not parsed.path.startswith("/") or not parsed.path.endswith((".css", ".js")):
            continue
        values = parse_qs(parsed.query).get("v", [])
        versioned_references.setdefault(parsed.path, []).extend(values)

    missing_queries = [
        asset for asset in query_assets
        if versioned_references.get(f"/{asset}") != [release]
    ]
    split_query_releases = sorted({
        value
        for values in versioned_references.values()
        for value in values
        if value != release
    })

    sw_release_match = re.search(r"\bconst\s+RELEASE\s*=\s*['\"]([^'\"]+)['\"]", sw_text)
    sw_release = sw_release_match.group(1) if sw_release_match else ""
    sw_core_missing = [asset for asset in query_assets if f"'/{asset}'" not in sw_text and f"\"/{asset}\"" not in sw_text]
    if major is not None and major >= 4:
        sw_valid = (
            sw_release == release
            and "?v=${RELEASE}" in sw_text
            and not sw_core_missing
        )
    else:
        release_family = release.rsplit("-", 1)[0]
        sw_valid = release in sw_text or release_family in sw_text

    i18n_match = re.search(r"\bI18N_DEFAULT_VER\s*=\s*['\"]([^'\"]+)['\"]", app_text)
    i18n_release = i18n_match.group(1) if i18n_match else ""
    i18n_asset_match = re.search(r"\bI18N_VERSION\s*=\s*['\"]([^'\"]+)['\"]", i18n_text)
    i18n_asset_release = i18n_asset_match.group(1) if i18n_asset_match else ""
    release_tokens = sorted(set(RELEASE_TOKEN_RE.findall("\n".join((index_text, app_text, i18n_text, sw_text)))))
    token_valid = release_tokens == [release]

    critical_paths = set(manifest_payload.get("required_critical_assets") or [])
    critical_records = manifest_payload.get("critical_assets") or []
    critical_by_path = {
        item.get("path"): item for item in critical_records
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    expected_critical = {f"static/{asset}" for asset in query_assets} | {
        "app.py", "assets.json", "static/index.html", "static/sw.js",
    }
    critical_missing = sorted(expected_critical - critical_paths)
    critical_hash_invalid = sorted(
        path for path in expected_critical
        if not re.fullmatch(r"[0-9a-f]{64}", str(critical_by_path.get(path, {}).get("sha256", "")))
    )
    critical_valid = not critical_missing and not critical_hash_invalid

    valid = (
        major is not None
        and not missing_queries
        and not split_query_releases
        and sw_valid
        and i18n_release == release
        and i18n_asset_release == release
        and token_valid
        and critical_valid
    )
    return valid, {
        "release": release,
        "major": major,
        "query_assets": list(query_assets),
        "missing_or_mismatched_queries": missing_queries,
        "split_query_releases": split_query_releases,
        "sw_release": sw_release or None,
        "sw_core_missing": sw_core_missing,
        "i18n_release": i18n_release or None,
        "i18n_asset_release": i18n_asset_release or None,
        "release_tokens": release_tokens,
        "critical_missing": critical_missing,
        "critical_hash_invalid": critical_hash_invalid,
        "critical_assets_sha256": manifest_payload.get("critical_assets_sha256"),
    }


def _validate_sw_cache_boundary(sw_text: str, release: str) -> tuple[bool, dict]:
    major = _release_major(release)
    forbidden_paths = {
        "/api/security", "/api/hardening", "/api/releases", "/api/config", "/api/logs", "/api/fleet",
    }
    if major is not None and major >= 4:
        core_match = re.search(r"const\s+CORE_PATHS\s*=\s*new\s+Set\s*\(\s*\[(.*?)\]\s*\)", sw_text, re.S)
        core_block = core_match.group(1) if core_match else ""
        required = {
            "same_origin_get_only": "request.method !== 'GET'" in sw_text and "url.origin !== self.location.origin" in sw_text,
            "private_request_bypass": "isPrivateRequest(request, url)" in sw_text and "url.pathname.startsWith('/api/')" in sw_text,
            "private_response_rejection": "responseIsPrivate(response)" in sw_text,
            "bad_response_rejection": "response.status !== 200" in sw_text and "response.redirected" in sw_text,
            "explicit_core_inventory": bool(core_block),
            "no_api_in_core": "/api" not in core_block.lower(),
        }
        valid = all(required.values()) and not any(path in core_block for path in forbidden_paths)
        return valid, {"mode": "r4-static-only", "checks": required}

    cacheable_block = ""
    if "const API_CACHEABLE" in sw_text:
        cacheable_block = sw_text.split("const API_CACHEABLE", 1)[1].split("]);", 1)[0]
    required = {
        "explicit_api_allowlist": bool(cacheable_block),
        "get_only": "e.request.method !== 'GET'" in sw_text,
        "forbidden_absent": all(path not in cacheable_block for path in forbidden_paths),
    }
    return all(required.values()), {"mode": "legacy-api-allowlist", "checks": required}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--phase", choices=("preflight", "deployed"), default="preflight")
    parser.add_argument("--output")
    parser.add_argument("--base-url", default="")
    parser.add_argument(
        "--browser-evidence-max-age",
        type=int,
        default=int(os.environ.get("SITE31_BROWSER_EVIDENCE_MAX_AGE_S", "86400")),
    )
    parser.add_argument(
        "--origin-evidence-max-age",
        type=int,
        default=int(os.environ.get("SITE31_ORIGIN_EVIDENCE_MAX_AGE_S", "86400")),
    )
    args = parser.parse_args()

    if args.browser_evidence_max_age <= 0:
        parser.error("--browser-evidence-max-age must be positive")
    if args.origin_evidence_max_age <= 0:
        parser.error("--origin-evidence-max-age must be positive")

    root = Path(args.root).resolve()
    tool_root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "tools"))
    os.environ.setdefault("XRD_CMD_TEST_MODE", "1")

    manifest_valid = False
    manifest_digest = ""
    manifest_artifact_sha256 = ""
    manifest_payload = {}
    manifest_detail = "asset-manifest.json is missing or invalid"
    try:
        from site31_asset_manifest import GENERATED_GATE_PATH, verify_manifest  # noqa: E402

        manifest_payload = verify_manifest(root, ignore_content_mismatch={GENERATED_GATE_PATH})
        manifest_valid = True
        manifest_digest = manifest_payload["manifest_digest"]
        manifest_artifact_sha256 = manifest_payload["artifact_sha256"]
        manifest_detail = (
            f"release={manifest_payload['release']}; digest={manifest_digest}; "
            f"files={manifest_payload['file_count']}"
        )
    except Exception as exc:
        manifest_detail = f"{type(exc).__name__}: {exc}"

    from app import ASSET_VER, app  # noqa: E402

    release_module = sys.modules.get("app")
    gate_db_tmp = tempfile.TemporaryDirectory(prefix="site31-gate-db-")
    release_module.DB_PATH = str(Path(gate_db_tmp.name) / "data.db")
    release_module._init_db()
    release_module._seed_defaults()
    release_not_before_s = _release_not_before(
        ASSET_VER, getattr(release_module, "RELEASED_AT", None),
    )
    audit_now = time.time()

    if manifest_valid and manifest_payload.get("release") != ASSET_VER:
        manifest_valid = False
        manifest_detail = (
            f"manifest release={manifest_payload.get('release')!r}; app release={ASSET_VER!r}"
        )

    static = root / "static"
    index_text = _read(static / "index.html")
    style_text = _read(static / "style.css")
    app_text = _read(static / "app.js")
    i18n_text = _read(static / "i18n.js")
    sw_text = _read(static / "sw.js")
    deploy_path = root / "tools" / "deploy_staged.sh"
    deploy_text = _read(deploy_path)
    unit_path = root / "systemd" / "xrd-cmdcenter.service"
    if not unit_path.exists():
        unit_path = tool_root / "systemd" / "xrd-cmdcenter.service"
    unit_text = _read(unit_path) if unit_path.exists() else ""

    doc = _DocumentAudit()
    doc.feed(index_text)
    duplicate_ids = sorted({item for item in doc.ids if doc.ids.count(item) > 1})
    unnamed_controls = doc.unnamed_controls()
    natural_controls = {"button", "a", "input", "select", "textarea"}
    nonsemantic_clicks = []
    for control in doc.controls:
        handler = control.get("onclick", "")
        if not handler or "event.target===this" in handler:
            continue
        semantic = control["tag"] in natural_controls
        semantic = semantic or (control.get("role") in _DocumentAudit.INTERACTIVE_ROLES
                                and control.get("tabindex") == "0")
        if not semantic:
            nonsemantic_clicks.append(control)
    browser_evidence_path = static / "quality" / "site31_browser_evidence.json"
    browser_evidence = {}
    browser_evidence_valid = False
    browser_evidence_sha256 = ""
    browser_evidence_detail = {
        "age_s": None,
        "max_age_s": args.browser_evidence_max_age,
        "missing_checks": sorted(REQUIRED_BROWSER_CHECKS),
        "release_matches": False,
        "manifest_matches": False,
    }
    try:
        browser_raw = browser_evidence_path.read_bytes()
        browser_evidence_sha256 = hashlib.sha256(browser_raw).hexdigest()
        browser_evidence = json.loads(browser_raw.decode("utf-8"))
        browser_evidence_valid, browser_evidence_detail = _validate_browser_evidence(
            browser_evidence,
            release=ASSET_VER,
            manifest_digest=manifest_digest,
            now=audit_now,
            max_age_s=args.browser_evidence_max_age,
            release_not_before_s=release_not_before_s,
        )
    except Exception as exc:
        browser_evidence = {}
        browser_evidence_detail["error"] = f"{type(exc).__name__}: {exc}"
    origin_evidence_path = static / "quality" / "site31_origin_evidence.json"
    origin_evidence = {}
    origin_evidence_valid = False
    origin_evidence_sha256 = ""
    origin_evidence_detail = {
        "age_s": None,
        "max_age_s": args.origin_evidence_max_age,
        "missing_checks": sorted(REQUIRED_ORIGIN_CHECKS),
        "release_matches": False,
        "manifest_required": (_release_major(ASSET_VER) or 0) >= 4,
        "manifest_matches": False,
        "newer_than_release": False,
    }
    try:
        origin_raw = origin_evidence_path.read_bytes()
        origin_evidence_sha256 = hashlib.sha256(origin_raw).hexdigest()
        origin_evidence = json.loads(origin_raw.decode("utf-8"))
        origin_evidence_valid, origin_evidence_detail = _validate_origin_evidence(
            origin_evidence,
            release=ASSET_VER,
            manifest_digest=manifest_digest,
            now=audit_now,
            max_age_s=args.origin_evidence_max_age,
            release_not_before_s=release_not_before_s,
            require_manifest=(_release_major(ASSET_VER) or 0) >= 4,
        )
    except Exception as exc:
        origin_evidence = {}
        origin_evidence_detail["error"] = f"{type(exc).__name__}: {exc}"

    site32_evidence_contract_valid, site32_evidence_contract = (
        _validate_site32_gate_evidence_contract(
            root,
            release=ASSET_VER,
            manifest_digest=manifest_digest,
            phase=args.phase,
            now=audit_now,
            browser_max_age_s=args.browser_evidence_max_age,
            origin_max_age_s=args.origin_evidence_max_age,
            release_not_before_s=release_not_before_s,
        )
    )

    checks: list[dict] = []

    def add(domain: str, key: str, label: str, max_points: float, passed: bool | None,
            evidence: str, *, critical: bool = False, residual_risk: str = "") -> None:
        state = "verified" if passed is True else ("failed" if passed is False else "manual-check")
        checks.append({
            "domain": domain,
            "key": key,
            "label": label,
            "state": state,
            "max_points": max_points,
            "earned_points": max_points if passed is True else 0.0,
            "critical": critical,
            "evidence": evidence,
            "residual_risk": residual_risk,
        })

    add(
        "security", "release.asset_manifest", "Release asset manifest integrity", 0.0,
        manifest_valid, manifest_detail, critical=True,
    )
    if site32_evidence_contract["required"]:
        add(
            "security", "release.site32_r0_evidence_contract",
            "Site32 R0 candidate evidence contract", 0.0,
            site32_evidence_contract_valid,
            (
                "browser/origin/style/site32_r0_baseline candidate binding; "
                f"failures={site32_evidence_contract['failures'] or 'none'}"
            ),
            critical=True,
        )

    with app.test_client() as client:
        host = {"Host": "localhost"}
        index = client.get("/", headers=host, base_url="https://localhost")
        method_results = []
        for method, path in (
            ("POST", "/api/config"), ("PUT", "/api/site31_portal"),
            ("PATCH", "/api/site31_portal"), ("DELETE", "/api/site31_portal"),
        ):
            response = client.open(path, method=method, headers=host, json={})
            method_results.append((method, path, response.status_code))
        judge_write = client.post(
            "/api/config", headers={**host, "X-User": "judge", "X-Role": "judge"}, json={}
        )
        cross_origin = client.post(
            "/api/config",
            headers={**host, "X-User": "member", "X-Role": "member", "Origin": "https://attacker.invalid"},
            json={},
        )
        bad_host = client.get("/api/site31_portal", headers={"Host": "attacker.invalid"})

        headers = {key.lower(): value for key, value in index.headers.items()}
        sensitive_headers = {key.lower(): value for key, value in client.get(
            "/api/site31_gate_evidence", headers=host, base_url="https://localhost"
        ).headers.items()}
        csp = headers.get("content-security-policy", "")
        required_headers = {
            "content-security-policy", "strict-transport-security", "x-content-type-options",
            "x-frame-options", "referrer-policy", "permissions-policy", "cross-origin-opener-policy",
            "cross-origin-resource-policy", "origin-agent-cluster", "x-permitted-cross-domain-policies",
        }
        required_csp = {
            "frame-ancestors", "base-uri", "form-action", "object-src", "worker-src",
            "manifest-src", "upgrade-insecure-requests", "block-all-mixed-content",
        }

        public_payloads = []
        public_ok = True
        public_statuses = {}
        for path in ("/api/public_manifest", "/api/assets", "/api/trust_center", "/api/hardening"):
            response = client.get(path, headers=host)
            public_statuses[path] = response.status_code
            public_ok = public_ok and response.status_code == 200
            public_payloads.append(response.get_data(as_text=True))

    method_ok = all(status == 405 for _, _, status in method_results)
    method_ok = method_ok and judge_write.status_code == 403 and cross_origin.status_code == 403
    add(
        "security", "app.method_boundary", "匿名与 judge 写方法 fail closed", 1.4, method_ok,
        f"method matrix={method_results}; judge={judge_write.status_code}; cross-origin={cross_origin.status_code}",
        critical=True,
    )
    add(
        "security", "app.host_body_boundary", "Host 白名单与请求体上限", 0.8,
        bad_host.status_code == 421 and app.config.get("MAX_CONTENT_LENGTH") == 2 * 1024 * 1024,
        f"bad_host={bad_host.status_code}; max_body={app.config.get('MAX_CONTENT_LENGTH')}", critical=True,
    )
    header_ok = index.status_code == 200 and required_headers.issubset(headers) and all(x in csp for x in required_csp)
    add(
        "security", "browser.security_headers", "浏览器安全响应头矩阵", 1.2, header_ok,
        f"headers={sorted(required_headers.intersection(headers))}; csp={sorted(x for x in required_csp if x in csp)}",
        critical=True,
    )

    public_text = "\n".join([index_text, app_text, sw_text, *public_payloads])
    private_patterns = {
        "private_ipv4": r"(?<!\d)(?:10|127)\.\d{1,3}\.\d{1,3}\.\d{1,3}(?!\d)|192\.168\.\d{1,3}\.\d{1,3}",
        "device_path": r"/dev/(?:tty|video|gpio|LD14|F407)",
        "ssh_command": r"\bssh\s+[A-Za-z0-9_.-]+@",
        "direct_actuation": r"(?:raw chassis velocity command|SERVO_WRITE|LIFT_SPIN|GPIO bitpulse|electromagnet\s+on)",
    }
    public_hits = {name: sorted(set(re.findall(pattern, public_text, flags=re.I)))[:5]
                   for name, pattern in private_patterns.items()}
    public_hits = {name: values for name, values in public_hits.items() if values}
    add(
        "security", "public.redaction_no_actuation", "公开字段脱敏与无物理执行入口", 1.2,
        public_ok and not public_hits,
        f"public API statuses={public_statuses}; hits={public_hits or 'none'}", critical=True,
    )

    auth_candidates = (root / "auth", root.parent / "auth", tool_root.parent / "auth")
    auth_dir = next((candidate for candidate in auth_candidates if candidate.exists()), auth_candidates[-1])
    auth_smoke = auth_dir / "security_smoke.py"
    auth_result = None
    auth_detail = "auth smoke unavailable"
    if auth_smoke.exists():
        proc = subprocess.run(
            [sys.executable, str(auth_smoke), str(auth_dir)], cwd=auth_dir,
            capture_output=True, text=True, timeout=30,
        )
        auth_result = proc.returncode == 0
        auth_detail = (proc.stdout or proc.stderr).strip()[-500:]
    add(
        "security", "gateway.sso_auth", "SSO 登录、角色与安全跳转回归", 1.1,
        auth_result, auth_detail,
        residual_risk="Caddy identity-header copy/strip configuration must be rechecked after gateway changes.",
    )
    auth_source = _read(auth_dir / "app.py") if (auth_dir / "app.py").exists() else ""
    auth_source_ok = all(token in auth_source for token in (
        "secure=True", "httponly=True", 'samesite="Lax"', "html.escape", "_safe_next", "LOGIN_LOCK_S"
    ))
    add(
        "security", "auth.session_redirect", "Cookie、爆破锁定与重定向防护", 0.8,
        auth_source_ok and auth_result is True, "auth source controls + isolated Flask regression",
    )

    sw_ok, sw_boundary_detail = _validate_sw_cache_boundary(sw_text, ASSET_VER)
    add(
        "security", "offline.cache_integrity", "Service Worker 只读白名单", 0.6, sw_ok,
        f"GET-only, same-origin static cache; private/API traffic excluded; {sw_boundary_detail}", critical=True,
    )

    offline_probe_ok = all(token in app_text for token in (
        "online_probe=", "method:'HEAD'", "cache:'no-store'", "offSetState(off)"
    ))
    add(
        "security", "offline.same_origin_probe", "离线提示以非缓存同源探测为准", 0.0,
        offline_probe_ok,
        "uncached HEAD probe bypasses Service Worker snapshots and avoids navigator.onLine false positives",
        critical=True,
    )

    release_bindings_valid, release_binding_detail = _validate_release_bindings(
        release=ASSET_VER,
        index_text=index_text,
        app_text=app_text,
        i18n_text=i18n_text,
        sw_text=sw_text,
        manifest_payload=manifest_payload,
    )
    release_ok = release_bindings_valid and all(token in deploy_text for token in (
        "py_compile", "site31_gate_audit.py", "site31_smoke.py", "restore_previous",
        "rollback snapshot", "rsync", "--delete",
    ))
    add(
        "security", "release.audit_rollback", "版本一致性、审计与回滚", 0.7, release_ok,
        f"bindings={release_binding_detail}; exact-tree promote and rollback hooks present",
        critical=True,
    )

    unit_tokens = [
        "--bind 127.0.0.1:29100", "NoNewPrivileges=true", "PrivateTmp=true",
        "PrivateDevices=true", "RestrictSUIDSGID=true",
    ]
    if ASSET_VER.startswith("site32-"):
        unit_tokens.extend((
            "User=xrd-cmdcenter", "Group=xrd-cmdcenter", "ProtectSystem=strict",
            "ProtectHome=tmpfs", "CapabilityBoundingSet=", "AmbientCapabilities=",
            "XRD_CMD_DB_PATH=/var/lib/xrd-cmdcenter/data.db",
        ))
        protect_system_ok = True
    else:
        protect_system_ok = any(
            token in unit_text for token in ("ProtectSystem=full", "ProtectSystem=strict")
        )
    unit_ok = protect_system_ok and all(token in unit_text for token in unit_tokens)
    attached_origin_evidence_valid = origin_evidence_valid
    deployed_origin_runtime_valid = None
    if args.phase == "deployed" and args.base_url:
        try:
            from urllib.request import urlopen
            with urlopen(args.base_url.rstrip("/") + "/api/public_status", timeout=8) as response:
                unit_ok = unit_ok and response.status == 200
        except Exception:
            unit_ok = False
        try:
            ufw = subprocess.run(["sudo", "-n", "ufw", "status", "verbose"], capture_output=True, text=True, timeout=10)
            sockets = subprocess.run(["ss", "-lnt"], capture_output=True, text=True, timeout=10)
            caddy = subprocess.run(
                ["sudo", "-n", "grep", "-E", "forward_auth|copy_headers|127.0.0.1:29100|127.0.0.1:29000", "/etc/caddy/Caddyfile"],
                capture_output=True, text=True, timeout=10,
            )
            spoof = subprocess.run(
                ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}",
                 "-H", "X-User: attacker", "-H", "X-Role: admin",
                 "https://xiaomiju.xyz/api/site31_scorecard"],
                capture_output=True, text=True, timeout=15,
            )
            deployed_origin_runtime_valid = (
                ufw.returncode == 0 and "Status: active" in ufw.stdout
                and "Default: deny (incoming)" in ufw.stdout
                and sockets.returncode == 0 and "127.0.0.1:29000" in sockets.stdout
                and "127.0.0.1:29100" in sockets.stdout
                and caddy.returncode == 0 and "forward_auth" in caddy.stdout and "copy_headers" in caddy.stdout
                and spoof.returncode == 0 and spoof.stdout.strip() == "401"
            )
            origin_evidence_detail["deployed_runtime"] = {
                "valid": deployed_origin_runtime_valid,
                "source": "deployed runtime commands",
            }
            if deployed_origin_runtime_valid:
                origin_evidence_detail["deployed_runtime"].update({
                    "completed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    "age_s": 0.0,
                    "max_age_s": args.origin_evidence_max_age,
                    "observed_checks": sorted(REQUIRED_ORIGIN_CHECKS),
                    "missing_checks": [],
                    "release_matches": True,
                    "manifest_required": True,
                    "manifest_matches": bool(manifest_digest),
                    "manifest_digest": manifest_digest or None,
                    "newer_than_release": True,
                })
        except Exception as exc:
            deployed_origin_runtime_valid = False
            origin_evidence_detail["deployed_runtime"] = {
                "valid": False,
                "source": "deployed runtime commands",
                "error": f"{type(exc).__name__}: {exc}",
            }
        origin_evidence_valid = (
            attached_origin_evidence_valid and deployed_origin_runtime_valid is True
        )
    unit_ok = unit_ok and origin_evidence_valid
    add(
        "security", "origin.loopback_systemd", "Loopback origin 与 systemd 沙箱", 0.8, unit_ok,
        f"unit hardening={'present' if unit_ok else 'incomplete'}; phase={args.phase}", critical=True,
    )

    transport_ok = "strict-transport-security" in headers and "no-store" in sensitive_headers.get("cache-control", "")
    add(
        "security", "transport.tls_secret_posture", "HTTPS/HSTS 与敏感响应缓存边界", 0.7,
        transport_ok, "HTTPS test-client response includes HSTS; auth and sensitive APIs use no-store",
        residual_risk="Certificate-chain and edge TLS policy require live operator/edge evidence.",
    )
    add(
        "security", "origin.vps_firewall", "VPS 防火墙、loopback origin 与身份伪造负测", 0.0,
        origin_evidence_valid,
        f"phase={args.phase}; origin evidence={'verified' if origin_evidence_valid else 'missing/failed'}",
        critical=True,
        residual_risk="Port 80 is also used by the existing FRP service; this is outside the cmdcenter origin binding.",
    )
    dependency_ok = all((static / name).exists() for name in ("three.min.js", "app.js", "twin.js"))
    dependency_ok = dependency_ok and "https://cdn" not in index_text.lower()
    add(
        "security", "dependency.self_hosted_inventory", "关键前端依赖自托管与静态清单", 0.5,
        dependency_ok, "three/app/twin are self-hosted; no CDN runtime dependency in index",
        residual_risk="Continuous CVE/SBOM scanning is not yet attached.",
    )
    add(
        "security", "edge.waf_rate_limit", "Cloudflare WAF 与边缘速率限制", 1.2, None,
        "requires Cloudflare dashboard/API rule evidence",
        residual_risk="Application controls do not replace verified edge managed rules or rate limiting.",
    )
    add(
        "security", "external.assurance", "外部渗透、持续漏洞扫描与独立复核", 1.0, None,
        "no independent report attached",
        residual_risk="Internal tests are not an independent penetration test or security certification.",
    )

    main_target_ok = "main" in doc.skip_targets and any(tag == "main" or role == "main" for tag, role in doc.landmarks)
    semantics_ok = doc.html_lang.lower().startswith("zh") and main_target_ok and bool(doc.live_regions)
    add(
        "accessibility", "document.semantics", "语言、跳转链接、地标与状态区域", 0.9,
        semantics_ok, f"lang={doc.html_lang}; landmarks={len(doc.landmarks)}; live_regions={doc.live_regions}",
        critical=True,
    )
    add(
        "accessibility", "controls.names_ids", "控件可访问名称与唯一 ID", 1.0,
        not unnamed_controls and not duplicate_ids and not nonsemantic_clicks,
        f"controls={len(doc.controls)}; unnamed={unnamed_controls[:100]}; "
        f"nonsemantic_clicks={nonsemantic_clicks[:100]}; duplicate_ids={duplicate_ids[:100]}",
        critical=True,
    )
    focus_ok = all(token in style_text for token in (":focus-visible", ".skip-link", "@media (forced-colors"))
    focus_ok = focus_ok and "aria-current" in app_text
    add(
        "accessibility", "focus.current_route", "可见焦点、skip link 与当前路由", 0.8,
        focus_ok, "focus-visible + forced-colors + aria-current route state",
    )
    motion_ok = "prefers-reduced-motion" in style_text and "prefers-reduced-transparency" in style_text
    motion_ok = motion_ok and "matchMedia('(prefers-reduced-motion: reduce)')" in app_text
    add(
        "accessibility", "motion.transparency", "减少动态与透明效果", 0.8,
        motion_ok, "CSS reduced motion/transparency and JS deterministic route path",
    )
    status_ok = "routeAnnouncer" in index_text and "announceRouteChange" in app_text
    add(
        "accessibility", "status.route_announcement", "路由状态播报且不抢焦点", 0.6,
        status_ok, "polite atomic live region + document title update",
    )
    responsive_ok = all(token in style_text for token in (
        "overflow-x:hidden", "overflow-wrap:anywhere", "@media (max-width:1366px)"
    ))
    add(
        "accessibility", "layout.zoom_readability", "宽屏笔记本、缩放与长文本可读性", 0.7,
        responsive_ok, "stable overflow rules and 1366px desktop breakpoint; browser matrix remains release evidence",
    )
    add(
        "accessibility", "browser.keyboard_matrix", "真实浏览器键盘、五档宽屏与等效 200% 缩放矩阵", 0.5,
        browser_evidence_valid,
        f"browser evidence valid={browser_evidence_valid}; sha256={browser_evidence_sha256 or 'missing'}; "
        f"binding={browser_evidence_detail}",
        critical=True,
    )
    add(
        "accessibility", "external.screen_reader", "NVDA/VoiceOver 与 WCAG-EM 人工审计", 0.7, None,
        "no independent/manual screen-reader report attached",
        residual_risk="Core automated checks do not prove complete WCAG 2.2 AA conformance.",
    )

    dimensions = {}
    for domain in ("security", "accessibility"):
        domain_checks = [item for item in checks if item["domain"] == domain]
        max_points = round(sum(item["max_points"] for item in domain_checks), 2)
        earned_points = round(sum(item["earned_points"] for item in domain_checks), 2)
        dimensions[domain] = {
            "max_points": max_points,
            "earned_points": earned_points,
            "ratio": round(earned_points / max_points, 4) if max_points else 0,
            "state": "verified-partial" if earned_points >= max_points * 0.75 else "work-in-progress",
        }

    critical_failures = [item["key"] for item in checks if item["critical"] and item["state"] != "verified"]
    failed_checks = [item["key"] for item in checks if item["state"] == "failed"]
    gate = "pass" if not critical_failures and all(item["ratio"] >= 0.75 for item in dimensions.values()) else "fail"
    payload = {
        "schema_version": "site31.gate_evidence.v1",
        "release": ASSET_VER,
        "generated_at": int(time.time()),
        "phase": args.phase,
        "gate": gate,
        "dimensions": dimensions,
        "checks": checks,
        "summary": {
            "verified": sum(item["state"] == "verified" for item in checks),
            "manual_check": sum(item["state"] == "manual-check" for item in checks),
            "failed": len(failed_checks),
            "critical_failures": critical_failures,
        },
        "browser_evidence": {
            "valid": browser_evidence_valid,
            "sha256": browser_evidence_sha256 or None,
            "browser": browser_evidence.get("browser"),
            "completed_at": browser_evidence.get("completed_at"),
            "base_url": browser_evidence.get("base_url"),
            "manifest_digest": browser_evidence.get("manifest_digest"),
            **browser_evidence_detail,
        },
        "asset_manifest": {
            "valid": manifest_valid,
            "manifest_digest": manifest_digest or None,
            "artifact_sha256": manifest_artifact_sha256 or None,
            "critical_assets_sha256": manifest_payload.get("critical_assets_sha256"),
            "critical_assets": manifest_payload.get("critical_assets") or [],
            "detail": manifest_detail,
        },
        "origin_evidence": {
            "valid": origin_evidence_valid,
            "sha256": origin_evidence_sha256 or None,
            "completed_at": origin_evidence.get("completed_at"),
            "source": (
                "attached VPS read-only audit + deployed runtime commands"
                if deployed_origin_runtime_valid is not None
                else "attached VPS read-only audit"
            ),
            "manifest_digest": origin_evidence.get("manifest_digest"),
            **origin_evidence_detail,
        },
        "site32_evidence_contract": site32_evidence_contract,
        "claim_boundary": (
            "Internal release evidence only; not a penetration test, WCAG certification, "
            "Cloudflare configuration proof, or third-party global ranking."
        ),
    }
    payload["artifact_sha256"] = _report_hash(payload)

    output = Path(args.output).resolve() if args.output else static / "quality" / "site31_gate_evidence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": gate == "pass", "release": ASSET_VER, "phase": args.phase, "gate": gate,
        "dimensions": dimensions, "critical_failures": critical_failures,
        "output": str(output), "sha256": payload["artifact_sha256"],
    }, ensure_ascii=False))
    return 0 if gate == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
