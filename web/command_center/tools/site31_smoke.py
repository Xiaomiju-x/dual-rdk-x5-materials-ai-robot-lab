#!/usr/bin/env python3
import json
import hashlib
import os
import re
import sys
from pathlib import Path


ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]).resolve()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from app import ASSET_VER, app  # noqa: E402


def expect(response, status, label):
    if response.status_code != status:
        raise AssertionError(f"{label}: expected {status}, got {response.status_code}: {response.data[:160]!r}")
    return response


def get_json(client, path, role=None):
    headers = {"Host": "localhost"}
    if role == "reviewer":
        headers.update({"X-User": "judge", "X-Role": "judge"})
    response = expect(client.get(path, headers=headers), 200, path)
    return response.get_json()


with app.test_client() as client:
    index = expect(client.get("/", headers={"Host": "localhost"}), 200, "index")
    assert ASSET_VER.encode() in index.data
    index.close()
    for path in ("/defense", "/benchmark", "/studio", "/atlas", "/command", "/models", "/assets", "/release",
                 "/evidence/ev:xrd:slam_shadow"):
        deep_link = expect(client.get(path, headers={
            "Host": "localhost", "X-User": "judge", "X-Role": "judge",
        }), 200, "SPA deep link " + path)
        assert ASSET_VER.encode() in deep_link.data
        deep_link.close()
    versioned_app = expect(
        client.get("/app.js?v=" + ASSET_VER, headers={"Host": "localhost"}),
        200,
        "versioned root static asset",
    )
    assert ASSET_VER.encode() in versioned_app.data
    versioned_app.close()
    expect(client.get("/definitely-not-a-page", headers={"Host": "localhost"}), 404, "unknown deep link")

    portal = get_json(client, "/api/site31_portal", role="reviewer")
    objects = get_json(client, "/api/evidence_objects")
    object_schema = get_json(client, "/api/evidence_objects/schema.json")
    passport = get_json(client, "/api/research_passport")
    search = get_json(client, "/api/search/federated?q=YAG%3ACr3%2B&limit=3")
    search_filtered = get_json(client, "/api/search/federated?q=YAG%3ACr3%2B&kind=material&status=replay&source=curated&limit=3")
    search_empty = get_json(client, "/api/search/federated?q=unrelated%20banana%20wafer&limit=3")
    bundle = get_json(client, "/api/evidence_bundle.json")
    bundle_limited = get_json(client, "/api/evidence_bundle.json?limit=1&q=missing")
    passport_limited = get_json(client, "/api/research_passport?limit=1&q=missing")
    materials_public = get_json(client, "/api/materials/explorer?limit=1000")
    public_status = get_json(client, "/api/public_status")
    hardening = get_json(client, "/api/hardening")
    trust = get_json(client, "/api/trust_center")
    benchmark = get_json(client, "/api/global_benchmark")
    scorecard = get_json(client, "/api/site31_scorecard", role="reviewer")
    gate_evidence = get_json(client, "/api/site31_gate_evidence", role="reviewer")

    assert portal["release"] == ASSET_VER
    assert objects["release"] == ASSET_VER
    assert objects["schema_version"] == "site31.evidence_index.v3"
    assert objects["count"] >= 7
    assert object_schema["properties"]["schema_version"]["const"] == "site31.evidence_object.v3"
    required = set(object_schema["required"])
    assert {"identifier", "version", "claims", "property_provenance", "validation",
            "uncertainty", "relations", "rights", "limitations", "citation",
            "distributions"}.issubset(required)
    assert passport["release"] == ASSET_VER
    assert passport_limited["counts"]["material_rows"] == passport["counts"]["material_rows"]
    assert bundle["release"] == ASSET_VER
    assert bundle_limited["materials_summary"] == bundle["materials_summary"]
    assert bundle_limited["materials_sample"] == bundle["materials_sample"]
    assert len(bundle["evidence_objects_summary"]) == objects["count"]
    assert all(not item.get("work_order") for item in materials_public["items"])
    assert search["release"] == ASSET_VER
    assert search["schema_version"] == "site32.research_search.v2"
    assert search["default_language"] == "zh-CN"
    assert search["total"] >= 1 and search["items"][0]["id"] == "seed-yag-cr3"
    assert search["query"]["share_href"].startswith("/?q=YAG%3ACr3%2B")
    assert [group["key"] for group in search["facet_groups"]] == ["kind", "status", "source"]
    assert search_filtered["total"] == 1 and search_filtered["items"][0]["id"] == "seed-yag-cr3"
    assert search_empty["total"] == 0 and search_empty["items"] == []
    assert public_status["summary"]["release"] == ASSET_VER
    assert hardening["release"] == ASSET_VER
    assert trust["certification_claim"].startswith("none")
    assert benchmark["score_type"].startswith("internal")
    assert scorecard["external_rank_claim"] == "none"
    assert scorecard["gate"] == "pass"
    assert gate_evidence["valid"] is True
    assert gate_evidence["release"] == ASSET_VER
    assert gate_evidence["gate"] == "pass"
    assert all(item["ratio"] >= 0.75 for item in gate_evidence["dimensions"].values())

    for item in objects["items"]:
        evidence_id = item["evidence_id"]
        assert re.fullmatch(r"ev:xrd:[a-z0-9_-]+", evidence_id)
        assert ASSET_VER not in evidence_id
        assert item["schema_version"] == "site31.evidence_object.v3"
        assert required.issubset(item)
        assert item["claims"] and item["distributions"]
        assert item["validation_status"] in {"verified", "implemented-partial", "manual-check", "planned"}
        assert item.get("title_en") and item.get("description_en")
        assert item.get("intended_use_en") and item.get("prohibited_use_en")
        assert item.get("limitations_en") and item.get("citation", {}).get("text_en")
        assert item.get("uncertainty", {}).get("statement_en")
        assert all(claim.get("statement_en") for claim in item["claims"])

        detail = get_json(client, "/api/evidence_objects/" + evidence_id)
        assert detail["evidence_id"] == evidence_id

        distribution = item["distributions"][0]
        snapshot = expect(client.get(distribution["href"], headers={"Host": "localhost"}),
                          200, distribution["href"])
        assert snapshot.headers.get("ETag")
        assert hashlib.sha256(snapshot.data).hexdigest() == distribution["sha256"]

    expect(client.get("/api/site31_portal", headers={"Host": "attacker.invalid"}), 421, "host allowlist")
    expect(client.get("/api/predictions/seed-yag-cr3", headers={"Host": "localhost"}), 404,
           "material id cannot identify a prediction")
    expect(client.get("/api/config", headers={"Host": "localhost"}), 401, "anonymous internal read")
    expect(client.get("/api/config", headers={"Host": "localhost", "X-User": "judge", "X-Role": "judge"}), 403, "judge internal read")
    expect(client.post("/api/config", headers={"Host": "localhost"}, json={}), 405, "anonymous write")
    expect(client.post("/api/config", headers={"Host": "localhost", "X-User": "judge", "X-Role": "judge"}, json={}), 403, "judge write")
    expect(client.post("/api/config", headers={"Host": "localhost", "X-User": "member", "X-Role": "member", "Origin": "https://attacker.invalid"}, json={}), 403, "cross-origin write")

    secure = expect(client.get("/", headers={"Host": "localhost"}, base_url="https://localhost"), 200, "security headers")
    for header in ("Content-Security-Policy", "Strict-Transport-Security", "Cross-Origin-Resource-Policy",
                   "Origin-Agent-Cluster", "X-Permitted-Cross-Domain-Policies"):
        assert secure.headers.get(header), header

print(json.dumps({
    "ok": True,
    "release": ASSET_VER,
    "evidence_objects": objects["count"],
    "evidence_schema": objects["schema_version"],
    "federated_search_results": search["total"],
    "spa_deep_links": 9,
    "trust_controls": len(trust.get("controls", [])),
    "scorecard_gate": scorecard["gate"],
    "scorecard_score": scorecard["score"],
    "gate_sha256": gate_evidence["artifact_sha256"],
}, ensure_ascii=False))
