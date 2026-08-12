"""Site32 product and access contracts exposed as read-only JSON."""

from __future__ import annotations

from flask import Blueprint, jsonify

from .access import public_matrix_payload
from .release import parse_release
from .route_contract import route_inventory, route_inventory_summary


def register_site32(app, *, release: str, released_at: str) -> Blueprint:
    identity = parse_release(release)
    blueprint = Blueprint("site32_contract", __name__)

    @blueprint.get("/api/site32/contract")
    def site32_contract():
        return jsonify({
            "schema_version": "site32.product_contract.v1",
            "release": release,
            "released_at": released_at,
            "product": {
                "name_zh": "荧光具身智研",
                "name_en": "Fluorescent Embodied Research",
                "category": "commercial-grade public materials research evidence portal",
                "audience": "global materials and near-infrared phosphor researchers",
            },
            "task_domains": ["research", "experiment", "evidence", "trust"],
            "access_layers": ["public", "reviewer", "internal"],
            "state_axes": ["runtime", "freshness", "scientific_conclusion"],
            "release_identity": {
                "product": identity.product,
                "version": identity.version,
                "date": identity.date,
                "generation": identity.generation,
            },
            "public_control_boundary": "read-only; no robot, chassis, arm or actuator authority",
            "claim_boundary": (
                "Built toward global top-tier commercial research-platform standards; "
                "not a self-certified global ranking, security certification, or WCAG certification."
            ),
        })

    @blueprint.get("/api/site32/access-matrix")
    def site32_access_matrix():
        payload = public_matrix_payload()
        payload["release"] = release
        inventory = route_inventory(app)
        payload["route_inventory"] = inventory
        payload["route_inventory_summary"] = route_inventory_summary(inventory)
        payload["enforcement"] = {
            "write_default_deny": True,
            "read_segmentation": "active in Flask; gateway and origin are additional layers",
        }
        return jsonify(payload)

    app.register_blueprint(blueprint)
    return blueprint
