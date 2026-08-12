#!/usr/bin/env python3
"""Contract tests for the Site32 Public Research Commons projection."""

from __future__ import annotations

import importlib
import json
import sqlite3
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmdcenter.research_collections import (  # noqa: E402
    SCHEMA_VERSION,
    build_research_collections,
    collection_detail,
)


RELEASE = "site32-global-commercial-v1.7-20260712"
RELEASED_AT = "2026-07-12T11:00:00Z"
PROHIBITED = {
    "work_order", "batch", "operator", "ip", "port", "ssh", "network",
    "device_path", "gpio", "pwm", "actuator", "cmd_vel", "control_url",
    "raw_command", "token", "secret", "private_prompt", "raw_log",
}


def material(material_id: str = "seed-yag-cr3", **overrides):
    row = {
        "id": material_id,
        "formula": "Y3Al5O12:Cr3+",
        "host": "YAG",
        "dopant": "Cr3+",
        "site": "Al",
        "verdict": "REFERENCE",
        "lambda_em": 714.0,
        "confidence_interval": "",
        "band": "nir_i",
        "method": "curated public fixture",
        "source": "curated",
        "state": "replay",
        "created": "",
        "uncertainty": "curated replay; no live uncertainty",
        "detail_url": "/materials/seed-yag-cr3",
        "work_order": "WO-PRIVATE",
        "batch": "BATCH-PRIVATE",
        "ip": "192.0.2.103",
        "cmd_vel": "1.0",
    }
    row.update(overrides)
    return row


def evidence(evidence_id: str, scope: str = "public_site", **overrides):
    row = {
        "evidence_id": evidence_id,
        "kind": "document",
        "scope": scope,
        "title": evidence_id,
        "title_en": evidence_id,
        "description": "public evidence fixture",
        "source_label": "mirror",
        "claim_status": "curated",
        "validation_status": "verified",
        "origin": ["curated"],
        "canonical_url": "/evidence/" + evidence_id.replace(":", "%3A"),
        "freshness": {"state": "mirror"},
        "uncertainty": {"statement": "project evidence only"},
        "rights": {"license": "LicenseRef-XRD-Public-Summary", "access": "public-read-only"},
        "limitations": ["not a certification"],
        "limitations_en": ["not a certification"],
        "relations": [],
        "raw_log": "must never appear",
        "token": "must never appear",
    }
    row.update(overrides)
    return row


def payload(materials=None, evidence_objects=None, params=None):
    objects = evidence_objects or [
        evidence("ev:xrd:passport"),
        evidence("ev:xrd:materials", "ai_brain"),
        evidence("ev:xrd:prediction_engine", "ai_brain"),
        evidence("ev:xrd:slam_shadow", "embodied_brain"),
        evidence("ev:xrd:arm01_redundancy", "arm01"),
    ]
    return build_research_collections(
        materials=materials or [material()],
        evidence_objects=objects,
        release=RELEASE,
        released_at=RELEASED_AT,
        params=params or {},
    )


def all_keys(value):
    keys = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key).lower())
            keys.update(all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(all_keys(child))
    return keys


class Site32ResearchCollectionsTests(unittest.TestCase):
    def test_module_import_has_no_runtime_side_effects(self):
        observations = {"sqlite": [], "threads": [], "subprocess": []}

        def blocked_connect(*args, **kwargs):
            observations["sqlite"].append(args)
            raise AssertionError("sqlite is forbidden during module import")

        def blocked_start(thread, *args, **kwargs):
            observations["threads"].append(thread.name)
            raise AssertionError("threads are forbidden during module import")

        def blocked_run(*args, **kwargs):
            observations["subprocess"].append(args)
            raise AssertionError("subprocess is forbidden during module import")

        with mock.patch.object(sqlite3, "connect", side_effect=blocked_connect), \
                mock.patch.object(threading.Thread, "start", new=blocked_start), \
                mock.patch.object(subprocess, "run", side_effect=blocked_run):
            importlib.reload(importlib.import_module("cmdcenter.research_collections"))
        self.assertEqual(observations, {"sqlite": [], "threads": [], "subprocess": []})

    def test_schema_collections_and_release_binding(self):
        data = payload()
        self.assertEqual(data["schema_version"], SCHEMA_VERSION)
        self.assertEqual(data["release"], RELEASE)
        self.assertEqual(data["as_of"], RELEASED_AT)
        self.assertEqual(data["count"], 5)
        self.assertEqual(data["items"][0]["collection_id"], "rc:xrd:materials-atlas")
        self.assertTrue(all(item["visibility"] == "public-read-only" for item in data["items"]))
        self.assertTrue(all(item["updated_at"] == RELEASED_AT for item in data["items"]))

    def test_projection_recursively_drops_private_and_control_fields(self):
        data = payload()
        rendered = json.dumps(data, ensure_ascii=False).lower()
        self.assertFalse(PROHIBITED & all_keys(data))
        self.assertNotIn("wo-private", rendered)
        self.assertNotIn("batch-private", rendered)
        self.assertNotIn("192.0.2.103", rendered)
        self.assertNotIn("must never appear", rendered)

    def test_material_identity_is_stable_and_never_uses_row_index(self):
        first = payload(materials=[material("observed-1", created="2026-07-01")])
        second = payload(materials=[material("observed-99", created="2026-07-01")])
        first_id = next(
            member["object_id"] for member in first["items"][0]["members"]
            if member["kind"] == "material"
        )
        second_id = next(
            member["object_id"] for member in second["items"][0]["members"]
            if member["kind"] == "material"
        )
        self.assertEqual(first_id, second_id)
        self.assertRegex(first_id, r"^mat:xrd:sha256-[0-9a-f]{24}$")
        self.assertNotIn("observed-", first_id)

    def test_filters_and_zero_result_have_explicit_terminal_state(self):
        embodied = payload(params={"scope": "embodied_brain", "has_kind": "evidence"})
        self.assertEqual(embodied["count"], 1)
        self.assertEqual(embodied["items"][0]["collection_id"], "rc:xrd:embodied-replay")
        missing = payload(params={"q": "no-such-public-collection"})
        self.assertEqual(missing["items"], [])
        self.assertEqual(missing["empty"]["reason"], "no_match")

    def test_detail_is_stable_and_unknown_collection_is_missing(self):
        data = payload()
        detail = collection_detail(data, "rc:xrd:arm01-redundancy")
        self.assertEqual(detail["item"]["scope"], "arm01")
        rendered = json.dumps(detail, ensure_ascii=False)
        self.assertIn("arm02", rendered)
        self.assertIn("CPU/OpenCV", rendered)
        self.assertIn("BPU is supporting", rendered)
        self.assertIsNone(collection_detail(data, "rc:xrd:missing"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
