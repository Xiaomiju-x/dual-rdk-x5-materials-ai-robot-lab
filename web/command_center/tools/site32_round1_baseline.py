#!/usr/bin/env python3
"""Generate a reproducible Site32 Round 1 architecture and asset baseline."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "site32.round1_baseline.v1"
R0_SCHEMA_VERSION = "site32.r0_baseline.v1"
TRACKED_ASSETS = (
    "app.py",
    "static/index.html",
    "static/app.js",
    "static/style.css",
    "static/i18n.js",
    "static/sw.js",
    "static/r4.css",
    "static/r4.js",
    "static/r4-performance.js",
    "static/r4-accessibility.js",
    "static/site32.css",
    "static/site32.js",
)
IMPORT_RE = re.compile(
    r"(?:^|\n)\s*(?:import\s+(?:[^'\"]+?\s+from\s+)?|export\s+[^'\"]+?\s+from\s+)"
    r"['\"]([^'\"]+)['\"]"
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _line_count(raw: bytes) -> int:
    text = raw.decode("utf-8-sig")
    return text.count("\n") + (0 if not text or text.endswith("\n") else 1)


def _asset_record(root: Path, relative: str) -> dict:
    path = root / relative
    raw = path.read_bytes()
    return {
        "path": relative,
        "bytes": len(raw),
        "gzip_bytes": len(gzip.compress(raw, compresslevel=9, mtime=0)),
        "lines": _line_count(raw),
        "sha256": _sha256(raw),
    }


def _site32_module_graph(root: Path) -> dict:
    module_root = root / "static" / "src" / "site32"
    modules = sorted(module_root.glob("*.js"))
    graph = {}
    for path in modules:
        text = path.read_text(encoding="utf-8-sig")
        graph[path.name] = sorted(set(IMPORT_RE.findall(text)))
    return {
        "module_count": len(modules),
        "modules": graph,
        "total_bytes": sum(path.stat().st_size for path in modules),
    }


def _load_runtime(root: Path):
    tools_root = root / "tools"
    for path in (str(root), str(tools_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from site31_r4_contract_test import _load_app_isolated

    module, observations, tempdir, error = _load_app_isolated(test_mode=True)
    if error is not None:
        tempdir.cleanup()
        raise RuntimeError(f"isolated app import failed: {error!r}")
    return module, observations, tempdir


def build_baseline(root: Path, *, r0: bool = False) -> dict:
    root = root.resolve()
    for path in (str(root), str(root / "tools")):
        if path not in sys.path:
            sys.path.insert(0, path)
    missing = [relative for relative in TRACKED_ASSETS if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"tracked assets are missing: {missing}")

    from site31_asset_manifest import build_manifest
    from site32_style_audit import collect_metrics
    from cmdcenter.route_contract import route_inventory, route_inventory_summary

    module, observations, tempdir = _load_runtime(root)
    try:
        routes = route_inventory(module.app)
        manifest = build_manifest(root)
        _css_files, css_aggregate = collect_metrics(root)
        assets = [_asset_record(root, relative) for relative in TRACKED_ASSETS]
        return {
            "schema_version": R0_SCHEMA_VERSION if r0 else SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "release": module.ASSET_VER,
            "manifest_digest": manifest["manifest_digest"],
            "manifest_file_count": manifest["file_count"],
            "assets": assets,
            "asset_totals": {
                "bytes": sum(item["bytes"] for item in assets),
                "gzip_bytes": sum(item["gzip_bytes"] for item in assets),
                "lines": sum(item["lines"] for item in assets),
            },
            "architecture": {
                "python_modules": sorted(
                    path.name for path in (root / "cmdcenter").glob("*.py")
                    if path.name != "__init__.py"
                ),
                "site32_frontend": _site32_module_graph(root),
                "routes": route_inventory_summary(routes),
                "route_role_method_source_matrix": routes,
            },
            "import_side_effects": {
                "thread_starts": observations["thread_starts"],
                "sqlite_connects": observations["sqlite_connects"],
                "subprocess_runs": observations["subprocess_runs"],
                "thread_delta": observations["threads_after"] - observations["threads_before"],
            },
            "css": css_aggregate,
            "interpretation": {
                "scope": (
                    "candidate-bound R0 release baseline"
                    if r0 else "local deterministic Round 1 baseline"
                ),
                "global_rank_claim": False,
                "field_performance": "not measured by this artifact",
                "external_security_assessment": "not measured by this artifact",
            },
        }
    finally:
        tempdir.cleanup()


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument(
        "--r0",
        action="store_true",
        help="emit the candidate-bound Site32 R0 baseline schema",
    )
    args = parser.parse_args()
    try:
        payload = build_baseline(args.root, r0=args.r0)
        if args.output:
            output = args.output if args.output.is_absolute() else args.root / args.output
            _atomic_write(output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
        return 0
    except Exception as exc:
        print(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
