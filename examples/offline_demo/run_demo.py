#!/usr/bin/env python3
"""Run a deterministic, hardware-free inspection of the public XRD release.

This command performs no network access, imports no robot SDK, and opens no
serial/camera device. It turns the committed contracts into one small JSON
receipt so a new contributor can verify the repository with stock Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workstation_public.interlock import fk_points  # noqa: E402


REGISTRY = ROOT / "ai_brain" / "icmat_foundry" / "finals_50model" / "contracts" / "model_registry.v3.json"
ACCEPTANCE = ROOT / "evidence" / "ai_brain" / "final_acceptance.v1.json"
STATUS_EXAMPLE = ROOT / "schemas" / "status_snapshot_example.json"


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_receipt() -> dict[str, object]:
    registry = _read_json(REGISTRY)
    acceptance = _read_json(ACCEPTANCE)
    status = _read_json(STATUS_EXAMPLE)

    if not isinstance(registry, dict) or not isinstance(registry.get("models"), list):
        raise ValueError("model registry does not contain a models list")
    if not isinstance(acceptance, dict) or not isinstance(acceptance.get("counts"), dict):
        raise ValueError("acceptance receipt does not contain counts")
    if not isinstance(status, dict):
        raise ValueError("status example is not a JSON object")

    pose = [0.0, 8.0, -127.0, 40.0, 0.0, 45.0]
    arm01_points = fk_points(pose)
    arm02_points = fk_points(pose, base=(400.0, 0.0, 0.0), base_yaw_deg=180.0)

    return {
        "schema": "xrd_smart_lab.offline_inspection.v1",
        "mode": "OFFLINE_SYNTHETIC_NO_ACTUATION",
        "side_effects": {
            "network": False,
            "camera": False,
            "serial": False,
            "robot_sdk": False,
            "writes": False,
        },
        "contracts": {
            "registry_models": len(registry["models"]),
            "release_ready_models": acceptance["counts"].get("release_ready_models"),
            "bpu_pc_toolchain_compiled": acceptance["counts"].get("bpu_pc_toolchain_compiled"),
            "acceptance_sha256": _sha256(ACCEPTANCE),
            "status_example_fields": sorted(status),
        },
        "kinematics_fixture": {
            "source": "synthetic known pose; not robot telemetry",
            "arm01_tool_xyz_mm": [round(value, 3) for value in arm01_points[-1]],
            "arm02_tool_xyz_mm": [round(value, 3) for value in arm02_points[-1]],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Omit it to keep the run read-only and print to stdout.",
    )
    args = parser.parse_args()

    rendered = json.dumps(build_receipt(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
