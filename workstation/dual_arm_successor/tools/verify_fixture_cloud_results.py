#!/usr/bin/env python3
"""Verify downloaded RTX 5090 fixture results and write a local acceptance receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


ACCEPTED_STATUS = "RTX5090_FIXTURE_REPLAY_ACCEPTED_NOT_REAL_POLICY"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def safe_member(member: tarfile.TarInfo) -> bool:
    value = PurePosixPath(member.name)
    return (
        not value.is_absolute()
        and ".." not in value.parts
        and (member.isfile() or member.isdir())
        and not member.issym()
        and not member.islnk()
        and not member.isdev()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence_dir.expanduser().resolve()
    archive_receipts = sorted(evidence.glob("xrd-cloud5090-results-*.json"))
    archives = sorted(evidence.glob("xrd-cloud5090-results-*.tar.gz"))
    if len(archive_receipts) != 1 or len(archives) != 1:
        raise SystemExit("exactly one result archive and receipt are required")
    archive_receipt = read_object(archive_receipts[0])
    archive = archives[0]
    if sha256_file(archive) != archive_receipt.get("archive_sha256"):
        raise SystemExit("result archive SHA-256 mismatch")

    extract_dir = evidence / "extracted"
    if extract_dir.exists():
        raise SystemExit(f"immutable extraction directory already exists: {extract_dir}")
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if not members or not all(safe_member(member) for member in members):
            raise SystemExit("unsafe archive member detected")
        bundle.extractall(extract_dir, filter="data")

    package_manifest_path = extract_dir / "MANIFEST.json"
    package_manifest = read_object(package_manifest_path)
    files = package_manifest.get("files")
    if not isinstance(files, list):
        raise SystemExit("package file manifest is missing")
    verified_files: list[dict[str, Any]] = []
    for item in files:
        relative = PurePosixPath(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"unsafe manifest path: {relative}")
        path = extract_dir / "results" / Path(*relative.parts)
        if not path.is_file():
            raise SystemExit(f"missing result file: {relative}")
        actual = sha256_file(path)
        if actual != item["sha256"] or path.stat().st_size != int(item["bytes"]):
            raise SystemExit(f"result file mismatch: {relative}")
        verified_files.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": actual,
            }
        )
    expected_paths = {f"results/{item['path']}" for item in files}
    actual_paths = {
        path.relative_to(extract_dir).as_posix()
        for path in extract_dir.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    if actual_paths != expected_paths:
        raise SystemExit("archive contains unmanifested or missing result files")

    run_receipt = read_object(evidence / "run_receipt.json")
    preflight = read_object(evidence / "fixture_preflight.json")
    aggregate_download = evidence / "fixture_aggregate.json"
    aggregate_packaged = extract_dir / "results" / "fixture_aggregate.json"
    if sha256_file(aggregate_download) != sha256_file(aggregate_packaged):
        raise SystemExit("downloaded and packaged aggregate receipts differ")
    aggregate = read_object(aggregate_download)
    if run_receipt.get("status") != "PASS":
        raise SystemExit("cloud run receipt did not pass")
    if preflight.get("status") != "PASS_FIXTURE_ONLY":
        raise SystemExit("fixture preflight did not pass")
    if aggregate.get("status") != "PASS_FIXTURE_ONLY":
        raise SystemExit("fixture aggregate did not pass")
    if not (preflight.get("dataset") or {}).get("gate_pass"):
        raise SystemExit("fixture dataset gate did not pass")
    gpu_devices = (
        ((run_receipt.get("machine") or {}).get("torch") or {}).get("devices") or []
    )
    if not gpu_devices or "RTX 5090" not in str(gpu_devices[0].get("name") or ""):
        raise SystemExit("RTX 5090 identity missing from run receipt")
    for truth in (
        preflight.get("truthfulness") or {},
        aggregate.get("truthfulness") or {},
    ):
        if (
            truth.get("motion_authority") is not False
            or truth.get("execution_allowed") is not False
            or truth.get("actuator_commands_issued") != 0
            or truth.get("real_robot_policy") is not False
            or truth.get("measured_robot_telemetry") is not False
        ):
            raise SystemExit("fixture truth boundary violation")

    metric_receipts: list[dict[str, Any]] = []
    for model in ("tiny_act", "world_model"):
        paths = sorted((extract_dir / "results" / model).glob("seed_*/metrics.json"))
        if len(paths) != 3:
            raise SystemExit(f"expected three {model} metrics receipts")
        for metrics_path in paths:
            metrics = read_object(metrics_path)
            if (
                metrics.get("status") != "PASS_FIXTURE_ONLY"
                or metrics.get("model") != model
                or (metrics.get("truthfulness") or {}).get("real_robot_policy")
                is not False
            ):
                raise SystemExit(f"invalid fixture metrics: {metrics_path}")
            checkpoint = metrics_path.parent / "checkpoint.pt"
            onnx_path = metrics_path.parent / "model.onnx"
            if sha256_file(checkpoint) != metrics["checkpoint"]["sha256"]:
                raise SystemExit(f"checkpoint hash mismatch: {metrics_path}")
            if sha256_file(onnx_path) != metrics["onnx"]["sha256"]:
                raise SystemExit(f"ONNX hash mismatch: {metrics_path}")
            if metrics["onnx"].get("checker") != "PASS":
                raise SystemExit(f"ONNX checker did not pass: {metrics_path}")
            metric_receipts.append(
                {
                    "model": model,
                    "seed": int(metrics["seed"]),
                    "metrics_sha256": sha256_file(metrics_path),
                    "checkpoint_sha256": metrics["checkpoint"]["sha256"],
                    "onnx_sha256": metrics["onnx"]["sha256"],
                    "evaluation": metrics["evaluation"],
                }
            )

    verification = {
        "schema_version": "xrd-fixture-cloud-verification-v1",
        "status": ACCEPTED_STATUS,
        "archive": {
            "name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
            "source_tree_sha256": package_manifest["source_tree_sha256"],
            "verified_files": len(verified_files),
        },
        "cloud": {
            "hostname": (run_receipt.get("machine") or {}).get("hostname"),
            "gpu": gpu_devices[0],
            "process_exit_code": run_receipt.get("process_exit_code"),
        },
        "dataset": {
            "sha256": aggregate["training_data_sha256"],
            "rows": preflight["dataset"]["rows"],
            "episodes": preflight["dataset"]["episodes"],
            "task_episode_counts": preflight["dataset"]["task_episode_counts"],
            "provenance_state": "COMMAND_DERIVED_DIGITAL_TWIN",
            "measured_robot_telemetry": False,
            "synchronized_camera_actions": False,
        },
        "selection": {
            "tiny_act_best_seed": aggregate["models"]["tiny_act"]["best_seed"],
            "tiny_act_best_onnx_sha256": aggregate["models"]["tiny_act"][
                "best_onnx_sha256"
            ],
            "world_model_best_seed": aggregate["models"]["world_model"][
                "best_seed"
            ],
            "world_model_best_onnx_sha256": aggregate["models"]["world_model"][
                "best_onnx_sha256"
            ],
        },
        "metrics": metric_receipts,
        "truthfulness": {
            "fixture_replay_only": True,
            "real_robot_policy": False,
            "motion_authority": False,
            "execution_allowed": False,
            "actuator_commands_issued": 0,
            "deployment_eligible": False,
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": verification["status"],
                "verified_files": len(verified_files),
                "models": len(metric_receipts),
                "output": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
