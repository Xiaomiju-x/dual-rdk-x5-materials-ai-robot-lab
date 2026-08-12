from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from workstation.dual_arm_successor.adapters import (
    EvidenceContractError,
    EvidencePathError,
    build_shadow_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
AUTHORITATIVE_EVIDENCE = (
    REPO_ROOT
    / "workstation"
    / "dual_arm"
    / "evidence"
    / "finals_part3_execute_20260720_052630_4956"
)
SUCCESSOR_ROOT = REPO_ROOT / "workstation" / "dual_arm_successor"


def _copy_required_evidence(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True)
    shutil.copy2(source / "result.json", destination / "result.json")
    for directory, names in {
        "apriltag_live": ["exact_gate_summary.json"],
        "overhead_live": ["cpu_result.json", "bpu_result.json"],
    }.items():
        target = destination / directory
        target.mkdir()
        for name in names:
            shutil.copy2(source / directory / name, target / name)
    return destination


def test_authoritative_evidence_builds_stage_only_dataset(tmp_path: Path) -> None:
    evidence = _copy_required_evidence(AUTHORITATIVE_EVIDENCE, tmp_path / "evidence")
    output = tmp_path / "dataset"

    receipt = build_shadow_dataset(evidence, output)

    episode = json.loads((output / "episode.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert receipt["physical_state"] == "PHYSICAL_STATE_UNAVAILABLE"
    assert episode["outcome"]["status"] == "CLOSED_LOOP_DONE"
    assert episode["observations"]["apriltag"]["required_id"] == 2
    assert episode["observations"]["overhead_cpu"]["decision"] == "BAG_PRESENT"
    assert episode["authority"]["motion_authority"] is False
    assert episode["authority"]["execution_allowed"] is False
    assert episode["authority"]["actuator_commands_issued"] == 0
    assert episode["physical_state"]["action_vector"] is None
    assert episode["physical_state"]["action_dimension"] is None
    assert manifest["action_dimension"] is None
    assert all(stage["action"] is None for stage in episode["stages"])
    assert len((output / "stages.jsonl").read_text(encoding="utf-8").splitlines()) == len(
        episode["stages"]
    )
    manifest_hash = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    assert receipt["manifest_sha256"] == manifest_hash
    assert (output / "manifest.sha256").read_text(encoding="ascii") == (
        f"{manifest_hash}  manifest.json\n"
    )


def test_missing_required_evidence_is_rejected(tmp_path: Path) -> None:
    evidence = _copy_required_evidence(AUTHORITATIVE_EVIDENCE, tmp_path / "evidence")
    (evidence / "overhead_live" / "bpu_result.json").unlink()

    with pytest.raises(EvidenceContractError, match="missing required evidence"):
        build_shadow_dataset(evidence, tmp_path / "dataset")


def test_suspicious_reverse_read_and_path_overlap_are_rejected(tmp_path: Path) -> None:
    evidence = _copy_required_evidence(AUTHORITATIVE_EVIDENCE, tmp_path / "output" / "evidence")

    with pytest.raises(EvidencePathError, match="must not contain each other"):
        build_shadow_dataset(evidence, tmp_path / "output")

    suspicious = SUCCESSOR_ROOT / "_test_suspicious_evidence"
    try:
        _copy_required_evidence(AUTHORITATIVE_EVIDENCE, suspicious)
        with pytest.raises(EvidencePathError, match="inside dual_arm_successor"):
            build_shadow_dataset(suspicious, tmp_path / "dataset")
    finally:
        if suspicious.exists():
            shutil.rmtree(suspicious)


def test_existing_output_is_rejected_without_overwrite(tmp_path: Path) -> None:
    evidence = _copy_required_evidence(AUTHORITATIVE_EVIDENCE, tmp_path / "evidence")
    output = tmp_path / "dataset"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(EvidencePathError, match="must be a new directory"):
        build_shadow_dataset(evidence, output)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_hashes_are_stable_across_output_locations(tmp_path: Path) -> None:
    evidence = _copy_required_evidence(AUTHORITATIVE_EVIDENCE, tmp_path / "evidence")

    first = build_shadow_dataset(evidence, tmp_path / "dataset-a")
    second = build_shadow_dataset(evidence, tmp_path / "dataset-b")

    assert first["manifest_sha256"] == second["manifest_sha256"]
    for name in ("episode.json", "stages.jsonl", "manifest.json", "manifest.sha256"):
        first_bytes = (tmp_path / "dataset-a" / name).read_bytes()
        second_bytes = (tmp_path / "dataset-b" / name).read_bytes()
        assert first_bytes == second_bytes
        assert hashlib.sha256(first_bytes).hexdigest() == hashlib.sha256(second_bytes).hexdigest()


def test_source_contains_no_robot_runtime_dependencies() -> None:
    banned_roots = {"pymycobot", "serial", "RPi", "GPIO", "rclpy"}
    python_files = sorted(
        [
            *SUCCESSOR_ROOT.joinpath("adapters").glob("*.py"),
            SUCCESSOR_ROOT / "tools" / "build_shadow_dataset.py",
        ]
    )
    for path in python_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported.isdisjoint(banned_roots), f"{path} imports forbidden dependencies: {imported}"
