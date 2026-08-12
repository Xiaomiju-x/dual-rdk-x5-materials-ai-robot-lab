"""Convert frozen finals evidence into a canonical, stage-only shadow episode."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SUCCESSOR_ROOT = Path(__file__).resolve().parents[1]
MAX_JSON_BYTES = 16 * 1024 * 1024


class DatasetAdapterError(RuntimeError):
    """Base error for deterministic dataset adaptation."""


class EvidencePathError(DatasetAdapterError):
    """Raised when an input or output path violates the read-only boundary."""


class EvidenceContractError(DatasetAdapterError):
    """Raised when frozen evidence is missing or violates its authority contract."""


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _resolve_source_file(evidence_dir: Path, relative_path: str) -> Path:
    candidate = evidence_dir.joinpath(*Path(relative_path).parts)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise EvidenceContractError(f"missing required evidence: {relative_path}") from exc
    if not resolved.is_file():
        raise EvidenceContractError(f"required evidence is not a file: {relative_path}")
    if not _is_within(resolved, evidence_dir):
        raise EvidencePathError(f"evidence path escapes --evidence-dir: {relative_path}")
    if resolved.stat().st_size > MAX_JSON_BYTES:
        raise EvidenceContractError(f"evidence JSON exceeds {MAX_JSON_BYTES} bytes: {relative_path}")
    return resolved


def _load_json(evidence_dir: Path, relative_path: str) -> tuple[dict[str, Any], Path]:
    path = _resolve_source_file(evidence_dir, relative_path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceContractError(f"invalid UTF-8 JSON evidence: {relative_path}") from exc
    if not isinstance(value, dict):
        raise EvidenceContractError(f"evidence root must be an object: {relative_path}")
    return value, path


def _require(value: bool, message: str) -> None:
    if not value:
        raise EvidenceContractError(message)


def _select_layout(result: dict[str, Any]) -> dict[str, str]:
    mode = result.get("mode")
    if mode == "EXECUTE":
        return {
            "result": "result.json",
            "apriltag": "apriltag_live/exact_gate_summary.json",
            "cpu": "overhead_live/cpu_result.json",
            "bpu": "overhead_live/bpu_result.json",
        }
    if mode in {"VALIDATE_ONLY", "VALIDATE"}:
        return {
            "result": "result.json",
            "apriltag": "apriltag_replay/exact_gate_summary.json",
            "cpu": "overhead_replay/cpu_result.json",
            "bpu": "overhead_replay/bpu_result.json",
        }
    raise EvidenceContractError(f"unsupported result mode: {mode!r}")


def _validate_contracts(
    result: dict[str, Any],
    apriltag: dict[str, Any],
    cpu: dict[str, Any],
    bpu: dict[str, Any],
) -> None:
    _require(
        result.get("schema_version") == "xrd-finals-part3-composed-v1",
        "result.json has an unsupported schema_version",
    )
    _require(
        apriltag.get("schema_version") == "xrd-finals-apriltag-exact-gate-v1",
        "AprilTag evidence has an unsupported schema_version",
    )
    _require(
        apriltag.get("required_dict") == "DICT_APRILTAG_36h11"
        and apriltag.get("required_id") == 2,
        "AprilTag evidence is not the frozen DICT_APRILTAG_36h11 id=2 gate",
    )
    _require(apriltag.get("motion_authority") is False, "AprilTag evidence claims motion authority")
    _require(
        cpu.get("schema_version") == "xrd-overhead-bag-presence-v3",
        "overhead CPU evidence has an unsupported schema_version",
    )
    _require(cpu.get("motion_authority") is False, "overhead CPU evidence claims motion authority")
    _require(
        bpu.get("schema_version") == "xrd-overhead-bpu-auxiliary-v1",
        "overhead BPU evidence has an unsupported schema_version",
    )
    _require(bpu.get("motion_authority") is False, "overhead BPU evidence claims motion authority")
    _require(
        bpu.get("bag_presence_authority") is False,
        "overhead BPU evidence incorrectly claims bag-presence authority",
    )
    _require(
        bpu.get("robot_sdk_serial_gpio_access") is False,
        "overhead BPU evidence reports robot/serial/GPIO access",
    )


def _frame_summary(frame: dict[str, Any], *, kind: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "kind": kind,
        "name": frame.get("name") or frame.get("frame"),
        "sha256": frame.get("sha256") or frame.get("frame_sha256"),
    }
    for key in ("exact_ok", "bag_present", "color_gate_passed", "baseline_change_gate_passed"):
        if key in frame:
            summary[key] = frame[key]
    if "metrics" in frame:
        summary["metrics"] = frame["metrics"]
    if "exact_hits" in frame:
        summary["exact_hits"] = [
            {
                "dict": hit.get("dict"),
                "id": hit.get("id"),
                "center": hit.get("center"),
                "edge_px": hit.get("edge_px"),
            }
            for hit in frame["exact_hits"]
            if isinstance(hit, dict)
        ]
    return summary


def _build_episode(
    result: dict[str, Any],
    apriltag: dict[str, Any],
    cpu: dict[str, Any],
    bpu: dict[str, Any],
    source_files: list[dict[str, Any]],
    episode_id: str,
) -> dict[str, Any]:
    events = result.get("events", [])
    _require(isinstance(events, list), "result events must be a list")
    stages = []
    for index, event in enumerate(events):
        _require(isinstance(event, dict), f"result event {index} must be an object")
        stages.append(
            {
                "index": index,
                "phase": event.get("phase"),
                "status": event.get("status"),
                "time": event.get("time"),
                "observation_only": True,
                "action": None,
                "physical_state": "PHYSICAL_STATE_UNAVAILABLE",
            }
        )

    empty_frames = cpu.get("empty", {}).get("files", [])
    occupied_frames = cpu.get("occupied", {}).get("files", [])
    tag_frames = apriltag.get("frames", [])
    _require(all(isinstance(item, dict) for item in tag_frames), "AprilTag frames must be objects")
    _require(all(isinstance(item, dict) for item in empty_frames), "empty frames must be objects")
    _require(all(isinstance(item, dict) for item in occupied_frames), "occupied frames must be objects")

    return {
        "schema_version": "xrd-dual-arm-shadow-episode-v1",
        "episode_id": episode_id,
        "source": {
            "kind": "frozen_finals_evidence_copy",
            "files": source_files,
        },
        "outcome": {
            "mode": result.get("mode"),
            "status": result.get("status"),
            "both_arms": next(
                (
                    event.get("both_arms")
                    for event in events
                    if isinstance(event, dict) and event.get("phase") == "dual_arm_v3"
                ),
                None,
            ),
        },
        "authority": {
            "motion_authority": False,
            "execution_allowed": False,
            "actuator_commands_issued": 0,
            "bag_presence_authority": "AI_X5_CPU_OPENCV",
            "bpu_role": "AUXILIARY_ONLY",
        },
        "physical_state": {
            "availability": "PHYSICAL_STATE_UNAVAILABLE",
            "joint_state": None,
            "action_vector": None,
            "action_dimension": None,
            "reason": "continuous joint and action telemetry is absent from the frozen evidence",
        },
        "observations": {
            "apriltag": {
                "required_dict": apriltag.get("required_dict"),
                "required_id": apriltag.get("required_id"),
                "passed": apriltag.get("passed"),
                "frames_exact_pass": apriltag.get("frames_exact_pass"),
                "frames_total": apriltag.get("frames_total"),
                "frames": [_frame_summary(frame, kind="APRILTAG_EXACT_GATE") for frame in tag_frames],
            },
            "overhead_cpu": {
                "decision": cpu.get("decision"),
                "positive_count": cpu.get("occupied", {}).get("positive_count"),
                "occupied_count": cpu.get("occupied", {}).get("count"),
                "empty_frames": [_frame_summary(frame, kind="EMPTY_DISH") for frame in empty_frames],
                "occupied_frames": [
                    _frame_summary(frame, kind="BAG_IN_DISH") for frame in occupied_frames
                ],
            },
            "overhead_bpu": {
                "forward_executed": bpu.get("bpu_forward_executed"),
                "forward_count": bpu.get("forward_count"),
                "model_name": bpu.get("model", {}).get("name"),
                "model_sha256": bpu.get("model", {}).get("sha256"),
                "measured_latency_ms": bpu.get("measured_latency_ms"),
                "role": bpu.get("role"),
            },
        },
        "stages": stages,
    }


def _prepare_paths(evidence_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    try:
        evidence = evidence_dir.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise EvidencePathError(f"--evidence-dir does not exist: {evidence_dir}") from exc
    if not evidence.is_dir():
        raise EvidencePathError(f"--evidence-dir is not a directory: {evidence_dir}")
    if _is_within(evidence, SUCCESSOR_ROOT):
        raise EvidencePathError("refusing to read evidence from inside dual_arm_successor")

    output = output_dir.expanduser().resolve(strict=False)
    if _is_within(evidence, output) or _is_within(output, evidence):
        raise EvidencePathError("evidence and output directories must not contain each other")
    if output.exists():
        raise EvidencePathError(f"--output-dir must be a new directory: {output}")
    try:
        output.parent.resolve(strict=True)
    except FileNotFoundError as exc:
        raise EvidencePathError(f"--output-dir parent does not exist: {output.parent}") from exc
    return evidence, output


def build_shadow_dataset(evidence_dir: Path | str, output_dir: Path | str) -> dict[str, Any]:
    """Build immutable canonical JSON/JSONL data from a frozen evidence directory."""

    evidence, output = _prepare_paths(Path(evidence_dir), Path(output_dir))
    result, result_path = _load_json(evidence, "result.json")
    layout = _select_layout(result)
    apriltag, apriltag_path = _load_json(evidence, layout["apriltag"])
    cpu, cpu_path = _load_json(evidence, layout["cpu"])
    bpu, bpu_path = _load_json(evidence, layout["bpu"])
    _validate_contracts(result, apriltag, cpu, bpu)

    resolved_files = [
        (layout["result"], result_path),
        (layout["apriltag"], apriltag_path),
        (layout["cpu"], cpu_path),
        (layout["bpu"], bpu_path),
    ]
    source_files = [
        {"path": relative, "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
        for relative, path in resolved_files
    ]
    episode_id = f"finals-part3-{source_files[0]['sha256'][:12]}"
    episode = _build_episode(result, apriltag, cpu, bpu, source_files, episode_id)

    episode_bytes = _canonical_bytes(episode)
    jsonl_rows = [
        {
            "schema_version": "xrd-dual-arm-shadow-stage-v1",
            "episode_id": episode_id,
            **stage,
        }
        for stage in episode["stages"]
    ]
    jsonl_bytes = b"".join(_canonical_bytes(row) for row in jsonl_rows)

    output.mkdir()
    episode_path = output / "episode.json"
    stages_path = output / "stages.jsonl"
    episode_path.write_bytes(episode_bytes)
    stages_path.write_bytes(jsonl_bytes)

    manifest = {
        "schema_version": "xrd-dual-arm-shadow-dataset-manifest-v1",
        "episode_id": episode_id,
        "physical_state": "PHYSICAL_STATE_UNAVAILABLE",
        "action_dimension": None,
        "source_files": source_files,
        "outputs": [
            {
                "path": "episode.json",
                "sha256": hashlib.sha256(episode_bytes).hexdigest(),
                "size_bytes": len(episode_bytes),
            },
            {
                "path": "stages.jsonl",
                "sha256": hashlib.sha256(jsonl_bytes).hexdigest(),
                "size_bytes": len(jsonl_bytes),
            },
        ],
    }
    manifest_bytes = _canonical_bytes(manifest)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    (output / "manifest.json").write_bytes(manifest_bytes)
    (output / "manifest.sha256").write_text(
        f"{manifest_hash}  manifest.json\n",
        encoding="ascii",
        newline="\n",
    )
    return {
        "output_dir": str(output),
        "episode_id": episode_id,
        "manifest_sha256": manifest_hash,
        "stage_count": len(jsonl_rows),
        "physical_state": "PHYSICAL_STATE_UNAVAILABLE",
    }
