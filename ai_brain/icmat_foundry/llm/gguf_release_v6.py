"""Fail-closed GGUF release builder for the ICMat v6 evidence pointer model.

This module consumes a model that has already been selected, calibrated, and
blind-tested. It never trains a model and never chooses a checkpoint. A release
is published only after the selected adapter is merged into an HF candidate,
converted to F16 GGUF, quantized to Q4_K_M, and shown to preserve strict pointer
and compiler behavior on a fixed non-blind golden set.

All runtime measurements produced here describe the local PC CPU. They are not
RDK X5 measurements and do not establish or use a BPU backend.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from icmat_foundry.llm import (
    blind_protocol_v6,
    calibration_eval_v6,
    pointer_hf_eval_v6,
    selection_freeze_v6,
)
from icmat_foundry.llm.evidence_pointer_v6 import (
    POINTER_KEYS,
    TRUSTED_FINISH_REASONS,
    compile_pointer,
)

RELEASE_BUILDER_VERSION = "icmat-gguf-release-v6.3.0"
PREFLIGHT_SCHEMA = "icmat_llm_gguf_release_preflight.v6"
GOLDEN_SET_SCHEMA = "icmat_pointer_release_golden_set.v6"
GOLDEN_SAMPLE_SCHEMA = "icmat_pointer_release_golden_sample.v6"
PARITY_SAMPLE_SCHEMA = "icmat_hf_gguf_pointer_parity_sample.v6"
PARITY_REPORT_SCHEMA = "icmat_hf_gguf_pointer_parity_report.v6"
RELEASE_RECEIPT_SCHEMA = "icmat_llm_gguf_release_receipt.v6"
FAILURE_RECEIPT_SCHEMA = "icmat_llm_gguf_release_failure_receipt.v6"

SELECTION_FREEZE_SCHEMA = selection_freeze_v6.SCHEMA
SELECTION_FREEZE_STATUS = selection_freeze_v6.STATUS
CALIBRATION_RECEIPT_SCHEMA = calibration_eval_v6.RECEIPT_SCHEMA
CALIBRATION_PASS_STATUS = "PASS_NONBLIND_CALIBRATION_MODEL_BOUND"
BLIND_RECEIPT_SCHEMA = blind_protocol_v6.RUN_RECEIPT_SCHEMA
BLIND_PASS_STATUS = "BLIND_EVALUATION_COMPLETE_FINAL_REPORT_ONLY"
BLIND_QUALIFICATION_SCHEMA = blind_protocol_v6.RELEASE_QUALIFICATION_SCHEMA
BLIND_QUALIFICATION_PASS_STATUS = blind_protocol_v6.RELEASE_QUALIFICATION_PASS_STATUS

PREFLIGHT_PASS_STATUS = "PASS_GGUF_RELEASE_PREFLIGHT_READY_NOT_EXECUTED"
PREFLIGHT_BLOCKED_STATUS = "BLOCKED_LOCAL_GGUF_TOOLCHAIN_OR_RUNTIME_UNAVAILABLE_NOT_EXECUTED"
PARITY_PASS_STATUS = "PASS_STRICT_HF_GGUF_POINTER_AND_COMPILER_PARITY"
PARITY_FAIL_STATUS = "FAIL_STRICT_HF_GGUF_POINTER_OR_COMPILER_PARITY"
RELEASE_PASS_STATUS = "PASS_PC_CPU_GGUF_RELEASE_NOT_ACTIVATED"
FIXTURE_PASS_STATUS = "PASS_FIXTURE_GGUF_PIPELINE_NOT_RELEASE_EVIDENCE"

DEFAULT_F16_NAME = "icmat-qwen05b-pointer-f16.gguf"
DEFAULT_Q4_NAME = "icmat-qwen05b-pointer-q4_k_m.gguf"
DEFAULT_MERGED_HF_NAME = "merged_hf"
DEFAULT_PREFLIGHT_NAME = "preflight.v6.json"
DEFAULT_PARITY_ROWS_NAME = "parity_samples.v6.jsonl"
DEFAULT_PARITY_REPORT_NAME = "hf_gguf_parity.v6.json"
DEFAULT_RECEIPT_NAME = "release_receipt.v6.json"
DEFAULT_DISABLED_MARKER = "ACTIVATION_DISABLED.txt"
DEFAULT_GOLDEN_NAME = "validation_golden_set.v6.json"

MAX_GOLDEN_BYTES = 32 * 1024 * 1024
MAX_INPUT_TOKENS = 1536
MAX_OUTPUT_TOKENS = 64
DEFAULT_SEED = 20260729
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONVERTER = (
    WORKSPACE_ROOT
    / "research"
    / "toolchains"
    / "llama_cpp_b10158_source"
    / "llama.cpp-b10158"
    / "convert_hf_to_gguf.py"
)
DEFAULT_QUANTIZER = (
    WORKSPACE_ROOT
    / "research"
    / "toolchains"
    / "llama_cpp_b10158_win_cuda13_3"
    / "runtime"
    / "llama-quantize.exe"
)
DEFAULT_LLAMA_SERVER = (
    WORKSPACE_ROOT
    / "research"
    / "toolchains"
    / "llama_cpp_b10158_win_cuda13_3"
    / "runtime"
    / "llama-server.exe"
)
DEFAULT_CONVERTER_SHA256 = "8f1bed9466221e57e434caa7ee720abe1569deb6bc2fe5a65da950ea66c8e737"
DEFAULT_QUANTIZER_SHA256 = "05d456e8ef4d5a670c7d63faf307c800426f1cb812e744d18f37e4bbc13248f3"
DEFAULT_LLAMA_SERVER_SHA256 = "bc2b10a5eb737eeaf14e95080a4fe0d16b9db4b92ffc5c3e35d86774b8e9561b"

CLAIM_BOUNDARY = {
    "execution_scope": "LOCAL_PC_CPU_ONLY_NOT_RDK_X5",
    "pc_cpu_latency_measured": True,
    "pc_memory_measured": True,
    "rdk_x5_measured": False,
    "bpu_used": False,
    "bpu_supported_or_claimed": False,
    "training_performed": False,
    "model_selection_performed": False,
    "production_activated": False,
    "services_modified": False,
    "network_required": False,
}

CommandRunner = Callable[[Sequence[str], Path], Any]
MergeHook = Callable[[Path, Path, Path], Mapping[str, Any] | None]
RuntimeRunner = Callable[
    [Sequence["GoldenRecord"], Path, "RuntimeConfig"],
    tuple[Mapping[str, "RuntimeObservation"], Mapping[str, Any]],
]
DependencyProbe = Callable[[str], bool]


class GgufReleaseV6Error(RuntimeError):
    """Raised when a release contract or execution gate fails."""


@dataclass(frozen=True)
class ReleaseInputs:
    """Pinned inputs for a release that was selected elsewhere."""

    base_model_dir: Path
    selected_adapter_dir: Path
    selection_freeze: Path
    selection_freeze_sha256: str
    calibration_receipt: Path
    calibration_receipt_sha256: str
    blind_receipt: Path
    blind_receipt_sha256: str
    blind_qualification_receipt: Path | None = None
    blind_qualification_receipt_sha256: str | None = None
    dataset_dir: Path | None = None
    # Deprecated constructor-only compatibility. Caller-supplied golden sets
    # are ignored; the release tool always derives its own validation set.
    golden_set: Path | None = None
    golden_set_sha256: str | None = None
    converter: Path = DEFAULT_CONVERTER
    converter_sha256: str = DEFAULT_CONVERTER_SHA256
    quantizer: Path = DEFAULT_QUANTIZER
    quantizer_sha256: str = DEFAULT_QUANTIZER_SHA256
    llama_server: Path = DEFAULT_LLAMA_SERVER
    llama_server_sha256: str = DEFAULT_LLAMA_SERVER_SHA256
    python_executable: Path = Path(sys.executable)


@dataclass(frozen=True)
class RuntimeConfig:
    """Fixed singleton greedy CPU parity configuration."""

    threads: int = 4
    context_size: int = MAX_INPUT_TOKENS
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    seed: int = DEFAULT_SEED
    startup_timeout_seconds: float = 120.0
    request_timeout_seconds: float = 180.0


@dataclass(frozen=True)
class GoldenRecord:
    example_id: str
    prompt: dict[str, Any]
    evidence: tuple[dict[str, Any], ...]
    expected_pointer: dict[str, Any]
    expected_compilation: dict[str, Any]


@dataclass(frozen=True)
class RuntimeObservation:
    raw_pointer: str
    finish_reason: str
    latency_ms: float
    peak_rss_bytes: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    generation_error: str | None = None


@dataclass(frozen=True)
class FixtureExecutionHarness:
    """Test-only hooks that can never produce a release-quality receipt."""

    merge_hook: MergeHook
    command_runner: CommandRunner
    hf_runner: RuntimeRunner
    gguf_runner: RuntimeRunner
    dependency_probe: DependencyProbe


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(values: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join((canonical_json(dict(value)) + "\n").encode("utf-8") for value in values)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GgufReleaseV6Error(f"duplicate JSON key rejected: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise GgufReleaseV6Error(f"non-finite JSON value rejected: {value}")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise GgufReleaseV6Error(f"{label} must be a lowercase SHA-256")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise GgufReleaseV6Error(f"{label} must be a non-empty string")
    return value


def _stable_regular_file(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[Path, bytes, dict[str, Any]]:
    raw = Path(path)
    if raw.is_symlink():
        raise GgufReleaseV6Error(f"{label} must not be a symlink: {raw}")
    try:
        mode = raw.stat(follow_symlinks=False).st_mode
    except OSError as exc:
        raise GgufReleaseV6Error(f"{label} is unavailable: {raw}") from exc
    if not stat.S_ISREG(mode):
        raise GgufReleaseV6Error(f"{label} must be a regular file: {raw}")
    resolved = raw.resolve(strict=True)
    before = resolved.stat()
    first = resolved.read_bytes()
    middle = resolved.stat()
    second = resolved.read_bytes()
    after = resolved.stat()
    if (
        first != second
        or len(
            {
                (before.st_size, before.st_mtime_ns),
                (middle.st_size, middle.st_mtime_ns),
                (after.st_size, after.st_mtime_ns),
            }
        )
        != 1
    ):
        raise GgufReleaseV6Error(f"{label} changed while it was read")
    actual = sha256_bytes(first)
    record = {
        "path": str(resolved),
        "bytes": len(first),
        "sha256": actual,
        "regular_file": True,
        "symlink": False,
    }
    if expected_sha256 is not None:
        expected = _require_sha256(expected_sha256, f"{label} expected SHA-256")
        if actual != expected:
            raise GgufReleaseV6Error(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
        record.update({"expected_sha256": expected, "sha256_match": True})
    return resolved, first, record


def _load_json(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    resolved, payload, record = _stable_regular_file(
        path,
        label=label,
        expected_sha256=expected_sha256,
    )
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GgufReleaseV6Error(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise GgufReleaseV6Error(f"{label} JSON root must be an object")
    return resolved, value, record


def _verify_optional_self_digest(value: Mapping[str, Any], label: str) -> None:
    for field in ("receipt_payload_sha256", "canonical_digest_sha256"):
        if field not in value:
            continue
        claimed = _require_sha256(value.get(field), f"{label}.{field}")
        body = dict(value)
        del body[field]
        actual = sha256_bytes(canonical_json(body).encode("utf-8"))
        if claimed != actual:
            raise GgufReleaseV6Error(f"{label}.{field} self-digest mismatch")
        return


def _resolve_tree(path: Path, *, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise GgufReleaseV6Error(f"{label} must not be a symlink: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise GgufReleaseV6Error(f"{label} is unavailable: {raw}") from exc
    if not resolved.is_dir():
        raise GgufReleaseV6Error(f"{label} must be a directory: {resolved}")
    return resolved


def tree_inventory(
    path: Path,
    *,
    label: str,
    selected_names: frozenset[str] | None = None,
) -> dict[str, Any]:
    root = _resolve_tree(path, label=label)
    candidates = list(root.rglob("*"))
    candidates.sort(
        key=lambda item: (
            item.relative_to(root).as_posix().casefold(),
            item.relative_to(root).as_posix(),
        )
    )
    files: list[dict[str, Any]] = []
    casefold_paths: set[str] = set()
    for candidate in candidates:
        if candidate.is_symlink():
            raise GgufReleaseV6Error(f"{label} contains a forbidden symlink: {candidate}")
        if candidate.is_dir():
            continue
        try:
            mode = candidate.stat(follow_symlinks=False).st_mode
        except OSError as exc:
            raise GgufReleaseV6Error(f"{label} entry cannot be inspected: {candidate}") from exc
        if not stat.S_ISREG(mode):
            raise GgufReleaseV6Error(f"{label} contains a non-regular entry: {candidate}")
        if selected_names is not None and candidate.name not in selected_names:
            continue
        relative = candidate.relative_to(root).as_posix()
        folded = relative.casefold()
        if folded in casefold_paths:
            raise GgufReleaseV6Error(
                f"{label} contains Windows-ambiguous paths"
            )
        casefold_paths.add(folded)
        before = candidate.stat()
        digest = sha256_file(candidate)
        after = candidate.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise GgufReleaseV6Error(f"{label} changed while hashing: {candidate}")
        files.append(
            {
                "path": relative,
                "bytes": after.st_size,
                "sha256": digest,
            }
        )
    if not files:
        raise GgufReleaseV6Error(f"{label} is empty: {root}")
    return {
        "path": str(root),
        "files": files,
        "file_count": len(files),
        "bytes": sum(int(item["bytes"]) for item in files),
        "tree_sha256": sha256_bytes(canonical_json(files).encode("utf-8")),
        "ordering": "windows_casefold_then_posix",
    }


def adapter_inventory(path: Path, *, label: str) -> dict[str, Any]:
    inventory = tree_inventory(
        path,
        label=label,
        selected_names=frozenset(
            {
                "adapter_config.json",
                "adapter_model.safetensors",
                "adapter_model.bin",
            }
        ),
    )
    names = [Path(str(item["path"])).name for item in inventory["files"]]
    if (
        len(names) != 2
        or "adapter_config.json" not in names
        or sum(
            name in {"adapter_model.safetensors", "adapter_model.bin"}
            for name in names
        )
        != 1
    ):
        raise GgufReleaseV6Error(
            f"{label} must contain exactly adapter_config.json and one adapter model"
        )
    return inventory


def _selected_adapter_record(selection: Mapping[str, Any]) -> Mapping[str, Any]:
    container = selection.get("selection")
    if isinstance(container, Mapping):
        for key in ("adapter", "selected_adapter"):
            candidate = container.get(key)
            if isinstance(candidate, Mapping):
                return candidate
    candidate = selection.get("selected_adapter")
    if isinstance(candidate, Mapping):
        return candidate
    raise GgufReleaseV6Error("selection freeze must contain a selected_adapter inventory")


def _validate_receipt_chain(
    inputs: ReleaseInputs,
    *,
    base_inventory: Mapping[str, Any],
    checkpoint_inventory: Mapping[str, Any],
    adapter_inventory: Mapping[str, Any],
    fixture_mode: bool,
) -> dict[str, Any]:
    if (
        inputs.blind_qualification_receipt is None
        or inputs.blind_qualification_receipt_sha256 is None
        or inputs.dataset_dir is None
    ):
        raise GgufReleaseV6Error("real blind qualification receipt and frozen dataset are required")
    _, selection, selection_file = _load_json(
        inputs.selection_freeze,
        label="selection freeze",
        expected_sha256=inputs.selection_freeze_sha256,
    )
    _, calibration, calibration_file = _load_json(
        inputs.calibration_receipt,
        label="calibration receipt",
        expected_sha256=inputs.calibration_receipt_sha256,
    )
    _, blind, blind_file = _load_json(
        inputs.blind_receipt,
        label="blind receipt",
        expected_sha256=inputs.blind_receipt_sha256,
    )
    _, qualification, qualification_file = _load_json(
        inputs.blind_qualification_receipt,
        label="blind release qualification receipt",
        expected_sha256=inputs.blind_qualification_receipt_sha256,
    )
    _verify_optional_self_digest(selection, "selection freeze")
    _verify_optional_self_digest(calibration, "calibration receipt")
    _verify_optional_self_digest(blind, "blind receipt")
    if (
        qualification.get("schema") != BLIND_QUALIFICATION_SCHEMA
        or qualification.get("status") != BLIND_QUALIFICATION_PASS_STATUS
        or qualification.get("qualified") is not True
    ):
        raise GgufReleaseV6Error("blind run completed but did not qualify GGUF release")
    authoritative_qualification: dict[str, Any] | None = None
    if not fixture_mode:
        try:
            authoritative_qualification = (
                blind_protocol_v6.verify_release_qualification_v6(
                    blind_receipt_path=inputs.blind_receipt,
                    blind_receipt_sha256=inputs.blind_receipt_sha256,
                    qualification_receipt_path=inputs.blind_qualification_receipt,
                    qualification_receipt_sha256=(
                        inputs.blind_qualification_receipt_sha256
                    ),
                )
            )
        except (
            blind_protocol_v6.BlindProtocolV6Error,
            OSError,
            ValueError,
        ) as exc:
            raise GgufReleaseV6Error(
                "blind release qualification failed independent per-sample verification"
            ) from exc

    if selection.get("schema") != SELECTION_FREEZE_SCHEMA:
        raise GgufReleaseV6Error("selection freeze schema is not final v6")
    if selection.get("status") != SELECTION_FREEZE_STATUS:
        raise GgufReleaseV6Error("selection freeze status is not final")
    selected = _selected_adapter_record(selection)
    selection_container = selection.get("selection")
    selected_checkpoint = (
        selection_container.get("checkpoint")
        if isinstance(selection_container, Mapping)
        else None
    )
    if not isinstance(selected_checkpoint, Mapping):
        raise GgufReleaseV6Error(
            "selection freeze has no full selected checkpoint inventory"
        )
    selected_checkpoint_tree = _require_sha256(
        selected_checkpoint.get("tree_sha256"),
        "selection.checkpoint.tree_sha256",
    )
    if selected_checkpoint_tree != checkpoint_inventory.get("tree_sha256"):
        raise GgufReleaseV6Error(
            "selected checkpoint tree does not match the selection freeze"
        )
    selected_tree = _require_sha256(
        selected.get("tree_sha256"),
        "selection.selected_adapter.tree_sha256",
    )
    if selected_tree != adapter_inventory.get("tree_sha256"):
        raise GgufReleaseV6Error("selected adapter tree does not match the selection freeze")
    selected_base = selection.get("base_model")
    if not isinstance(selected_base, Mapping) or selected_base.get(
        "training_tree_sha256"
    ) != base_inventory.get("tree_sha256"):
        raise GgufReleaseV6Error("base model tree does not match the selection freeze")
    selection_dataset = selection.get("dataset")
    selection_manifest = selection_dataset.get("manifest") if isinstance(selection_dataset, Mapping) else None
    if not isinstance(selection_manifest, Mapping):
        raise GgufReleaseV6Error("selection freeze has no dataset manifest binding")
    dataset_manifest_sha = _require_sha256(
        selection_manifest.get("sha256"),
        "selection.dataset.manifest.sha256",
    )

    if calibration.get("schema") != CALIBRATION_RECEIPT_SCHEMA:
        raise GgufReleaseV6Error("calibration receipt schema is not final v6")
    if calibration.get("status") != CALIBRATION_PASS_STATUS:
        raise GgufReleaseV6Error("calibration receipt did not pass")
    calibration_selection = calibration.get("selection_freeze")
    if (
        not isinstance(calibration_selection, Mapping)
        or calibration_selection.get("sha256") != selection_file["sha256"]
    ):
        raise GgufReleaseV6Error("calibration receipt does not bind the selection freeze")
    if calibration_selection.get("adapter_tree_sha256") != selected_tree:
        raise GgufReleaseV6Error("calibration receipt does not bind the selected adapter")
    if (
        calibration_selection.get("checkpoint_tree_sha256")
        != selected_checkpoint_tree
    ):
        raise GgufReleaseV6Error(
            "calibration receipt does not bind the selected checkpoint"
        )
    calibration_dataset = calibration.get("dataset")
    if (
        not isinstance(calibration_dataset, Mapping)
        or calibration_dataset.get("opened_split") != "calibration"
    ):
        raise GgufReleaseV6Error("calibration receipt must identify the calibration split")
    if (
        calibration_dataset.get("complete_split") is not True
        or calibration_dataset.get("rows") != blind_protocol_v6.EXPECTED_BLIND_EXAMPLES
        or calibration_dataset.get("blind_data_accessed") is not False
    ):
        raise GgufReleaseV6Error("calibration receipt must cover the complete split")
    calibration_execution = calibration.get("execution")
    if (
        not isinstance(calibration_execution, Mapping)
        or calibration_execution.get("checkpoint_reselection_performed") is not False
        or calibration_execution.get("blind_data_accessed") is not False
    ):
        raise GgufReleaseV6Error("calibration receipt changed selection or accessed blind data")

    if blind.get("schema") != BLIND_RECEIPT_SCHEMA:
        raise GgufReleaseV6Error("blind receipt schema is not final v6")
    if blind.get("status") != BLIND_PASS_STATUS:
        raise GgufReleaseV6Error("blind receipt did not pass")
    blind_backend = blind.get("backend")
    if not isinstance(blind_backend, Mapping) or blind_backend.get("mode") != "hf_model":
        raise GgufReleaseV6Error("only a model-bound HF blind run can qualify GGUF release")
    if blind.get("examples") != blind_protocol_v6.EXPECTED_BLIND_EXAMPLES:
        raise GgufReleaseV6Error("blind receipt is not a complete run")
    blind_dataset = blind.get("dataset")
    if (
        not isinstance(blind_dataset, Mapping)
        or blind_dataset.get("rows_read_once") != blind_protocol_v6.EXPECTED_BLIND_EXAMPLES
        or blind_dataset.get("manifest_sha256") != dataset_manifest_sha
    ):
        raise GgufReleaseV6Error("blind receipt dataset binding is incomplete")
    blind_model = blind.get("model")
    if (
        not isinstance(blind_model, Mapping)
        or blind_model.get("base_model_tree_sha256") != base_inventory.get("tree_sha256")
        or blind_model.get("checkpoint_tree_sha256") != selected_checkpoint_tree
        or blind_model.get("adapter_tree_sha256") != selected_tree
    ):
        raise GgufReleaseV6Error("blind receipt does not bind the frozen model")
    blind_gates = blind.get("gates")
    if not isinstance(blind_gates, Mapping):
        raise GgufReleaseV6Error("blind receipt has no upstream gates")
    blind_selection = blind_gates.get("selection_freeze")
    blind_calibration = blind_gates.get("calibration")
    blind_ablation = blind_gates.get("ablation")
    if not isinstance(blind_selection, Mapping) or blind_selection.get("sha256") != selection_file["sha256"]:
        raise GgufReleaseV6Error("blind receipt does not bind the selection freeze")
    if (
        not isinstance(blind_calibration, Mapping)
        or blind_calibration.get("sha256") != calibration_file["sha256"]
    ):
        raise GgufReleaseV6Error("blind receipt does not bind the calibration receipt")
    if not isinstance(blind_ablation, Mapping):
        raise GgufReleaseV6Error("blind receipt does not bind the ablation receipt")
    ablation_path = Path(str(blind_ablation.get("path")))
    ablation_sha = _require_sha256(
        blind_ablation.get("sha256"),
        "blind.gates.ablation.sha256",
    )
    _, _, ablation_file = _stable_regular_file(
        ablation_path,
        label="blind-bound ablation receipt",
        expected_sha256=ablation_sha,
    )

    blind_authorization = blind.get("authorization")
    blind_claim = blind.get("consumption_claim")
    if not isinstance(blind_authorization, Mapping) or not isinstance(blind_claim, Mapping):
        raise GgufReleaseV6Error("blind receipt must include authorization and one-time claim")
    claim_path = Path(str(blind_claim.get("path")))
    claim_sha = _require_sha256(
        blind_claim.get("sha256"),
        "blind.consumption_claim.sha256",
    )
    _, _, claim_file = _stable_regular_file(
        claim_path,
        label="blind consumption claim",
        expected_sha256=claim_sha,
    )

    if qualification.get("schema") != BLIND_QUALIFICATION_SCHEMA:
        raise GgufReleaseV6Error("blind release qualification schema is not final v6")
    if (
        qualification.get("status") != BLIND_QUALIFICATION_PASS_STATUS
        or qualification.get("qualified") is not True
    ):
        raise GgufReleaseV6Error("blind run completed but did not qualify GGUF release")
    if qualification.get("thresholds") != blind_protocol_v6.RELEASE_QUALIFICATION_POLICY:
        raise GgufReleaseV6Error("blind qualification thresholds differ from authorization")
    gate_results = qualification.get("gate_results")
    if (
        not isinstance(gate_results, list)
        or not gate_results
        or any(not isinstance(gate, Mapping) or gate.get("passed") is not True for gate in gate_results)
    ):
        raise GgufReleaseV6Error("blind qualification contains a failed or invalid gate")
    claimed_digest = _require_sha256(
        qualification.get("canonical_digest_sha256"),
        "blind qualification canonical digest",
    )
    qualification_body = dict(qualification)
    del qualification_body["canonical_digest_sha256"]
    if sha256_bytes(canonical_json(qualification_body).encode("utf-8")) != claimed_digest:
        raise GgufReleaseV6Error("blind qualification canonical digest is invalid")
    qualification_run = qualification.get("blind_run_receipt")
    qualification_upstream = qualification.get("upstream")
    qualification_claim = qualification.get("consumption_claim")
    if (
        not isinstance(qualification_run, Mapping)
        or qualification_run.get("sha256") != blind_file["sha256"]
        or qualification_run.get("schema") != BLIND_RECEIPT_SCHEMA
        or qualification_run.get("status") != BLIND_PASS_STATUS
        or not isinstance(qualification_upstream, Mapping)
        or not isinstance(qualification_claim, Mapping)
    ):
        raise GgufReleaseV6Error("blind qualification does not bind the real blind run")
    expected_upstream = {
        "selection_freeze_sha256": selection_file["sha256"],
        "calibration_receipt_sha256": calibration_file["sha256"],
        "ablation_receipt_sha256": ablation_sha,
        "dataset_manifest_sha256": dataset_manifest_sha,
        "blind_sha256": blind_dataset.get("blind_sha256"),
        "base_model_tree_sha256": base_inventory.get("tree_sha256"),
        "checkpoint_tree_sha256": selected_checkpoint_tree,
        "adapter_tree_sha256": selected_tree,
    }
    if dict(qualification_upstream) != expected_upstream:
        raise GgufReleaseV6Error("blind qualification upstream hashes do not match")
    if (
        qualification_claim.get("sha256") != claim_sha
        or qualification_claim.get("nonce_sha256") != blind_claim.get("nonce_sha256")
        or qualification_claim.get("failure_is_non_reusable") is not True
    ):
        raise GgufReleaseV6Error("blind qualification one-time claim binding is invalid")

    authorization = qualification.get("authorization")
    if not isinstance(authorization, Mapping):
        raise GgufReleaseV6Error("blind qualification has no authorization binding")
    if authorization.get("sha256") != blind_authorization.get("sha256") or authorization.get(
        "authorization_id"
    ) != blind_authorization.get("authorization_id"):
        raise GgufReleaseV6Error("blind qualification authorization binding differs from run")
    authorization_path = Path(str(authorization.get("path")))
    authorization_sha = _require_sha256(
        authorization.get("sha256"),
        "blind qualification authorization SHA-256",
    )
    _, authorization_receipt, authorization_file = _load_json(
        authorization_path,
        label="blind authorization",
        expected_sha256=authorization_sha,
    )
    if (
        authorization_receipt.get("schema") != blind_protocol_v6.AUTHORIZATION_SCHEMA
        or authorization_receipt.get("status") != blind_protocol_v6.AUTHORIZATION_STATUS
        or authorization_receipt.get("authorization_id") != authorization.get("authorization_id")
        or authorization_receipt.get("release_qualification_policy")
        != blind_protocol_v6.RELEASE_QUALIFICATION_POLICY
        or authorization.get("policy_sha256")
        != sha256_bytes(canonical_json(blind_protocol_v6.RELEASE_QUALIFICATION_POLICY).encode("utf-8"))
    ):
        raise GgufReleaseV6Error("blind qualification thresholds were not frozen by authorization")

    release_authorization = qualification.get("release_authorization")
    if not isinstance(release_authorization, Mapping):
        raise GgufReleaseV6Error("blind qualification has no release authorization")
    if release_authorization.get("gguf_release_authorized") is not True:
        raise GgufReleaseV6Error("GGUF release is not authorized")
    for field in (
        "activation_authorized",
        "deployment_authorized",
        "production_integration_authorized",
    ):
        if release_authorization.get(field) is not False:
            raise GgufReleaseV6Error(f"blind qualification must preserve {field}=false")

    blind_artifacts = blind.get("artifacts")
    qualification_artifacts = qualification.get("artifacts")
    if not isinstance(blind_artifacts, Mapping) or not isinstance(qualification_artifacts, Mapping):
        raise GgufReleaseV6Error("blind run or qualification artifact binding is absent")
    for name in ("sample_results.v6.jsonl", "summary.v6.json"):
        run_record = blind_artifacts.get(name)
        qualification_record = qualification_artifacts.get(name)
        if (
            not isinstance(run_record, Mapping)
            or not isinstance(qualification_record, Mapping)
            or run_record.get("sha256") != qualification_record.get("sha256")
        ):
            raise GgufReleaseV6Error(f"blind qualification artifact mismatch: {name}")
        artifact_path = Path(str(qualification_record.get("path")))
        expected_artifact_sha = _require_sha256(
            qualification_record.get("sha256"),
            f"blind qualification {name} SHA-256",
        )
        _stable_regular_file(
            artifact_path,
            label=f"blind qualification {name}",
            expected_sha256=expected_artifact_sha,
        )

    return {
        "selection_freeze": selection_file,
        "calibration_receipt": calibration_file,
        "blind_receipt": blind_file,
        "blind_release_qualification": qualification_file,
        "authoritative_blind_release_qualification": (
            authoritative_qualification
        ),
        "fixture_chain_only": fixture_mode,
        "blind_authorization": authorization_file,
        "blind_consumption_claim": claim_file,
        "ablation_receipt": ablation_file,
        "dataset_manifest_sha256": dataset_manifest_sha,
        "selected_adapter_tree_sha256": selected_tree,
        "selected_checkpoint_tree_sha256": selected_checkpoint_tree,
        "chain_verified": True,
        "training_invoked": False,
        "selection_invoked": False,
    }


def _parse_pointer_strict(value: str, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, GgufReleaseV6Error) as exc:
        raise GgufReleaseV6Error(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(parsed, dict) or set(parsed) != set(POINTER_KEYS):
        raise GgufReleaseV6Error(f"{label} must contain exactly task, decision, span_id")
    task = _require_string(parsed.get("task"), f"{label}.task")
    decision = parsed.get("decision")
    span_id = parsed.get("span_id")
    if decision not in {"ANSWER", "REFUSE"}:
        raise GgufReleaseV6Error(f"{label}.decision must be ANSWER or REFUSE")
    if decision == "ANSWER":
        if not isinstance(span_id, str) or not span_id:
            raise GgufReleaseV6Error(f"{label}.span_id must be non-empty for ANSWER")
    elif span_id is not None:
        raise GgufReleaseV6Error(f"{label}.span_id must be null for REFUSE")
    return {"task": task, "decision": decision, "span_id": span_id}


def _derive_validation_golden_set(
    *,
    dataset_dir: Path,
    selection_freeze_sha256: str,
    dataset_manifest_sha256: str,
) -> tuple[tuple[GoldenRecord, ...], bytes, dict[str, Any]]:
    """Derive one deterministic row per domain/task/decision validation stratum."""

    raw_root = Path(dataset_dir)
    if any("blind" in part.casefold() for part in raw_root.parts):
        raise GgufReleaseV6Error("golden source dataset must not be blind-labelled")
    if raw_root.is_symlink():
        raise GgufReleaseV6Error("golden source dataset must not be a symlink")
    root = raw_root.resolve(strict=True)
    manifest_path = root / "manifest.v6.json"
    _, manifest, manifest_file = _load_json(
        manifest_path,
        label="v6 dataset manifest",
        expected_sha256=dataset_manifest_sha256,
    )
    if (
        manifest.get("schema") != "icmat_evidence_pointer_manifest.v6"
        or manifest.get("status") != "DATASET_BUILT_BLIND_HASH_SEALED"
    ):
        raise GgufReleaseV6Error("golden source manifest is not the frozen v6 dataset")
    splits = manifest.get("splits")
    validation_descriptor = splits.get("validation") if isinstance(splits, Mapping) else None
    if not isinstance(validation_descriptor, Mapping):
        raise GgufReleaseV6Error("dataset manifest has no validation descriptor")
    try:
        selected = pointer_hf_eval_v6.select_dataset(
            dataset_dir=root,
            split="validation",
            max_samples=None,
        )
    except pointer_hf_eval_v6.PointerHFEvalV6Error as exc:
        raise GgufReleaseV6Error("frozen validation split cannot produce a golden set") from exc
    if (
        selected.rows_total != 150
        or validation_descriptor.get("count") != 150
        or validation_descriptor.get("sha256") != selected.split_sha256
        or validation_descriptor.get("bytes") != selected.split_bytes
    ):
        raise GgufReleaseV6Error("golden source validation split differs from its manifest")

    strata: dict[tuple[str, str, str], list[Any]] = {}
    for row in selected.rows:
        domain = row.metadata.get("domain")
        task = row.metadata.get("task")
        decision = row.expected_pointer.get("decision") if isinstance(row.expected_pointer, Mapping) else None
        if not isinstance(domain, str) or not isinstance(task, str) or decision not in {"ANSWER", "REFUSE"}:
            raise GgufReleaseV6Error(f"validation row has incomplete golden stratum: {row.example_id}")
        strata.setdefault((domain, task, decision), []).append(row)
    if len(strata) != 18:
        raise GgufReleaseV6Error("validation golden set requires all 3x3x2 strata")

    chosen = []
    for stratum, candidates in sorted(strata.items()):
        row = min(
            candidates,
            key=lambda item: sha256_bytes(
                (f"{DEFAULT_SEED}\0{stratum[0]}\0{stratum[1]}\0{stratum[2]}\0{item.example_id}").encode()
            ),
        )
        chosen.append((stratum, row))

    normalized: list[GoldenRecord] = []
    serialized_records: list[dict[str, Any]] = []
    for stratum, row in chosen:
        expected = _parse_pointer_strict(
            canonical_json(dict(row.expected_pointer)),
            label=f"validation golden {row.example_id}",
        )
        compilation = compile_pointer(
            prompt=row.compiler_prompt,
            evidence=row.compiler_evidence,
            raw_pointer=expected,
            finish_reason="eos_token",
        )
        if compilation.get("status") != "COMPILED" or compilation.get("fail_closed") is not False:
            raise GgufReleaseV6Error(f"validation golden pointer does not compile: {row.example_id}")
        normalized.append(
            GoldenRecord(
                example_id=row.example_id,
                prompt=json.loads(canonical_json(row.compiler_prompt)),
                evidence=tuple(json.loads(canonical_json(item)) for item in row.compiler_evidence),
                expected_pointer=expected,
                expected_compilation=compilation,
            )
        )
        serialized_records.append(
            {
                "schema": GOLDEN_SAMPLE_SCHEMA,
                "example_id": row.example_id,
                "stratum": {
                    "domain": stratum[0],
                    "task": stratum[1],
                    "decision": stratum[2],
                },
                "prompt": row.compiler_prompt,
                "evidence": row.compiler_evidence,
                "expected_pointer": expected,
            }
        )
    golden = {
        "schema": GOLDEN_SET_SCHEMA,
        "split": "validation",
        "selection_freeze_sha256": selection_freeze_sha256,
        "dataset_manifest_sha256": manifest_file["sha256"],
        "validation_sha256": selected.split_sha256,
        "sampling": {
            "fixed": True,
            "method": "sha256-min-per-domain-task-decision-stratum",
            "seed": DEFAULT_SEED,
            "strata": len(strata),
            "blind_data_accessed": False,
            "used_for_model_selection": False,
        },
        "records": serialized_records,
    }
    payload = _json_bytes(golden)
    if len(payload) > MAX_GOLDEN_BYTES:
        raise GgufReleaseV6Error("derived golden set exceeds the size limit")
    file_record = {
        "schema": GOLDEN_SET_SCHEMA,
        "path": DEFAULT_GOLDEN_NAME,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "split": "validation",
        "samples": len(normalized),
        "selection_freeze_sha256": selection_freeze_sha256,
        "dataset_manifest_sha256": manifest_file["sha256"],
        "validation_sha256": selected.split_sha256,
        "generated_by_gguf_tool": True,
        "blind_data_accessed": False,
    }
    return tuple(normalized), payload, file_record


def _tool_record(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    resolved, _, record = _stable_regular_file(
        path,
        label=label,
        expected_sha256=expected_sha256,
    )
    record["version_identity"] = f"sha256:{record['sha256']}"
    receipt_candidates = (
        resolved.parent.parent / "toolchain_receipt.v1.json",
        resolved.parent / "toolchain_receipt.v1.json",
        resolved.parent.parent / "source_receipt.v1.json",
        resolved.parent / "source_receipt.v1.json",
    )
    for receipt_path in receipt_candidates:
        if not receipt_path.is_file():
            continue
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(receipt, Mapping):
            record["toolchain_receipt"] = {
                "path": str(receipt_path.resolve()),
                "sha256": sha256_file(receipt_path),
                "tool": receipt.get("tool"),
                "release_tag": receipt.get("release_tag"),
                "commit": receipt.get("commit"),
            }
            if receipt.get("release_tag"):
                record["version_identity"] = str(receipt["release_tag"])
            break
    return record


def _dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _source_inventory() -> dict[str, Any]:
    paths = {
        "module": Path(__file__).resolve(),
        "cli": WORKSPACE_ROOT / "tools" / "build_icmat_llm_gguf_release_v6.py",
    }
    records: dict[str, Any] = {}
    for role, path in paths.items():
        _, _, record = _stable_regular_file(path, label=f"source {role}")
        records[role] = record
    return records


def preflight_release_v6(
    inputs: ReleaseInputs,
    *,
    runtime_config: RuntimeConfig | None = None,
    fixture_harness: FixtureExecutionHarness | None = None,
) -> dict[str, Any]:
    """Validate immutable release authorization without executing ML tools."""

    runtime_config = runtime_config or RuntimeConfig()
    if (
        runtime_config.threads <= 0
        or runtime_config.threads > 16
        or runtime_config.context_size != MAX_INPUT_TOKENS
        or runtime_config.max_output_tokens != MAX_OUTPUT_TOKENS
        or runtime_config.seed != DEFAULT_SEED
    ):
        raise GgufReleaseV6Error(
            "runtime config must preserve threads 1..16 and fixed 1536/64/20260729 parity limits"
        )
    base_inventory = tree_inventory(
        inputs.base_model_dir,
        label="base model",
    )
    checkpoint_inventory = tree_inventory(
        inputs.selected_adapter_dir,
        label="selected checkpoint",
    )
    selected_adapter_inventory = adapter_inventory(
        inputs.selected_adapter_dir,
        label="selected adapter",
    )
    base_config_path = Path(inputs.base_model_dir) / "config.json"
    adapter_config_path = Path(inputs.selected_adapter_dir) / "adapter_config.json"
    if not base_config_path.is_file() or not adapter_config_path.is_file():
        raise GgufReleaseV6Error("base and selected adapter must contain their config JSON files")
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(adapter_config, Mapping)
        or str(adapter_config.get("peft_type", "")).upper() != "LORA"
        or str(adapter_config.get("task_type", "")).upper() != "CAUSAL_LM"
    ):
        raise GgufReleaseV6Error("selected adapter must be a causal-LM LoRA adapter")

    selection_expected = _require_sha256(
        inputs.selection_freeze_sha256,
        "selection freeze expected SHA-256",
    )
    receipt_chain = _validate_receipt_chain(
        inputs,
        base_inventory=base_inventory,
        checkpoint_inventory=checkpoint_inventory,
        adapter_inventory=selected_adapter_inventory,
        fixture_mode=fixture_harness is not None,
    )
    golden_records, _, golden_file = _derive_validation_golden_set(
        dataset_dir=inputs.dataset_dir,
        selection_freeze_sha256=selection_expected,
        dataset_manifest_sha256=receipt_chain["dataset_manifest_sha256"],
    )

    blockers: list[dict[str, str]] = []
    tools: dict[str, Any] = {}
    for role, path, expected in (
        ("converter", inputs.converter, inputs.converter_sha256),
        ("quantizer", inputs.quantizer, inputs.quantizer_sha256),
        ("llama_server", inputs.llama_server, inputs.llama_server_sha256),
        ("python", inputs.python_executable, sha256_file(Path(inputs.python_executable))),
    ):
        try:
            tools[role] = _tool_record(
                Path(path),
                expected_sha256=expected,
                label=role,
            )
        except GgufReleaseV6Error as exc:
            if role == "python":
                raise
            blockers.append({"code": f"{role.upper()}_UNAVAILABLE", "detail": str(exc)})

    probe = (
        fixture_harness.dependency_probe
        if fixture_harness is not None
        else _dependency_available
    )
    dependencies = {}
    for name in ("torch", "transformers", "peft", "psutil"):
        available = bool(probe(name))
        dependencies[name] = {"available": available}
        if not available:
            blockers.append(
                {
                    "code": f"PYTHON_DEPENDENCY_{name.upper()}_UNAVAILABLE",
                    "detail": f"local Python dependency is unavailable: {name}",
                }
            )
    if Path(inputs.converter).suffix.casefold() != ".py":
        raise GgufReleaseV6Error("converter must be a Python source file")

    status = PREFLIGHT_PASS_STATUS if not blockers else PREFLIGHT_BLOCKED_STATUS
    fingerprint = sha256_bytes(
        canonical_json(
            {
                "base_tree_sha256": base_inventory["tree_sha256"],
                "checkpoint_tree_sha256": checkpoint_inventory["tree_sha256"],
                "adapter_tree_sha256": selected_adapter_inventory["tree_sha256"],
                "receipts": receipt_chain,
                "golden_set_sha256": golden_file["sha256"],
                "tools": tools,
                "runtime_config": runtime_config.__dict__,
                "source": _source_inventory(),
            }
        ).encode("utf-8")
    )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "builder_version": RELEASE_BUILDER_VERSION,
        "created_at": utc_now(),
        "status": status,
        "execution_ready": not blockers,
        "read_only": True,
        "network_used": False,
        "base_model": base_inventory,
        "selected_checkpoint": checkpoint_inventory,
        "selected_adapter": selected_adapter_inventory,
        "receipt_chain": receipt_chain,
        "golden_set": golden_file,
        "golden_samples_validated": len(golden_records),
        "tools": tools,
        "python_dependencies": dependencies,
        "runtime_config": {
            **runtime_config.__dict__,
            "generation": "singleton_greedy",
            "device": "cpu",
            "gpu_layers": 0,
        },
        "planned_outputs": {
            "merged_hf": DEFAULT_MERGED_HF_NAME,
            "gguf_f16": DEFAULT_F16_NAME,
            "gguf_q4_k_m": DEFAULT_Q4_NAME,
            "validation_golden_set": DEFAULT_GOLDEN_NAME,
        },
        "blockers": blockers,
        "input_fingerprint_sha256": fingerprint,
        "release_policy": {
            "overwrite_allowed": False,
            "activation_default": "DISABLED",
            "service_registration": "FORBIDDEN",
            "training": "FORBIDDEN",
            "selection": "FORBIDDEN",
            "fixture_execution": fixture_harness is not None,
            "release_quality_evidence": fixture_harness is None,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _default_merge_hook(
    base_model: Path,
    adapter: Path,
    output: Path,
) -> Mapping[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
    )
    peft = PeftModel.from_pretrained(
        base,
        str(adapter),
        local_files_only=True,
        is_trainable=False,
    )
    merged = peft.merge_and_unload(safe_merge=True)
    merged.save_pretrained(
        str(output),
        safe_serialization=True,
        max_shard_size="2GB",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model),
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.save_pretrained(str(output))
    return {
        "implementation": "transformers+peft",
        "operation": "merge_and_unload",
        "safe_merge": True,
        "device": "local_pc_cpu",
        "local_files_only": True,
        "trust_remote_code": False,
    }


def _default_command_runner(command: Sequence[str], cwd: Path) -> Any:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.casefold()
        not in {
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "ftp_proxy",
        }
    }
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    return subprocess.run(
        [str(value) for value in command],
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        check=False,
        env=environment,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _stream_record(value: Any) -> dict[str, Any]:
    text = "" if value is None else str(value)
    payload = text.encode("utf-8", errors="replace")
    return {
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "tail": text[-4000:],
    }


def _run_checked(
    runner: CommandRunner,
    command: Sequence[str],
    *,
    cwd: Path,
    role: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = runner(tuple(str(value) for value in command), cwd)
    returncode = int(getattr(result, "returncode", -1))
    record = {
        "role": role,
        "command": [str(value) for value in command],
        "cwd": str(cwd),
        "shell": False,
        "returncode": returncode,
        "wall_seconds": time.perf_counter() - started,
        "stdout": _stream_record(getattr(result, "stdout", "")),
        "stderr": _stream_record(getattr(result, "stderr", "")),
    }
    if returncode != 0:
        raise GgufReleaseV6Error(f"{role} failed with exit code {returncode}: {record['stderr']['tail']}")
    return record


class _RssSampler:
    def __init__(self, pid: int) -> None:
        try:
            import psutil
        except ImportError as exc:
            raise GgufReleaseV6Error("psutil is required for PC memory measurement") from exc
        self._process = psutil.Process(pid)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak = 0

    def _sample(self) -> None:
        while not self._stop.wait(0.01):
            try:
                self.peak = max(
                    self.peak,
                    int(self._process.memory_info().rss),
                )
            except Exception:
                return

    def __enter__(self) -> _RssSampler:
        try:
            self.peak = int(self._process.memory_info().rss)
        except Exception:
            self.peak = 0
        self._thread = threading.Thread(
            target=self._sample,
            name="icmat-rss-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        try:
            self.peak = max(
                self.peak,
                int(self._process.memory_info().rss),
            )
        except Exception:
            pass


def _host_record() -> dict[str, Any]:
    return {
        "scope": "LOCAL_PC_CPU_ONLY_NOT_RDK_X5",
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "pid": os.getpid(),
        "rdk_x5": False,
        "bpu": False,
    }


def _default_hf_runner(
    records: Sequence[GoldenRecord],
    merged_hf: Path,
    config: RuntimeConfig,
) -> tuple[Mapping[str, RuntimeObservation], Mapping[str, Any]]:
    from icmat_foundry.llm.pointer_hf_eval_v6 import (
        GenerationRequestV6,
        generate_hf_model,
    )

    requests = []
    for record in records:
        messages = record.prompt.get("messages")
        assert isinstance(messages, Sequence)
        requests.append(
            GenerationRequestV6(
                example_id=record.example_id,
                messages=(
                    dict(messages[0]),
                    dict(messages[1]),
                ),
            )
        )
    with _RssSampler(os.getpid()) as sampler:
        generated, backend = generate_hf_model(
            requests,
            base_model_dir=merged_hf,
            adapter_dir=None,
            device="cpu",
            seed=config.seed,
        )
    observations = {
        example_id: RuntimeObservation(
            raw_pointer=result.raw_pointer,
            finish_reason=result.finish_reason,
            latency_ms=result.latency_ms,
            peak_rss_bytes=sampler.peak,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            generation_error=result.generation_error,
        )
        for example_id, result in generated.items()
    }
    return observations, {
        "backend": backend,
        "host": _host_record(),
        "process_peak_rss_bytes": sampler.peak,
        "measurement": "local PC process RSS sampled at 10 ms",
        "device": "cpu",
        "bpu_used": False,
    }


def _default_gguf_runner(
    records: Sequence[GoldenRecord],
    q4_gguf: Path,
    config: RuntimeConfig,
    *,
    llama_server: Path,
) -> tuple[Mapping[str, RuntimeObservation], Mapping[str, Any]]:
    from icmat_foundry.llm.llama_cpp_eval_v5 import (
        LocalLlamaServer,
        ServerLaunchSpec,
        _extract_generation,
    )

    log_root = Path(tempfile.mkdtemp(prefix="icmat-v6-llama-server-logs-"))
    session = LocalLlamaServer(
        ServerLaunchSpec(
            executable=llama_server.resolve(strict=True),
            model=q4_gguf.resolve(strict=True),
            runtime_dir=llama_server.resolve(strict=True).parent,
            log_dir=log_root / "server",
            threads=config.threads,
            context_size=config.context_size,
            gpu_layers=0,
            startup_timeout_seconds=config.startup_timeout_seconds,
            request_timeout_seconds=config.request_timeout_seconds,
            seed=config.seed,
        )
    )
    observations: dict[str, RuntimeObservation] = {}
    sampler: _RssSampler | None = None
    trace: Mapping[str, Any] = {}
    try:
        session.start()
        process = getattr(session, "_process", None)
        if process is None:
            raise GgufReleaseV6Error("llama-server process was not created")
        sampler = _RssSampler(int(process.pid))
        sampler.__enter__()
        for record in records:
            started = time.perf_counter()
            response = session.chat(
                {
                    "messages": [dict(message) for message in record.prompt["messages"]],
                    "temperature": 0,
                    "max_tokens": config.max_output_tokens,
                    "seed": config.seed,
                    "stream": False,
                }
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            raw, response_trace = _extract_generation(response)
            usage = response_trace.get("usage")
            observations[record.example_id] = RuntimeObservation(
                raw_pointer=raw,
                finish_reason=str(response_trace.get("finish_reason") or "abnormal_end"),
                latency_ms=latency_ms,
                peak_rss_bytes=sampler.peak,
                input_tokens=(
                    int(usage["prompt_tokens"])
                    if isinstance(usage, Mapping) and isinstance(usage.get("prompt_tokens"), int)
                    else None
                ),
                output_tokens=(
                    int(usage["completion_tokens"])
                    if isinstance(usage, Mapping) and isinstance(usage.get("completion_tokens"), int)
                    else None
                ),
            )
    finally:
        if sampler is not None:
            sampler.__exit__(None, None, None)
        session.close()
        trace = session.trace_metadata()
    peak = 0 if sampler is None else sampler.peak
    for example_id, observation in list(observations.items()):
        observations[example_id] = RuntimeObservation(
            **{
                **observation.__dict__,
                "peak_rss_bytes": max(observation.peak_rss_bytes, peak),
            }
        )
    return observations, {
        "host": _host_record(),
        "server": dict(trace),
        "process_peak_rss_bytes": peak,
        "measurement": "local PC llama-server RSS sampled at 10 ms",
        "device": "cpu",
        "gpu_layers": 0,
        "bpu_used": False,
    }


def _runtime_observation(
    values: Mapping[str, RuntimeObservation],
    example_id: str,
    label: str,
) -> RuntimeObservation:
    value = values.get(example_id)
    if not isinstance(value, RuntimeObservation):
        raise GgufReleaseV6Error(f"{label} omitted runtime observation for {example_id}")
    if (
        not isinstance(value.raw_pointer, str)
        or not isinstance(value.finish_reason, str)
        or not value.finish_reason
        or value.latency_ms < 0
        or value.peak_rss_bytes < 0
    ):
        raise GgufReleaseV6Error(f"{label} returned invalid runtime observation for {example_id}")
    return value


def compare_runtime_parity(
    records: Sequence[GoldenRecord],
    *,
    hf_observations: Mapping[str, RuntimeObservation],
    gguf_observations: Mapping[str, RuntimeObservation],
    hf_metadata: Mapping[str, Any],
    gguf_metadata: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply strict raw-pointer, expected-pointer, and compiler parity gates."""

    rows: list[dict[str, Any]] = []
    for record in records:
        hf = _runtime_observation(
            hf_observations,
            record.example_id,
            "HF runtime",
        )
        gguf = _runtime_observation(
            gguf_observations,
            record.example_id,
            "GGUF runtime",
        )
        hf_pointer: dict[str, Any] | None = None
        gguf_pointer: dict[str, Any] | None = None
        hf_structure_error: str | None = None
        gguf_structure_error: str | None = None
        try:
            hf_pointer = _parse_pointer_strict(
                hf.raw_pointer,
                label=f"HF {record.example_id}",
            )
        except GgufReleaseV6Error as exc:
            hf_structure_error = str(exc)
        try:
            gguf_pointer = _parse_pointer_strict(
                gguf.raw_pointer,
                label=f"GGUF {record.example_id}",
            )
        except GgufReleaseV6Error as exc:
            gguf_structure_error = str(exc)

        hf_compilation = compile_pointer(
            prompt=record.prompt,
            evidence=record.evidence,
            raw_pointer=hf.raw_pointer,
            finish_reason=hf.finish_reason,
        )
        gguf_compilation = compile_pointer(
            prompt=record.prompt,
            evidence=record.evidence,
            raw_pointer=gguf.raw_pointer,
            finish_reason=gguf.finish_reason,
        )
        hf_compiled = (
            hf_compilation.get("status") == "COMPILED" and hf_compilation.get("fail_closed") is False
        )
        gguf_compiled = (
            gguf_compilation.get("status") == "COMPILED" and gguf_compilation.get("fail_closed") is False
        )
        pointer_backend_exact = (
            hf_pointer is not None and gguf_pointer is not None and hf_pointer == gguf_pointer
        )
        hf_expected = hf_pointer == record.expected_pointer
        gguf_expected = gguf_pointer == record.expected_pointer
        compiled_backend_exact = (
            hf_compiled
            and gguf_compiled
            and hf_compilation.get("compiler_decision") == gguf_compilation.get("compiler_decision")
            and hf_compilation.get("selected_span_id") == gguf_compilation.get("selected_span_id")
            and hf_compilation.get("compiled_answer") == gguf_compilation.get("compiled_answer")
        )
        expected_answer = record.expected_compilation.get("compiled_answer")
        hf_compiled_expected = hf_compiled and hf_compilation.get("compiled_answer") == expected_answer
        gguf_compiled_expected = gguf_compiled and gguf_compilation.get("compiled_answer") == expected_answer
        normal_stops = (
            hf.finish_reason in TRUSTED_FINISH_REASONS and gguf.finish_reason in TRUSTED_FINISH_REASONS
        )
        strict_pass = all(
            (
                hf_structure_error is None,
                gguf_structure_error is None,
                pointer_backend_exact,
                hf_expected,
                gguf_expected,
                hf_compiled,
                gguf_compiled,
                compiled_backend_exact,
                hf_compiled_expected,
                gguf_compiled_expected,
                normal_stops,
                hf.generation_error is None,
                gguf.generation_error is None,
            )
        )
        rows.append(
            {
                "schema": PARITY_SAMPLE_SCHEMA,
                "example_id": record.example_id,
                "expected_pointer": record.expected_pointer,
                "hf": {
                    **hf.__dict__,
                    "raw_pointer_sha256": sha256_bytes(hf.raw_pointer.encode("utf-8")),
                    "parsed_pointer": hf_pointer,
                    "structure_error": hf_structure_error,
                    "compilation": hf_compilation,
                },
                "gguf": {
                    **gguf.__dict__,
                    "raw_pointer_sha256": sha256_bytes(gguf.raw_pointer.encode("utf-8")),
                    "parsed_pointer": gguf_pointer,
                    "structure_error": gguf_structure_error,
                    "compilation": gguf_compilation,
                },
                "gates": {
                    "hf_structure_valid": hf_structure_error is None,
                    "gguf_structure_valid": gguf_structure_error is None,
                    "task_decision_span_backend_exact": pointer_backend_exact,
                    "hf_expected_pointer_exact": hf_expected,
                    "gguf_expected_pointer_exact": gguf_expected,
                    "hf_compiler_pass": hf_compiled,
                    "gguf_compiler_pass": gguf_compiled,
                    "compiler_output_backend_exact": compiled_backend_exact,
                    "hf_compiler_expected_exact": hf_compiled_expected,
                    "gguf_compiler_expected_exact": gguf_compiled_expected,
                    "trusted_finish_reasons": normal_stops,
                    "strict_pass": strict_pass,
                },
            }
        )

    passed = sum(bool(row["gates"]["strict_pass"]) for row in rows)
    latencies_hf = [float(row["hf"]["latency_ms"]) for row in rows]
    latencies_gguf = [float(row["gguf"]["latency_ms"]) for row in rows]
    report = {
        "schema": PARITY_REPORT_SCHEMA,
        "builder_version": RELEASE_BUILDER_VERSION,
        "created_at": utc_now(),
        "status": (PARITY_PASS_STATUS if passed == len(rows) else PARITY_FAIL_STATUS),
        "strict_gate_pass": passed == len(rows),
        "samples": len(rows),
        "samples_passed": passed,
        "samples_failed": len(rows) - passed,
        "required": {
            "strict_json_keys": ["task", "decision", "span_id"],
            "task_decision_span_backend_exact": "100%",
            "expected_pointer_exact": "100% on both backends",
            "compiler_pass": "100% on both backends",
            "compiler_output_backend_exact": "100%",
            "compiler_expected_exact": "100% on both backends",
            "trusted_finish_reason": "100% on both backends",
        },
        "runtime": {
            "hf": dict(hf_metadata),
            "gguf_q4_k_m": dict(gguf_metadata),
            "measurements": {
                "hf_latency_ms_mean": sum(latencies_hf) / len(latencies_hf),
                "hf_latency_ms_max": max(latencies_hf),
                "gguf_latency_ms_mean": sum(latencies_gguf) / len(latencies_gguf),
                "gguf_latency_ms_max": max(latencies_gguf),
                "hf_peak_rss_bytes": max(int(row["hf"]["peak_rss_bytes"]) for row in rows),
                "gguf_peak_rss_bytes": max(int(row["gguf"]["peak_rss_bytes"]) for row in rows),
                "measurement_scope": "LOCAL_PC_CPU_ONLY_NOT_RDK_X5",
            },
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    return rows, report


def _output_paths(output_dir: Path) -> tuple[Path, Path]:
    raw = Path(output_dir)
    if raw.name in {"", ".", ".."}:
        raise GgufReleaseV6Error("output must name a new directory")
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise GgufReleaseV6Error("output parent must already exist") from exc
    if not parent.is_dir():
        raise GgufReleaseV6Error("output parent must be a directory")
    final = parent / raw.name
    if os.path.lexists(final):
        raise FileExistsError(final)
    return parent, final


def _artifact_record(
    path: Path,
    *,
    kind: str,
    package_root: Path | None = None,
) -> dict[str, Any]:
    resolved, _, record = _stable_regular_file(path, label=kind)
    if package_root is None:
        recorded_path = str(resolved)
    else:
        root = package_root.resolve(strict=True)
        try:
            recorded_path = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise GgufReleaseV6Error(f"{kind} escapes the release package") from exc
    record.update({"kind": kind, "path": recorded_path})
    return record


def _failure_publish(
    *,
    parent: Path,
    final_name: str,
    run_id: str,
    stage: Path | None,
    active_stage: str,
    error: Exception,
) -> Path:
    rejected = parent / f".{final_name}.rejected-{run_id}-{uuid.uuid4().hex[:8]}"
    failure = {
        "schema": FAILURE_RECEIPT_SCHEMA,
        "builder_version": RELEASE_BUILDER_VERSION,
        "created_at": utc_now(),
        "status": "REJECTED_NOT_RELEASED_NOT_ACTIVATED",
        "run_id": run_id,
        "active_stage": active_stage,
        "error_type": type(error).__name__,
        "error": str(error)[:8000],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    if stage is None or not stage.exists():
        stage = parent / f".{final_name}.failure-{uuid.uuid4().hex}"
        stage.mkdir()
    _write_bytes_atomic(
        stage / "failure_receipt.v6.json",
        _json_bytes(failure),
    )
    os.replace(stage, rejected)
    return rejected


def build_gguf_release_v6(
    *,
    inputs: ReleaseInputs,
    output_dir: Path,
    runtime_config: RuntimeConfig | None = None,
    fixture_harness: FixtureExecutionHarness | None = None,
) -> dict[str, Any]:
    """Build and atomically publish a disabled PC CPU GGUF release."""

    runtime_config = runtime_config or RuntimeConfig()
    parent, final = _output_paths(output_dir)
    preflight = preflight_release_v6(
        inputs,
        runtime_config=runtime_config,
        fixture_harness=fixture_harness,
    )
    if preflight["status"] != PREFLIGHT_PASS_STATUS:
        blocker_text = "; ".join(str(item["detail"]) for item in preflight["blockers"])
        raise GgufReleaseV6Error(f"preflight blocked execution: {blocker_text}")
    source = _source_inventory()
    run_id = (
        "icmat-gguf-v6-"
        + sha256_bytes(
            canonical_json(
                {
                    "input": preflight["input_fingerprint_sha256"],
                    "source": source,
                    "output_name": final.name,
                }
            ).encode("utf-8")
        )[:20]
    )
    stage: Path | None = None
    active_stage = "staging"
    started = time.perf_counter()
    try:
        stage = parent / f".{final.name}.tmp-{run_id}-{uuid.uuid4().hex}"
        stage.mkdir()
        _write_bytes_atomic(
            stage / DEFAULT_PREFLIGHT_NAME,
            _json_bytes(preflight),
        )

        active_stage = "merge_selected_adapter"
        merged_hf = stage / DEFAULT_MERGED_HF_NAME
        merged_hf.mkdir()
        merge = (
            fixture_harness.merge_hook
            if fixture_harness is not None
            else _default_merge_hook
        )
        merge_metadata = dict(
            merge(
                Path(inputs.base_model_dir).resolve(strict=True),
                Path(inputs.selected_adapter_dir).resolve(strict=True),
                merged_hf,
            )
            or {}
        )
        merged_inventory = tree_inventory(
            merged_hf,
            label="merged HF candidate",
        )
        merged_inventory["path"] = DEFAULT_MERGED_HF_NAME
        if not (merged_hf / "config.json").is_file():
            raise GgufReleaseV6Error("merged HF candidate has no config.json")
        if not any(item["path"].endswith((".safetensors", ".bin")) for item in merged_inventory["files"]):
            raise GgufReleaseV6Error("merged HF candidate has no model weights")

        runner = (
            fixture_harness.command_runner
            if fixture_harness is not None
            else _default_command_runner
        )
        active_stage = "convert_f16"
        f16 = stage / DEFAULT_F16_NAME
        convert_command = [
            str(Path(inputs.python_executable).resolve(strict=True)),
            str(Path(inputs.converter).resolve(strict=True)),
            str(merged_hf),
            "--outfile",
            str(f16),
            "--outtype",
            "f16",
        ]
        convert_execution = _run_checked(
            runner,
            convert_command,
            cwd=Path(inputs.converter).resolve(strict=True).parent,
            role="llama_cpp_convert_hf_to_gguf_f16",
        )
        f16_record = _artifact_record(
            f16,
            kind="GGUF_F16",
            package_root=stage,
        )
        if f16_record["bytes"] <= 0:
            raise GgufReleaseV6Error("F16 GGUF is empty")

        active_stage = "quantize_q4_k_m"
        q4 = stage / DEFAULT_Q4_NAME
        quantize_command = [
            str(Path(inputs.quantizer).resolve(strict=True)),
            str(f16),
            str(q4),
            "Q4_K_M",
        ]
        quantize_execution = _run_checked(
            runner,
            quantize_command,
            cwd=Path(inputs.quantizer).resolve(strict=True).parent,
            role="llama_cpp_quantize_q4_k_m",
        )
        q4_record = _artifact_record(
            q4,
            kind="GGUF_Q4_K_M",
            package_root=stage,
        )
        if q4_record["bytes"] <= 0:
            raise GgufReleaseV6Error("Q4_K_M GGUF is empty")

        active_stage = "derive_fixed_validation_golden_set"
        golden_records, golden_payload, golden_file = _derive_validation_golden_set(
            dataset_dir=inputs.dataset_dir,
            selection_freeze_sha256=inputs.selection_freeze_sha256,
            dataset_manifest_sha256=preflight["receipt_chain"]["dataset_manifest_sha256"],
        )
        if golden_file["sha256"] != preflight["golden_set"]["sha256"]:
            raise GgufReleaseV6Error("derived validation golden set changed after preflight")
        _write_bytes_atomic(
            stage / DEFAULT_GOLDEN_NAME,
            golden_payload,
        )

        active_stage = "hf_pc_cpu_golden"
        selected_hf_runner = (
            fixture_harness.hf_runner
            if fixture_harness is not None
            else _default_hf_runner
        )
        hf_observations, hf_metadata = selected_hf_runner(
            golden_records,
            merged_hf,
            runtime_config,
        )

        active_stage = "gguf_pc_cpu_golden"
        if fixture_harness is None:
            gguf_observations, gguf_metadata = _default_gguf_runner(
                golden_records,
                q4,
                runtime_config,
                llama_server=Path(inputs.llama_server),
            )
        else:
            gguf_observations, gguf_metadata = fixture_harness.gguf_runner(
                golden_records,
                q4,
                runtime_config,
            )

        active_stage = "strict_parity_gate"
        parity_rows, parity_report = compare_runtime_parity(
            golden_records,
            hf_observations=hf_observations,
            gguf_observations=gguf_observations,
            hf_metadata=hf_metadata,
            gguf_metadata=gguf_metadata,
        )
        _write_bytes_atomic(
            stage / DEFAULT_PARITY_ROWS_NAME,
            _jsonl_bytes(parity_rows),
        )
        _write_bytes_atomic(
            stage / DEFAULT_PARITY_REPORT_NAME,
            _json_bytes(parity_report),
        )
        if parity_report["status"] != PARITY_PASS_STATUS:
            raise GgufReleaseV6Error("strict HF/GGUF pointer and compiler parity gate failed")

        active_stage = "disabled_release_manifest"
        disabled_text = (
            "ICMat v6 GGUF release candidate\n"
            "ACTIVATION=DISABLED\n"
            "This package is not registered as a service and must not replace "
            "any frozen runtime.\n"
            "Measurements are local PC CPU measurements, not RDK X5 or BPU "
            "measurements.\n"
        ).encode("ascii")
        _write_bytes_atomic(stage / DEFAULT_DISABLED_MARKER, disabled_text)
        artifacts = {
            "merged_hf": merged_inventory,
            "gguf_f16": f16_record,
            "gguf_q4_k_m": q4_record,
            "golden_set": _artifact_record(
                stage / DEFAULT_GOLDEN_NAME,
                kind="DETERMINISTIC_NONBLIND_VALIDATION_GOLDEN_SET",
                package_root=stage,
            ),
            "parity_samples": _artifact_record(
                stage / DEFAULT_PARITY_ROWS_NAME,
                kind="JSONL_PARITY_SAMPLES",
                package_root=stage,
            ),
            "parity_report": _artifact_record(
                stage / DEFAULT_PARITY_REPORT_NAME,
                kind="JSON_PARITY_REPORT",
                package_root=stage,
            ),
            "activation_disabled": _artifact_record(
                stage / DEFAULT_DISABLED_MARKER,
                kind="ACTIVATION_POLICY",
                package_root=stage,
            ),
        }
        receipt_body = {
            "schema": RELEASE_RECEIPT_SCHEMA,
            "builder_version": RELEASE_BUILDER_VERSION,
            "created_at": utc_now(),
            "status": (
                FIXTURE_PASS_STATUS
                if fixture_harness is not None
                else RELEASE_PASS_STATUS
            ),
            "run_id": run_id,
            "elapsed_seconds": time.perf_counter() - started,
            "activated": False,
            "deployable_by_this_receipt": False,
            "release_quality_evidence": fixture_harness is None,
            "fixture_execution": fixture_harness is not None,
            "service_registered": False,
            "overwrite_used": False,
            "training_invoked": False,
            "selection_invoked": False,
            "preflight_sha256": sha256_file(stage / DEFAULT_PREFLIGHT_NAME),
            "receipt_chain": preflight["receipt_chain"],
            "source": source,
            "tools": preflight["tools"],
            "commands": {
                "merge": merge_metadata,
                "convert_f16": convert_execution,
                "quantize_q4_k_m": quantize_execution,
            },
            "artifacts": artifacts,
            "parity": {
                "status": parity_report["status"],
                "report_sha256": artifacts["parity_report"]["sha256"],
                "samples_sha256": artifacts["parity_samples"]["sha256"],
                "samples": parity_report["samples"],
                "strict_gate_pass": True,
            },
            "pc_cpu_measurements": parity_report["runtime"]["measurements"],
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
        receipt = {
            **receipt_body,
            "receipt_payload_sha256": sha256_bytes(canonical_json(receipt_body).encode("utf-8")),
        }
        _write_bytes_atomic(
            stage / DEFAULT_RECEIPT_NAME,
            _json_bytes(receipt),
        )
        active_stage = "atomic_publish_disabled"
        if os.path.lexists(final):
            raise FileExistsError(final)
        os.replace(stage, final)
        stage = None
        return {
            "output_dir": str(final),
            "status": receipt["status"],
            "run_id": run_id,
            "receipt": receipt,
            "receipt_sha256": sha256_file(final / DEFAULT_RECEIPT_NAME),
            "activated": False,
            "bpu_used": False,
        }
    except Exception as exc:
        rejected = _failure_publish(
            parent=parent,
            final_name=final.name,
            run_id=run_id,
            stage=stage,
            active_stage=active_stage,
            error=exc,
        )
        if isinstance(exc, GgufReleaseV6Error):
            raise GgufReleaseV6Error(f"{exc}; rejected evidence: {rejected}") from exc
        raise


__all__ = [
    "BLIND_PASS_STATUS",
    "BLIND_QUALIFICATION_PASS_STATUS",
    "BLIND_QUALIFICATION_SCHEMA",
    "BLIND_RECEIPT_SCHEMA",
    "CALIBRATION_PASS_STATUS",
    "CALIBRATION_RECEIPT_SCHEMA",
    "DEFAULT_CONVERTER",
    "DEFAULT_CONVERTER_SHA256",
    "DEFAULT_LLAMA_SERVER",
    "DEFAULT_LLAMA_SERVER_SHA256",
    "DEFAULT_QUANTIZER",
    "DEFAULT_QUANTIZER_SHA256",
    "FIXTURE_PASS_STATUS",
    "FixtureExecutionHarness",
    "GOLDEN_SAMPLE_SCHEMA",
    "GOLDEN_SET_SCHEMA",
    "GgufReleaseV6Error",
    "PARITY_PASS_STATUS",
    "PREFLIGHT_BLOCKED_STATUS",
    "PREFLIGHT_PASS_STATUS",
    "RELEASE_PASS_STATUS",
    "ReleaseInputs",
    "RuntimeConfig",
    "RuntimeObservation",
    "SELECTION_FREEZE_SCHEMA",
    "SELECTION_FREEZE_STATUS",
    "build_gguf_release_v6",
    "canonical_json",
    "compare_runtime_parity",
    "preflight_release_v6",
    "sha256_bytes",
    "sha256_file",
    "tree_inventory",
]
