"""Fail-closed runtime qualification for the v8c3 infrastructure replay.

The qualification is deliberately separate from training.  It imports the
frozen local runtime, loads the pinned Qwen model in NF4, attaches the frozen
LoRA adapter shape, performs one finite synthetic forward, and exits without
constructing an optimizer or reading any dataset.  The canonical receipt is
created with ``O_EXCL`` only after every check has passed.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION_PATH = (
    WORKSPACE_ROOT
    / "docs"
    / "ai_brain_finals_20260728"
    / "ICMAT_POINTER_V8C3_INFRA_RECOVERY_PREREGISTRATION.json"
)
PREREGISTRATION_SHA256 = (
    "3c17a761b45fe14e4a5b48cb4eeb223a2d81b6dfad043a8060c0ca1ce7c06076"
)
CANONICAL_QUALIFICATION_RELATIVE_PATH = (
    "evaluation/icmat_foundry/llm/v8c3.runtime_qualification.v1.json"
)
CANONICAL_QUALIFICATION_PATH = (
    WORKSPACE_ROOT / CANONICAL_QUALIFICATION_RELATIVE_PATH
)
DEFAULT_BASE_MODEL_PATH = (
    WORKSPACE_ROOT
    / "research"
    / "model_assets"
    / "icmat_foundry"
    / "qwen25_05b_instruct"
    / "snapshot"
)

PREREGISTRATION_SCHEMA = (
    "icmat_pointer_v8c3_infra_recovery_preregistration.v1"
)
PROTOCOL_ID = "ICMAT-Pointer-v8c3-INFRA-RECOVERY-EXACT-REPLAY-r1"
QUALIFICATION_SCHEMA = "icmat_runtime_qualification_v8c3.v1"
QUALIFICATION_VERSION = "icmat-runtime-qualification-v8c3.0.0"
QUALIFICATION_STATUS = "PASS_V8C3_RUNTIME_QUALIFIED_ZERO_OPTIMIZER_STEP"
FAIL_STATUS = "FAIL_V8C3_RUNTIME_NOT_QUALIFIED_NO_ATTEMPT_AUTHORIZED"

EXPECTED_DEPENDENCIES = {
    "torch": "2.6.0+cu124",
    "transformers": "4.57.6",
    "tokenizers": "0.22.2",
    "peft": "0.19.1",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.0",
    "safetensors": "0.8.0",
}
EXPECTED_MODEL_CONFIG = {
    "model_type": "qwen2",
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
}
EXPECTED_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
EXPECTED_LORA_RANK = 8
EXPECTED_LORA_ALPHA = 16
EXPECTED_LORA_DROPOUT = 0.1
EXPECTED_HIDDEN_LAYERS = 24
EXPECTED_LINEAR4BIT_MODULES = 168
EXPECTED_TRAINABLE_PARAMETERS = 4_399_104
SYNTHETIC_PROMPT = (
    "ICMat v8c3 runtime qualification probe. Return exactly: QUALIFIED"
)
SYNTHETIC_PROMPT_SHA256 = hashlib.sha256(
    SYNTHETIC_PROMPT.encode("utf-8")
).hexdigest()

_READ_BLOCK_BYTES = 1024 * 1024
_MAX_JSON_BYTES = 4 * 1024 * 1024
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LAYER_NAME = re.compile(r"(?:^|\.)model\.layers\.(\d+)(?:\.|$)")
_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "WANDB_DISABLED": "true",
}
_NETWORK_AUDIT_EVENTS = frozenset(
    {
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
    }
)
_NETWORK_AUDIT_INSTALLED = False
_NETWORK_DENY_DEPTH = 0
_FALSE_AUTHORIZATION = {
    "v8c3_attempt_authorized": False,
    "training_authorized": False,
    "model_selected": False,
    "x5_deployment_authorized": False,
    "bpu_execution_authorized": False,
    "production_integration_authorized": False,
}
_CLAIM_BOUNDARY = (
    "This receipt qualifies only the frozen local v8c3 execution runtime "
    "before an attempt ledger. It uses one fixed synthetic forward and zero "
    "optimizer steps, reads no train, validation, calibration, or blind rows, "
    "and authorizes neither training nor X5, BPU, deployment, selection, or "
    "production claims."
)


class RuntimeQualificationV8C3Error(RuntimeError):
    """Raised when any frozen v8c3 runtime condition is not satisfied."""


@dataclass(frozen=True)
class StableFileSnapshotV8C3:
    path: Path
    sha256: str
    byte_count: int
    identity: tuple[int, int, int, int, int]
    payload: bytes | None = None

    def identity_receipt(self) -> dict[str, int]:
        return {
            "device": self.identity[0],
            "file_id": self.identity[1],
            "size": self.identity[2],
            "mtime_ns": self.identity[3],
            "ctime_ns": self.identity[4],
        }

    def binding(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "bytes": self.byte_count,
            "sha256": self.sha256,
            "stable_identity": self.identity_receipt(),
        }


@dataclass(frozen=True)
class RuntimeBindingsV8C3:
    """Imported runtime state; tests replace this with a deterministic fake."""

    executable: Path
    prefix: Path
    base_prefix: Path
    python_version: str
    versions: Mapping[str, str]
    modules: Mapping[str, ModuleType | Any]
    bitsandbytes_cextension: ModuleType | Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_nonfinite(value: str) -> NoReturn:
    raise RuntimeQualificationV8C3Error(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeQualificationV8C3Error(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeQualificationV8C3Error(
            f"{label} is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeQualificationV8C3Error(f"{label} must be one JSON object")
    return value


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        rendered = json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeQualificationV8C3Error(
            "qualification receipt is not finite canonical JSON"
        ) from exc
    return (rendered + "\n").encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeQualificationV8C3Error(
            "qualification value is not finite canonical JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise RuntimeQualificationV8C3Error(f"{label} fields mismatch")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HEX_SHA256.fullmatch(value) is not None


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path))))


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.fspath(_absolute(left))) == os.path.normcase(
        os.fspath(_absolute(right))
    )


def _is_reparse_point(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    marker = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(marker and attributes & marker)


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _assert_no_link_components(
    path: Path,
    *,
    label: str,
    allow_missing_leaf: bool = False,
) -> None:
    lexical = _absolute(path)
    parts = lexical.parts
    if not parts:
        raise RuntimeQualificationV8C3Error(f"{label} path is empty")
    current = Path(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf and index == len(parts) - 1:
                return
            raise RuntimeQualificationV8C3Error(
                f"{label} path component is missing: {current}"
            ) from None
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse_point(metadata):
            raise RuntimeQualificationV8C3Error(
                f"{label} must not contain a symlink, junction, or reparse point"
            )


def _directory_identity(path: Path, *, label: str) -> tuple[Path, tuple[int, int]]:
    lexical = _absolute(path)
    _assert_no_link_components(lexical, label=label)
    try:
        metadata = os.lstat(lexical)
    except FileNotFoundError as exc:
        raise RuntimeQualificationV8C3Error(f"{label} is missing") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeQualificationV8C3Error(f"{label} is not a directory")
    return lexical.resolve(strict=True), (int(metadata.st_dev), int(metadata.st_ino))


def _recheck_directory_identity(
    path: Path,
    expected: tuple[int, int],
    *,
    label: str,
) -> None:
    metadata = os.lstat(path)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse_point(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
        or (int(metadata.st_dev), int(metadata.st_ino)) != expected
    ):
        raise RuntimeQualificationV8C3Error(f"{label} identity changed")


def _snapshot_file(
    path: Path,
    *,
    label: str,
    capture_payload: bool = False,
    maximum_bytes: int | None = None,
) -> StableFileSnapshotV8C3:
    lexical = _absolute(path)
    _assert_no_link_components(lexical, label=label)
    try:
        before = os.lstat(lexical)
    except FileNotFoundError as exc:
        raise RuntimeQualificationV8C3Error(f"{label} is missing") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or _is_reparse_point(before)
        or not stat.S_ISREG(before.st_mode)
    ):
        raise RuntimeQualificationV8C3Error(
            f"{label} must be a regular non-reparse file"
        )
    if before.st_size < 1 or (
        maximum_bytes is not None and before.st_size > maximum_bytes
    ):
        raise RuntimeQualificationV8C3Error(
            f"{label} byte count is outside the fixed limit"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(lexical), flags)
    digest = hashlib.sha256()
    blocks: list[bytes] = []
    total = 0
    try:
        descriptor_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_before.st_mode)
            or _identity(descriptor_before) != _identity(before)
        ):
            raise RuntimeQualificationV8C3Error(
                f"{label} identity changed before read"
            )
        while True:
            block = os.read(descriptor, _READ_BLOCK_BYTES)
            if not block:
                break
            total += len(block)
            if maximum_bytes is not None and total > maximum_bytes:
                raise RuntimeQualificationV8C3Error(
                    f"{label} exceeded the fixed read limit"
                )
            digest.update(block)
            if capture_payload:
                blocks.append(block)
        descriptor_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(lexical)
    identities = {
        _identity(before),
        _identity(descriptor_before),
        _identity(descriptor_after),
        _identity(after),
    }
    if (
        len(identities) != 1
        or stat.S_ISLNK(after.st_mode)
        or _is_reparse_point(after)
        or not stat.S_ISREG(after.st_mode)
        or total != int(after.st_size)
    ):
        raise RuntimeQualificationV8C3Error(f"{label} changed while read")
    return StableFileSnapshotV8C3(
        path=lexical,
        sha256=digest.hexdigest(),
        byte_count=total,
        identity=_identity(after),
        payload=b"".join(blocks) if capture_payload else None,
    )


def _snapshot_model_tree(root: Path) -> dict[str, Any]:
    model_root, root_identity = _directory_identity(root, label="base model")
    file_records: list[dict[str, Any]] = []
    file_identities: list[dict[str, Any]] = []
    directory_identities: list[dict[str, Any]] = []
    config_payload: bytes | None = None

    def visit(directory: Path) -> None:
        nonlocal config_payload
        metadata = os.lstat(directory)
        entries = sorted(
            list(os.scandir(directory)),
            key=lambda item: (item.name.casefold(), item.name),
        )
        names = [item.name for item in entries]
        if len({name.casefold() for name in names}) != len(names):
            raise RuntimeQualificationV8C3Error(
                "base model contains case-colliding entries"
            )
        relative_directory = (
            "." if directory == model_root else directory.relative_to(model_root).as_posix()
        )
        directory_identities.append(
            {
                "path": relative_directory,
                "identity": list(_identity(metadata)),
                "entries": names,
            }
        )
        for entry in entries:
            child = directory / entry.name
            child_metadata = entry.stat(follow_symlinks=False)
            if (
                entry.is_symlink()
                or stat.S_ISLNK(child_metadata.st_mode)
                or _is_reparse_point(child_metadata)
            ):
                raise RuntimeQualificationV8C3Error(
                    "base model contains a link or reparse point"
                )
            if stat.S_ISDIR(child_metadata.st_mode):
                visit(child)
                continue
            if not stat.S_ISREG(child_metadata.st_mode):
                raise RuntimeQualificationV8C3Error(
                    "base model contains a non-regular entry"
                )
            relative = child.relative_to(model_root).as_posix()
            snapshot = _snapshot_file(
                child,
                label=f"base model file {relative}",
                capture_payload=relative == "config.json",
                maximum_bytes=_MAX_JSON_BYTES if relative == "config.json" else None,
            )
            file_records.append(
                {
                    "path": relative,
                    "bytes": snapshot.byte_count,
                    "sha256": snapshot.sha256,
                }
            )
            file_identities.append(
                {"path": relative, "identity": list(snapshot.identity)}
            )
            if relative == "config.json":
                config_payload = snapshot.payload

    visit(model_root)
    _recheck_directory_identity(model_root, root_identity, label="base model")
    if not file_records or config_payload is None:
        raise RuntimeQualificationV8C3Error(
            "base model is empty or config.json is missing"
        )
    file_records.sort(key=lambda item: (item["path"].casefold(), item["path"]))
    file_identities.sort(key=lambda item: (item["path"].casefold(), item["path"]))
    config = _strict_json(config_payload, label="base model config.json")
    for key, expected in EXPECTED_MODEL_CONFIG.items():
        if config.get(key) != expected:
            raise RuntimeQualificationV8C3Error(
                f"base model Qwen shape mismatch: {key}"
            )
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or "Qwen2ForCausalLM" not in architectures:
        raise RuntimeQualificationV8C3Error(
            "base model architecture is not Qwen2ForCausalLM"
        )
    return {
        "path": str(model_root),
        "tree_sha256": _canonical_sha256(file_records),
        "stable_identity_sha256": _canonical_sha256(
            {
                "directories": directory_identities,
                "files": file_identities,
            }
        ),
        "file_count": len(file_records),
        "bytes": sum(int(item["bytes"]) for item in file_records),
        "config": {**EXPECTED_MODEL_CONFIG, "architecture": "Qwen2ForCausalLM"},
    }


def _load_preregistration() -> tuple[StableFileSnapshotV8C3, dict[str, Any]]:
    snapshot = _snapshot_file(
        PREREGISTRATION_PATH,
        label="v8c3 preregistration",
        capture_payload=True,
        maximum_bytes=_MAX_JSON_BYTES,
    )
    if snapshot.sha256 != PREREGISTRATION_SHA256:
        raise RuntimeQualificationV8C3Error(
            "v8c3 preregistration SHA-256 mismatch"
        )
    preregistration = _strict_json(
        snapshot.payload or b"",
        label="v8c3 preregistration",
    )
    if (
        preregistration.get("schema") != PREREGISTRATION_SCHEMA
        or preregistration.get("protocol_id") != PROTOCOL_ID
        or preregistration.get("status")
        != "FROZEN_BEFORE_V8C3_IMPLEMENTATION_AND_TRAINING"
        or preregistration.get("scientific_change") != "NONE"
        or preregistration.get("recovery_scope")
        != "EXECUTION_INFRASTRUCTURE_ONLY"
    ):
        raise RuntimeQualificationV8C3Error(
            "v8c3 preregistration identity or scope mismatch"
        )
    runtime = preregistration.get("fixed_runtime_contract")
    gate = preregistration.get("runtime_qualification_gate")
    algorithm = preregistration.get("frozen_algorithm")
    frozen_data = preregistration.get("frozen_data")
    canonical = preregistration.get("canonical_artifacts")
    if not all(
        isinstance(value, Mapping)
        for value in (runtime, gate, algorithm, frozen_data, canonical)
    ):
        raise RuntimeQualificationV8C3Error(
            "v8c3 preregistration runtime sections are missing"
        )
    assert isinstance(runtime, Mapping)
    assert isinstance(gate, Mapping)
    assert isinstance(algorithm, Mapping)
    assert isinstance(frozen_data, Mapping)
    assert isinstance(canonical, Mapping)
    if runtime.get("dependencies") != EXPECTED_DEPENDENCIES:
        raise RuntimeQualificationV8C3Error(
            "v8c3 exact dependency contract mismatch"
        )
    if (
        runtime.get("network_allowed") is not False
        or algorithm.get("lora_rank") != EXPECTED_LORA_RANK
        or algorithm.get("lora_alpha") != EXPECTED_LORA_ALPHA
        or float(algorithm.get("lora_dropout", -1.0)) != EXPECTED_LORA_DROPOUT
        or algorithm.get("target_modules") != list(EXPECTED_TARGET_MODULES)
        or gate.get("required_before_attempt_ledger") is not True
        or gate.get("qualification_receipt")
        != CANONICAL_QUALIFICATION_RELATIVE_PATH
        or gate.get("qualification_receipt_create_mode")
        != "O_EXCL_AFTER_ALL_CHECKS_PASS"
        or gate.get("qwen_hidden_layers_expected") != EXPECTED_HIDDEN_LAYERS
        or gate.get("linear4bit_modules_expected")
        != EXPECTED_LINEAR4BIT_MODULES
        or gate.get("lora_trainable_parameters_expected")
        != EXPECTED_TRAINABLE_PARAMETERS
        or gate.get("optimizer_steps_allowed") != 0
        or gate.get("train_or_validation_rows_allowed_in_runtime_probe") != 0
        or gate.get("blind_access_allowed") is not False
        or canonical.get("attempt_ledger")
        != "evaluation/icmat_foundry/llm/v8c3.canary_attempt.v1.json"
        or not _is_sha256(frozen_data.get("base_model_tree_sha256"))
    ):
        raise RuntimeQualificationV8C3Error(
            "v8c3 frozen runtime gate invariants mismatch"
        )
    return snapshot, preregistration


def _load_runtime_bindings() -> RuntimeBindingsV8C3:
    modules = {
        name: importlib.import_module(name) for name in EXPECTED_DEPENDENCIES
    }
    versions = {
        name: importlib.metadata.version(name) for name in EXPECTED_DEPENDENCIES
    }
    cextension = importlib.import_module("bitsandbytes.cextension")
    return RuntimeBindingsV8C3(
        executable=Path(sys.executable),
        prefix=Path(sys.prefix),
        base_prefix=Path(sys.base_prefix),
        python_version=platform.python_version(),
        versions=versions,
        modules=modules,
        bitsandbytes_cextension=cextension,
    )


def _runtime_paths(
    preregistration: Mapping[str, Any],
) -> tuple[Path, Path, Path, tuple[Path, ...]]:
    runtime = preregistration["fixed_runtime_contract"]
    assert isinstance(runtime, Mapping)
    python_relative = Path(str(runtime["python_relative_path"]))
    prefix_relative = Path(str(runtime["workspace_venv_prefix_relative_path"]))
    if python_relative.is_absolute() or prefix_relative.is_absolute():
        raise RuntimeQualificationV8C3Error(
            "workspace runtime paths must remain relative"
        )
    if ".." in python_relative.parts or ".." in prefix_relative.parts:
        raise RuntimeQualificationV8C3Error(
            "workspace runtime paths must not escape the workspace"
        )
    python_path = _absolute(WORKSPACE_ROOT / python_relative)
    prefix = _absolute(WORKSPACE_ROOT / prefix_relative)
    base_prefix = _absolute(Path(str(runtime["base_prefix"])))
    roots: list[Path] = []
    for raw in runtime["allowed_module_roots"]:
        candidate = Path(str(raw))
        if not candidate.is_absolute():
            if ".." in candidate.parts:
                raise RuntimeQualificationV8C3Error(
                    "allowed module root escapes the workspace"
                )
            candidate = WORKSPACE_ROOT / candidate
        root, _ = _directory_identity(candidate, label="allowed module root")
        roots.append(root)
    return python_path, prefix, base_prefix, tuple(roots)


def _path_under(path: Path, roots: tuple[Path, ...]) -> Path | None:
    candidate = os.path.normcase(os.fspath(_absolute(path)))
    for root in roots:
        root_text = os.path.normcase(os.fspath(_absolute(root)))
        try:
            common = os.path.commonpath((candidate, root_text))
        except ValueError:
            continue
        if common == root_text:
            return root
    return None


def _observe_interpreter(
    bindings: RuntimeBindingsV8C3,
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = preregistration["fixed_runtime_contract"]
    assert isinstance(runtime, Mapping)
    expected_python, expected_prefix, expected_base, _ = _runtime_paths(
        preregistration
    )
    if not _same_path(bindings.executable, expected_python):
        raise RuntimeQualificationV8C3Error(
            "qualification must run under the fixed .venv-icmat interpreter"
        )
    if not _same_path(bindings.prefix, expected_prefix):
        raise RuntimeQualificationV8C3Error("sys.prefix is not the fixed venv")
    if not _same_path(bindings.base_prefix, expected_base):
        raise RuntimeQualificationV8C3Error(
            "sys.base_prefix is not the frozen xrd base environment"
        )
    expected_version = str(runtime["python_version_prefix"])
    if bindings.python_version != expected_version:
        raise RuntimeQualificationV8C3Error(
            "Python version does not exactly match the frozen runtime"
        )
    snapshot = _snapshot_file(expected_python, label="fixed Python interpreter")
    if snapshot.sha256 != runtime.get("python_sha256"):
        raise RuntimeQualificationV8C3Error(
            "fixed Python interpreter SHA-256 mismatch"
        )
    return {
        **snapshot.binding(),
        "version": bindings.python_version,
        "prefix": str(expected_prefix),
        "base_prefix": str(expected_base),
    }


def _observe_dependencies(
    bindings: RuntimeBindingsV8C3,
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = preregistration["fixed_runtime_contract"]
    assert isinstance(runtime, Mapping)
    _, _, _, allowed_roots = _runtime_paths(preregistration)
    if set(bindings.versions) != set(EXPECTED_DEPENDENCIES):
        raise RuntimeQualificationV8C3Error(
            "runtime dependency inventory keys mismatch"
        )
    if set(bindings.modules) != set(EXPECTED_DEPENDENCIES):
        raise RuntimeQualificationV8C3Error("runtime module inventory keys mismatch")
    receipts: dict[str, Any] = {}
    for name, expected_version in EXPECTED_DEPENDENCIES.items():
        if bindings.versions.get(name) != expected_version:
            raise RuntimeQualificationV8C3Error(
                f"runtime dependency version mismatch: {name}"
            )
        module_path_raw = getattr(bindings.modules[name], "__file__", None)
        if not isinstance(module_path_raw, (str, os.PathLike)):
            raise RuntimeQualificationV8C3Error(
                f"runtime dependency has no file source: {name}"
            )
        module_path = _absolute(Path(module_path_raw))
        matched_root = _path_under(module_path, allowed_roots)
        if matched_root is None:
            raise RuntimeQualificationV8C3Error(
                f"runtime dependency source is outside allowed roots: {name}"
            )
        source = _snapshot_file(
            module_path,
            label=f"runtime dependency source {name}",
        )
        receipts[name] = {
            "version": expected_version,
            "module_source": source.binding(),
            "allowed_root": str(matched_root),
        }
    return receipts


def _observe_cuda_and_bitsandbytes(
    bindings: RuntimeBindingsV8C3,
    preregistration: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = preregistration["fixed_runtime_contract"]
    assert isinstance(runtime, Mapping)
    torch = bindings.modules["torch"]
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not bool(cuda.is_available()) or int(cuda.device_count()) < 1:
        raise RuntimeQualificationV8C3Error("CUDA device 0 is unavailable")
    torch_cuda = str(getattr(getattr(torch, "version", None), "cuda", ""))
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    cudnn_version = None if cudnn is None else cudnn.version()
    device_name = str(cuda.get_device_name(0))
    capability = tuple(int(value) for value in cuda.get_device_capability(0))
    bf16_supported = bool(cuda.is_bf16_supported())
    free_bytes, total_bytes = cuda.mem_get_info(0)
    free_mib = int(free_bytes) // (1024 * 1024)
    total_mib = int(total_bytes) // (1024 * 1024)
    expected_capability = tuple(int(value) for value in runtime["compute_capability"])
    if (
        torch_cuda != runtime["cuda_version"]
        or cudnn_version != runtime["cudnn"]
        or device_name != runtime["device_name"]
        or capability != expected_capability
        or bf16_supported is not bool(runtime["bf16_required"])
        or free_mib < int(runtime["minimum_free_vram_mib"])
        or total_mib < free_mib
    ):
        raise RuntimeQualificationV8C3Error(
            "CUDA/BF16/device/capability/VRAM contract mismatch"
        )

    _, _, _, allowed_roots = _runtime_paths(preregistration)
    cextension = bindings.bitsandbytes_cextension
    extension_path_raw = getattr(cextension, "__file__", None)
    native_library = getattr(cextension, "lib", None)
    native_path_raw = getattr(native_library, "_name", None)
    if (
        not isinstance(extension_path_raw, (str, os.PathLike))
        or native_library is None
        or getattr(native_library, "compiled_with_cuda", None) is not True
        or not isinstance(native_path_raw, (str, os.PathLike))
    ):
        raise RuntimeQualificationV8C3Error(
            "bitsandbytes CUDA native extension is not loaded"
        )
    extension_path = _absolute(Path(extension_path_raw))
    native_path = _absolute(Path(native_path_raw))
    if (
        _path_under(extension_path, allowed_roots) is None
        or _path_under(native_path, allowed_roots) is None
    ):
        raise RuntimeQualificationV8C3Error(
            "bitsandbytes extension source is outside allowed roots"
        )
    expected_cuda_tag = "cuda" + str(runtime["cuda_version"]).replace(".", "")
    if expected_cuda_tag.casefold() not in native_path.name.casefold():
        raise RuntimeQualificationV8C3Error(
            "bitsandbytes native library does not match frozen CUDA"
        )
    extension_snapshot = _snapshot_file(
        extension_path,
        label="bitsandbytes cextension source",
    )
    native_snapshot = _snapshot_file(
        native_path,
        label="bitsandbytes CUDA native library",
    )
    return (
        {
            "torch_cuda": torch_cuda,
            "cudnn": cudnn_version,
            "device_index": 0,
            "device_count": int(cuda.device_count()),
            "device_name": device_name,
            "compute_capability": list(capability),
            "bf16_supported": bf16_supported,
            "free_vram_mib": free_mib,
            "total_vram_mib": total_mib,
            "minimum_free_vram_mib": int(runtime["minimum_free_vram_mib"]),
        },
        {
            "compiled_with_cuda": True,
            "native_library_class": type(native_library).__name__,
            "cextension_source": extension_snapshot.binding(),
            "native_library": native_snapshot.binding(),
            "cuda_tag": expected_cuda_tag,
        },
    )


def _normalize_task_type(value: Any) -> str:
    observed = getattr(value, "value", value)
    rendered = str(observed)
    if rendered.startswith("TaskType."):
        rendered = rendered.split(".", 1)[1]
    return rendered


def _execute_model_probe(
    bindings: RuntimeBindingsV8C3,
    *,
    model_dir: Path,
    model_tree: Mapping[str, Any],
) -> dict[str, Any]:
    torch = bindings.modules["torch"]
    transformers = bindings.modules["transformers"]
    peft = bindings.modules["peft"]
    bitsandbytes = bindings.modules["bitsandbytes"]
    model = None
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantization = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        torch.cuda.empty_cache()
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
            quantization_config=quantization,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        config_layers = int(getattr(model.config, "num_hidden_layers", -1))
        config_type = str(getattr(model.config, "model_type", ""))
        layer_indices = {
            int(match.group(1))
            for name, _ in model.named_modules()
            if (match := _LAYER_NAME.search(str(name))) is not None
        }
        if (
            config_layers != EXPECTED_HIDDEN_LAYERS
            or config_type != "qwen2"
            or layer_indices != set(range(EXPECTED_HIDDEN_LAYERS))
        ):
            raise RuntimeQualificationV8C3Error(
                "loaded model does not expose exactly 24 Qwen layers"
            )
        linear4bit_type = bitsandbytes.nn.Linear4bit
        linear4bit_count = sum(
            1 for module in model.modules() if isinstance(module, linear4bit_type)
        )
        if linear4bit_count != EXPECTED_LINEAR4BIT_MODULES:
            raise RuntimeQualificationV8C3Error(
                "loaded NF4 model does not expose exactly 168 Linear4bit modules"
            )

        model.config.use_cache = False
        model = peft.prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        requested_lora = peft.LoraConfig(
            r=EXPECTED_LORA_RANK,
            lora_alpha=EXPECTED_LORA_ALPHA,
            lora_dropout=EXPECTED_LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(EXPECTED_TARGET_MODULES),
        )
        model = peft.get_peft_model(model, requested_lora)
        peft_configs = getattr(model, "peft_config", None)
        if not isinstance(peft_configs, Mapping) or len(peft_configs) != 1:
            raise RuntimeQualificationV8C3Error(
                "exactly one attached LoRA adapter configuration is required"
            )
        attached_lora = next(iter(peft_configs.values()))
        attached_targets = tuple(
            sorted(str(value) for value in attached_lora.target_modules)
        )
        if (
            int(attached_lora.r) != EXPECTED_LORA_RANK
            or int(attached_lora.lora_alpha) != EXPECTED_LORA_ALPHA
            or not math.isclose(
                float(attached_lora.lora_dropout),
                EXPECTED_LORA_DROPOUT,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or str(attached_lora.bias) != "none"
            or _normalize_task_type(attached_lora.task_type) != "CAUSAL_LM"
            or attached_targets != tuple(sorted(EXPECTED_TARGET_MODULES))
        ):
            raise RuntimeQualificationV8C3Error(
                "attached LoRA configuration differs from rank8/alpha16/dropout0.1"
            )
        trainable_parameters = sum(
            int(parameter.numel())
            for parameter in model.parameters()
            if bool(parameter.requires_grad)
        )
        if trainable_parameters != EXPECTED_TRAINABLE_PARAMETERS:
            raise RuntimeQualificationV8C3Error(
                "LoRA trainable parameter count mismatch"
            )

        model.eval()
        encoded = tokenizer(
            SYNTHETIC_PROMPT,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=False,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise RuntimeQualificationV8C3Error(
                "synthetic prompt tokenization did not return input_ids"
            )
        input_shape = tuple(int(value) for value in encoded["input_ids"].shape)
        if len(input_shape) != 2 or input_shape[0] != 1 or input_shape[1] < 1:
            raise RuntimeQualificationV8C3Error(
                "synthetic prompt tokenization shape is invalid"
            )
        device_batch = {
            key: value.to(0) for key, value in encoded.items() if hasattr(value, "to")
        }
        with torch.inference_mode():
            output = model(**device_batch, use_cache=False)
        logits = getattr(output, "logits", None)
        if logits is None or int(logits.numel()) < 1:
            raise RuntimeQualificationV8C3Error(
                "synthetic forward produced no logits"
            )
        finite = bool(torch.isfinite(logits).all().item())
        if not finite:
            raise RuntimeQualificationV8C3Error(
                "synthetic NF4+LoRA forward produced non-finite logits"
            )
        return {
            "base_model": dict(model_tree),
            "qwen_hidden_layers_config": config_layers,
            "qwen_hidden_layers_observed": len(layer_indices),
            "qwen_layer_indices_contiguous": True,
            "quantization": {
                "load_in_4bit": True,
                "quant_type": "nf4",
                "double_quant": True,
                "compute_dtype": "bfloat16",
                "linear4bit_modules": linear4bit_count,
            },
            "lora": {
                "rank": EXPECTED_LORA_RANK,
                "alpha": EXPECTED_LORA_ALPHA,
                "dropout": EXPECTED_LORA_DROPOUT,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "target_modules": list(EXPECTED_TARGET_MODULES),
                "trainable_parameters": trainable_parameters,
            },
            "synthetic_forward": {
                "prompt_sha256": SYNTHETIC_PROMPT_SHA256,
                "prompt_source": "FIXED_SYNTHETIC_LITERAL_NOT_DATASET",
                "batch_size": 1,
                "token_count": input_shape[1],
                "forward_calls": 1,
                "logits_finite": True,
            },
            "optimizer_constructed": False,
            "optimizer_steps": 0,
        }
    finally:
        model = None
        gc.collect()
        try:
            bindings.modules["torch"].cuda.empty_cache()
        except Exception:
            pass


def _network_audit_hook(event: str, args: tuple[Any, ...]) -> None:
    del args
    if _NETWORK_DENY_DEPTH > 0 and event in _NETWORK_AUDIT_EVENTS:
        raise RuntimeQualificationV8C3Error(
            f"network operation is forbidden during v8c3 qualification: {event}"
        )


def _install_network_audit_hook() -> None:
    global _NETWORK_AUDIT_INSTALLED
    if not _NETWORK_AUDIT_INSTALLED:
        sys.addaudithook(_network_audit_hook)
        _NETWORK_AUDIT_INSTALLED = True


@contextmanager
def _offline_environment() -> Iterator[dict[str, str]]:
    global _NETWORK_DENY_DEPTH
    _install_network_audit_hook()
    previous = {name: os.environ.get(name) for name in _OFFLINE_ENVIRONMENT}
    os.environ.update(_OFFLINE_ENVIRONMENT)
    _NETWORK_DENY_DEPTH += 1
    try:
        yield dict(_OFFLINE_ENVIRONMENT)
    finally:
        _NETWORK_DENY_DEPTH -= 1
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _receipt_body(
    *,
    preregistration_snapshot: StableFileSnapshotV8C3,
    interpreter: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    cuda: Mapping[str, Any],
    bitsandbytes: Mapping[str, Any],
    model_probe: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": QUALIFICATION_SCHEMA,
        "version": QUALIFICATION_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": QUALIFICATION_STATUS,
        "created_at": _utc_now(),
        "preregistration": preregistration_snapshot.binding(),
        "interpreter": dict(interpreter),
        "dependencies": dict(dependencies),
        "cuda": dict(cuda),
        "bitsandbytes": dict(bitsandbytes),
        "model_probe": dict(model_probe),
        "access_boundary": {
            "synthetic_prompt_only": True,
            "train_rows_read": 0,
            "validation_rows_read": 0,
            "calibration_rows_read": 0,
            "blind_rows_read": 0,
            "dataset_paths_constructed": 0,
            "network_allowed": False,
            "network_used": False,
            "network_guard": "PYTHON_AUDIT_HOOK_DENY_CONNECT_AND_DNS",
            "offline_environment": dict(_OFFLINE_ENVIRONMENT),
        },
        "execution_boundary": {
            "model_loaded": True,
            "nf4_loaded": True,
            "lora_attached": True,
            "forward_calls": 1,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "trainer_constructed": False,
            "training_started": False,
            "attempt_ledger_created": False,
            "receipt_written_after_all_checks": True,
        },
        "authorization": dict(_FALSE_AUTHORIZATION),
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    return {**body, "canonical_payload_sha256": _canonical_sha256(body)}


def _exclusive_write(path: Path, payload: bytes) -> StableFileSnapshotV8C3:
    canonical = _absolute(CANONICAL_QUALIFICATION_PATH)
    target = _absolute(path)
    if not _same_path(target, canonical):
        raise RuntimeQualificationV8C3Error(
            "runtime qualification may only use the canonical v8c3 receipt path"
        )
    parent, parent_identity = _directory_identity(
        target.parent,
        label="v8c3 qualification receipt parent",
    )
    if os.path.lexists(target):
        raise FileExistsError(
            "canonical v8c3 runtime qualification receipt already exists"
        )
    _assert_no_link_components(
        target,
        label="v8c3 qualification receipt",
        allow_missing_leaf=True,
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(os.fspath(target), flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written < 1:
                raise OSError("v8c3 qualification receipt write stalled")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _recheck_directory_identity(
        parent,
        parent_identity,
        label="v8c3 qualification receipt parent",
    )
    snapshot = _snapshot_file(
        target,
        label="v8c3 qualification receipt",
        capture_payload=True,
        maximum_bytes=_MAX_JSON_BYTES,
    )
    if snapshot.payload != payload:
        raise RuntimeQualificationV8C3Error(
            "v8c3 qualification receipt changed after exclusive write"
        )
    return snapshot


def _validate_receipt_payload(
    receipt: Mapping[str, Any],
    *,
    preregistration_snapshot: StableFileSnapshotV8C3,
    preregistration: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema",
        "version",
        "protocol_id",
        "status",
        "created_at",
        "preregistration",
        "interpreter",
        "dependencies",
        "cuda",
        "bitsandbytes",
        "model_probe",
        "access_boundary",
        "execution_boundary",
        "authorization",
        "claim_boundary",
        "canonical_payload_sha256",
    }
    _require_exact_keys(
        receipt,
        expected_keys,
        label="v8c3 qualification receipt",
    )
    body = {key: value for key, value in receipt.items() if key != "canonical_payload_sha256"}
    created_at = receipt.get("created_at")
    try:
        parsed_created_at = datetime.fromisoformat(str(created_at))
    except ValueError as exc:
        raise RuntimeQualificationV8C3Error(
            "v8c3 qualification created_at is invalid"
        ) from exc
    if (
        receipt.get("schema") != QUALIFICATION_SCHEMA
        or receipt.get("version") != QUALIFICATION_VERSION
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("status") != QUALIFICATION_STATUS
        or receipt.get("canonical_payload_sha256") != _canonical_sha256(body)
        or receipt.get("preregistration") != preregistration_snapshot.binding()
        or receipt.get("authorization") != _FALSE_AUTHORIZATION
        or receipt.get("claim_boundary") != _CLAIM_BOUNDARY
        or parsed_created_at.tzinfo is None
    ):
        raise RuntimeQualificationV8C3Error(
            "v8c3 qualification receipt identity or digest mismatch"
        )
    access = receipt.get("access_boundary")
    execution = receipt.get("execution_boundary")
    expected_access = {
        "synthetic_prompt_only": True,
        "train_rows_read": 0,
        "validation_rows_read": 0,
        "calibration_rows_read": 0,
        "blind_rows_read": 0,
        "dataset_paths_constructed": 0,
        "network_allowed": False,
        "network_used": False,
        "network_guard": "PYTHON_AUDIT_HOOK_DENY_CONNECT_AND_DNS",
        "offline_environment": dict(_OFFLINE_ENVIRONMENT),
    }
    expected_execution = {
        "model_loaded": True,
        "nf4_loaded": True,
        "lora_attached": True,
        "forward_calls": 1,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "trainer_constructed": False,
        "training_started": False,
        "attempt_ledger_created": False,
        "receipt_written_after_all_checks": True,
    }
    if access != expected_access or execution != expected_execution:
        raise RuntimeQualificationV8C3Error(
            "v8c3 qualification access or zero-step boundary mismatch"
        )
    runtime = preregistration["fixed_runtime_contract"]
    frozen_data = preregistration["frozen_data"]
    gate = preregistration["runtime_qualification_gate"]
    assert isinstance(runtime, Mapping)
    assert isinstance(frozen_data, Mapping)
    assert isinstance(gate, Mapping)
    interpreter = receipt.get("interpreter")
    dependencies = receipt.get("dependencies")
    cuda = receipt.get("cuda")
    bnb = receipt.get("bitsandbytes")
    probe = receipt.get("model_probe")
    if not all(
        isinstance(value, Mapping)
        for value in (interpreter, dependencies, cuda, bnb, probe)
    ):
        raise RuntimeQualificationV8C3Error(
            "v8c3 qualification runtime evidence sections are missing"
        )
    assert isinstance(interpreter, Mapping)
    assert isinstance(dependencies, Mapping)
    assert isinstance(cuda, Mapping)
    assert isinstance(bnb, Mapping)
    assert isinstance(probe, Mapping)
    _require_exact_keys(
        interpreter,
        {
            "path",
            "bytes",
            "sha256",
            "stable_identity",
            "version",
            "prefix",
            "base_prefix",
        },
        label="v8c3 qualification interpreter",
    )
    _require_exact_keys(
        cuda,
        {
            "torch_cuda",
            "cudnn",
            "device_index",
            "device_count",
            "device_name",
            "compute_capability",
            "bf16_supported",
            "free_vram_mib",
            "total_vram_mib",
            "minimum_free_vram_mib",
        },
        label="v8c3 qualification CUDA",
    )
    _require_exact_keys(
        bnb,
        {
            "compiled_with_cuda",
            "native_library_class",
            "cextension_source",
            "native_library",
            "cuda_tag",
        },
        label="v8c3 qualification bitsandbytes",
    )
    _require_exact_keys(
        probe,
        {
            "base_model",
            "qwen_hidden_layers_config",
            "qwen_hidden_layers_observed",
            "qwen_layer_indices_contiguous",
            "quantization",
            "lora",
            "synthetic_forward",
            "optimizer_constructed",
            "optimizer_steps",
        },
        label="v8c3 qualification model probe",
    )
    expected_python, expected_prefix, expected_base, allowed_roots = _runtime_paths(
        preregistration
    )
    if (
        not _same_path(Path(str(interpreter.get("path"))), expected_python)
        or interpreter.get("sha256") != runtime["python_sha256"]
        or interpreter.get("version") != runtime["python_version_prefix"]
        or not _same_path(Path(str(interpreter.get("prefix"))), expected_prefix)
        or not _same_path(Path(str(interpreter.get("base_prefix"))), expected_base)
        or set(dependencies) != set(EXPECTED_DEPENDENCIES)
    ):
        raise RuntimeQualificationV8C3Error(
            "v8c3 qualification interpreter or dependency binding mismatch"
        )
    for name, version in EXPECTED_DEPENDENCIES.items():
        dependency = dependencies.get(name)
        if not isinstance(dependency, Mapping) or dependency.get("version") != version:
            raise RuntimeQualificationV8C3Error(
                f"v8c3 qualification dependency mismatch: {name}"
            )
        _require_exact_keys(
            dependency,
            {"version", "module_source", "allowed_root"},
            label=f"v8c3 qualification dependency {name}",
        )
        source = dependency.get("module_source")
        if not isinstance(source, Mapping):
            raise RuntimeQualificationV8C3Error(
                f"v8c3 qualification dependency source missing: {name}"
            )
        source_path = Path(str(source.get("path")))
        matched_root = _path_under(source_path, allowed_roots)
        if matched_root is None or not _same_path(
            Path(str(dependency.get("allowed_root"))), matched_root
        ):
            raise RuntimeQualificationV8C3Error(
                f"v8c3 qualification dependency source escaped: {name}"
            )
    expected_capability = [int(value) for value in runtime["compute_capability"]]
    if (
        cuda.get("torch_cuda") != runtime["cuda_version"]
        or cuda.get("cudnn") != runtime["cudnn"]
        or cuda.get("device_index") != 0
        or not isinstance(cuda.get("device_count"), int)
        or cuda["device_count"] < 1
        or cuda.get("device_name") != runtime["device_name"]
        or cuda.get("compute_capability") != expected_capability
        or cuda.get("bf16_supported") is not bool(runtime["bf16_required"])
        or not isinstance(cuda.get("free_vram_mib"), int)
        or cuda["free_vram_mib"] < runtime["minimum_free_vram_mib"]
        or not isinstance(cuda.get("total_vram_mib"), int)
        or cuda["total_vram_mib"] < cuda["free_vram_mib"]
        or cuda.get("minimum_free_vram_mib") != runtime["minimum_free_vram_mib"]
        or bnb.get("compiled_with_cuda") is not True
        or bnb.get("native_library_class") != "CudaBNBNativeLibrary"
        or bnb.get("cuda_tag")
        != "cuda" + str(runtime["cuda_version"]).replace(".", "")
    ):
        raise RuntimeQualificationV8C3Error(
            "v8c3 qualification CUDA or bitsandbytes binding mismatch"
        )
    for label, binding in (
        ("bitsandbytes cextension", bnb.get("cextension_source")),
        ("bitsandbytes native library", bnb.get("native_library")),
    ):
        if not isinstance(binding, Mapping):
            raise RuntimeQualificationV8C3Error(f"{label} binding is missing")
        _require_exact_keys(
            binding,
            {"path", "bytes", "sha256", "stable_identity"},
            label=label,
        )
        if _path_under(Path(str(binding.get("path"))), allowed_roots) is None:
            raise RuntimeQualificationV8C3Error(
                f"{label} source escaped allowed roots"
            )
    base_model = probe.get("base_model")
    quantization = probe.get("quantization")
    lora = probe.get("lora")
    forward = probe.get("synthetic_forward")
    if not all(
        isinstance(value, Mapping)
        for value in (base_model, quantization, lora, forward)
    ):
        raise RuntimeQualificationV8C3Error(
            "v8c3 qualification model probe sections are missing"
        )
    assert isinstance(base_model, Mapping)
    assert isinstance(quantization, Mapping)
    assert isinstance(lora, Mapping)
    assert isinstance(forward, Mapping)
    _require_exact_keys(
        base_model,
        {
            "path",
            "tree_sha256",
            "stable_identity_sha256",
            "file_count",
            "bytes",
            "config",
        },
        label="v8c3 qualification base model",
    )
    _require_exact_keys(
        quantization,
        {
            "load_in_4bit",
            "quant_type",
            "double_quant",
            "compute_dtype",
            "linear4bit_modules",
        },
        label="v8c3 qualification quantization",
    )
    _require_exact_keys(
        lora,
        {
            "rank",
            "alpha",
            "dropout",
            "bias",
            "task_type",
            "target_modules",
            "trainable_parameters",
        },
        label="v8c3 qualification LoRA",
    )
    _require_exact_keys(
        forward,
        {
            "prompt_sha256",
            "prompt_source",
            "batch_size",
            "token_count",
            "forward_calls",
            "logits_finite",
        },
        label="v8c3 qualification synthetic forward",
    )
    if (
        base_model.get("tree_sha256") != frozen_data["base_model_tree_sha256"]
        or probe.get("qwen_hidden_layers_config") != EXPECTED_HIDDEN_LAYERS
        or probe.get("qwen_hidden_layers_observed") != EXPECTED_HIDDEN_LAYERS
        or probe.get("qwen_layer_indices_contiguous") is not True
        or quantization
        != {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
            "linear4bit_modules": EXPECTED_LINEAR4BIT_MODULES,
        }
        or lora
        != {
            "rank": EXPECTED_LORA_RANK,
            "alpha": EXPECTED_LORA_ALPHA,
            "dropout": EXPECTED_LORA_DROPOUT,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": list(EXPECTED_TARGET_MODULES),
            "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        }
        or forward.get("prompt_sha256") != SYNTHETIC_PROMPT_SHA256
        or forward.get("prompt_source")
        != "FIXED_SYNTHETIC_LITERAL_NOT_DATASET"
        or forward.get("batch_size") != 1
        or not isinstance(forward.get("token_count"), int)
        or forward["token_count"] < 1
        or forward.get("forward_calls") != 1
        or forward.get("logits_finite") is not True
        or probe.get("optimizer_constructed") is not False
        or probe.get("optimizer_steps") != gate["optimizer_steps_allowed"]
    ):
        raise RuntimeQualificationV8C3Error(
            "v8c3 qualification NF4, LoRA, or finite-forward evidence mismatch"
        )


def _verify_bound_file(binding: Mapping[str, Any], *, label: str) -> None:
    expected_keys = {"path", "bytes", "sha256", "stable_identity"}
    core = {key: binding.get(key) for key in expected_keys}
    if any(key not in binding for key in expected_keys) or not _is_sha256(
        core.get("sha256")
    ):
        raise RuntimeQualificationV8C3Error(f"{label} binding fields mismatch")
    snapshot = _snapshot_file(Path(str(core["path"])), label=label)
    if snapshot.binding() != core:
        raise RuntimeQualificationV8C3Error(f"{label} binding changed")


def _verify_receipt_bound_files(receipt: Mapping[str, Any]) -> None:
    _verify_bound_file(receipt["interpreter"], label="qualified interpreter")
    for name, dependency in receipt["dependencies"].items():
        _verify_bound_file(
            dependency["module_source"],
            label=f"qualified dependency source {name}",
        )
    _verify_bound_file(
        receipt["bitsandbytes"]["cextension_source"],
        label="qualified bitsandbytes cextension source",
    )
    _verify_bound_file(
        receipt["bitsandbytes"]["native_library"],
        label="qualified bitsandbytes CUDA native library",
    )


def qualify_runtime_v8c3(
    *,
    base_model_dir: Path = DEFAULT_BASE_MODEL_PATH,
) -> dict[str, Any]:
    """Run the zero-step probe and exclusively create the canonical receipt."""

    canonical = _absolute(CANONICAL_QUALIFICATION_PATH)
    if os.path.lexists(canonical):
        raise FileExistsError(
            "canonical v8c3 runtime qualification receipt already exists"
        )
    prereg_snapshot, preregistration = _load_preregistration()
    expected_receipt = _absolute(
        WORKSPACE_ROOT
        / str(
            preregistration["runtime_qualification_gate"][
                "qualification_receipt"
            ]
        )
    )
    if not _same_path(expected_receipt, canonical):
        raise RuntimeQualificationV8C3Error(
            "v8c3 preregistration canonical receipt path mismatch"
        )
    model_tree_before = _snapshot_model_tree(base_model_dir)
    expected_model_hash = preregistration["frozen_data"]["base_model_tree_sha256"]
    if model_tree_before["tree_sha256"] != expected_model_hash:
        raise RuntimeQualificationV8C3Error(
            "base model tree does not match the v8c3 preregistration"
        )
    with _offline_environment():
        bindings = _load_runtime_bindings()
        interpreter = _observe_interpreter(bindings, preregistration)
        dependencies = _observe_dependencies(bindings, preregistration)
        cuda, bitsandbytes = _observe_cuda_and_bitsandbytes(
            bindings, preregistration
        )
        model_probe = _execute_model_probe(
            bindings,
            model_dir=_absolute(base_model_dir),
            model_tree=model_tree_before,
        )
        interpreter_after = _observe_interpreter(bindings, preregistration)
        dependencies_after = _observe_dependencies(bindings, preregistration)
        cuda_after, bitsandbytes_after = _observe_cuda_and_bitsandbytes(
            bindings, preregistration
        )
    if interpreter_after != interpreter or dependencies_after != dependencies:
        raise RuntimeQualificationV8C3Error(
            "interpreter or dependency identity changed during qualification"
        )
    cuda_static_keys = {
        key: value
        for key, value in cuda.items()
        if key not in {"free_vram_mib", "total_vram_mib"}
    }
    cuda_after_static = {
        key: value
        for key, value in cuda_after.items()
        if key not in {"free_vram_mib", "total_vram_mib"}
    }
    if cuda_static_keys != cuda_after_static or bitsandbytes_after != bitsandbytes:
        raise RuntimeQualificationV8C3Error(
            "CUDA or bitsandbytes identity changed during qualification"
        )
    model_tree_after = _snapshot_model_tree(base_model_dir)
    if model_tree_after != model_tree_before:
        raise RuntimeQualificationV8C3Error(
            "base model identity changed during qualification"
        )
    prereg_after, preregistration_after = _load_preregistration()
    if (
        prereg_after.binding() != prereg_snapshot.binding()
        or preregistration_after != preregistration
    ):
        raise RuntimeQualificationV8C3Error(
            "v8c3 preregistration changed during qualification"
        )
    if os.path.lexists(canonical):
        raise FileExistsError(
            "canonical v8c3 runtime qualification receipt appeared during probe"
        )
    receipt = _receipt_body(
        preregistration_snapshot=prereg_snapshot,
        interpreter=interpreter,
        dependencies=dependencies,
        cuda=cuda_after,
        bitsandbytes=bitsandbytes,
        model_probe=model_probe,
    )
    _validate_receipt_payload(
        receipt,
        preregistration_snapshot=prereg_snapshot,
        preregistration=preregistration,
    )
    snapshot = _exclusive_write(canonical, _canonical_json_bytes(receipt))
    return verify_runtime_qualification_v8c3(
        snapshot.path,
        revalidate_current_runtime=False,
    )


def verify_runtime_qualification_v8c3(
    receipt_path: Path = CANONICAL_QUALIFICATION_PATH,
    *,
    revalidate_current_runtime: bool = True,
) -> dict[str, Any]:
    """Verify the canonical receipt and return its stable binding for qlora."""

    canonical = _absolute(CANONICAL_QUALIFICATION_PATH)
    if not _same_path(receipt_path, canonical):
        raise RuntimeQualificationV8C3Error(
            "only the canonical v8c3 runtime qualification receipt is valid"
        )
    snapshot = _snapshot_file(
        canonical,
        label="v8c3 runtime qualification receipt",
        capture_payload=True,
        maximum_bytes=_MAX_JSON_BYTES,
    )
    receipt = _strict_json(
        snapshot.payload or b"",
        label="v8c3 runtime qualification receipt",
    )
    if snapshot.payload != _canonical_json_bytes(receipt):
        raise RuntimeQualificationV8C3Error(
            "v8c3 qualification receipt is not canonical JSON"
        )
    prereg_snapshot, preregistration = _load_preregistration()
    _validate_receipt_payload(
        receipt,
        preregistration_snapshot=prereg_snapshot,
        preregistration=preregistration,
    )
    _verify_receipt_bound_files(receipt)
    current_model_tree = _snapshot_model_tree(
        Path(str(receipt["model_probe"]["base_model"]["path"]))
    )
    if current_model_tree != receipt["model_probe"]["base_model"]:
        raise RuntimeQualificationV8C3Error(
            "qualified base model identity or content changed"
        )
    if revalidate_current_runtime:
        with _offline_environment():
            bindings = _load_runtime_bindings()
            interpreter = _observe_interpreter(bindings, preregistration)
            dependencies = _observe_dependencies(bindings, preregistration)
            cuda, bitsandbytes = _observe_cuda_and_bitsandbytes(
                bindings, preregistration
            )
        if interpreter != receipt["interpreter"]:
            raise RuntimeQualificationV8C3Error(
                "current interpreter differs from qualification receipt"
            )
        if dependencies != receipt["dependencies"]:
            raise RuntimeQualificationV8C3Error(
                "current dependencies differ from qualification receipt"
            )
        recorded_cuda = {
            key: value
            for key, value in receipt["cuda"].items()
            if key not in {"free_vram_mib", "total_vram_mib"}
        }
        current_cuda = {
            key: value
            for key, value in cuda.items()
            if key not in {"free_vram_mib", "total_vram_mib"}
        }
        if recorded_cuda != current_cuda or bitsandbytes != receipt["bitsandbytes"]:
            raise RuntimeQualificationV8C3Error(
                "current CUDA or bitsandbytes runtime differs from receipt"
            )
    return snapshot.binding()


__all__ = [
    "CANONICAL_QUALIFICATION_PATH",
    "DEFAULT_BASE_MODEL_PATH",
    "FAIL_STATUS",
    "QUALIFICATION_STATUS",
    "RuntimeBindingsV8C3",
    "RuntimeQualificationV8C3Error",
    "qualify_runtime_v8c3",
    "verify_runtime_qualification_v8c3",
]
