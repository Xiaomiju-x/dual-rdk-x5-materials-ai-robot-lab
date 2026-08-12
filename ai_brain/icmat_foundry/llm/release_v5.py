"""Deterministic, inactive release builder for the ICMat-Qwen v5 CPU sidecar."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, NoReturn

from icmat_foundry.deploy import ALLOWLIST_SCHEMA, build_package, verify_package
from icmat_foundry.deploy.package_v1 import (
    _input_file_within_root,
    _scan_file,
)
from icmat_foundry.release import build_release_manifest, verify_release_manifest

BUILD_RESULT_SCHEMA = "icmat_llm_release_build_result.v5"
EVALUATION_SUMMARY_SCHEMA = "icmat_llm_runtime_evaluation_summary.v5"
PARITY_SUMMARY_SCHEMA = "icmat_llm_runtime_hf_gguf_parity_summary.v5"
FIXTURE_CONTRACT_SCHEMA = "icmat_llm_runtime_prompt_fixture_contract.v5"
SPEC_SCHEMA = "icmat_candidate_spec.v1"
STAGE = "CPU_RUNTIME_VERIFIED"
CLAIM_STATUS = "LOCAL_CPU_RUNTIME_VERIFIED"
PRODUCT_ID = "ICMat-Qwen-0.5B"

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\)")
_POSIX_HOST_ABSOLUTE = re.compile(
    r"(?<![A-Za-z0-9:])/(?:Users|home|root|mnt|tmp|var|opt|private|Volumes|srv|data|workspace|build)/"
)
_FILE_URI = re.compile(r"(?i)\bfile://")

_EXPECTED_SCHEMAS = {
    "task_contract": "icmat_qwen_task_contract.v5",
    "preprocessing_contract": "icmat_qwen_preprocessing_contract.v5",
    "split_manifest": "icmat_evidence_sft_manifest.v5",
    "evaluation_report": "icmat_evidence_paired_comparison.v5",
    "source_catalog": "icmat.rag.licensed_source_catalog.v2",
    "rag_manifest": "icmat.rag.manifest.v2",
    "rag_audit": "icmat.rag.independent_audit.v1",
    "cpu_runtime_report": "icmat_hf_gguf_task_parity_report.v5",
    "training_receipt": "icmat_qlora_full_run_receipt.v5",
    "selection_freeze": "icmat_llm_selection_freeze.v5",
    "gguf_export_receipt": "icmat_gguf_export_receipt.v5",
    "prompt_fixture": "icmat_qwen_x5_prompt_fixture.v1",
    "prompt_fixture_build_receipt": ("icmat_qwen_x5_prompt_fixture_build_receipt.v1"),
}

_INPUT_ROLES = {
    "q4_gguf": "model_weights",
    "task_contract": "task_contract",
    "preprocessing_contract": "preprocessing_contract",
    "dataset_manifest": "split_manifest",
    "blind_paired_evaluation_report": "evaluation_report",
    "rag_source_catalog": "source_catalog",
    "rag_manifest": "rag_manifest",
    "rag_chunks": "rag_chunks",
    "rag_audit": "rag_audit",
    "hf_gguf_parity_report": "cpu_runtime_report",
    "training_receipt": "training_receipt",
    "selection_freeze": "selection_freeze",
    "export_receipt": "gguf_export_receipt",
    "x5_runtime_runner": "x5_runtime_runner",
    "x5_runtime_cli": "x5_runtime_cli",
    "x5_runtime_root_init": "icmat_python_package_init",
    "x5_runtime_llm_init": "llm_python_package_init",
    "prompt_fixture": "prompt_fixture",
    "prompt_fixture_contract": "prompt_fixture_build_receipt",
}

_PACKAGE_DESTINATIONS = {
    "model_weights": "artifacts/models/icmat-qwen05b-q4_k_m.gguf",
    "x5_runtime_runner": "bin/icmat_foundry/llm/x5_gguf_replay.py",
    "x5_runtime_cli": "bin/tools/x5_icmat_llm_replay.py",
    "icmat_python_package_init": "bin/icmat_foundry/__init__.py",
    "llm_python_package_init": "bin/icmat_foundry/llm/__init__.py",
    "task_contract": "contracts/task_contract.v5.json",
    "preprocessing_contract": "contracts/preprocessing_contract.v5.json",
    "source_catalog": "contracts/rag/licensed_source_catalog.v2.json",
    "rag_manifest": "contracts/rag/manifest.v2.json",
    "rag_chunks": "artifacts/rag/licensed_chunks.v1.jsonl",
    "prompt_fixture": "artifacts/prompt/prompt_fixture.v1.json",
    "prompt_fixture_contract": ("contracts/prompt/runtime_prompt_fixture_contract.v5.json"),
    "runtime_evaluation_summary": ("artifacts/evidence/blind_evaluation_summary.v5.json"),
    "runtime_cpu_parity_summary": ("artifacts/evidence/hf_gguf_parity_summary.v5.json"),
}

_RELEASE_ONLY_ROLES = frozenset(
    {
        "split_manifest",
        "evaluation_report",
        "rag_audit",
        "cpu_runtime_report",
        "training_receipt",
        "selection_freeze",
        "gguf_export_receipt",
        "prompt_fixture_build_receipt",
    }
)


class LlmReleaseV5Error(ValueError):
    """Raised when an LLM release input or output violates the v5 contract."""


@dataclass(frozen=True)
class LlmReleaseInputs:
    """Explicit files required to publish one local CPU-runtime candidate."""

    q4_gguf: Path
    task_contract: Path
    preprocessing_contract: Path
    dataset_manifest: Path
    blind_paired_evaluation_report: Path
    rag_source_catalog: Path
    rag_manifest: Path
    rag_chunks: Path
    rag_audit: Path
    hf_gguf_parity_report: Path
    training_receipt: Path
    selection_freeze: Path
    export_receipt: Path
    x5_runtime_runner: Path
    x5_runtime_cli: Path
    x5_runtime_root_init: Path
    x5_runtime_llm_init: Path
    prompt_fixture: Path
    prompt_fixture_contract: Path


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LlmReleaseV5Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    raise LlmReleaseV5Error(f"non-finite JSON constant is forbidden: {value}")


def _assert_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise LlmReleaseV5Error(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, label=f"{label}[{index}]")


def _load_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LlmReleaseV5Error(f"{role} must contain valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise LlmReleaseV5Error(f"{role} must contain one JSON object")
    _assert_finite(value, label=role)
    return value


def _load_rag_chunks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise LlmReleaseV5Error(f"rag_chunks contains blank row {line_number}")
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_pairs,
                    parse_constant=_reject_nonfinite,
                )
            except json.JSONDecodeError as exc:
                raise LlmReleaseV5Error(f"rag_chunks row {line_number} is invalid JSON") from exc
            if not isinstance(row, dict):
                raise LlmReleaseV5Error(f"rag_chunks row {line_number} must be an object")
            if (
                row.get("schema") != "icmat.rag.chunk.v1"
                or row.get("license_id") != "CC BY 4.0"
                or row.get("evidence_kind") != "literature_knowledge"
            ):
                raise LlmReleaseV5Error(
                    f"rag_chunks row {line_number} violates the licensed literature contract"
                )
            _assert_finite(row, label=f"rag_chunks[{line_number}]")
            rows.append(row)
    if not rows:
        raise LlmReleaseV5Error("rag_chunks must not be empty")
    return rows


def _assert_no_absolute_paths(value: Any, *, label: str) -> None:
    if isinstance(value, str):
        stripped = value.strip()
        if (
            stripped.startswith("/")
            or _WINDOWS_ABSOLUTE.search(value)
            or _POSIX_HOST_ABSOLUTE.search(value)
            or _FILE_URI.search(value)
        ):
            raise LlmReleaseV5Error(f"{label} contains an absolute host path or file URI")
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_no_absolute_paths(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_absolute_paths(item, label=f"{label}[{index}]")


def _strip_path_fields(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = key.casefold()
            if (
                lowered in {"directory", "root", "workspace"}
                or lowered == "path"
                or lowered.endswith("_path")
            ):
                continue
            result[key] = _strip_path_fields(item)
        return result
    if isinstance(value, list):
        return [_strip_path_fields(item) for item in value]
    return value


def _require_schema(
    records: dict[str, dict[str, Any]],
    *,
    role: str,
) -> None:
    expected = _EXPECTED_SCHEMAS[role]
    if records[role].get("schema") != expected:
        raise LlmReleaseV5Error(f"{role} schema must be {expected}")


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise LlmReleaseV5Error(f"{label} must be a lowercase SHA-256")
    return value


def _nested(value: dict[str, Any], *keys: str, label: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise LlmReleaseV5Error(f"{label} is missing {'.'.join(keys)}")
        current = current[key]
    return current


def _validate_contract_bindings(
    *,
    records: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    rag_rows: list[dict[str, Any]],
) -> None:
    for role in _EXPECTED_SCHEMAS:
        _require_schema(records, role=role)

    q4 = paths["model_weights"]
    with q4.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise LlmReleaseV5Error("model_weights is not a GGUF file")
    q4_sha256 = _sha256_file(q4)
    q4_bytes = q4.stat().st_size

    task = records["task_contract"]
    researcher = task.get("researcher_selection")
    if (
        not isinstance(researcher, dict)
        or researcher.get("hidden_task_router") is not False
        or researcher.get("model_and_task_selected_explicitly") is not True
    ):
        raise LlmReleaseV5Error("task_contract must preserve explicit researcher selection")

    preprocessing = records["preprocessing_contract"]
    runtime_policy = preprocessing.get("runtime_policy")
    if (
        not isinstance(runtime_policy, dict)
        or runtime_policy.get("gguf_backend") != "local llama.cpp Q4_K_M free generation"
        or runtime_policy.get("gguf_blind_test_allowed") is not False
    ):
        raise LlmReleaseV5Error("preprocessing_contract does not fix the local CPU GGUF policy")

    dataset = records["split_manifest"]
    if dataset.get("status") != "DATASET_BUILT_BLIND_TEST_SEALED":
        raise LlmReleaseV5Error("split_manifest is not the sealed v5 dataset")

    blind = records["evaluation_report"]
    if (
        blind.get("status") != "BLIND_PROMOTION_PASS"
        or blind.get("split") != "blind_test"
        or blind.get("promotion_context_valid") is not True
        or blind.get("promotion_allowed") is not True
        or blind.get("all_required_gates_pass") is not True
    ):
        raise LlmReleaseV5Error("evaluation_report is not a successful sealed blind comparison")

    source_catalog = records["source_catalog"]
    if source_catalog.get("status") != "LICENSED_FULLTEXT_CANDIDATE_OFFLINE" or source_catalog.get(
        "chunk_count"
    ) != len(rag_rows):
        raise LlmReleaseV5Error("source_catalog does not match the licensed RAG chunks")
    if records["rag_audit"].get("status") != "GO":
        raise LlmReleaseV5Error("rag_audit status must be GO")

    training = records["training_receipt"]
    if training.get("status") != "PASS_FULL_MULTI_SEED_TRAINING_COMPLETED_NOT_DEPLOYED":
        raise LlmReleaseV5Error("training_receipt is not a completed full run")

    selection = records["selection_freeze"]
    if (
        selection.get("status") != "SELECTION_FROZEN_BEFORE_BLIND_NOT_QUALITY_ACCEPTED"
        or selection.get("frozen_before_blind") is not True
        or selection.get("selection_locked") is not True
    ):
        raise LlmReleaseV5Error("selection_freeze is not locked before blind")
    selection_training_sha = _nested(
        selection,
        "training_receipt",
        "sha256",
        label="selection_freeze",
    )
    if selection_training_sha != _sha256_file(paths["training_receipt"]):
        raise LlmReleaseV5Error("selection_freeze does not bind the supplied training_receipt")
    selection_manifest_sha = _nested(
        selection,
        "dataset",
        "manifest",
        "sha256",
        label="selection_freeze",
    )
    if selection_manifest_sha != _sha256_file(paths["split_manifest"]):
        raise LlmReleaseV5Error("selection_freeze does not bind the supplied split_manifest")
    claimed_freeze_digest = _require_sha256(
        selection.get("canonical_digest_sha256"),
        label="selection_freeze.canonical_digest_sha256",
    )
    freeze_body = dict(selection)
    del freeze_body["canonical_digest_sha256"]
    if _sha256_bytes(_canonical_bytes(freeze_body)) != claimed_freeze_digest:
        raise LlmReleaseV5Error("selection_freeze canonical digest mismatch")

    export = records["gguf_export_receipt"]
    if (
        export.get("status") != "PASS_GGUF_EXPORT_COMPLETED_NOT_DEPLOYED"
        or export.get("x5_touched") is not False
        or export.get("autostart_created") is not False
    ):
        raise LlmReleaseV5Error("gguf_export_receipt has an invalid claim state")
    exported_q4 = _nested(
        export,
        "artifacts",
        "gguf_q4_k_m",
        label="gguf_export_receipt",
    )
    if (
        not isinstance(exported_q4, dict)
        or exported_q4.get("sha256") != q4_sha256
        or exported_q4.get("bytes") != q4_bytes
        or exported_q4.get("format") != "GGUF"
        or exported_q4.get("quantization") != "Q4_K_M"
    ):
        raise LlmReleaseV5Error("gguf_export_receipt does not bind the supplied Q4_K_M model")
    selected_adapter_sha = _nested(
        selection,
        "selection",
        "selected_adapter",
        "tree_sha256",
        label="selection_freeze",
    )
    exported_adapter_sha = _nested(
        export,
        "input_snapshot",
        "adapter",
        "tree_sha256",
        label="gguf_export_receipt",
    )
    if selected_adapter_sha != exported_adapter_sha:
        raise LlmReleaseV5Error("GGUF export adapter differs from the frozen selected adapter")

    parity = records["cpu_runtime_report"]
    if (
        parity.get("status") != "HF_GGUF_TASK_PARITY_PASS"
        or _nested(
            parity,
            "non_degradation_gate",
            "all_passed",
            label="cpu_runtime_report",
        )
        is not True
    ):
        raise LlmReleaseV5Error("cpu_runtime_report did not pass parity gates")
    parity_q4_sha = _nested(
        parity,
        "inputs",
        "gguf",
        "backend",
        "gguf_sha256",
        label="cpu_runtime_report",
    )
    if parity_q4_sha != q4_sha256:
        raise LlmReleaseV5Error("cpu_runtime_report does not bind the supplied Q4 model")

    fixture = records["prompt_fixture"]
    fixture_contract = records["prompt_fixture_build_receipt"]
    fixture_sha256 = _sha256_file(paths["prompt_fixture"])
    if (
        _nested(
            fixture_contract,
            "fixture",
            "sha256",
            label="prompt_fixture_build_receipt",
        )
        != fixture_sha256
    ):
        raise LlmReleaseV5Error("prompt_fixture_build_receipt does not bind prompt_fixture")
    if _nested(
        fixture_contract,
        "dataset",
        "manifest",
        "sha256",
        label="prompt_fixture_build_receipt",
    ) != _sha256_file(paths["split_manifest"]):
        raise LlmReleaseV5Error("prompt_fixture_build_receipt does not bind split_manifest")
    if fixture.get("fixture_id") != _nested(
        fixture_contract,
        "fixture",
        "fixture_id",
        label="prompt_fixture_build_receipt",
    ):
        raise LlmReleaseV5Error("prompt fixture_id differs from its build receipt")


def _source_binding(
    path: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": record.get("schema"),
        "status": record.get("status"),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _build_evaluation_summary(
    report: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    summary = {
        "schema": EVALUATION_SUMMARY_SCHEMA,
        "status": report["status"],
        "source_report": _source_binding(report_path, report),
        "scope": {key: report.get(key) for key in ("examples", "rows", "split", "ablations")},
        "baseline": _strip_path_fields(report.get("baseline", {})),
        "candidate": _strip_path_fields(report.get("candidate", {})),
        "bootstrap": _strip_path_fields(report.get("bootstrap", {})),
        "metrics": _strip_path_fields(report.get("metrics", {})),
        "family_strict_exact": _strip_path_fields(report.get("family_strict_exact", {})),
        "promotion_thresholds": report.get("promotion_thresholds", {}),
        "promotion_indicators": report.get("promotion_indicators", {}),
        "all_required_gates_pass": report["all_required_gates_pass"],
        "promotion_context_valid": report["promotion_context_valid"],
        "promotion_allowed": report["promotion_allowed"],
        "q4_non_inferiority_evaluated": report.get(
            "q4_non_inferiority_evaluated",
            False,
        ),
        "claim_boundary": report.get("claim_boundary"),
        "x5_contacted": False,
        "bpu_llm_claimed": False,
        "production_integration_allowed": False,
        "default_enabled": False,
        "autostart": False,
    }
    _assert_no_absolute_paths(summary, label="runtime_evaluation_summary")
    return summary


def _build_parity_summary(
    report: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    inputs = report.get("inputs", {})
    hf = inputs.get("hf", {}) if isinstance(inputs, dict) else {}
    gguf = inputs.get("gguf", {}) if isinstance(inputs, dict) else {}
    summary = {
        "schema": PARITY_SUMMARY_SCHEMA,
        "status": report["status"],
        "source_report": _source_binding(report_path, report),
        "claim_boundary": report.get("claim_boundary"),
        "configuration": _strip_path_fields(report.get("configuration", {})),
        "inputs": {
            "hf": {
                "hashes": _strip_path_fields(hf.get("hashes", {})),
                "backend": _strip_path_fields(hf.get("backend", {})),
            },
            "gguf": {
                "hashes": _strip_path_fields(gguf.get("hashes", {})),
                "backend": _strip_path_fields(gguf.get("backend", {})),
            },
            "dataset": _strip_path_fields(inputs.get("dataset", {})),
        },
        "comparisons": _strip_path_fields(report.get("comparisons", {})),
        "degradation_diagnostics_ablation_none": _strip_path_fields(
            report.get("degradation_diagnostics_ablation_none", {})
        ),
        "non_degradation_gate": report["non_degradation_gate"],
        "x5_contacted": False,
        "bpu_llm_claimed": False,
        "production_integration_allowed": False,
        "default_enabled": False,
        "autostart": False,
    }
    _assert_no_absolute_paths(summary, label="runtime_cpu_parity_summary")
    return summary


def _build_fixture_runtime_contract(
    *,
    fixture: dict[str, Any],
    fixture_path: Path,
    receipt: dict[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    dataset = receipt.get("dataset", {})
    manifest = dataset.get("manifest", {}) if isinstance(dataset, dict) else {}
    calibration = dataset.get("calibration", {}) if isinstance(dataset, dict) else {}
    receipt_fixture = receipt.get("fixture", {})
    contract = {
        "schema": FIXTURE_CONTRACT_SCHEMA,
        "status": "CPU_RUNTIME_FIXTURE_BOUND_NOT_X5_EXECUTED",
        "fixture": {
            "schema": fixture["schema"],
            "fixture_id": fixture["fixture_id"],
            "bytes": fixture_path.stat().st_size,
            "sha256": _sha256_file(fixture_path),
            "generation_fields": receipt_fixture.get("generation_fields"),
            "assistant_target_in_generation": receipt_fixture.get("assistant_target_in_generation"),
            "expected_contract_canonical_json_in_generation": (
                receipt_fixture.get("expected_contract_canonical_json_in_generation")
            ),
        },
        "source_build_receipt": _source_binding(receipt_path, receipt),
        "dataset": {
            "manifest": {key: manifest.get(key) for key in ("schema", "bytes", "sha256")},
            "calibration": {
                key: calibration.get(key) for key in ("bytes", "rows", "sha256", "manifest_sha256")
            },
        },
        "selection": _strip_path_fields(receipt.get("selection", {})),
        "claim_boundary": receipt.get("claim_boundary"),
        "x5_contacted": False,
        "bpu_llm_claimed": False,
        "production_integration_allowed": False,
        "default_enabled": False,
        "autostart": False,
    }
    _assert_no_absolute_paths(contract, label="runtime_prompt_fixture_contract")
    return contract


def _workspace_relative(root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(root).as_posix()
    except ValueError as exc:
        raise LlmReleaseV5Error(f"path escapes workspace: {path}") from exc


def _prepare_output_directory(root: Path, output_dir: Path) -> Path:
    raw = output_dir if output_dir.is_absolute() else root / output_dir
    absolute = Path(os.path.abspath(os.fspath(raw)))
    try:
        absolute.relative_to(root)
    except ValueError as exc:
        raise LlmReleaseV5Error("output_dir must stay inside workspace") from exc
    if os.path.lexists(absolute):
        raise FileExistsError(f"output_dir already exists; exclusive create required: {absolute}")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = absolute.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(root)
    except ValueError as exc:
        raise LlmReleaseV5Error("output_dir parent escapes workspace or uses a symlink") from exc
    absolute.mkdir()
    return absolute


def _prepare_package_root(root: Path, output_root: Path) -> Path:
    raw = output_root if output_root.is_absolute() else root / output_root
    absolute = Path(os.path.abspath(os.fspath(raw)))
    try:
        absolute.relative_to(root)
    except ValueError as exc:
        raise LlmReleaseV5Error("package_output_root must stay inside workspace") from exc
    absolute.parent.mkdir(parents=True, exist_ok=True)
    try:
        absolute.parent.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise LlmReleaseV5Error("package_output_root parent escapes workspace or uses a symlink") from exc
    return absolute


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _validate_role_rows(rows: list[dict[str, str]]) -> None:
    roles: set[str] = set()
    paths: set[str] = set()
    for row in rows:
        role = row["role"]
        path = row["path"]
        if role in roles:
            raise LlmReleaseV5Error(f"duplicate artifact role: {role}")
        if path.casefold() in paths:
            raise LlmReleaseV5Error(f"duplicate artifact path: {path}")
        roles.add(role)
        paths.add(path.casefold())


def _resolve_input_paths(
    root: Path,
    inputs: LlmReleaseInputs,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for field in fields(inputs):
        role = _INPUT_ROLES[field.name]
        supplied = getattr(inputs, field.name)
        path = _input_file_within_root(
            root,
            Path(supplied),
            field=role,
        )
        logical = _workspace_relative(root, path)
        _scan_file(path, logical_path=logical)
        resolved[role] = path
    return resolved


def _load_records(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {role: _load_json(paths[role], role=role) for role in _EXPECTED_SCHEMAS}


def _runtime_payload_preflight(
    *,
    records: dict[str, dict[str, Any]],
    rag_rows: list[dict[str, Any]],
    generated: dict[str, dict[str, Any]],
) -> None:
    for role in (
        "task_contract",
        "preprocessing_contract",
        "source_catalog",
        "rag_manifest",
        "prompt_fixture",
    ):
        _assert_no_absolute_paths(records[role], label=role)
    for index, row in enumerate(rag_rows, start=1):
        _assert_no_absolute_paths(row, label=f"rag_chunks[{index}]")
    for role, value in generated.items():
        _assert_no_absolute_paths(value, label=role)


def build_llm_release_v5(
    *,
    workspace_root: Path,
    candidate_id: str,
    created_at: str,
    output_dir: Path,
    inputs: LlmReleaseInputs,
    package_output_root: Path | None = None,
) -> dict[str, Any]:
    """Build one immutable local-CPU release and optional inactive X5 package."""

    root = Path(workspace_root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise LlmReleaseV5Error("workspace_root must be a regular non-symlink directory")

    paths = _resolve_input_paths(root, inputs)
    records = _load_records(paths)
    rag_rows = _load_rag_chunks(paths["rag_chunks"])
    _validate_contract_bindings(
        records=records,
        paths=paths,
        rag_rows=rag_rows,
    )

    generated = {
        "runtime_evaluation_summary": _build_evaluation_summary(
            records["evaluation_report"],
            paths["evaluation_report"],
        ),
        "runtime_cpu_parity_summary": _build_parity_summary(
            records["cpu_runtime_report"],
            paths["cpu_runtime_report"],
        ),
        "prompt_fixture_contract": _build_fixture_runtime_contract(
            fixture=records["prompt_fixture"],
            fixture_path=paths["prompt_fixture"],
            receipt=records["prompt_fixture_build_receipt"],
            receipt_path=paths["prompt_fixture_build_receipt"],
        ),
    }
    _runtime_payload_preflight(
        records=records,
        rag_rows=rag_rows,
        generated=generated,
    )

    release_dir = _prepare_output_directory(root, Path(output_dir))
    try:
        generated_paths = {
            "runtime_evaluation_summary": (release_dir / "runtime_evaluation_summary.v5.json"),
            "runtime_cpu_parity_summary": (release_dir / "runtime_hf_gguf_parity_summary.v5.json"),
            "prompt_fixture_contract": (release_dir / "runtime_prompt_fixture_contract.v5.json"),
        }
        for role, path in generated_paths.items():
            _write_exclusive(path, _pretty_bytes(generated[role]))
            _scan_file(path, logical_path=_workspace_relative(root, path))

        artifact_paths = {
            **paths,
            **generated_paths,
        }
        artifact_rows = [
            {
                "role": role,
                "path": _workspace_relative(root, artifact_paths[role]),
            }
            for role in sorted(artifact_paths)
        ]
        _validate_role_rows(artifact_rows)

        metadata = {
            "candidate_family": "X5-ICMat Foundry",
            "model": "ICMat-Qwen-0.5B Q4_K_M",
            "runtime_backend": "local llama.cpp CPU GGUF",
            "runtime_policy": "explicit one-shot sidecar; no hidden router",
            "stage": STAGE,
            "claim_status": CLAIM_STATUS,
            "x5_contacted": False,
            "x5_runtime_verified": False,
            "bpu_llm_claimed": False,
            "production_integration_allowed": False,
            "default_enabled": False,
            "autostart": False,
            "production_dependency": False,
            "release_only_roles": sorted(_RELEASE_ONLY_ROLES),
            "package_roles": sorted(_PACKAGE_DESTINATIONS),
            "builder_sha256": _sha256_file(Path(__file__)),
            "claim_boundary": (
                "Local HF/GGUF CPU behavior and evidence bindings are verified. "
                "This release does not claim X5 execution, BPU LLM execution, "
                "autostart, production integration, or replacement of frozen "
                "AI-brain services."
            ),
        }
        spec = {
            "schema": SPEC_SCHEMA,
            "candidate_id": candidate_id,
            "product_id": PRODUCT_ID,
            "created_at": created_at,
            "stage": STAGE,
            "artifacts": artifact_rows,
            "metadata": metadata,
        }
        spec_path = release_dir / "candidate_spec.v1.json"
        release_path = release_dir / "candidate_release.v1.json"
        allowlist_path = release_dir / "artifact_allowlist.v1.json"
        _write_exclusive(spec_path, _pretty_bytes(spec))
        release = build_release_manifest(root, spec_path, release_path)

        release_by_role = {row["role"]: row for row in release["artifacts"]}
        if set(_PACKAGE_DESTINATIONS) - set(release_by_role):
            raise AssertionError("package roles are absent from candidate release")
        if _RELEASE_ONLY_ROLES & set(_PACKAGE_DESTINATIONS):
            raise AssertionError("release-only evidence entered package allowlist")
        allowlist = {
            "schema": ALLOWLIST_SCHEMA,
            "release_id": candidate_id,
            "artifacts": [
                {
                    "role": role,
                    "source_path": release_by_role[role]["path"],
                    "package_path": _PACKAGE_DESTINATIONS[role],
                }
                for role in sorted(
                    _PACKAGE_DESTINATIONS,
                    key=lambda item: (
                        _PACKAGE_DESTINATIONS[item].casefold(),
                        item,
                    ),
                )
            ],
        }
        _write_exclusive(allowlist_path, _pretty_bytes(allowlist))
        _scan_file(
            allowlist_path,
            logical_path=_workspace_relative(root, allowlist_path),
        )

        release_verification = verify_release_manifest(root, release_path)
        package_result: dict[str, Any] | None = None
        if package_output_root is not None:
            package_root = _prepare_package_root(root, package_output_root)
            package_result = build_package(
                root,
                release_path,
                allowlist_path,
                package_root,
            )
            package_result = {
                **package_result,
                "independent_verification": verify_package(Path(package_result["package_manifest"])),
            }

        result = {
            "schema": BUILD_RESULT_SCHEMA,
            "status": (
                "CPU_RUNTIME_VERIFIED_INACTIVE_PACKAGE_BUILT"
                if package_result is not None
                else "CPU_RUNTIME_VERIFIED_RELEASE_BUILT"
            ),
            "candidate_id": candidate_id,
            "product_id": PRODUCT_ID,
            "stage": STAGE,
            "claim_status": CLAIM_STATUS,
            "production_integration_allowed": False,
            "x5_contacted": False,
            "bpu_llm_claimed": False,
            "default_enabled": False,
            "autostart": False,
            "outputs": {
                "candidate_spec": {
                    "path": _workspace_relative(root, spec_path),
                    "sha256": _sha256_file(spec_path),
                },
                "candidate_release": {
                    "path": _workspace_relative(root, release_path),
                    "sha256": _sha256_file(release_path),
                    "manifest_sha256": release["manifest_sha256"],
                },
                "artifact_allowlist": {
                    "path": _workspace_relative(root, allowlist_path),
                    "sha256": _sha256_file(allowlist_path),
                },
            },
            "release_verification": release_verification,
            "package": package_result,
        }
        _assert_no_absolute_paths(result["outputs"], label="build_result.outputs")
        return result
    except BaseException:
        shutil.rmtree(release_dir, ignore_errors=True)
        raise


__all__ = [
    "BUILD_RESULT_SCHEMA",
    "CLAIM_STATUS",
    "EVALUATION_SUMMARY_SCHEMA",
    "FIXTURE_CONTRACT_SCHEMA",
    "LlmReleaseInputs",
    "LlmReleaseV5Error",
    "PARITY_SUMMARY_SCHEMA",
    "PRODUCT_ID",
    "STAGE",
    "build_llm_release_v5",
]
