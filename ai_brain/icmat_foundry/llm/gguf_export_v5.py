"""Offline, atomic GGUF export for an ICMat v5 QLoRA adapter."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

EXPORTER_VERSION = "icmat-gguf-export-v5.1.0"
PREFLIGHT_SCHEMA = "icmat_gguf_export_preflight.v5"
EXPORT_RECEIPT_SCHEMA = "icmat_gguf_export_receipt.v5"
FAILURE_RECEIPT_SCHEMA = "icmat_gguf_export_failure_receipt.v5"
CLAIM_BOUNDARY = (
    "This receipt proves only that a fixed local base model and QLoRA adapter "
    "were merged with PEFT, converted by the recorded llama.cpp converter to "
    "GGUF F16, and quantized by the recorded llama-quantize tool to Q4_K_M. "
    "It does not establish model quality, scientific correctness, BPU "
    "conversion, RDK X5 execution, production integration, or service safety."
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_F16_NAME = "icmat-qwen05b-f16.gguf"
DEFAULT_Q4_NAME = "icmat-qwen05b-q4_k_m.gguf"

CommandRunner = Callable[[Sequence[str], Path], Any]
MergeHook = Callable[[Path, Path, Path], Mapping[str, Any] | None]


@dataclass(frozen=True)
class ExportInputs:
    base_model: Path
    adapter: Path
    converter: Path
    quantizer: Path
    converter_sha256: str
    quantizer_sha256: str
    python_executable: Path = Path(sys.executable)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_expected_sha256(value: str, role: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{role} expected SHA256 must be 64 hexadecimal characters")
    return normalized


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _tree_inventory(root: Path) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    records: list[dict[str, Any]] = []
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            target = path.resolve(strict=True)
            if not target.is_file():
                raise ValueError(f"tree symlink is not a regular file: {path}")
        elif not path.is_file():
            continue
        records.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not records:
        raise ValueError(f"input tree is empty: {resolved}")
    return {
        "path": str(resolved),
        "files": records,
        "file_count": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "tree_sha256": _canonical_sha256(records),
    }


def _load_json_object(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must contain a JSON object: {path}")
    return value


def _adapter_base_identifiers(base_model: Path, base_config: Mapping[str, Any]) -> set[str]:
    identifiers = {
        str(base_model),
        str(base_model).replace("\\", "/"),
        base_model.name,
    }
    for key in ("_name_or_path", "name_or_path"):
        value = base_config.get(key)
        if isinstance(value, str) and value.strip():
            identifiers.add(value.strip())
            identifiers.add(value.strip().replace("\\", "/"))
    return identifiers


def _validate_adapter_config(
    *,
    base_model: Path,
    adapter: Path,
    base_config: Mapping[str, Any],
    adapter_config: Mapping[str, Any],
) -> dict[str, Any]:
    peft_type = str(adapter_config.get("peft_type", "")).upper()
    task_type = str(adapter_config.get("task_type", "")).upper()
    if peft_type != "LORA":
        raise ValueError(f"adapter peft_type must be LORA, got {peft_type!r}")
    if task_type != "CAUSAL_LM":
        raise ValueError(f"adapter task_type must be CAUSAL_LM, got {task_type!r}")

    adapter_base = adapter_config.get("base_model_name_or_path")
    if not isinstance(adapter_base, str) or not adapter_base.strip():
        raise ValueError("adapter config must declare base_model_name_or_path")
    adapter_base = adapter_base.strip()
    identifiers = _adapter_base_identifiers(base_model, base_config)
    base_matches = (
        adapter_base in identifiers
        or adapter_base.replace("\\", "/") in identifiers
    )
    candidate = Path(adapter_base).expanduser()
    if candidate.is_absolute():
        try:
            base_matches = candidate.resolve(strict=True) == base_model
        except OSError:
            base_matches = False
    if not base_matches:
        raise ValueError(
            "adapter base_model_name_or_path does not identify the supplied base model"
        )

    weight_files = sorted(
        path.name
        for pattern in ("adapter_model*.safetensors", "adapter_model*.bin")
        for path in adapter.glob(pattern)
        if path.is_file()
    )
    if not weight_files:
        raise FileNotFoundError("adapter contains no adapter_model weights")
    return {
        "peft_type": peft_type,
        "task_type": task_type,
        "base_model_name_or_path": adapter_base,
        "base_match": True,
        "inference_mode": adapter_config.get("inference_mode"),
        "rank": adapter_config.get("r"),
        "lora_alpha": adapter_config.get("lora_alpha"),
        "target_modules": adapter_config.get("target_modules"),
        "weight_files": weight_files,
    }


def _tool_record(path: Path, expected_sha256: str, role: str) -> dict[str, Any]:
    record = _file_record(path)
    expected = _validate_expected_sha256(expected_sha256, role)
    if record["sha256"] != expected:
        raise PermissionError(
            f"{role} SHA256 mismatch: expected {expected}, got {record['sha256']}"
        )
    record["expected_sha256"] = expected
    record["sha256_match"] = True
    record["version_identity"] = f"sha256:{expected}"
    return record


def _source_inventory() -> dict[str, dict[str, Any]]:
    paths = {
        "exporter": Path(__file__).resolve(),
        "cli": WORKSPACE_ROOT / "tools" / "export_icmat_gguf_v5.py",
    }
    return {role: _file_record(path) for role, path in paths.items()}


def preflight_gguf_export(inputs: ExportInputs) -> dict[str, Any]:
    """Validate all inputs without importing ML libraries or running tools."""

    base_model = Path(inputs.base_model).resolve(strict=True)
    adapter = Path(inputs.adapter).resolve(strict=True)
    converter = Path(inputs.converter).resolve(strict=True)
    quantizer = Path(inputs.quantizer).resolve(strict=True)
    python_executable = Path(inputs.python_executable).resolve(strict=True)
    if not base_model.is_dir():
        raise NotADirectoryError(base_model)
    if not adapter.is_dir():
        raise NotADirectoryError(adapter)
    if converter.suffix.lower() != ".py":
        raise ValueError("llama.cpp converter must be a fixed Python source file")

    base_config_path = base_model / "config.json"
    adapter_config_path = adapter / "adapter_config.json"
    if not base_config_path.is_file():
        raise FileNotFoundError(base_config_path)
    if not adapter_config_path.is_file():
        raise FileNotFoundError(adapter_config_path)
    base_config = _load_json_object(base_config_path, "base config")
    adapter_config = _load_json_object(adapter_config_path, "adapter config")
    if not isinstance(base_config.get("model_type"), str):
        raise ValueError("base config must declare model_type")

    base_inventory = _tree_inventory(base_model)
    adapter_inventory = _tree_inventory(adapter)
    adapter_contract = _validate_adapter_config(
        base_model=base_model,
        adapter=adapter,
        base_config=base_config,
        adapter_config=adapter_config,
    )
    tools = {
        "converter": _tool_record(
            converter,
            inputs.converter_sha256,
            "converter",
        ),
        "quantizer": _tool_record(
            quantizer,
            inputs.quantizer_sha256,
            "quantizer",
        ),
        "python": _file_record(python_executable),
    }
    tools["converter"]["runtime_tree"] = _tree_inventory(converter.parent)
    tools["quantizer"]["runtime_tree"] = _tree_inventory(quantizer.parent)
    input_fingerprint = _canonical_sha256(
        {
            "base_tree_sha256": base_inventory["tree_sha256"],
            "adapter_tree_sha256": adapter_inventory["tree_sha256"],
            "converter_sha256": tools["converter"]["sha256"],
            "converter_runtime_tree_sha256": tools["converter"]["runtime_tree"][
                "tree_sha256"
            ],
            "quantizer_sha256": tools["quantizer"]["sha256"],
            "quantizer_runtime_tree_sha256": tools["quantizer"]["runtime_tree"][
                "tree_sha256"
            ],
            "python_sha256": tools["python"]["sha256"],
            "adapter_contract": adapter_contract,
        }
    )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "exporter_version": EXPORTER_VERSION,
        "created_at": _utc_now(),
        "status": "PASS_READ_ONLY_GGUF_EXPORT_PREFLIGHT_NOT_EXPORTED",
        "read_only": True,
        "network_used": False,
        "ml_runtime_imported": False,
        "x5_touched": False,
        "services_touched": False,
        "base_model": base_inventory,
        "base_model_type": base_config["model_type"],
        "adapter": adapter_inventory,
        "adapter_contract": adapter_contract,
        "tools": tools,
        "input_fingerprint_sha256": input_fingerprint,
        "planned_outputs": {
            "merged_hf": "merged_hf",
            "gguf_f16": DEFAULT_F16_NAME,
            "gguf_q4_k_m": DEFAULT_Q4_NAME,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _package_versions(names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


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
        dtype="auto",
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
    )
    peft_model = PeftModel.from_pretrained(
        base,
        str(adapter),
        is_trainable=False,
        local_files_only=True,
    )
    merged = peft_model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(
        str(output),
        safe_serialization=True,
        max_shard_size="2GB",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    tokenizer.save_pretrained(str(output))
    return {
        "implementation": "transformers.AutoModelForCausalLM+peft.PeftModel",
        "operation": "merge_and_unload",
        "safe_merge": True,
        "device": "cpu",
        "local_files_only": True,
        "trust_remote_code": False,
    }


def _default_command_runner(command: Sequence[str], cwd: Path) -> Any:
    return subprocess.run(
        list(command),
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        check=False,
    )


def _stream_record(value: Any) -> dict[str, Any]:
    text = "" if value is None else str(value)
    payload = text.encode("utf-8", errors="replace")
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "tail": text[-2000:],
    }


def _run_checked(
    *,
    runner: CommandRunner,
    command: Sequence[str],
    cwd: Path,
    role: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = runner(tuple(str(part) for part in command), cwd)
    returncode = int(getattr(result, "returncode", -1))
    record = {
        "role": role,
        "command": [str(part) for part in command],
        "cwd": str(cwd),
        "shell": False,
        "returncode": returncode,
        "wall_seconds": time.perf_counter() - started,
        "stdout": _stream_record(getattr(result, "stdout", "")),
        "stderr": _stream_record(getattr(result, "stderr", "")),
    }
    if returncode != 0:
        raise RuntimeError(
            f"{role} failed with exit code {returncode}: "
            f"{record['stderr']['tail']}"
        )
    return record


def _validate_merged_hf(path: Path) -> dict[str, Any]:
    inventory = _tree_inventory(path)
    names = {record["path"] for record in inventory["files"]}
    if "config.json" not in names:
        raise RuntimeError("merged HF output has no config.json")
    if not any(
        name.endswith(".safetensors") or name.startswith("pytorch_model")
        for name in names
    ):
        raise RuntimeError("merged HF output has no model weights")
    return inventory


def _validate_gguf(path: Path, role: str) -> dict[str, Any]:
    record = _file_record(path)
    if record["bytes"] <= 0:
        raise RuntimeError(f"{role} GGUF is empty")
    return record


def _safe_new_output(output_dir: Path) -> tuple[Path, Path]:
    raw = Path(output_dir)
    if raw.name in {"", ".", ".."}:
        raise ValueError("output must name a new directory")
    parent = raw.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    final = parent / raw.name
    if os.path.lexists(final):
        raise FileExistsError(final)
    return parent, final


def _failure_path(parent: Path, final_name: str, run_id: str) -> Path:
    return parent / f".{final_name}.failed-{run_id}-{uuid.uuid4().hex}"


def _publish_failure_only(
    *,
    parent: Path,
    final_name: str,
    run_id: str,
    failure: Mapping[str, Any],
    stage: Path | None,
) -> None:
    if stage is not None and os.path.lexists(stage):
        shutil.rmtree(stage, ignore_errors=True)
    temporary = parent / f".{final_name}.failure-tmp-{uuid.uuid4().hex}"
    failed = _failure_path(parent, final_name, run_id)
    try:
        os.mkdir(temporary)
        _write_json_atomic(temporary / "failure_receipt.v5.json", failure)
        os.replace(temporary, failed)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def export_gguf_v5(
    *,
    inputs: ExportInputs,
    output_dir: Path,
    merge_hook: MergeHook | None = None,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Merge, convert, quantize, and atomically publish an offline export."""

    parent, final_output = _safe_new_output(Path(output_dir))
    preflight: dict[str, Any] | None = None
    stage: Path | None = None
    run_id = "icmat-gguf-v5-" + uuid.uuid4().hex[:20]
    active_stage = "preflight"
    started = time.perf_counter()
    try:
        preflight = preflight_gguf_export(inputs)
        source_inventory = _source_inventory()
        run_id = "icmat-gguf-v5-" + _canonical_sha256(
            {
                "input_fingerprint": preflight["input_fingerprint_sha256"],
                "source_inventory": source_inventory,
                "output_name": final_output.name,
                "exporter_version": EXPORTER_VERSION,
            }
        )[:20]
        stage = parent / f".{final_output.name}.tmp-{run_id}-{uuid.uuid4().hex}"
        if os.path.lexists(stage):
            raise FileExistsError(stage)
        os.mkdir(stage)
        _write_json_atomic(stage / "preflight.v5.json", preflight)

        base_model = Path(inputs.base_model).resolve(strict=True)
        adapter = Path(inputs.adapter).resolve(strict=True)
        converter = Path(inputs.converter).resolve(strict=True)
        quantizer = Path(inputs.quantizer).resolve(strict=True)
        python_executable = Path(inputs.python_executable).resolve(strict=True)

        active_stage = "merge_hf"
        merged_hf = stage / "merged_hf"
        os.mkdir(merged_hf)
        merge = _default_merge_hook if merge_hook is None else merge_hook
        merge_metadata = dict(merge(base_model, adapter, merged_hf) or {})
        merged_inventory = _validate_merged_hf(merged_hf)

        active_stage = "convert_f16"
        f16_path = stage / DEFAULT_F16_NAME
        converter_command = [
            str(python_executable),
            str(converter),
            str(merged_hf),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        ]
        runner = _default_command_runner if command_runner is None else command_runner
        converter_execution = _run_checked(
            runner=runner,
            command=converter_command,
            cwd=stage,
            role="llama_cpp_converter_f16",
        )
        f16_record = _validate_gguf(f16_path, "F16")

        active_stage = "quantize_q4_k_m"
        q4_path = stage / DEFAULT_Q4_NAME
        quantizer_command = [
            str(quantizer),
            str(f16_path),
            str(q4_path),
            "Q4_K_M",
        ]
        quantizer_execution = _run_checked(
            runner=runner,
            command=quantizer_command,
            cwd=stage,
            role="llama_quantizer_q4_k_m",
        )
        q4_record = _validate_gguf(q4_path, "Q4_K_M")

        active_stage = "input_revalidation"
        final_snapshot = preflight_gguf_export(inputs)
        if (
            final_snapshot["input_fingerprint_sha256"]
            != preflight["input_fingerprint_sha256"]
        ):
            raise PermissionError("export inputs changed while the export was running")

        active_stage = "receipt"
        receipt = {
            "schema": EXPORT_RECEIPT_SCHEMA,
            "exporter_version": EXPORTER_VERSION,
            "created_at": _utc_now(),
            "status": "PASS_GGUF_EXPORT_COMPLETED_NOT_DEPLOYED",
            "run_id": run_id,
            "atomic_publish": True,
            "network_used": False,
            "x5_touched": False,
            "services_touched": False,
            "autostart_created": False,
            "input_snapshot": preflight,
            "source_inventory": source_inventory,
            "merge": {
                "required_operation": "PEFT merge_and_unload",
                "metadata": merge_metadata,
                "merged_hf": merged_inventory,
            },
            "commands": {
                "converter": converter_execution,
                "quantizer": quantizer_execution,
            },
            "artifacts": {
                "gguf_f16": {
                    **f16_record,
                    "path": DEFAULT_F16_NAME,
                    "format": "GGUF",
                    "quantization": "F16",
                },
                "gguf_q4_k_m": {
                    **q4_record,
                    "path": DEFAULT_Q4_NAME,
                    "format": "GGUF",
                    "quantization": "Q4_K_M",
                },
            },
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "dependencies": _package_versions(
                    (
                        "torch",
                        "transformers",
                        "tokenizers",
                        "peft",
                        "accelerate",
                        "safetensors",
                    )
                ),
                "converter_version_identity": preflight["tools"]["converter"][
                    "version_identity"
                ],
                "quantizer_version_identity": preflight["tools"]["quantizer"][
                    "version_identity"
                ],
            },
            "wall_seconds": time.perf_counter() - started,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        _write_json_atomic(stage / "export_receipt.v5.json", receipt)
        os.replace(stage, final_output)
        stage = None
        return receipt
    except BaseException as exc:
        failure = {
            "schema": FAILURE_RECEIPT_SCHEMA,
            "exporter_version": EXPORTER_VERSION,
            "created_at": _utc_now(),
            "status": "FAILED_NO_SUCCESS_EXPORT",
            "run_id": run_id,
            "active_stage": active_stage,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "final_output_created": False,
            "partial_artifacts_retained": False,
            "network_used": False,
            "x5_touched": False,
            "services_touched": False,
            "input_fingerprint_sha256": (
                preflight.get("input_fingerprint_sha256")
                if preflight is not None
                else None
            ),
            "claim_boundary": CLAIM_BOUNDARY,
        }
        try:
            _publish_failure_only(
                parent=parent,
                final_name=final_output.name,
                run_id=run_id,
                failure=failure,
                stage=stage,
            )
            stage = None
        except BaseException:
            pass
        raise


__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_F16_NAME",
    "DEFAULT_Q4_NAME",
    "EXPORTER_VERSION",
    "ExportInputs",
    "export_gguf_v5",
    "preflight_gguf_export",
]
