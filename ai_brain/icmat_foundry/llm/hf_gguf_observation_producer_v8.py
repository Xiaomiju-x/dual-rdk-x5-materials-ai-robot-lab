"""Controlled native-v8 HF/GGUF runtime observation production.

The producer is deliberately separate from parity scoring.  It builds one
target-free request set, starts the selected HF model in a child process and a
pinned loopback-only llama-server, captures their raw process evidence, and
publishes an authority receipt before publishing derived observation files.

This module never reads a reserved blind split, contacts an X5, registers a
service, or changes a production model registry.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from icmat_foundry.llm import (
    gguf_release_v8,
    llama_cpp_eval_v5,
    pointer_hf_eval_v6,
)

VERSION = "icmat-hf-gguf-observation-producer-v8.1.0"
AUTHORITY_SCHEMA = "icmat_runtime_observation_authority.v8"
AUTHORITY_STATUS = "PASS_CONTROLLED_RUNTIME_OBSERVATION_AUTHORITY_V8"
RUNTIME_PROVENANCE = "CONTROLLED_RUNTIME_MODEL_OUTPUT"
FIXTURE_PROVENANCE = "TEST_FIXTURE_NOT_MODEL_EVIDENCE"
REQUEST_SCHEMA = "icmat_target_free_runtime_request.v8"
RAW_RESULTS_SCHEMA = "icmat_runtime_raw_results.v8"
RAW_SAMPLE_SCHEMA = "icmat_runtime_raw_sample.v8"
HF_WORKER_STATUS = "COMPLETE_HF_SELECTED_ADAPTER_RUNTIME_RESULTS_V8"
GGUF_RUN_STATUS = "COMPLETE_GGUF_Q4_K_M_RUNTIME_RESULTS_V8"

EXPECTED_ROWS = 150
FIXED_SEED = 20260729
FIXED_MAX_NEW_TOKENS = 64
FIXED_CONTEXT_SIZE = 2048
FIXED_THREADS = 4
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024

REQUEST_FILENAME = "target_free_requests.v8.jsonl"
HF_STDOUT_FILENAME = "hf.stdout.v8.json"
HF_STDERR_FILENAME = "hf.stderr.v8.log"
GGUF_STDOUT_FILENAME = "gguf.stdout.v8.log"
GGUF_STDERR_FILENAME = "gguf.stderr.v8.log"
GGUF_RESULTS_FILENAME = "gguf.responses.v8.json"
AUTHORITY_FILENAME = "runtime_authority.v8.json"
HF_OBSERVATIONS_FILENAME = "hf_observations.v8.json"
GGUF_OBSERVATIONS_FILENAME = "gguf_observations.v8.json"

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CLI_PATH = WORKSPACE_ROOT / "tools" / "produce_icmat_hf_gguf_observations_v8.py"

_AUTHORITY_KEYS = {
    "schema",
    "version",
    "status",
    "created_at_utc",
    "provenance_kind",
    "preflight",
    "release_authority_inputs",
    "release_authority",
    "low_level_export",
    "request_set",
    "generation_policy",
    "implementation",
    "model_bindings",
    "executions",
    "execution_boundary",
    "claim_boundary",
    "canonical_digest_sha256",
}
_RAW_RESULT_KEYS = {
    "schema",
    "version",
    "status",
    "kind",
    "provenance_kind",
    "fixture_not_model_evidence",
    "model_invoked",
    "request_set_sha256",
    "generation_policy",
    "model",
    "backend",
    "samples",
    "canonical_digest_sha256",
}
_RAW_SAMPLE_KEYS = {
    "schema",
    "example_id",
    "raw_pointer",
    "finish_reason",
    "finish_category",
    "latency_ms",
    "peak_rss_bytes",
    "input_tokens",
    "output_tokens",
    "generation_error",
}
_RELEASE_FILE_ROLES = {
    "selection_freeze": "selection_freeze",
    "evaluation_index": "evaluation_index",
    "training_receipt": "training_receipt",
    "calibration_receipt": "calibration_receipt",
    "ablation_receipt": "ablation_receipt",
    "postfreeze_receipt": "postfreeze_receipt",
    "qualification_receipt": "qualification_receipt",
    "verification_receipt": "verification_receipt",
}


class ObservationProducerV8Error(RuntimeError):
    """Raised when controlled runtime evidence cannot be established."""


@dataclass(frozen=True)
class ProducerInputsV8:
    release_authority: gguf_release_v8.ReleaseAuthorityInputsV8
    preflight_receipt: Path
    preflight_receipt_sha256: str
    export_receipt: Path
    export_receipt_sha256: str
    gguf_model: Path
    gguf_model_sha256: str
    llama_server: Path
    llama_server_sha256: str
    output_dir: Path
    cli_runner_path: Path = EXPECTED_CLI_PATH
    python_executable: Path = Path(sys.executable)


@dataclass(frozen=True)
class ExecutionCaptureV8:
    document: Mapping[str, Any]
    stdout: bytes
    stderr: bytes
    returncode: int
    command: tuple[str, ...]
    trace: Mapping[str, Any]


@dataclass(frozen=True)
class VerifiedRuntimeAuthorityV8:
    snapshot: gguf_release_v8.FileSnapshotV8
    receipt: Mapping[str, Any]
    results: Mapping[str, Mapping[str, Mapping[str, Any]]]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical_sha(value: Any) -> str:
    return gguf_release_v8.canonical_sha256(value)


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
    return b"".join(
        (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        for value in values
    )


def _exact(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ObservationProducerV8Error(f"{label} exact field set mismatch")
    return value


def _sha(value: Any, *, label: str) -> str:
    try:
        return gguf_release_v8._require_sha256(value, label=label)
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise ObservationProducerV8Error(str(exc)) from exc


def _verify_digest(value: Mapping[str, Any], *, label: str) -> None:
    claimed = _sha(value.get("canonical_digest_sha256"), label=f"{label} digest")
    body = dict(value)
    del body["canonical_digest_sha256"]
    if _canonical_sha(body) != claimed:
        raise ObservationProducerV8Error(f"{label} canonical digest mismatch")


def _snapshot_file(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    maximum_bytes: int = MAX_ARTIFACT_BYTES,
) -> gguf_release_v8.FileSnapshotV8:
    try:
        return gguf_release_v8._snapshot_file(
            path,
            label=label,
            expected_sha256=expected_sha256,
            maximum_bytes=maximum_bytes,
        )
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise ObservationProducerV8Error(f"{label} rejected") from exc


def _snapshot_binary(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    maximum_bytes: int = 16 * 1024 * 1024 * 1024,
) -> gguf_release_v8.BinaryFileSnapshotV8:
    try:
        return gguf_release_v8._snapshot_binary_file(
            path,
            label=label,
            expected_sha256=expected_sha256,
            maximum_bytes=maximum_bytes,
        )
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise ObservationProducerV8Error(f"{label} rejected") from exc


def _write_exclusive(path: Path, payload: bytes) -> Path:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    return path.resolve(strict=True)


def generation_policy_v8() -> dict[str, Any]:
    return {
        "do_sample": False,
        "temperature": 0,
        "max_new_tokens": FIXED_MAX_NEW_TOKENS,
        "seed": FIXED_SEED,
        "stop_on_eos": True,
        "batch_size": 1,
        "device": "LOCAL_PC_CPU",
    }


def request_records_v8(records: Sequence[Any]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        messages = getattr(record, "messages", None)
        if (
            not isinstance(messages, Sequence)
            or isinstance(messages, (str, bytes))
            or len(messages) != 2
        ):
            raise ObservationProducerV8Error(
                f"validation record {index} has no target-free two-message request"
            )
        normalized: list[dict[str, str]] = []
        for message_index, message in enumerate(messages):
            if (
                not isinstance(message, Mapping)
                or set(message) != {"role", "content"}
                or message.get("role") not in {"system", "user"}
                or not isinstance(message.get("content"), str)
            ):
                raise ObservationProducerV8Error(
                    f"validation record {index} message {message_index} is invalid"
                )
            normalized.append(
                {"role": str(message["role"]), "content": str(message["content"])}
            )
        if any(message["role"] == "assistant" for message in normalized):
            raise ObservationProducerV8Error("target-free request contains assistant target")
        requests.append(
            {
                "schema": REQUEST_SCHEMA,
                "example_id": str(record.example_id),
                "compiler_prompt_sha256": str(record.prompt_sha256),
                "messages": normalized,
            }
        )
    if len(requests) != EXPECTED_ROWS or len(
        {request["example_id"] for request in requests}
    ) != EXPECTED_ROWS:
        raise ObservationProducerV8Error("runtime request set must contain 150 unique rows")
    return requests


def request_payload_v8(records: Sequence[Any]) -> bytes:
    return _jsonl_bytes(request_records_v8(records))


def _parse_request_payload(payload: bytes) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ObservationProducerV8Error("request set is not UTF-8") from exc
    if len(lines) != EXPECTED_ROWS or any(not line for line in lines):
        raise ObservationProducerV8Error("request set must contain 150 rows")
    values: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ObservationProducerV8Error(
                f"request row {index} is not JSON"
            ) from exc
        _exact(
            value,
            {"schema", "example_id", "compiler_prompt_sha256", "messages"},
            label=f"request row {index}",
        )
        if value["schema"] != REQUEST_SCHEMA:
            raise ObservationProducerV8Error(f"request row {index} schema mismatch")
        _sha(
            value["compiler_prompt_sha256"],
            label=f"request row {index} prompt SHA",
        )
        messages = value["messages"]
        if not isinstance(messages, list) or len(messages) != 2:
            raise ObservationProducerV8Error(f"request row {index} messages invalid")
        for message in messages:
            _exact(message, {"role", "content"}, label=f"request row {index} message")
            if message["role"] not in {"system", "user"} or not isinstance(
                message["content"], str
            ):
                raise ObservationProducerV8Error(
                    f"request row {index} is not target-free"
                )
        values.append(dict(value))
    ids = [str(value["example_id"]) for value in values]
    if len(set(ids)) != EXPECTED_ROWS:
        raise ObservationProducerV8Error("request set contains duplicate IDs")
    return values


def _model_bindings(
    authorities: gguf_release_v8.ReleaseAuthorityInputsV8,
    *,
    chain: Mapping[str, Any],
) -> dict[str, Any]:
    return _model_bindings_from_paths(
        base_model_dir=authorities.base_model_dir,
        selected_adapter_dir=authorities.selected_adapter_dir,
        chain=chain,
    )


def _model_bindings_from_paths(
    *,
    base_model_dir: Path,
    selected_adapter_dir: Path,
    chain: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        trees = gguf_release_v8._tree_bindings(
            base_model_dir=base_model_dir,
            selected_adapter_dir=selected_adapter_dir,
        )
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise ObservationProducerV8Error("runtime model trees rejected") from exc
    if (
        trees["base_model_tree_sha256"] != chain["base_model_tree_sha256"]
        or trees["checkpoint_tree_sha256"] != chain["checkpoint_tree_sha256"]
        or trees["adapter_tree_sha256"] != chain["adapter_tree_sha256"]
    ):
        raise ObservationProducerV8Error("runtime model trees differ from preflight")
    return trees


def _capture_release_authority_inputs(
    inputs: gguf_release_v8.ReleaseAuthorityInputsV8,
) -> dict[str, Any]:
    expected = {
        "selection_freeze": inputs.selection_freeze_sha256,
        "calibration_receipt": inputs.calibration_receipt_sha256,
        "ablation_receipt": inputs.ablation_receipt_sha256,
        "postfreeze_receipt": inputs.postfreeze_receipt_sha256,
        "qualification_receipt": inputs.qualification_receipt_sha256,
        "verification_receipt": inputs.verification_receipt_sha256,
    }
    files: dict[str, Any] = {}
    for role, attribute in _RELEASE_FILE_ROLES.items():
        snapshot = _snapshot_file(
            Path(getattr(inputs, attribute)),
            label=f"raw release authority {role}",
            expected_sha256=expected.get(role),
        )
        files[role] = snapshot.descriptor()
    return {
        "files": files,
        "dataset_dir": str(Path(inputs.dataset_dir).resolve(strict=True)),
        "base_model_dir": str(Path(inputs.base_model_dir).resolve(strict=True)),
        "selected_adapter_dir": str(
            Path(inputs.selected_adapter_dir).resolve(strict=True)
        ),
    }


def _restore_release_authority_inputs(
    value: Mapping[str, Any],
) -> gguf_release_v8.ReleaseAuthorityInputsV8:
    record = _exact(
        value,
        {"files", "dataset_dir", "base_model_dir", "selected_adapter_dir"},
        label="raw release authority inputs",
    )
    files = _exact(
        record["files"],
        set(_RELEASE_FILE_ROLES),
        label="raw release authority files",
    )
    verified: dict[str, gguf_release_v8.FileSnapshotV8] = {}
    for role in _RELEASE_FILE_ROLES:
        descriptor = _exact(
            files[role], {"path", "bytes", "sha256"}, label=f"raw authority {role}"
        )
        snapshot = _snapshot_file(
            Path(str(descriptor["path"])),
            label=f"raw authority {role}",
            expected_sha256=str(descriptor["sha256"]),
        )
        if snapshot.descriptor() != descriptor:
            raise ObservationProducerV8Error(f"raw authority {role} descriptor changed")
        verified[role] = snapshot
    try:
        dataset = Path(str(record["dataset_dir"])).resolve(strict=True)
        base = Path(str(record["base_model_dir"])).resolve(strict=True)
        adapter = Path(str(record["selected_adapter_dir"])).resolve(strict=True)
    except OSError as exc:
        raise ObservationProducerV8Error("raw authority directories are unavailable") from exc
    if not dataset.is_dir() or not base.is_dir() or not adapter.is_dir():
        raise ObservationProducerV8Error("raw authority directories are invalid")
    return gguf_release_v8.ReleaseAuthorityInputsV8(
        selection_freeze=verified["selection_freeze"].path,
        selection_freeze_sha256=verified["selection_freeze"].sha256,
        evaluation_index=verified["evaluation_index"].path,
        training_receipt=verified["training_receipt"].path,
        dataset_dir=dataset,
        base_model_dir=base,
        selected_adapter_dir=adapter,
        calibration_receipt=verified["calibration_receipt"].path,
        calibration_receipt_sha256=verified["calibration_receipt"].sha256,
        ablation_receipt=verified["ablation_receipt"].path,
        ablation_receipt_sha256=verified["ablation_receipt"].sha256,
        postfreeze_receipt=verified["postfreeze_receipt"].path,
        postfreeze_receipt_sha256=verified["postfreeze_receipt"].sha256,
        qualification_receipt=verified["qualification_receipt"].path,
        qualification_receipt_sha256=verified["qualification_receipt"].sha256,
        verification_receipt=verified["verification_receipt"].path,
        verification_receipt_sha256=verified["verification_receipt"].sha256,
    )


def _source_inventory(cli_runner_path: Path) -> dict[str, Any]:
    expected = EXPECTED_CLI_PATH.resolve(strict=True)
    supplied = Path(cli_runner_path).resolve(strict=True)
    if supplied != expected:
        raise ObservationProducerV8Error("producer CLI path is not the fixed v8 runner")
    roles = {
        "producer_module": Path(__file__).resolve(),
        "producer_cli": supplied,
        "hf_runner_source": Path(pointer_hf_eval_v6.__file__).resolve(),
        "llama_runner_source": Path(llama_cpp_eval_v5.__file__).resolve(),
    }
    return {
        role: _snapshot_file(path, label=role).descriptor()
        for role, path in roles.items()
    }


def _hf_command(
    *,
    python_executable: Path,
    cli_runner: Path,
    request_path: Path,
    request_sha256: str,
    model: Mapping[str, Any],
) -> tuple[str, ...]:
    return (
        str(Path(python_executable).resolve(strict=True)),
        str(Path(cli_runner).resolve(strict=True)),
        "_hf-worker",
        "--requests",
        str(request_path.resolve(strict=True)),
        "--requests-sha256",
        request_sha256,
        "--base-model",
        str(model["base_path"]),
        "--selected-adapter",
        str(model["checkpoint_path"]),
        "--base-model-tree-sha256",
        str(model["base_model_tree_sha256"]),
        "--checkpoint-tree-sha256",
        str(model["checkpoint_tree_sha256"]),
        "--adapter-tree-sha256",
        str(model["adapter_tree_sha256"]),
    )


def _offline_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.lower()
        not in {"http_proxy", "https_proxy", "all_proxy", "ftp_proxy", "no_proxy"}
    }
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    return environment


def _run_hf_subprocess(
    *,
    command: tuple[str, ...],
) -> ExecutionCaptureV8:
    completed = subprocess.run(
        list(command),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        cwd=WORKSPACE_ROOT,
        env=_offline_environment(),
        check=False,
        timeout=4 * 60 * 60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        raise ObservationProducerV8Error(
            f"HF worker exited with status {completed.returncode}"
        )
    try:
        document = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObservationProducerV8Error("HF worker stdout is not one JSON receipt") from exc
    return ExecutionCaptureV8(
        document=document,
        stdout=bytes(completed.stdout),
        stderr=bytes(completed.stderr),
        returncode=int(completed.returncode),
        command=command,
        trace={"transport": "subprocess", "completed": True},
    )


def _finish_category(finish_reason: str) -> str:
    lowered = finish_reason.casefold()
    if lowered in {"length", "max_tokens", "token_limit", "length_limit"}:
        return "LENGTH_LIMIT"
    if lowered in {"stop", "eos_token", "end_turn"}:
        return "TRUSTED_STOP"
    return "ABNORMAL"


def _run_gguf_server(
    *,
    requests: Sequence[Mapping[str, Any]],
    request_sha256: str,
    model: gguf_release_v8.BinaryFileSnapshotV8,
    llama_server: gguf_release_v8.BinaryFileSnapshotV8,
    server_factory: Callable[
        [llama_cpp_eval_v5.ServerLaunchSpec], llama_cpp_eval_v5.ServerSession
    ] = llama_cpp_eval_v5.LocalLlamaServer,
) -> ExecutionCaptureV8:
    with tempfile.TemporaryDirectory(prefix="icmat-v8-parity-llama-") as temporary:
        log_dir = Path(temporary) / "server"
        spec = llama_cpp_eval_v5.ServerLaunchSpec(
            executable=llama_server.path,
            model=model.path,
            runtime_dir=llama_server.path.parent,
            log_dir=log_dir,
            threads=FIXED_THREADS,
            context_size=FIXED_CONTEXT_SIZE,
            gpu_layers=0,
            startup_timeout_seconds=120.0,
            request_timeout_seconds=120.0,
            seed=FIXED_SEED,
        )
        server = server_factory(spec)
        samples: list[dict[str, Any]] = []
        started = False
        try:
            server.start()
            started = True
            for request in requests:
                sample_started = time.perf_counter()
                response = server.chat(
                    {
                        "messages": [dict(message) for message in request["messages"]],
                        "temperature": 0,
                        "max_tokens": FIXED_MAX_NEW_TOKENS,
                        "seed": FIXED_SEED,
                        "stream": False,
                    }
                )
                latency_ms = (time.perf_counter() - sample_started) * 1000.0
                raw, trace = llama_cpp_eval_v5._extract_generation(response)
                finish_reason = str(trace.get("finish_reason") or "abnormal_end")
                usage = trace.get("usage")
                samples.append(
                    {
                        "schema": RAW_SAMPLE_SCHEMA,
                        "example_id": request["example_id"],
                        "raw_pointer": raw,
                        "finish_reason": finish_reason,
                        "finish_category": _finish_category(finish_reason),
                        "latency_ms": latency_ms,
                        "peak_rss_bytes": 0,
                        "input_tokens": (
                            int(usage["prompt_tokens"])
                            if isinstance(usage, Mapping)
                            and isinstance(usage.get("prompt_tokens"), int)
                            else None
                        ),
                        "output_tokens": (
                            int(usage["completion_tokens"])
                            if isinstance(usage, Mapping)
                            and isinstance(usage.get("completion_tokens"), int)
                            else None
                        ),
                        "generation_error": None,
                    }
                )
        finally:
            server.close()
        trace = dict(server.trace_metadata())
        if not started or not server.port_released():
            raise ObservationProducerV8Error("llama-server did not close and release loopback")
        returncode = trace.get("returncode")
        command = trace.get("command")
        if not isinstance(returncode, int) or not isinstance(command, list) or not command:
            raise ObservationProducerV8Error("llama-server process trace is incomplete")
        stdout_path = log_dir / "stdout.log"
        stderr_path = log_dir / "stderr.log"
        stdout = stdout_path.read_bytes() if stdout_path.exists() else b""
        stderr = stderr_path.read_bytes() if stderr_path.exists() else b""
        document = _raw_results_document(
            kind="GGUF_Q4_K_M",
            status=GGUF_RUN_STATUS,
            request_sha256=request_sha256,
            model={
                "filename": model.path.name,
                "bytes": model.bytes,
                "sha256": model.sha256,
                "format": "GGUF",
                "architecture": "qwen2",
                "quantization": "Q4_K_M",
            },
            backend={
                "engine": "llama.cpp-server",
                "engine_version": llama_cpp_eval_v5.EVALUATOR_VERSION,
                "device": "LOCAL_PC_CPU",
                "runtime_artifact_sha256": llama_server.sha256,
                "loopback_only": True,
                "external_network_used": False,
            },
            samples=samples,
        )
        return ExecutionCaptureV8(
            document=document,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            command=tuple(str(item) for item in command),
            trace=trace,
        )


def _raw_results_document(
    *,
    kind: str,
    status: str,
    request_sha256: str,
    model: Mapping[str, Any],
    backend: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    body = {
        "schema": RAW_RESULTS_SCHEMA,
        "version": VERSION,
        "status": status,
        "kind": kind,
        "provenance_kind": RUNTIME_PROVENANCE,
        "fixture_not_model_evidence": False,
        "model_invoked": True,
        "request_set_sha256": request_sha256,
        "generation_policy": generation_policy_v8(),
        "model": dict(model),
        "backend": dict(backend),
        "samples": [dict(sample) for sample in samples],
    }
    return {**body, "canonical_digest_sha256": _canonical_sha(body)}


def run_hf_worker_v8(
    *,
    requests_path: Path,
    requests_sha256: str,
    base_model_dir: Path,
    selected_adapter_dir: Path,
    base_model_tree_sha256: str,
    checkpoint_tree_sha256: str,
    adapter_tree_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    """Run the selected HF adapter once; used only by the private child CLI."""

    request_snapshot = _snapshot_file(
        requests_path,
        label="HF target-free request set",
        expected_sha256=requests_sha256,
    )
    requests = _parse_request_payload(request_snapshot.payload)
    trees = _model_bindings_from_paths(
        base_model_dir=base_model_dir,
        selected_adapter_dir=selected_adapter_dir,
        chain={
            "base_model_tree_sha256": base_model_tree_sha256,
            "checkpoint_tree_sha256": checkpoint_tree_sha256,
            "adapter_tree_sha256": adapter_tree_sha256,
        },
    )
    generation_requests = tuple(
        pointer_hf_eval_v6.GenerationRequestV6(
            example_id=str(request["example_id"]),
            messages=(dict(request["messages"][0]), dict(request["messages"][1])),
        )
        for request in requests
    )
    model_stdout = io.StringIO()
    model_stderr = io.StringIO()
    with contextlib.redirect_stdout(model_stdout), contextlib.redirect_stderr(
        model_stderr
    ):
        generations, backend = pointer_hf_eval_v6.generate_hf_model(
            generation_requests,
            base_model_dir=Path(base_model_dir),
            adapter_dir=Path(selected_adapter_dir),
            device="cpu",
            seed=FIXED_SEED,
        )
    if set(generations) != {str(request["example_id"]) for request in requests}:
        raise ObservationProducerV8Error("HF worker generation membership mismatch")
    samples = [
        {
            "schema": RAW_SAMPLE_SCHEMA,
            "example_id": str(request["example_id"]),
            "raw_pointer": generations[str(request["example_id"])].raw_pointer,
            "finish_reason": generations[str(request["example_id"])].finish_reason,
            "finish_category": _finish_category(
                generations[str(request["example_id"])].finish_reason
            ),
            "latency_ms": generations[str(request["example_id"])].latency_ms,
            "peak_rss_bytes": 0,
            "input_tokens": generations[str(request["example_id"])].input_tokens,
            "output_tokens": generations[str(request["example_id"])].output_tokens,
            "generation_error": generations[
                str(request["example_id"])
            ].generation_error,
        }
        for request in requests
    ]
    document = _raw_results_document(
        kind="HF_SELECTED_ADAPTER",
        status=HF_WORKER_STATUS,
        request_sha256=request_snapshot.sha256,
        model={
            "base_model_tree_sha256": trees["base_model_tree_sha256"],
            "checkpoint_tree_sha256": trees["checkpoint_tree_sha256"],
            "adapter_tree_sha256": trees["adapter_tree_sha256"],
        },
        backend={
            "engine": "transformers_peft",
            "engine_version": pointer_hf_eval_v6.EVALUATOR_VERSION,
            "device": "LOCAL_PC_CPU",
            "runtime_artifact_sha256": None,
            "local_files_only": backend.get("local_files_only"),
            "assistant_target_visible": backend.get("assistant_target_visible"),
        },
        samples=samples,
    )
    diagnostics = (model_stdout.getvalue() + model_stderr.getvalue()).encode(
        "utf-8", errors="replace"
    )
    return document, diagnostics


def _validate_raw_results(
    document: Mapping[str, Any],
    *,
    kind: str,
    request_sha256: str,
    request_ids: Sequence[str],
    model: Mapping[str, Any],
    runtime_artifact_sha256: str | None,
) -> dict[str, Mapping[str, Any]]:
    _exact(document, _RAW_RESULT_KEYS, label=f"{kind} raw results")
    _verify_digest(document, label=f"{kind} raw results")
    expected_status = HF_WORKER_STATUS if kind == "HF_SELECTED_ADAPTER" else GGUF_RUN_STATUS
    if (
        document["schema"] != RAW_RESULTS_SCHEMA
        or document["version"] != VERSION
        or document["status"] != expected_status
        or document["kind"] != kind
        or document["provenance_kind"] != RUNTIME_PROVENANCE
        or document["fixture_not_model_evidence"] is not False
        or document["model_invoked"] is not True
        or document["request_set_sha256"] != request_sha256
        or document["generation_policy"] != generation_policy_v8()
        or document["model"] != model
    ):
        raise ObservationProducerV8Error(f"{kind} raw result authority mismatch")
    backend = document["backend"]
    if kind == "HF_SELECTED_ADAPTER":
        _exact(
            backend,
            {
                "engine",
                "engine_version",
                "device",
                "runtime_artifact_sha256",
                "local_files_only",
                "assistant_target_visible",
            },
            label="HF raw backend",
        )
        if (
            backend["engine"] != "transformers_peft"
            or backend["engine_version"] != pointer_hf_eval_v6.EVALUATOR_VERSION
            or backend["device"] != "LOCAL_PC_CPU"
            or backend["runtime_artifact_sha256"] is not None
            or backend["local_files_only"] is not True
            or backend["assistant_target_visible"] is not False
        ):
            raise ObservationProducerV8Error("HF raw backend authority mismatch")
    else:
        _exact(
            backend,
            {
                "engine",
                "engine_version",
                "device",
                "runtime_artifact_sha256",
                "loopback_only",
                "external_network_used",
            },
            label="GGUF raw backend",
        )
        if (
            backend["engine"] != "llama.cpp-server"
            or backend["engine_version"] != llama_cpp_eval_v5.EVALUATOR_VERSION
            or backend["device"] != "LOCAL_PC_CPU"
            or backend["runtime_artifact_sha256"] != runtime_artifact_sha256
            or backend["loopback_only"] is not True
            or backend["external_network_used"] is not False
        ):
            raise ObservationProducerV8Error("GGUF raw backend authority mismatch")
    samples = document["samples"]
    if not isinstance(samples, list) or len(samples) != EXPECTED_ROWS:
        raise ObservationProducerV8Error(f"{kind} raw result membership mismatch")
    by_id: dict[str, Mapping[str, Any]] = {}
    observed_order: list[str] = []
    for index, sample in enumerate(samples):
        _exact(sample, _RAW_SAMPLE_KEYS, label=f"{kind} raw sample {index}")
        if sample["schema"] != RAW_SAMPLE_SCHEMA:
            raise ObservationProducerV8Error(f"{kind} raw sample schema mismatch")
        example_id = sample["example_id"]
        if not isinstance(example_id, str) or not example_id or example_id in by_id:
            raise ObservationProducerV8Error(f"{kind} raw sample ID mismatch")
        if not isinstance(sample["raw_pointer"], str):
            raise ObservationProducerV8Error(f"{kind} raw pointer is invalid")
        if not isinstance(sample["finish_reason"], str) or not sample["finish_reason"]:
            raise ObservationProducerV8Error(f"{kind} finish reason is invalid")
        if sample["finish_category"] not in {
            "TRUSTED_STOP",
            "LENGTH_LIMIT",
            "ABNORMAL",
        }:
            raise ObservationProducerV8Error(f"{kind} finish category is invalid")
        latency = sample["latency_ms"]
        rss = sample["peak_rss_bytes"]
        if (
            not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or not math.isfinite(float(latency))
            or float(latency) < 0
            or not isinstance(rss, int)
            or isinstance(rss, bool)
            or rss < 0
        ):
            raise ObservationProducerV8Error(f"{kind} runtime measurements invalid")
        for token_field in ("input_tokens", "output_tokens"):
            token_count = sample[token_field]
            if token_count is not None and (
                not isinstance(token_count, int)
                or isinstance(token_count, bool)
                or token_count < 0
            ):
                raise ObservationProducerV8Error(
                    f"{kind} {token_field} is invalid"
                )
        generation_error = sample["generation_error"]
        if generation_error is not None and not isinstance(generation_error, str):
            raise ObservationProducerV8Error(
                f"{kind} generation_error is invalid"
            )
        if sample["finish_category"] != _finish_category(sample["finish_reason"]):
            raise ObservationProducerV8Error(f"{kind} finish category mismatch")
        observed_order.append(example_id)
        by_id[example_id] = sample
    if observed_order != list(request_ids):
        raise ObservationProducerV8Error(f"{kind} raw result order mismatch")
    return by_id


def _artifact(path: Path) -> dict[str, Any]:
    snapshot = _snapshot_file(path, label=path.name)
    return snapshot.descriptor()


def _execution_record(
    *,
    kind: str,
    capture: ExecutionCaptureV8,
    program: Mapping[str, Any],
    runner_source: Mapping[str, Any],
    request_sha256: str,
    stdout: Mapping[str, Any],
    stderr: Mapping[str, Any],
    raw_results: Mapping[str, Any],
    loopback_http_used: bool,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "process_started": True,
        "model_invoked": True,
        "fixture_not_model_evidence": False,
        "request_set_sha256": request_sha256,
        "generation_policy": generation_policy_v8(),
        "program": dict(program),
        "runner_source": dict(runner_source),
        "command": list(capture.command),
        "command_sha256": _canonical_sha(list(capture.command)),
        "returncode": capture.returncode,
        "stdout": dict(stdout),
        "stderr": dict(stderr),
        "raw_results": dict(raw_results),
        "trace": json.loads(gguf_release_v8.canonical_json(capture.trace)),
        "external_network_used": False,
        "loopback_http_used": loopback_http_used,
        "production_services_touched": False,
        "x5_contacted": False,
    }


def _observation_document(
    *,
    kind: str,
    records: Sequence[Any],
    dataset: Mapping[str, Any],
    preflight_snapshot: gguf_release_v8.FileSnapshotV8,
    preflight: Mapping[str, Any],
    authority_snapshot: gguf_release_v8.FileSnapshotV8,
    authority: Mapping[str, Any],
    raw_results_sha256: str,
    samples: Mapping[str, Mapping[str, Any]],
    gguf_model: gguf_release_v8.BinaryFileSnapshotV8,
) -> dict[str, Any]:
    if kind == "HF_SELECTED_ADAPTER":
        model = {
            "base_model_tree_sha256": preflight["chain_binding"][
                "base_model_tree_sha256"
            ],
            "checkpoint_tree_sha256": preflight["chain_binding"][
                "checkpoint_tree_sha256"
            ],
            "adapter_tree_sha256": preflight["chain_binding"][
                "adapter_tree_sha256"
            ],
        }
        engine = "transformers_peft"
        engine_version = pointer_hf_eval_v6.EVALUATOR_VERSION
        runtime_sha = None
    else:
        model = {
            "filename": gguf_model.path.name,
            "bytes": gguf_model.bytes,
            "sha256": gguf_model.sha256,
            "format": "GGUF",
            "architecture": "qwen2",
            "quantization": "Q4_K_M",
        }
        engine = "llama.cpp-server"
        engine_version = llama_cpp_eval_v5.EVALUATOR_VERSION
        runtime_sha = preflight["tool_binding"]["llama_server_sha256"]
    rows = []
    for record in records:
        raw = samples[str(record.example_id)]
        rows.append(
            {
                "schema": "icmat_pointer_runtime_observation.v8",
                "example_id": record.example_id,
                "prompt_sha256": record.prompt_sha256,
                "expected_pointer_sha256": record.expected_pointer_sha256,
                "raw_pointer": raw["raw_pointer"],
                "finish_reason": raw["finish_reason"],
                "generation_error": raw["generation_error"],
                "truncated": raw["finish_category"] == "LENGTH_LIMIT",
                "latency_ms": raw["latency_ms"],
                "peak_rss_bytes": raw["peak_rss_bytes"],
            }
        )
    body = {
        "schema": "icmat_pointer_runtime_observations.v8",
        "version": "icmat-hf-gguf-pointer-parity-v8.1.0",
        "status": "COMPLETE_NONBLIND_V8_VALIDATION_OBSERVATIONS",
        "backend": {
            "kind": kind,
            "engine": engine,
            "engine_version": engine_version,
            "device": "LOCAL_PC_CPU",
            "model": model,
            "runtime_artifact_sha256": runtime_sha,
        },
        "preflight": {
            "sha256": preflight_snapshot.sha256,
            "authorization_digest_sha256": preflight[
                "authorization_digest_sha256"
            ],
        },
        "dataset": {
            "manifest_sha256": dataset["manifest"]["sha256"],
            "split": "validation",
            "split_sha256": dataset["validation"]["sha256"],
            "samples": EXPECTED_ROWS,
            "example_id_order_sha256": dataset["example_id_order_sha256"],
        },
        "generation_policy": generation_policy_v8(),
        "samples": rows,
        "runtime_authority": {
            "schema": AUTHORITY_SCHEMA,
            "status": AUTHORITY_STATUS,
            **authority_snapshot.descriptor(),
            "canonical_digest_sha256": authority["canonical_digest_sha256"],
            "provenance_kind": RUNTIME_PROVENANCE,
            "execution_role": kind,
            "raw_results_sha256": raw_results_sha256,
        },
        "execution_boundary": {
            "model_invoked": True,
            "network_used": False,
            "reserved_blind_read": False,
            "x5_contacted": False,
            "production_services_touched": False,
        },
    }
    return {**body, "canonical_digest_sha256": _canonical_sha(body)}


def produce_runtime_observations_v8(inputs: ProducerInputsV8) -> dict[str, Any]:
    """Run both local backends and publish one controlled candidate bundle."""

    from icmat_foundry.llm import hf_gguf_parity_v8 as parity

    output = Path(inputs.output_dir).expanduser().absolute()
    if os.path.lexists(output):
        raise ObservationProducerV8Error("observation output directory already exists")
    output.parent.resolve(strict=True)
    preflight_snapshot, preflight = parity._load_preflight(
        inputs.preflight_receipt,
        expected_sha256=inputs.preflight_receipt_sha256,
    )
    records, dataset = parity._validation_records(
        inputs.release_authority.dataset_dir,
        preflight=preflight,
    )
    export_snapshot, model_snapshot, _ = parity._validate_low_level_export(
        receipt_path=inputs.export_receipt,
        receipt_sha256=inputs.export_receipt_sha256,
        gguf_model=inputs.gguf_model,
        gguf_model_sha256=inputs.gguf_model_sha256,
        preflight=preflight,
    )
    try:
        release_authority = gguf_release_v8.validate_authority_chain_v8(
            inputs.release_authority
        )
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise ObservationProducerV8Error("raw v8 release authority rejected") from exc
    if (
        release_authority["chain_binding"] != preflight["chain_binding"]
        or release_authority["authority_digest_sha256"]
        != preflight["authority_digest_sha256"]
        or release_authority["receipts"] != preflight["authority_receipts"]
    ):
        raise ObservationProducerV8Error("preflight differs from raw release authority")
    model_bindings = _model_bindings(
        inputs.release_authority,
        chain=preflight["chain_binding"],
    )
    llama_snapshot = _snapshot_binary(
        inputs.llama_server,
        label="pinned llama-server",
        expected_sha256=inputs.llama_server_sha256,
        maximum_bytes=1024 * 1024 * 1024,
    )
    if llama_snapshot.sha256 != preflight["tool_binding"]["llama_server_sha256"]:
        raise ObservationProducerV8Error("llama-server differs from preflight")
    python_snapshot = _snapshot_binary(
        inputs.python_executable,
        label="HF Python executable",
        maximum_bytes=1024 * 1024 * 1024,
    )
    implementation = _source_inventory(inputs.cli_runner_path)
    release_inputs = _capture_release_authority_inputs(inputs.release_authority)
    request_payload = request_payload_v8(records)
    request_sha = hashlib.sha256(request_payload).hexdigest()
    requests = _parse_request_payload(request_payload)

    os.mkdir(output)
    try:
        request_path = _write_exclusive(output / REQUEST_FILENAME, request_payload)
        request_record = _artifact(request_path)
        request_record.update(
            {
                "records": EXPECTED_ROWS,
                "example_id_order_sha256": _canonical_sha(
                    [request["example_id"] for request in requests]
                ),
                "target_free": True,
                "assistant_target_present": False,
            }
        )
        hf_command = _hf_command(
            python_executable=python_snapshot.path,
            cli_runner=Path(inputs.cli_runner_path),
            request_path=request_path,
            request_sha256=request_sha,
            model=model_bindings,
        )
        hf_capture = _run_hf_subprocess(command=hf_command)
        hf_stdout = _write_exclusive(output / HF_STDOUT_FILENAME, hf_capture.stdout)
        hf_stderr = _write_exclusive(output / HF_STDERR_FILENAME, hf_capture.stderr)
        hf_model = {
            "base_model_tree_sha256": model_bindings["base_model_tree_sha256"],
            "checkpoint_tree_sha256": model_bindings["checkpoint_tree_sha256"],
            "adapter_tree_sha256": model_bindings["adapter_tree_sha256"],
        }
        hf_samples = _validate_raw_results(
            hf_capture.document,
            kind="HF_SELECTED_ADAPTER",
            request_sha256=request_sha,
            request_ids=[request["example_id"] for request in requests],
            model=hf_model,
            runtime_artifact_sha256=None,
        )
        if _json_bytes(hf_capture.document) != hf_capture.stdout:
            raise ObservationProducerV8Error(
                "HF stdout is not the canonical raw result receipt"
            )

        gguf_capture = _run_gguf_server(
            requests=requests,
            request_sha256=request_sha,
            model=model_snapshot,
            llama_server=llama_snapshot,
        )
        gguf_stdout = _write_exclusive(
            output / GGUF_STDOUT_FILENAME, gguf_capture.stdout
        )
        gguf_stderr = _write_exclusive(
            output / GGUF_STDERR_FILENAME, gguf_capture.stderr
        )
        gguf_result_path = _write_exclusive(
            output / GGUF_RESULTS_FILENAME, _json_bytes(gguf_capture.document)
        )
        gguf_model = {
            "filename": model_snapshot.path.name,
            "bytes": model_snapshot.bytes,
            "sha256": model_snapshot.sha256,
            "format": "GGUF",
            "architecture": "qwen2",
            "quantization": "Q4_K_M",
        }
        gguf_samples = _validate_raw_results(
            gguf_capture.document,
            kind="GGUF_Q4_K_M",
            request_sha256=request_sha,
            request_ids=[request["example_id"] for request in requests],
            model=gguf_model,
            runtime_artifact_sha256=llama_snapshot.sha256,
        )

        authority_core = {
            "schema": AUTHORITY_SCHEMA,
            "version": VERSION,
            "status": AUTHORITY_STATUS,
            "created_at_utc": _utc_now(),
            "provenance_kind": RUNTIME_PROVENANCE,
            "preflight": {
                **preflight_snapshot.descriptor(),
                "authorization_digest_sha256": preflight[
                    "authorization_digest_sha256"
                ],
                "authority_digest_sha256": preflight["authority_digest_sha256"],
            },
            "release_authority_inputs": release_inputs,
            "release_authority": {
                "chain_binding": release_authority["chain_binding"],
                "receipts": release_authority["receipts"],
                "authority_digest_sha256": release_authority[
                    "authority_digest_sha256"
                ],
            },
            "low_level_export": {
                "receipt": export_snapshot.descriptor(),
                "gguf_model": model_snapshot.descriptor(),
            },
            "request_set": request_record,
            "generation_policy": generation_policy_v8(),
            "implementation": implementation,
            "model_bindings": {
                "hf_selected_adapter": model_bindings,
                "gguf_q4_k_m": gguf_model,
            },
            "executions": {
                "hf_selected_adapter": _execution_record(
                    kind="HF_SELECTED_ADAPTER",
                    capture=hf_capture,
                    program=python_snapshot.descriptor(),
                    runner_source=implementation["producer_cli"],
                    request_sha256=request_sha,
                    stdout=_artifact(hf_stdout),
                    stderr=_artifact(hf_stderr),
                    raw_results=_artifact(hf_stdout),
                    loopback_http_used=False,
                ),
                "gguf_q4_k_m": _execution_record(
                    kind="GGUF_Q4_K_M",
                    capture=gguf_capture,
                    program=llama_snapshot.descriptor(),
                    runner_source=implementation["llama_runner_source"],
                    request_sha256=request_sha,
                    stdout=_artifact(gguf_stdout),
                    stderr=_artifact(gguf_stderr),
                    raw_results=_artifact(gguf_result_path),
                    loopback_http_used=True,
                ),
            },
            "execution_boundary": {
                "both_models_invoked": True,
                "fixture_observations_used": False,
                "external_network_used": False,
                "loopback_http_used_for_llama_server": True,
                "reserved_blind_read": False,
                "x5_contacted": False,
                "production_services_touched": False,
                "production_registry_created": False,
            },
            "claim_boundary": (
                "Controlled local-PC nonblind runtime evidence only. This receipt "
                "does not authorize X5 execution, deployment, service registration, "
                "or production integration."
            ),
        }
        authority = {
            **authority_core,
            "canonical_digest_sha256": _canonical_sha(authority_core),
        }
        authority_path = _write_exclusive(
            output / AUTHORITY_FILENAME, _json_bytes(authority)
        )
        authority_snapshot = _snapshot_file(
            authority_path, label="published runtime authority"
        )
        hf_observations = _observation_document(
            kind="HF_SELECTED_ADAPTER",
            records=records,
            dataset=dataset,
            preflight_snapshot=preflight_snapshot,
            preflight=preflight,
            authority_snapshot=authority_snapshot,
            authority=authority,
            raw_results_sha256=_artifact(hf_stdout)["sha256"],
            samples=hf_samples,
            gguf_model=model_snapshot,
        )
        gguf_observations = _observation_document(
            kind="GGUF_Q4_K_M",
            records=records,
            dataset=dataset,
            preflight_snapshot=preflight_snapshot,
            preflight=preflight,
            authority_snapshot=authority_snapshot,
            authority=authority,
            raw_results_sha256=_artifact(gguf_result_path)["sha256"],
            samples=gguf_samples,
            gguf_model=model_snapshot,
        )
        hf_path = _write_exclusive(
            output / HF_OBSERVATIONS_FILENAME, _json_bytes(hf_observations)
        )
        gguf_path = _write_exclusive(
            output / GGUF_OBSERVATIONS_FILENAME, _json_bytes(gguf_observations)
        )
    except BaseException:
        # No authority is usable unless the complete receipt-last bundle exists.
        shutil.rmtree(output, ignore_errors=True)
        raise
    return {
        "status": AUTHORITY_STATUS,
        "output_dir": str(output.resolve(strict=True)),
        "runtime_authority": _artifact(authority_path),
        "hf_observations": _artifact(hf_path),
        "gguf_observations": _artifact(gguf_path),
        "fixture_observations_used": False,
        "x5_contacted": False,
        "production_services_touched": False,
    }


def _verify_source_inventory(value: Mapping[str, Any]) -> None:
    expected_paths = {
        "producer_module": Path(__file__).resolve(),
        "producer_cli": EXPECTED_CLI_PATH.resolve(strict=True),
        "hf_runner_source": Path(pointer_hf_eval_v6.__file__).resolve(),
        "llama_runner_source": Path(llama_cpp_eval_v5.__file__).resolve(),
    }
    inventory = _exact(value, set(expected_paths), label="runtime implementation")
    for role, expected_path in expected_paths.items():
        descriptor = _exact(
            inventory[role], {"path", "bytes", "sha256"}, label=role
        )
        if Path(str(descriptor["path"])).resolve(strict=True) != expected_path:
            raise ObservationProducerV8Error(f"{role} path differs from fixed source")
        snapshot = _snapshot_file(
            expected_path,
            label=role,
            expected_sha256=str(descriptor["sha256"]),
        )
        if snapshot.descriptor() != descriptor:
            raise ObservationProducerV8Error(f"{role} source changed")


def _load_artifact_descriptor(
    value: Any,
    *,
    label: str,
    binary: bool = False,
) -> gguf_release_v8.FileSnapshotV8 | gguf_release_v8.BinaryFileSnapshotV8:
    descriptor = _exact(value, {"path", "bytes", "sha256"}, label=label)
    if binary:
        snapshot = _snapshot_binary(
            Path(str(descriptor["path"])),
            label=label,
            expected_sha256=str(descriptor["sha256"]),
            maximum_bytes=1024 * 1024 * 1024,
        )
    else:
        snapshot = _snapshot_file(
            Path(str(descriptor["path"])),
            label=label,
            expected_sha256=str(descriptor["sha256"]),
        )
    if snapshot.descriptor() != descriptor:
        raise ObservationProducerV8Error(f"{label} descriptor changed")
    return snapshot


def _validate_gguf_command(
    command: Sequence[Any],
    *,
    executable: Path,
    model: Path,
) -> None:
    values = [str(item) for item in command]
    if not values or Path(values[0]).resolve(strict=True) != executable:
        raise ObservationProducerV8Error("GGUF launch command executable mismatch")
    required_pairs = {
        "-m": str(model),
        "-ngl": "0",
        "-t": str(FIXED_THREADS),
        "-c": str(FIXED_CONTEXT_SIZE),
        "-np": "1",
        "--host": "127.0.0.1",
        "--alias": llama_cpp_eval_v5.MODEL_ALIAS,
        "--reasoning": "off",
    }
    for option, expected in required_pairs.items():
        try:
            index = values.index(option)
            observed = values[index + 1]
        except (ValueError, IndexError) as exc:
            raise ObservationProducerV8Error(
                f"GGUF launch command omitted {option}"
            ) from exc
        if option == "-m":
            if Path(observed).resolve(strict=True) != model:
                raise ObservationProducerV8Error("GGUF command model path mismatch")
        elif observed != expected:
            raise ObservationProducerV8Error(f"GGUF command changed {option}")
    if "--no-webui" not in values or "--api-key" not in values:
        raise ObservationProducerV8Error("GGUF launch command safety flags missing")
    key_index = values.index("--api-key")
    if key_index + 1 >= len(values) or values[key_index + 1] != "<redacted-api-key>":
        raise ObservationProducerV8Error("GGUF launch trace did not redact API key")


def verify_runtime_authority_v8(
    *,
    authority_path: Path,
    authority_sha256: str,
    preflight_snapshot: gguf_release_v8.FileSnapshotV8,
    preflight: Mapping[str, Any],
    export_snapshot: gguf_release_v8.FileSnapshotV8,
    gguf_model: gguf_release_v8.BinaryFileSnapshotV8,
    expected_request_payload: bytes,
) -> VerifiedRuntimeAuthorityV8:
    """Revalidate raw authority inputs and every captured process artifact."""

    try:
        snapshot, receipt = gguf_release_v8._load_json(
            authority_path,
            label="controlled runtime observation authority",
            expected_sha256=authority_sha256,
        )
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise ObservationProducerV8Error("runtime observation authority rejected") from exc
    _exact(receipt, _AUTHORITY_KEYS, label="runtime observation authority")
    _verify_digest(receipt, label="runtime observation authority")
    if (
        receipt["schema"] != AUTHORITY_SCHEMA
        or receipt["version"] != VERSION
        or receipt["status"] != AUTHORITY_STATUS
        or receipt["provenance_kind"] != RUNTIME_PROVENANCE
    ):
        raise ObservationProducerV8Error(
            "fixture or legacy observation authority cannot authorize release"
        )
    boundary = receipt["execution_boundary"]
    if boundary != {
        "both_models_invoked": True,
        "fixture_observations_used": False,
        "external_network_used": False,
        "loopback_http_used_for_llama_server": True,
        "reserved_blind_read": False,
        "x5_contacted": False,
        "production_services_touched": False,
        "production_registry_created": False,
    }:
        raise ObservationProducerV8Error("runtime authority crossed its boundary")
    bound_preflight = _exact(
        receipt["preflight"],
        {
            "path",
            "bytes",
            "sha256",
            "authorization_digest_sha256",
            "authority_digest_sha256",
        },
        label="runtime authority preflight",
    )
    if bound_preflight != {
        **preflight_snapshot.descriptor(),
        "authorization_digest_sha256": preflight["authorization_digest_sha256"],
        "authority_digest_sha256": preflight["authority_digest_sha256"],
    }:
        raise ObservationProducerV8Error("runtime authority uses another preflight")

    raw_inputs = _restore_release_authority_inputs(
        _exact(
            receipt["release_authority_inputs"],
            {"files", "dataset_dir", "base_model_dir", "selected_adapter_dir"},
            label="runtime raw release authority",
        )
    )
    try:
        revalidated = gguf_release_v8.validate_authority_chain_v8(raw_inputs)
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise ObservationProducerV8Error(
            "runtime raw release authority no longer validates"
        ) from exc
    expected_release = {
        "chain_binding": revalidated["chain_binding"],
        "receipts": revalidated["receipts"],
        "authority_digest_sha256": revalidated["authority_digest_sha256"],
    }
    if receipt["release_authority"] != expected_release or (
        revalidated["chain_binding"] != preflight["chain_binding"]
        or revalidated["receipts"] != preflight["authority_receipts"]
        or revalidated["authority_digest_sha256"]
        != preflight["authority_digest_sha256"]
    ):
        raise ObservationProducerV8Error("raw release authority differs from preflight")

    low_level = _exact(
        receipt["low_level_export"],
        {"receipt", "gguf_model"},
        label="runtime low-level export",
    )
    if low_level["receipt"] != export_snapshot.descriptor() or low_level[
        "gguf_model"
    ] != gguf_model.descriptor():
        raise ObservationProducerV8Error("runtime authority export binding mismatch")
    request_record = _exact(
        receipt["request_set"],
        {
            "path",
            "bytes",
            "sha256",
            "records",
            "example_id_order_sha256",
            "target_free",
            "assistant_target_present",
        },
        label="runtime request set",
    )
    request_snapshot = _load_artifact_descriptor(
        {key: request_record[key] for key in ("path", "bytes", "sha256")},
        label="runtime request set",
    )
    assert isinstance(request_snapshot, gguf_release_v8.FileSnapshotV8)
    if (
        request_snapshot.payload != expected_request_payload
        or request_record.get("records") != EXPECTED_ROWS
        or request_record.get("target_free") is not True
        or request_record.get("assistant_target_present") is not False
    ):
        raise ObservationProducerV8Error("runtime request set differs from frozen validation")
    requests = _parse_request_payload(request_snapshot.payload)
    request_ids = [str(request["example_id"]) for request in requests]
    if request_record.get("example_id_order_sha256") != _canonical_sha(request_ids):
        raise ObservationProducerV8Error("runtime request order digest mismatch")
    if receipt["generation_policy"] != generation_policy_v8():
        raise ObservationProducerV8Error("runtime generation policy changed")
    _verify_source_inventory(receipt["implementation"])

    model_bindings = _model_bindings(
        raw_inputs,
        chain=preflight["chain_binding"],
    )
    expected_gguf_model = {
        "filename": gguf_model.path.name,
        "bytes": gguf_model.bytes,
        "sha256": gguf_model.sha256,
        "format": "GGUF",
        "architecture": "qwen2",
        "quantization": "Q4_K_M",
    }
    if receipt["model_bindings"] != {
        "hf_selected_adapter": model_bindings,
        "gguf_q4_k_m": expected_gguf_model,
    }:
        raise ObservationProducerV8Error("runtime model bindings changed")

    executions = _exact(
        receipt["executions"],
        {"hf_selected_adapter", "gguf_q4_k_m"},
        label="runtime executions",
    )
    result_maps: dict[str, Mapping[str, Mapping[str, Any]]] = {}
    for role, kind, expected_model, loopback in (
        (
            "hf_selected_adapter",
            "HF_SELECTED_ADAPTER",
            {
                "base_model_tree_sha256": model_bindings[
                    "base_model_tree_sha256"
                ],
                "checkpoint_tree_sha256": model_bindings[
                    "checkpoint_tree_sha256"
                ],
                "adapter_tree_sha256": model_bindings["adapter_tree_sha256"],
            },
            False,
        ),
        ("gguf_q4_k_m", "GGUF_Q4_K_M", expected_gguf_model, True),
    ):
        execution = _exact(
            executions[role],
            {
                "kind",
                "process_started",
                "model_invoked",
                "fixture_not_model_evidence",
                "request_set_sha256",
                "generation_policy",
                "program",
                "runner_source",
                "command",
                "command_sha256",
                "returncode",
                "stdout",
                "stderr",
                "raw_results",
                "trace",
                "external_network_used",
                "loopback_http_used",
                "production_services_touched",
                "x5_contacted",
            },
            label=f"{kind} execution",
        )
        if (
            execution["kind"] != kind
            or execution["process_started"] is not True
            or execution["model_invoked"] is not True
            or execution["fixture_not_model_evidence"] is not False
            or execution["request_set_sha256"] != request_snapshot.sha256
            or execution["generation_policy"] != generation_policy_v8()
            or not isinstance(execution["returncode"], int)
            or execution["external_network_used"] is not False
            or execution["loopback_http_used"] is not loopback
            or execution["production_services_touched"] is not False
            or execution["x5_contacted"] is not False
        ):
            raise ObservationProducerV8Error(f"{kind} execution boundary mismatch")
        command = execution["command"]
        if not isinstance(command, list) or execution[
            "command_sha256"
        ] != _canonical_sha(command):
            raise ObservationProducerV8Error(f"{kind} command digest mismatch")
        program = _load_artifact_descriptor(
            execution["program"], label=f"{kind} program", binary=True
        )
        runner = _load_artifact_descriptor(
            execution["runner_source"], label=f"{kind} runner source"
        )
        stdout = _load_artifact_descriptor(
            execution["stdout"], label=f"{kind} raw stdout"
        )
        _load_artifact_descriptor(execution["stderr"], label=f"{kind} raw stderr")
        raw_results = _load_artifact_descriptor(
            execution["raw_results"], label=f"{kind} raw results"
        )
        assert isinstance(program, gguf_release_v8.BinaryFileSnapshotV8)
        assert isinstance(runner, gguf_release_v8.FileSnapshotV8)
        assert isinstance(stdout, gguf_release_v8.FileSnapshotV8)
        assert isinstance(raw_results, gguf_release_v8.FileSnapshotV8)
        if kind == "HF_SELECTED_ADAPTER":
            if (
                Path(runner.path) != EXPECTED_CLI_PATH.resolve(strict=True)
                or execution["returncode"] != 0
                or raw_results.descriptor() != stdout.descriptor()
            ):
                raise ObservationProducerV8Error("HF subprocess authority mismatch")
            expected_command = _hf_command(
                python_executable=program.path,
                cli_runner=runner.path,
                request_path=request_snapshot.path,
                request_sha256=request_snapshot.sha256,
                model=model_bindings,
            )
            if list(expected_command) != command:
                raise ObservationProducerV8Error("HF subprocess command changed")
        else:
            if (
                program.sha256 != preflight["tool_binding"]["llama_server_sha256"]
                or Path(runner.path)
                != Path(llama_cpp_eval_v5.__file__).resolve(strict=True)
            ):
                raise ObservationProducerV8Error("GGUF runner authority mismatch")
            _validate_gguf_command(
                command,
                executable=program.path,
                model=gguf_model.path,
            )
        try:
            document = json.loads(raw_results.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ObservationProducerV8Error(f"{kind} raw results are invalid") from exc
        result_maps[kind] = _validate_raw_results(
            document,
            kind=kind,
            request_sha256=request_snapshot.sha256,
            request_ids=request_ids,
            model=expected_model,
            runtime_artifact_sha256=(
                None
                if kind == "HF_SELECTED_ADAPTER"
                else preflight["tool_binding"]["llama_server_sha256"]
            ),
        )
    return VerifiedRuntimeAuthorityV8(
        snapshot=snapshot,
        receipt=receipt,
        results=result_maps,
    )


__all__ = [
    "AUTHORITY_FILENAME",
    "AUTHORITY_SCHEMA",
    "AUTHORITY_STATUS",
    "ExecutionCaptureV8",
    "FIXTURE_PROVENANCE",
    "GGUF_OBSERVATIONS_FILENAME",
    "HF_OBSERVATIONS_FILENAME",
    "ObservationProducerV8Error",
    "ProducerInputsV8",
    "RUNTIME_PROVENANCE",
    "VerifiedRuntimeAuthorityV8",
    "generation_policy_v8",
    "produce_runtime_observations_v8",
    "request_payload_v8",
    "run_hf_worker_v8",
    "verify_runtime_authority_v8",
]
