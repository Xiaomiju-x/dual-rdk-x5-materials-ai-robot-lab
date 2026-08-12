"""Integrity-bound llama.cpp GGUF evaluation for the ICMat v5 student."""

from __future__ import annotations

import json
import os
import platform
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from icmat_foundry.llm.evidence_eval_v5 import (
    ALLOWED_ABLATIONS,
    ALLOWED_SPLITS,
    EvidenceEvalV5Error,
    GenerationRequestV5,
    build_generation_requests,
    canonical_json,
    load_dataset_selection,
    score_generations,
    sha256_bytes,
    sha256_file,
)

EVALUATOR_VERSION = "icmat-llama-cpp-eval-v5.1.0"
SUMMARY_SCHEMA = "icmat_llama_cpp_evidence_eval_summary.v5"
RUN_RECEIPT_SCHEMA = "icmat_llama_cpp_evidence_eval_run_receipt.v5"
FAILURE_RECEIPT_SCHEMA = "icmat_llama_cpp_evidence_eval_failure_receipt.v5"
QUANTIZATION = "Q4_K_M"
MODEL_ALIAS = "icmat-qwen05b-q4-k-m"
SHA256_CHARS = frozenset("0123456789abcdef")
MIN_HIGH_PORT = 49152
MAX_PORT = 65535
MAX_HTTP_BYTES = 4 * 1024 * 1024


class LlamaCppEvalV5Error(EvidenceEvalV5Error):
    """Raised when the local GGUF evaluation contract is violated."""


@dataclass(frozen=True)
class LlamaCppEvalConfig:
    dataset_dir: Path
    split: str
    output_dir: Path
    gguf_model: Path
    llama_server: Path
    expected_gguf_sha256: str
    expected_llama_server_sha256: str
    ablations: tuple[str, ...] = ("none",)
    max_samples: int | None = None
    blind_authorization_path: Path | None = None
    blind_authorization_sha256: str | None = None
    threads: int = 4
    context_size: int = 1536
    max_tokens: int = 320
    seed: int = 20260729
    startup_timeout_seconds: float = 90.0
    request_timeout_seconds: float = 180.0
    gpu_layers: int = 0
    runner_path: Path | None = None


@dataclass(frozen=True)
class ServerLaunchSpec:
    executable: Path
    model: Path
    runtime_dir: Path
    log_dir: Path
    threads: int
    context_size: int
    gpu_layers: int
    startup_timeout_seconds: float
    request_timeout_seconds: float
    seed: int


class ServerSession(Protocol):
    port: int

    def start(self) -> None: ...

    def chat(self, body: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def close(self) -> None: ...

    def trace_metadata(self) -> Mapping[str, Any]: ...

    def port_released(self) -> bool: ...


ServerFactory = Callable[[ServerLaunchSpec], ServerSession]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join((canonical_json(dict(record)) + "\n").encode("utf-8") for record in records)


def _validate_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or not set(normalized) <= SHA256_CHARS:
        raise LlamaCppEvalV5Error(f"{label} expected SHA256 must be 64 lowercase hexadecimal characters")
    return normalized


def _regular_file(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise LlamaCppEvalV5Error(f"{label} must not be a symlink: {candidate}")
    try:
        mode = candidate.stat(follow_symlinks=False).st_mode
    except OSError as exc:
        raise LlamaCppEvalV5Error(f"{label} is unavailable: {candidate}") from exc
    if not stat.S_ISREG(mode):
        raise LlamaCppEvalV5Error(f"{label} must be a regular file: {candidate}")
    resolved = candidate.resolve(strict=True)
    expected = _validate_sha256(expected_sha256, label)
    actual = sha256_file(resolved)
    if actual != expected:
        raise LlamaCppEvalV5Error(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return resolved, {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": actual,
        "expected_sha256": expected,
        "sha256_match": True,
        "symlink": False,
        "regular_file": True,
    }


def _tree_inventory(root: Path) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise LlamaCppEvalV5Error(f"runtime root is not a directory: {resolved}")
    records: list[dict[str, Any]] = []
    for candidate in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink():
            raise LlamaCppEvalV5Error(f"llama.cpp runtime tree contains a symlink: {candidate}")
        try:
            mode = candidate.stat(follow_symlinks=False).st_mode
        except OSError as exc:
            raise LlamaCppEvalV5Error(f"cannot stat llama.cpp runtime entry: {candidate}") from exc
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise LlamaCppEvalV5Error(f"llama.cpp runtime entry is not a regular file: {candidate}")
        records.append(
            {
                "path": candidate.relative_to(resolved).as_posix(),
                "bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    if not records:
        raise LlamaCppEvalV5Error("llama.cpp runtime directory is empty")
    material = canonical_json(records).encode("utf-8")
    return {
        "path": str(resolved),
        "files": records,
        "file_count": len(records),
        "bytes": sum(int(item["bytes"]) for item in records),
        "tree_sha256": sha256_bytes(material),
        "symlinks_allowed": False,
    }


def _source_inventory(runner_path: Path | None) -> dict[str, Any]:
    paths = {
        "llama_cpp_evaluator": Path(__file__).resolve(),
        "shared_evidence_evaluator": (Path(__file__).resolve().with_name("evidence_eval_v5.py")),
    }
    if runner_path is not None:
        paths["runner"] = runner_path.resolve(strict=True)
    return {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }


def _selected_ablations(values: Sequence[str]) -> tuple[str, ...]:
    if not values:
        return ("none",)
    selected: list[str] = []
    for value in values:
        if value not in ALLOWED_ABLATIONS:
            raise LlamaCppEvalV5Error(f"unsupported ablation: {value}")
        if value not in selected:
            selected.append(value)
    return tuple(selected)


def _validate_limits(config: LlamaCppEvalConfig) -> None:
    if config.split not in ALLOWED_SPLITS:
        raise LlamaCppEvalV5Error(f"unsupported split: {config.split}")
    if config.split != "calibration":
        raise LlamaCppEvalV5Error(
            "GGUF evaluation is calibration-only; validation and blind_test are forbidden"
        )
    integer_limits = {
        "threads": config.threads,
        "context_size": config.context_size,
        "max_tokens": config.max_tokens,
    }
    for label, value in integer_limits.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise LlamaCppEvalV5Error(f"{label} must be a positive integer")
    if config.threads > 16:
        raise LlamaCppEvalV5Error("threads must not exceed 16")
    if config.context_size > 8192:
        raise LlamaCppEvalV5Error("context_size must not exceed 8192")
    if config.max_tokens > 1024:
        raise LlamaCppEvalV5Error("max_tokens must not exceed 1024")
    if config.gpu_layers != 0:
        raise LlamaCppEvalV5Error("this evaluator requires n-gpu-layers=0")
    if config.startup_timeout_seconds <= 0 or config.request_timeout_seconds <= 0:
        raise LlamaCppEvalV5Error("timeouts must be positive")


def _free_high_port() -> int:
    for _ in range(256):
        candidate = secrets.randbelow(MAX_PORT - MIN_HIGH_PORT + 1) + MIN_HIGH_PORT
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
            try:
                handle.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate
    raise LlamaCppEvalV5Error("could not allocate a random loopback high port")


def _offline_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.lower()
        not in {
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "ftp_proxy",
            "no_proxy",
        }
    }
    environment.update(
        {
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return environment


class LocalLlamaServer:
    """One private llama-server process bound only to loopback."""

    def __init__(self, spec: ServerLaunchSpec) -> None:
        self.spec = spec
        self.port = _free_high_port()
        self._api_key = secrets.token_urlsafe(32)
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout: Any = None
        self._stderr: Any = None
        self._command: list[str] = []
        self._started_at: float | None = None
        self._closed_at: float | None = None
        self._stdout_tail = ""
        self._stderr_tail = ""
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _request(
        self,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        timeout: float,
    ) -> Mapping[str, Any]:
        url = self.endpoint + path
        if not url.startswith("http://127.0.0.1:"):
            raise LlamaCppEvalV5Error("non-loopback llama.cpp URL was rejected")
        payload = None if body is None else canonical_json(body).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
                **({"Content-Type": "application/json"} if payload is not None else {}),
            },
            method="POST" if payload is not None else "GET",
        )
        with self._opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_HTTP_BYTES + 1)
        if len(raw) > MAX_HTTP_BYTES:
            raise LlamaCppEvalV5Error("llama.cpp response exceeded size limit")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, Mapping):
            raise LlamaCppEvalV5Error("llama.cpp response root is not an object")
        return value

    def start(self) -> None:
        self.spec.log_dir.mkdir(parents=True, exist_ok=False)
        self._stdout = (self.spec.log_dir / "stdout.log").open("wb")
        self._stderr = (self.spec.log_dir / "stderr.log").open("wb")
        self._command = [
            str(self.spec.executable),
            "-m",
            str(self.spec.model),
            "-ngl",
            "0",
            "-t",
            str(self.spec.threads),
            "-c",
            str(self.spec.context_size),
            "-np",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--alias",
            MODEL_ALIAS,
            "--api-key",
            self._api_key,
            "--no-webui",
            "--reasoning",
            "off",
        ]
        self._started_at = time.monotonic()
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.DEVNULL,
            stdout=self._stdout,
            stderr=self._stderr,
            env=_offline_environment(),
            cwd=self.spec.runtime_dir,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.monotonic() + self.spec.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise LlamaCppEvalV5Error("local llama-server exited before becoming healthy")
            try:
                health = self._request("/health", timeout=2.0)
                if str(health.get("status", "")).lower() in {"ok", "ready"}:
                    return
            except (
                OSError,
                TimeoutError,
                ValueError,
                json.JSONDecodeError,
                urllib.error.URLError,
            ):
                pass
            time.sleep(0.2)
        raise TimeoutError("local llama-server health wait timed out")

    def chat(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        request_body = dict(body)
        request_body["model"] = MODEL_ALIAS
        return self._request(
            "/v1/chat/completions",
            body=request_body,
            timeout=self.spec.request_timeout_seconds,
        )

    def _capture_tail(self, path: Path) -> str:
        try:
            payload = path.read_bytes()[-16_384:]
            return payload.decode("utf-8", errors="replace")
        except OSError:
            return ""

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)
        for handle in (self._stdout, self._stderr):
            if handle is not None and not handle.closed:
                handle.close()
        self._stdout_tail = self._capture_tail(self.spec.log_dir / "stdout.log")
        self._stderr_tail = self._capture_tail(self.spec.log_dir / "stderr.log")
        self._closed_at = time.monotonic()

    def port_released(self) -> bool:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
                handle.settimeout(0.1)
                if handle.connect_ex(("127.0.0.1", self.port)) != 0:
                    return True
            time.sleep(0.05)
        return False

    def trace_metadata(self) -> Mapping[str, Any]:
        command = [
            "<redacted-api-key>" if previous == "--api-key" else value
            for previous, value in zip(["", *self._command[:-1]], self._command, strict=True)
        ]
        return {
            "pid": self._process.pid if self._process is not None else None,
            "returncode": (self._process.poll() if self._process is not None else None),
            "host": "127.0.0.1",
            "port": self.port,
            "command": command,
            "stdout_tail": self._stdout_tail,
            "stderr_tail": self._stderr_tail,
            "elapsed_seconds": (
                self._closed_at - self._started_at
                if self._closed_at is not None and self._started_at is not None
                else None
            ),
        }


def _extract_generation(response: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LlamaCppEvalV5Error("llama.cpp chat completion response shape is invalid") from exc
    if not isinstance(content, str):
        raise LlamaCppEvalV5Error("llama.cpp generation content is not a string")
    usage = response.get("usage")
    return content.strip(), {
        "response_id": response.get("id"),
        "finish_reason": choice.get("finish_reason") if isinstance(choice, Mapping) else None,
        "usage": dict(usage) if isinstance(usage, Mapping) else None,
    }


def _generate(
    requests: Sequence[GenerationRequestV5],
    *,
    server: ServerSession,
    config: LlamaCppEvalConfig,
) -> tuple[
    dict[tuple[str, str], str],
    dict[tuple[str, str], dict[str, Any]],
]:
    generations: dict[tuple[str, str], str] = {}
    traces: dict[tuple[str, str], dict[str, Any]] = {}
    for request in requests:
        body = {
            "messages": [dict(message) for message in request.messages],
            "temperature": 0,
            "max_tokens": config.max_tokens,
            "seed": config.seed,
            "stream": False,
        }
        if any(message.get("role") == "assistant" for message in body["messages"]):
            raise LlamaCppEvalV5Error("assistant gold leaked into a generation request")
        started = time.perf_counter()
        response = server.chat(body)
        latency_ms = (time.perf_counter() - started) * 1000.0
        generation, response_trace = _extract_generation(response)
        key = (request.ablation, request.example_id)
        generations[key] = generation
        traces[key] = {
            "prompt_sha256": request.prompt_sha256,
            "request_sha256": sha256_bytes(canonical_json(body).encode("utf-8")),
            "response_sha256": sha256_bytes(canonical_json(response).encode("utf-8")),
            "latency_ms": latency_ms,
            **response_trace,
        }
    return generations, traces


def _publish_directory(stage: Path, output: Path) -> None:
    if output.exists():
        raise LlamaCppEvalV5Error("evaluation output already exists; use a new immutable directory")
    stage.rename(output)


def _write_stage_file(stage: Path, name: str, payload: bytes) -> Path:
    path = stage / name
    temporary = stage / f".{name}.{uuid.uuid4().hex}.tmp"
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return path


def _failure_receipt(
    *,
    config: LlamaCppEvalConfig,
    error: BaseException,
    code: Mapping[str, Any] | None,
    inputs: Mapping[str, Any] | None,
    server_cleanup: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": FAILURE_RECEIPT_SCHEMA,
        "created_at_utc": _utc_now(),
        "status": "FAILED",
        "evaluator_version": EVALUATOR_VERSION,
        "split": config.split,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        "inputs": dict(inputs or {}),
        "code": dict(code or {}),
        "server_cleanup": dict(server_cleanup or {}),
        "backend": {
            "mode": "llama_cpp_gguf",
            "is_model": True,
            "free_generation_executed": False,
            "quantization": QUANTIZATION,
        },
        "claim_boundary": "FAILED_RUN_NO_MODEL_QUALITY_CLAIM",
        "production_integration_allowed": False,
    }


def run_llama_cpp_evaluation(
    config: LlamaCppEvalConfig,
    *,
    server_factory: ServerFactory = LocalLlamaServer,
) -> dict[str, Any]:
    """Run one immutable local llama.cpp evaluation and publish its evidence."""

    output = config.output_dir.expanduser().resolve()
    if output.exists():
        raise LlamaCppEvalV5Error("evaluation output already exists; use a new immutable directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.stage-{uuid.uuid4().hex}"
    stage.mkdir()
    code: dict[str, Any] | None = None
    inputs: dict[str, Any] | None = None
    server: ServerSession | None = None
    runtime_temp: tempfile.TemporaryDirectory[str] | None = None
    server_cleanup: dict[str, Any] | None = None
    try:
        _validate_limits(config)
        ablations = _selected_ablations(config.ablations)
        model, model_record = _regular_file(
            config.gguf_model,
            label="GGUF model",
            expected_sha256=config.expected_gguf_sha256,
        )
        executable, executable_record = _regular_file(
            config.llama_server,
            label="llama-server",
            expected_sha256=config.expected_llama_server_sha256,
        )
        runtime_dir = executable.parent
        runtime_before = _tree_inventory(runtime_dir)
        code = _source_inventory(config.runner_path)
        inputs = {
            "gguf_model": model_record,
            "llama_server": executable_record,
            "llama_cpp_runtime": runtime_before,
        }
        selection = load_dataset_selection(
            config.dataset_dir,
            split=config.split,
            max_samples=config.max_samples,
            blind_authorization_path=config.blind_authorization_path,
            blind_authorization_sha256=config.blind_authorization_sha256,
        )
        requests = tuple(
            request
            for ablation in ablations
            for request in build_generation_requests(
                selection.samples,
                ablation=ablation,
            )
        )
        runtime_temp = tempfile.TemporaryDirectory(
            prefix="icmat-llama-server-",
            dir=output.parent,
        )
        spec = ServerLaunchSpec(
            executable=executable,
            model=model,
            runtime_dir=runtime_dir,
            log_dir=Path(runtime_temp.name) / "logs",
            threads=config.threads,
            context_size=config.context_size,
            gpu_layers=0,
            startup_timeout_seconds=config.startup_timeout_seconds,
            request_timeout_seconds=config.request_timeout_seconds,
            seed=config.seed,
        )
        server = server_factory(spec)
        generation_started = time.perf_counter()
        try:
            server.start()
            generations, traces = _generate(requests, server=server, config=config)
        finally:
            server.close()
        elapsed_seconds = time.perf_counter() - generation_started
        port_released = server.port_released()
        server_cleanup = {
            "close_called": True,
            "port_released": port_released,
            "metadata": dict(server.trace_metadata()),
        }
        if not port_released:
            raise LlamaCppEvalV5Error("llama-server loopback port was not released")
        runtime_after = _tree_inventory(runtime_dir)
        if runtime_after["tree_sha256"] != runtime_before["tree_sha256"]:
            raise LlamaCppEvalV5Error("llama.cpp runtime tree changed during evaluation")
        if sha256_file(model) != model_record["sha256"]:
            raise LlamaCppEvalV5Error("GGUF model changed during evaluation")
        if sha256_file(executable) != executable_record["sha256"]:
            raise LlamaCppEvalV5Error("llama-server changed during evaluation")
        if _source_inventory(config.runner_path) != code:
            raise LlamaCppEvalV5Error("evaluation source code changed during evaluation")

        backend = {
            "mode": "llama_cpp_gguf",
            "is_model": True,
            "free_generation_executed": True,
            "assistant_target_visible_to_backend": False,
            "model_quality_evidence": True,
            "claim_boundary": "LOCAL_LLAMA_CPP_GGUF_FREE_GENERATION",
            "quantization": QUANTIZATION,
            "subject": "merged_qlora_student",
            "device": "cpu",
            "gpu_layers": 0,
            "network_allowed": False,
            "server_bind_host": "127.0.0.1",
            "sequential_generation": True,
            "seed": config.seed,
            "decoding": {
                "temperature": 0,
                "max_tokens": config.max_tokens,
                "context_size": config.context_size,
                "threads": config.threads,
            },
            "samples_generated": len(requests),
            "elapsed_seconds": elapsed_seconds,
            "gguf_model": model_record,
            "llama_server": executable_record,
            "llama_cpp_runtime": {
                **runtime_before,
                "unchanged_after_run": True,
            },
        }
        rows, summaries = score_generations(
            selection=selection,
            requests=requests,
            generations=generations,
            backend=backend,
            traces=traces,
        )
        per_sample_payload = _jsonl_bytes(rows)
        per_sample_sha256 = sha256_bytes(per_sample_payload)
        summary = {
            "schema": SUMMARY_SCHEMA,
            "split": config.split,
            "examples": len(selection.samples),
            "ablations": list(ablations),
            "dataset": {
                "manifest_path": str(selection.manifest_path),
                "manifest_sha256": selection.manifest_sha256,
                "split_path": str(selection.split_path),
                "split_sha256": selection.split_sha256,
            },
            "blind_test_authorization": selection.blind_test_authorization,
            "backend": backend,
            "model_quality_claim_allowed": True,
            "assistant_target_visible_to_backend": False,
            "summaries": summaries,
            "per_sample_sha256": per_sample_sha256,
        }
        summary_payload = _json_bytes(summary)
        summary_sha256 = sha256_bytes(summary_payload)
        _write_stage_file(stage, "per_sample.v5.jsonl", per_sample_payload)
        _write_stage_file(stage, "summary.v5.json", summary_payload)
        receipt = {
            "schema": RUN_RECEIPT_SCHEMA,
            "created_at_utc": _utc_now(),
            "status": "COMPLETED",
            "evaluator_version": EVALUATOR_VERSION,
            "split": config.split,
            "dataset": summary["dataset"],
            "blind_test_authorization": selection.blind_test_authorization,
            "backend": backend,
            "generation_contract": {
                "assistant_target_visible_to_backend": False,
                "free_generation_executed": True,
                "request_message_roles": ["system", "user"],
                "ablations": list(ablations),
                "seed": config.seed,
                "requests": [
                    {
                        "example_id": request.example_id,
                        "ablation": request.ablation,
                        "prompt_sha256": request.prompt_sha256,
                        **traces[(request.ablation, request.example_id)],
                    }
                    for request in requests
                ],
            },
            "server": {
                **server_cleanup["metadata"],
                "port_released": True,
                "process_cleanup_required": True,
                "process_exited": True,
            },
            "artifacts": {
                "per_sample.v5.jsonl": {
                    "bytes": len(per_sample_payload),
                    "sha256": per_sample_sha256,
                    "records": len(rows),
                },
                "summary.v5.json": {
                    "bytes": len(summary_payload),
                    "sha256": summary_sha256,
                },
            },
            "code": code,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "network_used_by_evaluator": False,
                "proxy_used": False,
                "loopback_only": True,
                "runtime_tree_unchanged": True,
            },
            "claim_boundary": "LOCAL_LLAMA_CPP_GGUF_FREE_GENERATION",
            "production_integration_allowed": False,
        }
        receipt["receipt_payload_sha256"] = sha256_bytes(canonical_json(receipt).encode("utf-8"))
        receipt_payload = _json_bytes(receipt)
        receipt_sha256 = sha256_bytes(receipt_payload)
        _write_stage_file(stage, "run_receipt.v5.json", receipt_payload)
        _publish_directory(stage, output)
        return {
            "output_dir": str(output),
            "paths": {
                "per_sample": str(output / "per_sample.v5.jsonl"),
                "summary": str(output / "summary.v5.json"),
                "run_receipt": str(output / "run_receipt.v5.json"),
            },
            "hashes": {
                "per_sample": per_sample_sha256,
                "summary": summary_sha256,
                "run_receipt": receipt_sha256,
            },
            "summary": summary,
            "receipt": receipt,
        }
    except BaseException as exc:
        if server is not None:
            cleanup_error: str | None = None
            try:
                server.close()
            except BaseException as close_exc:
                cleanup_error = f"{type(close_exc).__name__}: {close_exc}"
            try:
                port_released = server.port_released()
            except BaseException as port_exc:
                port_released = False
                cleanup_error = (
                    f"{cleanup_error}; " if cleanup_error else ""
                ) + f"{type(port_exc).__name__}: {port_exc}"
            try:
                metadata = dict(server.trace_metadata())
            except BaseException as metadata_exc:
                metadata = {"metadata_error": (f"{type(metadata_exc).__name__}: {metadata_exc}")}
            server_cleanup = {
                "close_called": True,
                "port_released": port_released,
                "cleanup_error": cleanup_error,
                "metadata": metadata,
            }
        shutil.rmtree(stage, ignore_errors=True)
        failure_stage = output.parent / f".{output.name}.failure-{uuid.uuid4().hex}"
        failure_stage.mkdir()
        failure = _failure_receipt(
            config=config,
            error=exc,
            code=code,
            inputs=inputs,
            server_cleanup=server_cleanup,
        )
        _write_stage_file(
            failure_stage,
            "failure_receipt.v5.json",
            _json_bytes(failure),
        )
        try:
            _publish_directory(failure_stage, output)
        except BaseException:
            shutil.rmtree(failure_stage, ignore_errors=True)
        raise
    finally:
        if runtime_temp is not None:
            runtime_temp.cleanup()
        shutil.rmtree(stage, ignore_errors=True)


__all__ = [
    "FAILURE_RECEIPT_SCHEMA",
    "LlamaCppEvalConfig",
    "LlamaCppEvalV5Error",
    "LocalLlamaServer",
    "QUANTIZATION",
    "RUN_RECEIPT_SCHEMA",
    "SUMMARY_SCHEMA",
    "ServerLaunchSpec",
    "run_llama_cpp_evaluation",
]
