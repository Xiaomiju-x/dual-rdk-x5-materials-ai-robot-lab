"""Hash-bound local llama.cpp teacher candidate generation.

The module deliberately emits unaudited candidates. It never promotes teacher
output into an SFT dataset and never contacts a non-loopback endpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import socket
import stat
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from jsonschema import Draft202012Validator

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
REQUEST_SCHEMA = "icmat_teacher_request.v1"
CANDIDATE_SCHEMA = "icmat_teacher_candidate.v1"
RECEIPT_SCHEMA = "icmat_local_teacher_run_receipt.v1"
RUNNER_VERSION = "icmat-local-teacher-1.1.0"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUEST_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,95}$")
ALLOWED_SPLITS = frozenset({"train", "validation", "calibration"})
ALLOWED_TASKS = frozenset(
    {
        "evidence_grounded_explanation",
        "evidence_bounded_comparison",
        "computed_experimental_boundary",
        "next_measurement_or_tool",
        "refusal_counterfactual",
    }
)
ALLOWED_LICENSES = frozenset({"CC BY 4.0"})
MAX_REQUEST_FILE_BYTES = 32 * 1024 * 1024
MAX_EVIDENCE_TEXT_CHARS = 6000
MAX_RESPONSE_TOKENS = 768
FILE_ATTRIBUTE_REPARSE_POINT = 0x400


class TeacherContractError(ValueError):
    """Raised when a local teacher artifact violates its immutable contract."""


@dataclass(frozen=True, slots=True)
class TeacherConfig:
    context_tokens: int = 4096
    gpu_layers: int = 99
    parallel_slots: int = 1
    request_timeout_seconds: int = 180
    startup_timeout_seconds: int = 120

    def validate(self) -> None:
        if not 512 <= self.context_tokens <= 32768:
            raise TeacherContractError("context_tokens must be in [512, 32768]")
        if not 0 <= self.gpu_layers <= 999:
            raise TeacherContractError("gpu_layers must be in [0, 999]")
        if self.parallel_slots != 1:
            raise TeacherContractError("the RTX4050 contract permits one teacher slot")
        if not 10 <= self.request_timeout_seconds <= 600:
            raise TeacherContractError("request timeout must be in [10, 600] seconds")
        if not 10 <= self.startup_timeout_seconds <= 300:
            raise TeacherContractError("startup timeout must be in [10, 300] seconds")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_reparse(metadata: os.stat_result) -> bool:
    return bool(
        int(getattr(metadata, "st_file_attributes", 0))
        & FILE_ATTRIBUTE_REPARSE_POINT
    )


def _is_unc(path: Path) -> bool:
    raw = os.fspath(path)
    normalized = raw.replace("/", "\\")
    return normalized.startswith("\\\\") or PureWindowsPath(raw).drive.startswith(
        "\\\\"
    )


def _workspace_file(path: Path, *, workspace_root: Path) -> Path:
    if _is_unc(path) or ".." in path.parts:
        raise TeacherContractError("teacher artifact path is unsafe")
    root = workspace_root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    candidate = candidate.resolve(strict=True)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise TeacherContractError(
            "teacher artifact must stay inside the workspace"
        ) from exc
    if any(":" in part for part in relative.parts):
        raise TeacherContractError("alternate data streams are forbidden")
    current = root
    for part in relative.parts:
        current = current / part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
            raise TeacherContractError(
                "teacher artifact path contains a symlink or reparse point"
            )
    if not candidate.is_file():
        raise TeacherContractError("teacher artifact must be a regular file")
    return candidate


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise TeacherContractError(f"{label} must be lowercase SHA-256")
    return value


def _read_bound_file(
    path: Path,
    *,
    expected_sha256: str,
    workspace_root: Path,
    maximum_bytes: int | None = None,
) -> tuple[Path, bytes]:
    expected = _require_sha256(expected_sha256, "expected_sha256")
    resolved = _workspace_file(path, workspace_root=workspace_root)
    before = os.stat(resolved)
    if maximum_bytes is not None and before.st_size > maximum_bytes:
        raise TeacherContractError("teacher artifact exceeds its bounded size")
    payload = resolved.read_bytes()
    after = os.stat(resolved)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise TeacherContractError("teacher artifact changed during read")
    if sha256_bytes(payload) != expected:
        raise TeacherContractError("teacher artifact SHA-256 mismatch")
    return resolved, payload


def _exact_object(
    value: object,
    keys: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise TeacherContractError(f"{label} has unexpected fields")
    return value


def _validate_evidence(value: object, request_id: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise TeacherContractError(f"{request_id}: evidence must be non-empty")
    records: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    keys = {
        "source_id",
        "document_id",
        "chunk_id",
        "locator",
        "text",
        "text_sha256",
        "license_id",
        "access_mode",
    }
    for index, raw in enumerate(value):
        record = _exact_object(raw, keys, f"{request_id}.evidence[{index}]")
        for field in (
            "source_id",
            "document_id",
            "chunk_id",
            "locator",
            "text",
            "license_id",
            "access_mode",
        ):
            if not isinstance(record[field], str) or not record[field]:
                raise TeacherContractError(
                    f"{request_id}: evidence {field} must be non-empty text"
                )
        if record["chunk_id"] in seen_chunks:
            raise TeacherContractError(f"{request_id}: duplicate evidence chunk")
        seen_chunks.add(record["chunk_id"])
        if record["license_id"] not in ALLOWED_LICENSES:
            raise TeacherContractError(
                f"{request_id}: evidence license is not training-authorized"
            )
        if record["access_mode"] != "licensed_fulltext_readonly":
            raise TeacherContractError(
                f"{request_id}: evidence is not licensed full text"
            )
        if len(record["text"]) > MAX_EVIDENCE_TEXT_CHARS:
            raise TeacherContractError(f"{request_id}: evidence text is too large")
        text_sha256 = _require_sha256(
            record["text_sha256"],
            f"{request_id}.evidence[{index}].text_sha256",
        )
        if sha256_bytes(record["text"].encode("utf-8")) != text_sha256:
            raise TeacherContractError(
                f"{request_id}: evidence text hash does not match"
            )
        records.append(record)
    return tuple(records)


def validate_teacher_request(value: object) -> dict[str, Any]:
    keys = {
        "schema",
        "request_id",
        "split",
        "task",
        "messages",
        "evidence",
        "response_schema",
        "generation",
    }
    request = _exact_object(value, keys, "teacher request")
    if request["schema"] != REQUEST_SCHEMA:
        raise TeacherContractError("unsupported teacher request schema")
    request_id = request["request_id"]
    if not isinstance(request_id, str) or not REQUEST_ID_RE.fullmatch(request_id):
        raise TeacherContractError("teacher request_id is invalid")
    if request["split"] not in ALLOWED_SPLITS:
        raise TeacherContractError(f"{request_id}: test/unknown split is forbidden")
    if request["task"] not in ALLOWED_TASKS:
        raise TeacherContractError(f"{request_id}: task is unsupported")

    messages = request["messages"]
    if not isinstance(messages, list) or len(messages) != 2:
        raise TeacherContractError(f"{request_id}: expected system and user messages")
    expected_roles = ("system", "user")
    for index, role in enumerate(expected_roles):
        message = _exact_object(
            messages[index],
            {"role", "content"},
            f"{request_id}.messages[{index}]",
        )
        if message["role"] != role:
            raise TeacherContractError(f"{request_id}: message role mismatch")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise TeacherContractError(f"{request_id}: message content is empty")

    evidence = _validate_evidence(request["evidence"], request_id)
    user_content = messages[1]["content"]
    for record in evidence:
        if record["chunk_id"] not in user_content:
            raise TeacherContractError(
                f"{request_id}: user prompt omits an evidence chunk id"
            )

    response_schema = request["response_schema"]
    if not isinstance(response_schema, dict):
        raise TeacherContractError(f"{request_id}: response_schema must be an object")
    try:
        Draft202012Validator.check_schema(response_schema)
    except Exception as exc:
        raise TeacherContractError(
            f"{request_id}: response_schema is invalid"
        ) from exc

    generation = _exact_object(
        request["generation"],
        {"temperature", "max_tokens", "seed"},
        f"{request_id}.generation",
    )
    if generation["temperature"] != 0:
        raise TeacherContractError(
            f"{request_id}: deterministic candidate generation requires temperature=0"
        )
    if (
        isinstance(generation["max_tokens"], bool)
        or not isinstance(generation["max_tokens"], int)
        or not 16 <= generation["max_tokens"] <= MAX_RESPONSE_TOKENS
    ):
        raise TeacherContractError(f"{request_id}: max_tokens is out of bounds")
    if isinstance(generation["seed"], bool) or not isinstance(
        generation["seed"], int
    ):
        raise TeacherContractError(f"{request_id}: seed must be an integer")
    return request


def load_teacher_requests(
    path: Path,
    *,
    expected_sha256: str,
    workspace_root: Path = WORKSPACE_ROOT,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    resolved, payload = _read_bound_file(
        path,
        expected_sha256=expected_sha256,
        workspace_root=workspace_root,
        maximum_bytes=MAX_REQUEST_FILE_BYTES,
    )
    requests: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TeacherContractError(
                f"teacher request line {line_number} is invalid JSON"
            ) from exc
        request = validate_teacher_request(value)
        if request["request_id"] in seen_ids:
            raise TeacherContractError("teacher request ids must be unique")
        seen_ids.add(request["request_id"])
        requests.append(request)
    if not requests:
        raise TeacherContractError("teacher request file is empty")
    root = workspace_root.resolve(strict=True)
    return tuple(requests), {
        "path": resolved.relative_to(root).as_posix(),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "request_count": len(requests),
    }


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _parse_exact_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.lstrip()
    try:
        value, end = json.JSONDecoder(
            object_pairs_hook=_reject_duplicate_json_keys
        ).raw_decode(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if stripped[end:].strip() or not isinstance(value, dict):
        return None
    return value


def generate_candidates_with_transport(
    requests: Iterable[Mapping[str, Any]],
    transport: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    candidates: list[dict[str, Any]] = []
    for request in requests:
        body = {
            "model": "local-pinned-teacher",
            "messages": request["messages"],
            "temperature": 0,
            "max_tokens": request["generation"]["max_tokens"],
            "seed": request["generation"]["seed"],
            "response_format": {
                "type": "json_schema",
                "schema": request["response_schema"],
            },
        }
        response = transport(body)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TeacherContractError(
                f"{request['request_id']}: teacher response shape is invalid"
            ) from exc
        if not isinstance(content, str) or not content:
            raise TeacherContractError(
                f"{request['request_id']}: teacher response content is empty"
            )
        parsed = _parse_exact_json_object(content)
        schema_valid = False
        if parsed is not None:
            schema_valid = not any(
                Draft202012Validator(request["response_schema"]).iter_errors(parsed)
            )
        candidates.append(
            {
                "schema": CANDIDATE_SCHEMA,
                "request_id": request["request_id"],
                "split": request["split"],
                "task": request["task"],
                "response_text": content,
                "response_text_sha256": sha256_bytes(content.encode("utf-8")),
                "parsed_response": parsed,
                "json_object_valid": parsed is not None,
                "response_schema_valid": schema_valid,
                "source_refs": [
                    {
                        "source_id": item["source_id"],
                        "document_id": item["document_id"],
                        "chunk_id": item["chunk_id"],
                        "locator": item["locator"],
                        "text_sha256": item["text_sha256"],
                    }
                    for item in request["evidence"]
                ],
                "candidate_only": True,
                "grounding_validated": False,
                "student_training_authorized": False,
            }
        )
    return tuple(candidates)


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


class _LocalLlamaServer:
    def __init__(
        self,
        *,
        executable: Path,
        model: Path,
        log_dir: Path,
        config: TeacherConfig,
        extra_args: Sequence[str] = (),
    ) -> None:
        self.executable = executable
        self.model = model
        self.log_dir = log_dir
        self.config = config
        self.extra_args = tuple(extra_args)
        self.port = _free_loopback_port()
        self._api_key = secrets.token_urlsafe(32)
        self._model_alias = "icmat-teacher-" + secrets.token_hex(16)
        self.process: subprocess.Popen[bytes] | None = None
        self._stdout: Any = None
        self._stderr: Any = None

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=False)
        self._stdout = (self.log_dir / "server_stdout.log").open("wb")
        self._stderr = (self.log_dir / "server_stderr.log").open("wb")
        command = [
            str(self.executable),
            "-m",
            str(self.model),
            "-ngl",
            str(self.config.gpu_layers),
            "-c",
            str(self.config.context_tokens),
            "-np",
            str(self.config.parallel_slots),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--alias",
            self._model_alias,
            "--api-key",
            self._api_key,
            "--no-webui",
            "--reasoning",
            "off",
        ]
        command.extend(self.extra_args)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self._stdout,
            stderr=self._stderr,
            creationflags=creationflags,
        )
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise TeacherContractError("local llama.cpp server exited during startup")
            try:
                health_request = urllib.request.Request(
                    self.endpoint + "/health",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                with urllib.request.urlopen(health_request, timeout=2) as response:
                    payload = json.loads(
                        response.read().decode("utf-8"),
                        object_pairs_hook=_reject_duplicate_json_keys,
                    )
                if payload.get("status") != "ok":
                    time.sleep(0.25)
                    continue
                models_request = urllib.request.Request(
                    self.endpoint + "/v1/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                with urllib.request.urlopen(models_request, timeout=2) as response:
                    models = json.loads(
                        response.read().decode("utf-8"),
                        object_pairs_hook=_reject_duplicate_json_keys,
                    )
                model_ids = {
                    item.get("id")
                    for item in models.get("data", [])
                    if isinstance(item, Mapping)
                }
                if self._model_alias in model_ids:
                    return
            except (
                OSError,
                TimeoutError,
                ValueError,
                json.JSONDecodeError,
                urllib.error.URLError,
            ):
                time.sleep(0.25)
        raise TeacherContractError("local llama.cpp server startup timed out")

    def chat(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        request_body = dict(body)
        request_body["model"] = self._model_alias
        request = urllib.request.Request(
            self.endpoint + "/v1/chat/completions",
            data=canonical_json(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.request_timeout_seconds,
            ) as response:
                payload = json.loads(
                    response.read().decode("utf-8"),
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
        except (
            OSError,
            TimeoutError,
            ValueError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ) as exc:
            raise TeacherContractError("local teacher request failed") from exc
        if not isinstance(payload, dict):
            raise TeacherContractError("local teacher response must be an object")
        return payload

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self._stdout is not None:
            self._stdout.close()
        if self._stderr is not None:
            self._stderr.close()

    def __enter__(self) -> _LocalLlamaServer:
        try:
            self.start()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    payload = "".join(canonical_json(record) + "\n" for record in records)
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def run_local_teacher(
    *,
    runtime_path: Path,
    runtime_sha256: str,
    model_path: Path,
    model_sha256: str,
    requests_path: Path,
    requests_sha256: str,
    output_dir: Path,
    config: TeacherConfig,
    max_requests: int | None = None,
    workspace_root: Path = WORKSPACE_ROOT,
) -> dict[str, Any]:
    config.validate()
    runtime, _ = _read_bound_file(
        runtime_path,
        expected_sha256=runtime_sha256,
        workspace_root=workspace_root,
    )
    model, _ = _read_bound_file(
        model_path,
        expected_sha256=model_sha256,
        workspace_root=workspace_root,
    )
    requests, request_receipt = load_teacher_requests(
        requests_path,
        expected_sha256=requests_sha256,
        workspace_root=workspace_root,
    )
    if max_requests is not None:
        if isinstance(max_requests, bool) or max_requests <= 0:
            raise TeacherContractError("max_requests must be a positive integer")
        requests = requests[:max_requests]

    root = workspace_root.resolve(strict=True)
    output = output_dir if output_dir.is_absolute() else root / output_dir
    output = Path(os.path.abspath(output))
    allowed = (root / "evaluation" / "icmat_foundry" / "llm").resolve()
    try:
        output.relative_to(allowed)
    except ValueError as exc:
        raise TeacherContractError(
            "teacher output must stay under evaluation/icmat_foundry/llm"
        ) from exc
    if output.exists():
        raise TeacherContractError("teacher output directory must not already exist")

    started_at = time.time()
    log_dir = output / "runtime"
    with _LocalLlamaServer(
        executable=runtime,
        model=model,
        log_dir=log_dir,
        config=config,
    ) as server:
        candidates = generate_candidates_with_transport(requests, server.chat)

    output.mkdir(parents=True, exist_ok=True)
    candidates_path = output / "teacher_candidates.jsonl"
    _write_jsonl_atomic(candidates_path, candidates)
    valid_json = sum(item["json_object_valid"] for item in candidates)
    valid_schema = sum(item["response_schema_valid"] for item in candidates)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "runner_version": RUNNER_VERSION,
        "status": "TEACHER_CANDIDATES_GENERATED_NOT_AUDITED",
        "started_unix_seconds": started_at,
        "completed_unix_seconds": time.time(),
        "runtime": {
            "path": runtime.relative_to(root).as_posix(),
            "sha256": runtime_sha256,
            "backend_request": "CUDA",
            "server_bind": "127.0.0.1",
        },
        "model": {
            "path": model.relative_to(root).as_posix(),
            "sha256": model_sha256,
        },
        "requests": request_receipt,
        "generated_request_count": len(candidates),
        "json_object_valid_count": valid_json,
        "response_schema_valid_count": valid_schema,
        "candidates": {
            "path": candidates_path.relative_to(root).as_posix(),
            "bytes": candidates_path.stat().st_size,
            "sha256": sha256_file(candidates_path),
        },
        "network_policy": {
            "server_bind": "loopback_only",
            "remote_api_used": False,
            "api_key_used": False,
            "pc_network_configuration_changed": False,
        },
        "authority": {
            "candidate_generation_only": True,
            "grounding_validated": False,
            "student_training_authorized": False,
            "x5_contacted": False,
            "production_modified": False,
        },
        "claim_boundary": (
            "The receipt proves only that a pinned local teacher generated candidate "
            "responses. Every candidate remains unauthorized for SFT until deterministic "
            "grounding checks and an external independent audit both pass."
        ),
    }
    _write_json_atomic(output / "teacher_run_receipt.v1.json", receipt)
    return receipt
