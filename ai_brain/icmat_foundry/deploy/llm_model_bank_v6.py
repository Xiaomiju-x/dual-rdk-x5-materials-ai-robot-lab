"""Build a deterministic, inactive ICMat-Qwen v6 model-bank candidate.

The builder consumes two already-finalized inputs:

* a PASS final LLM release bundle that explicitly authorizes packaging only;
* a PASS GGUF release whose strict local-PC parity gate already passed.

It does not train, select, deploy, activate, register, or start anything. The
result is an offline, content-addressed ZIP for a separate finals model bank.
Board measurements deliberately remain pending until a later validation phase.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, NoReturn

from icmat_foundry.llm.gguf_release_v6 import PARITY_PASS_STATUS
from icmat_foundry.llm.gguf_release_v6 import (
    RELEASE_PASS_STATUS as GGUF_RELEASE_PASS_STATUS,
)
from icmat_foundry.llm.gguf_release_v6 import (
    RELEASE_RECEIPT_SCHEMA as GGUF_RELEASE_RECEIPT_SCHEMA,
)
from icmat_foundry.llm.release_bundle_v6 import (
    FROZEN_SYSTEM_BOUNDARY as FINAL_FROZEN_SYSTEM_BOUNDARY,
)
from icmat_foundry.llm.release_bundle_v6 import (
    PACKAGE_TYPE as FINAL_RELEASE_PACKAGE_TYPE,
)
from icmat_foundry.llm.release_bundle_v6 import (
    PRODUCT_ID as FINAL_RELEASE_PRODUCT_ID,
)
from icmat_foundry.llm.release_bundle_v6 import (
    RELEASE_SCHEMA as FINAL_RELEASE_SCHEMA,
)
from icmat_foundry.llm.release_bundle_v6 import (
    RELEASE_STATUS as FINAL_RELEASE_PASS_STATUS,
)
from icmat_foundry.llm.release_bundle_v6 import verify_release_bundle_v6

BUILDER_VERSION = "icmat-llm-model-bank-v6.1.0"
PACKAGE_SCHEMA = "icmat_llm_model_bank_candidate.v6"
PACKAGE_KIND = "ICMAT_QWEN_V6_OFFLINE_FINALS_MODEL_BANK_CANDIDATE"
PACKAGE_STATUS = "BOARD_PENDING_NOT_DEPLOYED_NOT_ACTIVATED"
EVIDENCE_PIN_SCHEMA = "icmat_llm_model_bank_evidence_pin.v6"
RUNTIME_CONFIG_SCHEMA = "icmat_llm_model_bank_runtime_config.v6"
ROLLBACK_SCHEMA = "icmat_llm_model_bank_rollback_manifest.v6"
VERIFICATION_SCHEMA = "icmat_llm_model_bank_candidate_verification.v6"

PRODUCT_ID = FINAL_RELEASE_PRODUCT_ID
MODEL_BANK_RELATIVE_ROOT = PurePosixPath(
    "evaluation/icmat_foundry/packages/llm_model_bank_v6"
)
ARCHIVE_FAMILY_ROOT = PurePosixPath(
    "finals_model_bank/icmat-qwen-v6"
)
TARGET_FAMILY_ROOT = "~/icmat_foundry_finals/model_bank/icmat-qwen-v6"

PACKAGE_MANIFEST_NAME = "package_manifest.v6.json"
PACKAGE_ARCHIVE_NAME = "package.zip"
ARCHIVE_SHA256_NAME = "archive.sha256"
GGUF_RECEIPT_NAME = "release_receipt.v6.json"

MODEL_PATH = "artifacts/models/icmat-qwen05b-pointer-q4_k_m.gguf"
EVIDENCE_PIN_PATH = "contracts/evidence_pin.v6.json"
RUNTIME_CONFIG_PATH = "runtime/config.disabled.v6.json"
LAUNCHER_PATH = "runtime/launch_disabled.sh"
MODEL_CARD_PATH = "docs/MODEL_CARD.md"
ROLLBACK_PATH = "contracts/rollback_manifest.v6.json"
INTERNAL_MANIFEST_PATH = "contracts/package_manifest.v6.json"

FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
FIXED_FILE_MODE = stat.S_IFREG | 0o644
MAX_JSON_BYTES = 32 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")

PROTECTED_RUNTIME = {
    "production_files": ["dashboard.py", "start_x5.sh"],
    "production_ports": [5000, 5001, 8080, 8081, 8888],
    "existing_cpu_model_bank": "PRESERVE_UNCHANGED",
    "existing_bpu_swap_slots": "PRESERVE_UNCHANGED",
    "camera_ownership": "PRESERVE_UNCHANGED",
}

RESOURCE_BUDGET = {
    "measurement_scope": "BOARD_RUNTIME_PENDING",
    "latency_ms": {
        "status": "PENDING_BOARD_MEASUREMENT",
        "value": None,
    },
    "peak_rss_bytes": {
        "status": "PENDING_BOARD_MEASUREMENT",
        "value": None,
    },
    "steady_rss_bytes": {
        "status": "PENDING_BOARD_MEASUREMENT",
        "value": None,
    },
    "cpu_threads": {
        "status": "PENDING_BOARD_MEASUREMENT",
        "value": None,
    },
    "thermal_celsius": {
        "status": "PENDING_BOARD_MEASUREMENT",
        "value": None,
    },
    "bpu_usage": {
        "status": "NOT_APPLICABLE_CPU_GGUF_CANDIDATE",
        "value": None,
    },
}

COMMAND_POLICY = {
    "install": [],
    "systemd_enable": [],
    "systemd_start": [],
    "service_start": [],
    "production_file_replace": [],
    "port_bind": [],
}

ROLLBACK_POLICY = {
    "strategy": "REMOVE_NEW_CONTENT_ADDRESSED_DIRECTORY_ONLY",
    "activation_expected_before_board_acceptance": False,
    "production_rollback_required": False,
    "remove_only": ["THIS_CONTENT_ADDRESSED_CANDIDATE_DIRECTORY"],
    "preserve": [
        "dashboard.py",
        "start_x5.sh",
        "ports:5000,5001,8080,8081,8888",
        "existing_cpu_model_bank",
        "existing_bpu_swap_slots",
        "rb_voe_state",
    ],
    "commands": [],
}


class ModelBankV6Error(ValueError):
    """Raised when an input or package violates the offline candidate contract."""


@dataclass(frozen=True)
class ModelBankInputs:
    """Pinned workspace-local inputs for one candidate package."""

    workspace_root: Path
    final_release_bundle: Path
    final_release_bundle_sha256: str
    gguf_release_dir: Path
    gguf_release_receipt_sha256: str


@dataclass(frozen=True)
class _Material:
    relative_path: str
    kind: str
    role: str
    payload: bytes | None = None
    source: Path | None = None
    expected_bytes: int | None = None
    expected_sha256: str | None = None


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _pretty_json(value: Any) -> bytes:
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


def _sha256_stream(handle: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        total += len(block)
        digest.update(block)
    return total, digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)[1]


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelBankV6Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    raise ModelBankV6Error(f"non-finite JSON constant is forbidden: {value}")


def _assert_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ModelBankV6Error(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, label=f"{label}[{index}]")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ModelBankV6Error(f"{label} exceeds the JSON size limit")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelBankV6Error(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ModelBankV6Error(f"{label} must contain one JSON object")
    _assert_finite(value, label=label)
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ModelBankV6Error(f"{label} must be a lowercase SHA-256")
    return value


def _require_safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not SAFE_ID_RE.fullmatch(value):
        raise ModelBankV6Error(f"{label} is not a safe identifier")
    return value


def _is_reparse_or_symlink(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        info = path.stat(follow_symlinks=False)
    except (FileNotFoundError, OSError):
        return False
    attributes = getattr(info, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _ensure_no_link_chain(root: Path, path: Path, *, label: str) -> None:
    root_absolute = root.absolute()
    path_absolute = path.absolute()
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise ModelBankV6Error(f"{label} must stay inside workspace_root") from exc
    current = root_absolute
    if _is_reparse_or_symlink(current):
        raise ModelBankV6Error("workspace_root must not be a symlink/reparse point")
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current) and _is_reparse_or_symlink(current):
            raise ModelBankV6Error(f"{label} contains a symlink/reparse point")


def _workspace_file(root: Path, supplied: Path, *, label: str) -> Path:
    raw = Path(supplied)
    absolute = raw if raw.is_absolute() else root / raw
    _ensure_no_link_chain(root, absolute, label=label)
    try:
        resolved = absolute.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ModelBankV6Error(f"{label} does not exist") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ModelBankV6Error(f"{label} resolves outside workspace_root") from exc
    if not resolved.is_file() or _is_reparse_or_symlink(resolved):
        raise ModelBankV6Error(f"{label} must be a regular non-link file")
    return resolved


def _workspace_directory(root: Path, supplied: Path, *, label: str) -> Path:
    raw = Path(supplied)
    absolute = raw if raw.is_absolute() else root / raw
    _ensure_no_link_chain(root, absolute, label=label)
    try:
        resolved = absolute.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ModelBankV6Error(f"{label} does not exist") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ModelBankV6Error(f"{label} resolves outside workspace_root") from exc
    if not resolved.is_dir() or _is_reparse_or_symlink(resolved):
        raise ModelBankV6Error(f"{label} must be a regular non-link directory")
    return resolved


def _normalise_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelBankV6Error(f"{label} must be a non-empty relative path")
    if value != unicodedata.normalize("NFC", value):
        raise ModelBankV6Error(f"{label} must use NFC normalization")
    if "\x00" in value or "\\" in value:
        raise ModelBankV6Error(f"{label} must use POSIX separators")
    if value.startswith("/") or WINDOWS_DRIVE_RE.match(value):
        raise ModelBankV6Error(f"{label} must not be absolute")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or any(":" in part for part in pure.parts)
        or pure.as_posix() != value
    ):
        raise ModelBankV6Error(f"{label} is not a safe canonical path")
    return value


def _child_file(root: Path, relative_path: Any, *, label: str) -> Path:
    normalised = _normalise_relative_path(relative_path, label=label)
    candidate = root.joinpath(*PurePosixPath(normalised).parts)
    _ensure_no_link_chain(root, candidate, label=label)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ModelBankV6Error(f"{label} does not exist") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ModelBankV6Error(f"{label} escapes its release directory") from exc
    if not resolved.is_file() or _is_reparse_or_symlink(resolved):
        raise ModelBankV6Error(f"{label} must be a regular non-link file")
    return resolved


def _verify_file_record(
    release_dir: Path,
    record: Any,
    *,
    label: str,
    expected_kind: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, dict):
        raise ModelBankV6Error(f"{label} must be an artifact record")
    if record.get("kind") != expected_kind:
        raise ModelBankV6Error(f"{label} kind mismatch")
    source = _child_file(release_dir, record.get("path"), label=f"{label}.path")
    expected_size = record.get("bytes")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size <= 0
    ):
        raise ModelBankV6Error(f"{label}.bytes must be a positive integer")
    expected_sha = _require_sha256(record.get("sha256"), label=f"{label}.sha256")
    before = source.stat(follow_symlinks=False)
    actual_sha = sha256_file(source)
    after = source.stat(follow_symlinks=False)
    if (
        before.st_size != expected_size
        or after.st_size != expected_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
    ):
        raise ModelBankV6Error(f"{label} changed while being verified")
    if actual_sha != expected_sha:
        raise ModelBankV6Error(f"{label} SHA-256 mismatch")
    return source, {
        "path": str(record["path"]),
        "bytes": expected_size,
        "sha256": expected_sha,
        "kind": expected_kind,
    }


def _verify_optional_payload_digest(value: dict[str, Any], *, label: str) -> None:
    digest_fields = (
        "bundle_payload_sha256",
        "receipt_payload_sha256",
        "payload_sha256",
        "canonical_digest_sha256",
    )
    present = [field for field in digest_fields if field in value]
    if len(present) > 1:
        raise ModelBankV6Error(f"{label} has ambiguous payload digests")
    if not present:
        return
    field = present[0]
    expected = _require_sha256(value[field], label=f"{label}.{field}")
    body = dict(value)
    body.pop(field)
    actual = _sha256_bytes(canonical_json(body).encode("utf-8"))
    if actual != expected:
        raise ModelBankV6Error(f"{label} payload digest mismatch")


def _verify_final_release_bundle(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    expected = _require_sha256(
        expected_sha256,
        label="final_release_bundle_sha256",
    )
    if sha256_file(path) != expected:
        raise ModelBankV6Error("final release bundle file SHA-256 mismatch")
    value = _load_json(path, label="final release bundle")
    _verify_optional_payload_digest(value, label="final release bundle")
    if value.get("schema") != FINAL_RELEASE_SCHEMA:
        raise ModelBankV6Error("unsupported final release bundle schema")
    if value.get("status") != FINAL_RELEASE_PASS_STATUS:
        raise ModelBankV6Error("final release bundle is not PASS")
    if value.get("product_id") != PRODUCT_ID:
        raise ModelBankV6Error("final release bundle product_id mismatch")
    release_id = _require_safe_id(value.get("candidate_id"), label="candidate_id")
    if value.get("package_type") != FINAL_RELEASE_PACKAGE_TYPE:
        raise ModelBankV6Error("final release package_type mismatch")
    if value.get("system_boundary") != FINAL_FROZEN_SYSTEM_BOUNDARY:
        raise ModelBankV6Error("final release frozen system boundary changed")
    if path.parent.name != "manifest" or path.name != "release_manifest.v6.json":
        raise ModelBankV6Error("final release manifest is not in its canonical path")
    package_dir = path.parent.parent
    content_id = _require_sha256(value.get("content_id"), label="content_id")
    if package_dir.name != content_id:
        raise ModelBankV6Error("final release directory is not content-addressed")
    archive = package_dir.parent / f"{content_id}.zip"
    if not archive.is_file() or _is_reparse_or_symlink(archive):
        raise ModelBankV6Error("final release reproducible ZIP is missing or unsafe")
    try:
        verification = verify_release_bundle_v6(
            package_dir=package_dir,
            archive_path=archive,
        )
    except Exception as exc:
        raise ModelBankV6Error(
            f"final release independent verification failed: {exc}"
        ) from exc
    if verification.get("status") != "PASS_PC_OFFLINE_RELEASE_PACKAGE_VERIFIED":
        raise ModelBankV6Error("final release independent verification is not PASS")

    entries = value.get("entries")
    if not isinstance(entries, list):
        raise ModelBankV6Error("final release entries are missing")
    by_role: dict[str, dict[str, Any]] = {}
    for row in entries:
        if not isinstance(row, dict) or not isinstance(row.get("role"), str):
            raise ModelBankV6Error("final release entry is invalid")
        if row["role"] in by_role:
            raise ModelBankV6Error("final release contains duplicate roles")
        by_role[row["role"]] = row
    required_roles = {"gguf_model", "gguf_parity_receipt"}
    if not required_roles.issubset(by_role):
        raise ModelBankV6Error("final release GGUF bindings are missing")
    gguf_model = by_role["gguf_model"]
    gguf_receipt = by_role["gguf_parity_receipt"]
    bundled_model = _child_file(
        package_dir,
        gguf_model.get("path"),
        label="final release GGUF model",
    )
    bundled_receipt = _child_file(
        package_dir,
        gguf_receipt.get("path"),
        label="final release GGUF receipt",
    )
    for label, row, artifact in (
        ("final release GGUF model", gguf_model, bundled_model),
        ("final release GGUF receipt", gguf_receipt, bundled_receipt),
    ):
        expected_size = row.get("bytes")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size <= 0
            or artifact.stat().st_size != expected_size
        ):
            raise ModelBankV6Error(f"{label} byte count mismatch")
        expected_digest = _require_sha256(
            row.get("sha256"),
            label=f"{label} sha256",
        )
        if sha256_file(artifact) != expected_digest:
            raise ModelBankV6Error(f"{label} SHA-256 mismatch")
    gguf = {
        "status": GGUF_RELEASE_PASS_STATUS,
        "receipt_sha256": gguf_receipt["sha256"],
        "q4_gguf_sha256": gguf_model["sha256"],
    }
    return {
        "value": value,
        "file_sha256": expected,
        "release_id": release_id,
        "gguf_binding": gguf,
        "verification": verification,
    }


def _verify_gguf_release(
    release_dir: Path,
    *,
    expected_receipt_sha256: str,
    final_binding: dict[str, Any],
) -> dict[str, Any]:
    receipt = _child_file(
        release_dir,
        GGUF_RECEIPT_NAME,
        label="GGUF release receipt",
    )
    expected_receipt = _require_sha256(
        expected_receipt_sha256,
        label="gguf_release_receipt_sha256",
    )
    actual_receipt = sha256_file(receipt)
    if actual_receipt != expected_receipt:
        raise ModelBankV6Error("GGUF release receipt file SHA-256 mismatch")
    if final_binding["receipt_sha256"] != actual_receipt:
        raise ModelBankV6Error("final release GGUF receipt binding mismatch")

    value = _load_json(receipt, label="GGUF release receipt")
    _verify_optional_payload_digest(value, label="GGUF release receipt")
    if value.get("schema") != GGUF_RELEASE_RECEIPT_SCHEMA:
        raise ModelBankV6Error("unsupported GGUF release receipt schema")
    if value.get("status") != GGUF_RELEASE_PASS_STATUS:
        raise ModelBankV6Error("GGUF release is not PASS")
    for field in ("activated", "service_registered", "deployable_by_this_receipt"):
        if value.get(field) is not False:
            raise ModelBankV6Error(f"GGUF release {field} must remain false")
    if value.get("training_invoked") is not False:
        raise ModelBankV6Error("GGUF release must not train during release")
    if value.get("selection_invoked") is not False:
        raise ModelBankV6Error("GGUF release must not select during release")

    parity = value.get("parity")
    if (
        not isinstance(parity, dict)
        or parity.get("status") != PARITY_PASS_STATUS
        or parity.get("strict_gate_pass") is not True
    ):
        raise ModelBankV6Error("GGUF strict parity gate did not PASS")
    claim = value.get("claim_boundary")
    if not isinstance(claim, dict):
        raise ModelBankV6Error("GGUF claim boundary is missing")
    for field in (
        "rdk_x5_measured",
        "bpu_used",
        "bpu_supported_or_claimed",
        "production_activated",
        "services_modified",
    ):
        if claim.get(field) is not False:
            raise ModelBankV6Error(f"GGUF claim boundary {field} must be false")

    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ModelBankV6Error("GGUF release artifacts are missing")
    q4_path, q4 = _verify_file_record(
        release_dir,
        artifacts.get("gguf_q4_k_m"),
        label="GGUF Q4_K_M",
        expected_kind="GGUF_Q4_K_M",
    )
    with q4_path.open("rb") as handle:
        q4_magic = handle.read(4)
    if q4_magic != b"GGUF":
        raise ModelBankV6Error("GGUF Q4_K_M file has invalid magic")
    if q4["sha256"] != final_binding["q4_gguf_sha256"]:
        raise ModelBankV6Error("final release Q4_K_M binding mismatch")

    parity_path, parity_record = _verify_file_record(
        release_dir,
        artifacts.get("parity_report"),
        label="GGUF parity report",
        expected_kind="JSON_PARITY_REPORT",
    )
    parity_value = _load_json(parity_path, label="GGUF parity report")
    if parity_value.get("status") != PARITY_PASS_STATUS:
        raise ModelBankV6Error("GGUF parity report status is not PASS")
    if parity_record["sha256"] != parity.get("report_sha256"):
        raise ModelBankV6Error("GGUF parity report receipt binding mismatch")

    marker_path, marker = _verify_file_record(
        release_dir,
        artifacts.get("activation_disabled"),
        label="GGUF activation marker",
        expected_kind="ACTIVATION_POLICY",
    )
    try:
        marker_text = marker_path.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        raise ModelBankV6Error("GGUF activation marker must be ASCII") from exc
    if "ACTIVATION=DISABLED" not in marker_text:
        raise ModelBankV6Error("GGUF activation marker does not disable activation")

    return {
        "receipt": value,
        "receipt_sha256": actual_receipt,
        "q4_path": q4_path,
        "q4": q4,
        "parity": parity_record,
        "activation_marker": marker,
    }


def _runtime_config() -> dict[str, Any]:
    return {
        "schema": RUNTIME_CONFIG_SCHEMA,
        "status": PACKAGE_STATUS,
        "enabled": False,
        "default_enabled": False,
        "autostart": False,
        "service_registered": False,
        "execution_mode": "EXPLICIT_RESEARCHER_ONE_SHOT_CPU_ONLY",
        "backend": "llama.cpp CPU GGUF",
        "model_relative_path": "../" + MODEL_PATH,
        "input_tokens_max": 1536,
        "output_tokens_max": 64,
        "bind_address": None,
        "port": None,
        "bpu_backend": False,
        "hidden_router": False,
        "rb_voe_dependency": False,
        "protected_runtime": PROTECTED_RUNTIME,
        "resource_budget": RESOURCE_BUDGET,
    }


def _launcher_bytes() -> bytes:
    return (
        "#!/bin/sh\n"
        "set -eu\n"
        'if [ "${ICMAT_FINALS_EXPLICIT_ONE_SHOT:-}" != "ALLOW" ]; then\n'
        '  echo "ICMat-Qwen v6 candidate is disabled; explicit one-shot approval is required." >&2\n'
        "  exit 78\n"
        "fi\n"
        'if [ "${ICMAT_FINALS_PRODUCTION_OVERRIDE:-}" = "1" ]; then\n'
        '  echo "Production override is forbidden for this candidate." >&2\n'
        "  exit 78\n"
        "fi\n"
        'LLAMA_CLI_BIN="${LLAMA_CLI_BIN:?set LLAMA_CLI_BIN after board validation}"\n'
        'ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"\n'
        'MODEL="$ROOT/artifacts/models/icmat-qwen05b-pointer-q4_k_m.gguf"\n'
        'exec "$LLAMA_CLI_BIN" -m "$MODEL" -c 1536 -n 64 "$@"\n'
    ).encode("ascii")


def _model_card_bytes(
    *,
    release_id: str,
    q4_sha256: str,
    final_bundle_sha256: str,
    gguf_receipt_sha256: str,
) -> bytes:
    text = f"""# ICMat-Qwen v6 offline candidate

- Product: `{PRODUCT_ID}`
- Release ID: `{release_id}`
- Status: `{PACKAGE_STATUS}`
- Runtime target: explicit researcher-selected llama.cpp CPU one-shot
- Q4_K_M SHA-256: `{q4_sha256}`
- Final release bundle SHA-256: `{final_bundle_sha256}`
- GGUF release receipt SHA-256: `{gguf_receipt_sha256}`

## Verified before packaging

The final LLM release bundle passed its release gate. The GGUF release passed
strict local-PC HF/GGUF pointer and compiler parity. Every selected package
artifact was re-hashed while building and independently verified from the ZIP.

## Deliberately not claimed

This package is not deployed, not activated, not registered as a service, and
does not claim board latency, board memory, BPU execution, or production
integration. All board resource fields remain `PENDING_BOARD_MEASUREMENT`.

## Isolation

The launcher is disabled unless an explicit one-shot environment flag is
provided. It does not bind a port or call systemd. It must not replace
`dashboard.py`, `start_x5.sh`, the five frozen ports, the existing CPU model
bank, or any BPU swap slot. RB-VoE is not a dependency.

## Rollback

Before activation, rollback means removing only this new content-addressed
candidate directory after verifying its package manifest. No production file
or service rollback is required.
"""
    return text.encode("ascii")


def _rollback_bytes() -> bytes:
    return _pretty_json(
        {
            "schema": ROLLBACK_SCHEMA,
            "status": PACKAGE_STATUS,
            "default_enabled": False,
            "activated": False,
            "policy": ROLLBACK_POLICY,
            "protected_runtime": PROTECTED_RUNTIME,
        }
    )


def _evidence_pin_bytes(
    *,
    release_id: str,
    final_bundle_sha256: str,
    gguf_receipt_sha256: str,
    q4: dict[str, Any],
    parity: dict[str, Any],
    activation_marker: dict[str, Any],
) -> bytes:
    return _pretty_json(
        {
            "schema": EVIDENCE_PIN_SCHEMA,
            "status": PACKAGE_STATUS,
            "product_id": PRODUCT_ID,
            "release_id": release_id,
            "final_release_bundle": {
                "schema": FINAL_RELEASE_SCHEMA,
                "status": FINAL_RELEASE_PASS_STATUS,
                "sha256": final_bundle_sha256,
            },
            "gguf_release": {
                "schema": GGUF_RELEASE_RECEIPT_SCHEMA,
                "status": GGUF_RELEASE_PASS_STATUS,
                "receipt_sha256": gguf_receipt_sha256,
                "strict_parity_status": PARITY_PASS_STATUS,
                "parity_report_sha256": parity["sha256"],
                "activation_marker_sha256": activation_marker["sha256"],
                "q4_gguf_sha256": q4["sha256"],
                "q4_gguf_bytes": q4["bytes"],
            },
            "authorization": {
                "offline_candidate_packaging": True,
                "deployment": False,
                "activation": False,
                "production_integration": False,
            },
        }
    )


def _material_record(material: _Material) -> dict[str, Any]:
    _normalise_relative_path(material.relative_path, label="package path")
    if (material.payload is None) == (material.source is None):
        raise AssertionError("material must contain exactly one source")
    if material.payload is not None:
        payload = material.payload
        return {
            "kind": material.kind,
            "role": material.role,
            "path": material.relative_path,
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
    if material.expected_bytes is None or material.expected_sha256 is None:
        raise AssertionError("file material requires expected size and digest")
    return {
        "kind": material.kind,
        "role": material.role,
        "path": material.relative_path,
        "bytes": material.expected_bytes,
        "sha256": material.expected_sha256,
    }


def _entry_key(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()


def _validate_materials(materials: list[_Material]) -> list[dict[str, Any]]:
    allowed_paths = {
        MODEL_PATH,
        EVIDENCE_PIN_PATH,
        RUNTIME_CONFIG_PATH,
        LAUNCHER_PATH,
        MODEL_CARD_PATH,
        ROLLBACK_PATH,
    }
    if {item.relative_path for item in materials} != allowed_paths:
        raise ModelBankV6Error("package material set violates the path allowlist")
    seen_paths: set[str] = set()
    seen_roles: set[str] = set()
    records = []
    for material in materials:
        key = _entry_key(material.relative_path)
        if key in seen_paths:
            raise ModelBankV6Error("duplicate normalized package path")
        if material.role in seen_roles:
            raise ModelBankV6Error("duplicate package role")
        seen_paths.add(key)
        seen_roles.add(material.role)
        records.append(_material_record(material))
    return sorted(records, key=lambda row: _entry_key(row["path"]))


def _content_descriptor(
    *,
    release_id: str,
    input_pins: dict[str, Any],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": PACKAGE_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "package_kind": PACKAGE_KIND,
        "status": PACKAGE_STATUS,
        "content_algorithm": "sha256",
        "product_id": PRODUCT_ID,
        "release_id": release_id,
        "default_enabled": False,
        "activated": False,
        "autostart": False,
        "service_registered": False,
        "production_dependency": False,
        "production_files_modified": False,
        "bpu_claimed": False,
        "rb_voe_dependency": False,
        "commands": COMMAND_POLICY,
        "protected_runtime": PROTECTED_RUNTIME,
        "resource_budget": RESOURCE_BUDGET,
        "input_pins": input_pins,
        "entries": entries,
        "rollback": ROLLBACK_POLICY,
    }


def _package_manifest(
    *,
    release_id: str,
    input_pins: dict[str, Any],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    descriptor = _content_descriptor(
        release_id=release_id,
        input_pins=input_pins,
        entries=entries,
    )
    content_id = _sha256_bytes(canonical_json(descriptor).encode("ascii"))
    archive_root = (ARCHIVE_FAMILY_ROOT / content_id).as_posix()
    return {
        **descriptor,
        "content_id": content_id,
        "archive_root": archive_root,
        "internal_manifest_path": f"{archive_root}/{INTERNAL_MANIFEST_PATH}",
        "target_install_path": f"{TARGET_FAMILY_ROOT}/{content_id}",
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = FIXED_FILE_MODE << 16
    info.extra = b""
    info.comment = b""
    return info


def _write_file_member(
    bundle: zipfile.ZipFile,
    *,
    archive_path: str,
    source: Path,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    digest = hashlib.sha256()
    total = 0
    before = source.stat(follow_symlinks=False)
    with source.open("rb") as handle, bundle.open(
        _zip_info(archive_path),
        mode="w",
        force_zip64=True,
    ) as destination:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            total += len(block)
            digest.update(block)
            destination.write(block)
    after = source.stat(follow_symlinks=False)
    if (
        total != expected_bytes
        or digest.hexdigest() != expected_sha256
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
    ):
        raise ModelBankV6Error("source artifact changed during ZIP construction")


def _prepare_model_bank_root(workspace_root: Path, supplied: Path) -> Path:
    expected = workspace_root.joinpath(*MODEL_BANK_RELATIVE_ROOT.parts)
    raw = Path(supplied)
    absolute = raw if raw.is_absolute() else workspace_root / raw
    if absolute.absolute() != expected.absolute():
        raise ModelBankV6Error(
            "output_root is outside the independent finals model-bank allowlist"
        )
    _ensure_no_link_chain(workspace_root, expected, label="output_root")
    expected.mkdir(parents=True, exist_ok=True)
    _ensure_no_link_chain(workspace_root, expected, label="output_root")
    if not expected.is_dir() or _is_reparse_or_symlink(expected):
        raise ModelBankV6Error("output_root must be a regular non-link directory")
    return expected.resolve(strict=True)


def _strict_pending_resource_budget(value: Any) -> None:
    if value != RESOURCE_BUDGET:
        raise ModelBankV6Error("resource budget must remain board-measurement pending")
    for key in (
        "latency_ms",
        "peak_rss_bytes",
        "steady_rss_bytes",
        "cpu_threads",
        "thermal_celsius",
    ):
        row = value[key]
        if row["status"] != "PENDING_BOARD_MEASUREMENT" or row["value"] is not None:
            raise ModelBankV6Error(f"resource budget {key} was prefilled")


def build_model_bank_candidate_v6(
    *,
    inputs: ModelBankInputs,
    output_root: Path,
) -> dict[str, Any]:
    """Build and independently verify one disabled content-addressed candidate."""

    try:
        workspace_root = Path(inputs.workspace_root).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ModelBankV6Error("workspace_root does not exist") from exc
    if not workspace_root.is_dir() or _is_reparse_or_symlink(workspace_root):
        raise ModelBankV6Error("workspace_root must be a regular non-link directory")

    final_path = _workspace_file(
        workspace_root,
        inputs.final_release_bundle,
        label="final release bundle",
    )
    gguf_dir = _workspace_directory(
        workspace_root,
        inputs.gguf_release_dir,
        label="GGUF release directory",
    )
    final = _verify_final_release_bundle(
        final_path,
        expected_sha256=inputs.final_release_bundle_sha256,
    )
    gguf = _verify_gguf_release(
        gguf_dir,
        expected_receipt_sha256=inputs.gguf_release_receipt_sha256,
        final_binding=final["gguf_binding"],
    )

    evidence_pin = _evidence_pin_bytes(
        release_id=final["release_id"],
        final_bundle_sha256=final["file_sha256"],
        gguf_receipt_sha256=gguf["receipt_sha256"],
        q4=gguf["q4"],
        parity=gguf["parity"],
        activation_marker=gguf["activation_marker"],
    )
    materials = [
        _Material(
            relative_path=MODEL_PATH,
            kind="MODEL",
            role="gguf_q4_k_m",
            source=gguf["q4_path"],
            expected_bytes=gguf["q4"]["bytes"],
            expected_sha256=gguf["q4"]["sha256"],
        ),
        _Material(
            relative_path=EVIDENCE_PIN_PATH,
            kind="CONTRACT",
            role="evidence_pin",
            payload=evidence_pin,
        ),
        _Material(
            relative_path=RUNTIME_CONFIG_PATH,
            kind="CONFIG",
            role="disabled_runtime_config",
            payload=_pretty_json(_runtime_config()),
        ),
        _Material(
            relative_path=LAUNCHER_PATH,
            kind="LAUNCHER",
            role="disabled_one_shot_launcher",
            payload=_launcher_bytes(),
        ),
        _Material(
            relative_path=MODEL_CARD_PATH,
            kind="DOCUMENT",
            role="model_card",
            payload=_model_card_bytes(
                release_id=final["release_id"],
                q4_sha256=gguf["q4"]["sha256"],
                final_bundle_sha256=final["file_sha256"],
                gguf_receipt_sha256=gguf["receipt_sha256"],
            ),
        ),
        _Material(
            relative_path=ROLLBACK_PATH,
            kind="CONTRACT",
            role="rollback_manifest",
            payload=_rollback_bytes(),
        ),
    ]
    entries = _validate_materials(materials)
    input_pins = {
        "final_release_bundle_sha256": final["file_sha256"],
        "gguf_release_receipt_sha256": gguf["receipt_sha256"],
        "gguf_q4_k_m_sha256": gguf["q4"]["sha256"],
        "gguf_q4_k_m_bytes": gguf["q4"]["bytes"],
        "gguf_parity_report_sha256": gguf["parity"]["sha256"],
        "gguf_activation_marker_sha256": gguf["activation_marker"]["sha256"],
    }
    manifest = _package_manifest(
        release_id=final["release_id"],
        input_pins=input_pins,
        entries=entries,
    )
    manifest_bytes = _pretty_json(manifest)
    bank_root = _prepare_model_bank_root(workspace_root, output_root)
    final_dir = bank_root / manifest["content_id"]
    if os.path.lexists(final_dir):
        raise FileExistsError(
            "content-addressed candidate already exists; overwrite is forbidden"
        )

    stage = Path(tempfile.mkdtemp(prefix=".candidate-", dir=bank_root))
    try:
        archive = stage / PACKAGE_ARCHIVE_NAME
        by_path = {material.relative_path: material for material in materials}
        with zipfile.ZipFile(
            archive,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as bundle:
            for row in entries:
                material = by_path[row["path"]]
                archive_path = f"{manifest['archive_root']}/{row['path']}"
                if material.payload is not None:
                    bundle.writestr(_zip_info(archive_path), material.payload)
                else:
                    assert material.source is not None
                    _write_file_member(
                        bundle,
                        archive_path=archive_path,
                        source=material.source,
                        expected_bytes=row["bytes"],
                        expected_sha256=row["sha256"],
                    )
            bundle.writestr(
                _zip_info(manifest["internal_manifest_path"]),
                manifest_bytes,
            )

        archive_sha256 = sha256_file(archive)
        (stage / PACKAGE_MANIFEST_NAME).write_bytes(manifest_bytes)
        (stage / ARCHIVE_SHA256_NAME).write_text(
            f"{archive_sha256}  {PACKAGE_ARCHIVE_NAME}\n",
            encoding="ascii",
            newline="\n",
        )
        os.replace(stage, final_dir)
        verification = verify_model_bank_candidate_v6(
            final_dir / PACKAGE_MANIFEST_NAME
        )
        return {
            "schema": "icmat_llm_model_bank_build_result.v6",
            "status": PACKAGE_STATUS,
            "content_id": manifest["content_id"],
            "package_directory": str(final_dir),
            "package_manifest": str(final_dir / PACKAGE_MANIFEST_NAME),
            "package_manifest_sha256": sha256_file(
                final_dir / PACKAGE_MANIFEST_NAME
            ),
            "archive": str(final_dir / PACKAGE_ARCHIVE_NAME),
            "archive_sha256": archive_sha256,
            "default_enabled": False,
            "deployed": False,
            "activated": False,
            "service_registered": False,
            "bpu_claimed": False,
            "rb_voe_dependency": False,
            "verification": verification,
        }
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        raise


def _load_canonical_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or _is_reparse_or_symlink(path):
        raise ModelBankV6Error("package manifest is missing or unsafe")
    payload = path.read_bytes()
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelBankV6Error("package manifest is not valid JSON") from exc
    if not isinstance(value, dict) or payload != _pretty_json(value):
        raise ModelBankV6Error("package manifest is not canonical")
    return value, payload


def _verify_manifest_contract(manifest: dict[str, Any]) -> None:
    descriptor_keys = set(
        _content_descriptor(
            release_id="fixture",
            input_pins={},
            entries=[],
        )
    )
    expected_keys = descriptor_keys | {
        "content_id",
        "archive_root",
        "internal_manifest_path",
        "target_install_path",
    }
    if set(manifest) != expected_keys:
        raise ModelBankV6Error("package manifest has unsupported or missing fields")
    if manifest["schema"] != PACKAGE_SCHEMA:
        raise ModelBankV6Error("unsupported package schema")
    if manifest["builder_version"] != BUILDER_VERSION:
        raise ModelBankV6Error("unsupported package builder version")
    if manifest["package_kind"] != PACKAGE_KIND:
        raise ModelBankV6Error("unsupported package kind")
    if manifest["status"] != PACKAGE_STATUS:
        raise ModelBankV6Error("package status is not board-pending inactive")
    if manifest["content_algorithm"] != "sha256":
        raise ModelBankV6Error("unsupported content algorithm")
    if manifest["product_id"] != PRODUCT_ID:
        raise ModelBankV6Error("package product_id mismatch")
    release_id = _require_safe_id(manifest["release_id"], label="release_id")
    for field in (
        "default_enabled",
        "activated",
        "autostart",
        "service_registered",
        "production_dependency",
        "production_files_modified",
        "bpu_claimed",
        "rb_voe_dependency",
    ):
        if manifest[field] is not False:
            raise ModelBankV6Error(f"package {field} must remain false")
    if manifest["commands"] != COMMAND_POLICY:
        raise ModelBankV6Error("package commands must remain empty")
    if manifest["protected_runtime"] != PROTECTED_RUNTIME:
        raise ModelBankV6Error("protected runtime contract changed")
    _strict_pending_resource_budget(manifest["resource_budget"])
    if manifest["rollback"] != ROLLBACK_POLICY:
        raise ModelBankV6Error("rollback policy changed")

    input_pins = manifest["input_pins"]
    expected_pin_keys = {
        "final_release_bundle_sha256",
        "gguf_release_receipt_sha256",
        "gguf_q4_k_m_sha256",
        "gguf_q4_k_m_bytes",
        "gguf_parity_report_sha256",
        "gguf_activation_marker_sha256",
    }
    if not isinstance(input_pins, dict) or set(input_pins) != expected_pin_keys:
        raise ModelBankV6Error("package input pins are invalid")
    for field in expected_pin_keys - {"gguf_q4_k_m_bytes"}:
        _require_sha256(input_pins[field], label=f"input_pins.{field}")
    if (
        not isinstance(input_pins["gguf_q4_k_m_bytes"], int)
        or isinstance(input_pins["gguf_q4_k_m_bytes"], bool)
        or input_pins["gguf_q4_k_m_bytes"] <= 0
    ):
        raise ModelBankV6Error("package Q4 byte-count pin is invalid")

    entries = manifest["entries"]
    if not isinstance(entries, list) or not entries:
        raise ModelBankV6Error("package entries are missing")
    paths = []
    roles = []
    for row in entries:
        if not isinstance(row, dict) or set(row) != {
            "kind",
            "role",
            "path",
            "bytes",
            "sha256",
        }:
            raise ModelBankV6Error("package entry has invalid fields")
        paths.append(_normalise_relative_path(row["path"], label="entry path"))
        if not isinstance(row["kind"], str) or not row["kind"]:
            raise ModelBankV6Error("package entry kind is invalid")
        if not isinstance(row["role"], str) or not row["role"]:
            raise ModelBankV6Error("package entry role is invalid")
        roles.append(row["role"])
        if (
            not isinstance(row["bytes"], int)
            or isinstance(row["bytes"], bool)
            or row["bytes"] <= 0
        ):
            raise ModelBankV6Error("package entry bytes are invalid")
        _require_sha256(row["sha256"], label="entry sha256")
    allowed_paths = {
        MODEL_PATH,
        EVIDENCE_PIN_PATH,
        RUNTIME_CONFIG_PATH,
        LAUNCHER_PATH,
        MODEL_CARD_PATH,
        ROLLBACK_PATH,
    }
    if set(paths) != allowed_paths:
        raise ModelBankV6Error("package entry paths violate the allowlist")
    if len({_entry_key(path) for path in paths}) != len(paths):
        raise ModelBankV6Error("package has duplicate normalized paths")
    if len(set(roles)) != len(roles):
        raise ModelBankV6Error("package has duplicate roles")
    if entries != sorted(entries, key=lambda row: _entry_key(row["path"])):
        raise ModelBankV6Error("package entries are not deterministically sorted")
    model_entry = next(row for row in entries if row["path"] == MODEL_PATH)
    if (
        model_entry["sha256"] != input_pins["gguf_q4_k_m_sha256"]
        or model_entry["bytes"] != input_pins["gguf_q4_k_m_bytes"]
    ):
        raise ModelBankV6Error("packaged model does not match its input pins")

    content_id = _require_sha256(manifest["content_id"], label="content_id")
    descriptor = {
        key: manifest[key]
        for key in _content_descriptor(
            release_id=release_id,
            input_pins={},
            entries=[],
        )
    }
    if _sha256_bytes(canonical_json(descriptor).encode("ascii")) != content_id:
        raise ModelBankV6Error("package content_id mismatch")
    expected_archive_root = (ARCHIVE_FAMILY_ROOT / content_id).as_posix()
    if manifest["archive_root"] != expected_archive_root:
        raise ModelBankV6Error("package archive_root mismatch")
    if (
        manifest["internal_manifest_path"]
        != f"{expected_archive_root}/{INTERNAL_MANIFEST_PATH}"
    ):
        raise ModelBankV6Error("internal manifest path mismatch")
    if (
        manifest["target_install_path"]
        != f"{TARGET_FAMILY_ROOT}/{content_id}"
    ):
        raise ModelBankV6Error("target install path mismatch")


def _read_archive_sidecar(package_dir: Path) -> str:
    sidecar = package_dir / ARCHIVE_SHA256_NAME
    if not sidecar.is_file() or _is_reparse_or_symlink(sidecar):
        raise ModelBankV6Error("archive SHA-256 sidecar is missing or unsafe")
    try:
        text = sidecar.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        raise ModelBankV6Error("archive SHA-256 sidecar must be ASCII") from exc
    match = re.fullmatch(
        rf"([0-9a-f]{{64}})  {re.escape(PACKAGE_ARCHIVE_NAME)}\n",
        text,
    )
    if not match:
        raise ModelBankV6Error("archive SHA-256 sidecar is invalid")
    return match.group(1)


def _validate_zip_info(info: zipfile.ZipInfo) -> None:
    _normalise_relative_path(info.filename, label="ZIP member")
    if info.is_dir():
        raise ModelBankV6Error("directory ZIP members are forbidden")
    if info.date_time != FIXED_ZIP_TIME:
        raise ModelBankV6Error("ZIP member timestamp is not deterministic")
    if info.compress_type != zipfile.ZIP_STORED:
        raise ModelBankV6Error("ZIP member compression is not deterministic")
    if info.extra or info.comment:
        raise ModelBankV6Error("ZIP member metadata is not canonical")
    mode = (info.external_attr >> 16) & 0xFFFF
    if mode != FIXED_FILE_MODE:
        raise ModelBankV6Error("ZIP member mode is not canonical")


def verify_model_bank_candidate_v6(package_manifest_path: Path) -> dict[str, Any]:
    """Independently verify an offline candidate from its sidecars and ZIP."""

    manifest_path = Path(package_manifest_path).absolute()
    package_dir = manifest_path.parent
    if _is_reparse_or_symlink(package_dir):
        raise ModelBankV6Error("package directory must not be a symlink")
    manifest, manifest_bytes = _load_canonical_manifest(manifest_path)
    _verify_manifest_contract(manifest)
    if package_dir.name != manifest["content_id"]:
        raise ModelBankV6Error("package directory is not content-addressed")
    children = sorted(path.name for path in package_dir.iterdir())
    if children != sorted(
        [ARCHIVE_SHA256_NAME, PACKAGE_ARCHIVE_NAME, PACKAGE_MANIFEST_NAME]
    ):
        raise ModelBankV6Error("package directory contains unexpected files")

    archive = package_dir / PACKAGE_ARCHIVE_NAME
    if not archive.is_file() or _is_reparse_or_symlink(archive):
        raise ModelBankV6Error("package archive is missing or unsafe")
    expected_archive_sha = _read_archive_sidecar(package_dir)
    actual_archive_sha = sha256_file(archive)
    if actual_archive_sha != expected_archive_sha:
        raise ModelBankV6Error("package archive SHA-256 mismatch")

    expected: dict[str, dict[str, Any]] = {}
    for row in manifest["entries"]:
        archive_path = f"{manifest['archive_root']}/{row['path']}"
        expected[_entry_key(archive_path)] = row
    internal_key = _entry_key(manifest["internal_manifest_path"])
    actual_keys: set[str] = set()
    retained: dict[str, bytes] = {}
    with zipfile.ZipFile(archive, mode="r") as bundle:
        if bundle.comment:
            raise ModelBankV6Error("ZIP archive comment is forbidden")
        for info in bundle.infolist():
            _validate_zip_info(info)
            key = _entry_key(info.filename)
            if key in actual_keys:
                raise ModelBankV6Error("duplicate normalized ZIP member")
            actual_keys.add(key)
            if key != internal_key and key not in expected:
                raise ModelBankV6Error("unexpected ZIP member")
            with bundle.open(info, mode="r") as handle:
                digest = hashlib.sha256()
                total = 0
                prefix = bytearray()
                keep = key == internal_key or (
                    key in expected and expected[key]["path"] != MODEL_PATH
                )
                payload = bytearray() if keep else None
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    total += len(block)
                    digest.update(block)
                    if len(prefix) < 4:
                        prefix.extend(block[: 4 - len(prefix)])
                    if payload is not None:
                        payload.extend(block)
            if key == internal_key:
                if bytes(payload or b"") != manifest_bytes:
                    raise ModelBankV6Error("internal manifest differs from sidecar")
                continue
            row = expected[key]
            if total != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ModelBankV6Error("ZIP artifact size or SHA-256 mismatch")
            if row["path"] == MODEL_PATH and bytes(prefix) != b"GGUF":
                raise ModelBankV6Error("packaged model has invalid GGUF magic")
            if payload is not None:
                retained[row["path"]] = bytes(payload)

    if actual_keys != set(expected) | {internal_key}:
        raise ModelBankV6Error("ZIP member set does not match the manifest")

    config = json.loads(retained[RUNTIME_CONFIG_PATH].decode("utf-8"))
    if config != _runtime_config():
        raise ModelBankV6Error("packaged disabled runtime config changed")
    rollback = json.loads(retained[ROLLBACK_PATH].decode("utf-8"))
    if rollback != json.loads(_rollback_bytes().decode("utf-8")):
        raise ModelBankV6Error("packaged rollback manifest changed")
    launcher = retained[LAUNCHER_PATH].decode("ascii")
    forbidden_launcher_terms = (
        "systemctl enable",
        "systemctl start",
        "service ",
        "dashboard.py",
        "start_x5.sh",
        "RB_VOE",
        "--port",
    )
    if any(term in launcher for term in forbidden_launcher_terms):
        raise ModelBankV6Error("packaged launcher violates isolation policy")
    if "ICMAT_FINALS_EXPLICIT_ONE_SHOT" not in launcher:
        raise ModelBankV6Error("packaged launcher is not explicitly gated")
    evidence = json.loads(retained[EVIDENCE_PIN_PATH].decode("utf-8"))
    if (
        evidence.get("schema") != EVIDENCE_PIN_SCHEMA
        or evidence.get("status") != PACKAGE_STATUS
        or evidence.get("authorization")
        != {
            "offline_candidate_packaging": True,
            "deployment": False,
            "activation": False,
            "production_integration": False,
        }
    ):
        raise ModelBankV6Error("packaged evidence pin is invalid")
    if (
        evidence["gguf_release"]["q4_gguf_sha256"]
        != manifest["input_pins"]["gguf_q4_k_m_sha256"]
    ):
        raise ModelBankV6Error("evidence pin Q4 hash mismatch")
    if (
        evidence["final_release_bundle"]["sha256"]
        != manifest["input_pins"]["final_release_bundle_sha256"]
        or evidence["gguf_release"]["receipt_sha256"]
        != manifest["input_pins"]["gguf_release_receipt_sha256"]
        or evidence["gguf_release"]["parity_report_sha256"]
        != manifest["input_pins"]["gguf_parity_report_sha256"]
        or evidence["gguf_release"]["activation_marker_sha256"]
        != manifest["input_pins"]["gguf_activation_marker_sha256"]
        or evidence["gguf_release"]["q4_gguf_bytes"]
        != manifest["input_pins"]["gguf_q4_k_m_bytes"]
    ):
        raise ModelBankV6Error("evidence pin does not match package input pins")

    return {
        "schema": VERIFICATION_SCHEMA,
        "ok": True,
        "status": PACKAGE_STATUS,
        "content_id": manifest["content_id"],
        "release_id": manifest["release_id"],
        "archive_sha256": actual_archive_sha,
        "artifact_count": len(manifest["entries"]),
        "default_enabled": False,
        "deployed": False,
        "activated": False,
        "service_registered": False,
        "production_files_modified": False,
        "bpu_claimed": False,
        "rb_voe_dependency": False,
        "board_resource_measurements": "PENDING_BOARD_MEASUREMENT",
    }


__all__ = [
    "ARCHIVE_SHA256_NAME",
    "FINAL_RELEASE_PASS_STATUS",
    "FINAL_RELEASE_SCHEMA",
    "GGUF_RELEASE_PASS_STATUS",
    "ModelBankInputs",
    "ModelBankV6Error",
    "MODEL_BANK_RELATIVE_ROOT",
    "PACKAGE_ARCHIVE_NAME",
    "PACKAGE_MANIFEST_NAME",
    "PACKAGE_STATUS",
    "RESOURCE_BUDGET",
    "build_model_bank_candidate_v6",
    "canonical_json",
    "sha256_file",
    "verify_model_bank_candidate_v6",
]
