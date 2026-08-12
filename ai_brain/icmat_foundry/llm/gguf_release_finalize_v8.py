"""Finalize one disabled native-v8 GGUF candidate after offline parity.

This module does not export a model or execute a runtime.  It independently
recomputes the native-v8 parity receipt from its bound nonblind validation
observations, then emits the exact disabled release contract consumed by the
v8 pre-board replay planner.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icmat_foundry.llm import (
    gguf_release_v8,
    hf_gguf_observation_producer_v8,
    hf_gguf_parity_v8,
    x5_gguf_replay_v8,
)

VERSION = x5_gguf_replay_v8.RELEASE_VERSION
RELEASE_SCHEMA = x5_gguf_replay_v8.RELEASE_SCHEMA
RELEASE_STATUS = x5_gguf_replay_v8.RELEASE_STATUS

_PARITY_KEYS = {
    "schema",
    "version",
    "created_at_utc",
    "status",
    "preflight",
    "chain_binding",
    "low_level_export",
    "model",
    "dataset",
    "observations",
    "observation_authority",
    "metrics",
    "gates",
    "failing_example_ids",
    "integrity",
    "strict_pointer_and_compiler_parity",
    "legacy_v5_comparator_used",
    "authorization",
    "execution_boundary",
    "claim_boundary",
    "canonical_digest_sha256",
}
_OBSERVATION_AUTHORITY_KEYS = {
    "schema",
    "version",
    "status",
    "path",
    "bytes",
    "sha256",
    "canonical_digest_sha256",
    "provenance_kind",
    "request_set_sha256",
    "release_authority_digest_sha256",
    "producer_source_sha256",
    "producer_cli_sha256",
    "hf_runner_source_sha256",
    "llama_runner_source_sha256",
    "hf_program_sha256",
    "hf_stdout_sha256",
    "hf_stderr_sha256",
    "hf_exit_status",
    "gguf_program_sha256",
    "gguf_stdout_sha256",
    "gguf_stderr_sha256",
    "gguf_exit_status",
    "raw_authority_inputs_revalidated",
    "raw_process_artifacts_revalidated",
    "fixture_observations_used",
}


class GgufReleaseFinalizeV8Error(RuntimeError):
    """Raised when a v8 release candidate fails closed."""


@dataclass(frozen=True)
class FinalizeInputsV8:
    preflight_receipt: Path
    preflight_receipt_sha256: str
    export_receipt: Path
    export_receipt_sha256: str
    parity_receipt: Path
    parity_receipt_sha256: str
    gguf_model: Path
    gguf_model_sha256: str


def _exact(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise GgufReleaseFinalizeV8Error(f"{label} exact field set mismatch")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _verify_digest(value: Mapping[str, Any], *, label: str) -> None:
    try:
        claimed = gguf_release_v8._require_sha256(
            value.get("canonical_digest_sha256"),
            label=f"{label}.canonical_digest_sha256",
        )
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise GgufReleaseFinalizeV8Error(str(exc)) from exc
    body = dict(value)
    del body["canonical_digest_sha256"]
    if _digest(body) != claimed:
        raise GgufReleaseFinalizeV8Error(f"{label} canonical digest mismatch")


def _resolved_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise GgufReleaseFinalizeV8Error(f"{label} path is invalid")
    try:
        path = Path(value).resolve(strict=True)
    except OSError as exc:
        raise GgufReleaseFinalizeV8Error(f"{label} path is unavailable") from exc
    if not path.is_file():
        raise GgufReleaseFinalizeV8Error(f"{label} must be a file")
    return path


def _without_time_and_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("created_at_utc", None)
    result.pop("canonical_digest_sha256", None)
    return result


def _load_parity(
    inputs: FinalizeInputsV8,
) -> tuple[gguf_release_v8.FileSnapshotV8, Mapping[str, Any]]:
    try:
        snapshot, receipt = gguf_release_v8._load_json(
            inputs.parity_receipt,
            label="native-v8 parity receipt",
            expected_sha256=inputs.parity_receipt_sha256,
        )
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise GgufReleaseFinalizeV8Error("native-v8 parity receipt rejected") from exc
    _exact(receipt, _PARITY_KEYS, label="native-v8 parity receipt")
    _verify_digest(receipt, label="native-v8 parity receipt")
    if (
        receipt["schema"] != hf_gguf_parity_v8.PARITY_SCHEMA
        or receipt["version"] != hf_gguf_parity_v8.VERSION
        or receipt["status"] != hf_gguf_parity_v8.PARITY_PASS_STATUS
        or receipt["strict_pointer_and_compiler_parity"] is not True
        or receipt["legacy_v5_comparator_used"] is not False
        or receipt.get("gates", {}).get("all_passed") is not True
    ):
        raise GgufReleaseFinalizeV8Error("native-v8 parity did not pass")
    authorization = _exact(
        receipt["authorization"],
        {
            "pc_gguf_release_receipt_authorized",
            "x5_execution_authorized",
            "deployment_authorized",
            "production_integration_authorized",
        },
        label="native-v8 parity authorization",
    )
    if authorization != {
        "pc_gguf_release_receipt_authorized": True,
        "x5_execution_authorized": False,
        "deployment_authorized": False,
        "production_integration_authorized": False,
    }:
        raise GgufReleaseFinalizeV8Error("native-v8 parity authorization is unsafe")
    authority = _exact(
        receipt["observation_authority"],
        _OBSERVATION_AUTHORITY_KEYS,
        label="native-v8 parity observation authority",
    )
    if (
        authority["schema"]
        != hf_gguf_observation_producer_v8.AUTHORITY_SCHEMA
        or authority["version"]
        != hf_gguf_observation_producer_v8.VERSION
        or authority["status"]
        != hf_gguf_observation_producer_v8.AUTHORITY_STATUS
        or authority["provenance_kind"]
        != hf_gguf_observation_producer_v8.RUNTIME_PROVENANCE
        or authority["raw_authority_inputs_revalidated"] is not True
        or authority["raw_process_artifacts_revalidated"] is not True
        or authority["fixture_observations_used"] is not False
        or authority["hf_exit_status"] != 0
        or not isinstance(authority["gguf_exit_status"], int)
        or isinstance(authority["gguf_exit_status"], bool)
    ):
        raise GgufReleaseFinalizeV8Error(
            "native-v8 parity lacks controlled runtime authority"
        )
    for key, value in authority.items():
        if key.endswith("_sha256"):
            try:
                gguf_release_v8._require_sha256(
                    value,
                    label=f"native-v8 observation authority {key}",
                )
            except gguf_release_v8.GgufReleaseV8Error as exc:
                raise GgufReleaseFinalizeV8Error(str(exc)) from exc
    try:
        authority_snapshot = gguf_release_v8._snapshot_file(
            _resolved_file(authority["path"], label="runtime authority"),
            label="runtime authority",
            expected_sha256=str(authority["sha256"]),
        )
    except gguf_release_v8.GgufReleaseV8Error as exc:
        raise GgufReleaseFinalizeV8Error(
            "runtime observation authority artifact rejected"
        ) from exc
    if authority_snapshot.descriptor() != {
        "path": authority["path"],
        "bytes": authority["bytes"],
        "sha256": authority["sha256"],
    }:
        raise GgufReleaseFinalizeV8Error(
            "runtime observation authority artifact changed"
        )
    boundary = _exact(
        receipt["execution_boundary"],
        {
            "model_invoked_by_this_validator",
            "observation_files_consumed",
            "reserved_blind_read",
            "network_used",
            "x5_contacted",
            "production_services_touched",
        },
        label="native-v8 parity execution boundary",
    )
    if boundary != {
        "model_invoked_by_this_validator": False,
        "observation_files_consumed": True,
        "reserved_blind_read": False,
        "network_used": False,
        "x5_contacted": False,
        "production_services_touched": False,
    }:
        raise GgufReleaseFinalizeV8Error("native-v8 parity crossed its boundary")
    return snapshot, receipt


def _recompute_parity(
    inputs: FinalizeInputsV8,
    *,
    parity: Mapping[str, Any],
) -> Mapping[str, Any]:
    preflight = _exact(
        parity["preflight"],
        {"path", "bytes", "sha256", "authorization_digest_sha256"},
        label="native-v8 parity preflight",
    )
    export = _exact(
        parity["low_level_export"],
        {"path", "bytes", "sha256"},
        label="native-v8 parity export",
    )
    model = _exact(
        parity["model"],
        {
            "path",
            "bytes",
            "sha256",
            "filename",
            "format",
            "architecture",
            "quantization",
        },
        label="native-v8 parity model",
    )
    observations = _exact(
        parity["observations"],
        {"hf_selected_adapter", "gguf_q4_k_m"},
        label="native-v8 parity observations",
    )
    hf = _exact(
        observations["hf_selected_adapter"],
        {"path", "bytes", "sha256"},
        label="native-v8 HF observations",
    )
    gguf = _exact(
        observations["gguf_q4_k_m"],
        {"path", "bytes", "sha256"},
        label="native-v8 GGUF observations",
    )
    dataset = parity["dataset"]
    manifest = (
        dataset.get("manifest") if isinstance(dataset, Mapping) else None
    )
    if not isinstance(manifest, Mapping):
        raise GgufReleaseFinalizeV8Error("native-v8 parity dataset is incomplete")

    explicit_preflight = Path(inputs.preflight_receipt).resolve(strict=True)
    explicit_export = Path(inputs.export_receipt).resolve(strict=True)
    explicit_model = Path(inputs.gguf_model).resolve(strict=True)
    if (
        _resolved_file(preflight["path"], label="bound preflight")
        != explicit_preflight
        or _resolved_file(export["path"], label="bound export") != explicit_export
        or _resolved_file(model["path"], label="bound GGUF") != explicit_model
        or preflight["sha256"] != inputs.preflight_receipt_sha256
        or export["sha256"] != inputs.export_receipt_sha256
        or model["sha256"] != inputs.gguf_model_sha256
    ):
        raise GgufReleaseFinalizeV8Error(
            "explicit release inputs differ from the parity receipt"
        )
    recomputed = hf_gguf_parity_v8.verify_hf_gguf_parity_v8(
        hf_gguf_parity_v8.ParityInputsV8(
            preflight_receipt=explicit_preflight,
            preflight_receipt_sha256=inputs.preflight_receipt_sha256,
            dataset_dir=_resolved_file(
                manifest.get("path"),
                label="bound nonblind manifest",
            ).parent,
            export_receipt=explicit_export,
            export_receipt_sha256=inputs.export_receipt_sha256,
            gguf_model=explicit_model,
            gguf_model_sha256=inputs.gguf_model_sha256,
            hf_observations=_resolved_file(
                hf["path"],
                label="bound HF observations",
            ),
            hf_observations_sha256=str(hf["sha256"]),
            gguf_observations=_resolved_file(
                gguf["path"],
                label="bound GGUF observations",
            ),
            gguf_observations_sha256=str(gguf["sha256"]),
        )
    )
    if (
        recomputed["status"] != hf_gguf_parity_v8.PARITY_PASS_STATUS
        or _without_time_and_digest(recomputed)
        != _without_time_and_digest(parity)
    ):
        raise GgufReleaseFinalizeV8Error(
            "native-v8 parity could not be independently reproduced"
        )
    return recomputed


def finalize_release_v8(inputs: FinalizeInputsV8) -> dict[str, Any]:
    """Recompute parity and build one disabled PC release receipt."""

    try:
        parity_snapshot, parity = _load_parity(inputs)
        recomputed = _recompute_parity(inputs, parity=parity)
    except GgufReleaseFinalizeV8Error:
        raise
    except (
        hf_gguf_parity_v8.HfGgufParityV8Error,
        gguf_release_v8.GgufReleaseV8Error,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise GgufReleaseFinalizeV8Error(
            "native-v8 release inputs failed independent recomputation"
        ) from exc
    model = recomputed["model"]
    release = {
        "schema": RELEASE_SCHEMA,
        "version": VERSION,
        "status": RELEASE_STATUS,
        "preflight": {
            "schema": gguf_release_v8.PREFLIGHT_SCHEMA,
            "status": gguf_release_v8.PREFLIGHT_PASS_STATUS,
            "sha256": recomputed["preflight"]["sha256"],
            "authorization_digest_sha256": recomputed["preflight"][
                "authorization_digest_sha256"
            ],
        },
        "chain_binding": dict(recomputed["chain_binding"]),
        "model": {
            "filename": model["filename"],
            "bytes": model["bytes"],
            "sha256": model["sha256"],
            "format": "GGUF",
            "architecture": "qwen2",
            "quantization": "Q4_K_M",
        },
        "parity": {
            "schema": hf_gguf_parity_v8.PARITY_SCHEMA,
            "status": hf_gguf_parity_v8.PARITY_PASS_STATUS,
            "receipt_sha256": parity_snapshot.sha256,
            "strict_pointer_and_compiler_parity": True,
            "legacy_v5_comparator_used": False,
        },
        "runtime_policy": {
            "default_enabled": False,
            "autostart": False,
            "service_registered": False,
            "production_dependency": False,
            "production_files_modified": False,
            "x5_runtime_verified": False,
            "board_validation_pending": True,
        },
    }
    release["canonical_digest_sha256"] = _digest(release)
    return release


def write_release_receipt_v8(path: Path, receipt: Mapping[str, Any]) -> Path:
    """Write one immutable disabled release receipt."""

    _verify_digest(receipt, label="native-v8 GGUF release receipt")
    output = Path(path).expanduser().absolute()
    try:
        parent = output.parent.resolve(strict=True)
    except OSError as exc:
        raise GgufReleaseFinalizeV8Error("release output parent must exist") from exc
    if output.name in {"", ".", ".."} or os.path.lexists(output):
        raise GgufReleaseFinalizeV8Error("release output must be a new file")
    payload = (
        json.dumps(
            receipt,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        parent / output.name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if os.path.lexists(output):
            os.unlink(output)
        raise
    return output.resolve(strict=True)


__all__ = [
    "FinalizeInputsV8",
    "GgufReleaseFinalizeV8Error",
    "RELEASE_SCHEMA",
    "RELEASE_STATUS",
    "VERSION",
    "finalize_release_v8",
    "write_release_receipt_v8",
]
