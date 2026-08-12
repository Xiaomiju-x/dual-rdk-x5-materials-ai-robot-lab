#!/usr/bin/env python3
"""Producer-side event recorder for an A0 overhead acquisition manifest.

This module is instrumentation, not a camera or inference implementation. The
real arm02/AI-X5 runner must call each method immediately after the named
event. The recorder hashes the exact files produced by that event chain,
including the running arm02 camera-service identity, and emits the only
in-process attestation accepted by the production sealer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from workstation.dual_arm import rb_voe_overhead_actual_record as contract
except ModuleNotFoundError:  # pragma: no cover - direct device-side import
    import rb_voe_overhead_actual_record as contract  # type: ignore[no-redef]


@dataclass(frozen=True)
class ProducerHostIdentity:
    hostname: str
    device_id: str
    boot_id: str
    session_id: str

    def as_manifest_value(self) -> dict[str, str]:
        return {
            "hostname": self.hostname,
            "device_id": self.device_id,
            "boot_id": self.boot_id,
            "session_id": self.session_id,
        }


class A0AcquisitionRecorder:
    """Record one ordered camera-to-inference producer execution."""

    def __init__(
        self,
        *,
        expected: contract.ExpectedAcquisition,
        capture_host: ProducerHostIdentity,
        inference_host: ProducerHostIdentity,
        producer_started_at_ms: int,
    ) -> None:
        expected.validate()
        if capture_host.hostname != contract.FROZEN_CAPTURE_HOSTNAME:
            raise contract.ActualRecordError("capture producer must run for arm02 host er")
        if inference_host.hostname != contract.FROZEN_INFERENCE_HOSTNAME:
            raise contract.ActualRecordError("inference producer must run for host xrd-ai")
        identity_pairs = (
            (capture_host.device_id, expected.capture_device_id, "capture device"),
            (capture_host.boot_id, expected.capture_boot_id, "capture boot"),
            (capture_host.session_id, expected.capture_session_id, "capture session"),
            (inference_host.device_id, expected.inference_device_id, "inference device"),
            (inference_host.boot_id, expected.inference_boot_id, "inference boot"),
            (inference_host.session_id, expected.inference_session_id, "inference session"),
        )
        for actual, required, label in identity_pairs:
            if actual != required:
                raise contract.ActualRecordError(f"{label} does not match the R2 challenge")
        self._expected = expected
        self._capture_host = capture_host
        self._inference_host = inference_host
        self._producer_started_at_ms = producer_started_at_ms
        self._camera_service_identity: Path | None = None
        self._camera_opened_at_ms: int | None = None
        self._frame_captured_at_ms: int | None = None
        self._frame_bundle_bound_at_ms: int | None = None
        self._inference_started_at_ms: int | None = None
        self._inference_completed_at_ms: int | None = None
        self._frame: Path | None = None
        self._frame_bundle: Path | None = None
        self._result: Path | None = None
        self._capture_pipeline: Path | None = None
        self._inference_pipeline: Path | None = None
        self._production_attestation: contract.RunnerProductionAttestation | None = None

    def record_camera_service_identity(
        self,
        *,
        camera_service_identity_artifact: Path | str,
    ) -> None:
        if self._camera_service_identity is not None:
            raise contract.ActualRecordError("camera-service identity was already recorded")
        if self._camera_opened_at_ms is not None:
            raise contract.ActualRecordError("camera-service identity must be recorded before camera-open")
        self._camera_service_identity = Path(camera_service_identity_artifact)

    def record_camera_opened(self, *, opened_at_ms: int) -> None:
        if self._camera_service_identity is None:
            raise contract.ActualRecordError("camera-service identity must precede camera-open")
        if self._camera_opened_at_ms is not None:
            raise contract.ActualRecordError("camera-open event was already recorded")
        self._camera_opened_at_ms = opened_at_ms

    def record_frame_captured(
        self,
        *,
        raw_frame: Path | str,
        capture_pipeline_artifact: Path | str,
        captured_at_ms: int,
    ) -> None:
        if self._camera_opened_at_ms is None:
            raise contract.ActualRecordError("camera-open event must precede frame capture")
        if self._frame is not None:
            raise contract.ActualRecordError("frame-capture event was already recorded")
        self._frame = Path(raw_frame)
        self._capture_pipeline = Path(capture_pipeline_artifact)
        self._frame_captured_at_ms = captured_at_ms

    def record_input_frame_bundle(
        self,
        *,
        frame_bundle_artifact: Path | str,
        bound_at_ms: int,
    ) -> None:
        if self._frame is None:
            raise contract.ActualRecordError("frame capture must precede input-frame bundle binding")
        if self._frame_bundle is not None:
            raise contract.ActualRecordError("input-frame bundle event was already recorded")
        self._frame_bundle = Path(frame_bundle_artifact)
        self._frame_bundle_bound_at_ms = bound_at_ms

    def record_inference_started(self, *, started_at_ms: int) -> None:
        if self._frame_bundle is None:
            raise contract.ActualRecordError("a bound input-frame bundle must precede inference")
        if self._inference_started_at_ms is not None:
            raise contract.ActualRecordError("inference-start event was already recorded")
        self._inference_started_at_ms = started_at_ms

    def record_inference_completed(
        self,
        *,
        result_json: Path | str,
        inference_pipeline_artifact: Path | str,
        completed_at_ms: int,
    ) -> None:
        if self._inference_started_at_ms is None:
            raise contract.ActualRecordError("inference-start event must precede completion")
        if self._result is not None:
            raise contract.ActualRecordError("inference-completion event was already recorded")
        self._result = Path(result_json)
        self._inference_pipeline = Path(inference_pipeline_artifact)
        self._inference_completed_at_ms = completed_at_ms

    def _require_complete(self) -> tuple[Path, Path, Path, Path, Path, Path]:
        values = (
            self._frame,
            self._frame_bundle,
            self._result,
            self._capture_pipeline,
            self._inference_pipeline,
            self._camera_service_identity,
        )
        if any(value is None for value in values):
            raise contract.ActualRecordError("producer event chain is incomplete")
        return values  # type: ignore[return-value]

    def build(self, *, manifest_emitted_at_ms: int) -> dict[str, Any]:
        """Build and self-validate a manifest after the actual producer run."""

        (
            frame_path,
            frame_bundle_path,
            result_path,
            capture_path,
            inference_path,
            camera_service_identity_path,
        ) = self._require_complete()
        frame = contract._read_regular_file(frame_path, label="raw frame", max_bytes=contract.MAX_FRAME_BYTES)
        frame_bundle = contract._read_regular_file(
            frame_bundle_path,
            label="input frame bundle artifact",
            max_bytes=contract.MAX_INPUT_FRAME_BUNDLE_BYTES,
        )
        parsed_frame_bundle = contract._validate_input_frame_bundle_value(
            contract._load_json_bytes(frame_bundle.data, label="input frame bundle artifact")
        )
        result = contract._read_regular_file(
            result_path, label="result JSON", max_bytes=contract.MAX_RESULT_JSON_BYTES
        )
        capture = contract._read_regular_file(
            capture_path,
            label="capture pipeline artifact",
            max_bytes=contract.MAX_PIPELINE_ARTIFACT_BYTES,
        )
        inference = contract._read_regular_file(
            inference_path,
            label="inference pipeline artifact",
            max_bytes=contract.MAX_PIPELINE_ARTIFACT_BYTES,
        )
        camera_service_identity_file = contract._read_regular_file(
            camera_service_identity_path,
            label="camera service identity artifact",
            max_bytes=contract.MAX_CAMERA_SERVICE_IDENTITY_BYTES,
        )
        camera_service_identity = contract._validate_camera_service_identity_value(
            contract._load_json_bytes(
                camera_service_identity_file.data,
                label="camera service identity artifact",
            ),
            expected=self._expected,
        )
        width, height = contract.inspect_jpeg(frame.data)
        semantics = contract.inspect_result_json(
            result.data,
            raw_frame_sha256=frame.sha256,
            raw_frame_name=frame.path.name,
        )
        contract._validate_task_result_semantics(semantics, self._expected)
        contract._validate_frame_bundle_binding(
            parsed_frame_bundle,
            semantics,
            raw_frame=frame,
        )
        if (width, height) != (semantics.width, semantics.height):
            raise contract.ActualRecordError("producer frame dimensions differ from result")
        inference_contract = contract.INFERENCE_PIPELINE_CONTRACTS[semantics.schema]
        contract._validate_pipeline_artifact(
            capture,
            contract.CAPTURE_PIPELINE_CONTRACT,
            label="capture pipeline artifact",
        )
        contract._validate_pipeline_artifact(
            inference,
            inference_contract,
            label="inference pipeline artifact",
        )

        camera_opened = self._camera_opened_at_ms is not None
        frame_captured = self._frame_captured_at_ms is not None
        inference_started = self._inference_started_at_ms is not None
        inference_completed = self._inference_completed_at_ms is not None
        manifest: dict[str, Any] = {
            "schema_version": contract.ACQUISITION_SCHEMA_VERSION,
            "acquisition_id": self._expected.acquisition_id,
            "a0_run_id": self._expected.a0_run_id,
            "r2_run_id": self._expected.r2_run_id,
            "r2_run_nonce_sha256": hashlib.sha256(self._expected.r2_run_nonce.encode("utf-8")).hexdigest(),
            "challenge_sha256": self._expected.challenge_sha256,
            "challenge_issued_at_ms": self._expected.challenge_issued_at_ms,
            "challenge_expires_at_ms": self._expected.challenge_expires_at_ms,
            "replay_identity_sha256": contract.replay_identity_sha256_for_expected(
                self._expected,
                camera_service_identity_sha256=camera_service_identity.artifact_sha256,
            ),
            "release_id": self._expected.release_id,
            "config_sha256": self._expected.config_sha256,
            "case_id": self._expected.case_id,
            "sample_id": self._expected.sample_id,
            "sample_lineage_sha256": self._expected.sample_lineage_sha256,
            "parent_evidence_root_sha256": self._expected.parent_evidence_root_sha256,
            "bag_empty_baseline_sha256": self._expected.bag_empty_baseline_sha256,
            "task_kind": self._expected.task_kind,
            "result_schema": self._expected.result_schema,
            "success_state": self._expected.success_state,
            "dual_arm_semantic_profile_sha256": (contract.DUAL_ARM_SEMANTIC_PROFILE_SHA256),
            "capture_host": self._capture_host.as_manifest_value(),
            "inference_host": self._inference_host.as_manifest_value(),
            "camera": {
                "owner": contract.FROZEN_CAMERA_OWNER,
                "source": contract.FROZEN_CAMERA_SOURCE,
                "usb_id": contract.FROZEN_CAMERA_USB_ID,
                "backend": contract.FROZEN_BACKEND,
            },
            "camera_service_identity": contract._camera_service_identity_reference(
                camera_service_identity_file,
                camera_service_identity,
            ),
            "authority": {
                "domain": contract.ACQUISITION_AUTHORITY_DOMAIN,
                "hardware_touched": camera_opened and frame_captured,
                "camera_opened": camera_opened,
                "inference_triggered": inference_started and inference_completed,
                "motion_authority": False,
                "robot_sdk_opened": False,
                "serial_opened": False,
                "gpio_opened": False,
                "actuator_commands_issued": 0,
            },
            "events": {
                "producer_started_at_ms": self._producer_started_at_ms,
                "camera_service_identity_observed_at_ms": (camera_service_identity.observed_at_ms),
                "camera_opened_at_ms": self._camera_opened_at_ms,
                "frame_captured_at_ms": self._frame_captured_at_ms,
                "input_frame_bundle_bound_at_ms": self._frame_bundle_bound_at_ms,
                "inference_started_at_ms": self._inference_started_at_ms,
                "inference_completed_at_ms": self._inference_completed_at_ms,
                "manifest_emitted_at_ms": manifest_emitted_at_ms,
            },
            "frame": {
                "file_name": frame.path.name,
                "sha256": frame.sha256,
                "size_bytes": frame.size_bytes,
                "media_type": "image/jpeg",
                "width": width,
                "height": height,
            },
            "frame_bundle": {
                "file_name": frame_bundle.path.name,
                "file_sha256": frame_bundle.sha256,
                "size_bytes": frame_bundle.size_bytes,
                "schema": contract.INPUT_FRAME_BUNDLE_SCHEMA_VERSION,
                "bundle_sha256": parsed_frame_bundle.bundle_sha256,
                "entry_count": parsed_frame_bundle.entry_count,
                "total_bytes": parsed_frame_bundle.total_bytes,
            },
            "result": {
                "file_name": result.path.name,
                "sha256": result.sha256,
                "size_bytes": result.size_bytes,
                "schema": semantics.schema,
                "state": semantics.state,
                "success": semantics.success,
                "input_frame_sha256": frame.sha256,
                "baseline_sha256": semantics.baseline_sha256,
                "derivation_sha256": semantics.derivation_sha256,
            },
            "artifacts": {
                "capture_pipeline": contract._artifact_value(capture, contract.CAPTURE_PIPELINE_CONTRACT),
                "inference_pipeline": contract._artifact_value(inference, inference_contract),
                "model_contract": contract.NO_EXTERNAL_MODEL_CONTRACT,
                "models": [],
            },
        }
        manifest["manifest_sha256"] = contract.canonical_sha256(manifest)
        contract.validate_acquisition_manifest_value(
            manifest,
            raw_frame=frame_path,
            frame_bundle_artifact=frame_bundle_path,
            result_json=result_path,
            capture_pipeline_artifact=capture_path,
            inference_pipeline_artifact=inference_path,
            camera_service_identity_artifact=camera_service_identity_path,
            expected=self._expected,
            now_ms=manifest_emitted_at_ms,
        )
        return manifest

    def emit_once(self, output: Path | str, *, manifest_emitted_at_ms: int) -> dict[str, Any]:
        manifest = self.build(manifest_emitted_at_ms=manifest_emitted_at_ms)
        contract.write_json_once(output, manifest)
        if self._frame_bundle is None or self._camera_service_identity is None:
            raise contract.ActualRecordError("producer event chain lost required artifacts")
        self._production_attestation = contract._mint_runner_production_attestation(
            acquisition_manifest=output,
            camera_service_identity_artifact=self._camera_service_identity,
            frame_bundle_artifact=self._frame_bundle,
            expected=self._expected,
        )
        return manifest

    def production_attestation(self) -> contract.RunnerProductionAttestation:
        if self._production_attestation is None:
            raise contract.ActualRecordError(
                "production attestation is unavailable before successful manifest emission"
            )
        return self._production_attestation
