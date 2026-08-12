from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from embodied_brain.finals_cortex.recorder import (
    MessageSample,
    Provenance,
    SampleSynchronizer,
    SessionRecorder,
    ValidationError,
    verify_manifest,
)
from embodied_brain.finals_cortex.recorder.integrity import IntegrityDetector
from embodied_brain.finals_cortex.recorder.session import ZERO_PUBLISHER_PERMISSIONS


class RecorderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.provenance = Provenance(
            state="live_sensor",
            source_id="lidar.front",
            device_id="ld14",
            clock_domain="monotonic.raw",
            capture_host="embodied-x5",
            metadata={"calibration_sha256": "a" * 64},
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_payload(self, relative: str, content: bytes) -> tuple[str, int]:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return hashlib.sha256(content).hexdigest(), len(content)

    def sample(
        self,
        stream: str,
        sequence: int,
        timestamp_ns: int,
        *,
        relative: str | None = None,
        content: bytes | None = None,
        declared_hash: str | None = None,
        provenance: Provenance | None = None,
    ) -> MessageSample:
        relative = relative or f"payload/{stream}.{sequence}.bin"
        content = content if content is not None else f"{stream}:{sequence}".encode()
        actual_hash, size = self.write_payload(relative, content)
        return MessageSample(
            stream=stream,
            message_type="sensor_msgs/msg/Test",
            sequence=sequence,
            timestamp_ns=timestamp_ns,
            received_timestamp_ns=timestamp_ns + 100,
            receive_clock_domain=(provenance or self.provenance).clock_domain,
            payload_file=relative,
            payload_sha256=declared_hash or actual_hash,
            payload_size_bytes=size,
            provenance=provenance or self.provenance,
        )

    def test_sample_validation_rejects_bad_hash_and_path_escape(self) -> None:
        digest, size = self.write_payload("payload/a.bin", b"a")
        common = {
            "stream": "scan",
            "message_type": "sensor_msgs/msg/LaserScan",
            "sequence": 0,
            "timestamp_ns": 1,
            "received_timestamp_ns": 2,
            "receive_clock_domain": "monotonic.raw",
            "payload_size_bytes": size,
            "provenance": self.provenance,
        }
        with self.assertRaises(ValidationError):
            MessageSample(
                payload_file="../a.bin", payload_sha256=digest, **common
            )
        with self.assertRaises(ValidationError):
            MessageSample(
                payload_file="payload/a.bin", payload_sha256="BAD", **common
            )

    def test_ros_topic_names_are_valid_stream_identifiers(self) -> None:
        sample = self.sample("/scan", 0, 1_000)
        self.assertEqual("/scan", sample.stream)

    def test_sequence_first_then_timestamp_fallback(self) -> None:
        samples = [
            self.sample("scan", 10, 1_000),
            self.sample("depth", 10, 1_040),
            self.sample("scan", 11, 2_000),
            self.sample("depth", 99, 2_030),
        ]
        result = SampleSynchronizer(
            ("scan", "depth"), anchor_stream="scan", tolerance_ns=50
        ).synchronize(samples)
        self.assertEqual(2, len(result.groups))
        self.assertEqual("sequence", result.groups[0].match_modes["depth"])
        self.assertEqual("timestamp", result.groups[1].match_modes["depth"])
        depth_stats = next(
            stats for stats in result.offset_stats if stats.stream == "depth"
        )
        self.assertEqual(35.0, depth_stats.mean_ns)
        self.assertEqual(2, depth_stats.count)

    def test_duplicate_out_of_order_and_missing_detection(self) -> None:
        detector = IntegrityDetector(("scan",), expected_start_sequences={"scan": 0})
        first = self.sample("scan", 0, 1_000)
        third = self.sample("scan", 2, 3_000)
        late = self.sample("scan", 1, 2_000)
        self.assertTrue(detector.add(first))
        self.assertTrue(detector.add(third))
        self.assertFalse(detector.add(third))
        self.assertTrue(detector.add(late))
        report = detector.report()
        codes = {issue.code for issue in report.issues}
        self.assertIn("duplicate_sequence", codes)
        self.assertIn("out_of_order_sequence", codes)
        self.assertIn("out_of_order_timestamp", codes)
        self.assertNotIn("missing_sequence", codes)
        self.assertTrue(report.valid)

    def test_missing_sequence_and_required_stream_are_errors(self) -> None:
        detector = IntegrityDetector(("scan", "imu"), {"scan": 0, "imu": 0})
        detector.add(self.sample("scan", 0, 1_000))
        detector.add(self.sample("scan", 2, 3_000))
        report = detector.report()
        codes = {issue.code for issue in report.issues}
        self.assertIn("missing_sequence", codes)
        self.assertIn("missing_stream", codes)
        self.assertFalse(report.valid)

    def test_conflicting_duplicate_is_critical(self) -> None:
        detector = IntegrityDetector(("scan",))
        detector.add(self.sample("scan", 0, 1_000, content=b"first"))
        accepted = detector.add(
            self.sample(
                "scan",
                0,
                1_000,
                relative="payload/scan.0.conflict.bin",
                content=b"second",
            )
        )
        self.assertFalse(accepted)
        report = detector.report()
        self.assertFalse(report.valid)
        self.assertIn(
            "conflicting_duplicate", {issue.code for issue in report.issues}
        )

    def test_finalize_writes_read_only_manifest_and_valid_hashes(self) -> None:
        recorder = SessionRecorder(
            "session-001",
            self.root,
            required_streams=("scan", "depth"),
            anchor_stream="scan",
            tolerance_ns=100,
            expected_start_sequences={"scan": 0, "depth": 0},
            metadata={"scenario": "stationary-contract-test"},
        )
        recorder.add_sample(self.sample("scan", 0, 1_000))
        recorder.add_sample(self.sample("depth", 0, 1_050))
        manifest_path = recorder.finalize()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        permissions = manifest["permissions"]
        self.assertEqual([], permissions["publishers"])
        self.assertEqual([], permissions["services"])
        self.assertEqual([], permissions["actions"])
        self.assertEqual([], permissions["tf_broadcasters"])
        self.assertEqual([], permissions["serial_devices"])
        self.assertFalse(permissions["control_authority"])
        self.assertTrue(manifest["integrity"]["valid"])
        self.assertEqual(1, manifest["synchronization"]["complete_group_count"])
        self.assertTrue(manifest_path.with_name("session_manifest.json.sha256").is_file())
        self.assertTrue(verify_manifest(manifest_path).valid)

    def test_json_contracts_parse_and_permission_contract_matches_runtime(self) -> None:
        contracts = (
            Path(__file__).parents[1] / "contracts"
        )
        policy = json.loads(
            (contracts / "recorder_permissions.v1.json").read_text(encoding="utf-8")
        )
        schema = json.loads(
            (contracts / "session_manifest.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(ZERO_PUBLISHER_PERMISSIONS, policy)
        self.assertEqual(
            "https://json-schema.org/draft/2020-12/schema", schema["$schema"]
        )
        self.assertEqual(
            "x5-real-sensor-session.v1",
            schema["properties"]["schema_version"]["const"],
        )

    def test_payload_tamper_is_detected_after_finalize(self) -> None:
        recorder = SessionRecorder(
            "session-002",
            self.root,
            required_streams=("scan",),
            anchor_stream="scan",
            tolerance_ns=0,
        )
        sample = self.sample("scan", 0, 1_000)
        recorder.add_sample(sample)
        manifest_path = recorder.finalize()
        (self.root / sample.payload_file).write_bytes(b"tampered")
        verification = verify_manifest(manifest_path)
        self.assertFalse(verification.valid)
        self.assertIn(
            f"payload_hash_mismatch:{sample.payload_file}", verification.issues
        )

    def test_declared_hash_tamper_is_recorded_in_manifest(self) -> None:
        recorder = SessionRecorder(
            "session-003",
            self.root,
            required_streams=("scan",),
            anchor_stream="scan",
            tolerance_ns=0,
        )
        recorder.add_sample(
            self.sample("scan", 0, 1_000, declared_hash="0" * 64)
        )
        manifest_path = recorder.finalize()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        codes = {issue["code"] for issue in manifest["integrity"]["issues"]}
        self.assertIn("payload_hash_mismatch", codes)
        self.assertFalse(manifest["integrity"]["valid"])

    def test_payload_alias_with_conflicting_declaration_is_critical(self) -> None:
        recorder = SessionRecorder(
            "session-alias",
            self.root,
            required_streams=("scan", "depth"),
            anchor_stream="scan",
            tolerance_ns=100,
        )
        shared = "payload/shared.bin"
        recorder.add_sample(
            self.sample("scan", 0, 1_000, relative=shared, content=b"shared")
        )
        recorder.add_sample(
            self.sample(
                "depth",
                0,
                1_010,
                relative=shared,
                content=b"shared",
                declared_hash="f" * 64,
            )
        )
        manifest = json.loads(recorder.finalize().read_text(encoding="utf-8"))
        codes = {issue["code"] for issue in manifest["integrity"]["issues"]}
        self.assertIn("payload_alias_conflict", codes)
        self.assertFalse(manifest["integrity"]["valid"])

    def test_manifest_tamper_breaks_sidecar(self) -> None:
        recorder = SessionRecorder(
            "session-004",
            self.root,
            required_streams=("scan",),
            anchor_stream="scan",
            tolerance_ns=0,
        )
        recorder.add_sample(self.sample("scan", 0, 1_000))
        manifest_path = recorder.finalize()
        manifest_path.write_text(
            manifest_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
        verification = verify_manifest(manifest_path)
        self.assertFalse(verification.valid)
        self.assertIn("manifest_hash_mismatch", verification.issues)

    def test_missing_synchronized_sample_is_recorded(self) -> None:
        recorder = SessionRecorder(
            "session-005",
            self.root,
            required_streams=("scan", "depth"),
            anchor_stream="scan",
            tolerance_ns=20,
        )
        recorder.add_sample(self.sample("scan", 0, 1_000))
        recorder.add_sample(self.sample("depth", 0, 2_000))
        manifest = json.loads(recorder.finalize().read_text(encoding="utf-8"))
        codes = {issue["code"] for issue in manifest["integrity"]["issues"]}
        self.assertIn("missing_synchronized_sample", codes)
        self.assertFalse(manifest["integrity"]["valid"])


if __name__ == "__main__":
    unittest.main()
