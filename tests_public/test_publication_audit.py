from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDITOR_PATH = REPOSITORY_ROOT / "tools" / "publication" / "audit_release.py"
SPEC = importlib.util.spec_from_file_location("audit_release", AUDITOR_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import infrastructure guard
    raise RuntimeError("could not load publication auditor")
audit_release = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit_release
SPEC.loader.exec_module(audit_release)


def _pending_status() -> str:
    return "pending_" + "official_announcement"


def _write_award_status(root: Path, *, duplicate_marker: bool = False) -> None:
    target = root / "docs" / "competition" / "award_status.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = "\n# " + _pending_status() if duplicate_marker else ""
    target.write_text(
        "schema_version: 1\n"
        "national:\n"
        "  stage: national_final\n"
        f"  status: {_pending_status()}\n"
        "  result: null\n"
        "  source_url: null\n"
        "  evidence_path: null\n"
        "  announced_at: null\n"
        + suffix,
        encoding="utf-8",
    )


class PublicationAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        _write_award_status(self.root)
        (self.root / "valid.json").write_text(
            json.dumps({"status": "public"}), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def rules(self, result: object) -> set[str]:
        return {item.rule for item in result.findings}

    def test_clean_tree_and_auditor_source_do_not_self_trigger(self) -> None:
        destination = self.root / "tools" / "publication" / "audit_release.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(AUDITOR_PATH, destination)
        test_destination = self.root / "tests_public" / "test_publication_audit.py"
        test_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__), test_destination)
        result = audit_release.scan_tree(self.root)
        self.assertTrue(result.ok, result.findings)

    def test_detects_api_token_without_echoing_it(self) -> None:
        synthetic_token = "sk" + "-" + ("A" * 32)
        (self.root / "config.txt").write_text("api=" + synthetic_token, encoding="utf-8")
        result = audit_release.scan_tree(self.root)
        self.assertIn("api_token", self.rules(result))
        self.assertNotIn(synthetic_token, json.dumps(result.to_dict()))

    def test_blocks_hardcoded_baidu_credentials_but_allows_empty_env_fallback(self) -> None:
        variable_name = "BAIDU_" + "API_" + "KEY"
        unsafe_value = "nonempty-" + "fixture-value"
        source = self.root / "speech.py"
        source.write_text(
            f'{variable_name} = "{unsafe_value}"\n',
            encoding="utf-8",
        )
        result = audit_release.scan_tree(self.root)
        self.assertIn("baidu_credential_assignment", self.rules(result))
        self.assertNotIn(unsafe_value, json.dumps(result.to_dict()))

        source.write_text(
            f'{variable_name} = os.environ.get("{variable_name}", "")\n',
            encoding="utf-8",
        )
        result = audit_release.scan_tree(self.root)
        self.assertNotIn("baidu_credential_assignment", self.rules(result))

    def test_detects_private_key_and_host_key(self) -> None:
        private_header = "-----BEGIN " + "PRIVATE KEY-----"
        host_key = "ssh-" + "ed25519 " + ("A" * 64)
        (self.root / "keys.txt").write_text(
            private_header + "\n" + host_key, encoding="utf-8"
        )
        result = audit_release.scan_tree(self.root)
        self.assertTrue({"private_key", "host_key"} <= self.rules(result))

    def test_environment_lookup_and_credential_url_placeholder_handling(self) -> None:
        source = self.root / "settings.py"
        source.write_text(
            'API_KEY = os.environ.get("API_KEY")\n'
            'EMPTY = os.environ.get("API_KEY", "")\n'
            'TEMPLATE = os.environ.get("API_KEY", "${API_KEY}")\n'
            'EXAMPLE_URL = "https://user:password@example.invalid"\n',
            encoding="utf-8",
        )
        result = audit_release.scan_tree(self.root)
        self.assertTrue(result.ok, result.findings)

        fallback = "fallback_" + ("Z" * 24)
        source.write_text(
            'API_KEY = os.environ.get("API_KEY", "' + fallback + '")\n',
            encoding="utf-8",
        )
        result = audit_release.scan_tree(self.root)
        self.assertIn("credential_default", self.rules(result))
        self.assertNotIn(fallback, json.dumps(result.to_dict()))

        source.write_text(
            'ENDPOINT = "https://account:' + ("Q" * 24) + '@example.invalid"\n',
            encoding="utf-8",
        )
        result = audit_release.scan_tree(self.root)
        self.assertIn("credential_url", self.rules(result))

    def test_public_template_paths_are_allowed_but_real_user_paths_fail(self) -> None:
        public_posix = "/home/" + "rdk/project"
        public_windows = "C:" + "\\Users\\YOUR_USER\\project"
        path_file = self.root / "paths.txt"
        path_file.write_text(public_posix + "\n" + public_windows, encoding="utf-8")
        result = audit_release.scan_tree(self.root)
        self.assertTrue(result.ok, result.findings)

        private_windows = "C:" + "\\Users\\alice\\project"
        path_file.write_text(private_windows, encoding="utf-8")
        result = audit_release.scan_tree(self.root)
        self.assertIn("local_path", self.rules(result))

    def test_detects_private_ip_and_machine_local_path(self) -> None:
        private_ip = "192.168." + "1.9"
        local_path = "C:" + "\\Users\\alice\\workspace"
        (self.root / "machine.txt").write_text(
            private_ip + "\n" + local_path, encoding="utf-8"
        )
        result = audit_release.scan_tree(self.root)
        self.assertTrue({"private_ip", "local_path"} <= self.rules(result))

    def test_detects_forbidden_weight_and_configurable_large_file(self) -> None:
        (self.root / ("weights" + ".gguf")).write_bytes(b"weight")
        (self.root / "oversize.dat").write_bytes(b"x" * 65)
        result = audit_release.scan_tree(self.root, max_file_bytes=64)
        self.assertTrue({"forbidden_weight", "large_file"} <= self.rules(result))

    def test_detects_sensitive_file_and_directory_names(self) -> None:
        (self.root / ".env.local").write_text("placeholder", encoding="utf-8")
        secret_dir = self.root / "secrets"
        secret_dir.mkdir()
        (secret_dir / "opaque.txt").write_text("opaque", encoding="utf-8")
        result = audit_release.scan_tree(self.root)
        self.assertTrue(
            {"sensitive_filename", "sensitive_directory"} <= self.rules(result)
        )

    def test_detects_jpeg_exif_container(self) -> None:
        exif_payload = b"Exif\x00\x00" + b"II*\x00\x08\x00\x00\x00\x00\x00"
        segment = b"\xff\xe1" + (len(exif_payload) + 2).to_bytes(2, "big") + exif_payload
        (self.root / "photo.jpg").write_bytes(b"\xff\xd8" + segment + b"\xff\xd9")
        result = audit_release.scan_tree(self.root)
        self.assertIn("image_exif", self.rules(result))

    def test_detects_invalid_json(self) -> None:
        (self.root / "broken.json").write_text('{"missing":', encoding="utf-8")
        result = audit_release.scan_tree(self.root)
        self.assertIn("invalid_json", self.rules(result))

    def test_enforces_single_pending_award_boundary(self) -> None:
        (self.root / "README.md").write_text(_pending_status(), encoding="utf-8")
        result = audit_release.scan_tree(self.root)
        self.assertIn("award_pending_outside_ssot", self.rules(result))

        (self.root / "README.md").unlink()
        _write_award_status(self.root, duplicate_marker=True)
        result = audit_release.scan_tree(self.root)
        self.assertIn("award_pending_cardinality", self.rules(result))

    def test_announced_award_requires_evidence(self) -> None:
        target = self.root / "docs" / "competition" / "award_status.yaml"
        target.write_text(
            "national:\n"
            "  status: official_verified\n"
            "  result: first_prize\n"
            "  source_url: null\n"
            "  evidence_path: null\n"
            "  evidence_sha256: null\n"
            "  announced_at: null\n",
            encoding="utf-8",
        )
        result = audit_release.scan_tree(self.root)
        self.assertIn("award_announced_incomplete", self.rules(result))

    def test_ci_compatibility_arguments_are_accepted(self) -> None:
        args = audit_release._build_parser().parse_args(
            ["--root", str(self.root), "--strict"]
        )
        self.assertEqual(args.root_option, self.root)
        self.assertTrue(args.strict)


if __name__ == "__main__":
    unittest.main()
