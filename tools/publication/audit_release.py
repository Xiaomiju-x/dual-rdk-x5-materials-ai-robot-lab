#!/usr/bin/env python3
"""Fail-closed, dependency-free audit for the public release tree.

The scanner deliberately reports only rule identifiers and relative paths.  It
never prints matched credential material.  Exit code 0 means the checked tree
passed; 1 means findings were detected; 2 means the audit could not run.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence


DEFAULT_MAX_FILE_BYTES = 50 * 1024 * 1024

# Model weights and compiled accelerator payloads are distributed only through
# separately licensed release channels, never through the source repository.
FORBIDDEN_WEIGHT_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".engine",
        ".gguf",
        ".h5",
        ".hbm",
        ".hdf5",
        ".mlmodel",
        ".onnx",
        ".plan",
        ".pt",
        ".pth",
        ".safetensors",
        ".tflite",
    }
)

IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
SKIPPED_DIRECTORY_NAMES = frozenset({".git"})
FORBIDDEN_DIRECTORY_NAMES = frozenset(
    {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)
CANONICAL_AWARD_PATH = Path("docs/competition/award_status.yaml")
PENDING_AWARD_STATUS = "pending_" + "official_announcement"
ANNOUNCED_AWARD_STATUS = "official_" + "verified"

PRIVATE_V4_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in (
        "10." + "0.0.0/8",
        "172." + "16.0.0/12",
        "192." + "168.0.0/16",
    )
)

# The expressions intentionally describe formats instead of embedding example
# secrets.  This keeps the scanner from detecting its own implementation.
KNOWN_TOKEN_RE = re.compile(
    r"(?:"
    r"\bAKIA[0-9A-Z]{16}\b|"
    r"\bAIza[0-9A-Za-z_-]{30,}\b|"
    r"\bgithub_pat_[0-9A-Za-z_]{20,}\b|"
    r"\bgh[pousr]_[0-9A-Za-z]{30,}\b|"
    r"\bhf_[0-9A-Za-z]{20,}\b|"
    r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b|"
    r"\b(?:sk|rk)_live_[0-9A-Za-z]{16,}\b|"
    r"\bsk-[0-9A-Za-z_-]{20,}\b"
    r")"
)
PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:OPENSSH |RSA |EC |DSA |PGP )?PRIVATE KEY-----"
)
CREDENTIAL_URL_RE = re.compile(
    r"\b(?:https?|ssh|postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://"
    r"(?P<username>[^\s/@:]{2,}):(?P<password>[^\s/@]{4,})@",
    re.IGNORECASE,
)
GENERIC_CREDENTIAL_RE = re.compile(
    r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|bearer[_-]?token|"
    r"client[_-]?secret|password|passwd|secret[_-]?key)\b"
    r"\s*[:=]\s*[\"']?([^\"'\s,;}{]{8,})",
    re.IGNORECASE,
)
ENV_DEFAULT_RE = re.compile(
    r"(?:os\.(?:getenv|environ\.get)|env\.get)\(\s*[\"'](?P<name>[^\"']+)[\"']"
    r"\s*,\s*[\"'](?P<value>[^\"']*)[\"']",
    re.IGNORECASE,
)
CREDENTIAL_ENV_NAME_RE = re.compile(
    r"(?:^|_)(?:API_?KEY|ACCESS_?TOKEN|AUTH_?TOKEN|BEARER_?TOKEN|"
    r"CLIENT_?SECRET|COOKIE|HMAC_?KEY|PASS(?:WORD|WD)?|PRIVATE_?KEY|"
    r"PWD|SECRET(?:_?KEY)?|TOKEN|WEBHOOK)(?:$|_)",
    re.IGNORECASE,
)
BAIDU_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"\b_?(?:DEFAULT_)?BAIDU_(?:TTS_)?(?:APP_ID|API_KEY|SECRET_KEY)\b"
    r"\s*=\s*[\"'](?P<value>[^\"']+)[\"']",
    re.IGNORECASE,
)
HOST_PUBLIC_KEY_RE = re.compile(
    r"\b(?:ssh-(?:rsa|ed25519)|ecdsa-sha2-nistp\d+)\s+"
    r"[A-Za-z0-9+/]{40,}={0,3}\b"
)
HOST_FINGERPRINT_RE = re.compile(
    r"(?:host\s*key|ED25519|ECDSA|RSA).{0,80}"
    r"SHA256:[A-Za-z0-9+/]{20,}={0,3}",
    re.IGNORECASE,
)
IPV4_RE = re.compile(r"(?<![0-9.])(?:\d{1,3}\.){3}\d{1,3}(?![0-9.])")
LOCAL_PATH_RES = (
    re.compile(r"\b[A-Za-z]:\\(?:\\)?Users\\(?:\\)?[^\\/\s\"']+", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9_])/(?:home|Users)/[^/\s\"']+"),
)
PUBLIC_PATH_PLACEHOLDER_USERS = frozenset(
    {"demo", "example", "rdk", "test", "user", "username", "your-user", "your_user"}
)

SENSITIVE_EXACT_NAMES = frozenset(
    {
        ".env",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "authorized_keys",
        "cookies",
        "credentials",
        "known_hosts",
        "login data",
    }
)
SENSITIVE_SUFFIXES = frozenset({".jks", ".key", ".p12", ".pem", ".pfx"})
SENSITIVE_DIRECTORY_NAMES = frozenset(
    {"authentication_file", "credentials", "secrets"}
)

PLACEHOLDER_MARKERS = (
    "${",
    "<",
    "changeme",
    "dummy",
    "example",
    "fake",
    "os.environ",
    "os.getenv",
    "process.env",
    "re.compile",
    "regex]::new",
    "redacted",
    "replace-me",
    "replace_me",
    "your-",
    "your_",
)


@dataclass(frozen=True, order=True)
class Finding:
    severity: str
    rule: str
    path: str
    line: int | None
    message: str


@dataclass
class AuditResult:
    files_scanned: int
    bytes_scanned: int
    findings: list[Finding]

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": "pass" if self.ok else "fail",
            "files_scanned": self.files_scanned,
            "bytes_scanned": self.bytes_scanned,
            "finding_count": len(self.findings),
            "findings": [asdict(item) for item in self.findings],
        }


class _FindingCollector:
    def __init__(self) -> None:
        self._items: list[Finding] = []
        self._keys: set[tuple[str, str, int | None]] = set()

    def add(
        self,
        severity: str,
        rule: str,
        path: str,
        message: str,
        line: int | None = None,
    ) -> None:
        key = (rule, path, line)
        if key in self._keys:
            return
        self._keys.add(key)
        self._items.append(Finding(severity, rule, path, line, message))

    def sorted(self) -> list[Finding]:
        severity_order = {"BLOCKER": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        return sorted(
            self._items,
            key=lambda item: (
                severity_order.get(item.severity, 99),
                item.path,
                item.line or 0,
                item.rule,
            ),
        )


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _iter_files(root: Path, collector: _FindingCollector) -> Iterator[Path]:
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = current_path / name
            if name in SKIPPED_DIRECTORY_NAMES:
                continue
            if name.lower() in SENSITIVE_DIRECTORY_NAMES:
                collector.add(
                    "BLOCKER",
                    "sensitive_directory",
                    _relative(candidate, root),
                    "Credential and secret directories are not publishable.",
                )
                continue
            if name in FORBIDDEN_DIRECTORY_NAMES or name.startswith(".venv-"):
                collector.add(
                    "BLOCKER",
                    "forbidden_directory",
                    _relative(candidate, root),
                    "Dependency, cache, or build directories are not publishable.",
                )
                continue
            if candidate.is_symlink():
                collector.add(
                    "BLOCKER",
                    "symlink",
                    _relative(candidate, root),
                    "Directory symlinks are not allowed in the publication tree.",
                )
                continue
            retained_directories.append(name)
        directory_names[:] = retained_directories

        for name in sorted(file_names):
            candidate = current_path / name
            if candidate.is_symlink():
                collector.add(
                    "BLOCKER",
                    "symlink",
                    _relative(candidate, root),
                    "File symlinks are not allowed in the publication tree.",
                )
                continue
            yield candidate


def _is_sensitive_name(path: Path) -> bool:
    lower_name = path.name.lower()
    if lower_name == ".env.example":
        return False
    if lower_name.startswith(".env."):
        return True
    if lower_name in SENSITIVE_EXACT_NAMES:
        return True
    if lower_name.startswith("ssh_host_") or lower_name.startswith("id_rsa"):
        return True
    if lower_name.startswith("id_ed25519") or lower_name.startswith("id_ecdsa"):
        return True
    return path.suffix.lower() in SENSITIVE_SUFFIXES


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().strip("\"'").lower()
    if normalized in {"none", "null", "true", "false"}:
        return True
    if not normalized or set(normalized) <= {"*", "x", "-", "_"}:
        return True
    return any(marker in normalized for marker in PLACEHOLDER_MARKERS)


def _credential_url_is_placeholder(match: re.Match[str]) -> bool:
    url_username = match.group("username").strip().lower()
    url_password = match.group("password").strip().lower()
    placeholder_users = {"corp-id", "demo", "example", "test", "user", "username"}
    placeholder_passwords = {
        "changeme",
        "demo",
        "example",
        "pass",
        "password",
        "secret",
        "test",
        "token",
    }
    return url_username in placeholder_users and url_password in placeholder_passwords


def _looks_textual(data: bytes) -> bool:
    if not data:
        return True
    if b"\x00" in data[:8192]:
        return False
    try:
        data[:65536].decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _private_ipv4s(line: str) -> Iterator[str]:
    for match in IPV4_RE.finditer(line):
        value = match.group(0)
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            continue
        if any(address in network for network in PRIVATE_V4_NETWORKS):
            yield value


def _local_path_is_public_template(match: re.Match[str]) -> bool:
    normalized = match.group(0).replace("\\", "/")
    parts = normalized.split("/")
    if len(parts) < 3:
        return False
    username = parts[2].strip().lower()
    return username in PUBLIC_PATH_PLACEHOLDER_USERS


def _scan_text(
    text: str,
    relative_path: str,
    collector: _FindingCollector,
) -> None:
    for line_number, line in enumerate(text.splitlines(), start=1):
        if KNOWN_TOKEN_RE.search(line):
            collector.add(
                "BLOCKER",
                "api_token",
                relative_path,
                "A value matching a known API/token format is present.",
                line_number,
            )
        if PRIVATE_KEY_RE.search(line):
            collector.add(
                "BLOCKER",
                "private_key",
                relative_path,
                "A private-key header is present.",
                line_number,
            )
        for match in CREDENTIAL_URL_RE.finditer(line):
            if not _credential_url_is_placeholder(match):
                collector.add(
                    "BLOCKER",
                    "credential_url",
                    relative_path,
                    "A URL with embedded credentials is present.",
                    line_number,
                )
                break
        if HOST_PUBLIC_KEY_RE.search(line) or HOST_FINGERPRINT_RE.search(line):
            collector.add(
                "HIGH",
                "host_key",
                relative_path,
                "An SSH host/public key or host-key fingerprint is present.",
                line_number,
            )

        for match in GENERIC_CREDENTIAL_RE.finditer(line):
            if not _is_placeholder(match.group(1)):
                collector.add(
                    "BLOCKER",
                    "credential_assignment",
                    relative_path,
                    "A non-placeholder credential assignment is present.",
                    line_number,
                )
                break

        for match in ENV_DEFAULT_RE.finditer(line):
            if CREDENTIAL_ENV_NAME_RE.search(match.group("name")) and not _is_placeholder(
                match.group("value")
            ):
                collector.add(
                    "BLOCKER",
                    "credential_default",
                    relative_path,
                    "An environment lookup contains a non-placeholder fallback credential.",
                    line_number,
                )
                break

        for match in BAIDU_CREDENTIAL_ASSIGNMENT_RE.finditer(line):
            if not _is_placeholder(match.group("value")):
                collector.add(
                    "BLOCKER",
                    "baidu_credential_assignment",
                    relative_path,
                    "A Baidu speech credential is hard-coded instead of read from the environment.",
                    line_number,
                )
                break

        if next(_private_ipv4s(line), None) is not None:
            collector.add(
                "HIGH",
                "private_ip",
                relative_path,
                "An RFC 1918 private IPv4 address is present.",
                line_number,
            )

        local_path_matches = (
            match for expression in LOCAL_PATH_RES for match in expression.finditer(line)
        )
        if any(not _local_path_is_public_template(match) for match in local_path_matches):
            collector.add(
                "HIGH",
                "local_path",
                relative_path,
                "A machine-local user path is present.",
                line_number,
            )


def _jpeg_has_exif(data: bytes) -> bool:
    if not data.startswith(b"\xff\xd8"):
        return False
    cursor = 2
    while cursor + 4 <= len(data):
        if data[cursor] != 0xFF:
            cursor += 1
            continue
        marker = data[cursor + 1]
        cursor += 2
        if marker in {0xD8, 0xD9}:
            continue
        if marker == 0xDA:
            break
        if cursor + 2 > len(data):
            break
        segment_length = int.from_bytes(data[cursor : cursor + 2], "big")
        if segment_length < 2 or cursor + segment_length > len(data):
            break
        payload = data[cursor + 2 : cursor + segment_length]
        if marker == 0xE1 and payload.startswith(b"Exif\x00\x00"):
            return True
        cursor += segment_length
    return False


def _tiff_has_sensitive_metadata(data: bytes) -> bool:
    if len(data) < 8 or data[:2] not in {b"II", b"MM"}:
        return False
    endian = "<" if data[:2] == b"II" else ">"
    try:
        ifd_offset = struct.unpack_from(endian + "I", data, 4)[0]
        if ifd_offset + 2 > len(data):
            return False
        entry_count = struct.unpack_from(endian + "H", data, ifd_offset)[0]
        sensitive_tags = {
            270,  # ImageDescription
            271,  # Make
            272,  # Model
            305,  # Software
            306,  # DateTime
            315,  # Artist
            33432,  # Copyright
            34665,  # Exif IFD
            34853,  # GPS IFD
        }
        for index in range(entry_count):
            offset = ifd_offset + 2 + index * 12
            if offset + 12 > len(data):
                break
            tag = struct.unpack_from(endian + "H", data, offset)[0]
            if tag in sensitive_tags:
                return True
    except (IndexError, struct.error):
        return False
    return False


def _image_has_exif(path: Path) -> bool:
    # EXIF containers are located near the beginning of these formats.  Four
    # MiB is ample while keeping the audit bounded for unusually large images.
    try:
        with path.open("rb") as handle:
            data = handle.read(4 * 1024 * 1024)
    except OSError:
        return False
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return _jpeg_has_exif(data)
    if suffix == ".png":
        return data.startswith(b"\x89PNG\r\n\x1a\n") and b"eXIf" in data
    if suffix == ".webp":
        return data.startswith(b"RIFF") and b"WEBP" in data[:16] and b"EXIF" in data
    if suffix in {".tif", ".tiff"}:
        return _tiff_has_sensitive_metadata(data)
    return False


def _parse_simple_yaml_section(text: str, section_name: str) -> dict[str, str]:
    """Parse immediate scalar keys from one top-level YAML mapping.

    This is intentionally narrow: the award file is a flat, human-editable
    contract.  Full YAML parsing would require a non-stdlib dependency.
    """

    result: dict[str, str] = {}
    in_section = False
    section_indent = 0
    key_value_re = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.*?)\s*$")
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if not in_section:
            if raw_line.strip() == section_name + ":":
                in_section = True
                section_indent = indent
            continue
        if indent <= section_indent:
            break
        match = key_value_re.match(raw_line)
        if not match:
            continue
        value = match.group(2).split(" #", 1)[0].strip().strip("\"'")
        result[match.group(1)] = value
    return result


def _is_yaml_null(value: str | None) -> bool:
    return value is None or value.strip().lower() in {"", "null", "~"}


def _check_official_award_evidence(
    root: Path,
    national: dict[str, str],
    canonical_rel: str,
    collector: _FindingCollector,
) -> None:
    required = ("result", "source_url", "evidence_path", "evidence_sha256", "announced_at")
    if any(_is_yaml_null(national.get(key)) for key in required):
        collector.add(
            "BLOCKER",
            "award_announced_incomplete",
            canonical_rel,
            "An official result requires result, source, evidence, hash, and announcement date.",
        )
        return

    source_url = national["source_url"]
    if not source_url.startswith("https://"):
        collector.add(
            "BLOCKER",
            "award_source_invalid",
            canonical_rel,
            "The official award source must use HTTPS.",
        )

    evidence_path = national["evidence_path"]
    candidate = (root / evidence_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        collector.add(
            "BLOCKER",
            "award_evidence_outside_tree",
            canonical_rel,
            "Award evidence must remain inside the publication tree.",
        )
        return
    if not candidate.is_file():
        collector.add(
            "BLOCKER",
            "award_evidence_missing",
            canonical_rel,
            "The declared award evidence file is missing.",
        )
        return

    expected_hash = national["evidence_sha256"]
    if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        collector.add(
            "BLOCKER",
            "award_evidence_hash_invalid",
            canonical_rel,
            "Award evidence SHA-256 must be 64 lowercase hexadecimal characters.",
        )
        return
    try:
        actual_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    except OSError:
        collector.add(
            "BLOCKER",
            "award_evidence_unreadable",
            canonical_rel,
            "The declared award evidence file could not be read.",
        )
        return
    if actual_hash != expected_hash:
        collector.add(
            "BLOCKER",
            "award_evidence_hash_mismatch",
            canonical_rel,
            "Award evidence SHA-256 does not match the declared digest.",
        )


def _check_award_boundary(
    root: Path,
    files: Sequence[Path],
    pending_locations: Sequence[str],
    collector: _FindingCollector,
) -> None:
    canonical = root / CANONICAL_AWARD_PATH
    canonical_rel = CANONICAL_AWARD_PATH.as_posix()
    competing_files = [
        _relative(path, root)
        for path in files
        if path.name.lower().startswith("award_status.")
        and _relative(path, root) != canonical_rel
    ]
    for path in competing_files:
        collector.add(
            "BLOCKER",
            "award_single_source",
            path,
            "Award status must have exactly one canonical source file.",
        )

    if not canonical.is_file():
        collector.add(
            "BLOCKER",
            "award_status_missing",
            canonical_rel,
            "The canonical competition award status file is missing.",
        )
        return
    try:
        text = canonical.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        collector.add(
            "BLOCKER",
            "award_status_unreadable",
            canonical_rel,
            "The canonical award status file is not readable UTF-8 text.",
        )
        return

    national = _parse_simple_yaml_section(text, "national")
    status = national.get("status")
    if status not in {PENDING_AWARD_STATUS, ANNOUNCED_AWARD_STATUS}:
        collector.add(
            "BLOCKER",
            "award_status_invalid",
            canonical_rel,
            "National status is not one of the permitted publication states.",
        )
        return

    outside_pending = [path for path in pending_locations if path != canonical_rel]
    for path in sorted(set(outside_pending)):
        collector.add(
            "BLOCKER",
            "award_pending_outside_ssot",
            path,
            "The pending national-award marker may appear only in the canonical file.",
        )

    pending_count_in_canonical = text.count(PENDING_AWARD_STATUS)
    if status == PENDING_AWARD_STATUS:
        if pending_count_in_canonical != 1:
            collector.add(
                "BLOCKER",
                "award_pending_cardinality",
                canonical_rel,
                "The canonical pending marker must occur exactly once.",
            )
        for key in (
            "result",
            "source_url",
            "evidence_path",
            "evidence_sha256",
            "announced_at",
        ):
            if not _is_yaml_null(national.get(key)):
                collector.add(
                    "BLOCKER",
                    "award_pending_boundary",
                    canonical_rel,
                    "Pending national status cannot carry a result or announcement evidence.",
                )
                break
    else:
        if pending_count_in_canonical or outside_pending:
            collector.add(
                "BLOCKER",
                "award_announced_has_pending",
                canonical_rel,
                "An officially announced award cannot retain a pending marker.",
            )
        _check_official_award_evidence(root, national, canonical_rel, collector)


def scan_tree(root: Path | str, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES) -> AuditResult:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError("audit root must be an existing directory")
    if max_file_bytes < 1:
        raise ValueError("max_file_bytes must be positive")

    collector = _FindingCollector()
    files = list(_iter_files(root_path, collector))
    files_scanned = 0
    bytes_scanned = 0
    pending_locations: list[str] = []

    for path in files:
        relative_path = _relative(path, root_path)
        files_scanned += 1
        try:
            size = path.stat().st_size
        except OSError:
            collector.add(
                "BLOCKER",
                "stat_error",
                relative_path,
                "A file could not be inspected.",
            )
            continue
        bytes_scanned += size

        if _is_sensitive_name(path):
            collector.add(
                "BLOCKER",
                "sensitive_filename",
                relative_path,
                "A credential, key, or SSH state filename is not publishable.",
            )
        if path.suffix.lower() in FORBIDDEN_WEIGHT_SUFFIXES:
            collector.add(
                "BLOCKER",
                "forbidden_weight",
                relative_path,
                "A model-weight or compiled-accelerator extension is forbidden.",
            )
        if size > max_file_bytes:
            collector.add(
                "BLOCKER",
                "large_file",
                relative_path,
                "The file exceeds the publication size limit.",
            )
            # A size violation is already fail-closed.  Do not load an
            # accidental multi-gigabyte model or video into memory merely to
            # produce additional findings for the same unpublishable file.
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES and _image_has_exif(path):
            collector.add(
                "HIGH",
                "image_exif",
                relative_path,
                "Image EXIF or equivalent embedded metadata must be stripped.",
            )

        try:
            data = path.read_bytes()
        except OSError:
            collector.add(
                "BLOCKER",
                "read_error",
                relative_path,
                "A file could not be read during the audit.",
            )
            continue

        if path.suffix.lower() == ".json":
            try:
                json.loads(data.decode("utf-8-sig"))
            except UnicodeError:
                collector.add(
                    "HIGH",
                    "invalid_json",
                    relative_path,
                    "JSON is not valid UTF-8 text.",
                )
            except json.JSONDecodeError as error:
                collector.add(
                    "HIGH",
                    "invalid_json",
                    relative_path,
                    "JSON syntax is invalid.",
                    error.lineno,
                )

        if not _looks_textual(data):
            continue
        try:
            text = data.decode("utf-8-sig")
        except UnicodeError:
            continue
        _scan_text(text, relative_path, collector)
        if PENDING_AWARD_STATUS in text:
            pending_locations.append(relative_path)

    _check_award_boundary(root_path, files, pending_locations, collector)
    return AuditResult(files_scanned, bytes_scanned, collector.sorted())


def _render_text(result: AuditResult) -> str:
    status = "PASS" if result.ok else "FAIL"
    lines = [
        f"PUBLICATION_AUDIT={status}",
        f"files_scanned={result.files_scanned}",
        f"bytes_scanned={result.bytes_scanned}",
        f"finding_count={len(result.findings)}",
    ]
    for finding in result.findings:
        location = finding.path
        if finding.line is not None:
            location += f":{finding.line}"
        lines.append(
            f"[{finding.severity}] {finding.rule} {location} - {finding.message}"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=None,
        help="release tree to audit (default: repository root)",
    )
    parser.add_argument(
        "--root",
        dest="root_option",
        type=Path,
        help="compatibility form of the positional release-tree argument",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="compatibility flag; the audit is always fail-closed and strict",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="machine-readable JSON or concise text output",
    )
    parser.add_argument(
        "--max-file-mib",
        type=float,
        default=50.0,
        help="maximum allowed file size in MiB (default: 50)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        max_file_bytes = int(args.max_file_mib * 1024 * 1024)
        root = args.root_option or args.root or Path(__file__).resolve().parents[2]
        result = scan_tree(root, max_file_bytes=max_file_bytes)
    except (OSError, ValueError) as error:
        print(f"PUBLICATION_AUDIT=ERROR\nreason={type(error).__name__}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(_render_text(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
