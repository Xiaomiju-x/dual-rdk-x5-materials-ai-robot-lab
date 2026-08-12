"""Fail-closed append-only JSONL ledger with a terminal integrity anchor.

The anchor detects suffix truncation before a later append. It is an integrity
anchor, not a signature; callers that need adversarial rewrite protection must
publish or sign the anchor outside this module.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from rb_voe.contracts.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    is_sha256,
    to_primitive,
)

LEDGER_SCHEMA = "xrd-rb-voe-audit-ledger-v1"
ANCHOR_SCHEMA = "xrd-rb-voe-audit-ledger-anchor-v1"
GENESIS_HASH = "0" * 64

ROW_FIELDS = frozenset(
    {
        "schema_version",
        "sequence",
        "record_id",
        "nonce",
        "record_type",
        "payload",
        "payload_sha256",
        "previous_record_sha256",
        "record_sha256",
    }
)
ANCHOR_FIELDS = frozenset(
    {
        "schema_version",
        "record_count",
        "terminal_record_sha256",
        "ledger_file_sha256",
        "anchor_sha256",
    }
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_NONCE_RE = re.compile(r"^[!-~]{1,256}$")


class AuditLedgerError(ValueError):
    """Raised when a ledger operation fails closed."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        line_number: int | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.line_number = line_number
        location = f" at line {line_number}" if line_number is not None else ""
        super().__init__(f"{code}{location}: {message}")

    @property
    def diagnostic(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.line_number is not None:
            result["line_number"] = self.line_number
        return result


class LedgerVerificationError(AuditLedgerError):
    """Raised with a machine-readable diagnostic when verification fails."""


def _verification_failure(
    code: str,
    message: str,
    *,
    line_number: int | None = None,
) -> LedgerVerificationError:
    return LedgerVerificationError(code, message, line_number=line_number)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _record_digest(row: Mapping[str, Any]) -> str:
    unsigned = dict(row)
    unsigned.pop("record_sha256", None)
    return canonical_sha256(unsigned)


def _anchor_digest(anchor: Mapping[str, Any]) -> str:
    unsigned = dict(anchor)
    unsigned.pop("anchor_sha256", None)
    return canonical_sha256(unsigned)


def _require_hash(value: object, field: str, *, line_number: int | None = None) -> str:
    if not is_sha256(value):
        raise _verification_failure(
            "HASH_FORMAT_INVALID",
            f"{field} must be a 64-character lowercase hexadecimal SHA-256 digest",
            line_number=line_number,
        )
    return value


def _require_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise AuditLedgerError(
            "IDENTIFIER_INVALID",
            f"{field} must be 1..128 characters using letters, digits, '.', '_', ':', or '-'",
        )
    return value


def _require_nonce(value: object) -> str:
    if not isinstance(value, str) or _NONCE_RE.fullmatch(value) is None:
        raise AuditLedgerError(
            "NONCE_INVALID",
            "nonce must contain 1..256 printable non-whitespace ASCII characters",
        )
    return value


def _absolute_path(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def terminal_anchor_path(ledger_path: str | Path) -> Path:
    """Return the terminal anchor path paired with ``ledger_path``."""
    return Path(str(_absolute_path(ledger_path)) + ".anchor.json")


def _lock_path(ledger_path: Path) -> Path:
    return Path(str(ledger_path) + ".lock")


@contextlib.contextmanager
def _exclusive_lock(ledger_path: Path) -> Iterator[None]:
    lock_path = _lock_path(ledger_path)
    descriptor = -1
    lock_created = False
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        lock_created = True
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
    except FileExistsError as exc:
        raise AuditLedgerError(
            "LEDGER_LOCKED",
            f"audit ledger is locked: {lock_path}",
        ) from exc
    except OSError:
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        if lock_created:
            lock_path.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _make_anchor(
    ledger_bytes: bytes,
    record_count: int,
    terminal_record_sha256: str,
) -> dict[str, Any]:
    anchor: dict[str, Any] = {
        "schema_version": ANCHOR_SCHEMA,
        "record_count": record_count,
        "terminal_record_sha256": terminal_record_sha256,
        "ledger_file_sha256": _sha256_bytes(ledger_bytes),
    }
    anchor["anchor_sha256"] = _anchor_digest(anchor)
    return anchor


def _write_anchor_atomic(path: Path, anchor: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(anchor) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def initialize_ledger(ledger_path: str | Path) -> Path:
    """Create a new empty ledger and fixed-genesis terminal anchor."""
    path = _absolute_path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anchor_path = terminal_anchor_path(path)
    with _exclusive_lock(path):
        if path.exists() or path.is_symlink() or anchor_path.exists() or anchor_path.is_symlink():
            raise AuditLedgerError(
                "OVERWRITE_FORBIDDEN",
                "ledger or terminal anchor already exists",
            )
        with path.open("xb") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        _write_anchor_atomic(anchor_path, _make_anchor(b"", 0, GENESIS_HASH))
    return path


def _validate_row_shape(row: Mapping[str, Any], line_number: int) -> None:
    if set(row) != ROW_FIELDS:
        missing = sorted(ROW_FIELDS - set(row))
        extra = sorted(set(row) - ROW_FIELDS)
        raise _verification_failure(
            "ROW_FIELDS_INVALID",
            f"row fields must match the v1 contract; missing={missing}, extra={extra}",
            line_number=line_number,
        )
    if row.get("schema_version") != LEDGER_SCHEMA:
        raise _verification_failure(
            "ROW_SCHEMA_INVALID",
            "unsupported ledger row schema",
            line_number=line_number,
        )
    sequence = row.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise _verification_failure(
            "SEQUENCE_INVALID",
            "sequence must be a positive integer",
            line_number=line_number,
        )
    for field in ("record_id", "record_type"):
        value = row.get(field)
        if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
            raise _verification_failure(
                "IDENTIFIER_INVALID",
                f"{field} is invalid",
                line_number=line_number,
            )
    nonce = row.get("nonce")
    if not isinstance(nonce, str) or _NONCE_RE.fullmatch(nonce) is None:
        raise _verification_failure(
            "NONCE_INVALID",
            "nonce is invalid",
            line_number=line_number,
        )
    if not isinstance(row.get("payload"), dict):
        raise _verification_failure(
            "PAYLOAD_INVALID",
            "payload must be a JSON object",
            line_number=line_number,
        )
    for field in ("payload_sha256", "previous_record_sha256", "record_sha256"):
        _require_hash(row.get(field), field, line_number=line_number)


def _replay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    previous_hash = GENESIS_HASH
    record_ids: set[str] = set()
    nonces: set[str] = set()

    for expected_sequence, row in enumerate(rows, start=1):
        _validate_row_shape(row, expected_sequence)
        if row["sequence"] != expected_sequence:
            raise _verification_failure(
                "REORDER_DETECTED",
                f"expected sequence {expected_sequence}, found {row['sequence']}",
                line_number=expected_sequence,
            )
        if row["payload_sha256"] != canonical_sha256(row["payload"]):
            raise _verification_failure(
                "PAYLOAD_TAMPER_DETECTED",
                "payload does not match payload_sha256",
                line_number=expected_sequence,
            )
        if row["record_id"] in record_ids:
            raise _verification_failure(
                "DUPLICATE_RECORD_ID",
                f"record_id was already used: {row['record_id']}",
                line_number=expected_sequence,
            )
        if row["nonce"] in nonces:
            raise _verification_failure(
                "DUPLICATE_NONCE",
                f"nonce was already used: {row['nonce']}",
                line_number=expected_sequence,
            )
        if row["previous_record_sha256"] != previous_hash:
            raise _verification_failure(
                "HASH_CHAIN_TAMPER_DETECTED",
                "previous_record_sha256 does not match the preceding record",
                line_number=expected_sequence,
            )
        if row["record_sha256"] != _record_digest(row):
            raise _verification_failure(
                "RECORD_TAMPER_DETECTED",
                "record does not match record_sha256",
                line_number=expected_sequence,
            )
        record_ids.add(row["record_id"])
        nonces.add(row["nonce"])
        previous_hash = row["record_sha256"]

    return {
        "record_count": len(rows),
        "terminal_record_sha256": previous_hash,
        "record_ids": record_ids,
        "nonces": nonces,
    }


def _parse_ledger(ledger_bytes: bytes) -> list[dict[str, Any]]:
    if not ledger_bytes:
        return []
    if not ledger_bytes.endswith(b"\n"):
        raise _verification_failure(
            "LEDGER_TRUNCATED_LINE",
            "ledger must end with a newline",
        )
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(ledger_bytes[:-1].split(b"\n"), start=1):
        if not raw_line:
            raise _verification_failure(
                "BLANK_ROW",
                "blank JSONL rows are forbidden",
                line_number=line_number,
            )
        try:
            decoded = raw_line.decode("utf-8")
            row = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _verification_failure(
                "ROW_JSON_INVALID",
                "ledger row is not valid UTF-8 JSON",
                line_number=line_number,
            ) from exc
        if not isinstance(row, dict):
            raise _verification_failure(
                "ROW_TYPE_INVALID",
                "ledger row must be a JSON object",
                line_number=line_number,
            )
        try:
            canonical = canonical_json_bytes(row)
        except (TypeError, ValueError) as exc:
            raise _verification_failure(
                "ROW_CANONICALIZATION_INVALID",
                "ledger row cannot be encoded canonically",
                line_number=line_number,
            ) from exc
        if raw_line != canonical:
            raise _verification_failure(
                "ROW_NOT_CANONICAL",
                "ledger row is not canonical JSON",
                line_number=line_number,
            )
        rows.append(row)
    return rows


def _parse_anchor(anchor_bytes: bytes) -> dict[str, Any]:
    if not anchor_bytes.endswith(b"\n") or anchor_bytes.count(b"\n") != 1:
        raise _verification_failure(
            "ANCHOR_FORMAT_INVALID",
            "terminal anchor must contain exactly one canonical JSON line",
        )
    try:
        decoded = json.loads(anchor_bytes[:-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _verification_failure(
            "ANCHOR_JSON_INVALID",
            "terminal anchor is not valid UTF-8 JSON",
        ) from exc
    if not isinstance(decoded, dict) or set(decoded) != ANCHOR_FIELDS:
        raise _verification_failure(
            "ANCHOR_FIELDS_INVALID",
            "terminal anchor fields do not match the v1 contract",
        )
    try:
        canonical = canonical_json_bytes(decoded)
    except (TypeError, ValueError) as exc:
        raise _verification_failure(
            "ANCHOR_CANONICALIZATION_INVALID",
            "terminal anchor cannot be encoded canonically",
        ) from exc
    if anchor_bytes[:-1] != canonical:
        raise _verification_failure(
            "ANCHOR_NOT_CANONICAL",
            "terminal anchor is not canonical JSON",
        )
    if decoded.get("schema_version") != ANCHOR_SCHEMA:
        raise _verification_failure(
            "ANCHOR_SCHEMA_INVALID",
            "unsupported terminal anchor schema",
        )
    record_count = decoded.get("record_count")
    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count < 0:
        raise _verification_failure(
            "ANCHOR_COUNT_INVALID",
            "terminal anchor record_count must be a non-negative integer",
        )
    for field in ("terminal_record_sha256", "ledger_file_sha256", "anchor_sha256"):
        _require_hash(decoded.get(field), field)
    if decoded["anchor_sha256"] != _anchor_digest(decoded):
        raise _verification_failure(
            "ANCHOR_TAMPER_DETECTED",
            "terminal anchor does not match anchor_sha256",
        )
    return decoded


def _verified_state(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], bytes]:
    anchor_path = terminal_anchor_path(path)
    if path.is_symlink() or not path.is_file():
        raise _verification_failure(
            "LEDGER_FILE_INVALID",
            "ledger must be an existing regular file",
        )
    if anchor_path.is_symlink() or not anchor_path.is_file():
        raise _verification_failure(
            "ANCHOR_FILE_INVALID",
            "terminal anchor must be an existing regular file",
        )
    try:
        ledger_bytes = path.read_bytes()
        anchor_bytes = anchor_path.read_bytes()
    except OSError as exc:
        raise _verification_failure(
            "AUDIT_FILES_UNREADABLE",
            "ledger or terminal anchor could not be read",
        ) from exc

    rows = _parse_ledger(ledger_bytes)
    state = _replay(rows)
    anchor = _parse_anchor(anchor_bytes)
    actual_count = state["record_count"]
    anchored_count = anchor["record_count"]
    if anchored_count > actual_count:
        raise _verification_failure(
            "TRUNCATION_DETECTED",
            f"terminal anchor binds {anchored_count} records but ledger contains {actual_count}",
        )
    if anchored_count < actual_count:
        raise _verification_failure(
            "ANCHOR_STALE",
            f"ledger contains {actual_count} records but terminal anchor binds {anchored_count}",
        )
    if anchor["terminal_record_sha256"] != state["terminal_record_sha256"]:
        raise _verification_failure(
            "TERMINAL_HASH_MISMATCH",
            "terminal record hash differs from the terminal anchor",
        )
    ledger_file_sha256 = _sha256_bytes(ledger_bytes)
    if anchor["ledger_file_sha256"] != ledger_file_sha256:
        raise _verification_failure(
            "LEDGER_DIGEST_MISMATCH",
            "ledger bytes differ from the terminal anchor",
        )

    report = {
        "ok": True,
        "schema_version": LEDGER_SCHEMA,
        "record_count": actual_count,
        "terminal_record_sha256": state["terminal_record_sha256"],
        "ledger_file_sha256": ledger_file_sha256,
        "ledger_digest": ledger_file_sha256,
        "anchor_sha256": anchor["anchor_sha256"],
        "diagnostics": [],
    }
    return report, rows, ledger_bytes


def verify_ledger(ledger_path: str | Path) -> dict[str, Any]:
    """Verify canonical bytes, full hash chain, uniqueness, and terminal anchor."""
    path = _absolute_path(ledger_path)
    report, _, _ = _verified_state(path)
    return report


def diagnose_ledger(ledger_path: str | Path) -> dict[str, Any]:
    """Return a non-throwing verification report with one precise diagnostic."""
    try:
        return verify_ledger(ledger_path)
    except AuditLedgerError as exc:
        return {
            "ok": False,
            "diagnostics": [exc.diagnostic],
        }


def append_record(
    ledger_path: str | Path,
    *,
    record_id: str,
    nonce: str,
    record_type: str,
    payload: Any,
) -> dict[str, Any]:
    """Append one canonical record after verifying the complete prior ledger."""
    checked_record_id = _require_identifier(record_id, "record_id")
    checked_nonce = _require_nonce(nonce)
    checked_record_type = _require_identifier(record_type, "record_type")
    try:
        checked_payload = to_primitive(payload)
    except (TypeError, ValueError) as exc:
        raise AuditLedgerError(
            "PAYLOAD_INVALID",
            "payload cannot be represented as canonical JSON",
        ) from exc
    if not isinstance(checked_payload, dict):
        raise AuditLedgerError("PAYLOAD_INVALID", "payload must encode to a JSON object")

    path = _absolute_path(ledger_path)
    if not path.parent.is_dir():
        raise AuditLedgerError("LEDGER_PARENT_INVALID", "ledger parent directory does not exist")
    with _exclusive_lock(path):
        _, rows, existing_bytes = _verified_state(path)
        state = _replay(rows)
        row: dict[str, Any] = {
            "schema_version": LEDGER_SCHEMA,
            "sequence": state["record_count"] + 1,
            "record_id": checked_record_id,
            "nonce": checked_nonce,
            "record_type": checked_record_type,
            "payload": checked_payload,
            "payload_sha256": canonical_sha256(checked_payload),
            "previous_record_sha256": state["terminal_record_sha256"],
        }
        row["record_sha256"] = _record_digest(row)
        _replay([*rows, row])

        encoded = canonical_json_bytes(row) + b"\n"
        with path.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        updated_bytes = existing_bytes + encoded
        _write_anchor_atomic(
            terminal_anchor_path(path),
            _make_anchor(updated_bytes, len(rows) + 1, row["record_sha256"]),
        )
        return row
