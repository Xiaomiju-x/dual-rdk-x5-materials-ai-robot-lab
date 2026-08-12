"""Fail-closed physical-start admission and durable replay protection."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from rb_voe.contracts.canonical import canonical_json_bytes, canonical_sha256, is_sha256
from rb_voe.contracts.models import ExecutionChallenge, JointPermit

PHYSICAL_AUTHORITY_DOMAIN = "SUPERVISED_TRIAL_AUTH"
PHYSICAL_CHALLENGE_SIGNATURE_DOMAIN = "xrd-rb-voe/physical-challenge/v1"
PHYSICAL_PERMIT_SIGNATURE_DOMAIN = "xrd-rb-voe/physical-permit/v1"
PHYSICAL_CHALLENGE_KEY_DOMAIN = "RB_VOE_EMBODIED_CHALLENGE_ED25519_V1"
PHYSICAL_PERMIT_KEY_DOMAIN = "RB_VOE_SUPERVISED_TRIAL_PERMIT_ED25519_V1"
SIMULATION_SIGNATURE_PREFIX = "SIMULATED_ONLY:"
R1_PHYSICAL_ADMISSION_REASON = "R1_PRODUCTION_PHYSICAL_AUTHORITY_NOT_IMPLEMENTED"
_ZERO_HASH = "0" * 64


class ChallengeSignatureVerifier(Protocol):
    """Verifier dedicated to the physical challenge signature domain."""

    def verify_challenge(
        self,
        challenge: ExecutionChallenge,
        *,
        signature_domain: str,
        authority_domain: str,
    ) -> tuple[bool, str]: ...


class PermitSignatureVerifier(Protocol):
    """Verifier dedicated to the physical permit signature domain."""

    def verify_permit(
        self,
        permit: JointPermit,
        *,
        signature_domain: str,
        authority_domain: str,
    ) -> tuple[bool, str]: ...


class RejectingChallengeSignatureVerifier:
    def verify_challenge(
        self,
        challenge: ExecutionChallenge,
        *,
        signature_domain: str,
        authority_domain: str,
    ) -> tuple[bool, str]:
        del challenge, signature_domain, authority_domain
        return False, "CHALLENGE_SIGNATURE_VERIFIER_NOT_CONFIGURED"


class RejectingPermitSignatureVerifier:
    def verify_permit(
        self,
        permit: JointPermit,
        *,
        signature_domain: str,
        authority_domain: str,
    ) -> tuple[bool, str]:
        del permit, signature_domain, authority_domain
        return False, "PERMIT_SIGNATURE_VERIFIER_NOT_CONFIGURED"


def _signature_payload(contract: ExecutionChallenge | JointPermit) -> dict[str, object]:
    payload = contract.to_dict()
    payload.pop("signature", None)
    return payload


def _test_signature(
    secret: bytes,
    contract: ExecutionChallenge | JointPermit,
    *,
    signature_domain: str,
    authority_domain: str,
    prefix: str,
) -> str:
    message = canonical_json_bytes(
        {
            "authority_domain": authority_domain,
            "contract": _signature_payload(contract),
            "signature_domain": signature_domain,
        }
    )
    return prefix + hmac.new(secret, message, hashlib.sha256).hexdigest()


class TestOnlyChallengeSignatureVerifier:
    """Content-bound test verifier; never configure it in a physical release."""

    PREFIX = "TEST_ONLY_PHYSICAL_CHALLENGE:"
    __test__ = False

    def __init__(self, secret: bytes) -> None:
        if not secret:
            raise ValueError("test verifier secret cannot be empty")
        self._secret = bytes(secret)

    def sign(
        self,
        challenge: ExecutionChallenge,
        *,
        authority_domain: str = PHYSICAL_AUTHORITY_DOMAIN,
    ) -> str:
        return _test_signature(
            self._secret,
            challenge,
            signature_domain=PHYSICAL_CHALLENGE_SIGNATURE_DOMAIN,
            authority_domain=authority_domain,
            prefix=self.PREFIX,
        )

    def verify_challenge(
        self,
        challenge: ExecutionChallenge,
        *,
        signature_domain: str,
        authority_domain: str,
    ) -> tuple[bool, str]:
        expected = _test_signature(
            self._secret,
            challenge,
            signature_domain=signature_domain,
            authority_domain=authority_domain,
            prefix=self.PREFIX,
        )
        if hmac.compare_digest(challenge.signature, expected):
            return True, "TEST_ONLY_CHALLENGE_SIGNATURE_VALID"
        return False, "CHALLENGE_SIGNATURE_INVALID"


class TestOnlyPermitSignatureVerifier:
    """Content-bound test verifier in a domain distinct from challenges."""

    PREFIX = "TEST_ONLY_PHYSICAL_PERMIT:"
    __test__ = False

    def __init__(self, secret: bytes) -> None:
        if not secret:
            raise ValueError("test verifier secret cannot be empty")
        self._secret = bytes(secret)

    def sign(
        self,
        permit: JointPermit,
        *,
        authority_domain: str = PHYSICAL_AUTHORITY_DOMAIN,
    ) -> str:
        return _test_signature(
            self._secret,
            permit,
            signature_domain=PHYSICAL_PERMIT_SIGNATURE_DOMAIN,
            authority_domain=authority_domain,
            prefix=self.PREFIX,
        )

    def verify_permit(
        self,
        permit: JointPermit,
        *,
        signature_domain: str,
        authority_domain: str,
    ) -> tuple[bool, str]:
        expected = _test_signature(
            self._secret,
            permit,
            signature_domain=signature_domain,
            authority_domain=authority_domain,
            prefix=self.PREFIX,
        )
        if hmac.compare_digest(permit.signature, expected):
            return True, "TEST_ONLY_PERMIT_SIGNATURE_VALID"
        return False, "PERMIT_SIGNATURE_INVALID"


class SimulationSignatureVerifier:
    """Simulation-only validation that is intentionally not a physical verifier."""

    PREFIX = SIMULATION_SIGNATURE_PREFIX

    def verify_simulation_permit(self, permit: JointPermit) -> tuple[bool, str]:
        if permit.signature.startswith(self.PREFIX):
            return True, "SIMULATED_SIGNATURE_VALID"
        return False, "SIMULATED_SIGNATURE_REQUIRED"


def validate_simulation_permit(
    permit: JointPermit,
    verifier: SimulationSignatureVerifier,
) -> tuple[bool, str]:
    """Validate a simulation watermark without granting execution authority."""
    return verifier.verify_simulation_permit(permit)


@dataclass(frozen=True, slots=True)
class TrustedPhysicalAdmissionContext:
    """Live values supplied by the trusted physical-start boundary."""

    boot_id: str
    session_id: str
    authority_domain: str
    operator_armed: bool
    case_id: str
    intent_id: str
    plan_epoch: int
    option_id: str
    challenge_id: str
    challenge_nonce: str
    permit_id: str
    attempt_id: str
    sample_lineage_sha256: str
    macro_id: str
    macro_contract_sha256: str
    command_envelope_sha256: str
    release_manifest_sha256: str
    embodied_manifest_sha256: str
    route_plan_sha256: str
    current_capability_hashes: tuple[str, ...]
    reserved_routes: tuple[str, ...]
    reserved_stations: tuple[str, ...]
    reserved_zones: tuple[str, ...]
    roles: Mapping[str, str]
    fallback: str
    challenge_key_domain: str
    permit_key_domain: str
    local_gate_states: Mapping[str, bool]

    def __post_init__(self) -> None:
        required_text = (
            self.boot_id,
            self.session_id,
            self.authority_domain,
            self.case_id,
            self.intent_id,
            self.option_id,
            self.challenge_id,
            self.challenge_nonce,
            self.permit_id,
            self.attempt_id,
            self.macro_id,
            self.fallback,
            self.challenge_key_domain,
            self.permit_key_domain,
        )
        if any(not item for item in required_text):
            raise ValueError("trusted admission identifiers cannot be empty")
        if self.plan_epoch < 0:
            raise ValueError("trusted plan_epoch cannot be negative")
        for name, digest in (
            ("sample_lineage_sha256", self.sample_lineage_sha256),
            ("macro_contract_sha256", self.macro_contract_sha256),
            ("command_envelope_sha256", self.command_envelope_sha256),
            ("release_manifest_sha256", self.release_manifest_sha256),
            ("embodied_manifest_sha256", self.embodied_manifest_sha256),
            ("route_plan_sha256", self.route_plan_sha256),
        ):
            if not is_sha256(digest):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not self.current_capability_hashes:
            raise ValueError("current capability hashes cannot be empty")
        if len(set(self.current_capability_hashes)) != len(self.current_capability_hashes):
            raise ValueError("current capability hashes cannot contain duplicates")
        if any(not is_sha256(digest) for digest in self.current_capability_hashes):
            raise ValueError("current capability hashes must be lowercase SHA-256 digests")
        if any(not isinstance(value, bool) for value in self.local_gate_states.values()):
            raise ValueError("local gate states must be booleans")
        if any(not key or not value for key, value in self.roles.items()):
            raise ValueError("trusted role bindings cannot be empty")


@dataclass(frozen=True, slots=True)
class PermitConsumeResult:
    accepted: bool
    reason_code: str
    permit_id: str
    attempt_id: str
    consumed: bool
    execution_deadline_ms: int | None = None
    execution_timeout_ms: int | None = None

    def __post_init__(self) -> None:
        if (
            self.accepted
            or self.consumed
            or self.execution_deadline_ms is not None
            or self.execution_timeout_ms is not None
        ):
            raise ValueError("R1 cannot construct a physical authorization result")

    def execution_authorized_at(self, now_ms: int) -> bool:
        del now_ms
        return False


class ReplayStoreError(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class _ReplayState:
    count: int
    terminal_sha256: str
    permit_ids: frozenset[str]
    attempt_ids: frozenset[str]
    challenge_nonces: frozenset[str]


class DurableReplayStore:
    """Boot/session-bound local replay ledger for simulation and unit tests.

    Exclusive claim files burn identifiers before admission returns. The ledger
    and terminal anchor then provide deterministic audit and rollback detection.
    A stale or partially restored store fails closed. This local filesystem
    store is not a production physical authority: replacing the whole directory
    also replaces its trust state, so R1 physical admission never accepts it.
    """

    SCHEMA_VERSION = "xrd-rb-voe-replay-store-v1"
    TRUST_TIER = "LOCAL_TEST_ONLY"

    def __init__(self, path: str | Path, *, boot_id: str, session_id: str) -> None:
        if not boot_id or not session_id:
            raise ValueError("replay store boot_id and session_id are required")
        self.path = Path(path)
        self.anchor_path = self.path.with_suffix(self.path.suffix + ".anchor")
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.claims_path = self.path.with_suffix(self.path.suffix + ".claims")
        self.boot_id = boot_id
        self.session_id = session_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.claims_path.mkdir(parents=True, exist_ok=True)
        with self._exclusive_lock():
            if not self.path.exists() and not self.anchor_path.exists():
                self._initialize()
            elif not self.path.exists() or not self.anchor_path.exists():
                raise ReplayStoreError("REPLAY_STORE_INCOMPLETE")
            self._load_verified()

    class _Lock:
        def __init__(self, owner: DurableReplayStore) -> None:
            self.owner = owner
            self.fd: int | None = None

        def __enter__(self) -> None:
            try:
                self.fd = os.open(
                    self.owner.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError as exc:
                raise ReplayStoreError("REPLAY_STORE_LOCKED") from exc
            os.write(self.fd, f"{os.getpid()}:{threading.get_ident()}".encode("ascii"))
            os.fsync(self.fd)

        def __exit__(self, exc_type, exc, traceback) -> None:
            del exc_type, exc, traceback
            if self.fd is not None:
                os.close(self.fd)
            try:
                self.owner.lock_path.unlink()
            except FileNotFoundError:
                pass

    def _exclusive_lock(self) -> DurableReplayStore._Lock:
        return self._Lock(self)

    def _initialize(self) -> None:
        base = {
            "boot_id": self.boot_id,
            "kind": "GENESIS",
            "previous_sha256": _ZERO_HASH,
            "schema_version": self.SCHEMA_VERSION,
            "sequence": 0,
            "session_id": self.session_id,
        }
        genesis = dict(base, record_sha256=canonical_sha256(base))
        self._atomic_write(self.path, canonical_json_bytes(genesis) + b"\n")
        self._write_anchor(genesis["record_sha256"], 1)

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with temp.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
            self._fsync_directory(path.parent)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def _anchor_payload(self, terminal_sha256: str, count: int) -> dict[str, object]:
        ledger_bytes = self.path.read_bytes()
        return {
            "boot_id": self.boot_id,
            "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
            "ledger_size": len(ledger_bytes),
            "record_count": count,
            "schema_version": self.SCHEMA_VERSION,
            "session_id": self.session_id,
            "terminal_sha256": terminal_sha256,
        }

    def _write_anchor(self, terminal_sha256: str, count: int) -> None:
        payload = self._anchor_payload(terminal_sha256, count)
        anchor = dict(payload, anchor_sha256=canonical_sha256(payload))
        self._atomic_write(self.anchor_path, canonical_json_bytes(anchor) + b"\n")

    def _load_verified(self) -> _ReplayState:
        raw = self.path.read_bytes()
        lines = raw.splitlines()
        if not lines or raw != b"\n".join(lines) + b"\n":
            raise ReplayStoreError("REPLAY_STORE_LEDGER_FORMAT_INVALID")
        permits: set[str] = set()
        attempts: set[str] = set()
        nonces: set[str] = set()
        previous = _ZERO_HASH
        for sequence, line in enumerate(lines):
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ReplayStoreError("REPLAY_STORE_LEDGER_JSON_INVALID") from exc
            if not isinstance(record, dict) or canonical_json_bytes(record) != line:
                raise ReplayStoreError("REPLAY_STORE_LEDGER_NOT_CANONICAL")
            stored_hash = record.get("record_sha256")
            payload = {key: value for key, value in record.items() if key != "record_sha256"}
            if not is_sha256(stored_hash) or canonical_sha256(payload) != stored_hash:
                raise ReplayStoreError("REPLAY_STORE_RECORD_HASH_MISMATCH")
            if record.get("schema_version") != self.SCHEMA_VERSION:
                raise ReplayStoreError("REPLAY_STORE_SCHEMA_MISMATCH")
            if record.get("boot_id") != self.boot_id or record.get("session_id") != self.session_id:
                raise ReplayStoreError("REPLAY_STORE_BOOT_SESSION_MISMATCH")
            if record.get("sequence") != sequence or record.get("previous_sha256") != previous:
                raise ReplayStoreError("REPLAY_STORE_CHAIN_MISMATCH")
            if sequence == 0:
                if record.get("kind") != "GENESIS":
                    raise ReplayStoreError("REPLAY_STORE_GENESIS_INVALID")
            else:
                if record.get("kind") != "CONSUMED":
                    raise ReplayStoreError("REPLAY_STORE_RECORD_KIND_INVALID")
                permit_id = record.get("permit_id")
                attempt_id = record.get("attempt_id")
                nonce = record.get("challenge_nonce")
                if not all(isinstance(item, str) and item for item in (permit_id, attempt_id, nonce)):
                    raise ReplayStoreError("REPLAY_STORE_IDENTIFIER_INVALID")
                if permit_id in permits or attempt_id in attempts or nonce in nonces:
                    raise ReplayStoreError("REPLAY_STORE_DUPLICATE_IDENTIFIER")
                permits.add(permit_id)
                attempts.add(attempt_id)
                nonces.add(nonce)
                self._require_claim("permit", permit_id)
                self._require_claim("attempt", attempt_id)
                self._require_claim("nonce", nonce)
            previous = stored_hash
        self._verify_anchor(previous, len(lines), raw)
        return _ReplayState(
            count=len(lines),
            terminal_sha256=previous,
            permit_ids=frozenset(permits),
            attempt_ids=frozenset(attempts),
            challenge_nonces=frozenset(nonces),
        )

    def _verify_anchor(self, terminal_sha256: str, count: int, ledger_bytes: bytes) -> None:
        try:
            raw = self.anchor_path.read_bytes()
            anchor = json.loads(raw)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ReplayStoreError("REPLAY_STORE_ANCHOR_INVALID") from exc
        if raw != canonical_json_bytes(anchor) + b"\n":
            raise ReplayStoreError("REPLAY_STORE_ANCHOR_NOT_CANONICAL")
        stored = anchor.get("anchor_sha256")
        payload = {key: value for key, value in anchor.items() if key != "anchor_sha256"}
        if not is_sha256(stored) or canonical_sha256(payload) != stored:
            raise ReplayStoreError("REPLAY_STORE_ANCHOR_HASH_MISMATCH")
        expected = {
            "boot_id": self.boot_id,
            "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
            "ledger_size": len(ledger_bytes),
            "record_count": count,
            "schema_version": self.SCHEMA_VERSION,
            "session_id": self.session_id,
            "terminal_sha256": terminal_sha256,
        }
        if payload != expected:
            raise ReplayStoreError("REPLAY_STORE_ANCHOR_MISMATCH")

    def _claim_path(self, kind: str, value: str) -> Path:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return self.claims_path / f"{kind}-{digest}.claim"

    def _claim_payload(self, kind: str, value: str) -> dict[str, object]:
        payload = {
            "boot_id": self.boot_id,
            "identifier_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            "kind": kind,
            "schema_version": self.SCHEMA_VERSION,
            "session_id": self.session_id,
        }
        return dict(payload, claim_sha256=canonical_sha256(payload))

    def _require_claim(self, kind: str, value: str) -> None:
        path = self._claim_path(kind, value)
        try:
            raw = path.read_bytes()
            claim = json.loads(raw)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ReplayStoreError("REPLAY_STORE_CLAIM_MISSING_OR_INVALID") from exc
        expected = self._claim_payload(kind, value)
        if raw != canonical_json_bytes(claim) + b"\n" or claim != expected:
            raise ReplayStoreError("REPLAY_STORE_CLAIM_MISMATCH")

    def _create_claim(self, kind: str, value: str) -> None:
        path = self._claim_path(kind, value)
        payload = canonical_json_bytes(self._claim_payload(kind, value)) + b"\n"
        try:
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            self._fsync_directory(path.parent)
        except FileExistsError as exc:
            reason = {
                "permit": "PERMIT_REPLAYED",
                "attempt": "ATTEMPT_REPLAYED",
                "nonce": "CHALLENGE_NONCE_REPLAYED",
            }[kind]
            raise ReplayStoreError(reason) from exc

    def consume(
        self,
        *,
        permit_id: str,
        attempt_id: str,
        challenge_nonce: str,
        admitted_at_ms: int,
        execution_deadline_ms: int,
    ) -> None:
        with self._exclusive_lock():
            state = self._load_verified()
            if permit_id in state.permit_ids:
                raise ReplayStoreError("PERMIT_REPLAYED")
            if attempt_id in state.attempt_ids:
                raise ReplayStoreError("ATTEMPT_REPLAYED")
            if challenge_nonce in state.challenge_nonces:
                raise ReplayStoreError("CHALLENGE_NONCE_REPLAYED")

            # Claims are intentionally burned first. A crash can refuse future
            # work, but can never make a physical authorization reusable.
            self._create_claim("permit", permit_id)
            self._create_claim("attempt", attempt_id)
            self._create_claim("nonce", challenge_nonce)

            base = {
                "admitted_at_ms": admitted_at_ms,
                "attempt_id": attempt_id,
                "boot_id": self.boot_id,
                "challenge_nonce": challenge_nonce,
                "execution_deadline_ms": execution_deadline_ms,
                "kind": "CONSUMED",
                "permit_id": permit_id,
                "previous_sha256": state.terminal_sha256,
                "schema_version": self.SCHEMA_VERSION,
                "sequence": state.count,
                "session_id": self.session_id,
            }
            record = dict(base, record_sha256=canonical_sha256(base))
            with self.path.open("ab") as handle:
                handle.write(canonical_json_bytes(record) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._write_anchor(record["record_sha256"], state.count + 1)
            self._load_verified()

    def diagnostic_snapshot(self) -> Mapping[str, object]:
        """Return diagnostics only; this payload cannot restore replay state."""
        with self._exclusive_lock():
            state = self._load_verified()
        return MappingProxyType(
            {
                "boot_id": self.boot_id,
                "session_id": self.session_id,
                "record_count": state.count,
                "terminal_sha256": state.terminal_sha256,
                "trust_tier": self.TRUST_TIER,
            }
        )

    @classmethod
    def from_snapshot(cls, *args, **kwargs) -> DurableReplayStore:
        del args, kwargs
        raise ReplayStoreError("REPLAY_STORE_SNAPSHOT_RESTORE_FORBIDDEN")


class PermitReplayGuard:
    """Fail-closed R1 physical boundary.

    R1 intentionally has no production signature verifier or externally
    monotonic replay backend. The complete preflight contract remains available
    for negative testing, but no verifier object supplied by application code
    can turn this class into physical execution authority.
    """

    def __init__(
        self,
        challenge_verifier: ChallengeSignatureVerifier | None = None,
        permit_verifier: PermitSignatureVerifier | None = None,
        *,
        replay_store: DurableReplayStore | None = None,
        mandatory_gates_by_macro: Mapping[str, tuple[str, ...]] | None = None,
        authority_domain: str = PHYSICAL_AUTHORITY_DOMAIN,
        challenge_key_domain: str = PHYSICAL_CHALLENGE_KEY_DOMAIN,
        permit_key_domain: str = PHYSICAL_PERMIT_KEY_DOMAIN,
    ) -> None:
        if isinstance(challenge_verifier, SimulationSignatureVerifier) or isinstance(
            permit_verifier, SimulationSignatureVerifier
        ):
            raise TypeError("simulation verifiers cannot be installed in physical admission")
        self._challenge_verifier = challenge_verifier or RejectingChallengeSignatureVerifier()
        self._permit_verifier = permit_verifier or RejectingPermitSignatureVerifier()
        self._replay_store = replay_store
        self._authority_domain = authority_domain
        self._challenge_key_domain = challenge_key_domain
        self._permit_key_domain = permit_key_domain
        gates = mandatory_gates_by_macro or {}
        self._mandatory_gates = MappingProxyType(
            {macro: tuple(sorted(set(values))) for macro, values in gates.items()}
        )

    @staticmethod
    def _rejected(permit: JointPermit, reason: str) -> PermitConsumeResult:
        return PermitConsumeResult(False, reason, permit.permit_id, permit.attempt_id, False)

    def consume_for_physical_start(
        self,
        challenge: ExecutionChallenge,
        permit: JointPermit,
        *,
        now_ms: int,
        context: TrustedPhysicalAdmissionContext,
    ) -> PermitConsumeResult:
        rejection = self._preflight_reason(challenge, permit, now_ms=now_ms, context=context)
        if rejection is not None:
            return self._rejected(permit, rejection)

        # R1 is replay/simulation only. Test HMAC verifiers and a local
        # filesystem replay ledger are useful for contract tests, but neither is
        # an independent production trust root. Keep this unconditional until a
        # hardware-backed verifier and externally monotonic consume backend are
        # implemented and separately reviewed.
        return self._rejected(permit, R1_PHYSICAL_ADMISSION_REASON)

    def consume_for_start(
        self,
        challenge: ExecutionChallenge,
        permit: JointPermit,
        *,
        now_ms: int,
        context: TrustedPhysicalAdmissionContext,
    ) -> PermitConsumeResult:
        """Compatibility alias with the same physical-only contract."""
        return self.consume_for_physical_start(challenge, permit, now_ms=now_ms, context=context)

    def _preflight_reason(
        self,
        challenge: ExecutionChallenge,
        permit: JointPermit,
        *,
        now_ms: int,
        context: TrustedPhysicalAdmissionContext,
    ) -> str | None:
        if challenge.signature.startswith(SIMULATION_SIGNATURE_PREFIX) or permit.signature.startswith(
            SIMULATION_SIGNATURE_PREFIX
        ):
            return "SIMULATION_SIGNATURE_FORBIDDEN_IN_PHYSICAL_ADMISSION"
        if context.authority_domain != self._authority_domain:
            return "AUTHORITY_DOMAIN_MISMATCH"
        if permit.authority_domain != self._authority_domain:
            return "PERMIT_AUTHORITY_DOMAIN_MISMATCH"
        if challenge.key_domain != self._challenge_key_domain or (
            context.challenge_key_domain != self._challenge_key_domain
        ):
            return "CHALLENGE_KEY_DOMAIN_MISMATCH"
        if permit.key_domain != self._permit_key_domain or (
            context.permit_key_domain != self._permit_key_domain
        ):
            return "PERMIT_KEY_DOMAIN_MISMATCH"
        if not context.operator_armed or not permit.operator_armed:
            return "OPERATOR_NOT_ARMED"
        if self._replay_store is not None and (
            context.boot_id != self._replay_store.boot_id
            or context.session_id != self._replay_store.session_id
        ):
            return "BOOT_SESSION_CHANGED"
        if now_ms < challenge.issued_at_ms:
            return "CHALLENGE_NOT_YET_VALID"
        if now_ms >= challenge.expires_at_ms:
            return "CHALLENGE_EXPIRED"
        if now_ms < permit.issued_at_ms:
            return "PERMIT_NOT_YET_VALID"
        if now_ms >= permit.start_expires_at_ms:
            return "PERMIT_START_EXPIRED"
        if not (challenge.issued_at_ms <= permit.issued_at_ms < challenge.expires_at_ms):
            return "PERMIT_ISSUANCE_OUTSIDE_CHALLENGE"
        expected_challenge = (
            (challenge.case_id, context.case_id, "CASE_BINDING_MISMATCH"),
            (challenge.intent_id, context.intent_id, "INTENT_BINDING_MISMATCH"),
            (challenge.plan_epoch, context.plan_epoch, "PLAN_EPOCH_MISMATCH"),
            (challenge.source_boot_id, context.boot_id, "CHALLENGE_BOOT_ID_MISMATCH"),
            (
                challenge.source_session_id,
                context.session_id,
                "CHALLENGE_SESSION_ID_MISMATCH",
            ),
            (challenge.challenge_id, context.challenge_id, "CHALLENGE_ID_MISMATCH"),
            (challenge.nonce, context.challenge_nonce, "CHALLENGE_NONCE_MISMATCH"),
            (
                challenge.embodied_manifest_sha256,
                context.embodied_manifest_sha256,
                "EMBODIED_MANIFEST_HASH_MISMATCH",
            ),
            (challenge.route_plan_sha256, context.route_plan_sha256, "ROUTE_PLAN_HASH_MISMATCH"),
            (challenge.reserved_routes, context.reserved_routes, "RESERVED_ROUTES_MISMATCH"),
            (
                challenge.reserved_stations,
                context.reserved_stations,
                "RESERVED_STATIONS_MISMATCH",
            ),
            (challenge.reserved_zones, context.reserved_zones, "RESERVED_ZONES_MISMATCH"),
        )
        for actual, expected, reason in expected_challenge:
            if actual != expected:
                return reason
        expected_permit = (
            (permit.case_id, challenge.case_id, "PERMIT_CASE_BINDING_MISMATCH"),
            (permit.intent_id, challenge.intent_id, "PERMIT_INTENT_BINDING_MISMATCH"),
            (permit.plan_epoch, challenge.plan_epoch, "PERMIT_PLAN_EPOCH_MISMATCH"),
            (permit.plan_epoch, context.plan_epoch, "PERMIT_CONTEXT_PLAN_EPOCH_MISMATCH"),
            (permit.option_id, context.option_id, "OPTION_ID_MISMATCH"),
            (permit.challenge_id, challenge.challenge_id, "PERMIT_CHALLENGE_BINDING_MISMATCH"),
            (permit.challenge_nonce, challenge.nonce, "PERMIT_CHALLENGE_NONCE_MISMATCH"),
            (permit.permit_id, context.permit_id, "PERMIT_ID_MISMATCH"),
            (permit.attempt_id, context.attempt_id, "ATTEMPT_ID_MISMATCH"),
            (
                permit.sample_lineage_sha256,
                context.sample_lineage_sha256,
                "SAMPLE_LINEAGE_HASH_MISMATCH",
            ),
            (permit.macro_id, context.macro_id, "MACRO_ID_MISMATCH"),
            (
                permit.macro_contract_sha256,
                context.macro_contract_sha256,
                "MACRO_CONTRACT_HASH_MISMATCH",
            ),
            (permit.roles, context.roles, "ROLE_BINDING_MISMATCH"),
            (permit.zones, context.reserved_zones, "PERMIT_ZONE_BINDING_MISMATCH"),
            (
                permit.command_envelope_sha256,
                context.command_envelope_sha256,
                "COMMAND_ENVELOPE_HASH_MISMATCH",
            ),
            (permit.fallback, context.fallback, "FALLBACK_MISMATCH"),
            (
                permit.release_manifest_sha256,
                context.release_manifest_sha256,
                "RELEASE_MANIFEST_HASH_MISMATCH",
            ),
        )
        for actual, expected, reason in expected_permit:
            if actual != expected:
                return reason
        if len(set(permit.required_capability_hashes)) != len(permit.required_capability_hashes):
            return "PERMIT_CAPABILITY_HASH_DUPLICATE"
        if tuple(sorted(permit.required_capability_hashes)) != tuple(
            sorted(context.current_capability_hashes)
        ):
            return "CAPABILITY_HASH_MISMATCH"
        if challenge.embodied_manifest_sha256 not in context.current_capability_hashes:
            return "EMBODIED_MANIFEST_NOT_IN_CAPABILITY_SET"
        mandatory = self._mandatory_gates.get(permit.macro_id)
        if mandatory is None or not mandatory:
            return "MANDATORY_GATE_POLICY_NOT_CONFIGURED"
        if tuple(sorted(set(permit.required_local_gates))) != mandatory:
            return "PERMIT_GATE_SET_MISMATCH"
        missing = [gate for gate in mandatory if gate not in context.local_gate_states]
        if missing:
            return "TRUSTED_LOCAL_GATE_MISSING:" + ",".join(missing)
        rejected = [gate for gate in mandatory if not context.local_gate_states[gate]]
        if rejected:
            return "TRUSTED_LOCAL_GATE_REJECTED:" + ",".join(rejected)
        return None


__all__ = [
    "ChallengeSignatureVerifier",
    "DurableReplayStore",
    "PHYSICAL_AUTHORITY_DOMAIN",
    "PHYSICAL_CHALLENGE_KEY_DOMAIN",
    "PHYSICAL_CHALLENGE_SIGNATURE_DOMAIN",
    "PHYSICAL_PERMIT_SIGNATURE_DOMAIN",
    "PHYSICAL_PERMIT_KEY_DOMAIN",
    "R1_PHYSICAL_ADMISSION_REASON",
    "PermitConsumeResult",
    "PermitReplayGuard",
    "PermitSignatureVerifier",
    "RejectingChallengeSignatureVerifier",
    "RejectingPermitSignatureVerifier",
    "ReplayStoreError",
    "SimulationSignatureVerifier",
    "TestOnlyChallengeSignatureVerifier",
    "TestOnlyPermitSignatureVerifier",
    "TrustedPhysicalAdmissionContext",
    "validate_simulation_permit",
]
