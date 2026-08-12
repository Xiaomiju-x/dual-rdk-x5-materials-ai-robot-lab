"""Read-only hard-case candidate mining with persistent dedupe and hash chain."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scene_graph import _finite_number, _json_object, _open_database

SCHEMA_VERSION = "x5-hard-case-candidate/1.0"
GENESIS_HASH = "0" * 64


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class MinerConfig:
    ood_threshold: float = 0.80
    disagreement_threshold: float = 0.50
    max_data_age_s: float = 0.50
    dedupe_window_s: float = 5.0

    def __post_init__(self) -> None:
        for name in (
            "ood_threshold",
            "disagreement_threshold",
            "max_data_age_s",
            "dedupe_window_s",
        ):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name),
            )
        if not 0.0 <= self.ood_threshold <= 1.0:
            raise ValueError("ood_threshold must be in [0, 1]")
        if not 0.0 <= self.disagreement_threshold <= 1.0:
            raise ValueError("disagreement_threshold must be in [0, 1]")
        if self.max_data_age_s <= 0.0:
            raise ValueError("max_data_age_s must be positive")
        if self.dedupe_window_s < 0.0:
            raise ValueError("dedupe_window_s must be non-negative")


@dataclass(frozen=True, slots=True)
class HardCaseCandidate:
    sequence: int
    candidate_id: str
    event_timestamp: float
    created_timestamp: float
    reasons: tuple[str, ...]
    fingerprint: str
    signal: Mapping[str, Any]
    episode: Mapping[str, Any]
    provenance: Mapping[str, Any]
    previous_hash: str
    record_hash: str

    def chain_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "sequence": self.sequence,
            "candidate_id": self.candidate_id,
            "event_timestamp": self.event_timestamp,
            "created_timestamp": self.created_timestamp,
            "reasons": list(self.reasons),
            "fingerprint": self.fingerprint,
            "signal": dict(self.signal),
            "episode": dict(self.episode),
            "provenance": dict(self.provenance),
            "previous_hash": self.previous_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.chain_payload(),
            "record_hash": self.record_hash,
            "shadow_only": True,
            "controls_devices": False,
            "online_training": False,
        }


@dataclass(frozen=True, slots=True)
class MiningDecision:
    triggered: bool
    reasons: tuple[str, ...]
    candidate: HardCaseCandidate | None = None
    duplicate: bool = False
    duplicate_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "reasons": list(self.reasons),
            "candidate": (
                None if self.candidate is None else self.candidate.to_dict()
            ),
            "duplicate": self.duplicate,
            "duplicate_of": self.duplicate_of,
            "shadow_only": True,
            "controls_devices": False,
            "online_training": False,
        }


@dataclass(frozen=True, slots=True)
class ChainVerification:
    valid: bool
    checked_records: int
    head_hash: str
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checked_records": self.checked_records,
            "head_hash": self.head_hash,
            "errors": list(self.errors),
        }


class HardCaseMiner:
    """Mine immutable episode references without training or device control."""

    def __init__(
        self,
        database: str | Path = ":memory:",
        *,
        config: MinerConfig | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._connection = _open_database(database)
        self._config = config or MinerConfig()
        self._clock = clock
        self._lock = threading.RLock()
        self._create_schema()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS hard_case_candidates (
                    sequence INTEGER PRIMARY KEY,
                    candidate_id TEXT NOT NULL UNIQUE,
                    event_timestamp REAL NOT NULL,
                    created_timestamp REAL NOT NULL,
                    reasons_json TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    signal_json TEXT NOT NULL,
                    episode_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    record_hash TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS hard_case_dedupe (
                    fingerprint TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    last_event_timestamp REAL NOT NULL,
                    duplicate_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS hard_case_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_hard_case_event_time
                    ON hard_case_candidates(event_timestamp);
                """
            )
            self._connection.execute(
                """
                INSERT INTO hard_case_state(key, value)
                VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (SCHEMA_VERSION,),
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> HardCaseMiner:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @staticmethod
    def _candidate_from_row(row: sqlite3.Row) -> HardCaseCandidate:
        return HardCaseCandidate(
            sequence=int(row["sequence"]),
            candidate_id=row["candidate_id"],
            event_timestamp=float(row["event_timestamp"]),
            created_timestamp=float(row["created_timestamp"]),
            reasons=tuple(json.loads(row["reasons_json"])),
            fingerprint=row["fingerprint"],
            signal=json.loads(row["signal_json"]),
            episode=json.loads(row["episode_json"]),
            provenance=json.loads(row["provenance_json"]),
            previous_hash=row["previous_hash"],
            record_hash=row["record_hash"],
        )

    def _get_candidate_by_id(
        self,
        candidate_id: str,
    ) -> HardCaseCandidate | None:
        row = self._connection.execute(
            "SELECT * FROM hard_case_candidates WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()
        return None if row is None else self._candidate_from_row(row)

    def observe(
        self,
        *,
        timestamp: float | None = None,
        ood: bool | None = None,
        ood_score: float | None = None,
        cross_modal_disagreement: float | None = None,
        guard_state: str | None = None,
        data_age_s: float | None = None,
        episode: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        dedupe_key: str | None = None,
    ) -> MiningDecision:
        event_timestamp = _finite_number(
            self._clock() if timestamp is None else timestamp,
            "timestamp",
        )
        if ood is not None and not isinstance(ood, bool):
            raise TypeError("ood must be boolean or None")
        if ood_score is not None:
            ood_score = _finite_number(ood_score, "ood_score")
            if not 0.0 <= ood_score <= 1.0:
                raise ValueError("ood_score must be in [0, 1]")
        if cross_modal_disagreement is not None:
            cross_modal_disagreement = _finite_number(
                cross_modal_disagreement,
                "cross_modal_disagreement",
            )
            if not 0.0 <= cross_modal_disagreement <= 1.0:
                raise ValueError("cross_modal_disagreement must be in [0, 1]")
        if data_age_s is not None:
            data_age_s = _finite_number(data_age_s, "data_age_s")
            if data_age_s < 0.0:
                raise ValueError("data_age_s must be non-negative")
        if guard_state is not None:
            if not isinstance(guard_state, str) or not guard_state.strip():
                raise ValueError("guard_state must be a non-empty string or None")
            guard_state = guard_state.strip()
        if dedupe_key is not None:
            if not isinstance(dedupe_key, str) or not dedupe_key.strip():
                raise ValueError("dedupe_key must be a non-empty string or None")
            dedupe_key = dedupe_key.strip()
        episode_value = _json_object(episode, "episode")
        provenance_value = _json_object(provenance, "provenance")

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                previous_row = self._connection.execute(
                    """
                    SELECT value FROM hard_case_state
                    WHERE key = 'last_guard_state'
                    """
                ).fetchone()
                previous_guard = (
                    None if previous_row is None else previous_row["value"]
                )

                reasons: list[str] = []
                if ood is True or (
                    ood_score is not None
                    and ood_score >= self._config.ood_threshold
                ):
                    reasons.append("ood")
                if (
                    cross_modal_disagreement is not None
                    and cross_modal_disagreement
                    >= self._config.disagreement_threshold
                ):
                    reasons.append("cross_modal_disagreement")
                if (
                    guard_state is not None
                    and previous_guard is not None
                    and guard_state != previous_guard
                ):
                    reasons.append("guard_state_transition")
                if (
                    data_age_s is not None
                    and data_age_s > self._config.max_data_age_s
                ):
                    reasons.append("stale_data")
                reasons = sorted(set(reasons))

                if guard_state is not None:
                    self._connection.execute(
                        """
                        INSERT INTO hard_case_state(key, value)
                        VALUES('last_guard_state', ?)
                        ON CONFLICT(key) DO UPDATE SET value = excluded.value
                        """,
                        (guard_state,),
                    )

                if not reasons:
                    self._connection.commit()
                    return MiningDecision(triggered=False, reasons=())

                signal = {
                    "ood": ood,
                    "ood_score": ood_score,
                    "cross_modal_disagreement": cross_modal_disagreement,
                    "guard_state": guard_state,
                    "previous_guard_state": previous_guard,
                    "data_age_s": data_age_s,
                }
                fingerprint_payload = {
                    "reasons": reasons,
                    "guard_state": guard_state,
                    "previous_guard_state": (
                        previous_guard
                        if "guard_state_transition" in reasons
                        else None
                    ),
                    "dedupe_key": dedupe_key,
                    "episode": episode_value,
                    "provenance": provenance_value,
                }
                fingerprint = _digest(fingerprint_payload)
                duplicate_row = self._connection.execute(
                    """
                    SELECT candidate_id, last_event_timestamp
                    FROM hard_case_dedupe
                    WHERE fingerprint = ?
                    """,
                    (fingerprint,),
                ).fetchone()
                if (
                    duplicate_row is not None
                    and abs(
                        event_timestamp
                        - float(duplicate_row["last_event_timestamp"])
                    )
                    <= self._config.dedupe_window_s
                ):
                    duplicate_of = duplicate_row["candidate_id"]
                    self._connection.execute(
                        """
                        UPDATE hard_case_dedupe
                        SET last_event_timestamp = ?,
                            duplicate_count = duplicate_count + 1
                        WHERE fingerprint = ?
                        """,
                        (event_timestamp, fingerprint),
                    )
                    candidate = self._get_candidate_by_id(duplicate_of)
                    self._connection.commit()
                    return MiningDecision(
                        triggered=True,
                        reasons=tuple(reasons),
                        candidate=candidate,
                        duplicate=True,
                        duplicate_of=duplicate_of,
                    )

                tail = self._connection.execute(
                    """
                    SELECT sequence, record_hash
                    FROM hard_case_candidates
                    ORDER BY sequence DESC
                    LIMIT 1
                    """
                ).fetchone()
                sequence = 1 if tail is None else int(tail["sequence"]) + 1
                previous_hash = (
                    GENESIS_HASH if tail is None else tail["record_hash"]
                )
                created_timestamp = _finite_number(
                    self._clock(),
                    "created_timestamp",
                )
                identity_payload = {
                    "sequence": sequence,
                    "event_timestamp": event_timestamp,
                    "fingerprint": fingerprint,
                    "previous_hash": previous_hash,
                }
                candidate_id = f"hc-{_digest(identity_payload)[:24]}"
                candidate = HardCaseCandidate(
                    sequence=sequence,
                    candidate_id=candidate_id,
                    event_timestamp=event_timestamp,
                    created_timestamp=created_timestamp,
                    reasons=tuple(reasons),
                    fingerprint=fingerprint,
                    signal=signal,
                    episode=episode_value,
                    provenance=provenance_value,
                    previous_hash=previous_hash,
                    record_hash="",
                )
                record_hash = _digest(candidate.chain_payload())
                candidate = HardCaseCandidate(
                    sequence=candidate.sequence,
                    candidate_id=candidate.candidate_id,
                    event_timestamp=candidate.event_timestamp,
                    created_timestamp=candidate.created_timestamp,
                    reasons=candidate.reasons,
                    fingerprint=candidate.fingerprint,
                    signal=candidate.signal,
                    episode=candidate.episode,
                    provenance=candidate.provenance,
                    previous_hash=candidate.previous_hash,
                    record_hash=record_hash,
                )
                self._connection.execute(
                    """
                    INSERT INTO hard_case_candidates(
                        sequence, candidate_id, event_timestamp,
                        created_timestamp, reasons_json, fingerprint,
                        signal_json, episode_json, provenance_json,
                        previous_hash, record_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.sequence,
                        candidate.candidate_id,
                        candidate.event_timestamp,
                        candidate.created_timestamp,
                        json.dumps(list(candidate.reasons), separators=(",", ":")),
                        candidate.fingerprint,
                        json.dumps(
                            candidate.signal,
                            allow_nan=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        json.dumps(
                            candidate.episode,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        json.dumps(
                            candidate.provenance,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        candidate.previous_hash,
                        candidate.record_hash,
                    ),
                )
                self._connection.execute(
                    """
                    INSERT INTO hard_case_dedupe(
                        fingerprint, candidate_id, last_event_timestamp,
                        duplicate_count
                    ) VALUES (?, ?, ?, 0)
                    ON CONFLICT(fingerprint) DO UPDATE SET
                        candidate_id = excluded.candidate_id,
                        last_event_timestamp = excluded.last_event_timestamp,
                        duplicate_count = 0
                    """,
                    (
                        candidate.fingerprint,
                        candidate.candidate_id,
                        candidate.event_timestamp,
                    ),
                )
                self._connection.commit()
                return MiningDecision(
                    triggered=True,
                    reasons=candidate.reasons,
                    candidate=candidate,
                )
            except Exception:
                self._connection.rollback()
                raise

    def list_candidates(self) -> list[HardCaseCandidate]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM hard_case_candidates ORDER BY sequence"
            ).fetchall()
        return [self._candidate_from_row(row) for row in rows]

    def verify_chain(self) -> ChainVerification:
        errors: list[str] = []
        previous_hash = GENESIS_HASH
        expected_sequence = 1
        rows: list[sqlite3.Row]
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM hard_case_candidates ORDER BY sequence"
            ).fetchall()
        for row in rows:
            try:
                candidate = self._candidate_from_row(row)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"sequence_{expected_sequence}:decode_error:{exc}")
                expected_sequence += 1
                continue
            if candidate.sequence != expected_sequence:
                errors.append(
                    f"sequence_{candidate.sequence}:expected_{expected_sequence}"
                )
            if candidate.previous_hash != previous_hash:
                errors.append(
                    f"sequence_{candidate.sequence}:previous_hash_mismatch"
                )
            calculated = _digest(candidate.chain_payload())
            if calculated != candidate.record_hash:
                errors.append(
                    f"sequence_{candidate.sequence}:record_hash_mismatch"
                )
            previous_hash = candidate.record_hash
            expected_sequence += 1
        return ChainVerification(
            valid=not errors,
            checked_records=len(rows),
            head_hash=previous_hash,
            errors=tuple(errors),
        )

    @staticmethod
    def safety_boundary() -> dict[str, bool]:
        return {
            "shadow_only": True,
            "controls_devices": False,
            "online_training": False,
            "publishes_ros": False,
        }
