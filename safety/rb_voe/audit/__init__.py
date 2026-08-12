"""Append-only audit primitives for deterministic RB-VoE evidence runs."""

from rb_voe.audit.ledger import (
    ANCHOR_SCHEMA,
    GENESIS_HASH,
    LEDGER_SCHEMA,
    AuditLedgerError,
    LedgerVerificationError,
    append_record,
    diagnose_ledger,
    initialize_ledger,
    terminal_anchor_path,
    verify_ledger,
)
from rb_voe.audit.semantic import (
    CASE_EVENT,
    OBSERVATION_EVENT,
    PLAN_EVENT,
    TERMINAL_EVENT,
    SemanticLedgerReport,
    require_verified_semantic_report,
    verify_r1_semantic_ledger,
)

__all__ = [
    "ANCHOR_SCHEMA",
    "GENESIS_HASH",
    "LEDGER_SCHEMA",
    "AuditLedgerError",
    "LedgerVerificationError",
    "append_record",
    "diagnose_ledger",
    "initialize_ledger",
    "terminal_anchor_path",
    "verify_ledger",
    "CASE_EVENT",
    "PLAN_EVENT",
    "OBSERVATION_EVENT",
    "TERMINAL_EVENT",
    "SemanticLedgerReport",
    "require_verified_semantic_report",
    "verify_r1_semantic_ledger",
]
