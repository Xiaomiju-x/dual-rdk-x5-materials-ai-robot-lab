"""Offline evidence bootstrap for the future XRD/PL assay work line.

This package inventories existing bytes and freezes future file-drop semantics.
It does not implement an instrument controller, grant execution authority, or
change the frozen RB-VoE R1 release.
"""

from rb_voe.assay_bootstrap.builder import (
    BOOTSTRAP_STAGE,
    build_bootstrap_bundle,
    verify_bootstrap_bundle,
)
from rb_voe.assay_bootstrap.claim_gate import evaluate_claim_gate
from rb_voe.assay_bootstrap.intake import (
    DiagnosticReplayGuard,
    FileDropValidationError,
    validate_file_drop_record,
)
from rb_voe.assay_bootstrap.models import (
    ClaimEvidenceSummary,
    EvidenceTier,
    Modality,
    SourceRecord,
    SourceStatus,
)

__all__ = [
    "BOOTSTRAP_STAGE",
    "ClaimEvidenceSummary",
    "DiagnosticReplayGuard",
    "EvidenceTier",
    "FileDropValidationError",
    "Modality",
    "SourceRecord",
    "SourceStatus",
    "build_bootstrap_bundle",
    "evaluate_claim_gate",
    "validate_file_drop_record",
    "verify_bootstrap_bundle",
]
