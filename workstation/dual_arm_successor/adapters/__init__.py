"""Read-only adapters for frozen dual-arm evidence."""

from .evidence_adapter import (
    DatasetAdapterError,
    EvidenceContractError,
    EvidencePathError,
    build_shadow_dataset,
)

__all__ = [
    "DatasetAdapterError",
    "EvidenceContractError",
    "EvidencePathError",
    "build_shadow_dataset",
]
