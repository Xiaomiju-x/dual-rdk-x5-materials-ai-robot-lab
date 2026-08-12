"""Passive 4K FSD-style semantic BEV bridge for finals vNext."""

from .bridge import BridgeResult, READ_ONLY_AUTHORITY, ReadOnlySemanticBridge
from .codec import MAGIC, PayloadError, decode_payload, encode_payload
from .contracts import (
    CameraExtrinsics,
    CameraIntrinsics,
    ContractError,
    FrameProvenance,
    OdometryDelta,
    PayloadLimits,
    ProvenanceState,
    QualityMetrics,
    SCHEMA_VERSION,
    SemanticBEVFrame,
    SparseVectorToken,
    VectorTokenKind,
)
from .memory import (
    DualBEVMemory,
    MemoryConfig,
    MemorySnapshot,
    MemoryUpdateError,
    compute_ghost_risk,
    warp_bev_nearest,
)
from .validation import (
    FrameAssessment,
    FrameRejectedError,
    FreshnessQualityPolicy,
    assess_frame,
    require_acceptable_frame,
)


__all__ = [
    "BridgeResult",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "ContractError",
    "DualBEVMemory",
    "FrameAssessment",
    "FrameProvenance",
    "FrameRejectedError",
    "FreshnessQualityPolicy",
    "MAGIC",
    "MemoryConfig",
    "MemorySnapshot",
    "MemoryUpdateError",
    "OdometryDelta",
    "PayloadError",
    "PayloadLimits",
    "ProvenanceState",
    "QualityMetrics",
    "READ_ONLY_AUTHORITY",
    "ReadOnlySemanticBridge",
    "SCHEMA_VERSION",
    "SemanticBEVFrame",
    "SparseVectorToken",
    "VectorTokenKind",
    "assess_frame",
    "compute_ghost_risk",
    "decode_payload",
    "encode_payload",
    "require_acceptable_frame",
    "warp_bev_nearest",
]
