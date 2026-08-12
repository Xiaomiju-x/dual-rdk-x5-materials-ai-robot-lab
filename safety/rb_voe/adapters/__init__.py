"""Target and read-only adapter entry points for RB-VoE."""

from rb_voe.adapters.ai_x5 import (
    AiX5Adapter,
    AiX5CapabilityBinding,
    AiX5ReadOnlyAdapter,
    AiX5TargetAdapter,
)
from rb_voe.adapters.assay_station import AssayStationAdapter, AssayStationTargetAdapter
from rb_voe.adapters.base import AdapterRefusal, AdapterResult, TargetOnlyAdapter
from rb_voe.adapters.dual_arm import (
    DualArmAdapter,
    DualArmCapabilityBinding,
    DualArmReadOnlyAdapter,
    DualArmTargetAdapter,
)
from rb_voe.adapters.embodied_x5 import (
    EmbodiedX5Adapter,
    EmbodiedX5CapabilityBinding,
    EmbodiedX5ReadOnlyAdapter,
    EmbodiedX5TargetAdapter,
)
from rb_voe.adapters.read_only import (
    CapabilityReadResult,
    FileJsonSnapshotTransport,
    HttpJsonSnapshotTransport,
    MappingJsonSnapshotTransport,
    PrefetchedJsonSnapshotTransport,
    ReadSourceKind,
    SshJsonSnapshotTransport,
)

__all__ = [
    "AdapterRefusal",
    "AdapterResult",
    "AiX5Adapter",
    "AiX5CapabilityBinding",
    "AiX5ReadOnlyAdapter",
    "AiX5TargetAdapter",
    "AssayStationAdapter",
    "AssayStationTargetAdapter",
    "DualArmAdapter",
    "DualArmCapabilityBinding",
    "DualArmReadOnlyAdapter",
    "DualArmTargetAdapter",
    "EmbodiedX5Adapter",
    "EmbodiedX5CapabilityBinding",
    "EmbodiedX5ReadOnlyAdapter",
    "EmbodiedX5TargetAdapter",
    "CapabilityReadResult",
    "FileJsonSnapshotTransport",
    "HttpJsonSnapshotTransport",
    "MappingJsonSnapshotTransport",
    "PrefetchedJsonSnapshotTransport",
    "ReadSourceKind",
    "SshJsonSnapshotTransport",
    "TargetOnlyAdapter",
]
