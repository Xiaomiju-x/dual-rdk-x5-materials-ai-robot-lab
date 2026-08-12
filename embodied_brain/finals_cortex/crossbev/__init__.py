"""PC-only CrossBEV knowledge-distillation contracts.

This package implements a lightweight, auditable monocular temporal BEV
student contract. It borrows public BEV and cross-modal distillation ideas; it
does not claim to reproduce Tesla software, data, weights, or architecture.
"""

from .contracts import (
    CROSSBEV_LAYER_NAMES,
    CalibrationRecord,
    ContractError,
    CrossBEVMaps,
    GateDecision,
    GatePolicy,
    ProvenanceState,
    TemporalFrameProvenance,
    TemporalMonocularInput,
    assess_temporal_input,
    require_accepted_temporal_input,
)
from .distillation import (
    DEFAULT_LAYER_WEIGHTS,
    DistillationLoss,
    crossbev_distillation_loss,
)
from .model import (
    TORCH_AVAILABLE,
    CrossBEVStudent,
    crossbev_probabilities,
)

__all__ = [
    "CROSSBEV_LAYER_NAMES",
    "DEFAULT_LAYER_WEIGHTS",
    "TORCH_AVAILABLE",
    "CalibrationRecord",
    "ContractError",
    "CrossBEVMaps",
    "CrossBEVStudent",
    "DistillationLoss",
    "GateDecision",
    "GatePolicy",
    "ProvenanceState",
    "TemporalFrameProvenance",
    "TemporalMonocularInput",
    "assess_temporal_input",
    "crossbev_distillation_loss",
    "crossbev_probabilities",
    "require_accepted_temporal_input",
]
