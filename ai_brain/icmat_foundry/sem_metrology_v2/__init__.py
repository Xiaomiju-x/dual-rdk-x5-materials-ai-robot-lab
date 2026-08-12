"""Fail-closed SEM metrology v2 candidate.

The package is isolated from the frozen SEM-Metrology-X5 baseline and from all
production/X5 entry points.
"""

from .contracts import CLAIM_BOUNDARY, NON_TEST_GATE, SEALED_TEST_SET, TRAIN_SETS
from .model import TinyUNetQuality

__all__ = [
    "CLAIM_BOUNDARY",
    "NON_TEST_GATE",
    "SEALED_TEST_SET",
    "TRAIN_SETS",
    "TinyUNetQuality",
]
