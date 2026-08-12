"""Finals-only ICMat-PropNet research candidate.

This package is intentionally isolated from the frozen AI Brain production
entry points. Importing it never starts a service, contacts a device, or
changes an existing model slot.
"""

from .contracts import FEATURE_NAMES, TARGET_SPECS
from .model import PropNet

__all__ = ["FEATURE_NAMES", "TARGET_SPECS", "PropNet"]
