"""Small, dependency-free security primitives shared by public XRD services."""

from .path_safety import (
    UnsafePathError,
    read_contained_bytes,
    resolve_contained_path,
    resolve_safe_basename,
)

__all__ = (
    "UnsafePathError",
    "read_contained_bytes",
    "resolve_contained_path",
    "resolve_safe_basename",
)
