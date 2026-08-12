"""Filesystem path validation for HTTP-facing services.

The helpers deliberately return a fully resolved path only after it has been
shown to remain below a trusted root.  Resolving both sides also prevents a
symlink inside the root from escaping to another directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


class UnsafePathError(ValueError):
    """Raised when an untrusted path cannot be confined to its trusted root."""


def _normalized_suffixes(values: Iterable[str] | None) -> frozenset[str]:
    if values is None:
        return frozenset()
    return frozenset(value.lower() if value.startswith(".") else f".{value.lower()}" for value in values)


def resolve_contained_path(
    root: str | os.PathLike[str],
    untrusted_path: str | os.PathLike[str],
    *,
    allowed_suffixes: Iterable[str] | None = None,
    require_file: bool = False,
) -> Path:
    """Resolve *untrusted_path* below *root* or fail closed.

    Absolute paths, NUL bytes, sibling-prefix tricks, traversal components and
    symlink escapes are rejected.  ``require_file`` is intended for reads;
    callers preparing a new upload can leave it false and create the trusted
    root separately.
    """

    base = Path(os.path.realpath(os.fspath(root)))
    raw = os.fspath(untrusted_path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise UnsafePathError("path must be a non-empty text value")

    relative = Path(raw)
    if relative.is_absolute():
        raise UnsafePathError("absolute paths are not allowed")

    candidate = Path(os.path.realpath(os.path.join(os.fspath(base), raw)))
    normalized_base = os.path.normcase(os.fspath(base))
    normalized_candidate = os.path.normcase(os.fspath(candidate))
    base_prefix = normalized_base.rstrip(os.sep) + os.sep
    if normalized_candidate != normalized_base and not normalized_candidate.startswith(base_prefix):
        raise UnsafePathError("path escapes the trusted root")

    suffixes = _normalized_suffixes(allowed_suffixes)
    if suffixes and candidate.suffix.lower() not in suffixes:
        raise UnsafePathError("file type is not allowed")
    if require_file and not candidate.is_file():
        raise FileNotFoundError(candidate.name)
    return candidate


def resolve_safe_basename(
    root: str | os.PathLike[str],
    untrusted_name: str,
    *,
    allowed_suffixes: Iterable[str] | None = None,
    require_file: bool = False,
) -> Path:
    """Resolve a single filename below *root*, rejecting directory syntax."""

    if not isinstance(untrusted_name, str) or not untrusted_name:
        raise UnsafePathError("filename must be a non-empty text value")
    if untrusted_name in {".", ".."} or Path(untrusted_name).name != untrusted_name:
        raise UnsafePathError("directory components are not allowed")
    if "/" in untrusted_name or "\\" in untrusted_name:
        raise UnsafePathError("directory separators are not allowed")
    return resolve_contained_path(
        root,
        untrusted_name,
        allowed_suffixes=allowed_suffixes,
        require_file=require_file,
    )
