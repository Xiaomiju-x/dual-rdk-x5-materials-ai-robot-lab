"""Filesystem path validation for HTTP-facing services.

Untrusted path text is never used for a filesystem lookup.  The resolver first
validates the path lexically, then walks directory entries that originate from
the trusted root and selects entries by name.  This data-flow boundary is both
easier to audit and safer than joining a request value and checking the result
afterwards.

For reads, :func:`read_contained_bytes` also opens the selected regular file
without following a final symlink and verifies that the opened descriptor still
refers to the file that was inspected.  This narrows the usual check/open race;
the trusted root must still not be writable by untrusted users.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterable
from pathlib import Path, PureWindowsPath


class UnsafePathError(ValueError):
    """Raised when an untrusted path cannot be confined to its trusted root."""


def _normalized_suffixes(values: Iterable[str] | None) -> frozenset[str]:
    if values is None:
        return frozenset()
    return frozenset(value.lower() if value.startswith(".") else f".{value.lower()}" for value in values)


def _relative_parts(untrusted_path: str | os.PathLike[str]) -> tuple[str, ...]:
    """Return validated relative components without touching the filesystem."""

    raw = os.fspath(untrusted_path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise UnsafePathError("path must be a non-empty text value")
    # PureWindowsPath catches drive-qualified, rooted and UNC paths even when
    # validation runs on Linux.  Path catches native absolute paths.
    if Path(raw).is_absolute() or PureWindowsPath(raw).is_absolute() or PureWindowsPath(raw).drive:
        raise UnsafePathError("absolute paths are not allowed")

    parts = tuple(raw.replace("\\", "/").split("/"))
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise UnsafePathError("empty and traversal components are not allowed")
    for part in parts:
        if ":" in part:
            # Also blocks NTFS alternate data streams (for example file.raw:log).
            raise UnsafePathError("colon characters are not allowed")
        if part.endswith((" ", ".")) or any(ord(character) < 32 for character in part):
            raise UnsafePathError("ambiguous or control characters are not allowed")
    return parts


def _entry_for_name(directory: Path, wanted_name: str) -> tuple[Path, os.stat_result] | None:
    """Select an actual directory entry by name.

    The returned path is built from ``DirEntry.path`` (trusted filesystem data),
    not from ``wanted_name``.  Case folding mirrors the host filesystem.
    """

    wanted_key = os.path.normcase(wanted_name)
    with os.scandir(directory) as entries:
        for entry in entries:
            if os.path.normcase(entry.name) != wanted_key:
                continue
            metadata = entry.stat(follow_symlinks=False)
            return Path(entry.path), metadata
    return None


def _existing_regular_file(base: Path, parts: tuple[str, ...]) -> Path:
    """Walk only real directories below *base* and return a regular file."""

    current = base
    for index, wanted_name in enumerate(parts):
        match = _entry_for_name(current, wanted_name)
        if match is None:
            raise FileNotFoundError(parts[-1])
        entry_path, metadata = match
        if stat.S_ISLNK(metadata.st_mode):
            raise UnsafePathError("symbolic links are not allowed")
        is_last = index == len(parts) - 1
        if is_last:
            if not stat.S_ISREG(metadata.st_mode):
                raise FileNotFoundError(entry_path.name)
        elif not stat.S_ISDIR(metadata.st_mode):
            raise FileNotFoundError(parts[-1])
        current = entry_path
    return current


def _existing_directory(base: Path, parts: tuple[str, ...]) -> Path:
    """Walk existing non-symlink directories, returning the last directory."""

    current = base
    for wanted_name in parts:
        match = _entry_for_name(current, wanted_name)
        if match is None:
            raise FileNotFoundError(wanted_name)
        entry_path, metadata = match
        if stat.S_ISLNK(metadata.st_mode):
            raise UnsafePathError("symbolic links are not allowed")
        if not stat.S_ISDIR(metadata.st_mode):
            raise FileNotFoundError(wanted_name)
        current = entry_path
    return current


def _is_below(base: Path, candidate: Path) -> bool:
    normalized_base = os.path.normcase(os.fspath(base))
    normalized_candidate = os.path.normcase(os.fspath(candidate))
    prefix = normalized_base.rstrip(os.sep) + os.sep
    return normalized_candidate == normalized_base or normalized_candidate.startswith(prefix)


def resolve_contained_path(
    root: str | os.PathLike[str],
    untrusted_path: str | os.PathLike[str],
    *,
    allowed_suffixes: Iterable[str] | None = None,
    require_file: bool = False,
) -> Path:
    """Resolve *untrusted_path* below *root* or fail closed.

    Absolute paths, NUL bytes, sibling-prefix tricks, traversal components,
    NTFS alternate data streams and symlinks are rejected.  Existing files are
    selected from directory entries rooted at *root*, so request text is never
    sent to a filesystem lookup.
    """

    base = Path(os.path.realpath(os.fspath(root)))
    parts = _relative_parts(untrusted_path)
    suffixes = _normalized_suffixes(allowed_suffixes)
    if suffixes and Path(parts[-1]).suffix.lower() not in suffixes:
        raise UnsafePathError("file type is not allowed")

    if require_file:
        candidate = _existing_regular_file(base, parts)
    else:
        parent = _existing_directory(base, parts[:-1])
        existing = _entry_for_name(parent, parts[-1])
        if existing is not None:
            candidate, metadata = existing
            if stat.S_ISLNK(metadata.st_mode):
                raise UnsafePathError("symbolic links are not allowed")
        else:
            # basename is a recognized path sanitizer; _relative_parts already
            # proved it contains no platform separator or alternate-data syntax.
            safe_name = os.path.basename(parts[-1])
            candidate = parent / safe_name

    resolved = Path(os.path.realpath(os.fspath(candidate)))
    if not _is_below(base, resolved):
        raise UnsafePathError("path escapes the trusted root")
    return resolved


def resolve_safe_basename(
    root: str | os.PathLike[str],
    untrusted_name: str,
    *,
    allowed_suffixes: Iterable[str] | None = None,
    require_file: bool = False,
) -> Path:
    """Resolve a single filename below *root*, rejecting directory syntax."""

    parts = _relative_parts(untrusted_name)
    if len(parts) != 1:
        raise UnsafePathError("directory components are not allowed")
    return resolve_contained_path(
        root,
        untrusted_name,
        allowed_suffixes=allowed_suffixes,
        require_file=require_file,
    )


def read_contained_bytes(
    root: str | os.PathLike[str],
    untrusted_path: str | os.PathLike[str],
    *,
    allowed_suffixes: Iterable[str] | None = None,
    max_bytes: int = 16 * 1024 * 1024,
) -> bytes:
    """Read one confined regular file with symlink/race and size defenses."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    base = Path(os.path.realpath(os.fspath(root)))
    candidate = resolve_contained_path(
        base,
        untrusted_path,
        allowed_suffixes=allowed_suffixes,
        require_file=True,
    )
    before_real = Path(os.path.realpath(os.fspath(candidate)))
    if not _is_below(base, before_real):
        raise UnsafePathError("path escapes the trusted root")
    before = os.lstat(candidate)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise UnsafePathError("only regular files may be read")

    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    try:
        opened = os.fstat(descriptor)
        after = os.lstat(candidate)
        after_real = Path(os.path.realpath(os.fspath(candidate)))
        if not stat.S_ISREG(opened.st_mode):
            raise UnsafePathError("only regular files may be read")
        if not os.path.samestat(before, opened) or not os.path.samestat(after, opened):
            raise UnsafePathError("file changed while it was being opened")
        if before_real != after_real or not _is_below(base, after_real):
            raise UnsafePathError("path changed while it was being opened")
        if opened.st_size > max_bytes:
            raise UnsafePathError("file exceeds the configured size limit")

        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise UnsafePathError("file exceeds the configured size limit")
        return payload
    finally:
        os.close(descriptor)
