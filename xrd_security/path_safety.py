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
    if len(parts) > 32 or len(raw) > 4096:
        raise UnsafePathError("path exceeds the configured component limit")
    for part in parts:
        if len(part) > 255:
            raise UnsafePathError("path component exceeds the configured length limit")
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


def _is_reparse_or_junction(path: Path, metadata: os.stat_result) -> bool:
    """Recognize Windows reparse points, including directory junctions."""

    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if getattr(metadata, "st_file_attributes", 0) & reparse_flag:
        return True
    is_junction = getattr(os.path, "isjunction", None)
    return bool(is_junction(path)) if callable(is_junction) else False


def _verify_opened_descriptor(
    descriptor: int,
    selected_metadata: os.stat_result,
    *,
    require_directory: bool,
) -> os.stat_result:
    """Verify an opened object and close its descriptor on every failure path."""

    try:
        opened = os.fstat(descriptor)
        expected_type = (
            stat.S_ISDIR(opened.st_mode)
            if require_directory
            else stat.S_ISREG(opened.st_mode)
        )
        if not expected_type or not os.path.samestat(selected_metadata, opened):
            object_type = "directory" if require_directory else "file"
            raise UnsafePathError(f"{object_type} changed while it was being opened")
        return opened
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


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


def _open_regular_file_at(
    base: Path,
    parts: tuple[str, ...],
    flags: int,
) -> tuple[int, os.stat_result]:
    """Open a scanned entry with descriptor-relative traversal on POSIX.

    Every name passed to ``os.open(..., dir_fd=...)`` originates from a
    ``DirEntry``.  Request components participate only in equality tests.
    Directory descriptors pin each ancestor, so renaming or replacing a parent
    after it was selected cannot redirect the remainder of the walk.
    """

    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fds = [os.open(base, directory_flags)]
    try:
        for index in range(len(parts)):
            current_fd = directory_fds[-1]
            requested_key = os.path.normcase(parts[index])
            selected_name: str | None = None
            selected_metadata: os.stat_result | None = None
            with os.scandir(current_fd) as entries:
                for entry in entries:
                    if os.path.normcase(entry.name) != requested_key:
                        continue
                    selected_name = entry.name
                    selected_metadata = entry.stat(follow_symlinks=False)
                    break
            if selected_name is None or selected_metadata is None:
                raise FileNotFoundError(parts[-1])
            if stat.S_ISLNK(selected_metadata.st_mode):
                raise UnsafePathError("symbolic links are not allowed")

            is_last = index == len(parts) - 1
            if is_last:
                if not stat.S_ISREG(selected_metadata.st_mode):
                    raise FileNotFoundError(parts[-1])
                descriptor = os.open(selected_name, flags, dir_fd=current_fd)
                opened = _verify_opened_descriptor(
                    descriptor, selected_metadata, require_directory=False
                )
                return descriptor, opened

            if not stat.S_ISDIR(selected_metadata.st_mode):
                raise FileNotFoundError(parts[-1])
            next_fd = os.open(selected_name, directory_flags, dir_fd=current_fd)
            _verify_opened_descriptor(
                next_fd, selected_metadata, require_directory=True
            )
            directory_fds.append(next_fd)
    finally:
        for directory_fd in reversed(directory_fds):
            try:
                os.close(directory_fd)
            except OSError:
                pass

    raise FileNotFoundError(parts[-1])  # pragma: no cover - non-empty parts return/raise


def _open_regular_file_from_entries(
    base: Path,
    parts: tuple[str, ...],
    flags: int,
) -> tuple[int, os.stat_result]:
    """Open a scanned entry on platforms without descriptor-relative paths.

    Each matched ``DirEntry.path`` is resolved and restatted without following
    links to obtain authoritative Windows file-index metadata.  The final
    descriptor is checked against that pre-open identity before any bytes are
    read.  The trusted root and its parent directories must also not be
    writable by untrusted users.
    """

    current_directory = base
    current_metadata: os.stat_result | None = None
    selected_path: Path | None = None
    selected_metadata: os.stat_result | None = None
    for index in range(len(parts)):
        if current_metadata is not None:
            observed_directory = os.stat(current_directory, follow_symlinks=False)
            if _is_reparse_or_junction(current_directory, observed_directory):
                raise UnsafePathError("reparse points and junctions are not allowed")
            if not os.path.samestat(current_metadata, observed_directory):
                raise UnsafePathError("directory changed during path traversal")
            current_real = Path(os.path.realpath(os.fspath(current_directory)))
            if not _is_below(base, current_real):
                raise UnsafePathError("path escapes the trusted root")
        requested_key = os.path.normcase(parts[index])
        selected_path = None
        selected_metadata = None
        with os.scandir(current_directory) as entries:
            for entry in entries:
                if os.path.normcase(entry.name) != requested_key:
                    continue
                selected_path = Path(entry.path)
                selected_metadata = entry.stat(follow_symlinks=False)
                break
        if selected_path is None or selected_metadata is None:
            raise FileNotFoundError(parts[-1])
        if stat.S_ISLNK(selected_metadata.st_mode) or _is_reparse_or_junction(
            selected_path, selected_metadata
        ):
            raise UnsafePathError("symbolic links, reparse points and junctions are not allowed")
        selected_real = Path(os.path.realpath(os.fspath(selected_path)))
        if not _is_below(base, selected_real):
            raise UnsafePathError("path escapes the trusted root")
        # On some Windows Python builds DirEntry.stat has zero st_dev/st_ino;
        # refresh from the trusted entry path so later identity checks compare
        # stable file-index metadata.  Recheck the reparse bit after the lookup.
        selected_metadata = os.stat(selected_real, follow_symlinks=False)
        if _is_reparse_or_junction(selected_real, selected_metadata):
            raise UnsafePathError("reparse points and junctions are not allowed")
        if index == len(parts) - 1:
            if not stat.S_ISREG(selected_metadata.st_mode):
                raise FileNotFoundError(parts[-1])
        elif stat.S_ISDIR(selected_metadata.st_mode):
            current_directory = selected_real
            current_metadata = selected_metadata
        else:
            raise FileNotFoundError(parts[-1])

    if selected_path is None or selected_metadata is None:
        raise FileNotFoundError(parts[-1])  # pragma: no cover - parts is non-empty
    descriptor = os.open(selected_path, flags)
    opened = _verify_opened_descriptor(
        descriptor, selected_metadata, require_directory=False
    )
    return descriptor, opened


def read_contained_bytes(
    root: str | os.PathLike[str],
    untrusted_path: str | os.PathLike[str],
    *,
    allowed_suffixes: Iterable[str] | None = None,
    max_bytes: int = 16 * 1024 * 1024,
) -> bytes:
    """Read one confined regular file with symlink/race and size defenses.

    The read path is deliberately selected again here instead of consuming the
    :class:`Path` returned by :func:`resolve_contained_path`.  Request text is
    used only as an equality key while walking ``os.scandir`` results.  POSIX
    walks are descriptor-relative and pin every ancestor; the Windows fallback
    rejects reparse points, checks real-path containment at every component and
    verifies file identity after opening.  The trusted root itself must not be
    writable by untrusted users.
    """

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    base = Path(os.path.realpath(os.fspath(root)))
    parts = _relative_parts(untrusted_path)
    suffixes = _normalized_suffixes(allowed_suffixes)
    suffix = "." + parts[-1].rsplit(".", 1)[-1].lower() if "." in parts[-1] else ""
    if suffixes and suffix not in suffixes:
        raise UnsafePathError("file type is not allowed")

    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    supports_descriptor_walk = (
        os.open in os.supports_dir_fd and os.scandir in os.supports_fd
    )
    if supports_descriptor_walk:
        descriptor, opened = _open_regular_file_at(base, parts, flags)
    else:
        descriptor, opened = _open_regular_file_from_entries(base, parts, flags)
    try:
        if opened.st_size > max_bytes:
            raise UnsafePathError("file exceeds the configured size limit")

        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise UnsafePathError("file exceeds the configured size limit")
        return payload
    finally:
        os.close(descriptor)
