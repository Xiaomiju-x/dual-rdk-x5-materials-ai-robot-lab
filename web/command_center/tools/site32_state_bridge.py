#!/usr/bin/env python3
"""Atomically bridge the command-center SQLite state between release layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import stat
import tempfile
from pathlib import Path
from urllib.parse import quote


def _regular_file(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{label} is empty: {path}")


def _sqlite_ro_uri(path: Path) -> str:
    return "file:" + quote(os.fspath(path), safe="/:\\") + "?mode=ro"


def _quick_check(connection: sqlite3.Connection) -> str:
    row = connection.execute("PRAGMA quick_check").fetchone()
    return str(row[0]) if row else "missing-result"


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(os.fspath(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def bridge_sqlite(
    source: Path,
    target: Path,
    *,
    mode: int = 0o600,
    uid: int | None = None,
    gid: int | None = None,
) -> dict[str, object]:
    source = source.expanduser().absolute()
    _regular_file(source, label="source database")
    source = source.resolve(strict=True)
    target = target.expanduser().absolute()

    target_parent = target.parent.resolve(strict=True)
    target = target_parent / target.name
    if target.exists() or target.is_symlink():
        _regular_file(target, label="target database")
    if source == target:
        connection = sqlite3.connect(_sqlite_ro_uri(source), uri=True)
        try:
            result = _quick_check(connection)
        finally:
            connection.close()
        if result != "ok":
            raise ValueError(f"source database quick_check failed: {result}")
        return {
            "changed": False,
            "quick_check": result,
            "source": os.fspath(source),
            "target": os.fspath(target),
        }

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.rollback-new.", dir=os.fspath(target_parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        source_connection = sqlite3.connect(_sqlite_ro_uri(source), uri=True, timeout=30)
        target_connection = sqlite3.connect(os.fspath(temporary), timeout=30)
        try:
            source_result = _quick_check(source_connection)
            if source_result != "ok":
                raise ValueError(f"source database quick_check failed: {source_result}")
            source_connection.backup(target_connection, pages=256, sleep=0.01)
            target_connection.commit()
            target_result = _quick_check(target_connection)
            if target_result != "ok":
                raise ValueError(f"bridged database quick_check failed: {target_result}")
        finally:
            target_connection.close()
            source_connection.close()

        os.chmod(temporary, mode)
        if uid is not None or gid is not None:
            os.chown(temporary, -1 if uid is None else uid, -1 if gid is None else gid)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
            handle.seek(0)
            hasher = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
            digest = hasher.hexdigest()
        size = temporary.stat().st_size
        os.replace(temporary, target)
        _fsync_directory(target_parent)
        return {
            "bytes": size,
            "changed": True,
            "quick_check": "ok",
            "sha256": digest,
            "source": os.fspath(source),
            "target": os.fspath(target),
        }
    finally:
        temporary.unlink(missing_ok=True)


def _identity(name: str | None, *, group: bool) -> int | None:
    if name is None:
        return None
    if name.isdecimal():
        return int(name)
    if group:
        import grp

        return grp.getgrnam(name).gr_gid
    import pwd

    return pwd.getpwnam(name).pw_uid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--owner")
    parser.add_argument("--group")
    parser.add_argument("--mode", default="0600")
    args = parser.parse_args()
    mode = int(args.mode, 8)
    if mode & 0o077:
        parser.error("database mode must not grant group or other permissions")
    result = bridge_sqlite(
        args.source,
        args.target,
        mode=mode,
        uid=_identity(args.owner, group=False),
        gid=_identity(args.group, group=True),
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
