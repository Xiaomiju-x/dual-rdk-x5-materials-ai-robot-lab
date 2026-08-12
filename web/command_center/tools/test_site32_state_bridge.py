#!/usr/bin/env python3
"""Regression tests for the cross-release SQLite state bridge."""

from __future__ import annotations

import os
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path

from site32_state_bridge import bridge_sqlite


class Site32StateBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="site32-state-bridge-")
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.db"
        self.target = self.root / "target.db"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _create_source(self, *, rows: int = 3) -> None:
        connection = sqlite3.connect(self.source)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE evidence(id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO evidence(value) VALUES(?)",
            [(f"row-{index}",) for index in range(rows)],
        )
        connection.commit()
        connection.close()

    def test_bridge_copies_wal_state_and_replaces_target_atomically(self) -> None:
        self._create_source(rows=5)
        connection = sqlite3.connect(self.target)
        try:
            connection.execute("CREATE TABLE stale(value TEXT)")
            connection.commit()
        finally:
            connection.close()

        result = bridge_sqlite(self.source, self.target, mode=0o600)

        self.assertTrue(result["changed"])
        self.assertEqual(result["quick_check"], "ok")
        self.assertEqual(len(str(result["sha256"])), 64)
        connection = sqlite3.connect(self.target)
        try:
            self.assertEqual(connection.execute("PRAGMA quick_check").fetchone()[0], "ok")
            self.assertEqual(connection.execute("SELECT count(*) FROM evidence").fetchone()[0], 5)
            self.assertIsNone(connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='stale'"
            ).fetchone())
        finally:
            connection.close()
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(self.target.stat().st_mode), 0o600)
        self.assertEqual(list(self.root.glob(".target.db.rollback-new.*")), [])

    def test_corrupt_source_is_rejected_without_touching_target(self) -> None:
        self.source.write_bytes(b"not-sqlite")
        self.target.write_bytes(b"keep-me")

        with self.assertRaises((ValueError, sqlite3.DatabaseError)):
            bridge_sqlite(self.source, self.target)

        self.assertEqual(self.target.read_bytes(), b"keep-me")
        self.assertEqual(list(self.root.glob(".target.db.rollback-new.*")), [])

    @unittest.skipIf(os.name == "nt", "Windows symlink creation requires optional privileges")
    def test_source_and_target_symlinks_are_rejected(self) -> None:
        self._create_source()
        link_source = self.root / "source-link.db"
        link_source.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            bridge_sqlite(link_source, self.target)

        real_target = self.root / "real-target.db"
        real_target.write_bytes(b"keep")
        self.target.symlink_to(real_target)
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            bridge_sqlite(self.source, self.target)
        self.assertEqual(real_target.read_bytes(), b"keep")

    def test_same_path_is_verified_without_rewrite(self) -> None:
        self._create_source()
        before = self.source.stat()
        result = bridge_sqlite(self.source, self.source)
        after = self.source.stat()
        self.assertFalse(result["changed"])
        self.assertEqual(result["quick_check"], "ok")
        self.assertEqual(before.st_ino, after.st_ino)


if __name__ == "__main__":
    unittest.main(verbosity=2)
