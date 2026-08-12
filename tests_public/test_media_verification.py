from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = REPOSITORY_ROOT / "tools" / "publication" / "verify_media.py"
SPEC = importlib.util.spec_from_file_location("verify_media", VERIFIER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load media verifier")
verify_media = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_media
SPEC.loader.exec_module(verify_media)


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def _minimal_mp4(handler: bytes = b"vide", extra: bytes = b"") -> bytes:
    ftyp = _box(b"ftyp", b"isom\x00\x00\x02\x00isommp42")
    hdlr = _box(b"hdlr", b"\x00\x00\x00\x00\x00\x00\x00\x00" + handler + b"\x00" * 12)
    stsd = _box(b"stsd", b"\x00\x00\x00\x00\x00\x00\x00\x01" + b"avc1" + b"\x00" * 16)
    moov = _box(b"moov", _box(b"trak", _box(b"mdia", hdlr + _box(b"minf", _box(b"stbl", stsd)))))
    return ftyp + moov + extra


class MediaVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "assets/media").mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_manifest(self, path: Path) -> None:
        import hashlib

        document = {
            "schema_version": 1,
            "entries": [{
                "path": path.relative_to(self.root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "kind": path.suffix.lstrip("."),
            }],
        }
        (self.root / "assets/media/MEDIA_MANIFEST.json").write_text(
            json.dumps(document), encoding="utf-8"
        )

    def test_repository_manifest_passes(self) -> None:
        self.assertEqual(verify_media.verify_manifest(REPOSITORY_ROOT), [])

    def test_hash_and_size_mismatch_fail(self) -> None:
        target = self.root / "assets/media/demo.gif"
        target.write_bytes(b"GIF89a")
        self._write_manifest(target)
        target.write_bytes(b"GIF89a changed")
        rules = {item.rule for item in verify_media.verify_manifest(self.root)}
        self.assertTrue({"bytes", "sha256"} <= rules)

    def test_jpeg_exif_is_rejected(self) -> None:
        payload = b"Exif\x00\x00" + b"II*\x00\x08\x00\x00\x00\x00\x00"
        target = self.root / "assets/media/photo.jpg"
        target.write_bytes(b"\xff\xd8\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload + b"\xff\xd9")
        self._write_manifest(target)
        rules = {item.rule for item in verify_media.verify_manifest(self.root)}
        self.assertIn("image_metadata", rules)

    def test_mp4_audio_and_location_are_rejected(self) -> None:
        target = self.root / "assets/media/demo.mp4"
        target.write_bytes(_minimal_mp4(b"soun", b"location"))
        self._write_manifest(target)
        rules = {item.rule for item in verify_media.verify_manifest(self.root)}
        self.assertTrue({"mp4_audio", "mp4_location_metadata"} <= rules)


if __name__ == "__main__":
    unittest.main()
