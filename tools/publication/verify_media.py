#!/usr/bin/env python3
"""Verify the deterministic public-media manifest using only the stdlib."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


DEFAULT_MANIFEST = Path("assets/media/MEDIA_MANIFEST.json")
GIF_MAX_BYTES = 8 * 1024 * 1024
PHONE_MARKERS = (
    b"com.android",
    b"com.xiaomi",
    b"xiaomi.exifinfo",
    b"location-eng",
)


@dataclass(frozen=True, order=True)
class MediaFinding:
    rule: str
    path: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_png_chunks(data: bytes) -> Iterator[tuple[bytes, bytes]]:
    cursor = 8
    while cursor + 12 <= len(data):
        length = int.from_bytes(data[cursor : cursor + 4], "big")
        kind = data[cursor + 4 : cursor + 8]
        end = cursor + 12 + length
        if end > len(data):
            return
        yield kind, data[cursor + 8 : cursor + 8 + length]
        cursor = end
        if kind == b"IEND":
            return


def _jpeg_markers(data: bytes) -> Iterator[tuple[int, bytes]]:
    if not data.startswith(b"\xff\xd8"):
        return
    cursor = 2
    while cursor + 4 <= len(data):
        if data[cursor] != 0xFF:
            cursor += 1
            continue
        marker = data[cursor + 1]
        cursor += 2
        if marker == 0xDA:
            return
        if marker in {0xD8, 0xD9}:
            continue
        length = int.from_bytes(data[cursor : cursor + 2], "big")
        if length < 2 or cursor + length > len(data):
            return
        yield marker, data[cursor + 2 : cursor + length]
        cursor += length


def _image_metadata_found(path: Path, data: bytes) -> bool:
    suffix = path.suffix.lower()
    lower = data.lower()
    if suffix in {".jpg", ".jpeg"}:
        for marker, payload in _jpeg_markers(data):
            if marker == 0xE1 and (
                payload.startswith(b"Exif\x00\x00")
                or b"http://ns.adobe.com/xap/1.0/" in payload
            ):
                return True
            if marker == 0xE2 and payload.startswith(b"ICC_PROFILE\x00"):
                return True
        return False
    if suffix == ".png":
        sensitive = {b"eXIf", b"iCCP", b"iTXt", b"tEXt", b"zTXt"}
        return any(kind in sensitive for kind, _ in _iter_png_chunks(data))
    if suffix == ".webp":
        return any(marker in data for marker in (b"EXIF", b"XMP ", b"ICCP"))
    if suffix == ".gif":
        return b"XMP DataXMP" in data or b"ICCRGBG1012" in data
    return b"exif\x00\x00" in lower or b"xmp" in lower or b"icc_profile" in lower


def _walk_mp4_boxes(data: bytes, start: int = 0, end: int | None = None) -> Iterator[tuple[bytes, bytes]]:
    end = len(data) if end is None else min(end, len(data))
    cursor = start
    containers = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"udta"}
    while cursor + 8 <= end:
        size = int.from_bytes(data[cursor : cursor + 4], "big")
        kind = data[cursor + 4 : cursor + 8]
        header = 8
        if size == 1:
            if cursor + 16 > end:
                return
            size = int.from_bytes(data[cursor + 8 : cursor + 16], "big")
            header = 16
        elif size == 0:
            size = end - cursor
        if size < header or cursor + size > end:
            return
        payload_start = cursor + header
        if kind == b"meta":
            payload_start += 4
        payload = data[payload_start : cursor + size]
        yield kind, payload
        if kind in containers or kind == b"meta":
            yield from _walk_mp4_boxes(data, payload_start, cursor + size)
        cursor += size


def _mp4_findings(path: Path, data: bytes) -> set[str]:
    rules: set[str] = set()
    if len(data) < 12 or data[4:8] != b"ftyp":
        rules.add("mp4_container")
        return rules
    handler_types: set[bytes] = set()
    saw_video_sample_entry = False
    for kind, payload in _walk_mp4_boxes(data):
        if kind == b"hdlr" and len(payload) >= 12:
            handler_types.add(payload[8:12])
        if kind == b"stsd" and any(codec in payload for codec in (b"avc1", b"hvc1", b"hev1", b"vp09", b"av01")):
            saw_video_sample_entry = True
    if b"soun" in handler_types:
        rules.add("mp4_audio")
    if b"vide" not in handler_types or not saw_video_sample_entry:
        rules.add("mp4_video_missing")
    lower = data.lower()
    if any(marker in lower for marker in PHONE_MARKERS):
        rules.add("mp4_phone_metadata")
    if b"\xa9xyz" in data or b"location" in lower:
        rules.add("mp4_location_metadata")
    return rules


def verify_manifest(root: Path | str, manifest: Path | str = DEFAULT_MANIFEST) -> list[MediaFinding]:
    root_path = Path(root).resolve()
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = root_path / manifest_path
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("entries"), list):
        return [MediaFinding("manifest_schema", manifest_path.relative_to(root_path).as_posix())]

    findings: list[MediaFinding] = []
    seen: set[str] = set()
    for entry in document["entries"]:
        relative = entry.get("path")
        if not isinstance(relative, str) or relative in seen:
            findings.append(MediaFinding("manifest_path", str(relative)))
            continue
        seen.add(relative)
        candidate = (root_path / relative).resolve()
        try:
            candidate.relative_to(root_path)
        except ValueError:
            findings.append(MediaFinding("manifest_outside_root", relative))
            continue
        if not candidate.is_file():
            findings.append(MediaFinding("missing", relative))
            continue
        data = candidate.read_bytes()
        if candidate.stat().st_size != entry.get("bytes"):
            findings.append(MediaFinding("bytes", relative))
        if _sha256(candidate) != entry.get("sha256"):
            findings.append(MediaFinding("sha256", relative))
        suffix = candidate.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"} and _image_metadata_found(candidate, data):
            findings.append(MediaFinding("image_metadata", relative))
        if suffix == ".gif" and len(data) > GIF_MAX_BYTES:
            findings.append(MediaFinding("gif_size", relative))
        if suffix == ".mp4":
            findings.extend(MediaFinding(rule, relative) for rule in sorted(_mp4_findings(candidate, data)))
    return sorted(set(findings))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    try:
        findings = verify_manifest(args.root, args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"MEDIA_VERIFICATION=ERROR\nreason={type(error).__name__}", file=sys.stderr)
        return 2
    print("MEDIA_VERIFICATION=" + ("PASS" if not findings else "FAIL"))
    print(f"finding_count={len(findings)}")
    for finding in findings:
        print(f"{finding.rule} {finding.path}")
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
