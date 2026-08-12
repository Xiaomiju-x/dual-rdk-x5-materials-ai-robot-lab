#!/usr/bin/env python3
"""Prepare synthetic RGB float32 calibration tensors for CamSemLite tooling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from embodied_brain.finals_successor.x5_tribev_flow.camsem_synthetic import (
    QUALITY_CLASS_NAMES,
    generate_camsem_sample,
)


EXPECTED_SHAPE = (1, 3, 288, 512)
EXPECTED_BYTES = int(np.prod(EXPECTED_SHAPE) * np.dtype("<f4").itemsize)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(
        path,
        (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8"),
    )


def prepare(
    output_directory: Path,
    *,
    first_index: int = 480,
    count: int = 100,
    seed: int = 20260734,
) -> dict[str, Any]:
    if count < 100:
        raise ValueError("at least 100 calibration images are required")
    samples_directory = output_directory / "samples"
    samples_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for offset in range(count):
        index = first_index + offset
        sample = generate_camsem_sample(index, seed=seed)
        tensor = np.ascontiguousarray(
            sample.rgb_u8.transpose(2, 0, 1)[None],
            dtype="<f4",
        )
        if tensor.shape != EXPECTED_SHAPE:
            raise RuntimeError(f"sample {index}: {tensor.shape} != {EXPECTED_SHAPE}")
        target = samples_directory / f"{offset:04d}_synthetic_{index:06d}.bin"
        _atomic_bytes(target, tensor.tobytes(order="C"))
        if target.stat().st_size != EXPECTED_BYTES:
            raise RuntimeError(f"{target}: invalid calibration sample size")
        records.append(
            {
                "file": target.name,
                "sha256": _sha256(target),
                "bytes": target.stat().st_size,
                "synthetic_index": index,
                "quality_label": int(sample.quality_label),
                "quality_name": QUALITY_CLASS_NAMES[sample.quality_label],
            }
        )
    manifest = {
        "schema_version": "cam-sem-lite-calibration/1.0",
        "seed": seed,
        "first_index": first_index,
        "sample_count": count,
        "shape": list(EXPECTED_SHAPE),
        "dtype": "little-endian-float32",
        "value_range": [0.0, 255.0],
        "scale_value": 1.0 / 255.0,
        "bytes_per_sample": EXPECTED_BYTES,
        "samples_directory": str(samples_directory.resolve()),
        "records": records,
        "synthetic_only": True,
        "real_camera_calibration_required_before_real_claim": True,
        "shadow_only": True,
        "cmd_vel_authority": False,
    }
    manifest_path = output_directory / "manifest.json"
    _atomic_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-index", type=int, default=480)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260734)
    args = parser.parse_args(argv)
    result = prepare(
        args.output,
        first_index=args.first_index,
        count=args.count,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "sample_count",
                    "shape",
                    "dtype",
                    "value_range",
                    "scale_value",
                    "samples_directory",
                    "manifest_path",
                    "manifest_sha256",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
