#!/usr/bin/env python3
"""Prepare content-audited TinyOccFlow float32 PTQ calibration tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from embodied_brain.finals_successor.x5_tribev_flow.dataset import (
    build_episode_refs,
    flatten_tribev_history,
    load_episode,
    split_episode_refs,
)


EXPECTED_SHAPE = (1, 40, 64, 64)
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
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_bytes(path, encoded)


def prepare(
    dataset_root: Path,
    output_directory: Path,
    *,
    count: int = 128,
    seed: int = 20260734,
) -> dict[str, Any]:
    if count < 100:
        raise ValueError("at least 100 independent episode tensors are required")
    refs = build_episode_refs(dataset_root)
    dataset_manifest_path = dataset_root / "dataset_manifest.json"
    if not dataset_manifest_path.is_file():
        raise RuntimeError(
            "dataset_manifest.json is required for auditable PTQ calibration"
        )
    dataset_manifest = json.loads(
        dataset_manifest_path.read_text(encoding="utf-8")
    )
    if int(dataset_manifest.get("episode_count", -1)) != len(refs):
        raise RuntimeError("dataset manifest episode_count does not match corpus")
    splits = split_episode_refs(refs, seed=seed)
    candidates = sorted(
        splits["train"],
        key=lambda ref: hashlib.sha256(
            f"{seed}:{ref.episode_id}".encode("utf-8")
        ).digest(),
    )
    if len(candidates) < count:
        raise RuntimeError(
            f"train split has only {len(candidates)} episodes, requested {count}"
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    samples_directory = output_directory / "samples"
    samples_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, ref in enumerate(candidates[:count]):
        record = load_episode(ref.path, validate=True)
        flattened = flatten_tribev_history(
            record["arrays"]["tribev_input"]
        )[None]
        tensor = np.ascontiguousarray(flattened, dtype="<f4")
        if tensor.shape != EXPECTED_SHAPE:
            raise RuntimeError(
                f"{ref.episode_id}: {tensor.shape} != {EXPECTED_SHAPE}"
            )
        target = samples_directory / f"{index:04d}_{ref.episode_id}.bin"
        _atomic_bytes(target, tensor.tobytes(order="C"))
        if target.stat().st_size != EXPECTED_BYTES:
            raise RuntimeError(f"{target}: invalid size after write")
        records.append(
            {
                "file": target.name,
                "sha256": _sha256(target),
                "bytes": target.stat().st_size,
                "episode_id": ref.episode_id,
                "session_id": ref.session_id,
                "scenario_id": ref.scenario_id,
                "source_kind": ref.source_kind,
                "split": "train",
            }
        )

    manifest = {
        "schema_version": "tiny-occ-flow-calibration/1.0",
        "dataset_root": str(dataset_root.resolve()),
        "dataset_manifest": {
            "path": str(dataset_manifest_path.resolve()),
            "sha256": _sha256(dataset_manifest_path),
            "schema_version": dataset_manifest.get("schema_version"),
            "duplicate_input_count": dataset_manifest.get(
                "duplicate_input_count"
            ),
        },
        "seed": seed,
        "sample_count": len(records),
        "shape": list(EXPECTED_SHAPE),
        "dtype": "little-endian-float32",
        "bytes_per_sample": EXPECTED_BYTES,
        "samples_directory": str(samples_directory.resolve()),
        "records": records,
        "test_split_used": False,
        "shadow_only": True,
        "cmd_vel_authority": False,
        "claim_boundary": (
            "Synthetic tensors are valid for toolchain calibration and pipeline "
            "testing only. Board calibration must later include representative "
            "real TriBEV captures before a real-world model claim."
        ),
    }
    manifest_path = output_directory / "manifest.json"
    _atomic_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260734)
    args = parser.parse_args(argv)
    result = prepare(
        args.dataset,
        args.output,
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
                    "bytes_per_sample",
                    "samples_directory",
                    "test_split_used",
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
