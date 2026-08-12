"""Run fixed inputs through the isolated successor models on an RDK X5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def _sha256_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _meminfo() -> dict[str, int]:
    wanted = {"MemAvailable", "CmaTotal", "CmaFree"}
    result: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, rest = line.split(":", 1)
        if key in wanted:
            result[f"{key}_kib"] = int(rest.strip().split()[0])
    return result


def _self_pss_kib() -> int | None:
    path = Path(f"/proc/{os.getpid()}/smaps_rollup")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Pss:"):
            return int(line.split()[1])
    return None


def _thermal() -> dict[str, float]:
    result: dict[str, float] = {}
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            name = (zone / "type").read_text(encoding="utf-8").strip()
            raw = float((zone / "temp").read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        result[name] = raw / 1000.0 if raw > 1000.0 else raw
    return result


def _run_model(runner: object, input_value: np.ndarray, iterations: int, method: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    latencies: list[float] = []
    hashes: list[dict[str, str]] = []
    first_raw: dict[str, np.ndarray] | None = None
    for _ in range(iterations):
        result = getattr(runner, method)(input_value)
        raw = {name: np.ascontiguousarray(value, dtype=np.float32) for name, value in result["raw"].items()}
        if first_raw is None:
            first_raw = raw
        hashes.append({name: _sha256_bytes(value) for name, value in raw.items()})
        latencies.append(float(result["latency_ms"]))
    assert first_raw is not None
    unique_counts = {name: len({row[name] for row in hashes}) for name in first_raw}
    return first_raw, {
        "samples": len(latencies),
        "p50_ms": _percentile(latencies, 50.0),
        "p95_ms": _percentile(latencies, 95.0),
        "p99_ms": _percentile(latencies, 99.0),
        "mean_ms": float(statistics.fmean(latencies)),
        "max_ms": max(latencies),
        "output_sha256": {name: _sha256_bytes(value) for name, value in first_raw.items()},
        "unique_output_hashes_across_iterations": unique_counts,
        "identity": result["model"],
        "output_layouts": result["output_layouts"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--tiny-model", type=Path, required=True)
    parser.add_argument("--cam-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--ack-idle", action="store_true")
    args = parser.parse_args()
    if not args.ack_idle:
        parser.error("--ack-idle is required")
    if args.iterations < 200:
        parser.error("--iterations must be at least 200")

    release_root = args.release_root.resolve()
    home_candidates = (Path.home() / "xrd_candidates").resolve()
    for path in (release_root, args.tiny_model.resolve(), args.cam_model.resolve()):
        if home_candidates not in (path, *path.parents):
            parser.error(f"path is outside isolated candidates: {path}")
    sys.path.insert(0, str(release_root))
    from x5_tribev_flow.bpu_runtime import CamSemLiteBpuRunner, TinyOccFlowBpuRunner

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with np.load(args.inputs, allow_pickle=False) as bundle:
        tiny_input = np.ascontiguousarray(bundle["tiny_input"], dtype=np.float32)
        cam_bgr = np.ascontiguousarray(bundle["cam_bgr"], dtype=np.uint8)

    baseline = {"memory": _meminfo(), "pss_kib": _self_pss_kib(), "thermal_c": _thermal()}
    tiny_runner = TinyOccFlowBpuRunner(args.tiny_model)
    tiny_raw, tiny_report = _run_model(tiny_runner, tiny_input, args.iterations, "infer")
    after_tiny = {"memory": _meminfo(), "pss_kib": _self_pss_kib(), "thermal_c": _thermal()}
    cam_runner = CamSemLiteBpuRunner(args.cam_model)
    cam_raw, cam_report = _run_model(cam_runner, cam_bgr, args.iterations, "infer_bgr")
    after_cam = {"memory": _meminfo(), "pss_kib": _self_pss_kib(), "thermal_c": _thermal()}

    arrays = {
        **{f"tiny__{name}": value for name, value in tiny_raw.items()},
        **{f"cam__{name}": value for name, value in cam_raw.items()},
    }
    output_npz = output / "x5_int8_outputs.npz"
    np.savez_compressed(output_npz, **arrays)
    report = {
        "schema_version": 1,
        "kind": "x5-successor-fixed-probe-actual-board",
        "host": os.uname().nodename,
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "actual_backend": "hbm_runtime",
        "inputs_sha256": _sha256_file(args.inputs),
        "models": {"tiny_occ_flow": tiny_report, "cam_sem_lite": cam_report},
        "resources": {"baseline": baseline, "after_tiny": after_tiny, "after_cam": after_cam},
        "output_bundle_sha256": _sha256_file(output_npz),
        "shadow_only": True,
        "motion_authority": False,
        "services_restarted": [],
        "measurement_boundary": "Fixed synthetic inputs on actual X5 BPU; no ROS, camera, serial, F407 or motion",
    }
    (output / "x5_probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
