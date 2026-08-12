#!/usr/bin/env python3
"""Compile one isolated X5BiSkillTCN ONNX candidate for Bayes-e."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


IMAGE = "openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(root: Path, command: str, log_name: str) -> dict:
    completed = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{root}:/work", "-w", "/work", IMAGE, "bash", "-lc", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log = (completed.stdout or "") + (completed.stderr or "")
    path = root / log_name
    path.write_text(log, encoding="utf-8", errors="replace")
    return {"command": command, "returncode": completed.returncode, "log": log_name, "log_sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    root = args.candidate.resolve()
    receipt_path = root / "bpu_compile_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"refusing to replace {receipt_path}")
    config = root / "config_bpu.yaml"
    config.write_text("""model_parameters:
  onnx_model: './x5_biskill_tcn_fixture.onnx'
  march: 'bayes-e'
  output_model_file_prefix: 'x5_biskill_tcn_fixture_int8'
  working_dir: './model_output'
  layer_out_dump: False
  log_level: 'debug'

input_parameters:
  input_name: 'features'
  input_shape: '1x48x16x1'
  input_type_rt: 'featuremap'
  input_layout_rt: 'NCHW'
  input_type_train: 'featuremap'
  input_layout_train: 'NCHW'
  norm_type: 'no_preprocess'

calibration_parameters:
  cal_data_dir: './calibration_data'
  cal_data_type: 'float32'
  calibration_type: 'max'
  per_channel: True

compiler_parameters:
  compile_mode: 'latency'
  optimize_level: 'O3'
  debug: False
  core_num: 1
""", encoding="utf-8", newline="\n")
    checker = run(root, "hb_mapper checker --model-type onnx --model x5_biskill_tcn_fixture.onnx --march bayes-e --input-shape features 1x48x16x1", "hb_mapper_checker.log")
    maker = run(root, "hb_mapper makertbin --model-type onnx --config config_bpu.yaml", "hb_mapper_makertbin.log") if checker["returncode"] == 0 else {"returncode": -1}
    bins = sorted((root / "model_output").glob("*.bin")) if (root / "model_output").exists() else []
    clean = ANSI_RE.sub("", (root / "hb_mapper_makertbin.log").read_text(encoding="utf-8", errors="replace")) if (root / "hb_mapper_makertbin.log").exists() else ""
    placement = {"bpu_lines": [line.strip() for line in clean.splitlines() if re.search(r"\sBPU\s", line)][:200]}
    passed = checker["returncode"] == 0 and maker.get("returncode") == 0 and len(bins) == 1 and "Convert to runtime bin file successfully!" in clean and bool(placement["bpu_lines"])
    receipt = {
        "schema_version": "xrd-dual-arm-x5-biskill-bayes-e-compile-v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "BPU_COMPILED_BOARD_PENDING" if passed else "BPU_COMPILE_FAILED",
        "toolchain": {"image": IMAGE, "march": "bayes-e"},
        "source_onnx_sha256": sha256(root / "x5_biskill_tcn_fixture.onnx"),
        "config_sha256": sha256(config),
        "checker": checker,
        "makertbin": maker,
        "placement": placement,
        "runtime_binary": ({"file": bins[0].name, "path": f"model_output/{bins[0].name}", "bytes": bins[0].stat().st_size, "sha256": sha256(bins[0])} if len(bins) == 1 else None),
        "board_execution": False,
        "production_modified": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
