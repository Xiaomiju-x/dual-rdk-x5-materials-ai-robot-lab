"""Create a hash-bound, read-only audit of the frozen 24-layer BPU LLM chain."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


SCRIPT = Path(__file__).resolve()
HERE = SCRIPT.parent
REPO = SCRIPT.parents[3]
EVIDENCE = HERE / "evidence"

LEGACY_FILES = (
    "tools/bpu_transformer/qwen2_manual.py",
    "tools/bpu_transformer/qwen2_bpu_split.py",
    "tools/bpu_transformer/gen_calib_local.py",
    "tools/bpu_transformer/run_c2_makertbin.sh",
    "tools/bpu_transformer/config_c2_nir_part1.yaml",
    "tools/bpu_transformer/config_c2_nir_part2.yaml",
    "tools/bpu_transformer/config_c2_verdict_part1.yaml",
    "tools/bpu_transformer/config_c2_verdict_part2.yaml",
    "embodied_brain/car_llm/bpu_llm_server.py",
    "docs/bpu_llm_cluster_v2_report_2026-06-11.md",
)

LEGACY_LOGS = (
    "tools/bpu_transformer/config_c2_nir_part1.log",
    "tools/bpu_transformer/config_c2_nir_part2.log",
    "tools/bpu_transformer/config_c2_verdict_part1.log",
    "tools/bpu_transformer/config_c2_verdict_part2.log",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    files = []
    for relative in LEGACY_FILES:
        path = REPO / relative
        files.append(
            {
                "path": relative,
                "present": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256(path) if path.is_file() else None,
            }
        )

    logs = []
    for relative in LEGACY_LOGS:
        path = REPO / relative
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        logs.append(
            {
                "path": relative,
                "present": path.is_file(),
                "sha256": sha256(path) if path.is_file() else None,
                "conversion_success_marker": "Convert to runtime bin file successfully!" in text,
                "bayes_e_marker": "BPU march           : bayes-e" in text,
                "bpu_operator_rows": len(re.findall(r"\sBPU\s+id\(\d+\)", text)),
                "error_rows": len(re.findall(r"\b(?:ERROR|FAILED|Traceback)\b", text, re.IGNORECASE)),
            }
        )

    old_output_dirs = list((REPO / "tools/bpu_transformer").glob("model_output_c2_*"))
    old_bins = [path for directory in old_output_dirs for path in directory.rglob("*.bin")]
    result = {
        "schema": "x5_icmat_foundry.bpu_llm_legacy_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "READ_ONLY_PC_AUDIT",
        "x5_access_performed": False,
        "production_files_modified": False,
        "conclusions": {
            "manual_24_layer_implementation_present": all(item["present"] for item in files[:2]),
            "historical_two_segment_shape": "layers 0-11 and 12-23, batch=1, seq=64, hidden=896",
            "official_compiler": "OpenExplorer hb_mapper, march bayes-e",
            "all_four_historical_compile_logs_successful": all(
                item["conversion_success_marker"] and item["bayes_e_marker"] for item in logs
            ),
            "all_historical_logs_contain_bpu_operator_rows": all(item["bpu_operator_rows"] > 0 for item in logs),
            "legacy_runtime_contract": "two bins plus CPU embedding/final RMSNorm/LM head; CMA released by process exit",
            "historical_bins_present_in_pc_backup": bool(old_bins),
            "historical_bin_limitation": (
                "The checked-in PC backup preserves compiler logs but not the four historical runtime bins. "
                "The logs prove past compiler success; they do not substitute for new model bins or X5 validation."
            ),
        },
        "source_files": files,
        "compiler_logs": logs,
        "historical_bins": [
            {"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in old_bins
        ],
        "truth_boundary": [
            "No new model was compiled by this audit.",
            "No X5 or Bayes-e runtime was accessed.",
            "Historical latency and CMA numbers remain historical evidence until repeated on the finals X5 image.",
        ],
    }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    output = EVIDENCE / "legacy_24layer_chain_audit.v1.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(sha256(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
