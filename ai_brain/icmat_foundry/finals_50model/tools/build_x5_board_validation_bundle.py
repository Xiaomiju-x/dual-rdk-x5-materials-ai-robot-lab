"""Build small, content-addressed X5 validation fixtures for the frozen staging ZIP.

This tool does not rebuild the release.  It derives fixed board inputs and PC
FP32 references from already accepted artifacts, placing them in a separate
validation bundle.  The release manifest and all frozen artifacts are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[3]
FINAL = ROOT / "icmat_foundry" / "finals_50model"
RELEASE_ZIP = FINAL / "releases" / (
    "x5-icmat-foundry-50model-x5-staging-c5fa215a58168c0c.zip"
)
RELEASE_SHA256 = "c5fa215a58168c0cb7274c2b1cf6d66bcd0f3c1e70d3f4cf13749e9b57dafb52"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_release_manifest() -> dict[str, Any]:
    import zipfile

    if sha256(RELEASE_ZIP) != RELEASE_SHA256:
        raise RuntimeError("authoritative staging ZIP hash mismatch")
    with zipfile.ZipFile(RELEASE_ZIP) as archive:
        manifest = json.loads(archive.read("release_manifest.json"))
        policy = json.loads(archive.read("release_policy.json"))
    assert manifest["kind"] == "x5-staging"
    assert len(manifest["models"]) == 38
    assert policy["automatic_start"] is False
    assert policy["production_overwrite"] is False
    assert policy["rb_voe_state"] == "DEPLOYED_OFF"
    return manifest


def first_np_array(path: Path, preferred: tuple[str, ...] = ()) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        for key in preferred:
            if key in loaded.files:
                return np.asarray(loaded[key])
        for key in loaded.files:
            value = np.asarray(loaded[key])
            if value.dtype.kind in "fiu" and value.size:
                return value
        raise ValueError(f"no numeric array in {path}")
    return np.asarray(loaded)


CPU_ONNX_FIXTURES: dict[str, tuple[str, str | None]] = {
    "F-MAT-06": ("icmat_foundry/finals_50model/artifacts/material_bank/F-MAT-06/fixed_input.npz", None),
    "F-MAT-07": ("icmat_foundry/finals_50model/artifacts/material_bank/F-MAT-07/fixed_input.npz", None),
    "F-MAT-08": ("icmat_foundry/finals_50model/artifacts/material_bank/F-MAT-08/fixed_input.npz", None),
    "F-PROC-07": ("icmat_foundry/finals_50model/artifacts/process_bank/F-PROC-07/ort_sample_input.npy", None),
    "F-PROC-08": ("icmat_foundry/finals_50model/artifacts/process_bank/F-PROC-08/ort_sample_input.npy", None),
    "F-PROC-09": ("icmat_foundry/finals_50model/artifacts/process_bank/F-PROC-09/ort_sample_input.npy", None),
    "F-SEM-05": ("icmat_foundry/finals_50model/artifacts/sem_bank/F-SEM-05/fixed_ort_fixture.npz", "input_fp32"),
    "F-SEM-06": ("icmat_foundry/finals_50model/artifacts/sem_bank/F-SEM-06/fixed_ort_fixture.npz", "input_fp32"),
    "F-PKG-04": ("icmat_foundry/finals_50model/artifacts/package_bank/f_pkg_04/input_fixture.npy", None),
}


GGUF_PROMPTS: dict[str, tuple[str, dict[str, Any]]] = {
    "F-LLM-01": (
        "icmat_foundry/finals_50model/contracts/F-LLM-01-smoke-prompt.txt",
        {
            "required_literals": ["28.4", "980", "UNKNOWN", "MEASURED_VALUES_ONLY"],
            "max_tokens": 96,
        },
    ),
    "F-LLM-02": (
        "icmat_foundry/finals_50model/evidence/llm/F-LLM-02/llama_cpp_cpu_smoke_prompt.txt",
        {
            "required_literals": ["evidence_selection", "ANSWER", "E2.S2"],
            "max_tokens": 96,
        },
    ),
}


def source_onnx_reference(model_path: Path, input_value: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    info = session.get_inputs()[0]
    value = np.ascontiguousarray(input_value, dtype=np.float32)
    # Preserve a dynamic batch dimension.  Reshape only when every declared
    # dimension is static; filtering the symbolic batch would drop rank.
    if info.shape and all(isinstance(item, int) for item in info.shape):
        static = [int(item) for item in info.shape]
        if int(np.prod(static)) == value.size:
            value = value.reshape(tuple(static))
    return np.asarray(session.run(None, {info.name: value})[0]), value


def build_cpu_onnx(
    model: dict[str, Any], fixtures: Path
) -> dict[str, Any]:
    inventory_id = model["inventory_id"]
    model_rel = model["files"][0]
    model_path = ROOT / model_rel
    source_rel, key = CPU_ONNX_FIXTURES[inventory_id]
    source_fixture = ROOT / source_rel
    preferred = (key,) if key else ()
    raw = first_np_array(source_fixture, preferred)
    expected, runtime_input = source_onnx_reference(model_path, raw)
    target = fixtures / f"{inventory_id}.npz"
    np.savez_compressed(
        target,
        input=np.ascontiguousarray(runtime_input, dtype=np.float32),
        expected=np.ascontiguousarray(expected),
    )
    return {
        "inventory_id": inventory_id,
        "primary_backend": "CPU",
        "method": "onnxruntime_cpu",
        "release_files": model["files"],
        "model_sha256": sha256(model_path),
        "fixture": f"fixtures/{target.name}",
        "fixture_sha256": sha256(target),
        "source_fixture": source_rel,
        "source_fixture_sha256": sha256(source_fixture),
    }


def compile_receipt(inventory_id: str) -> tuple[dict[str, Any], Path]:
    if inventory_id == "F-MAT-01":
        source = ROOT / "evaluation/icmat_foundry/propnet/task8_candidate_v2_locked/model_fp32.onnx"
        fixture = ROOT / "icmat_foundry/finals_50model/fixtures/F-MAT-01/input.npz"
        return (
            {
                "source": {"path": str(source.relative_to(ROOT)).replace("\\", "/")},
                "calibration": {"path": str(fixture.relative_to(ROOT)).replace("\\", "/")},
                "compatibility": {"staged_input_shape": [1, 1, 1, 149]},
            },
            FINAL / "evidence" / "material_bank" / "F-MAT-01.receipt.v1.json",
        )
    path = FINAL / "bpu" / "compiled" / inventory_id / "compile_receipt.v1.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def build_bpu_fixed(
    model: dict[str, Any], fixtures: Path
) -> dict[str, Any]:
    inventory_id = model["inventory_id"]
    receipt, receipt_path = compile_receipt(inventory_id)
    source_rel = receipt["source"]["path"].replace("\\", "/")
    source_model = ROOT / source_rel
    fixture_rel = receipt["calibration"]["path"].replace("\\", "/")
    source_fixture = ROOT / fixture_rel
    raw = first_np_array(
        source_fixture,
        ("input_fp32", "features_normalized_fp32", "features_fp32", "xrd_degraded_fp32", "xrd_profile_fp32"),
    )
    if raw.shape[0] > 1:
        raw = raw[:1]
    expected, source_input = source_onnx_reference(source_model, raw)
    runtime_shape = receipt.get("compatibility", {}).get("staged_input_shape")
    runtime_input = source_input
    if runtime_shape and int(np.prod(runtime_shape)) == source_input.size:
        runtime_input = source_input.reshape(tuple(int(item) for item in runtime_shape))
    target = fixtures / f"{inventory_id}.npz"
    np.savez_compressed(
        target,
        input=np.ascontiguousarray(runtime_input, dtype=np.float32),
        expected=np.ascontiguousarray(expected),
    )
    runtime_model = ROOT / model["files"][0]
    return {
        "inventory_id": inventory_id,
        "primary_backend": "BPU",
        "method": "pyeasy_dnn_fixed_diff",
        "release_files": model["files"],
        "model_sha256": sha256(runtime_model),
        "fixture": f"fixtures/{target.name}",
        "fixture_sha256": sha256(target),
        "source_fixture": fixture_rel,
        "source_fixture_sha256": sha256(source_fixture),
        "pc_reference_onnx": source_rel,
        "pc_reference_onnx_sha256": sha256(source_model),
        "compile_receipt": str(receipt_path.relative_to(ROOT)).replace("\\", "/"),
    }


def extract_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        return "\n".join(
            str(item.get("content", "")) for item in messages if isinstance(item, dict)
        )
    raise ValueError("LLM fixture row has no messages")


def build_bpu_llm(model: dict[str, Any], fixtures: Path) -> dict[str, Any]:
    from tokenizers import Tokenizer

    inventory_id = model["inventory_id"]
    files = model["files"]
    embed_rel = next(item for item in files if item.endswith("embed_tokens_fp16.npy"))
    tokenizer_rel = next(item for item in files if item.endswith("tokenizer.json"))
    part1_rel = next(item for item in files if item.endswith("part1.bin"))
    part2_rel = next(item for item in files if item.endswith("part2.bin"))
    norm_rel = next(item for item in files if item.endswith("norm_final_fp32.npy"))
    dataset = FINAL / "data" / "llm_sft" / inventory_id / "train.jsonl"
    first_row = json.loads(next(line for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()))
    prompt = extract_prompt(first_row)
    tokenizer = Tokenizer.from_file(str(ROOT / tokenizer_rel))
    ids = tokenizer.encode(prompt).ids
    if len(ids) <= 64:
        raise RuntimeError(f"{inventory_id} fixed prompt is too short")
    token_ids = np.asarray(ids[:64], dtype=np.int64)
    expected_next = int(ids[64])
    embed = np.load(ROOT / embed_rel, mmap_mode="r")
    hidden = np.asarray(embed[token_ids], dtype=np.float32)[None, ...]
    target = fixtures / f"{inventory_id}.npz"
    np.savez_compressed(
        target,
        input=hidden,
        token_ids=token_ids,
        expected_next_token_id=np.asarray([expected_next], dtype=np.int64),
        score_position=np.asarray([63], dtype=np.int64),
    )
    return {
        "inventory_id": inventory_id,
        "primary_backend": "BPU",
        "method": "bpu_llm_two_process_fixed_token_diff",
        "release_files": files,
        "part1": part1_rel,
        "part2": part2_rel,
        "embed": embed_rel,
        "norm": norm_rel,
        "tokenizer": tokenizer_rel,
        "part1_sha256": sha256(ROOT / part1_rel),
        "part2_sha256": sha256(ROOT / part2_rel),
        "fixture": f"fixtures/{target.name}",
        "fixture_sha256": sha256(target),
        "expected_next_token_id": expected_next,
        "prompt_source": str(dataset.relative_to(ROOT)).replace("\\", "/"),
        "prompt_source_sha256": sha256(dataset),
        "claim_boundary": (
            "Two bins are run in separate processes. The fixed gate is one-token exact semantic continuity; "
            "it is not a general free-generation claim or a numeric FP32 hidden-state differential."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing bundle: {output}")
    fixtures = output / "fixtures"
    prompts = output / "prompts"
    fixtures.mkdir(parents=True)
    prompts.mkdir()
    manifest = load_release_manifest()
    entries: list[dict[str, Any]] = []
    for model in manifest["models"]:
        inventory_id = model["inventory_id"]
        if model["primary_backend"] == "CPU":
            suffix = Path(model["files"][0]).suffix.lower()
            if suffix == ".onnx":
                entries.append(build_cpu_onnx(model, fixtures))
            elif suffix == ".gguf":
                prompt_rel, gate = GGUF_PROMPTS[inventory_id]
                source = ROOT / prompt_rel
                target = prompts / f"{inventory_id}.txt"
                shutil.copy2(source, target)
                entries.append(
                    {
                        "inventory_id": inventory_id,
                        "primary_backend": "CPU",
                        "method": "llama_server_cpu",
                        "release_files": model["files"],
                        "model_sha256": sha256(ROOT / model["files"][0]),
                        "prompt": f"prompts/{target.name}",
                        "prompt_sha256": sha256(target),
                        **gate,
                    }
                )
            elif suffix == ".safetensors":
                entries.append(
                    {
                        "inventory_id": inventory_id,
                        "primary_backend": "CPU",
                        "method": "staging_asset_audit",
                        "release_files": model["files"],
                        "model_sha256": sha256(ROOT / model["files"][0]),
                        "expected_status": "BOARD_REJECTED",
                        "reason": "STAGING_HAS_WEIGHT_ONLY_NO_CONFIG_TOKENIZER_OR_EXECUTABLE_LOADER",
                    }
                )
            else:
                raise ValueError(f"unsupported CPU artifact for {inventory_id}: {suffix}")
        elif inventory_id in {"F-LLM-03", "F-LLM-04", "F-LLM-05"}:
            entries.append(build_bpu_llm(model, fixtures))
        else:
            entries.append(build_bpu_fixed(model, fixtures))
    if len(entries) != 38:
        raise RuntimeError(f"expected 38 entries, got {len(entries)}")
    bundle_manifest = {
        "schema": "x5_icmat_foundry.board_validation_bundle.v1",
        "release_zip_sha256": RELEASE_SHA256,
        "registry_sha256": manifest["registry_sha256"],
        "automatic_start": False,
        "production_overwrite": False,
        "rb_voe_state": "DEPLOYED_OFF",
        "entries": entries,
    }
    write_json(output / "bundle_manifest.json", bundle_manifest)
    inventory = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        inventory.append(
            {
                "path": str(path.relative_to(output)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_json(output / "bundle_inventory.json", {"files": inventory})
    print(json.dumps({"entries": len(entries), "files": len(inventory), "output": str(output)}))


if __name__ == "__main__":
    main()
