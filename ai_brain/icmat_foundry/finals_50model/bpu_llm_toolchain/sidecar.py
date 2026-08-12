"""Content-addressed export, differential test, and OpenExplorer compile CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from manual_qwen2 import ManualRMSNorm, ManualSegment, Qwen2StaticConfig, load_segment_weights


HERE = Path(__file__).resolve().parent
FINAL_ROOT = HERE.parent
CONTRACT = json.loads((HERE / "contracts/models.v1.json").read_text(encoding="utf-8"))
ARCH = CONTRACT["architecture_contract"]
MODEL_MAP = {item["inventory_id"]: item for item in CONTRACT["models"]}
IMAGE = "openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(items: list[dict[str, Any]]) -> str:
    payload = json.dumps(items, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_from_final_root(relative: str) -> Path:
    return (FINAL_ROOT / relative).resolve()


def model_spec(model_id: str) -> dict[str, Any]:
    if model_id not in MODEL_MAP:
        raise ValueError(f"unknown model id: {model_id}")
    return MODEL_MAP[model_id]


def snapshot_files(model_dir: Path) -> list[Path]:
    required = [model_dir / "config.json", model_dir / "model.safetensors", model_dir / "tokenizer.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing merged HF files: " + ", ".join(missing))
    names = (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
    )
    return [model_dir / name for name in names if (model_dir / name).is_file()]


def inspect_model(model_id: str, write: bool = True) -> dict[str, Any]:
    spec = model_spec(model_id)
    model_dir = resolve_from_final_root(spec["merged_hf"])
    if not model_dir.is_dir():
        result = {
            "schema": "x5_icmat_foundry.bpu_llm_inspect.v1",
            "created_at": utc_now(),
            "inventory_id": model_id,
            "state": "WAITING_FOR_MERGED_HF",
            "merged_hf": str(model_dir),
            "x5_access_performed": False,
        }
        if write:
            atomic_json(HERE / "evidence" / model_id / "inspect.v1.json", result)
        return result

    try:
        files = snapshot_files(model_dir)
    except FileNotFoundError as exc:
        result = {
            "schema": "x5_icmat_foundry.bpu_llm_inspect.v1",
            "created_at": utc_now(),
            "inventory_id": model_id,
            "state": "MERGED_HF_INCOMPLETE",
            "merged_hf": str(model_dir),
            "error": str(exc),
            "x5_access_performed": False,
        }
        if write:
            atomic_json(HERE / "evidence" / model_id / "inspect.v1.json", result)
        return result

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    actual = {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "hidden_size": config.get("hidden_size"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "intermediate_size": config.get("intermediate_size"),
    }
    expected = {key: ARCH[key] for key in actual if key != "architectures"}
    architecture_ok = all(actual[key] == expected[key] for key in expected)
    architecture_ok = architecture_ok and "Qwen2ForCausalLM" in actual["architectures"]
    manifest = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    content_hash = canonical_hash(manifest)
    result = {
        "schema": "x5_icmat_foundry.bpu_llm_inspect.v1",
        "created_at": utc_now(),
        "inventory_id": model_id,
        "name": spec["name"],
        "domain": spec["domain"],
        "state": "MERGED_HF_ACCEPTED" if architecture_ok else "ARCHITECTURE_REJECTED",
        "merged_hf": str(model_dir),
        "content_hash": content_hash,
        "content_id": content_hash[:16],
        "architecture_ok": architecture_ok,
        "expected_architecture": ARCH,
        "actual_architecture": actual,
        "files": manifest,
        "x5_access_performed": False,
    }
    if write:
        atomic_json(HERE / "evidence" / model_id / "inspect.v1.json", result)
    return result


def accepted_inspect(model_id: str) -> dict[str, Any]:
    result = inspect_model(model_id)
    if result["state"] != "MERGED_HF_ACCEPTED":
        raise RuntimeError(f"{model_id}: {result['state']}")
    return result


def build_segment(model_dir: Path, part: int, state: dict[str, torch.Tensor]) -> tuple[ManualSegment, dict[str, Any]]:
    cfg = Qwen2StaticConfig.from_hf_config(model_dir / "config.json", int(ARCH["sequence_length"]))
    segment_spec = ARCH["segments"][part - 1]
    start = int(segment_spec["layer_start"])
    end = int(segment_spec["layer_end_inclusive"]) + 1
    segment = ManualSegment(cfg, start, end).eval()
    mapping = load_segment_weights(segment, state, start)
    return segment, mapping


def extract_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        return "\n".join(str(item.get("content", "")) for item in messages if isinstance(item, dict))
    for key in ("prompt", "text", "input", "question"):
        if row.get(key):
            return str(row[key])
    return json.dumps(row, ensure_ascii=False)


def calibration_prompts(path: Path, count: int) -> list[str]:
    prompts: list[str] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            prompts.append(extract_prompt(json.loads(line)))
            if len(prompts) == count:
                break
    if not prompts:
        raise RuntimeError(f"no calibration prompts found in {path}")
    while len(prompts) < count:
        prompts.extend(prompts[: count - len(prompts)])
    return prompts


def export_model(model_id: str, calibration_count: int) -> dict[str, Any]:
    inspected = accepted_inspect(model_id)
    spec = model_spec(model_id)
    model_dir = Path(inspected["merged_hf"])
    output = HERE / "work" / model_id / inspected["content_id"]
    onnx_dir = output / "onnx"
    tensor_dir = output / "cpu_tensors"
    calibration_root = output / "calibration"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)
    state = load_file(str(model_dir / "model.safetensors"), device="cpu")
    cfg = Qwen2StaticConfig.from_hf_config(model_dir / "config.json", int(ARCH["sequence_length"]))
    hidden = torch.randn(1, cfg.max_seq_len, cfg.hidden_size, generator=torch.Generator().manual_seed(20260801))
    segment_reports = []
    segments: list[ManualSegment] = []
    for part in (1, 2):
        segment, mapping = build_segment(model_dir, part, state)
        segments.append(segment)
        destination = onnx_dir / f"{model_id.lower()}_{inspected['content_id']}_part{part}.onnx"
        with torch.inference_mode():
            dry_output = segment(hidden)
        torch.onnx.export(
            segment,
            (hidden,),
            str(destination),
            input_names=["hidden_in"],
            output_names=["hidden_out"],
            opset_version=int(ARCH["onnx_opset"]),
            dynamic_axes=None,
            do_constant_folding=True,
            export_params=True,
        )
        import onnx

        graph = onnx.load(str(destination), load_external_data=False)
        onnx.checker.check_model(graph)
        segment_reports.append(
            {
                "part": part,
                **mapping,
                "input_shape": list(hidden.shape),
                "output_shape": list(dry_output.shape),
                "onnx": str(destination.relative_to(HERE)),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "opset": int(ARCH["onnx_opset"]),
            }
        )

    embed = state["model.embed_tokens.weight"].cpu().numpy().astype(np.float16)
    norm = state["model.norm.weight"].cpu().numpy().astype(np.float32)
    embed_path = tensor_dir / "embed_tokens_fp16.npy"
    norm_path = tensor_dir / "norm_final_fp32.npy"
    np.save(embed_path, embed)
    np.save(norm_path, norm)
    shutil.copy2(model_dir / "tokenizer.json", tensor_dir / "tokenizer.json")

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    pad_id = json.loads((model_dir / "config.json").read_text(encoding="utf-8")).get("pad_token_id")
    if pad_id is None:
        pad_id = json.loads((model_dir / "config.json").read_text(encoding="utf-8")).get("eos_token_id", 0)
    prompts = calibration_prompts(resolve_from_final_root(spec["calibration_jsonl"]), calibration_count)
    part1_cal = calibration_root / "part1"
    part2_cal = calibration_root / "part2"
    part1_cal.mkdir(parents=True, exist_ok=True)
    part2_cal.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt).ids[: cfg.max_seq_len]
        ids += [int(pad_id)] * (cfg.max_seq_len - len(ids))
        part1_input = torch.from_numpy(embed[np.asarray(ids, dtype=np.int64)].astype(np.float32)).unsqueeze(0)
        with torch.inference_mode():
            part2_input = segments[0](part1_input)
        (part1_cal / f"calib_{index:03d}.bin").write_bytes(part1_input.numpy().astype(np.float32).tobytes())
        (part2_cal / f"calib_{index:03d}.bin").write_bytes(part2_input.numpy().astype(np.float32).tobytes())

    cpu_files = [embed_path, norm_path, tensor_dir / "tokenizer.json"]
    receipt = {
        "schema": "x5_icmat_foundry.bpu_llm_export.v1",
        "created_at": utc_now(),
        "inventory_id": model_id,
        "state": "STATIC_ONNX_EXPORTED_CPU_TENSORS_READY",
        "merged_hf_content_hash": inspected["content_hash"],
        "content_id": inspected["content_id"],
        "architecture": ARCH,
        "segments": segment_reports,
        "cpu_tensors": [
            {
                "path": str(path.relative_to(HERE)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in cpu_files
        ],
        "calibration": {
            "source": spec["calibration_jsonl"],
            "samples_per_part": calibration_count,
            "part1_is_token_embedding": True,
            "part2_is_actual_part1_fp32_output": True,
        },
        "x5_access_performed": False,
        "bpu_runtime_tested": False,
    }
    atomic_json(HERE / "evidence" / model_id / "export.v1.json", receipt)
    del state, segments
    return receipt


def compare_values(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    flat_reference = reference.reshape(-1).astype(np.float64)
    flat_candidate = candidate.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(flat_reference) * np.linalg.norm(flat_candidate)
    cosine = float(np.dot(flat_reference, flat_candidate) / denominator) if denominator else 1.0
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()), "cosine": cosine}


def diff_model(model_id: str) -> dict[str, Any]:
    inspected = accepted_inspect(model_id)
    model_dir = Path(inspected["merged_hf"])
    output = HERE / "work" / model_id / inspected["content_id"]
    export_receipt_path = HERE / "evidence" / model_id / "export.v1.json"
    if not export_receipt_path.is_file():
        raise RuntimeError("run export before diff")
    state = load_file(str(model_dir / "model.safetensors"), device="cpu")
    cfg = Qwen2StaticConfig.from_hf_config(model_dir / "config.json", int(ARCH["sequence_length"]))
    segment1, _ = build_segment(model_dir, 1, state)
    segment2, _ = build_segment(model_dir, 2, state)

    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    generator = torch.Generator().manual_seed(20260801)
    input_ids = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len), generator=generator)
    captures: dict[int, torch.Tensor] = {}

    def capture(index: int):
        def hook(_module, _inputs, output):
            captures[index] = (output[0] if isinstance(output, tuple) else output).detach().cpu()
        return hook

    hook11 = hf.model.layers[11].register_forward_hook(capture(11))
    hook23 = hf.model.layers[23].register_forward_hook(capture(23))
    with torch.inference_mode():
        hf.model(input_ids=input_ids, use_cache=False, return_dict=True)
        hidden0 = hf.model.embed_tokens(input_ids).detach().cpu().float()
        manual12 = segment1(hidden0)
        manual24 = segment2(manual12)
    hook11.remove()
    hook23.remove()
    hf_metrics = {
        "part1": compare_values(captures[11].numpy(), manual12.numpy()),
        "part2": compare_values(captures[23].numpy(), manual24.numpy()),
    }
    del hf, captures

    import onnxruntime as ort

    onnx_metrics = {}
    current = hidden0.numpy().astype(np.float32)
    manual_values = [manual12.numpy(), manual24.numpy()]
    for part in (1, 2):
        pattern = list((output / "onnx").glob(f"*_part{part}.onnx"))
        if len(pattern) != 1:
            raise RuntimeError(f"expected one part{part} ONNX, found {len(pattern)}")
        session = ort.InferenceSession(str(pattern[0]), providers=["CPUExecutionProvider"])
        current = session.run(["hidden_out"], {"hidden_in": current})[0]
        onnx_metrics[f"part{part}"] = compare_values(manual_values[part - 1], current)
        del session

    hf_pass = all(item["cosine"] >= 0.99999 and item["mean_abs"] <= 1e-4 for item in hf_metrics.values())
    # ORT may reassociate the large static MatMul/Softmax graph. The observed
    # absolute error is scale-local while cosine remains effectively one; a
    # 2e-5 mean-absolute gate avoids treating harmless FP32 reassociation as a
    # weight-mapping failure.
    onnx_pass = all(item["cosine"] >= 0.999999 and item["mean_abs"] <= 2e-5 for item in onnx_metrics.values())
    receipt = {
        "schema": "x5_icmat_foundry.bpu_llm_diff.v1",
        "created_at": utc_now(),
        "inventory_id": model_id,
        "state": "FP32_DIFFERENTIAL_PASS" if hf_pass and onnx_pass else "FP32_DIFFERENTIAL_FAIL",
        "merged_hf_content_hash": inspected["content_hash"],
        "input_seed": 20260801,
        "manual_vs_hf_layer_outputs": hf_metrics,
        "onnxruntime_vs_manual": onnx_metrics,
        "thresholds": {
            "manual_vs_hf_min_cosine": 0.99999,
            "manual_vs_hf_max_mean_abs": 0.0001,
            "onnx_vs_manual_min_cosine": 0.999999,
            "onnx_vs_manual_max_mean_abs": 0.00002,
        },
        "x5_access_performed": False,
        "bpu_int8_differential_pending": True,
    }
    atomic_json(HERE / "evidence" / model_id / "diff.v1.json", receipt)
    return receipt


def mapper_yaml(onnx_name: str, prefix: str, working_dir: str, calibration_dir: str) -> str:
    return f"""model_parameters:
  onnx_model: '{onnx_name}'
  march: 'bayes-e'
  output_model_file_prefix: '{prefix}'
  working_dir: '{working_dir}'
  log_level: 'info'

input_parameters:
  input_name: 'hidden_in'
  input_shape: '1x64x896'
  input_type_rt: 'featuremap'
  input_layout_rt: 'NCHW'
  input_type_train: 'featuremap'
  input_layout_train: 'NCHW'
  norm_type: 'no_preprocess'

calibration_parameters:
  cal_data_dir: '{calibration_dir}'
  cal_data_type: 'float32'
  calibration_type: 'kl'

compiler_parameters:
  optimize_level: 'O1'
  debug: False
  core_num: 1
"""


def compile_model(model_id: str, calibration_count: int = 1) -> dict[str, Any]:
    inspected = accepted_inspect(model_id)
    diff_path = HERE / "evidence" / model_id / "diff.v1.json"
    if not diff_path.is_file() or json.loads(diff_path.read_text(encoding="utf-8"))["state"] != "FP32_DIFFERENTIAL_PASS":
        raise RuntimeError("FP32 differential receipt must pass before compilation")
    output = HERE / "work" / model_id / inspected["content_id"]
    compile_root = output / "openexplorer"
    compile_root.mkdir(parents=True, exist_ok=True)
    if calibration_count < 1:
        raise ValueError("calibration_count must be positive")
    segment_results = []
    for part in (1, 2):
        onnx_files = list((output / "onnx").glob(f"*_part{part}.onnx"))
        if len(onnx_files) != 1:
            raise RuntimeError(f"expected one part{part} ONNX")
        prefix = f"{model_id.lower()}_{inspected['content_id']}_part{part}"
        config_path = compile_root / f"part{part}.yaml"
        source_calibration = sorted((output / "calibration" / f"part{part}").glob("calib_*.bin"))
        if len(source_calibration) < calibration_count:
            raise RuntimeError(
                f"part{part} has {len(source_calibration)} calibration samples; {calibration_count} requested"
            )
        compile_calibration = compile_root / "calibration_compile" / f"part{part}"
        if compile_calibration.exists():
            shutil.rmtree(compile_calibration)
        compile_calibration.mkdir(parents=True)
        for source in source_calibration[:calibration_count]:
            shutil.copy2(source, compile_calibration / source.name)
        part_output = compile_root / f"part{part}_output"
        if part_output.exists():
            shutil.rmtree(part_output)
        config_path.write_text(
            mapper_yaml(
                f"../onnx/{onnx_files[0].name}",
                prefix,
                f"part{part}_output",
                f"calibration_compile/part{part}",
            ),
            encoding="utf-8",
        )
        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{output}:/work",
            "-w",
            "/work/openexplorer",
            IMAGE,
            "hb_mapper",
            "makertbin",
            "--config",
            f"part{part}.yaml",
            "--model-type",
            "onnx",
        ]
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log_path = compile_root / f"part{part}.log"
        log_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
        bins = list((compile_root / f"part{part}_output").glob("*.bin"))
        success_marker = "Convert to runtime bin file successfully!" in completed.stdout
        if completed.returncode != 0 or not success_marker or len(bins) != 1:
            raise RuntimeError(
                f"part{part} compile failed: rc={completed.returncode}, marker={success_marker}, bins={len(bins)}; "
                f"see {log_path}"
            )
        segment_results.append(
            {
                "part": part,
                "config": str(config_path.relative_to(HERE)),
                "config_sha256": sha256(config_path),
                "log": str(log_path.relative_to(HERE)),
                "log_sha256": sha256(log_path),
                "bin": str(bins[0].relative_to(HERE)),
                "bin_bytes": bins[0].stat().st_size,
                "bin_sha256": sha256(bins[0]),
                "success_marker": success_marker,
                "bpu_operator_rows": completed.stdout.count(" BPU  id("),
                "calibration_samples_used": calibration_count,
            }
        )
    receipt = {
        "schema": "x5_icmat_foundry.bpu_llm_compile.v1",
        "created_at": utc_now(),
        "inventory_id": model_id,
        "state": "BAYES_E_BINS_COMPILED_PC_X5_PENDING",
        "merged_hf_content_hash": inspected["content_hash"],
        "openexplorer_image": IMAGE,
        "segments": segment_results,
        "deployment": "ON_DEMAND_ONLY",
        "autostart": False,
        "decision_authority": False,
        "x5_access_performed": False,
        "x5_runtime_tested": False,
        "compiler_output_is_not_x5_runtime_evidence": True,
    }
    atomic_json(HERE / "evidence" / model_id / "compile.v1.json", receipt)
    return receipt


def status() -> dict[str, Any]:
    models = []
    for model_id in MODEL_MAP:
        inspected = inspect_model(model_id)
        entry = {"inventory_id": model_id, "inspect_state": inspected["state"]}
        for name in ("export", "diff", "compile"):
            path = HERE / "evidence" / model_id / f"{name}.v1.json"
            entry[f"{name}_state"] = json.loads(path.read_text(encoding="utf-8"))["state"] if path.is_file() else "NOT_RUN"
        models.append(entry)
    result = {
        "schema": "x5_icmat_foundry.bpu_llm_sidecar_status.v1",
        "created_at": utc_now(),
        "state": "SIDECAR_STAGED",
        "models": models,
        "default_state": "DEPLOYED_OFF",
        "x5_access_performed": False,
    }
    atomic_json(HERE / "evidence" / "sidecar_status.v1.json", result)
    return result


def print_result(result: dict[str, Any]) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "export", "diff", "compile", "all"):
        child = subparsers.add_parser(command)
        child.add_argument("--model-id", required=True, choices=sorted(MODEL_MAP))
        if command in ("export", "all"):
            child.add_argument("--calibration-count", type=int, default=8)
        if command == "compile":
            child.add_argument("--calibration-count", type=int, default=1)
        if command == "all":
            child.add_argument("--compile-calibration-count", type=int, default=1)
    subparsers.add_parser("status")
    args = parser.parse_args()
    if args.command == "inspect":
        print_result(inspect_model(args.model_id))
    elif args.command == "export":
        print_result(export_model(args.model_id, args.calibration_count))
    elif args.command == "diff":
        print_result(diff_model(args.model_id))
    elif args.command == "compile":
        print_result(compile_model(args.model_id, args.calibration_count))
    elif args.command == "all":
        print_result(export_model(args.model_id, args.calibration_count))
        print_result(diff_model(args.model_id))
        print_result(compile_model(args.model_id, args.compile_calibration_count))
    else:
        print_result(status())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
