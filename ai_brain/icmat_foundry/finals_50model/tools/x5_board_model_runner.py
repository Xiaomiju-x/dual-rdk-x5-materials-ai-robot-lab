#!/usr/bin/env python3
"""Run exactly one isolated X5 candidate artifact and emit one JSON receipt.

The process has no service, camera, network-configuration, GPIO, serial, or
production-write authority.  BPU memory is released by process exit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def memory_kib() -> dict[str, int]:
    wanted = {"MemAvailable", "SwapFree", "CmaTotal", "CmaFree"}
    result: dict[str, int] = {}
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in wanted:
            result[key] = int(value.split()[0])
    return result


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = reference.reshape(-1).astype(np.float64)
    out = candidate.reshape(-1).astype(np.float64)
    if ref.size != out.size:
        return {
            "shape_compatible": False,
            "reference_size": int(ref.size),
            "candidate_size": int(out.size),
            "task_gate_pass": False,
        }
    delta = np.abs(out - ref)
    denom = float(np.sqrt(np.mean(ref * ref))) + 1e-12
    nrmse = float(np.sqrt(np.mean((out - ref) ** 2)) / denom)
    norm_product = float(np.linalg.norm(ref) * np.linalg.norm(out))
    cosine = float(np.dot(ref, out) / norm_product) if norm_product else 1.0
    sign_agreement = float(np.mean(np.signbit(ref) == np.signbit(out)))
    argmax_equal = int(np.argmax(ref)) == int(np.argmax(out))
    max_abs = float(delta.max())
    if ref.size == 1:
        task_gate = max_abs <= max(0.1, 0.1 * abs(float(ref[0])))
        gate = "SCALAR_ABS_OR_REL_10_PERCENT"
    elif ref.size <= 128:
        task_gate = argmax_equal and cosine >= 0.95 and nrmse <= 0.25
        gate = "SMALL_VECTOR_ARGMAX_COSINE_NRMSE"
    else:
        task_gate = cosine >= 0.99 and nrmse <= 0.10 and sign_agreement >= 0.95
        gate = "DENSE_OUTPUT_COSINE_NRMSE_SIGN"
    return {
        "shape_compatible": True,
        "max_abs_diff": max_abs,
        "mean_abs_diff": float(delta.mean()),
        "nrmse": nrmse,
        "cosine": cosine,
        "sign_agreement": sign_agreement,
        "argmax_equal": argmax_equal,
        "task_gate": gate,
        "task_gate_pass": bool(task_gate),
    }


def common(
    inventory_id: str,
    backend: str,
    model: pathlib.Path,
    fixture: pathlib.Path | None,
    before: dict[str, int],
    mid: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema": "x5_icmat_foundry.single_model_board_receipt.v1",
        "inventory_id": inventory_id,
        "backend": backend,
        "model_path": str(model),
        "model_bytes": model.stat().st_size,
        "model_sha256": sha256(model),
        "fixture_path": str(fixture) if fixture else None,
        "fixture_sha256": sha256(fixture) if fixture else None,
        "pid": os.getpid(),
        "resource_before_kib": before,
        "resource_after_load_kib": mid,
        "network_configuration_accessed": False,
        "production_files_modified": False,
        "service_registered": False,
        "camera_accessed": False,
        "robot_io_accessed": False,
    }


def run_onnx(args: argparse.Namespace) -> dict[str, Any]:
    import onnxruntime as ort

    before = memory_kib()
    fixture = np.load(args.fixture, allow_pickle=False)
    value = np.ascontiguousarray(fixture["input"], dtype=np.float32)
    expected = np.asarray(fixture["expected"])
    started = time.perf_counter()
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - started) * 1000.0
    mid = memory_kib()
    input_name = session.get_inputs()[0].name
    started = time.perf_counter()
    output = np.asarray(session.run(None, {input_name: value})[0])
    inference_ms = (time.perf_counter() - started) * 1000.0
    metrics = compare(expected, output)
    result = common(
        args.inventory_id,
        "onnxruntime.CPUExecutionProvider",
        args.model,
        args.fixture,
        before,
        mid,
    )
    result.update(
        {
            "actual_backend": "CPU",
            "input_shape": list(value.shape),
            "input_tensor_sha256": tensor_sha256(value),
            "output_shape": list(output.shape),
            "output_tensor_sha256": tensor_sha256(output),
            "output_preview": output.reshape(-1)[:16].astype(float).tolist(),
            "finite": bool(np.isfinite(output).all()),
            "load_ms": load_ms,
            "inference_ms": inference_ms,
            "differential": metrics,
            "resource_after_inference_kib": memory_kib(),
        }
    )
    result["status"] = (
        "X5_VALIDATED"
        if result["finite"] and metrics.get("task_gate_pass")
        else "BOARD_REJECTED"
    )
    return result


def run_bpu(args: argparse.Namespace) -> dict[str, Any]:
    from hobot_dnn import pyeasy_dnn as dnn

    before = memory_kib()
    fixture = np.load(args.fixture, allow_pickle=False)
    value = np.ascontiguousarray(fixture["input"], dtype=np.float32)
    expected = np.asarray(fixture["expected"])
    started = time.perf_counter()
    models = dnn.load(str(args.model))
    load_ms = (time.perf_counter() - started) * 1000.0
    if len(models) != 1:
        raise RuntimeError(f"expected one packed model, got {len(models)}")
    model = models[0]
    mid = memory_kib()
    started = time.perf_counter()
    outputs = model.forward(value)
    inference_ms = (time.perf_counter() - started) * 1000.0
    if not outputs:
        raise RuntimeError("empty BPU output")
    output = np.asarray(outputs[0].buffer)
    metrics = compare(expected, output)
    result = common(
        args.inventory_id,
        "hobot_dnn.pyeasy_dnn",
        args.model,
        args.fixture,
        before,
        mid,
    )
    result.update(
        {
            "actual_backend": "BPU",
            "input_shape": list(value.shape),
            "input_tensor_sha256": tensor_sha256(value),
            "output_shape": list(output.shape),
            "output_tensor_sha256": tensor_sha256(output),
            "output_preview": output.reshape(-1)[:16].astype(float).tolist(),
            "reference_preview": expected.reshape(-1)[:16].astype(float).tolist(),
            "finite": bool(np.isfinite(output).all()),
            "load_ms": load_ms,
            "inference_ms": inference_ms,
            "differential": metrics,
            "resource_after_inference_kib": memory_kib(),
        }
    )
    result["status"] = (
        "X5_VALIDATED"
        if result["finite"] and metrics.get("task_gate_pass")
        else "BOARD_EXPERIMENTAL"
    )
    return result


def parse_shape(text: str, label: str) -> tuple[int, ...]:
    match = re.search(rf"{re.escape(label)}:\s*\(([^)]*)\)", text)
    if not match:
        raise ValueError(f"model_info missing {label}")
    return tuple(int(item.strip()) for item in match.group(1).split(",") if item.strip())


def run_hrt(args: argparse.Namespace) -> dict[str, Any]:
    before = memory_kib()
    fixture = np.load(args.fixture, allow_pickle=False)
    value = np.ascontiguousarray(fixture["input"], dtype="<f4")
    expected = np.asarray(fixture["expected"])
    if args.scratch.exists():
        raise FileExistsError(f"scratch already exists: {args.scratch}")
    dump = args.scratch / "dump"
    dump.mkdir(parents=True)
    input_path = args.scratch / "input_f32.bin"
    value.tofile(input_path)
    info_process = subprocess.run(
        ["hrt_model_exec", "model_info", "--model_file", str(args.model)],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    info_text = info_process.stdout + "\n" + info_process.stderr
    if info_process.returncode:
        raise RuntimeError(f"hrt model_info failed: {info_text[-2000:]}")
    name_match = re.search(r"\[model name\]:\s*(\S+)", info_text)
    if not name_match:
        raise ValueError("unable to parse hrt model name")
    model_name = name_match.group(1)
    output_section_match = re.search(
        r"output\[0\]:(.*?)(?:\noutput\[1\]:|\n-+)", info_text, re.DOTALL
    )
    if not output_section_match:
        raise ValueError("unable to parse first hrt output contract")
    output_section = output_section_match.group(1)
    valid_shape = parse_shape(output_section, "valid shape")
    aligned_shape = parse_shape(output_section, "aligned shape")
    command = [
        "hrt_model_exec",
        "infer",
        "--model_file",
        str(args.model),
        "--model_name",
        model_name,
        "--input_file",
        str(input_path),
        "--enable_dump",
        "true",
        "--dump_path",
        str(dump),
        "--dump_format",
        "txt",
        "--dump_precision",
        "9",
    ]
    started = time.perf_counter()
    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    if process.returncode:
        raise RuntimeError(
            f"hrt infer failed: {(process.stdout + process.stderr)[-3000:]}"
        )
    output_files = sorted(dump.glob("model_infer_output_0_*.txt"))
    if len(output_files) != 1:
        raise RuntimeError(f"expected one first-output dump, got {len(output_files)}")
    values = np.fromstring(output_files[0].read_text(encoding="utf-8"), sep=" ")
    valid_size = int(np.prod(valid_shape))
    aligned_size = int(np.prod(aligned_shape))
    if values.size == aligned_size:
        aligned = values.reshape(aligned_shape)
        slices = tuple(slice(0, size) for size in valid_shape)
        output = np.asarray(aligned[slices]).reshape(valid_shape)
    elif values.size == valid_size:
        output = values.reshape(valid_shape)
    else:
        raise ValueError(
            f"unexpected hrt dump size {values.size}; valid={valid_size}, aligned={aligned_size}"
        )
    output = output.astype(np.float32)
    metrics = compare(expected, output)
    load_match = re.search(r"Load model to DDR cost\s+([0-9.]+)ms", process.stdout)
    infer_match = re.search(r"Infer time:\s*([0-9.]+)\s*ms", process.stdout)
    result = common(
        args.inventory_id,
        "hrt_model_exec Bayes-e BPU",
        args.model,
        args.fixture,
        before,
        memory_kib(),
    )
    result.update(
        {
            "actual_backend": "BPU",
            "model_name": model_name,
            "input_shape": list(value.shape),
            "input_tensor_sha256": tensor_sha256(value),
            "output_valid_shape": list(valid_shape),
            "output_aligned_shape": list(aligned_shape),
            "output_tensor_sha256": tensor_sha256(output),
            "output_preview": output.reshape(-1)[:16].astype(float).tolist(),
            "reference_preview": expected.reshape(-1)[:16].astype(float).tolist(),
            "finite": bool(np.isfinite(output).all()),
            "load_ms": float(load_match.group(1)) if load_match else None,
            "inference_ms": float(infer_match.group(1)) if infer_match else None,
            "wall_ms": wall_ms,
            "differential": metrics,
            "dump_file": str(output_files[0]),
            "dump_file_sha256": sha256(output_files[0]),
            "hrt_stdout": process.stdout,
            "hrt_stderr": process.stderr,
            "resource_after_inference_kib": memory_kib(),
        }
    )
    result["status"] = (
        "X5_VALIDATED"
        if result["finite"] and metrics.get("task_gate_pass")
        else "BOARD_EXPERIMENTAL"
    )
    return result


def run_asset_audit(args: argparse.Namespace) -> dict[str, Any]:
    before = memory_kib()
    missing = [
        name for name in ("config.json", "tokenizer.json") if not (args.model.parent / name).is_file()
    ]
    try:
        import transformers  # type: ignore  # noqa: F401

        transformers_available = True
    except Exception:
        transformers_available = False
    result = common(
        args.inventory_id,
        "NO_EXECUTABLE_LOADER",
        args.model,
        None,
        before,
        memory_kib(),
    )
    result.update(
        {
            "actual_backend": None,
            "missing_companion_assets": missing,
            "transformers_available": transformers_available,
            "status": "BOARD_REJECTED",
            "reason": "STAGING_RUNTIME_ASSET_INCOMPLETE",
            "resource_after_inference_kib": memory_kib(),
        }
    )
    return result


def http_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run_gguf(args: argparse.Namespace) -> dict[str, Any]:
    before = memory_kib()
    prompt_text = args.prompt.read_text(encoding="utf-8")
    command = [
        str(args.llama_server),
        "-m",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "-c",
        "2048",
        "-t",
        "4",
        "-ngl",
        "0",
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    output = ""
    try:
        health_url = f"http://127.0.0.1:{args.port}/health"
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate(timeout=5)
                raise RuntimeError(f"llama-server exited early: {stdout[-1000:]} {stderr[-1000:]}")
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(0.5)
        else:
            raise TimeoutError("llama-server health timeout")
        load_ms = (time.perf_counter() - started) * 1000.0
        mid = memory_kib()
        infer_started = time.perf_counter()
        response = http_json(
            f"http://127.0.0.1:{args.port}/completion",
            {
                "prompt": prompt_text,
                "n_predict": args.max_tokens,
                "temperature": 0,
                "seed": 20260804,
                "stream": False,
                "cache_prompt": False,
            },
            180.0,
        )
        inference_ms = (time.perf_counter() - infer_started) * 1000.0
        output = str(response.get("content", ""))
        literals = json.loads(args.required_literals)
        literal_checks = {item: item in output for item in literals}
        result = common(
            args.inventory_id,
            "llama.cpp CPU llama-server",
            args.model,
            args.prompt,
            before,
            mid,
        )
        result.update(
            {
                "actual_backend": "CPU",
                "candidate_pid": process.pid,
                "candidate_port": args.port,
                "load_ms": load_ms,
                "inference_ms": inference_ms,
                "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                "output_preview": output[:2000],
                "required_literal_checks": literal_checks,
                "resource_after_inference_kib": memory_kib(),
                "status": "X5_VALIDATED" if output and all(literal_checks.values()) else "BOARD_EXPERIMENTAL",
            }
        )
        return result
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()


def run_llm_part1(args: argparse.Namespace) -> dict[str, Any]:
    from hobot_dnn import pyeasy_dnn as dnn

    before = memory_kib()
    fixture = np.load(args.fixture, allow_pickle=False)
    value = np.ascontiguousarray(fixture["input"], dtype=np.float32)
    started = time.perf_counter()
    models = dnn.load(str(args.model))
    load_ms = (time.perf_counter() - started) * 1000.0
    if len(models) != 1:
        raise RuntimeError("unexpected packed model count")
    mid = memory_kib()
    started = time.perf_counter()
    outputs = models[0].forward(value)
    inference_ms = (time.perf_counter() - started) * 1000.0
    if not outputs:
        raise RuntimeError("empty BPU output")
    output = np.asarray(outputs[0].buffer)
    np.save(args.output, np.ascontiguousarray(output, dtype=np.float32))
    result = common(
        args.inventory_id,
        "hobot_dnn.pyeasy_dnn part1",
        args.model,
        args.fixture,
        before,
        mid,
    )
    result.update(
        {
            "segment": "part1_layers_0_11",
            "actual_backend": "BPU",
            "input_shape": list(value.shape),
            "output_shape": list(output.shape),
            "output_tensor_sha256": tensor_sha256(output),
            "finite": bool(np.isfinite(output).all()),
            "load_ms": load_ms,
            "inference_ms": inference_ms,
            "resource_after_inference_kib": memory_kib(),
            "status": "SEGMENT_X5_EXECUTED" if np.isfinite(output).all() else "BOARD_REJECTED",
        }
    )
    return result


def run_llm_part2(args: argparse.Namespace) -> dict[str, Any]:
    from hobot_dnn import pyeasy_dnn as dnn

    before = memory_kib()
    fixture = np.load(args.fixture, allow_pickle=False)
    expected = int(fixture["expected_next_token_id"][0])
    position = int(fixture["score_position"][0])
    value = np.ascontiguousarray(np.load(args.input), dtype=np.float32)
    started = time.perf_counter()
    models = dnn.load(str(args.model))
    load_ms = (time.perf_counter() - started) * 1000.0
    if len(models) != 1:
        raise RuntimeError("unexpected packed model count")
    mid = memory_kib()
    started = time.perf_counter()
    outputs = models[0].forward(value)
    inference_ms = (time.perf_counter() - started) * 1000.0
    if not outputs:
        raise RuntimeError("empty BPU output")
    output = np.asarray(outputs[0].buffer)
    if output.size != 64 * 896:
        raise ValueError(f"unexpected hidden size/shape {output.size} {output.shape}")
    hidden = output.reshape(1, 64, 896)[0, position].astype(np.float32)
    norm = np.asarray(np.load(args.norm), dtype=np.float32)
    hidden = hidden / math.sqrt(float(np.mean(hidden * hidden)) + 1e-6) * norm
    embed = np.load(args.embed, mmap_mode="r")
    best_id = -1
    best_score = -float("inf")
    for start in range(0, embed.shape[0], 4096):
        block = np.asarray(embed[start : start + 4096], dtype=np.float32)
        scores = block @ hidden
        local = int(np.argmax(scores))
        score = float(scores[local])
        if score > best_score:
            best_score = score
            best_id = start + local
    result = common(
        args.inventory_id,
        "hobot_dnn.pyeasy_dnn part2 + CPU tied LM head",
        args.model,
        args.fixture,
        before,
        mid,
    )
    result.update(
        {
            "segment": "part2_layers_12_23",
            "actual_backend": "BPU_WITH_CPU_FINAL_NORM_AND_TIED_LM_HEAD",
            "input_shape": list(value.shape),
            "input_tensor_sha256": tensor_sha256(value),
            "output_shape": list(output.shape),
            "output_tensor_sha256": tensor_sha256(output),
            "finite": bool(np.isfinite(output).all()),
            "load_ms": load_ms,
            "inference_ms": inference_ms,
            "expected_next_token_id": expected,
            "actual_next_token_id": best_id,
            "next_token_exact": best_id == expected,
            "best_logit": best_score,
            "resource_after_inference_kib": memory_kib(),
            "status": "X5_VALIDATED_FIXED_TOKEN_CONTRACT" if best_id == expected else "BOARD_EXPERIMENTAL",
            "claim_boundary": "One fixed next-token contract only; not general free generation or FP32 hidden-state parity.",
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("onnx", "bpu", "asset-audit"):
        child = sub.add_parser(name)
        child.add_argument("--inventory-id", required=True)
        child.add_argument("--model", type=pathlib.Path, required=True)
        if name in {"onnx", "bpu"}:
            child.add_argument("--fixture", type=pathlib.Path, required=True)
    hrt = sub.add_parser("hrt")
    hrt.add_argument("--inventory-id", required=True)
    hrt.add_argument("--model", type=pathlib.Path, required=True)
    hrt.add_argument("--fixture", type=pathlib.Path, required=True)
    hrt.add_argument("--scratch", type=pathlib.Path, required=True)
    gguf = sub.add_parser("gguf")
    gguf.add_argument("--inventory-id", required=True)
    gguf.add_argument("--model", type=pathlib.Path, required=True)
    gguf.add_argument("--prompt", type=pathlib.Path, required=True)
    gguf.add_argument("--llama-server", type=pathlib.Path, required=True)
    gguf.add_argument("--port", type=int, required=True)
    gguf.add_argument("--max-tokens", type=int, required=True)
    gguf.add_argument("--required-literals", required=True)
    part1 = sub.add_parser("llm-part1")
    part1.add_argument("--inventory-id", required=True)
    part1.add_argument("--model", type=pathlib.Path, required=True)
    part1.add_argument("--fixture", type=pathlib.Path, required=True)
    part1.add_argument("--output", type=pathlib.Path, required=True)
    part2 = sub.add_parser("llm-part2")
    part2.add_argument("--inventory-id", required=True)
    part2.add_argument("--model", type=pathlib.Path, required=True)
    part2.add_argument("--fixture", type=pathlib.Path, required=True)
    part2.add_argument("--input", type=pathlib.Path, required=True)
    part2.add_argument("--embed", type=pathlib.Path, required=True)
    part2.add_argument("--norm", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "onnx":
        result = run_onnx(args)
    elif args.command == "bpu":
        result = run_bpu(args)
    elif args.command == "asset-audit":
        result = run_asset_audit(args)
    elif args.command == "hrt":
        result = run_hrt(args)
    elif args.command == "gguf":
        result = run_gguf(args)
    elif args.command == "llm-part1":
        result = run_llm_part1(args)
    elif args.command == "llm-part2":
        result = run_llm_part2(args)
    else:
        raise AssertionError(args.command)
    print("RECEIPT_JSON=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            "ERROR_JSON="
            + json.dumps(
                {
                    "schema": "x5_icmat_foundry.single_model_board_error.v1",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "resource_at_error_kib": memory_kib(),
                },
                separators=(",", ":"),
            )
        )
        raise
