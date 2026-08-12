"""Static ONNX export, checker validation, and ONNX Runtime parity."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .contracts import CLAIM_BOUNDARY, INPUT_SIZE, MODEL_NAME, MODEL_VERSION
from .data import sha256_file, write_json_atomic
from .model import LiteSemSeg


def export_static_onnx(
    model: LiteSemSeg,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "sem_metrology_x5_lite_static_128.onnx"
    example = torch.linspace(
        0.0,
        1.0,
        steps=INPUT_SIZE * INPUT_SIZE,
        dtype=torch.float32,
    ).reshape(1, 1, INPUT_SIZE, INPUT_SIZE)
    model = model.cpu().eval()
    with torch.inference_mode():
        expected = model(example).numpy()
    torch.onnx.export(
        model,
        example,
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    graph = onnx.load(onnx_path)
    onnx.checker.check_model(graph, full_check=True)
    input_dims = [
        dimension.dim_value
        for dimension in graph.graph.input[0].type.tensor_type.shape.dim
    ]
    if input_dims != [1, 1, INPUT_SIZE, INPUT_SIZE]:
        raise ValueError(f"ONNX input is not static: {input_dims}")
    operator_counts: dict[str, int] = {}
    for node in graph.graph.node:
        operator_counts[node.op_type] = operator_counts.get(node.op_type, 0) + 1
    forbidden = {"ConvTranspose", "Loop", "If", "Scan"}
    found_forbidden = forbidden.intersection(operator_counts)
    if found_forbidden:
        raise ValueError(f"forbidden ONNX operators: {sorted(found_forbidden)}")

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(["logits"], {"image": example.numpy()})[0]
    difference = np.abs(expected - actual)
    parity = {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "allclose_atol_1e_5_rtol_1e_4": bool(
            np.allclose(expected, actual, atol=1e-5, rtol=1e-4)
        ),
    }
    if not parity["allclose_atol_1e_5_rtol_1e_4"]:
        raise ValueError(f"ONNX Runtime parity failed: {parity}")

    weights_path = output_dir / "sem_metrology_x5_lite_fp32.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(f"weights missing: {weights_path}")
    manifest = {
        "schema": "icmat_sem_model_manifest.v1",
        "candidate_status": "OFFICIAL_SUBSET_BASELINE",
        "release_eligible": False,
        "deployment_status": "PC_FP32_AND_ONNX_ONLY_NOT_MAPPED_NOT_X5_TESTED",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "input_contract": {
            "name": "image",
            "shape": [1, 1, INPUT_SIZE, INPUT_SIZE],
            "dtype": "float32",
            "range": [0.0, 1.0],
            "dynamic_axes": False,
        },
        "output_contract": {
            "name": "logits",
            "shape": [1, 1, INPUT_SIZE, INPUT_SIZE],
            "activation_external": "sigmoid",
        },
        "architecture": {
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "operator_counts": operator_counts,
            "forbidden_operators_absent": sorted(forbidden),
            "bayes_e_intent": (
                "Fixed-shape Conv/ReLU/MaxPool/nearest-Resize candidate; "
                "mapper and actual Bayes-e support remain unverified."
            ),
        },
        "artifacts": {
            "weights": {
                "path": weights_path.name,
                "bytes": weights_path.stat().st_size,
                "sha256": sha256_file(weights_path),
            },
            "onnx": {
                "path": onnx_path.name,
                "bytes": onnx_path.stat().st_size,
                "sha256": sha256_file(onnx_path),
                "graph_sha256": hashlib.sha256(
                    graph.SerializeToString()
                ).hexdigest(),
            },
        },
        "onnx_runtime_parity": parity,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write_json_atomic(output_dir / "model_manifest.v1.json", manifest)
    return manifest
