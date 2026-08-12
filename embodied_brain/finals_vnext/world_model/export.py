"""Static ONNX export helpers for TinyOccFlowV2."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .model import (
    INPUT_NAME,
    INPUT_SHAPE,
    OUTPUT_NAMES,
    OUTPUT_SHAPES,
    TinyOccFlowV2,
    parameter_statistics,
)

ONNX_OPSET = 11
ALLOWED_ONNX_OPERATORS = frozenset(
    {
        "Add",
        "Concat",
        "Constant",
        "Conv",
        "Relu",
        "Reshape",
        "Resize",
    }
)


def _output_tuple(outputs: object) -> tuple[Tensor, ...]:
    if isinstance(outputs, Tensor):
        return (outputs,)
    if isinstance(outputs, tuple) and all(isinstance(item, Tensor) for item in outputs):
        return outputs
    raise TypeError("model outputs must be a Tensor tuple")


def _shape_from_value_info(value_info: Any) -> tuple[int, ...]:
    tensor_type = value_info.type.tensor_type
    return tuple(int(dimension.dim_value) for dimension in tensor_type.shape.dim)


def validate_onnx_operator_policy(onnx_path: str | Path) -> dict[str, object]:
    """Check static I/O shapes and the Bayes-e operator allowlist."""

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX validation requires the 'onnx' package") from exc

    path = Path(onnx_path)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph, full_check=True)
    operator_counts = Counter(node.op_type for node in graph.graph.node)
    unexpected = sorted(set(operator_counts) - ALLOWED_ONNX_OPERATORS)
    if unexpected:
        raise AssertionError(f"unexpected ONNX operators: {unexpected}")

    input_shapes = tuple(_shape_from_value_info(value) for value in graph.graph.input)
    output_shapes = tuple(_shape_from_value_info(value) for value in graph.graph.output)
    if input_shapes != (INPUT_SHAPE,):
        raise AssertionError(f"unexpected ONNX input shapes: {input_shapes}")
    if output_shapes != OUTPUT_SHAPES:
        raise AssertionError(f"unexpected ONNX output shapes: {output_shapes}")
    return {
        "valid": True,
        "opset": ONNX_OPSET,
        "operator_counts": dict(sorted(operator_counts.items())),
        "input_shapes": [list(shape) for shape in input_shapes],
        "output_shapes": [list(shape) for shape in output_shapes],
    }


def export_tiny_occ_flow_v2_onnx(
    output_path: str | Path,
    *,
    model: nn.Module | None = None,
    seed: int = 20260728,
    validate: bool = True,
) -> dict[str, object]:
    """Export a deterministic, fixed-shape ONNX graph and return its report.

    When ``model`` is omitted, deterministically initialized *untrained*
    weights are exported for graph/toolchain verification only.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if model is None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = TinyOccFlowV2()
        weight_source = f"deterministic_untrained_initialization_seed_{seed}"
    else:
        weight_source = "caller_supplied_model"
    model = model.cpu().eval()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    sample = torch.randn(INPUT_SHAPE, generator=generator, dtype=torch.float32)
    with torch.inference_mode():
        outputs = _output_tuple(model(sample))
    actual_shapes = tuple(tuple(output.shape) for output in outputs)
    if actual_shapes != OUTPUT_SHAPES:
        raise RuntimeError(
            f"PyTorch output shapes {actual_shapes} do not match {OUTPUT_SHAPES}"
        )

    torch.onnx.export(
        model,
        sample,
        path,
        export_params=True,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=list(OUTPUT_NAMES),
        opset_version=ONNX_OPSET,
        dynamic_axes=None,
        dynamo=False,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    report: dict[str, object] = {
        "model": "TinyOccFlowV2",
        "onnx_path": str(path.resolve()),
        "onnx_sha256": digest,
        "opset": ONNX_OPSET,
        "input": {
            "name": INPUT_NAME,
            "shape": list(INPUT_SHAPE),
            "dtype": "float32",
        },
        "outputs": [
            {"name": name, "shape": list(shape), "dtype": "float32"}
            for name, shape in zip(OUTPUT_NAMES, OUTPUT_SHAPES, strict=True)
        ],
        "parameters": parameter_statistics(model),
        "weight_source": weight_source,
        "diagnostic_only": True,
    }
    if validate:
        report["operator_policy"] = validate_onnx_operator_policy(path)
    return report


__all__ = [
    "ALLOWED_ONNX_OPERATORS",
    "ONNX_OPSET",
    "export_tiny_occ_flow_v2_onnx",
    "validate_onnx_operator_policy",
]
