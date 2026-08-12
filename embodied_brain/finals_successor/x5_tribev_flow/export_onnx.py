"""Export X5-TriBEV-Flow candidate models as static ONNX opset 11 graphs.

Examples:

    python export_onnx.py --model all --output-dir ./onnx --check
    python export_onnx.py --model tiny_occ_flow --tiny-weights checkpoint.pt --check

When no checkpoint is supplied, the script exports deterministic initialized
weights for graph/toolchain validation only. Such an artifact is not a trained
navigation model and must not be promoted to a board runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

if __package__:
    from .models import (
        CAM_SEM_INPUT_NAME,
        CAM_SEM_INPUT_SHAPE,
        CAM_SEM_OUTPUT_NAMES,
        TINY_OCC_FLOW_INPUT_NAME,
        TINY_OCC_FLOW_INPUT_SHAPE,
        TINY_OCC_FLOW_OUTPUT_NAMES,
        CamSemLite,
        TinyOccFlowStudent,
        parameter_statistics,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from models import (  # type: ignore[no-redef]
        CAM_SEM_INPUT_NAME,
        CAM_SEM_INPUT_SHAPE,
        CAM_SEM_OUTPUT_NAMES,
        TINY_OCC_FLOW_INPUT_NAME,
        TINY_OCC_FLOW_INPUT_SHAPE,
        TINY_OCC_FLOW_OUTPUT_NAMES,
        CamSemLite,
        TinyOccFlowStudent,
        parameter_statistics,
    )


OPSET_VERSION = 11
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "onnx"
ALLOWED_ONNX_OPERATORS = frozenset(
    {
        "Add",
        "Concat",
        "Constant",
        "Conv",
        "GlobalAveragePool",
        "Relu",
        "Reshape",
        "Resize",
    }
)


@dataclass(frozen=True)
class ExportDefinition:
    """Immutable ONNX export contract for one candidate model."""

    name: str
    filename: str
    input_name: str
    input_shape: tuple[int, int, int, int]
    output_names: tuple[str, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    builder: Callable[[], nn.Module]


EXPORT_DEFINITIONS = {
    "tiny_occ_flow": ExportDefinition(
        name="tiny_occ_flow",
        filename="tiny_occ_flow_student_opset11.onnx",
        input_name=TINY_OCC_FLOW_INPUT_NAME,
        input_shape=TINY_OCC_FLOW_INPUT_SHAPE,
        output_names=TINY_OCC_FLOW_OUTPUT_NAMES,
        output_shapes=(
            (1, 3, 64, 64),
            (1, 6, 32, 32),
            (1, 6, 64, 64),
            (1, 9),
        ),
        builder=TinyOccFlowStudent,
    ),
    "cam_sem_lite": ExportDefinition(
        name="cam_sem_lite",
        filename="cam_sem_lite_opset11.onnx",
        input_name=CAM_SEM_INPUT_NAME,
        input_shape=CAM_SEM_INPUT_SHAPE,
        output_names=CAM_SEM_OUTPUT_NAMES,
        output_shapes=((1, 6, 72, 128), (1, 4)),
        builder=CamSemLite,
    ),
}


def _extract_state_dict(checkpoint: object) -> Mapping[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a state_dict or a mapping containing one")

    candidate: object = checkpoint
    for key in ("state_dict", "model_state_dict", "model"):
        nested = checkpoint.get(key)
        if isinstance(nested, Mapping):
            candidate = nested
            break
    if not isinstance(candidate, Mapping):
        raise TypeError("checkpoint does not contain a state_dict mapping")

    normalized: dict[str, Tensor] = {}
    for raw_key, value in candidate.items():
        if not isinstance(raw_key, str) or not isinstance(value, Tensor):
            raise TypeError("state_dict keys must be strings and values must be tensors")
        key = raw_key.removeprefix("module.")
        normalized[key] = value
    return normalized


def load_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    """Load a strict CPU state_dict from a training checkpoint."""

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)


def _deterministic_sample(shape: Sequence[int], seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32)


def _as_output_tuple(outputs: object) -> tuple[Tensor, ...]:
    if isinstance(outputs, Tensor):
        return (outputs,)
    if isinstance(outputs, tuple) and all(isinstance(item, Tensor) for item in outputs):
        return outputs
    raise TypeError("model output must be a Tensor or tuple of Tensors")


def export_model(
    definition: ExportDefinition,
    output_path: Path,
    *,
    checkpoint_path: Path | None = None,
    seed: int = 20260728,
) -> tuple[nn.Module, Tensor, dict[str, object]]:
    """Instantiate and export one fixed-shape model.

    Returns the evaluated PyTorch model, deterministic comparison input, and
    a JSON-serializable parameter/export report.
    """

    torch.manual_seed(seed)
    model = definition.builder().cpu().eval()
    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path)

    sample = _deterministic_sample(definition.input_shape, seed + 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        outputs = _as_output_tuple(model(sample))
    actual_shapes = tuple(tuple(output.shape) for output in outputs)
    if actual_shapes != definition.output_shapes:
        raise RuntimeError(
            f"{definition.name} PyTorch output shapes {actual_shapes} "
            f"do not match contract {definition.output_shapes}"
        )

    torch.onnx.export(
        model,
        sample,
        output_path,
        export_params=True,
        do_constant_folding=True,
        input_names=[definition.input_name],
        output_names=list(definition.output_names),
        opset_version=OPSET_VERSION,
    )

    report: dict[str, object] = {
        "model": definition.name,
        "onnx_path": str(output_path.resolve()),
        "opset": OPSET_VERSION,
        "input": {
            "name": definition.input_name,
            "shape": list(definition.input_shape),
            "dtype": "float32",
        },
        "outputs": [
            {"name": name, "shape": list(shape), "dtype": "float32"}
            for name, shape in zip(
                definition.output_names,
                definition.output_shapes,
                strict=True,
            )
        ],
        "parameters": parameter_statistics(model),
        "trained_checkpoint_supplied": checkpoint_path is not None,
        "weights": (
            str(checkpoint_path.resolve())
            if checkpoint_path is not None
            else f"deterministic_untrained_initialization_seed_{seed}"
        ),
    }
    return model, sample, report


def _onnx_shape(value_info: object) -> tuple[int, ...]:
    tensor_type = value_info.type.tensor_type  # type: ignore[attr-defined]
    return tuple(int(dimension.dim_value) for dimension in tensor_type.shape.dim)


def check_export(
    definition: ExportDefinition,
    model: nn.Module,
    sample: Tensor,
    output_path: Path,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, object]:
    """Run ONNX checker and compare every output against ONNX Runtime."""

    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("--check requires both onnx and onnxruntime") from exc

    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph, full_check=True)

    operator_counts = Counter(node.op_type for node in graph.graph.node)
    unexpected_operators = set(operator_counts) - ALLOWED_ONNX_OPERATORS
    if unexpected_operators:
        raise AssertionError(
            f"ONNX graph contains operators outside the Bayes-e candidate policy: "
            f"{sorted(unexpected_operators)}"
        )
    for node in graph.graph.node:
        if node.op_type != "Resize":
            continue
        attributes = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        if attributes.get("mode", b"nearest") != b"nearest":
            raise AssertionError("ONNX Resize must use nearest-neighbor mode")

    graph_inputs = {value.name: _onnx_shape(value) for value in graph.graph.input}
    graph_outputs = {value.name: _onnx_shape(value) for value in graph.graph.output}
    expected_inputs = {definition.input_name: definition.input_shape}
    expected_outputs = dict(
        zip(definition.output_names, definition.output_shapes, strict=True)
    )
    if graph_inputs != expected_inputs:
        raise AssertionError(f"ONNX input contract mismatch: {graph_inputs}")
    if graph_outputs != expected_outputs:
        raise AssertionError(f"ONNX output contract mismatch: {graph_outputs}")

    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    with torch.inference_mode():
        torch_outputs = _as_output_tuple(model(sample))
    ort_outputs = session.run(
        list(definition.output_names),
        {definition.input_name: sample.numpy()},
    )

    output_reports: list[dict[str, object]] = []
    for name, torch_output, ort_output in zip(
        definition.output_names,
        torch_outputs,
        ort_outputs,
        strict=True,
    ):
        expected = torch_output.detach().cpu().numpy()
        if not np.isfinite(expected).all() or not np.isfinite(ort_output).all():
            raise AssertionError(f"{name} contains a non-finite value")
        np.testing.assert_allclose(
            ort_output,
            expected,
            atol=atol,
            rtol=rtol,
            err_msg=f"PyTorch/ORT mismatch for {name}",
        )
        absolute_error = np.abs(ort_output - expected)
        output_reports.append(
            {
                "name": name,
                "max_abs_error": float(absolute_error.max(initial=0.0)),
                "mean_abs_error": float(absolute_error.mean()),
                "allclose_atol": atol,
                "allclose_rtol": rtol,
            }
        )

    return {
        "onnx_checker": "pass",
        "onnx_operator_policy": "pass",
        "onnx_operator_counts": dict(sorted(operator_counts.items())),
        "onnxruntime_provider": session.get_providers()[0],
        "pytorch_ort_allclose": "pass",
        "outputs": output_reports,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("all", *EXPORT_DEFINITIONS),
        default="all",
        help="model to export; default exports both contracts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for static ONNX artifacts",
    )
    parser.add_argument(
        "--tiny-weights",
        type=Path,
        help="optional TinyOccFlowStudent state_dict/checkpoint",
    )
    parser.add_argument(
        "--cam-weights",
        type=Path,
        help="optional CamSemLite state_dict/checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260728,
        help="deterministic initialization and check-input seed",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="run onnx.checker and PyTorch/ONNX Runtime numerical comparison",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--report-file",
        type=Path,
        help="optional UTF-8 JSON file for the complete export/check receipt",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for deterministic static ONNX export."""

    args = _parse_args()
    selected = (
        tuple(EXPORT_DEFINITIONS)
        if args.model == "all"
        else (args.model,)
    )
    checkpoint_paths = {
        "tiny_occ_flow": args.tiny_weights,
        "cam_sem_lite": args.cam_weights,
    }

    reports: list[dict[str, object]] = []
    for model_name in selected:
        definition = EXPORT_DEFINITIONS[model_name]
        checkpoint_path = checkpoint_paths[model_name]
        output_path = args.output_dir / definition.filename
        model, sample, report = export_model(
            definition,
            output_path,
            checkpoint_path=checkpoint_path,
            seed=args.seed,
        )
        if args.check:
            report["validation"] = check_export(
                definition,
                model,
                sample,
                output_path,
                atol=args.atol,
                rtol=args.rtol,
            )
        reports.append(report)

    payload = {"exports": reports}
    if args.report_file is not None:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if any(not report["trained_checkpoint_supplied"] for report in reports):
        print(
            "WARNING: untrained deterministic weights are for graph validation only.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
