#!/usr/bin/env python3
"""Compile the accepted ICMat ONNX bank for Bayes-e without contacting X5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, version_converter


ROOT = Path(__file__).resolve().parents[3]
FINAL_ROOT = ROOT / "icmat_foundry" / "finals_50model"
BPU_ROOT = FINAL_ROOT / "bpu"
COMPILED_ROOT = BPU_ROOT / "compiled"
EVIDENCE_ROOT = BPU_ROOT / "evidence"
TOOLCHAIN_IMAGE = "openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310"
SUCCESS_STATUS = "BPU_COMPILED_BOARD_PENDING"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True)
class Target:
    model_id: str
    onnx_path: str
    fixture_path: str | None = None
    fixture_key: str | None = None
    calibration_path: str | None = None
    evidence_class: str = "MODEL_SPECIFIC"
    quality_status: str = "NOT_REEVALUATED_BY_BPU_COMPILER"
    quality_state: str | None = None
    source_receipt: str | None = None


TARGETS = (
    Target(
        "F-XRD-01",
        "icmat_foundry/finals_50model/artifacts/xrd_bank/F-XRD-01/model.onnx",
        "icmat_foundry/finals_50model/artifacts/xrd_bank/F-XRD-01/fixed_input.npz",
        "xrd_degraded_fp32",
        evidence_class="PUBLIC_COMPUTATIONAL_XRD",
    ),
    Target(
        "F-XRD-02",
        "icmat_foundry/finals_50model/artifacts/xrd_bank/F-XRD-02/model.onnx",
        "icmat_foundry/finals_50model/artifacts/xrd_bank/F-XRD-02/fixed_input.npz",
        "xrd_profile_fp32",
        evidence_class="PUBLIC_COMPUTATIONAL_XRD",
    ),
    Target(
        "F-PKG-01",
        "icmat_foundry/finals_50model/artifacts/package_bank/f_pkg_01/model_fp32.onnx",
        "icmat_foundry/finals_50model/artifacts/package_bank/f_pkg_01/input_fixture.npy",
        evidence_class="SIM_ONLY",
    ),
    Target(
        "F-PKG-02",
        "icmat_foundry/finals_50model/artifacts/package_bank/f_pkg_02/model_fp32.onnx",
        "icmat_foundry/finals_50model/artifacts/package_bank/f_pkg_02/input_fixture.npy",
        evidence_class="SIM_ONLY",
    ),
    Target(
        "F-PKG-03",
        "icmat_foundry/finals_50model/artifacts/package_bank/f_pkg_03/model_fp32.onnx",
        "icmat_foundry/finals_50model/artifacts/package_bank/f_pkg_03/input_fixture.npy",
        evidence_class="SIM_ONLY",
    ),
    Target(
        "F-MAT-02",
        "icmat_foundry/finals_50model/artifacts/material_bank/F-MAT-02/model.onnx",
        calibration_path="icmat_foundry/finals_50model/artifacts/material_bank/F-MAT-02/calibration_inputs.npy",
        evidence_class="PUBLIC_COMPUTATIONAL_DFT",
    ),
    Target(
        "F-MAT-03",
        "icmat_foundry/finals_50model/artifacts/material_bank/F-MAT-03/model.onnx",
        calibration_path="icmat_foundry/finals_50model/artifacts/material_bank/F-MAT-03/calibration_inputs.npy",
        evidence_class="PUBLIC_COMPUTATIONAL_DFT",
    ),
    Target(
        "F-MAT-04",
        "icmat_foundry/finals_50model/artifacts/material_bank/F-MAT-04/model.onnx",
        calibration_path="icmat_foundry/finals_50model/artifacts/material_bank/F-MAT-04/calibration_inputs.npy",
        evidence_class="PUBLIC_COMPUTATIONAL_DFT",
    ),
    Target(
        "F-MAT-05",
        "icmat_foundry/finals_50model/artifacts/material_bank/F-MAT-05/model.onnx",
        calibration_path="icmat_foundry/finals_50model/artifacts/material_bank/F-MAT-05/calibration_inputs.npy",
        evidence_class="PUBLIC_COMPUTATIONAL_DFT",
    ),
    Target(
        "F-KNW-03",
        "icmat_foundry/finals_50model/artifacts/knowledge_bank/F-KNW-03/chem_entity_mlp.static_1x73.onnx",
        "icmat_foundry/finals_50model/artifacts/knowledge_bank/F-KNW-03/fixed_input.v1.npz",
        "features_fp32",
        evidence_class="LICENSED_LITERATURE_DERIVED_FEATURES",
    ),
    Target(
        "F-SEM-01",
        "icmat_foundry/finals_50model/artifacts/sem_bank/F-SEM-01/model_static_opset11_ir7.onnx",
        "icmat_foundry/finals_50model/artifacts/sem_bank/F-SEM-01/fixed_ort_fixture.npz",
        "input_fp32",
        evidence_class="PUBLIC_REAL_SEM_CC_BY_4_0",
        quality_status="QUALITY_ACCEPTED_PC_CANDIDATE",
        source_receipt="icmat_foundry/finals_50model/evidence/sem_bank/F-SEM-01/training_receipt.v1.json",
    ),
    Target(
        "F-SEM-02",
        "icmat_foundry/finals_50model/artifacts/sem_bank/F-SEM-02/model_static_opset11_ir7.onnx",
        "icmat_foundry/finals_50model/artifacts/sem_bank/F-SEM-02/fixed_ort_fixture.npz",
        "input_fp32",
        evidence_class="PUBLIC_REAL_SEM_EXPERT_MASKS_CC_BY_4_0",
        quality_status="QUALITY_ACCEPTED_PC_CANDIDATE",
        source_receipt="icmat_foundry/finals_50model/evidence/sem_bank/F-SEM-02/training_receipt.v1.json",
    ),
    Target(
        "F-SEM-03",
        "icmat_foundry/finals_50model/artifacts/sem_bank/F-SEM-03/model_static_opset11_ir7.onnx",
        "icmat_foundry/finals_50model/artifacts/sem_bank/F-SEM-03/fixed_ort_fixture.npz",
        "input_fp32",
        evidence_class="NIST_MASK_DERIVED_SIM_ONLY",
        quality_status="QUALITY_ACCEPTED_PC_CANDIDATE_SIM_ONLY",
        source_receipt="icmat_foundry/finals_50model/evidence/sem_bank/F-SEM-03/training_receipt.v1.json",
    ),
    Target(
        "F-SEM-04",
        "icmat_foundry/finals_50model/artifacts/sem_bank/F-SEM-04/model_static_opset11_ir7.onnx",
        "icmat_foundry/finals_50model/artifacts/sem_bank/F-SEM-04/fixed_ort_fixture.npz",
        "input_fp32",
        evidence_class="PHYSICS_SURROGATE_SIM_ONLY",
        quality_status="QUALITY_ACCEPTED_PC_CANDIDATE_SIM_ONLY",
        source_receipt="icmat_foundry/finals_50model/evidence/sem_bank/F-SEM-04/training_receipt.v1.json",
    ),
    Target(
        "F-PROC-01",
        "icmat_foundry/finals_50model/artifacts/process_bank/F-PROC-01/model.onnx",
        "icmat_foundry/finals_50model/artifacts/process_bank/F-PROC-01/ort_sample_input.npy",
        evidence_class="PUBLIC_PROCESS_DATA_DERIVED",
        quality_status="QUALITY_ACCEPTED_PC_CANDIDATE",
        source_receipt="icmat_foundry/finals_50model/evidence/process_bank/F-PROC-01/receipt.json",
    ),
    Target(
        "F-PROC-02",
        "icmat_foundry/finals_50model/artifacts/process_bank/F-PROC-02/model.onnx",
        "icmat_foundry/finals_50model/artifacts/process_bank/F-PROC-02/ort_sample_input.npy",
        evidence_class="PUBLIC_PROCESS_DATA_DERIVED",
        quality_status="QUALITY_ACCEPTED_PC_CANDIDATE",
        source_receipt="icmat_foundry/finals_50model/evidence/process_bank/F-PROC-02/receipt.json",
    ),
    Target(
        "F-PROC-03",
        "icmat_foundry/finals_50model/artifacts/process_bank/F-PROC-03/model.onnx",
        "icmat_foundry/finals_50model/artifacts/process_bank/F-PROC-03/ort_sample_input.npy",
        evidence_class="PUBLIC_PROCESS_DATA_DERIVED_NEGATIVE_EVIDENCE",
        quality_status="QUALITY_LIMITED",
        quality_state="QUALITY_LIMITED_NOT_PROMOTED",
        source_receipt="icmat_foundry/finals_50model/evidence/process_bank/F-PROC-03/receipt.json",
    ),
    Target(
        "F-PROC-04",
        "CIMC_candidates/ICMat_PhosFab_Foundry_R1_20260731/artifacts/thermal_sim/N02/candidate/model.onnx",
        "icmat_foundry/finals_50model/fixtures/F-PROC-04/input.npz",
        "mlx_sequence_ambient_emissivity",
        evidence_class="SIM_ONLY",
    ),
    Target(
        "F-PROC-05",
        "CIMC_candidates/ICMat_PhosFab_Foundry_R1_20260731/artifacts/thermal_sim/N03/candidate/model.onnx",
        "icmat_foundry/finals_50model/fixtures/F-PROC-05/input.npz",
        "synchronized_sensor_features",
        evidence_class="SIM_ONLY",
    ),
    Target(
        "F-PROC-06",
        "CIMC_candidates/ICMat_PhosFab_Foundry_R1_20260731/artifacts/process_sim/N04/candidate/model.onnx",
        "icmat_foundry/finals_50model/fixtures/F-PROC-06/input.npz",
        "ptc_power_chain_features",
        evidence_class="SIM_ONLY",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def model_contract(model: onnx.ModelProto) -> tuple[str, list[int | str]]:
    if len(model.graph.input) != 1:
        raise ValueError(f"exactly one model input is required, got {len(model.graph.input)}")
    value = model.graph.input[0]
    if value.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError("only float32 model input is supported")
    shape: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        else:
            shape.append(dim.dim_param or "?")
    return value.name, shape


def opsets(model: onnx.ModelProto) -> dict[str, int]:
    return {item.domain or "ai.onnx": int(item.version) for item in model.opset_import}


def reexport_process_compatibility(target: Target, output: Path) -> None:
    """Re-export the three accepted legacy candidates without changing their sources."""
    import joblib
    import torch
    from torch import nn

    candidate_root = ROOT / "CIMC_candidates" / "ICMat_PhosFab_Foundry_R1_20260731"
    if str(candidate_root) not in sys.path:
        sys.path.insert(0, str(candidate_root))
    fixture = load_fixture(ROOT / str(target.fixture_path), target.fixture_key)

    if target.model_id in {"F-PROC-04", "F-PROC-05"}:
        from phosfab.models.thermal_sim.networks import (
            TCIRFusionNet,
            ThermalFieldHotspotNet,
        )

        source_id = "N02" if target.model_id == "F-PROC-04" else "N03"
        checkpoint_path = (
            candidate_root / "artifacts" / "thermal_sim" / source_id / "candidate" / "checkpoint.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if source_id == "N02":
            model: nn.Module = ThermalFieldHotspotNet()
            input_tensor = torch.from_numpy(np.asarray(fixture[:1], dtype=np.float32))
            output_names = ["temperature_field_fp32", "hotspot_logit_fp32"]
        else:
            core = TCIRFusionNet(checkpoint["feature_mean"], checkpoint["feature_std"])

            class NchwWrapper(nn.Module):
                def __init__(self, wrapped: nn.Module) -> None:
                    super().__init__()
                    self.wrapped = wrapped

                def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    return self.wrapped(values.flatten(1))

            model = NchwWrapper(core)
            input_tensor = torch.from_numpy(
                np.asarray(fixture[:1], dtype=np.float32).reshape(1, 1, 1, 24)
            )
            output_names = ["fused_temperature_fp32", "sigma_fp32"]
        state_target = model.wrapped if source_id == "N03" else model
        state_target.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        torch.onnx.export(
            model,
            input_tensor,
            str(output),
            input_names=[target.fixture_key or "input_fp32"],
            output_names=output_names,
            opset_version=11,
            do_constant_folding=True,
            dynamo=False,
        )
        return

    if target.model_id == "F-PROC-06":
        pipeline = joblib.load(
            candidate_root / "artifacts" / "process_sim" / "N04" / "candidate" / "checkpoint.joblib"
        )
        scaler = pipeline.named_steps["scale"]
        regressor = pipeline.named_steps["regressor"]

        class PtcWrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "feature_mean", torch.tensor(scaler.mean_, dtype=torch.float32).reshape(1, -1)
                )
                self.register_buffer(
                    "feature_scale", torch.tensor(scaler.scale_, dtype=torch.float32).reshape(1, -1)
                )
                layers: list[nn.Module] = []
                for index, (weight, bias) in enumerate(
                    zip(regressor.coefs_, regressor.intercepts_, strict=True)
                ):
                    linear = nn.Linear(weight.shape[0], weight.shape[1])
                    linear.weight.data.copy_(torch.tensor(weight.T, dtype=torch.float32))
                    linear.bias.data.copy_(torch.tensor(bias, dtype=torch.float32))
                    layers.append(linear)
                    if index < len(regressor.coefs_) - 1:
                        layers.append(nn.Tanh())
                self.network = nn.Sequential(*layers)

            def forward(self, values: torch.Tensor) -> torch.Tensor:
                flat = values.flatten(1)
                return self.network((flat - self.feature_mean) / self.feature_scale)

        model = PtcWrapper().eval()
        input_tensor = torch.from_numpy(
            np.asarray(fixture[:1], dtype=np.float32).reshape(1, 1, 1, 10)
        )
        torch.onnx.export(
            model,
            input_tensor,
            str(output),
            input_names=[target.fixture_key or "input_fp32"],
            output_names=["power_delta_fp32"],
            opset_version=11,
            do_constant_folding=True,
            dynamo=False,
        )
        return
    raise ValueError(f"no checkpoint re-export contract exists for {target.model_id}")


def make_compatible_copy(target: Target, source: Path, output: Path) -> dict[str, Any]:
    model = onnx.load(str(source), load_external_data=True)
    input_name, original_shape = model_contract(model)
    original_opsets = opsets(model)
    fixes: list[str] = []

    main_opset = original_opsets.get("ai.onnx", 0)
    if main_opset > 11:
        try:
            model = version_converter.convert_version(model, 11)
            fixes.append(f"ai.onnx_opset_{main_opset}_to_11")
        except Exception:
            if not target.model_id.startswith("F-PROC-"):
                raise
            reexport_process_compatibility(target, output)
            staged = onnx.load(str(output), load_external_data=False)
            staged_name, staged_shape = model_contract(staged)
            staged_opsets = opsets(staged)
            return {
                "source_input_name": input_name,
                "source_input_shape": original_shape,
                "source_opsets": original_opsets,
                "staged_input_name": staged_name,
                "staged_input_shape": staged_shape,
                "staged_opsets": staged_opsets,
                "staged_ir_version": int(staged.ir_version),
                "compatibility_fixes": [
                    "checkpoint_reexport_static_opset11",
                    "source_onnx_version_converter_unavailable",
                ],
            }

    input_name, shape = model_contract(model)
    value = model.graph.input[0]
    if not isinstance(shape[0], int) or shape[0] != 1:
        first_dim = value.type.tensor_type.shape.dim[0]
        first_dim.ClearField("dim_param")
        first_dim.dim_value = 1
        shape[0] = 1
        fixes.append("fixed_batch_to_1")
    for index, dim in enumerate(shape[1:], start=1):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"unresolved non-batch input dimension at index {index}: {dim}")

    if len(shape) == 2:
        width = int(shape[1])
        flat_name = f"{input_name}__flat_nc"
        for node in model.graph.node:
            for index, node_input in enumerate(node.input):
                if node_input == input_name:
                    node.input[index] = flat_name
        flatten = helper.make_node(
            "Flatten", [input_name], [flat_name], name="bpu_compat_flatten_nchw_to_nc", axis=1
        )
        model.graph.node.insert(0, flatten)
        tensor_shape = value.type.tensor_type.shape
        del tensor_shape.dim[:]
        for dim_value in (1, 1, 1, width):
            tensor_shape.dim.add().dim_value = dim_value
        fixes.append("wrapped_rank2_input_as_1x1x1xN")
    elif len(shape) != 4:
        raise ValueError(f"unsupported input rank {len(shape)}; expected 2 or 4")

    if model.ir_version > 7:
        fixes.append(f"ir_{model.ir_version}_to_7")
        model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, str(output))
    staged = onnx.load(str(output), load_external_data=False)
    staged_name, staged_shape = model_contract(staged)
    staged_opsets = opsets(staged)
    if staged_opsets.get("ai.onnx") != 11:
        raise ValueError(f"staged ai.onnx opset is not 11: {staged_opsets}")
    if staged.ir_version > 7:
        raise ValueError(f"staged IR version is not compatible: {staged.ir_version}")
    if staged_shape[0] != 1 or len(staged_shape) != 4:
        raise ValueError(f"staged shape is not static NCHW: {staged_shape}")
    return {
        "source_input_name": input_name,
        "source_input_shape": original_shape,
        "source_opsets": original_opsets,
        "staged_input_name": staged_name,
        "staged_input_shape": staged_shape,
        "staged_opsets": staged_opsets,
        "staged_ir_version": int(staged.ir_version),
        "compatibility_fixes": fixes,
    }


def load_fixture(path: Path, key: str | None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    with np.load(path, allow_pickle=False) as payload:
        selected = key or payload.files[0]
        if selected not in payload.files:
            raise ValueError(f"fixture key {selected!r} not found in {path}")
        return np.asarray(payload[selected], dtype=np.float32)


def shape_for_staged(sample: np.ndarray, staged_shape: list[int | str]) -> np.ndarray:
    expected = tuple(int(item) for item in staged_shape)
    array = np.asarray(sample, dtype=np.float32)
    if array.shape == expected:
        return np.ascontiguousarray(array)
    if array.ndim >= 1 and array.shape[0] > 1:
        array = array[:1]
    if array.size == int(np.prod(expected)):
        return np.ascontiguousarray(array.reshape(expected))
    raise ValueError(f"fixture shape {sample.shape} cannot satisfy staged shape {expected}")


def perturb_fixture(base: np.ndarray, seed_text: str, rows: int = 32) -> np.ndarray:
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    sample_shape = base.shape[1:]
    anchor = np.asarray(base[0], dtype=np.float32)
    scale = np.maximum(np.abs(anchor), np.float32(0.05))
    collection = [anchor.copy()]
    for index in range(1, rows):
        amplitude = np.float32(0.001 + 0.0005 * (index % 7))
        noise = rng.standard_normal(sample_shape, dtype=np.float32) * scale * amplitude
        collection.append(np.asarray(anchor + noise, dtype=np.float32))
    result = np.ascontiguousarray(np.stack(collection, axis=0), dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("deterministic calibration perturbation produced non-finite values")
    return result


def calibration_inputs(
    target: Target, staged_shape: list[int | str], source_sha: str
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    expected = tuple(int(item) for item in staged_shape)
    if target.calibration_path:
        source = ROOT / target.calibration_path
        if source.exists():
            loaded = np.asarray(np.load(source, allow_pickle=False), dtype=np.float32)
            if loaded.ndim == len(expected) and loaded.shape[1:] == expected[1:]:
                rows = loaded
            elif loaded.ndim == len(expected) + 1 and loaded.shape[1:] == expected:
                rows = loaded.reshape((-1, *expected[1:]))
            else:
                raise ValueError(
                    f"real calibration shape {loaded.shape} does not match {expected}"
                )
            if rows.shape[0] < 2 or not np.all(np.isfinite(rows)):
                raise ValueError("real calibration inputs are insufficient or non-finite")
            first = np.ascontiguousarray(rows[:1].reshape(expected), dtype=np.float32)
            return (
                np.ascontiguousarray(rows, dtype=np.float32),
                {
                    "kind": "REAL_MODEL_TUNE_FEATURES",
                    "path": relative(source),
                    "sha256": sha256_file(source),
                    "rows": int(rows.shape[0]),
                    "deterministic_perturbation": False,
                },
                first,
            )
    if not target.fixture_path:
        raise ValueError("neither real calibration inputs nor a fixed fixture is available")
    fixture = ROOT / target.fixture_path
    base = shape_for_staged(load_fixture(fixture, target.fixture_key), staged_shape)
    rows = perturb_fixture(base, f"{target.model_id}:{source_sha}:calibration-v1")
    return (
        rows,
        {
            "kind": "DETERMINISTIC_FINITE_PERTURBATION_FROM_FIXED_FIXTURE",
            "path": relative(fixture),
            "sha256": sha256_file(fixture),
            "rows": int(rows.shape[0]),
            "seed_contract": f"sha256({target.model_id}:source_sha256:calibration-v1)",
            "deterministic_perturbation": True,
            "claim_boundary": "Calibration coverage is synthetic around one fixed accepted fixture.",
        },
        base,
    )


def verify_staged_parity(
    source: Path, staged: Path, staged_input: np.ndarray
) -> dict[str, Any]:
    source_session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    staged_session = ort.InferenceSession(str(staged), providers=["CPUExecutionProvider"])
    source_meta = source_session.get_inputs()[0]
    source_shape = source_meta.shape
    source_sample = staged_input
    if len(source_shape) == 2 and staged_input.ndim == 4:
        source_sample = staged_input.reshape((1, staged_input.shape[-1]))
    elif len(source_shape) == 4:
        source_sample = staged_input
    outputs_a = source_session.run(None, {source_meta.name: source_sample.astype(np.float32)})
    staged_name = staged_session.get_inputs()[0].name
    outputs_b = staged_session.run(None, {staged_name: staged_input.astype(np.float32)})
    if len(outputs_a) != len(outputs_b):
        raise ValueError("compatibility copy changed output count")
    max_diff = 0.0
    shapes: list[list[int]] = []
    for left, right in zip(outputs_a, outputs_b, strict=True):
        left_array = np.asarray(left, dtype=np.float32)
        right_array = np.asarray(right, dtype=np.float32)
        if left_array.shape != right_array.shape or not np.all(np.isfinite(right_array)):
            raise ValueError("compatibility copy changed output shape or finiteness")
        max_diff = max(max_diff, float(np.max(np.abs(left_array - right_array))))
        shapes.append(list(right_array.shape))
    if max_diff > 1e-4:
        raise ValueError(f"compatibility parity exceeds tolerance: {max_diff}")
    return {"output_shapes": shapes, "max_abs_diff": max_diff, "tolerance": 1e-4}


def mapper_yaml(input_name: str, input_shape: list[int | str], prefix: str) -> str:
    shape_text = "x".join(str(item) for item in input_shape)
    return f"""model_parameters:
  onnx_model: './model_staged.onnx'
  march: 'bayes-e'
  output_model_file_prefix: '{prefix}'
  working_dir: './model_output'
  layer_out_dump: False
  log_level: 'debug'

input_parameters:
  input_name: '{input_name}'
  input_shape: '{shape_text}'
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
"""


def run_mapper(model_dir: Path, command: str, log_name: str) -> dict[str, Any]:
    relative_dir = model_dir.resolve().relative_to(BPU_ROOT.resolve()).as_posix()
    invocation = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{BPU_ROOT.resolve()}:/work",
        "-w",
        f"/work/{relative_dir}",
        TOOLCHAIN_IMAGE,
        "bash",
        "-lc",
        command,
    ]
    completed = subprocess.run(
        invocation,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log = (completed.stdout or "") + (completed.stderr or "")
    log_path = model_dir / log_name
    log_path.write_text(log, encoding="utf-8", errors="replace", newline="\n")
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "log_path": relative(log_path),
        "log_sha256": sha256_file(log_path),
    }


def node_evidence(log_path: Path) -> dict[str, Any]:
    clean = ANSI_RE.sub("", log_path.read_text(encoding="utf-8", errors="replace"))
    bpu: list[str] = []
    cpu: list[str] = []
    for line in clean.splitlines():
        match = re.match(r"^\s*(\S.*?)\s{2,}(BPU|CPU)\s+", line)
        if not match:
            continue
        node, placement = match.groups()
        if node.lower().startswith("node"):
            continue
        destination = bpu if placement == "BPU" else cpu
        if node not in destination:
            destination.append(node)
    return {
        "bpu_node_count": len(bpu),
        "cpu_node_count": len(cpu),
        "bpu_nodes": bpu,
        "cpu_nodes": cpu,
        "runtime_bin_success_marker": "Convert to runtime bin file successfully!" in clean,
    }


def compile_target(target: Target, force: bool) -> dict[str, Any]:
    source = ROOT / target.onnx_path
    if not source.exists():
        return {"model_id": target.model_id, "result": "SKIPPED_SOURCE_MISSING"}
    model_dir = COMPILED_ROOT / target.model_id
    evidence_path = EVIDENCE_ROOT / f"{target.model_id}.compile.v1.json"
    if model_dir.exists() and not force:
        existing = model_dir / "compile_receipt.v1.json"
        if existing.exists():
            return json.loads(existing.read_text(encoding="utf-8"))
        raise FileExistsError(f"partial output exists; rerun with --force: {model_dir}")
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    source_sha = sha256_file(source)
    staged = model_dir / "model_staged.onnx"
    base_receipt: dict[str, Any] = {
        "schema": "x5_icmat_foundry.bayes_e_compile.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": target.model_id,
        "result": "IN_PROGRESS",
        "source": {
            "path": relative(source),
            "sha256": source_sha,
            "evidence_class": target.evidence_class,
        },
        "quality_status": target.quality_status,
        "quality_state": target.quality_state or target.quality_status,
        "quality_note": "Compilation success does not promote model quality.",
        "quality_boundary": {
            "bpu_compilation_promotes_model_quality": False,
            "source_receipt": (
                {
                    "path": target.source_receipt,
                    "sha256": sha256_file(ROOT / target.source_receipt),
                }
                if target.source_receipt
                else None
            ),
        },
        "toolchain": {
            "image": TOOLCHAIN_IMAGE,
            "march": "bayes-e",
            "calibration_type": "max",
            "per_channel": True,
        },
        "execution_boundary": {
            "x5_contacted": False,
            "board_execution": False,
            "production_modified": False,
            "registry_modified": False,
            "overlay_modified": False,
            "agents_modified": False,
        },
    }
    try:
        compatibility = make_compatible_copy(target, source, staged)
        staged_shape = compatibility["staged_input_shape"]
        rows, calibration, first_input = calibration_inputs(
            target, staged_shape, source_sha
        )
        parity = verify_staged_parity(source, staged, first_input)
        np.save(model_dir / "calibration_inputs.npy", rows, allow_pickle=False)
        calibration_dir = model_dir / "calibration_data"
        calibration_dir.mkdir(exist_ok=True)
        records = []
        for index, row in enumerate(rows):
            path = calibration_dir / f"calib_{index:03d}.bin"
            payload = np.ascontiguousarray(row, dtype="<f4").tobytes(order="C")
            path.write_bytes(payload)
            records.append(
                {"index": index, "bytes": len(payload), "sha256": sha256_file(path)}
            )
        prefix = f"icmat_{target.model_id.lower().replace('-', '_')}_int8"
        config = model_dir / "config_bpu.yaml"
        config.write_text(
            mapper_yaml(compatibility["staged_input_name"], staged_shape, prefix),
            encoding="utf-8",
            newline="\n",
        )
        shape_text = "x".join(str(item) for item in staged_shape)
        checker = run_mapper(
            model_dir,
            (
                "hb_mapper checker --model-type onnx --model model_staged.onnx "
                f"--march bayes-e --input-shape {compatibility['staged_input_name']} {shape_text}"
            ),
            "hb_mapper_checker.log",
        )
        if checker["returncode"] != 0:
            raise RuntimeError(f"hb_mapper checker failed with rc={checker['returncode']}")
        makertbin = run_mapper(
            model_dir,
            "hb_mapper makertbin --model-type onnx --config config_bpu.yaml",
            "hb_mapper_makertbin.log",
        )
        if makertbin["returncode"] != 0:
            raise RuntimeError(f"hb_mapper makertbin failed with rc={makertbin['returncode']}")
        bins = sorted((model_dir / "model_output").glob("*.bin"))
        if len(bins) != 1 or bins[0].stat().st_size <= 0:
            raise RuntimeError(f"expected one non-empty runtime bin, found {len(bins)}")
        nodes = node_evidence(model_dir / "hb_mapper_makertbin.log")
        if not nodes["runtime_bin_success_marker"]:
            raise RuntimeError("mapper success marker is absent")
        if nodes["bpu_node_count"] < 1:
            raise RuntimeError("no core operator was reported on BPU")
        binary = bins[0]
        receipt = {
            **base_receipt,
            "result": "PASS",
            "status": SUCCESS_STATUS,
            "compatibility": compatibility,
            "staged_model": {
                "path": relative(staged),
                "sha256": sha256_file(staged),
                "onnx_checker": "PASS",
                "source_parity": parity,
            },
            "calibration": {
                **calibration,
                "tensor_shape": list(rows.shape),
                "dtype": "float32_little_endian",
                "materialized_path": relative(model_dir / "calibration_inputs.npy"),
                "materialized_sha256": sha256_file(model_dir / "calibration_inputs.npy"),
                "records": records,
            },
            "mapper": {
                "config_path": relative(config),
                "config_sha256": sha256_file(config),
                "checker": checker,
                "makertbin": makertbin,
            },
            "runtime_binary": {
                "path": relative(binary),
                "bytes": binary.stat().st_size,
                "sha256": sha256_file(binary),
            },
            "placement": nodes,
            "claim_boundary": (
                "Compiled with the pinned PC OpenExplorer Bayes-e toolchain. "
                "X5 board execution, latency, memory, and production deployment remain pending."
            ),
        }
    except Exception as error:
        receipt = {
            **base_receipt,
            "result": "FAILED_NOT_COMPILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "claim_boundary": "No successful BPU compilation claim is authorized.",
        }
        for name in ("hb_mapper_checker.log", "hb_mapper_makertbin.log"):
            path = model_dir / name
            if path.exists():
                receipt.setdefault("available_logs", []).append(
                    {"path": relative(path), "sha256": sha256_file(path)}
                )
    write_json(model_dir / "compile_receipt.v1.json", receipt)
    write_json(evidence_path, receipt)
    return receipt


def image_record() -> dict[str, Any]:
    completed = subprocess.run(
        ["docker", "image", "inspect", TOOLCHAIN_IMAGE, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"required Docker image is unavailable: {completed.stderr.strip()}")
    return {"name": TOOLCHAIN_IMAGE, "id": completed.stdout.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="*", help="Optional model IDs; default compiles the complete target bank."
    )
    parser.add_argument("--force", action="store_true", help="Replace prior per-model staging output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = list(TARGETS)
    if args.models:
        wanted = set(args.models)
        selected = [target for target in TARGETS if target.model_id in wanted]
        missing = sorted(wanted - {target.model_id for target in selected})
        if missing:
            raise ValueError(f"unknown model IDs: {missing}")
    COMPILED_ROOT.mkdir(parents=True, exist_ok=True)
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    image = image_record()
    results = []
    for target in selected:
        print(f"[{target.model_id}] compiling", flush=True)
        result = compile_target(target, force=args.force)
        results.append(result)
        print(f"[{target.model_id}] {result['result']}", flush=True)
    passed = [item for item in results if item.get("status") == SUCCESS_STATUS]
    failed = [item for item in results if item.get("status") != SUCCESS_STATUS]
    verified_models: list[dict[str, Any]] = []
    for item in passed:
        binary = ROOT / item["runtime_binary"]["path"]
        recomputed_sha = sha256_file(binary)
        if binary.stat().st_size <= 0 or recomputed_sha != item["runtime_binary"]["sha256"]:
            raise RuntimeError(f"runtime binary independent verification failed: {item['model_id']}")
        verified_models.append(
            {
                "model_id": item["model_id"],
                "bin_path": item["runtime_binary"]["path"],
                "bin_bytes": binary.stat().st_size,
                "bin_sha256": recomputed_sha,
                "bpu_node_count": item["placement"]["bpu_node_count"],
                "cpu_node_count": item["placement"]["cpu_node_count"],
                "source_parity_max_abs_diff": item["staged_model"]["source_parity"][
                    "max_abs_diff"
                ],
                "calibration_kind": item["calibration"]["kind"],
                "quality_status": item.get(
                    "quality_status", "NOT_REEVALUATED_BY_BPU_COMPILER"
                ),
                "quality_state": item.get(
                    "quality_state",
                    item.get("quality_status", "NOT_REEVALUATED_BY_BPU_COMPILER"),
                ),
                "quality_source_receipt": item.get("quality_boundary", {}).get(
                    "source_receipt"
                ),
                "quality_note": item.get(
                    "quality_note", "Compilation success does not promote model quality."
                ),
                "compatibility_fixes": item["compatibility"]["compatibility_fixes"],
            }
        )
    summary = {
        "schema": "x5_icmat_foundry.bayes_e_compile_bank.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result": "PASS" if not failed else "PARTIAL",
        "status": SUCCESS_STATUS if not failed else None,
        "toolchain": image,
        "selected_count": len(selected),
        "compiled_count": len(passed),
        "failed_count": len(failed),
        "compiled_model_ids": [item["model_id"] for item in passed],
        "verified_models": verified_models,
        "placement_totals": {
            "bpu_node_count": sum(item["bpu_node_count"] for item in verified_models),
            "cpu_node_count": sum(item["cpu_node_count"] for item in verified_models),
        },
        "independent_binary_verification": {
            "nonempty_bins": len(verified_models),
            "sha256_matches_receipt": len(verified_models),
        },
        "failed_models": [
            {"model_id": item["model_id"], "result": item["result"], "error": item.get("error")}
            for item in failed
        ],
        "receipts": [
            {
                "model_id": item["model_id"],
                "path": relative(EVIDENCE_ROOT / f"{item['model_id']}.compile.v1.json"),
                "sha256": sha256_file(EVIDENCE_ROOT / f"{item['model_id']}.compile.v1.json"),
            }
            for item in results
        ],
        "execution_boundary": {
            "x5_contacted": False,
            "board_execution": False,
            "production_modified": False,
            "registry_overlay_agents_modified": False,
        },
        "compiler": {
            "path": relative(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    write_json(EVIDENCE_ROOT / "compile_bank_summary.v1.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    return 0 if not failed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
