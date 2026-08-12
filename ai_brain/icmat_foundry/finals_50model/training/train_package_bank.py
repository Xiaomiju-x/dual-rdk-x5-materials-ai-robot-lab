"""Train four independent packaging surrogate models from explicit physics proxies.

The referenced NIST records are not present as local data files. This script therefore
does not synthesize fake measurements: every generated sample and artifact is labelled
SIM_ONLY and remains a PC candidate until separate BPU/X5 validation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

sys.dont_write_bytecode = True

import numpy as np
import onnx
import onnxruntime as ort
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = ROOT / "icmat_foundry/finals_50model"
ARTIFACT_ROOT = CANDIDATE / "artifacts/package_bank"
EVIDENCE_ROOT = CANDIDATE / "evidence/package_bank"
SEED = 20260801
SAMPLE_COUNT = 18_000
HOLDOUT_LOW = 0.44
HOLDOUT_HIGH = 0.56


@dataclass(frozen=True)
class Parameter:
    name: str
    unit: str
    low: float
    high: float


@dataclass(frozen=True)
class Output:
    name: str
    unit: str
    transform: str = "log1p"


@dataclass(frozen=True)
class Task:
    inventory_id: str
    model_id: str
    backend: str
    nist_record: str
    nist_url: str
    modality_scope: str
    equation_scope: str
    parameters: tuple[Parameter, ...]
    outputs: tuple[Output, ...]
    holdout_parameter: str
    simulator: Callable[[np.ndarray], np.ndarray]


class CompactSurrogate(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, backend: str) -> None:
        super().__init__()
        hidden = (128, 96, 48) if backend == "BPU" else (160, 96, 48)
        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], hidden[2]),
            nn.ReLU(),
            nn.Linear(hidden[2], output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value.reshape(-1, self.input_dim))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_hash_sidecar(path: Path) -> str:
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def cure_kinetics(inputs: np.ndarray) -> np.ndarray:
    t_start, t_end, beta, hold_s, log_a1, ea1, log_a2, ea2, m_exp, n_exp = inputs.T
    ramp_s = (t_end - t_start) * 60.0 / beta
    total_s = ramp_s + hold_s
    alpha = np.full(len(inputs), 1.0e-5, dtype=np.float64)
    peak_rate = np.zeros(len(inputs), dtype=np.float64)
    gas_constant = 8.314462618
    steps = 128
    dt = total_s / steps
    for step in range(steps):
        elapsed = (step + 0.5) * dt
        temperature_c = np.minimum(t_start + beta * elapsed / 60.0, t_end)
        temperature_k = temperature_c + 273.15
        k1 = np.power(10.0, log_a1) * np.exp(-(ea1 * 1000.0) / (gas_constant * temperature_k))
        k2 = np.power(10.0, log_a2) * np.exp(-(ea2 * 1000.0) / (gas_constant * temperature_k))
        rate = (k1 + k2 * np.power(alpha, m_exp)) * np.power(1.0 - alpha, n_exp)
        peak_rate = np.maximum(peak_rate, rate)
        alpha = np.clip(alpha + rate * dt, 1.0e-6, 0.999999)
    return np.column_stack((alpha, peak_rate))


def residual_stress(inputs: np.ndarray) -> np.ndarray:
    (
        modulus_gpa,
        poisson,
        resin_cte,
        substrate_cte,
        cure_shrink_pct,
        cure_degree,
        glass_transition_c,
        cure_temperature_c,
        service_temperature_c,
        relaxation,
        restraint,
    ) = inputs.T
    thermal_strain = np.abs(resin_cte - substrate_cte) * 1.0e-6 * np.abs(
        cure_temperature_c - service_temperature_c
    )
    chemical_strain = cure_shrink_pct * 0.01 * np.power(cure_degree, 1.35)
    glass_factor = 0.35 + 0.65 / (
        1.0 + np.exp(-(glass_transition_c - service_temperature_c) / 18.0)
    )
    effective_modulus_mpa = modulus_gpa * 1000.0 * glass_factor / (1.0 - poisson)
    stress_mpa = effective_modulus_mpa * (thermal_strain + chemical_strain)
    stress_mpa *= restraint * np.power(1.0 - relaxation, 1.25)
    return stress_mpa[:, None]


def package_warpage(inputs: np.ndarray) -> np.ndarray:
    (
        die_modulus,
        substrate_modulus,
        interface_modulus,
        die_cte,
        substrate_cte,
        interface_cte,
        die_thickness,
        substrate_thickness,
        interface_thickness,
        package_length,
        aspect_ratio,
        delta_temperature,
        interface_restraint,
    ) = inputs.T
    top_cte = (die_cte * die_thickness + interface_cte * interface_thickness) / (
        die_thickness + interface_thickness
    )
    mismatch = np.abs(substrate_cte - top_cte) * 1.0e-6
    top_rigidity = die_modulus * die_thickness**3 + interface_modulus * interface_thickness**3
    substrate_rigidity = substrate_modulus * substrate_thickness**3
    rigidity_ratio = substrate_rigidity / np.maximum(top_rigidity, 1.0e-9)
    coupling = 6.0 * rigidity_ratio / np.square(1.0 + rigidity_ratio)
    total_thickness = die_thickness + substrate_thickness + interface_thickness
    curvature_per_mm = (
        mismatch * delta_temperature * coupling * interface_restraint / total_thickness
    )
    center_warpage_um = curvature_per_mm * package_length**2 * 1000.0 / 8.0
    peak_warpage_um = center_warpage_um * (1.15 + 0.22 * (aspect_ratio - 1.0))
    interface_stress_mpa = (
        interface_modulus
        * 1000.0
        * mismatch
        * delta_temperature
        * interface_restraint
        / (1.0 + 3.0 * interface_thickness / substrate_thickness)
    )
    return np.column_stack((center_warpage_um, peak_warpage_um, interface_stress_mpa))


def copper_afm_mechanics(inputs: np.ndarray) -> np.ndarray:
    (
        indentation_nm,
        tip_radius_nm,
        modulus_gpa,
        poisson,
        yield_strength_gpa,
        roughness_nm,
        oxide_nm,
        loading_rate_un_s,
        hold_ms,
        multistep_ratio,
    ) = inputs.T
    effective_depth_nm = np.maximum(indentation_nm - 0.20 * roughness_nm - 0.10 * oxide_nm, 0.1)
    depth_m = effective_depth_nm * 1.0e-9
    radius_m = tip_radius_nm * 1.0e-9
    reduced_modulus_pa = modulus_gpa * 1.0e9 / (1.0 - poisson**2)
    hertz_force_n = (4.0 / 3.0) * reduced_modulus_pa * np.sqrt(radius_m) * depth_m**1.5
    contact_radius_m = np.sqrt(np.maximum(radius_m * depth_m, 1.0e-30))
    mean_pressure_pa = hertz_force_n / np.maximum(np.pi * contact_radius_m**2, 1.0e-30)
    plastic_weight = 1.0 / (
        1.0 + np.exp(-(mean_pressure_pa / (yield_strength_gpa * 1.0e9) - 1.0) * 5.0)
    )
    hardness_pa = 2.8 * yield_strength_gpa * 1.0e9
    plastic_force_n = hardness_pa * np.pi * contact_radius_m**2
    force_n = (1.0 - plastic_weight) * hertz_force_n + plastic_weight * plastic_force_n
    protocol_factor = (1.0 + 0.025 * np.log1p(loading_rate_un_s)) * (
        1.0 - 0.035 * np.log1p(hold_ms / 10.0)
    )
    protocol_factor *= 1.0 + 0.03 * multistep_ratio
    roughness_factor = np.exp(-roughness_nm / np.maximum(4.0 * tip_radius_nm, 1.0))
    force_un = force_n * 1.0e6 * protocol_factor * roughness_factor
    stiffness_n_m = 2.0 * reduced_modulus_pa * contact_radius_m
    stiffness_n_m *= (1.0 - 0.35 * plastic_weight) * roughness_factor
    return np.column_stack((force_un, stiffness_n_m))


TASKS = (
    Task(
        inventory_id="F-PKG-01",
        model_id="Encapsulant-CureKinetics-X5",
        backend="BPU",
        nist_record="mds2-3702",
        nist_url="https://data.nist.gov/od/id/mds2-3702",
        modality_scope="DSC/FTIR/Raman liquid encapsulant cure reference",
        equation_scope="Kamal-Sourour dual-Arrhenius cure proxy under ramp-plus-hold schedules",
        parameters=(
            Parameter("start_temperature", "degC", 60.0, 120.0),
            Parameter("end_temperature", "degC", 130.0, 220.0),
            Parameter("heating_rate", "degC/min", 0.5, 20.0),
            Parameter("hold_time", "s", 60.0, 3600.0),
            Parameter("log10_A1", "log10(1/s)", 2.0, 7.0),
            Parameter("Ea1", "kJ/mol", 35.0, 80.0),
            Parameter("log10_A2", "log10(1/s)", 3.0, 9.0),
            Parameter("Ea2", "kJ/mol", 45.0, 100.0),
            Parameter("m", "1", 0.4, 1.8),
            Parameter("n", "1", 1.0, 3.0),
        ),
        outputs=(Output("final_cure_fraction", "1", "logit"), Output("peak_cure_rate", "1/s")),
        holdout_parameter="heating_rate",
        simulator=cure_kinetics,
    ),
    Task(
        inventory_id="F-PKG-02",
        model_id="Thermoset-ResidualStress-X5",
        backend="BPU",
        nist_record="mds2-3698",
        nist_url="https://data.nist.gov/od/id/mds2-3698",
        modality_scope="DSC/DIC/rheology/residual-stress packaging thermoset reference",
        equation_scope="restrained thermo-chemo-elastic residual-stress proxy with viscoelastic relaxation",
        parameters=(
            Parameter("encapsulant_modulus", "GPa", 1.0, 15.0),
            Parameter("poisson_ratio", "1", 0.25, 0.42),
            Parameter("encapsulant_cte", "ppm/K", 20.0, 80.0),
            Parameter("substrate_cte", "ppm/K", 2.5, 20.0),
            Parameter("cure_shrinkage", "%", 0.10, 3.00),
            Parameter("cure_degree", "1", 0.70, 0.995),
            Parameter("glass_transition", "degC", 80.0, 220.0),
            Parameter("cure_temperature", "degC", 120.0, 220.0),
            Parameter("service_temperature", "degC", -40.0, 125.0),
            Parameter("relaxation_fraction", "1", 0.10, 0.85),
            Parameter("interface_restraint", "1", 0.20, 1.00),
        ),
        outputs=(Output("residual_stress", "MPa"),),
        holdout_parameter="cure_shrinkage",
        simulator=residual_stress,
    ),
    Task(
        inventory_id="F-PKG-03",
        model_id="Package-Warpage-FEA-X5",
        backend="BPU",
        nist_record="mds2-3698+mds2-3725",
        nist_url="https://data.nist.gov/od/id/mds2-3725",
        modality_scope="packaging stress plus local mechanical-property references",
        equation_scope="reduced-order laminated-plate thermo-mechanical proxy; not a validated FEA solver",
        parameters=(
            Parameter("die_modulus", "GPa", 120.0, 190.0),
            Parameter("substrate_modulus", "GPa", 10.0, 35.0),
            Parameter("interface_modulus", "GPa", 1.0, 15.0),
            Parameter("die_cte", "ppm/K", 2.3, 4.5),
            Parameter("substrate_cte", "ppm/K", 10.0, 22.0),
            Parameter("interface_cte", "ppm/K", 20.0, 70.0),
            Parameter("die_thickness", "mm", 0.05, 0.30),
            Parameter("substrate_thickness", "mm", 0.20, 1.20),
            Parameter("interface_thickness", "mm", 0.01, 0.12),
            Parameter("package_length", "mm", 5.0, 30.0),
            Parameter("aspect_ratio", "1", 1.0, 2.0),
            Parameter("temperature_delta", "K", 40.0, 220.0),
            Parameter("interface_restraint", "1", 0.20, 1.00),
        ),
        outputs=(
            Output("center_warpage", "um"),
            Output("peak_warpage", "um"),
            Output("interface_stress", "MPa"),
        ),
        holdout_parameter="package_length",
        simulator=package_warpage,
    ),
    Task(
        inventory_id="F-PKG-04",
        model_id="HybridBond-CuMechanics-CPU",
        backend="CPU",
        nist_record="mds2-3867",
        nist_url="https://data.nist.gov/od/id/mds2-3867",
        modality_scope="single-step/multi-step AFM indentation reference for hybrid-bond-ready Cu",
        equation_scope="Hertz elastic/contact-pressure plastic-blend proxy with protocol corrections",
        parameters=(
            Parameter("indentation_depth", "nm", 5.0, 250.0),
            Parameter("tip_radius", "nm", 20.0, 200.0),
            Parameter("cu_modulus", "GPa", 80.0, 140.0),
            Parameter("poisson_ratio", "1", 0.28, 0.38),
            Parameter("yield_strength", "GPa", 0.30, 2.00),
            Parameter("surface_roughness", "nm", 0.10, 10.0),
            Parameter("oxide_thickness", "nm", 0.0, 8.0),
            Parameter("loading_rate", "uN/s", 0.05, 10.0),
            Parameter("hold_time", "ms", 0.0, 1000.0),
            Parameter("multistep_ratio", "1", 0.0, 1.0),
        ),
        outputs=(Output("indentation_force", "uN"), Output("contact_stiffness", "N/m")),
        holdout_parameter="indentation_depth",
        simulator=copper_afm_mechanics,
    ),
)


def probe_local_sources() -> dict[str, Any]:
    data_extensions = {
        ".csv", ".tsv", ".txt", ".xls", ".xlsx", ".zip", ".tar", ".gz",
        ".h5", ".hdf5", ".mat", ".npy", ".npz", ".jsonl", ".parquet",
    }
    excluded = {
        ".git", ".venv", ".venv-icmat", "node_modules", "__pycache__",
        "artifacts", "evidence", "backups",
    }
    search_roots = [
        ROOT / "research/data_assets",
        ROOT / "icmat_foundry",
        ROOT / "evaluation/icmat_foundry",
        ROOT / "CIMC_candidates",
    ]
    records: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        ids = [item for item in task.nist_record.split("+") if item.startswith("mds2-")]
        matches: list[str] = []
        for search_root in search_roots:
            if not search_root.exists():
                continue
            for current, dirnames, filenames in os.walk(search_root):
                dirnames[:] = [name for name in dirnames if name not in excluded]
                current_path = Path(current)
                for name in filenames:
                    path = current_path / name
                    lowered = path.as_posix().lower()
                    if path.suffix.lower() in data_extensions and any(item in lowered for item in ids):
                        matches.append(path.relative_to(ROOT).as_posix())
        records[task.inventory_id] = {
            "nist_record": task.nist_record,
            "canonical_url": task.nist_url,
            "local_data_files": sorted(set(matches)),
            "local_data_available": bool(matches),
            "decision": "USE_LOCAL_DATA" if matches else "SIM_ONLY_PHYSICS_PROXY",
        }
    return {
        "schema": "x5_icmat_foundry.package_source_probe.v1",
        "searched_roots": [path.relative_to(ROOT).as_posix() for path in search_roots],
        "data_extensions": sorted(data_extensions),
        "records": records,
        "result": "ALL_REFERENCED_NIST_DATA_FILES_ABSENT_USE_SIM_ONLY"
        if not any(item["local_data_available"] for item in records.values())
        else "LOCAL_DATA_REQUIRES_SEPARATE_SCHEMA_REVIEW",
    }


def transform_outputs(values: np.ndarray, outputs: tuple[Output, ...]) -> np.ndarray:
    transformed = np.empty_like(values, dtype=np.float64)
    for index, output in enumerate(outputs):
        column = values[:, index]
        if output.transform == "logit":
            clipped = np.clip(column, 1.0e-6, 1.0 - 1.0e-6)
            transformed[:, index] = np.log(clipped / (1.0 - clipped))
        elif output.transform == "log1p":
            transformed[:, index] = np.log1p(np.maximum(column, 0.0))
        else:
            transformed[:, index] = column
    return transformed


def inverse_outputs(values: np.ndarray, outputs: tuple[Output, ...]) -> np.ndarray:
    restored = np.empty_like(values, dtype=np.float64)
    for index, output in enumerate(outputs):
        column = values[:, index]
        if output.transform == "logit":
            clipped = np.clip(column, -40.0, 40.0)
            restored[:, index] = 1.0 / (1.0 + np.exp(-clipped))
        elif output.transform == "log1p":
            restored[:, index] = np.maximum(
                np.expm1(np.clip(column, -20.0, 30.0)), 0.0
            )
        else:
            restored[:, index] = column
    return restored


def make_dataset(task: Task) -> dict[str, np.ndarray]:
    task_index = next(index for index, item in enumerate(TASKS) if item.inventory_id == task.inventory_id)
    rng = np.random.default_rng(SEED + 1009 * task_index)
    unit = rng.random((SAMPLE_COUNT, len(task.parameters)), dtype=np.float64)
    lows = np.asarray([item.low for item in task.parameters], dtype=np.float64)
    highs = np.asarray([item.high for item in task.parameters], dtype=np.float64)
    inputs = lows + unit * (highs - lows)
    targets = task.simulator(inputs)
    if targets.shape != (SAMPLE_COUNT, len(task.outputs)):
        raise ValueError(f"unexpected simulator output for {task.inventory_id}: {targets.shape}")
    if not np.isfinite(inputs).all() or not np.isfinite(targets).all() or (targets < 0.0).any():
        raise ValueError(f"non-finite or negative physics output for {task.inventory_id}")
    anchor = [item.name for item in task.parameters].index(task.holdout_parameter)
    test_mask = (unit[:, anchor] >= HOLDOUT_LOW) & (unit[:, anchor] <= HOLDOUT_HIGH)
    pool = np.flatnonzero(~test_mask)
    rng.shuffle(pool)
    tune_size = max(1000, int(0.14 * len(pool)))
    return {
        "inputs": inputs.astype(np.float32),
        "targets": targets.astype(np.float32),
        "train_indices": pool[tune_size:].astype(np.int64),
        "tune_indices": pool[:tune_size].astype(np.int64),
        "test_indices": np.flatnonzero(test_mask).astype(np.int64),
        "anchor_unit": unit[:, anchor].astype(np.float32),
    }


def evaluate(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    task: Task,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    baseline: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), 4096):
            outputs.append(model(torch.from_numpy(x[start : start + 4096]).to(device)).cpu().numpy())
    normalized_prediction = np.concatenate(outputs, axis=0)
    latent_prediction = normalized_prediction * y_std + y_mean
    prediction = inverse_outputs(latent_prediction, task.outputs)
    ranges = np.ptp(y, axis=0)
    ranges[ranges < 1.0e-9] = 1.0
    mae = np.asarray(
        [mean_absolute_error(y[:, index], prediction[:, index]) for index in range(y.shape[1])]
    )
    baseline_mae = np.asarray(
        [mean_absolute_error(y[:, index], baseline[:, index]) for index in range(y.shape[1])]
    )
    r2 = np.asarray([r2_score(y[:, index], prediction[:, index]) for index in range(y.shape[1])])
    physical = {
        "prediction_all_finite": bool(np.isfinite(prediction).all()),
        "prediction_nonnegative": bool((prediction >= 0.0).all()),
        "target_all_finite": bool(np.isfinite(y).all()),
        "target_nonnegative": bool((y >= 0.0).all()),
    }
    if task.inventory_id == "F-PKG-01":
        physical["cure_fraction_in_unit_interval"] = bool(
            ((prediction[:, 0] >= 0.0) & (prediction[:, 0] <= 1.0)).all()
        )
    metrics = {
        "output_names": [item.name for item in task.outputs],
        "mae": mae.tolist(),
        "median_baseline_mae": baseline_mae.tolist(),
        "normalized_mae_by_test_range": (mae / ranges).tolist(),
        "median_baseline_normalized_mae": (baseline_mae / ranges).tolist(),
        "mean_normalized_mae": float(np.mean(mae / ranges)),
        "mean_baseline_normalized_mae": float(np.mean(baseline_mae / ranges)),
        "r2": r2.tolist(),
        "beats_median_baseline_all_outputs": bool((mae < baseline_mae).all()),
        "physical_checks": physical,
    }
    return metrics, normalized_prediction.astype(np.float32), prediction.astype(np.float32)


def train_one(task: Task, device: torch.device) -> dict[str, Any]:
    started = time.time()
    task_index = next(index for index, item in enumerate(TASKS) if item.inventory_id == task.inventory_id)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    dataset = make_dataset(task)
    train_idx = dataset["train_indices"]
    tune_idx = dataset["tune_indices"]
    test_idx = dataset["test_indices"]
    raw_x = dataset["inputs"]
    raw_y = dataset["targets"]

    x_mean = raw_x[train_idx].mean(axis=0).astype(np.float32)
    x_std = raw_x[train_idx].std(axis=0).astype(np.float32)
    x_std[x_std < 1.0e-7] = 1.0
    x = ((raw_x - x_mean) / x_std).astype(np.float32)
    transformed_y = transform_outputs(raw_y, task.outputs).astype(np.float32)
    y_mean = transformed_y[train_idx].mean(axis=0).astype(np.float32)
    y_std = transformed_y[train_idx].std(axis=0).astype(np.float32)
    y_std[y_std < 1.0e-7] = 1.0
    y = ((transformed_y - y_mean) / y_std).astype(np.float32)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])),
        batch_size=512,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )
    model = CompactSurrogate(x.shape[1], y.shape[1], task.backend).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-5)
    loss_fn = nn.SmoothL1Loss(beta=0.25)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    patience = 18
    stale = 0
    for epoch in range(1, 121):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            tune_prediction = model(torch.from_numpy(x[tune_idx]).to(device))
            tune_loss = float(loss_fn(tune_prediction, torch.from_numpy(y[tune_idx]).to(device)).item())
        if tune_loss < best_loss - 1.0e-6:
            best_loss = tune_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"training did not produce a checkpoint for {task.inventory_id}")
    model.load_state_dict(best_state)
    median = np.median(raw_y[train_idx], axis=0, keepdims=True)
    baseline = np.repeat(median, len(test_idx), axis=0)
    metrics, _, physical_prediction = evaluate(
        model, x[test_idx], raw_y[test_idx], task, y_mean, y_std, baseline, device
    )

    slug = task.inventory_id.lower().replace("-", "_")
    artifact_dir = ARTIFACT_ROOT / slug
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weight_path = artifact_dir / "model_fp32.pt"
    normalization_path = artifact_dir / "normalization.npz"
    onnx_path = artifact_dir / "model_fp32.onnx"
    input_fixture_path = artifact_dir / "input_fixture.npy"
    output_fixture_path = artifact_dir / "output_fixture.npy"
    torch.save(
        {
            "state_dict": best_state,
            "inventory_id": task.inventory_id,
            "model_id": task.model_id,
            "seed": SEED,
            "input_dim": x.shape[1],
            "output_dim": y.shape[1],
            "backend_candidate": task.backend,
            "evidence_class": "SIM_ONLY",
        },
        weight_path,
    )
    np.savez(
        normalization_path,
        input_mean=x_mean,
        input_std=x_std,
        output_latent_mean=y_mean,
        output_latent_std=y_std,
    )
    model = model.cpu().eval()
    fixture = x[test_idx[:1]].astype(np.float32)
    torch.onnx.export(
        model,
        torch.from_numpy(fixture),
        onnx_path,
        input_names=["features_normalized_fp32"],
        output_names=["outputs_normalized_fp32"],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    graph = onnx.load(onnx_path)
    graph.ir_version = 7
    onnx.checker.check_model(graph)
    onnx.save(graph, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_output = session.run(None, {"features_normalized_fp32": fixture})[0]
    with torch.inference_mode():
        torch_output = model(torch.from_numpy(fixture)).numpy()
    parity = float(np.max(np.abs(ort_output - torch_output)))
    latent_fixture = ort_output * y_std + y_mean
    physical_fixture = inverse_outputs(latent_fixture, task.outputs).astype(np.float32)
    np.save(input_fixture_path, fixture)
    np.save(output_fixture_path, physical_fixture)

    all_physical = metrics["physical_checks"]
    accepted = bool(
        metrics["beats_median_baseline_all_outputs"]
        and metrics["mean_normalized_mae"] <= 0.10
        and all(all_physical.values())
        and parity <= 1.0e-5
    )
    status = (
        "PC_ACCEPTED_SIM_ONLY_BPU_CANDIDATE_BOARD_PENDING"
        if accepted and task.backend == "BPU"
        else "PC_ACCEPTED_SIM_ONLY_CPU_ONNX"
        if accepted
        else "PC_SIM_ONLY_QUALITY_HOLD"
    )
    artifact_hashes = {
        path.name: sha256(path)
        for path in (
            weight_path,
            normalization_path,
            onnx_path,
            input_fixture_path,
            output_fixture_path,
        )
    }
    receipt = {
        "schema": "x5_icmat_foundry.package_model_receipt.v1",
        "inventory_id": task.inventory_id,
        "model_id": task.model_id,
        "status": status,
        "accepted": accepted,
        "evidence_class": "SIM_ONLY",
        "claim_boundary": (
            "Physics-generated surrogate candidate only. It is not NIST experimental training, "
            "not production calibration, not a compiled Bayes-e binary, and not X5-verified."
        ),
        "nist_reference": {
            "record": task.nist_record,
            "canonical_url": task.nist_url,
            "modality_scope": task.modality_scope,
            "local_data_used": False,
        },
        "physics_proxy": task.equation_scope,
        "parameter_envelope": [asdict(item) for item in task.parameters],
        "outputs": [asdict(item) for item in task.outputs],
        "split": {
            "type": "PARAMETER_RANGE_BLOCK_HOLDOUT",
            "holdout_parameter": task.holdout_parameter,
            "heldout_normalized_interval": [HOLDOUT_LOW, HOLDOUT_HIGH],
            "train_count": int(len(train_idx)),
            "tune_count": int(len(tune_idx)),
            "test_count": int(len(test_idx)),
            "random_seed": SEED,
        },
        "training": {
            "device": str(device),
            "single_seed": SEED,
            "best_epoch": best_epoch,
            "best_tune_loss": best_loss,
            "parameter_count": int(sum(value.numel() for value in model.parameters())),
            "elapsed_seconds": round(time.time() - started, 3),
        },
        "metrics": metrics,
        "export": {
            "backend_target": task.backend,
            "static_batch": 1,
            "onnx_opset": 11,
            "onnx_ir": 7,
            "onnx_checker": "PASS",
            "onnxruntime_provider": "CPUExecutionProvider",
            "torch_ort_max_abs": parity,
            "bpu_compiled": False,
            "x5_contacted": False,
        },
        "artifacts": artifact_hashes,
        "script_sha256": sha256(Path(__file__)),
    }
    receipt_path = EVIDENCE_ROOT / f"{slug}_receipt.v1.json"
    write_json(receipt_path, receipt)
    receipt_sha = write_hash_sidecar(receipt_path)
    result = {
        "inventory_id": task.inventory_id,
        "model_id": task.model_id,
        "status": status,
        "accepted": accepted,
        "backend": task.backend,
        "weight_sha256": artifact_hashes["model_fp32.pt"],
        "onnx_sha256": artifact_hashes["model_fp32.onnx"],
        "receipt": receipt_path.relative_to(ROOT).as_posix(),
        "receipt_sha256": receipt_sha,
        "mean_normalized_mae": metrics["mean_normalized_mae"],
        "baseline_mean_normalized_mae": metrics["mean_baseline_normalized_mae"],
        "torch_ort_max_abs": parity,
    }
    print(
        f"[{task.inventory_id}] {status} nMAE={metrics['mean_normalized_mae']:.5f} "
        f"baseline={metrics['mean_baseline_normalized_mae']:.5f} parity={parity:.3e}",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[task.inventory_id for task in TASKS],
        help="Train only the selected tasks, then summarize all available task receipts.",
    )
    args = parser.parse_args()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    source_probe = probe_local_sources()
    probe_path = EVIDENCE_ROOT / "source_probe.v1.json"
    write_json(probe_path, source_probe)
    write_hash_sidecar(probe_path)
    if any(
        entry["local_data_available"] for entry in source_probe["records"].values()
    ):
        raise RuntimeError(
            "Local NIST files were found. This fast-track SIM_ONLY trainer refuses to ingest them "
            "without a separate schema and license review."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = set(args.only or [task.inventory_id for task in TASKS])
    trained = {task.inventory_id: train_one(task, device) for task in TASKS if task.inventory_id in selected}
    results: list[dict[str, Any]] = []
    for task in TASKS:
        if task.inventory_id in trained:
            results.append(trained[task.inventory_id])
            continue
        slug = task.inventory_id.lower().replace("-", "_")
        receipt_path = EVIDENCE_ROOT / f"{slug}_receipt.v1.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(f"missing receipt for unselected task: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        results.append(
            {
                "inventory_id": task.inventory_id,
                "model_id": task.model_id,
                "status": receipt["status"],
                "accepted": bool(receipt["accepted"]),
                "backend": task.backend,
                "weight_sha256": receipt["artifacts"]["model_fp32.pt"],
                "onnx_sha256": receipt["artifacts"]["model_fp32.onnx"],
                "receipt": receipt_path.relative_to(ROOT).as_posix(),
                "receipt_sha256": sha256(receipt_path),
                "mean_normalized_mae": receipt["metrics"]["mean_normalized_mae"],
                "baseline_mean_normalized_mae": receipt["metrics"][
                    "mean_baseline_normalized_mae"
                ],
                "torch_ort_max_abs": receipt["export"]["torch_ort_max_abs"],
            }
        )
    weight_hashes = [item["weight_sha256"] for item in results]
    distinct_weights = len(set(weight_hashes)) == len(TASKS)
    accepted = all(item["accepted"] for item in results) and distinct_weights
    summary = {
        "schema": "x5_icmat_foundry.package_bank_run_summary.v1",
        "status": "PC_ACCEPTED_SIM_ONLY_X5_BOARD_PENDING" if accepted else "PC_QUALITY_HOLD",
        "accepted": accepted,
        "evidence_class": "SIM_ONLY",
        "model_count": len(results),
        "independent_weight_count": len(set(weight_hashes)),
        "all_weight_hashes_distinct": distinct_weights,
        "single_seed": SEED,
        "local_nist_data_used": False,
        "x5_contacted": False,
        "bpu_binary_produced": False,
        "results": results,
        "source_probe_sha256": sha256(probe_path),
        "script_sha256": sha256(Path(__file__)),
    }
    summary_path = EVIDENCE_ROOT / "run_summary.v1.json"
    write_json(summary_path, summary)
    summary_sha = write_hash_sidecar(summary_path)
    print(f"[package-bank] {summary['status']} summary_sha256={summary_sha}", flush=True)
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
