"""Train the fast-track theoretical-XRD restoration and retrieval models.

The source spectra are calculated locally from version-pinned JARVIS-DFT
structures. Degraded inputs are synthetic instrument-like perturbations. No
experimental XRD measurement is used or claimed by this script.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
import torch
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Composition, Lattice, Structure
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = ROOT / "icmat_foundry/finals_50model"
SOURCE = ROOT / "research/data_assets/icmat_foundry/nist_jarvis_dft/raw/jdft_3d-9-24-2025.json.zip"
SOURCE_RECEIPT = ROOT / "research/data_assets/icmat_foundry/nist_jarvis_dft/acquisition_receipt.v1.json"
ARTIFACT_ROOT = CANDIDATE / "artifacts/xrd_bank"
EVIDENCE_ROOT = CANDIDATE / "evidence/xrd_bank"
DATASET_PATH = ARTIFACT_ROOT / "theoretical_xrd_dataset.v1.npz"
DATASET_RECEIPT = EVIDENCE_ROOT / "theoretical_xrd_dataset.receipt.v1.json"

SEED = 20260801
GRID_MIN_DEG = 10.0
GRID_MAX_DEG = 90.0
GRID_POINTS = 512
GRID = np.linspace(GRID_MIN_DEG, GRID_MAX_DEG, GRID_POINTS, dtype=np.float32)
SPLIT_NAMES = {0: "train", 1: "tune", 2: "test"}
SPLIT_TARGETS = {0: 3800, 1: 500, 2: 500}


@dataclass(frozen=True)
class DatasetBank:
    spectra: np.ndarray
    split_codes: np.ndarray
    jids: np.ndarray
    formulas: np.ndarray
    structure_families: np.ndarray


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(1, 5), padding=(0, 2))
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(1, 5), padding=(0, 2))
        self.activation = nn.ReLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.activation(self.conv1(value))
        return self.activation(value + self.conv2(residual))


class DenoisePeakRestore(nn.Module):
    """BPU-friendly fixed-length residual CNN producing a restored profile."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=(1, 9), padding=(0, 4)),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(ResidualBlock(24), ResidualBlock(24), ResidualBlock(24))
        self.head = nn.Conv2d(24, 1, kernel_size=(1, 9), padding=(0, 4))

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.blocks(self.stem(spectrum))))


class PhaseEmbedding(nn.Module):
    """Independent Siamese encoder for candidate-phase retrieval."""

    def __init__(self, embedding_dim: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=(1, 9), stride=(1, 2), padding=(0, 4)),
            nn.ReLU(),
            nn.Conv2d(12, 20, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3)),
            nn.ReLU(),
            nn.Conv2d(20, 28, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.ReLU(),
            nn.Conv2d(28, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        embedding = self.projector(self.features(spectrum))
        norm = torch.sqrt(torch.sum(embedding * embedding, dim=1, keepdim=True) + 1.0e-12)
        return embedding / norm


def set_determinism() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_hash_sidecar(path: Path) -> str:
    digest = sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def canonical_formula(raw_formula: Any) -> str | None:
    try:
        formula = Composition(str(raw_formula)).reduced_formula
    except Exception:
        return None
    return formula if formula else None


def split_for_formula(formula: str) -> int:
    bucket = stable_int(f"split-v1|{formula}") % 100
    if bucket < 80:
        return 0
    if bucket < 90:
        return 1
    return 2


def structure_family(row: dict[str, Any], formula: str) -> str:
    atoms = row["atoms"]
    lattice = np.asarray(atoms["lattice_mat"], dtype=np.float64)
    lengths = np.linalg.norm(lattice, axis=1)
    scale = max(float(np.min(lengths)), 1.0e-8)
    ratios = ",".join(f"{value / scale:.2f}" for value in np.sort(lengths))
    return "|".join(
        [
            formula,
            str(row.get("spg_number") or row.get("spg") or "UNKNOWN"),
            str(len(atoms.get("elements", ()))),
            ratios,
        ]
    )


def row_to_structure(row: dict[str, Any]) -> Structure:
    atoms = row["atoms"]
    return Structure(
        Lattice(atoms["lattice_mat"]),
        atoms["elements"],
        atoms["coords"],
        coords_are_cartesian=bool(atoms.get("cartesian", True)),
    )


def pattern_to_grid(pattern: Any) -> np.ndarray | None:
    impulses = np.zeros(GRID_POINTS, dtype=np.float32)
    step = float(GRID[1] - GRID[0])
    for position, intensity in zip(pattern.x, pattern.y, strict=True):
        if not math.isfinite(float(position)) or not math.isfinite(float(intensity)):
            continue
        fractional = (float(position) - GRID_MIN_DEG) / step
        lower = int(math.floor(fractional))
        fraction = fractional - lower
        if 0 <= lower < GRID_POINTS:
            impulses[lower] += float(intensity) * (1.0 - fraction)
        if 0 <= lower + 1 < GRID_POINTS:
            impulses[lower + 1] += float(intensity) * fraction
    if float(np.max(impulses)) <= 0.0:
        return None
    # A finite 0.20-degree reference width avoids an unphysical delta target.
    clean = gaussian_filter1d(impulses, sigma=0.55, mode="nearest")
    maximum = float(np.max(clean))
    return (clean / maximum).astype(np.float32) if maximum > 0.0 else None


def load_jarvis_rows() -> list[dict[str, Any]]:
    with zipfile.ZipFile(SOURCE) as archive:
        members = archive.namelist()
        if len(members) != 1:
            raise ValueError(f"expected one JSON member, found {members}")
        with archive.open(members[0]) as stream:
            return json.load(stream)


def validate_split_isolation(bank: DatasetBank) -> dict[str, Any]:
    formula_sets: dict[int, set[str]] = {}
    structure_sets: dict[int, set[str]] = {}
    for code in SPLIT_NAMES:
        mask = bank.split_codes == code
        formula_sets[code] = set(bank.formulas[mask].tolist())
        structure_sets[code] = set(bank.structure_families[mask].tolist())
    formula_overlap = sum(
        len(formula_sets[left] & formula_sets[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    structure_overlap = sum(
        len(structure_sets[left] & structure_sets[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    if formula_overlap or structure_overlap:
        raise ValueError("formula or structure-family leakage detected")
    return {
        "split_counts": {
            SPLIT_NAMES[code]: int(np.sum(bank.split_codes == code)) for code in SPLIT_NAMES
        },
        "formula_family_overlap_count": formula_overlap,
        "structure_family_overlap_count": structure_overlap,
        "split_unit": "canonical reduced formula; one deterministic structure representative per formula",
        "structure_family_key": "reduced_formula|space_group|natoms|rounded_normalized_lattice_lengths",
    }


def build_or_load_dataset() -> tuple[DatasetBank, dict[str, Any]]:
    source_digest = sha256(SOURCE)
    expected_source = "f9e0a3309f0000d5de1ec9e49c93963109ea45e63f451103f8f6595d2eabf7f5"
    if source_digest != expected_source:
        raise ValueError(f"JARVIS source digest mismatch: {source_digest}")
    if DATASET_PATH.is_file() and DATASET_RECEIPT.is_file():
        receipt = json.loads(DATASET_RECEIPT.read_text(encoding="utf-8"))
        if receipt.get("source_sha256") == source_digest:
            with np.load(DATASET_PATH, allow_pickle=False) as loaded:
                bank = DatasetBank(
                    spectra=loaded["spectra"].astype(np.float32),
                    split_codes=loaded["split_codes"],
                    jids=loaded["jids"],
                    formulas=loaded["formulas"],
                    structure_families=loaded["structure_families"],
                )
            isolation = validate_split_isolation(bank)
            return bank, {**receipt, "split_isolation": isolation, "cache_reused": True}

    print("[xrd-bank] loading version-pinned JARVIS-DFT source", flush=True)
    rows = load_jarvis_rows()
    ordered = sorted(rows, key=lambda row: stable_int(f"row-v1|{row.get('jid', '')}"))
    calculator = XRDCalculator(wavelength="CuKa")
    spectra: list[np.ndarray] = []
    split_codes: list[int] = []
    jids: list[str] = []
    formulas: list[str] = []
    structure_families: list[str] = []
    used_formulas: set[str] = set()
    counts = {code: 0 for code in SPLIT_NAMES}
    failures = 0
    started = time.time()
    warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
    for row in ordered:
        formula = canonical_formula(row.get("formula"))
        if formula is None or formula in used_formulas:
            continue
        code = split_for_formula(formula)
        if counts[code] >= SPLIT_TARGETS[code]:
            continue
        atoms = row.get("atoms")
        if not isinstance(atoms, dict):
            continue
        natoms = len(atoms.get("elements", ()))
        if natoms < 1 or natoms > 48:
            continue
        try:
            structure = row_to_structure(row)
            pattern = calculator.get_pattern(
                structure,
                scaled=True,
                two_theta_range=(GRID_MIN_DEG, GRID_MAX_DEG),
            )
            spectrum = pattern_to_grid(pattern)
            if spectrum is None or int(np.sum(spectrum >= 0.05)) < 2:
                raise ValueError("empty or degenerate diffraction pattern")
            family = structure_family(row, formula)
        except Exception:
            failures += 1
            continue
        used_formulas.add(formula)
        spectra.append(spectrum)
        split_codes.append(code)
        jids.append(str(row.get("jid") or "UNKNOWN"))
        formulas.append(formula)
        structure_families.append(family)
        counts[code] += 1
        if len(spectra) % 250 == 0:
            print(
                f"[xrd-bank] generated {len(spectra)}/{sum(SPLIT_TARGETS.values())} "
                f"theoretical patterns in {time.time() - started:.1f}s",
                flush=True,
            )
        if all(counts[split] >= target for split, target in SPLIT_TARGETS.items()):
            break
    if counts != SPLIT_TARGETS:
        raise RuntimeError(f"could not fill deterministic split targets: {counts}")
    bank = DatasetBank(
        spectra=np.stack(spectra).astype(np.float32),
        split_codes=np.asarray(split_codes, dtype=np.uint8),
        jids=np.asarray(jids, dtype="U32"),
        formulas=np.asarray(formulas, dtype="U64"),
        structure_families=np.asarray(structure_families, dtype="U160"),
    )
    isolation = validate_split_isolation(bank)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATASET_PATH,
        spectra=bank.spectra.astype(np.float16),
        split_codes=bank.split_codes,
        jids=bank.jids,
        formulas=bank.formulas,
        structure_families=bank.structure_families,
        grid_2theta_deg=GRID,
    )
    source_receipt = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    receipt = {
        "schema": "x5_icmat_foundry.theoretical_xrd_dataset_receipt.v1",
        "state": "JARVIS_DFT_THEORETICAL_PLUS_SYNTHETIC_DEGRADATION_READY",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_path": SOURCE.relative_to(ROOT).as_posix(),
        "source_sha256": source_digest,
        "source_version": source_receipt.get("source_version"),
        "source_license": source_receipt.get("license_name"),
        "source_reuse_gate": source_receipt.get("reuse_gate"),
        "source_rows": len(rows),
        "selected_patterns": len(bank.spectra),
        "generation_failures_before_fill": failures,
        "radiation": "CuKa",
        "two_theta_grid_deg": [GRID_MIN_DEG, GRID_MAX_DEG, GRID_POINTS],
        "split_isolation": isolation,
        "dataset_path": DATASET_PATH.relative_to(ROOT).as_posix(),
        "dataset_sha256": sha256(DATASET_PATH),
        "claim_boundary": {
            "clean_target": "JARVIS_DFT_THEORETICAL_XRD_CALCULATED_FROM_STRUCTURE",
            "input_degradation": "SYNTHETIC_INSTRUMENT_LIKE_PHYSICAL_DEGRADATION",
            "experimental_measurement_used": False,
            "x5_contacted": False,
            "forbidden_claim": "Not experimental phase accuracy, instrument validation, or X5 runtime evidence.",
        },
        "cache_reused": False,
    }
    write_json(DATASET_RECEIPT, receipt)
    write_hash_sidecar(DATASET_RECEIPT)
    return bank, receipt


def degrade_spectrum(clean: np.ndarray, seed: int) -> np.ndarray:
    """Apply bounded XRD-like instrument and counting perturbations."""
    rng = np.random.default_rng(seed)
    step = float(GRID[1] - GRID[0])
    broaden_sigma_bins = float(rng.uniform(0.55, 2.20))
    broadened = gaussian_filter1d(clean, sigma=broaden_sigma_bins, mode="nearest")
    zero_shift_deg = float(rng.uniform(-0.25, 0.25))
    shifted_grid = GRID - zero_shift_deg
    shifted = np.interp(shifted_grid, GRID, broadened, left=0.0, right=0.0).astype(np.float32)

    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    envelope = 1.0 + float(rng.uniform(0.0, 0.20)) * np.sin(
        np.linspace(0.0, 2.0 * math.pi, GRID_POINTS, dtype=np.float32) + phase
    )
    signal = np.clip(shifted * envelope, 0.0, None)
    x = np.linspace(-1.0, 1.0, GRID_POINTS, dtype=np.float32)
    background = (
        float(rng.uniform(0.0, 0.06))
        + float(rng.uniform(-0.025, 0.025)) * x
        + float(rng.uniform(0.0, 0.06)) * x * x
    )
    hump_center = float(rng.uniform(-0.55, 0.55))
    hump_width = float(rng.uniform(0.10, 0.32))
    hump = float(rng.uniform(0.0, 0.14)) * np.exp(-0.5 * ((x - hump_center) / hump_width) ** 2)
    intensity = np.clip(float(rng.uniform(0.65, 1.15)) * signal + background + hump, 0.0, None)
    counts = float(rng.uniform(700.0, 7000.0))
    counted = rng.poisson(intensity * counts).astype(np.float32) / counts
    read_noise = rng.normal(0.0, rng.uniform(0.002, 0.018), GRID_POINTS).astype(np.float32)
    degraded = np.clip(counted + read_noise, 0.0, None)
    maximum = float(np.max(degraded))
    if maximum > 0.0:
        degraded /= maximum
    # Keep this explicit even though the step is not consumed by the model.
    _ = step
    return degraded.astype(np.float32)


def degraded_views(clean: np.ndarray, tag: str, views: int = 1) -> np.ndarray:
    result = []
    for view in range(views):
        rows = [
            degrade_spectrum(spectrum, stable_int(f"degrade-v1|{tag}|{view}|{index}|{SEED}"))
            for index, spectrum in enumerate(clean)
        ]
        result.append(np.stack(rows))
    return np.stack(result).astype(np.float32)


def shaped(values: np.ndarray) -> np.ndarray:
    return values[:, None, None, :].astype(np.float32)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def denoise_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = 1.0 + 8.0 * target + 4.0 * (target >= 0.08).to(target.dtype)
    profile = torch.mean(weights * torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none"))
    pred_delta = prediction[..., 1:] - prediction[..., :-1]
    target_delta = target[..., 1:] - target[..., :-1]
    derivative = torch.nn.functional.l1_loss(pred_delta, target_delta)
    return profile + 0.35 * derivative


def infer_torch(model: nn.Module, values: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            tensor = torch.from_numpy(shaped(values[start : start + batch_size])).to(device)
            outputs.append(model(tensor).cpu().numpy()[:, 0, 0])
    return np.concatenate(outputs, axis=0)


def peak_counts(reference: np.ndarray, candidate: np.ndarray) -> tuple[int, int, int]:
    truth, _ = find_peaks(reference, height=0.08, prominence=0.035, distance=2)
    predicted, _ = find_peaks(candidate, height=0.08, prominence=0.035, distance=2)
    used: set[int] = set()
    matched = 0
    for peak in truth:
        options = [index for index, other in enumerate(predicted) if index not in used and abs(int(other) - int(peak)) <= 2]
        if options:
            chosen = min(options, key=lambda index: abs(int(predicted[index]) - int(peak)))
            used.add(chosen)
            matched += 1
    return matched, len(predicted) - matched, len(truth) - matched


def profile_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    candidate = np.clip(candidate, 0.0, 1.0)
    mae = float(np.mean(np.abs(reference - candidate)))
    rmse = float(np.sqrt(np.mean((reference - candidate) ** 2)))
    cosine = np.sum(reference * candidate, axis=1) / (
        np.linalg.norm(reference, axis=1) * np.linalg.norm(candidate, axis=1) + 1.0e-12
    )
    tp = fp = fn = 0
    for truth, predicted in zip(reference, candidate, strict=True):
        row_tp, row_fp, row_fn = peak_counts(truth, predicted)
        tp += row_tp
        fp += row_fp
        fn += row_fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        "mae": mae,
        "rmse": rmse,
        "mean_cosine": float(np.mean(cosine)),
        "peak_precision": float(precision),
        "peak_recall": float(recall),
        "peak_f1": float(f1),
    }


def export_static_onnx(
    model: nn.Module,
    output_path: Path,
    input_name: str,
    output_name: str,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = copy.deepcopy(model).cpu().eval()
    dummy = torch.zeros((1, 1, 1, GRID_POINTS), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    proto = onnx.load(output_path)
    proto.ir_version = 7
    onnx.checker.check_model(proto)
    onnx.save(proto, output_path)
    checked = onnx.load(output_path)
    onnx.checker.check_model(checked)
    opsets = {item.domain or "ai.onnx": item.version for item in checked.opset_import}
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    return {
        "ir_version": checked.ir_version,
        "opsets": opsets,
        "input_name": session.get_inputs()[0].name,
        "input_shape": session.get_inputs()[0].shape,
        "output_name": session.get_outputs()[0].name,
        "output_shape": session.get_outputs()[0].shape,
        "providers": session.get_providers(),
    }


def train_denoiser(bank: DatasetBank, device: torch.device, source_receipt: dict[str, Any]) -> dict[str, Any]:
    output_dir = ARTIFACT_ROOT / "F-XRD-01"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_clean = bank.spectra[bank.split_codes == 0]
    tune_clean = bank.spectra[bank.split_codes == 1]
    test_clean = bank.spectra[bank.split_codes == 2]
    train_views = degraded_views(train_clean, "denoise-train", views=2).reshape(-1, GRID_POINTS)
    train_targets = np.repeat(train_clean[None, ...], 2, axis=0).reshape(-1, GRID_POINTS)
    tune_input = degraded_views(tune_clean, "denoise-tune")[0]
    test_input = degraded_views(test_clean, "denoise-test")[0]

    loader = DataLoader(
        TensorDataset(torch.from_numpy(shaped(train_views)), torch.from_numpy(shaped(train_targets))),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )
    model = DenoisePeakRestore().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1.0e-5)
    best_state = copy.deepcopy(model.state_dict())
    best_tune = float("inf")
    patience = 0
    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(1, 31):
        model.train()
        total = 0.0
        seen = 0
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = denoise_loss(model(noisy), clean)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(noisy)
            seen += len(noisy)
        tune_prediction = infer_torch(model, tune_input, device)
        tune_mae = float(np.mean(np.abs(tune_prediction - tune_clean)))
        history.append({"epoch": epoch, "train_loss": total / seen, "tune_mae": tune_mae})
        print(f"[F-XRD-01] epoch={epoch:02d} loss={total / seen:.6f} tune_mae={tune_mae:.6f}", flush=True)
        if tune_mae < best_tune - 1.0e-5:
            best_tune = tune_mae
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= 5:
            break
    model.load_state_dict(best_state)
    test_prediction = infer_torch(model, test_input, device)
    baseline_metrics = profile_metrics(test_clean, test_input)
    model_metrics = profile_metrics(test_clean, test_prediction)

    pt_path = output_dir / "model.pt"
    onnx_path = output_dir / "model.onnx"
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "architecture": "DenoisePeakRestore residual Conv2d",
            "input_shape": [1, 1, 1, GRID_POINTS],
            "seed": SEED,
        },
        pt_path,
    )
    onnx_contract = export_static_onnx(model, onnx_path, "xrd_degraded_fp32", "xrd_restored_fp32")
    fixed_input = shaped(test_input[:1])
    fixed_torch = model(torch.from_numpy(fixed_input).to(device)).detach().cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    fixed_ort = session.run(None, {"xrd_degraded_fp32": fixed_input})[0]
    parity = float(np.max(np.abs(fixed_torch - fixed_ort)))
    fixed_input_path = output_dir / "fixed_input.npz"
    fixed_output_path = output_dir / "fixed_output.npz"
    np.savez_compressed(fixed_input_path, xrd_degraded_fp32=fixed_input, grid_2theta_deg=GRID)
    np.savez_compressed(fixed_output_path, xrd_restored_fp32=fixed_ort)
    accepted = (
        model_metrics["mae"] < baseline_metrics["mae"]
        and model_metrics["peak_f1"] > baseline_metrics["peak_f1"]
        and parity <= 1.0e-5
    )
    artifacts = {
        path.name: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in (pt_path, onnx_path, fixed_input_path, fixed_output_path)
    }
    receipt = {
        "schema": "x5_icmat_foundry.xrd_model_receipt.v1",
        "inventory_id": "F-XRD-01",
        "model_id": "XRD-DenoisePeakRestore-X5",
        "state": "PC_ACCEPTED_BPU_ONNX_X5_PENDING" if accepted else "PC_TRAINED_QUALITY_GATE_FAILED",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": SEED,
        "device": str(device),
        "duration_seconds": round(time.time() - started, 3),
        "parameters": count_parameters(model),
        "training_samples": len(train_views),
        "tune_samples": len(tune_clean),
        "test_samples": len(test_clean),
        "task": "Restore a clean theoretical 1D diffraction profile from a synthetically degraded profile; peak positions/intensities are derived from the restored profile.",
        "baseline_degraded_input": baseline_metrics,
        "model_test": model_metrics,
        "quality_gate": {
            "mae_better_than_degraded_input": model_metrics["mae"] < baseline_metrics["mae"],
            "peak_f1_better_than_degraded_input": model_metrics["peak_f1"] > baseline_metrics["peak_f1"],
            "ort_max_abs_diff_le_1e-5": parity <= 1.0e-5,
            "pass": accepted,
        },
        "onnx": {**onnx_contract, "ort_max_abs_diff": parity, "bpu_conversion_performed": False},
        "artifacts": artifacts,
        "history": history,
        "source_dataset_receipt_sha256": sha256(DATASET_RECEIPT),
        "source_dataset_state": source_receipt["state"],
        "claim_boundary": {
            "clean_target": "JARVIS_DFT_THEORETICAL_XRD",
            "degradation": "SYNTHETIC_PHYSICAL_DEGRADATION",
            "experimental_xrd_used": False,
            "x5_contacted": False,
            "x5_runtime_verified": False,
        },
    }
    receipt_path = EVIDENCE_ROOT / "F-XRD-01.receipt.v1.json"
    write_json(receipt_path, receipt)
    receipt["receipt_sha256"] = write_hash_sidecar(receipt_path)
    return receipt


def contrastive_loss(left: torch.Tensor, right: torch.Tensor, temperature: float = 0.08) -> torch.Tensor:
    labels = torch.arange(len(left), device=left.device)
    logits = left @ right.transpose(0, 1) / temperature
    return 0.5 * (
        torch.nn.functional.cross_entropy(logits, labels)
        + torch.nn.functional.cross_entropy(logits.transpose(0, 1), labels)
    )


def infer_embeddings(model: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), 256):
            tensor = torch.from_numpy(shaped(values[start : start + 256])).to(device)
            outputs.append(model(tensor).cpu().numpy())
    result = np.concatenate(outputs, axis=0)
    result /= np.linalg.norm(result, axis=1, keepdims=True) + 1.0e-12
    return result


def retrieval_metrics(query: np.ndarray, gallery: np.ndarray) -> dict[str, float]:
    similarity = query @ gallery.T
    order = np.argsort(-similarity, axis=1)
    labels = np.arange(len(query))
    ranks = np.empty(len(query), dtype=np.int64)
    for index in range(len(query)):
        ranks[index] = int(np.flatnonzero(order[index] == labels[index])[0]) + 1
    return {
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "top10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "gallery_size": int(len(gallery)),
        "random_top1": float(1.0 / len(gallery)),
        "random_top5": float(min(5.0 / len(gallery), 1.0)),
    }


def train_embedding(bank: DatasetBank, device: torch.device, source_receipt: dict[str, Any]) -> dict[str, Any]:
    output_dir = ARTIFACT_ROOT / "F-XRD-02"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_clean = bank.spectra[bank.split_codes == 0]
    tune_clean = bank.spectra[bank.split_codes == 1]
    test_clean = bank.spectra[bank.split_codes == 2]
    train_views = degraded_views(train_clean, "embedding-train", views=2)
    tune_query = degraded_views(tune_clean, "embedding-tune")[0]
    test_query = degraded_views(test_clean, "embedding-test")[0]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(shaped(train_views[0])), torch.from_numpy(shaped(train_views[1]))),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED + 1),
        num_workers=0,
        drop_last=True,
    )
    model = PhaseEmbedding().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_top1 = -1.0
    patience = 0
    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(1, 31):
        model.train()
        total = 0.0
        seen = 0
        for left, right in loader:
            left = left.to(device)
            right = right.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = contrastive_loss(model(left), model(right))
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(left)
            seen += len(left)
        tune_metrics = retrieval_metrics(
            infer_embeddings(model, tune_query, device),
            infer_embeddings(model, tune_clean, device),
        )
        history.append({"epoch": epoch, "train_loss": total / seen, "tune_top1": tune_metrics["top1"]})
        print(
            f"[F-XRD-02] epoch={epoch:02d} loss={total / seen:.6f} tune_top1={tune_metrics['top1']:.4f}",
            flush=True,
        )
        if tune_metrics["top1"] > best_top1 + 1.0e-6:
            best_top1 = tune_metrics["top1"]
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= 6:
            break
    model.load_state_dict(best_state)
    test_metrics = retrieval_metrics(
        infer_embeddings(model, test_query, device),
        infer_embeddings(model, test_clean, device),
    )

    pt_path = output_dir / "model.pt"
    onnx_path = output_dir / "model.onnx"
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "architecture": "PhaseEmbedding Siamese Conv2d encoder",
            "embedding_dim": 32,
            "input_shape": [1, 1, 1, GRID_POINTS],
            "seed": SEED,
        },
        pt_path,
    )
    onnx_contract = export_static_onnx(model, onnx_path, "xrd_profile_fp32", "phase_embedding_fp32")
    fixed_input = shaped(test_query[:1])
    fixed_torch = model(torch.from_numpy(fixed_input).to(device)).detach().cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    fixed_ort = session.run(None, {"xrd_profile_fp32": fixed_input})[0]
    parity = float(np.max(np.abs(fixed_torch - fixed_ort)))
    fixed_input_path = output_dir / "fixed_input.npz"
    fixed_output_path = output_dir / "fixed_output.npz"
    gallery_path = output_dir / "test_gallery.npz"
    np.savez_compressed(fixed_input_path, xrd_profile_fp32=fixed_input, grid_2theta_deg=GRID)
    np.savez_compressed(fixed_output_path, phase_embedding_fp32=fixed_ort)
    test_mask = bank.split_codes == 2
    gallery_embeddings = infer_embeddings(model, test_clean, device)
    np.savez_compressed(
        gallery_path,
        embeddings=gallery_embeddings.astype(np.float32),
        jids=bank.jids[test_mask],
        formulas=bank.formulas[test_mask],
    )
    accepted = test_metrics["top1"] >= 0.50 and test_metrics["top5"] >= 0.80 and parity <= 1.0e-5
    artifacts = {
        path.name: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in (pt_path, onnx_path, fixed_input_path, fixed_output_path, gallery_path)
    }
    receipt = {
        "schema": "x5_icmat_foundry.xrd_model_receipt.v1",
        "inventory_id": "F-XRD-02",
        "model_id": "XRD-PhaseEmbedding-X5",
        "state": "PC_ACCEPTED_BPU_ONNX_X5_PENDING" if accepted else "PC_TRAINED_QUALITY_GATE_FAILED",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": SEED,
        "device": str(device),
        "duration_seconds": round(time.time() - started, 3),
        "parameters": count_parameters(model),
        "embedding_dim": 32,
        "training_phases": len(train_clean),
        "tune_phases": len(tune_clean),
        "test_phases": len(test_clean),
        "task": "Independent Siamese embedding retrieval from a degraded/query spectrum to a held-out theoretical phase gallery.",
        "test_retrieval": test_metrics,
        "quality_gate": {
            "top1_ge_0.50": test_metrics["top1"] >= 0.50,
            "top5_ge_0.80": test_metrics["top5"] >= 0.80,
            "ort_max_abs_diff_le_1e-5": parity <= 1.0e-5,
            "pass": accepted,
        },
        "onnx": {**onnx_contract, "ort_max_abs_diff": parity, "bpu_conversion_performed": False},
        "artifacts": artifacts,
        "history": history,
        "source_dataset_receipt_sha256": sha256(DATASET_RECEIPT),
        "source_dataset_state": source_receipt["state"],
        "claim_boundary": {
            "gallery": "HELD_OUT_JARVIS_DFT_THEORETICAL_XRD",
            "query_degradation": "SYNTHETIC_PHYSICAL_DEGRADATION",
            "experimental_xrd_used": False,
            "exact_phase_retrieval_only": True,
            "x5_contacted": False,
            "x5_runtime_verified": False,
        },
    }
    receipt_path = EVIDENCE_ROOT / "F-XRD-02.receipt.v1.json"
    write_json(receipt_path, receipt)
    receipt["receipt_sha256"] = write_hash_sidecar(receipt_path)
    return receipt


def artifact_manifest(paths: Iterable[Path]) -> dict[str, Any]:
    files = []
    for path in sorted(set(paths), key=lambda item: item.as_posix()):
        files.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {"schema": "x5_icmat_foundry.xrd_bank_artifact_manifest.v1", "files": files}


def main() -> None:
    set_determinism()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[xrd-bank] device={device} seed={SEED}", flush=True)
    bank, dataset_receipt = build_or_load_dataset()
    denoise_receipt = train_denoiser(bank, device, dataset_receipt)
    embedding_receipt = train_embedding(bank, device, dataset_receipt)

    artifact_paths = [path for path in ARTIFACT_ROOT.rglob("*") if path.is_file()]
    manifest = artifact_manifest(artifact_paths)
    manifest_path = EVIDENCE_ROOT / "artifact_manifest.v1.json"
    write_json(manifest_path, manifest)
    manifest_digest = write_hash_sidecar(manifest_path)
    summary = {
        "schema": "x5_icmat_foundry.xrd_bank_summary.v1",
        "state": (
            "PC_ACCEPTED_BPU_ONNX_X5_PENDING"
            if denoise_receipt["quality_gate"]["pass"] and embedding_receipt["quality_gate"]["pass"]
            else "PC_PARTIAL_OR_FAILED"
        ),
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": str(device),
        "seed": SEED,
        "models": {
            "F-XRD-01": {
                "state": denoise_receipt["state"],
                "test": denoise_receipt["model_test"],
            },
            "F-XRD-02": {
                "state": embedding_receipt["state"],
                "test": embedding_receipt["test_retrieval"],
            },
        },
        "dataset_receipt": DATASET_RECEIPT.relative_to(ROOT).as_posix(),
        "dataset_receipt_sha256": sha256(DATASET_RECEIPT),
        "artifact_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "artifact_manifest_sha256": manifest_digest,
        "production_files_modified": False,
        "registry_or_overlay_modified": False,
        "x5_contacted": False,
        "claim_boundary": "Theoretical JARVIS-DFT XRD plus synthetic degradation; not experimental-XRD or X5-runtime evidence.",
    }
    summary_path = EVIDENCE_ROOT / "summary.v1.json"
    write_json(summary_path, summary)
    write_hash_sidecar(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["state"] != "PC_ACCEPTED_BPU_ONNX_X5_PENDING":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
