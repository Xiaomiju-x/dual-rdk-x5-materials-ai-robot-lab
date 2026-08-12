"""Deterministic PC training and one-shot locked-set evaluation."""
from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .contracts import (
    BOUNDARY_TOLERANCE_PX,
    CLAIM_BOUNDARY,
    DEFAULT_THRESHOLD,
    INPUT_SIZE,
    LOCKED_TEST_SET,
    MODEL_NAME,
    MODEL_VERSION,
    VALID_SUBSET_TRAIN_SETS,
)
from .data import (
    OfficialSubsetPatchDataset,
    load_official_pair,
    tile_origins,
    validate_finite_probability_map,
    write_json_atomic,
)
from .metrics import otsu_baseline, segmentation_metrics
from .model import LiteSemSeg, segmentation_loss


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260728
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    patch_stride: int = 64
    inference_stride: int = 64
    threshold: float = DEFAULT_THRESHOLD
    boundary_tolerance_px: int = BOUNDARY_TOLERANCE_PX
    device: str = "auto"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


@torch.inference_mode()
def predict_tiled(
    model: LiteSemSeg,
    image: np.ndarray,
    *,
    device: torch.device,
    stride: int,
) -> np.ndarray:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("predict_tiled requires a 2D uint8 image")
    height, width = image.shape
    probability_sum = np.zeros((height, width), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.float64)
    model.eval()
    for top, left in tile_origins(
        height,
        width,
        tile_size=INPUT_SIZE,
        stride=stride,
    ):
        patch = image[top : top + INPUT_SIZE, left : left + INPUT_SIZE]
        inputs = (
            torch.from_numpy(np.ascontiguousarray(patch))
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .div_(255.0)
            .to(device)
        )
        probability = torch.sigmoid(model(inputs))[0, 0].cpu().numpy()
        probability_sum[
            top : top + INPUT_SIZE,
            left : left + INPUT_SIZE,
        ] += probability
        count[top : top + INPUT_SIZE, left : left + INPUT_SIZE] += 1.0
    if np.any(count == 0):
        raise RuntimeError("tiled inference left uncovered pixels")
    probability = (probability_sum / count).astype(np.float32)
    validate_finite_probability_map(probability)
    return probability


def _save_uint8(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L").save(path)


def train_official_subset_baseline(
    mask_archive: Path,
    output_dir: Path,
    config: TrainingConfig,
) -> tuple[LiteSemSeg, dict[str, Any]]:
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = OfficialSubsetPatchDataset(
        mask_archive,
        patch_size=INPUT_SIZE,
        stride=config.patch_stride,
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    model = LiteSemSeg().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    peak_memory = 0

    for epoch in range(config.epochs):
        model.train()
        cumulative = 0.0
        samples = 0
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(inputs)
                loss = segmentation_loss(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            cumulative += float(loss.detach().cpu()) * inputs.shape[0]
            samples += int(inputs.shape[0])
        if device.type == "cuda":
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(device)))
        history.append(
            {
                "epoch": epoch + 1,
                "mean_loss": cumulative / max(samples, 1),
            }
        )

    training_seconds = time.perf_counter() - started
    model = model.cpu().eval()
    weights_path = output_dir / "sem_metrology_x5_lite_fp32.pt"
    torch.save(model.state_dict(), weights_path)

    # Locked set 6 is opened exactly once, after all training choices are fixed.
    test_image, test_mask_u8 = load_official_pair(mask_archive, LOCKED_TEST_SET)
    test_mask = test_mask_u8 > 0
    probability = predict_tiled(
        model.to(device),
        test_image,
        device=device,
        stride=config.inference_stride,
    )
    model = model.cpu().eval()
    prediction = probability >= config.threshold
    learned_metrics = segmentation_metrics(
        prediction,
        test_mask,
        boundary_tolerance_px=config.boundary_tolerance_px,
    )
    baseline_prediction = otsu_baseline(test_image)
    baseline_metrics = segmentation_metrics(
        baseline_prediction,
        test_mask,
        boundary_tolerance_px=config.boundary_tolerance_px,
    )

    _save_uint8(output_dir / "set6_input_simulated_sem.png", test_image)
    _save_uint8(output_dir / "set6_ground_truth.png", test_mask.astype(np.uint8) * 255)
    _save_uint8(output_dir / "set6_prediction.png", prediction.astype(np.uint8) * 255)
    error = np.zeros_like(test_image)
    error[prediction & ~test_mask] = 170
    error[~prediction & test_mask] = 255
    _save_uint8(output_dir / "set6_error_fp170_fn255.png", error)

    report: dict[str, Any] = {
        "schema": "icmat_sem_training_evaluation.v1",
        "candidate_status": "OFFICIAL_SUBSET_BASELINE",
        "release_eligible": False,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "config": asdict(config),
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "training_seconds": training_seconds,
            "peak_cuda_memory_bytes": peak_memory if device.type == "cuda" else None,
        },
        "training_data": {
            "sets": list(VALID_SUBSET_TRAIN_SETS),
            "patches_with_geometric_augmentation": len(dataset),
            "official_full_intensity_corpus_used": False,
            "official_set4_used": False,
        },
        "locked_test": {
            "set": LOCKED_TEST_SET,
            "access_count": 1,
            "used_for_model_selection": False,
            "threshold_fixed_before_access": config.threshold,
            "learned_model": learned_metrics,
            "otsu_reference_baseline": baseline_metrics,
        },
        "history": history,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write_json_atomic(output_dir / "training_evaluation.v1.json", report)
    return model, report
