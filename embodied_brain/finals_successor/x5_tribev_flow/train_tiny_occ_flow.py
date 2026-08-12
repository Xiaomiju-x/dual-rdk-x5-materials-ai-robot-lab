#!/usr/bin/env python3
"""Train and evaluate TinyOccFlowStudent on contract-valid NPZ episodes.

This utility is deliberately offline. It has no ROS, serial, F407, TF, or
``/cmd_vel`` interface. Synthetic results validate the learning/export
pipeline only and are never promoted as real-world navigation accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.optim import AdamW

if __package__:
    from .dataset import (
        HISTORY_FRAMES,
        SPLIT_NAMES,
        TRIBEV_CHANNEL_NAMES,
        EpisodeRef,
        build_episode_refs,
        flatten_tribev_history,
        load_episode,
        split_episode_refs,
        summarize_splits,
    )
    from .metrics import (
        binary_occupancy_metrics,
        expected_calibration_error,
        flow_endpoint_error,
        trajectory_distribution_metrics,
    )
    from .models import TINY_OCC_FLOW_INPUT_SHAPE, TinyOccFlowStudent
    from .shadow_guard import SplitConformalEpisodeCalibrator
else:
    from dataset import (  # type: ignore[no-redef]
        HISTORY_FRAMES,
        SPLIT_NAMES,
        TRIBEV_CHANNEL_NAMES,
        EpisodeRef,
        build_episode_refs,
        flatten_tribev_history,
        load_episode,
        split_episode_refs,
        summarize_splits,
    )
    from metrics import (  # type: ignore[no-redef]
        binary_occupancy_metrics,
        expected_calibration_error,
        flow_endpoint_error,
        trajectory_distribution_metrics,
    )
    from models import TINY_OCC_FLOW_INPUT_SHAPE, TinyOccFlowStudent  # type: ignore[no-redef]
    from shadow_guard import SplitConformalEpisodeCalibrator  # type: ignore[no-redef]


SHADOW_ONLY = True
CMD_VEL_AUTHORITY = False
EXPECTED_FRAME_CHANNELS = TINY_OCC_FLOW_INPUT_SHAPE[1] // HISTORY_FRAMES


@dataclass(frozen=True)
class LossWeights:
    occupancy: float = 1.0
    flow: float = 0.80
    dynamic: float = 0.65
    uncertainty: float = 0.20
    trajectory: float = 0.65


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _validate_contract_dimensions() -> None:
    frame_channels = len(TRIBEV_CHANNEL_NAMES)
    if frame_channels != EXPECTED_FRAME_CHANNELS:
        raise RuntimeError(
            "dataset/model contract mismatch: "
            f"{HISTORY_FRAMES} frames x {frame_channels} channels = "
            f"{HISTORY_FRAMES * frame_channels}, model requires "
            f"{TINY_OCC_FLOW_INPUT_SHAPE[1]}"
        )


def _episode_tensors(
    ref: EpisodeRef,
    device: torch.device,
) -> tuple[Tensor, dict[str, Tensor], dict[str, Any]]:
    record = load_episode(ref.path, validate=True)
    arrays = record["arrays"]
    input_array = arrays["tribev_input"]
    expected = (
        HISTORY_FRAMES,
        EXPECTED_FRAME_CHANNELS,
        TINY_OCC_FLOW_INPUT_SHAPE[2],
        TINY_OCC_FLOW_INPUT_SHAPE[3],
    )
    if input_array.shape != expected:
        raise RuntimeError(
            f"{ref.episode_id}: tribev_input={input_array.shape}, expected={expected}"
        )
    model_input = flatten_tribev_history(input_array)[None]
    inputs = torch.from_numpy(model_input).to(device=device, dtype=torch.float32)
    targets = {
        "occupancy": torch.from_numpy(arrays["future_occupancy"])[None].to(
            device=device,
            dtype=torch.float32,
        ),
        "flow": torch.from_numpy(
            arrays["future_flow_m"].reshape(6, 64, 64)
        )[None].to(device=device, dtype=torch.float32),
        "dynamic": torch.from_numpy(arrays["dynamic_mask"])[None].to(
            device=device,
            dtype=torch.float32,
        ),
        "uncertainty": torch.from_numpy(arrays["uncertainty_target"])[None].to(
            device=device,
            dtype=torch.float32,
        ),
        "trajectory": torch.from_numpy(arrays["trajectory_soft_labels"])[None].to(
            device=device,
            dtype=torch.float32,
        ),
    }
    metadata = {
        "episode_id": ref.episode_id,
        "session_id": ref.session_id,
        "scenario_id": ref.scenario_id,
        "source_kind": ref.source_kind,
    }
    return inputs, targets, metadata


def _batch_tensors(
    refs: Sequence[EpisodeRef],
    device: torch.device,
) -> tuple[Tensor, dict[str, Tensor]]:
    rows = [_episode_tensors(ref, device) for ref in refs]
    inputs = torch.cat([row[0] for row in rows], dim=0)
    targets = {
        name: torch.cat([row[1][name] for row in rows], dim=0)
        for name in rows[0][1]
    }
    return inputs, targets


def _positive_weight(target: Tensor, maximum: float = 24.0) -> Tensor:
    positives = target.sum()
    negatives = target.numel() - positives
    ratio = negatives / torch.clamp(positives, min=1.0)
    return torch.clamp(ratio.detach(), min=1.0, max=maximum)


def _loss_terms(
    outputs: Any,
    targets: dict[str, Tensor],
    weights: LossWeights,
) -> tuple[Tensor, dict[str, Tensor]]:
    occupancy_loss = functional.binary_cross_entropy_with_logits(
        outputs.future_occupancy,
        targets["occupancy"],
        pos_weight=_positive_weight(targets["occupancy"]),
    )

    dynamic_logits = outputs.dynamic_uncertainty[:, :3]
    uncertainty_logits = outputs.dynamic_uncertainty[:, 3:]
    dynamic_loss = functional.binary_cross_entropy_with_logits(
        dynamic_logits,
        targets["dynamic"],
        pos_weight=_positive_weight(targets["dynamic"]),
    )
    dynamic_probability = torch.sigmoid(dynamic_logits)
    dice_numerator = 2.0 * (
        dynamic_probability * targets["dynamic"]
    ).sum(dim=(2, 3))
    dice_denominator = (
        dynamic_probability.sum(dim=(2, 3))
        + targets["dynamic"].sum(dim=(2, 3))
    )
    dynamic_dice_loss = 1.0 - (
        (dice_numerator + 1.0) / (dice_denominator + 1.0)
    ).mean()
    dynamic_loss = dynamic_loss + 0.5 * dynamic_dice_loss
    uncertainty_loss = functional.binary_cross_entropy_with_logits(
        uncertainty_logits,
        targets["uncertainty"],
    )

    flow_target = functional.interpolate(
        targets["flow"],
        size=outputs.flow.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    dynamic_half = functional.interpolate(
        targets["dynamic"],
        size=outputs.flow.shape[-2:],
        mode="nearest",
    )
    flow_mask = dynamic_half.repeat_interleave(2, dim=1)
    flow_difference = functional.smooth_l1_loss(
        outputs.flow,
        flow_target,
        reduction="none",
        beta=0.05,
    )
    active = flow_mask.sum()
    if float(active.detach().cpu()) > 0.0:
        flow_loss = (flow_difference * flow_mask).sum() / active
    else:
        flow_loss = flow_difference.mean() * 0.0

    trajectory_log_probability = functional.log_softmax(
        outputs.trajectory_logits,
        dim=1,
    )
    trajectory_loss = -(
        targets["trajectory"] * trajectory_log_probability
    ).sum(dim=1).mean()

    terms = {
        "occupancy": occupancy_loss,
        "flow": flow_loss,
        "dynamic": dynamic_loss,
        "uncertainty": uncertainty_loss,
        "trajectory": trajectory_loss,
    }
    total = (
        weights.occupancy * occupancy_loss
        + weights.flow * flow_loss
        + weights.dynamic * dynamic_loss
        + weights.uncertainty * uncertainty_loss
        + weights.trajectory * trajectory_loss
    )
    return total, terms


def _ordered_refs(refs: Sequence[EpisodeRef], seed: int, epoch: int) -> list[EpisodeRef]:
    order = list(refs)
    random.Random(seed + epoch * 1009).shuffle(order)
    return order


def _run_epoch(
    model: TinyOccFlowStudent,
    refs: Sequence[EpisodeRef],
    *,
    device: torch.device,
    weights: LossWeights,
    optimizer: AdamW | None,
    seed: int,
    epoch: int,
    batch_size: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "total": 0.0,
        "occupancy": 0.0,
        "flow": 0.0,
        "dynamic": 0.0,
        "uncertainty": 0.0,
        "trajectory": 0.0,
    }
    count = 0
    ordered = _ordered_refs(refs, seed, epoch) if training else list(refs)
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for start in range(0, len(ordered), batch_size):
            batch_refs = ordered[start : start + batch_size]
            inputs, targets = _batch_tensors(batch_refs, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss, terms = _loss_terms(outputs, targets, weights)
            if not torch.isfinite(loss):
                episode_ids = ",".join(ref.episode_id for ref in batch_refs)
                raise RuntimeError(f"non-finite loss for episodes {episode_ids}")
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            batch_count = len(batch_refs)
            totals["total"] += float(loss.detach().cpu()) * batch_count
            for name, value in terms.items():
                totals[name] += float(value.detach().cpu()) * batch_count
            count += batch_count
    if not count:
        raise RuntimeError("split contains no episodes")
    return {name: value / count for name, value in totals.items()}


def _flatten_flow_target(targets: dict[str, Tensor], output_shape: Sequence[int]) -> np.ndarray:
    resized = functional.interpolate(
        targets["flow"],
        size=tuple(output_shape),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0].detach().cpu().numpy()


def evaluate(
    model: TinyOccFlowStudent,
    refs: Sequence[EpisodeRef],
    *,
    device: torch.device,
    weights: LossWeights,
    trajectory_temperature: float = 1.0,
) -> tuple[dict[str, Any], list[float]]:
    if not math.isfinite(trajectory_temperature) or trajectory_temperature <= 0.0:
        raise ValueError("trajectory_temperature must be finite and positive")
    model.eval()
    losses: list[float] = []
    occupancy_predictions: list[np.ndarray] = []
    occupancy_targets: list[np.ndarray] = []
    dynamic_predictions: list[np.ndarray] = []
    dynamic_targets: list[np.ndarray] = []
    uncertainty_predictions: list[np.ndarray] = []
    uncertainty_targets: list[np.ndarray] = []
    flow_predictions: list[np.ndarray] = []
    flow_targets: list[np.ndarray] = []
    flow_masks: list[np.ndarray] = []
    trajectory_rows: list[dict[str, Any]] = []
    confidence_rows: list[float] = []
    correctness_rows: list[float] = []
    nonconformity: list[float] = []

    with torch.inference_mode():
        for ref in refs:
            inputs, targets, metadata = _episode_tensors(ref, device)
            outputs = model(inputs)
            total, _ = _loss_terms(outputs, targets, weights)
            losses.append(float(total.cpu()))

            occupancy_logits = outputs.future_occupancy[0].cpu().numpy()
            occupancy_target = targets["occupancy"][0].cpu().numpy()
            occupancy_predictions.append(occupancy_logits)
            occupancy_targets.append(occupancy_target)
            occupancy_probability = 1.0 / (
                1.0 + np.exp(-np.clip(occupancy_logits, -40.0, 40.0))
            )
            per_episode_error = np.abs(
                occupancy_probability - occupancy_target
            )
            nonconformity.append(float(np.quantile(per_episode_error, 0.95)))

            dynamic_predictions.append(
                outputs.dynamic_uncertainty[0, :3].cpu().numpy()
            )
            dynamic_targets.append(targets["dynamic"][0].cpu().numpy())
            uncertainty_predictions.append(
                outputs.dynamic_uncertainty[0, 3:].cpu().numpy()
            )
            uncertainty_targets.append(targets["uncertainty"][0].cpu().numpy())

            flow_predictions.append(outputs.flow[0].cpu().numpy())
            flow_targets.append(
                _flatten_flow_target(targets, outputs.flow.shape[-2:])
            )
            dynamic_half = functional.interpolate(
                targets["dynamic"],
                size=outputs.flow.shape[-2:],
                mode="nearest",
            )[0].cpu().numpy()
            flow_masks.append(dynamic_half)

            trajectory_logits = (
                outputs.trajectory_logits[0].cpu().numpy()
                / trajectory_temperature
            )
            trajectory_target = targets["trajectory"][0].cpu().numpy()
            row = trajectory_distribution_metrics(
                trajectory_logits,
                trajectory_target,
            )
            row["episode_id"] = metadata["episode_id"]
            trajectory_rows.append(row)
            shifted = trajectory_logits - trajectory_logits.max()
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum()
            confidence_rows.append(float(probabilities.max()))
            correctness_rows.append(float(row["top1_agreement"]))

    occupancy_prediction = np.stack(occupancy_predictions, axis=0)
    occupancy_target = np.stack(occupancy_targets, axis=0)
    dynamic_prediction = np.stack(dynamic_predictions, axis=0)
    dynamic_target = np.stack(dynamic_targets, axis=0)
    uncertainty_prediction = np.stack(uncertainty_predictions, axis=0)
    uncertainty_target = np.stack(uncertainty_targets, axis=0)
    flow_prediction = np.stack(flow_predictions, axis=0)
    flow_target = np.stack(flow_targets, axis=0)
    flow_mask = np.stack(flow_masks, axis=0)

    occupancy_rows = [
        {
            "horizon_index": horizon,
            **binary_occupancy_metrics(
                occupancy_prediction[:, horizon],
                occupancy_target[:, horizon],
                from_logits=True,
            ),
        }
        for horizon in range(occupancy_prediction.shape[1])
    ]
    occupancy = {
        "horizons": occupancy_rows,
        "mean_iou": float(np.mean([row["iou"] for row in occupancy_rows])),
        "mean_f1": float(np.mean([row["f1"] for row in occupancy_rows])),
    }
    dynamic = binary_occupancy_metrics(
        dynamic_prediction,
        dynamic_target,
        from_logits=True,
    )
    uncertainty_probability = 1.0 / (
        1.0 + np.exp(-np.clip(uncertainty_prediction, -40.0, 40.0))
    )
    flow = flow_endpoint_error(
        flow_prediction,
        flow_target,
        valid_mask=flow_mask > 0.5,
    )
    trajectory_top1 = np.mean(
        [float(row["top1_agreement"]) for row in trajectory_rows]
    )
    return (
        {
            "episodes": len(refs),
            "mean_loss": float(np.mean(losses)),
            "occupancy": occupancy,
            "dynamic": dynamic,
            "flow": flow,
            "uncertainty_mae": float(
                np.mean(np.abs(uncertainty_probability - uncertainty_target))
            ),
            "trajectory": {
                "top1_agreement": float(trajectory_top1),
                "mean_kl": float(
                    np.mean(
                        [
                            float(row["kl_target_to_prediction"])
                            for row in trajectory_rows
                        ]
                    )
                ),
                "ece": expected_calibration_error(
                    np.asarray(confidence_rows),
                    np.asarray(correctness_rows),
                ),
            },
        },
        nonconformity,
    )


def fit_trajectory_temperature(
    model: TinyOccFlowStudent,
    refs: Sequence[EpisodeRef],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Fit one deterministic temperature on the independent calibration split."""

    logits_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for ref in refs:
            inputs, targets, _ = _episode_tensors(ref, device)
            outputs = model(inputs)
            logits_rows.append(outputs.trajectory_logits[0].cpu().numpy())
            target_rows.append(targets["trajectory"][0].cpu().numpy())
    logits = np.stack(logits_rows).astype(np.float64)
    targets = np.stack(target_rows).astype(np.float64)
    candidates = np.exp(np.linspace(math.log(0.25), math.log(4.0), 241))
    losses: list[float] = []
    for temperature in candidates:
        scaled = logits / float(temperature)
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        log_probability = shifted - np.log(
            np.exp(shifted).sum(axis=1, keepdims=True)
        )
        losses.append(float(-(targets * log_probability).sum(axis=1).mean()))
    best_index = int(np.argmin(losses))
    return {
        "method": "deterministic_log_grid_soft_cross_entropy",
        "fit_split": "calibration",
        "episode_count": len(refs),
        "temperature": float(candidates[best_index]),
        "soft_cross_entropy": float(losses[best_index]),
        "search_min": float(candidates[0]),
        "search_max": float(candidates[-1]),
        "search_count": len(candidates),
    }


def _save_checkpoint(
    path: Path,
    model: TinyOccFlowStudent,
    *,
    epoch: int,
    seed: int,
    split_summary: dict[str, Any],
    validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": int(epoch),
            "seed": int(seed),
            "validation_loss": float(validation_loss),
            "split_summary": split_summary,
            "shadow_only": SHADOW_ONLY,
            "cmd_vel_authority": CMD_VEL_AUTHORITY,
        },
        temporary,
    )
    os.replace(temporary, path)


def train(
    dataset_root: Path,
    output_directory: Path,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device_name: str,
    patience: int,
    batch_size: int,
) -> dict[str, Any]:
    _validate_contract_dimensions()
    if epochs <= 0 or patience <= 0 or batch_size <= 0:
        raise ValueError("epochs, patience, and batch_size must be positive")
    _seed_everything(seed)
    device = torch.device(device_name)
    refs = build_episode_refs(dataset_root)
    splits = split_episode_refs(refs, seed=seed)
    for name in SPLIT_NAMES:
        if not splits[name]:
            raise RuntimeError(
                f"{name} split is empty; generate more independent sessions"
            )
    split_summary = summarize_splits(splits)
    model = TinyOccFlowStudent().to(device)
    weights = LossWeights()
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "tiny_occ_flow_best.pt"
    history: list[dict[str, Any]] = []
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_losses = _run_epoch(
            model,
            splits["train"],
            device=device,
            weights=weights,
            optimizer=optimizer,
            seed=seed,
            epoch=epoch,
            batch_size=batch_size,
        )
        validation_losses = _run_epoch(
            model,
            splits["validation"],
            device=device,
            weights=weights,
            optimizer=None,
            seed=seed,
            epoch=epoch,
            batch_size=batch_size,
        )
        row = {
            "epoch": epoch,
            "train": train_losses,
            "validation": validation_losses,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        current = validation_losses["total"]
        if current < best_validation - 1e-6:
            best_validation = current
            best_epoch = epoch
            stale_epochs = 0
            _save_checkpoint(
                checkpoint_path,
                model,
                epoch=epoch,
                seed=seed,
                split_summary=split_summary,
                validation_loss=current,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    trajectory_calibration = fit_trajectory_temperature(
        model,
        splits["calibration"],
        device=device,
    )
    trajectory_temperature = float(trajectory_calibration["temperature"])
    calibration_metrics, calibration_scores = evaluate(
        model,
        splits["calibration"],
        device=device,
        weights=weights,
        trajectory_temperature=trajectory_temperature,
    )
    test_metrics, test_scores = evaluate(
        model,
        splits["test"],
        device=device,
        weights=weights,
        trajectory_temperature=trajectory_temperature,
    )
    calibrator = SplitConformalEpisodeCalibrator(alpha=0.10)
    conformal = calibrator.fit(calibration_scores)
    test_p_values = [
        calibrator.calibration_p_value(score)["p_value"]
        for score in test_scores
    ]
    conformal["test_p_value_summary"] = {
        "minimum": float(np.min(test_p_values)),
        "median": float(np.median(test_p_values)),
        "maximum": float(np.max(test_p_values)),
    }
    q_hat = conformal.get("q_hat")
    empirical_coverage = (
        float(np.mean(np.asarray(test_scores) <= float(q_hat)))
        if q_hat is not None
        else None
    )
    conformal["test_empirical_coverage"] = empirical_coverage
    conformal["test_nominal_coverage"] = 0.90
    conformal["test_absolute_coverage_gap"] = (
        abs(empirical_coverage - 0.90)
        if empirical_coverage is not None
        else None
    )

    dataset_manifest_path = dataset_root / "dataset_manifest.json"
    if not dataset_manifest_path.is_file():
        raise RuntimeError(
            "dataset_manifest.json is required for auditable v5 training"
        )
    dataset_manifest = json.loads(
        dataset_manifest_path.read_text(encoding="utf-8")
    )
    if int(dataset_manifest.get("episode_count", -1)) != len(refs):
        raise RuntimeError("dataset manifest episode_count does not match corpus")

    report = {
        "schema_version": "x5-tribev-flow-training/1.1",
        "created_at_unix_s": time.time(),
        "elapsed_s": time.perf_counter() - started,
        "seed": seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "dataset_root": str(dataset_root.resolve()),
        "episode_count": len(refs),
        "dataset_manifest": {
            "path": str(dataset_manifest_path.resolve()),
            "sha256": _sha256_file(dataset_manifest_path),
            "schema_version": dataset_manifest.get("schema_version"),
            "generator": dataset_manifest.get("generator"),
            "duplicate_input_count": dataset_manifest.get(
                "duplicate_input_count"
            ),
        },
        "split_summary": split_summary,
        "input_contract": {
            "shape": list(TINY_OCC_FLOW_INPUT_SHAPE),
            "history_frames": HISTORY_FRAMES,
            "frame_channels": list(TRIBEV_CHANNEL_NAMES),
            "episode_storage_order": "oldest_to_newest",
            "model_history_order": "newest_to_oldest_t0_first",
        },
        "loss_weights": asdict(weights),
        "hyperparameters": {
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "patience": patience,
            "batch_size": batch_size,
        },
        "history": history,
        "calibration_metrics": calibration_metrics,
        "test_metrics": test_metrics,
        "trajectory_calibration": trajectory_calibration,
        "conformal": conformal,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256_file(checkpoint_path),
        },
        "shadow_only": SHADOW_ONLY,
        "cmd_vel_authority": CMD_VEL_AUTHORITY,
        "claim_boundary": (
            "Metrics from synthetic episodes validate only the deterministic "
            "training/export pipeline. Real sensor and navigation claims require "
            "independent captured episodes and later X5 shadow validation."
        ),
    }
    report_path = output_directory / "training_report.json"
    _atomic_json(report_path, report)
    report["report"] = {
        "path": str(report_path.resolve()),
        "sha256": _sha256_file(report_path),
    }
    return report


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    report = train(
        args.dataset,
        args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device_name=args.device,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "checkpoint": report["checkpoint"],
                "report": report["report"],
                "test_metrics": report["test_metrics"],
                "shadow_only": True,
                "cmd_vel_authority": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
