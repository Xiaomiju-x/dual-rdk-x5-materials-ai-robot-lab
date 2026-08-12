#!/usr/bin/env python3
"""Procedurally pretrain CamSemLite and emit a truthful offline report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch.optim import AdamW

if __package__:
    from .camsem_synthetic import (
        QUALITY_CLASS_NAMES,
        SEMANTIC_CLASS_NAMES,
        generate_camsem_sample,
    )
    from .models import CamSemLite
else:
    from camsem_synthetic import (  # type: ignore[no-redef]
        QUALITY_CLASS_NAMES,
        SEMANTIC_CLASS_NAMES,
        generate_camsem_sample,
    )
    from models import CamSemLite  # type: ignore[no-redef]


SHADOW_ONLY = True
CMD_VEL_AUTHORITY = False


def _sha256(path: Path) -> str:
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


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _tensors(index: int, seed: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    sample = generate_camsem_sample(index, seed=seed)
    image = (
        torch.from_numpy(sample.rgb_u8)
        .permute(2, 0, 1)[None]
        .to(device=device, dtype=torch.float32)
        / 255.0
    )
    mask = torch.from_numpy(sample.semantic_mask.astype(np.int64))[None].to(device)
    quality = torch.tensor([sample.quality_label], dtype=torch.int64, device=device)
    return image, mask, quality


def _loss(
    outputs: Any,
    mask: torch.Tensor,
    quality: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    class_weights = torch.tensor(
        [0.35, 0.55, 1.20, 1.60, 2.20, 1.40],
        dtype=torch.float32,
        device=mask.device,
    )
    semantic = functional.cross_entropy(
        outputs.semantic_logits,
        mask,
        weight=class_weights,
    )
    quality_loss = functional.cross_entropy(outputs.quality_logits, quality)
    total = semantic + 0.35 * quality_loss
    return total, {"semantic": semantic, "quality": quality_loss}


def _run_epoch(
    model: CamSemLite,
    indices: Sequence[int],
    *,
    seed: int,
    device: torch.device,
    optimizer: AdamW | None,
    epoch: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    ordered = list(indices)
    if training:
        random.Random(seed + epoch * 997).shuffle(ordered)
    totals = {"total": 0.0, "semantic": 0.0, "quality": 0.0}
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for index in ordered:
            image, mask, quality = _tensors(index, seed, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(image)
            total, terms = _loss(outputs, mask, quality)
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite loss at synthetic index {index}")
            if optimizer is not None:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            totals["total"] += float(total.detach().cpu())
            for name, value in terms.items():
                totals[name] += float(value.detach().cpu())
    if not ordered:
        raise RuntimeError("empty split")
    return {name: value / len(ordered) for name, value in totals.items()}


def _evaluate(
    model: CamSemLite,
    indices: Sequence[int],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    intersections = np.zeros(len(SEMANTIC_CLASS_NAMES), dtype=np.int64)
    unions = np.zeros(len(SEMANTIC_CLASS_NAMES), dtype=np.int64)
    quality_correct = 0
    with torch.inference_mode():
        model.eval()
        for index in indices:
            image, mask, quality = _tensors(index, seed, device)
            outputs = model(image)
            prediction = outputs.semantic_logits.argmax(dim=1)
            quality_prediction = outputs.quality_logits.argmax(dim=1)
            quality_correct += int((quality_prediction == quality).item())
            pred_np = prediction[0].cpu().numpy()
            mask_np = mask[0].cpu().numpy()
            for class_index in range(len(SEMANTIC_CLASS_NAMES)):
                chosen = pred_np == class_index
                truth = mask_np == class_index
                intersections[class_index] += int(np.logical_and(chosen, truth).sum())
                unions[class_index] += int(np.logical_or(chosen, truth).sum())
    class_iou = {
        name: (
            float(intersections[index] / unions[index])
            if unions[index]
            else 1.0
        )
        for index, name in enumerate(SEMANTIC_CLASS_NAMES)
    }
    return {
        "samples": len(indices),
        "class_iou": class_iou,
        "mean_iou": float(np.mean(list(class_iou.values()))),
        "quality_accuracy": float(quality_correct / len(indices)),
    }


def train(
    output_directory: Path,
    *,
    train_samples: int,
    calibration_samples: int,
    test_samples: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    if min(train_samples, calibration_samples, test_samples, epochs, patience) <= 0:
        raise ValueError("sample counts, epochs, and patience must be positive")
    _seed(seed)
    device = torch.device(device_name)
    train_indices = list(range(0, train_samples))
    calibration_indices = list(
        range(train_samples, train_samples + calibration_samples)
    )
    test_indices = list(
        range(
            train_samples + calibration_samples,
            train_samples + calibration_samples + test_samples,
        )
    )
    model = CamSemLite().to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "cam_sem_lite_best.pt"
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        training = _run_epoch(
            model,
            train_indices,
            seed=seed,
            device=device,
            optimizer=optimizer,
            epoch=epoch,
        )
        calibration = _run_epoch(
            model,
            calibration_indices,
            seed=seed,
            device=device,
            optimizer=None,
            epoch=epoch,
        )
        row = {"epoch": epoch, "train": training, "calibration": calibration}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if calibration["total"] < best_loss - 1e-6:
            best_loss = calibration["total"]
            best_epoch = epoch
            stale = 0
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                    "synthetic_pretrain_only": True,
                    "shadow_only": True,
                    "cmd_vel_authority": False,
                },
                temporary,
            )
            os.replace(temporary, checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    calibration_metrics = _evaluate(
        model,
        calibration_indices,
        seed=seed,
        device=device,
    )
    test_metrics = _evaluate(
        model,
        test_indices,
        seed=seed,
        device=device,
    )
    report = {
        "schema_version": "cam-sem-lite-training/1.0",
        "elapsed_s": time.perf_counter() - started,
        "seed": seed,
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "synthetic_split": {
            "train": train_samples,
            "calibration": calibration_samples,
            "test": test_samples,
            "index_ranges_do_not_overlap": True,
        },
        "semantic_classes": list(SEMANTIC_CLASS_NAMES),
        "quality_classes": list(QUALITY_CLASS_NAMES),
        "hyperparameters": {
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "learning_rate": learning_rate,
            "patience": patience,
        },
        "history": history,
        "calibration_metrics": calibration_metrics,
        "test_metrics": test_metrics,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256(checkpoint_path),
        },
        "synthetic_pretrain_only": True,
        "board_runtime_eligible": False,
        "shadow_only": SHADOW_ONLY,
        "cmd_vel_authority": CMD_VEL_AUTHORITY,
        "claim_boundary": (
            "Procedural metrics prove only that CamSemLite, export, and BPU "
            "conversion plumbing learn a controlled fixture. Real 4K semantic "
            "claims require provenance-preserved camera captures, labels or "
            "teacher distillation, and independent real-session validation."
        ),
    }
    report_path = output_directory / "training_report.json"
    _atomic_json(report_path, report)
    report["report"] = {
        "path": str(report_path.resolve()),
        "sha256": _sha256(report_path),
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=480)
    parser.add_argument("--calibration-samples", type=int, default=120)
    parser.add_argument("--test-samples", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260734)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args(argv)
    report = train(
        args.output,
        train_samples=args.train_samples,
        calibration_samples=args.calibration_samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "checkpoint": report["checkpoint"],
                "report": report["report"],
                "test_metrics": report["test_metrics"],
                "synthetic_pretrain_only": True,
                "board_runtime_eligible": False,
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
