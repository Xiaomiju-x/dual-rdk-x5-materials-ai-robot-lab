#!/usr/bin/env python3
"""Train fixture-only Tiny-ACT or temporal world-model shadow candidates."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from cloud_common import (
    FIXTURE_TRUTH,
    load_yaml,
    sha256_file,
    utc_now,
    write_json,
)
from train_models import (
    TinyACT,
    epoch_loss,
    load_rows,
    make_loaders,
)


JOINT_INDICES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIPPER_INDEX = 6


class ResidualTemporalWorldModel(nn.Module):
    """Predict a bounded correction to the persistence state baseline."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        stage_count: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        hidden = int(config["hidden_dim"])
        self.rnn = nn.GRU(
            state_dim + action_dim,
            hidden,
            num_layers=int(config["recurrent_layers"]),
            dropout=(
                float(config["dropout"])
                if int(config["recurrent_layers"]) > 1
                else 0.0
            ),
            batch_first=True,
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, state_dim),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        self.stage_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, stage_count)
        )

    def forward(
        self, states: torch.Tensor, actions_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.rnn(torch.cat([states, actions_in], dim=-1))
        final = encoded[:, -1]
        next_state = states[:, -1] + self.delta_head(final)
        return next_state, self.stage_head(final)


def create_model(
    model_kind: str,
    config: dict[str, Any],
    state_dim: int,
    action_dim: int,
    stage_count: int,
) -> nn.Module:
    if model_kind == "tiny_act":
        return TinyACT(
            state_dim,
            action_dim,
            int(config["tiny_act"]["history_len"]),
            int(config["tiny_act"]["chunk_len"]),
            config["tiny_act"],
        )
    if config["world_model"].get("residual_prediction") is not True:
        raise ValueError("fixture world model must preserve the persistence residual")
    return ResidualTemporalWorldModel(
        state_dim,
        action_dim,
        stage_count,
        config["world_model"],
    )


def fixture_train_attempt(
    args: argparse.Namespace,
    config: dict[str, Any],
    batch_size: int,
    accumulation: int,
) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    episodes = load_rows(Path(args.train_jsonl))
    first = next(iter(episodes.values()))[0]
    state_dim, action_dim = len(first.state), len(first.action)
    train_loader, val_loader, metadata = make_loaders(
        episodes, args.model, config, args.seed, batch_size
    )
    if metadata["parent_overlap"]:
        raise RuntimeError("parent episode leakage detected")
    model = create_model(
        args.model,
        config,
        state_dim,
        action_dim,
        len(metadata["stage_to_id"]),
    ).to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    amp = bool(config["training"]["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history: list[dict[str, Any]] = []
    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss, train_stage_acc = epoch_loss(
            model,
            train_loader,
            args.model,
            torch.device("cuda"),
            amp,
            optimizer,
            scaler,
            accumulation,
            float(config["world_model"]["stage_loss_weight"]),
        )
        with torch.no_grad():
            val_loss, val_stage_acc = epoch_loss(
                model,
                val_loader,
                args.model,
                torch.device("cuda"),
                amp,
                None,
                None,
                accumulation,
                float(config["world_model"]["stage_loss_weight"]),
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_stage_accuracy": (
                    train_stage_acc if args.model == "world_model" else None
                ),
                "val_stage_accuracy": (
                    val_stage_acc if args.model == "world_model" else None
                ),
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    return {
        "model_state_dict": best_state,
        "model_config": {
            "kind": args.model,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "residual_prediction": args.model == "world_model",
        },
        "metadata": metadata,
        "history": history,
        "best_val_loss": best_val,
        "batch_size": batch_size,
        "gradient_accumulation": accumulation,
    }


def validation_metrics(
    args: argparse.Namespace,
    config: dict[str, Any],
    model_state: dict[str, torch.Tensor],
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any], nn.Module]:
    episodes = load_rows(Path(args.train_jsonl))
    first = next(iter(episodes.values()))[0]
    state_dim, action_dim = len(first.state), len(first.action)
    _, val_loader, metadata = make_loaders(
        episodes, args.model, config, args.seed, batch_size
    )
    model = create_model(
        args.model,
        config,
        state_dim,
        action_dim,
        len(metadata["stage_to_id"]),
    )
    model.load_state_dict(model_state)
    model.to("cuda").eval()
    normalization = metadata["normalization"]
    native_mean = np.asarray(
        normalization["action_mean"]
        if args.model == "tiny_act"
        else normalization["state_mean"],
        dtype=np.float32,
    )
    native_std = np.asarray(
        normalization["action_std"]
        if args.model == "tiny_act"
        else normalization["state_std"],
        dtype=np.float32,
    )
    mean = torch.tensor(native_mean, device="cuda")
    std = torch.tensor(native_std, device="cuda")
    learned_joint_sum = 0.0
    baseline_joint_sum = 0.0
    learned_gripper_sum = 0.0
    baseline_gripper_sum = 0.0
    normalized_sum = 0.0
    baseline_normalized_sum = 0.0
    joint_count = 0
    gripper_count = 0
    normalized_count = 0
    stage_correct = 0
    stage_total = 0
    stage_labels: Counter[int] = Counter()
    with torch.no_grad():
        for batch in val_loader:
            states = batch["states"].cuda(non_blocking=True)
            actions_in = batch["actions_in"].cuda(non_blocking=True)
            target = batch["target"].cuda(non_blocking=True)
            if args.model == "tiny_act":
                prediction = model(states, actions_in)
                baseline = actions_in[:, -1:, :].expand_as(target)
            else:
                prediction, stage_logits = model(states, actions_in)
                baseline = states[:, -1, :]
                labels = batch["stage"].cuda(non_blocking=True)
                stage_correct += int(
                    (stage_logits.argmax(dim=-1) == labels).sum().item()
                )
                stage_total += int(labels.numel())
                stage_labels.update(int(value) for value in labels.cpu().tolist())
            learned_native = prediction * std + mean
            baseline_native = baseline * std + mean
            target_native = target * std + mean
            learned_abs = (learned_native - target_native).abs()
            baseline_abs = (baseline_native - target_native).abs()
            learned_joint_sum += float(learned_abs[..., JOINT_INDICES].sum().item())
            baseline_joint_sum += float(
                baseline_abs[..., JOINT_INDICES].sum().item()
            )
            learned_gripper_sum += float(
                learned_abs[..., GRIPPER_INDEX].sum().item()
            )
            baseline_gripper_sum += float(
                baseline_abs[..., GRIPPER_INDEX].sum().item()
            )
            normalized_sum += float((prediction - target).abs().sum().item())
            baseline_normalized_sum += float(
                (baseline - target).abs().sum().item()
            )
            multiplier = math.prod(target.shape[:-1])
            joint_count += multiplier * len(JOINT_INDICES)
            gripper_count += multiplier
            normalized_count += target.numel()

    learned_joint = learned_joint_sum / max(joint_count, 1)
    baseline_joint = baseline_joint_sum / max(joint_count, 1)
    learned_gripper = learned_gripper_sum / max(gripper_count, 1)
    baseline_gripper = baseline_gripper_sum / max(gripper_count, 1)
    learned_normalized = normalized_sum / max(normalized_count, 1)
    baseline_normalized = baseline_normalized_sum / max(normalized_count, 1)
    evaluation: dict[str, Any] = {
        "validation_windows": len(val_loader.dataset),
        "learned_native_joint_mae_deg": learned_joint,
        "persistence_native_joint_mae_deg": baseline_joint,
        "joint_mae_improvement_fraction": (
            (baseline_joint - learned_joint) / baseline_joint
            if baseline_joint > 0
            else 0.0
        ),
        "learned_native_gripper_mae": learned_gripper,
        "persistence_native_gripper_mae": baseline_gripper,
        "learned_normalized_mae": learned_normalized,
        "persistence_normalized_mae": baseline_normalized,
        "normalized_mae_improvement_fraction": (
            (baseline_normalized - learned_normalized) / baseline_normalized
            if baseline_normalized > 0
            else 0.0
        ),
    }
    if args.model == "world_model":
        majority = max(stage_labels.values(), default=0)
        evaluation.update(
            {
                "learned_stage_accuracy": stage_correct / max(stage_total, 1),
                "majority_stage_accuracy": majority / max(stage_total, 1),
            }
        )
    return evaluation, metadata, model.cpu()


def export_onnx(
    model: nn.Module,
    model_kind: str,
    config: dict[str, Any],
    state_dim: int,
    action_dim: int,
    path: Path,
) -> dict[str, Any]:
    model.eval()
    history = int(
        config["tiny_act"]["history_len"]
        if model_kind == "tiny_act"
        else config["world_model"]["history_len"]
    )
    states = torch.zeros(1, history, state_dim)
    actions = torch.zeros(1, history, action_dim)
    output_names = (
        ["action_chunk"]
        if model_kind == "tiny_act"
        else ["next_state", "stage_logits"]
    )
    dynamic_axes: dict[str, dict[int, str]] = {
        "states": {0: "batch"},
        "actions_in": {0: "batch"},
    }
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}
    torch.onnx.export(
        model,
        (states, actions),
        str(path),
        input_names=["states", "actions_in"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    checker = "NOT_RUN"
    try:
        import onnx

        onnx.checker.check_model(onnx.load(str(path)))
        checker = "PASS"
    except ImportError:
        checker = "ONNX_PACKAGE_MISSING"
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "checker": checker,
        "opset": 17,
        "inputs": {
            "states": [1, history, state_dim],
            "actions_in": [1, history, action_dim],
        },
        "outputs": output_names,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["tiny_act", "world_model"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml(Path(args.config).resolve())
    preflight_path = Path(args.preflight).expanduser().resolve()
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    fixture = preflight.get("dataset") or {}
    if (
        preflight.get("status") != "PASS_FIXTURE_ONLY"
        or not fixture.get("gate_pass")
        or fixture.get("real_robot_policy") is not False
    ):
        raise SystemExit("fixture-only preflight gate did not pass")
    train_path = Path(args.train_jsonl).expanduser().resolve()
    if sha256_file(train_path) != fixture.get("train_jsonl_sha256"):
        raise SystemExit("fixture JSONL changed after preflight")
    if args.seed not in [int(value) for value in config["training"]["seeds"]]:
        raise SystemExit("seed is outside the frozen fixture configuration")

    batch = int(config["training"]["initial_batch_size"])
    minimum_batch = int(config["training"]["minimum_batch_size"])
    accumulation = int(config["training"]["gradient_accumulation"])
    attempts: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    while batch >= minimum_batch:
        try:
            result = fixture_train_attempt(args, config, batch, accumulation)
            attempts.append(
                {
                    "batch_size": batch,
                    "gradient_accumulation": accumulation,
                    "status": "PASS",
                }
            )
            break
        except torch.cuda.OutOfMemoryError as exc:
            attempts.append(
                {
                    "batch_size": batch,
                    "gradient_accumulation": accumulation,
                    "status": "CUDA_OOM",
                    "error": str(exc),
                }
            )
            gc.collect()
            torch.cuda.empty_cache()
            if batch == minimum_batch:
                break
            batch = max(minimum_batch, batch // 2)
            accumulation *= 2
        except Exception as exc:
            write_json(
                out_dir / "failure.json",
                {
                    "schema_version": "xrd-fixture-training-failure-v1",
                    "created_at": utc_now(),
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "attempts": attempts,
                    "truthfulness": FIXTURE_TRUTH,
                },
            )
            raise
    if result is None:
        write_json(
            out_dir / "failure.json",
            {
                "schema_version": "xrd-fixture-training-failure-v1",
                "created_at": utc_now(),
                "status": "FAILED_CUDA_OOM_AT_MINIMUM_BATCH",
                "attempts": attempts,
                "truthfulness": FIXTURE_TRUTH,
            },
        )
        return 3

    evaluation, evaluation_metadata, export_model = validation_metrics(
        args, config, result["model_state_dict"], result["batch_size"]
    )
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "format": f"xrd-fixture-shadow-{args.model}-v1",
            "created_at": utc_now(),
            "seed": args.seed,
            "status": "FIXTURE_REPLAY_NOT_REAL_POLICY",
            "truthfulness": FIXTURE_TRUTH,
            "training_data_sha256": sha256_file(train_path),
            "model_config": result["model_config"],
            "metadata": result["metadata"],
            "model_state_dict": result["model_state_dict"],
        },
        checkpoint_path,
    )
    onnx_receipt = export_onnx(
        export_model,
        args.model,
        config,
        int(result["model_config"]["state_dim"]),
        int(result["model_config"]["action_dim"]),
        out_dir / "model.onnx",
    )
    metrics = {
        "schema_version": "xrd-fixture-shadow-training-metrics-v1",
        "created_at": utc_now(),
        "status": "PASS_FIXTURE_ONLY",
        "model": args.model,
        "seed": args.seed,
        "truthfulness": FIXTURE_TRUTH,
        "training_data_sha256": sha256_file(train_path),
        "preflight_sha256": sha256_file(preflight_path),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "onnx": onnx_receipt,
        "attempts": attempts,
        "effective_batch_size": result["batch_size"],
        "effective_gradient_accumulation": result["gradient_accumulation"],
        "best_val_loss": result["best_val_loss"],
        "model_config": result["model_config"],
        "split": {
            "train_episode_ids": result["metadata"]["train_episode_ids"],
            "val_episode_ids": result["metadata"]["val_episode_ids"],
            "parent_overlap": result["metadata"]["parent_overlap"],
        },
        "evaluation": evaluation,
        "stage_to_id": evaluation_metadata["stage_to_id"],
        "history": result["history"],
    }
    write_json(out_dir / "metrics.json", metrics)
    print(
        json.dumps(
            {
                "status": "PASS_FIXTURE_ONLY",
                "model": args.model,
                "seed": args.seed,
                "joint_mae_deg": evaluation["learned_native_joint_mae_deg"],
                "metrics": str(out_dir / "metrics.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
