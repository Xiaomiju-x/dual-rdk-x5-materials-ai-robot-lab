#!/usr/bin/env python3
"""Train shadow-only Tiny-ACT or an action-conditioned temporal world model."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cloud_common import TRUTH, load_yaml, sha256_file, utc_now, write_json


@dataclass(frozen=True)
class Row:
    parent: str
    episode: str
    timestamp: float
    state: list[float]
    action: list[float]
    stage: str


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def load_rows(path: Path) -> dict[str, list[Row]]:
    episodes: dict[str, list[Row]] = defaultdict(list)
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        episode = str(value.get("episode_id") or "").strip()
        parent = str(value.get("parent_episode_id") or episode).strip()
        state = [float(item) for item in value["observation_state"]]
        action = [float(item) for item in value["action"]]
        if not episode or not parent or len(state) != len(action):
            raise ValueError(f"{path}:{line_no}: invalid episode/state/action")
        if not all(math.isfinite(item) for item in state + action):
            raise ValueError(f"{path}:{line_no}: non-finite state/action")
        timestamp = float(value.get("timestamp", value.get("t", line_no)))
        episodes[episode].append(
            Row(
                parent=parent,
                episode=episode,
                timestamp=timestamp,
                state=state,
                action=action,
                stage=str(value.get("stage") or "UNKNOWN"),
            )
        )
    for rows in episodes.values():
        rows.sort(key=lambda item: item.timestamp)
    if not episodes:
        raise RuntimeError("no training episodes")
    dims = {(len(row.state), len(row.action)) for rows in episodes.values() for row in rows}
    if len(dims) != 1:
        raise RuntimeError(f"inconsistent dimensions: {sorted(dims)}")
    return dict(episodes)


class WindowDataset(Dataset):
    def __init__(
        self,
        episodes: dict[str, list[Row]],
        episode_ids: list[str],
        *,
        model_kind: str,
        history_len: int,
        chunk_len: int,
        stage_to_id: dict[str, int],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ) -> None:
        self.model_kind = model_kind
        self.history_len = history_len
        self.chunk_len = chunk_len
        self.stage_to_id = stage_to_id
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.samples: list[tuple[str, int]] = []
        self.episodes = episodes
        for episode_id in episode_ids:
            rows = episodes[episode_id]
            required_future = chunk_len if model_kind == "tiny_act" else 1
            for index in range(history_len - 1, len(rows) - required_future):
                self.samples.append((episode_id, index))
        if not self.samples:
            raise RuntimeError(f"no windows for {model_kind}; increase episode length or reduce history/chunk")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_id, cursor = self.samples[index]
        rows = self.episodes[episode_id]
        history = rows[cursor - self.history_len + 1 : cursor + 1]
        states = (np.asarray([row.state for row in history], dtype=np.float32) - self.state_mean) / self.state_std
        actions_in = (np.asarray([row.action for row in history], dtype=np.float32) - self.action_mean) / self.action_std
        if self.model_kind == "tiny_act":
            future = rows[cursor + 1 : cursor + 1 + self.chunk_len]
            actions = (np.asarray([row.action for row in future], dtype=np.float32) - self.action_mean) / self.action_std
            return {
                "states": torch.from_numpy(states),
                "actions_in": torch.from_numpy(actions_in),
                "target": torch.from_numpy(actions),
            }
        target_row = rows[cursor + 1]
        next_state = (np.asarray(target_row.state, dtype=np.float32) - self.state_mean) / self.state_std
        return {
            "states": torch.from_numpy(states),
            "actions_in": torch.from_numpy(actions_in),
            "target": torch.from_numpy(next_state),
            "stage": torch.tensor(self.stage_to_id[target_row.stage], dtype=torch.long),
        }


class TinyACT(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, history_len: int, chunk_len: int, config: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden_dim"])
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.input = nn.Linear(state_dim + action_dim, hidden)
        self.position = nn.Parameter(torch.zeros(1, history_len, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(config["transformer_heads"]),
            dim_feedforward=hidden * 4,
            dropout=float(config["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(config["transformer_layers"]))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, chunk_len * action_dim))

    def forward(self, states: torch.Tensor, actions_in: torch.Tensor) -> torch.Tensor:
        latent = self.input(torch.cat([states, actions_in], dim=-1)) + self.position[:, : states.shape[1]]
        encoded = self.encoder(latent)
        return self.head(encoded[:, -1]).reshape(-1, self.chunk_len, self.action_dim)


class TemporalWorldModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, stage_count: int, config: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden_dim"])
        self.rnn = nn.GRU(
            state_dim + action_dim,
            hidden,
            num_layers=int(config["recurrent_layers"]),
            dropout=float(config["dropout"]) if int(config["recurrent_layers"]) > 1 else 0.0,
            batch_first=True,
        )
        self.state_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, state_dim))
        self.stage_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, stage_count))

    def forward(self, states: torch.Tensor, actions_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.rnn(torch.cat([states, actions_in], dim=-1))
        final = encoded[:, -1]
        return self.state_head(final), self.stage_head(final)


def split_episodes(episodes: dict[str, list[Row]], seed: int, val_fraction: float) -> tuple[list[str], list[str]]:
    parents = sorted({rows[0].parent for rows in episodes.values()})
    val_parents = {parent for parent in parents if stable_fraction(parent, seed) < val_fraction}
    if not val_parents:
        val_parents.add(parents[-1])
    if val_parents == set(parents):
        val_parents.remove(parents[0])
    train = sorted(ep for ep, rows in episodes.items() if rows[0].parent not in val_parents)
    val = sorted(ep for ep, rows in episodes.items() if rows[0].parent in val_parents)
    if not train or not val:
        raise RuntimeError("episode-level split produced an empty partition")
    return train, val


def normalization(episodes: dict[str, list[Row]], train_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = np.asarray([row.state for ep in train_ids for row in episodes[ep]], dtype=np.float32)
    actions = np.asarray([row.action for ep in train_ids for row in episodes[ep]], dtype=np.float32)
    state_std = np.maximum(states.std(axis=0), 1e-5)
    action_std = np.maximum(actions.std(axis=0), 1e-5)
    return states.mean(axis=0), state_std, actions.mean(axis=0), action_std


def make_loaders(
    episodes: dict[str, list[Row]],
    model_kind: str,
    config: dict[str, Any],
    seed: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    train_ids, val_ids = split_episodes(episodes, int(config["dataset"]["split_seed"]) + seed, float(config["dataset"]["validation_fraction"]))
    state_mean, state_std, action_mean, action_std = normalization(episodes, train_ids)
    stages = sorted({row.stage for rows in episodes.values() for row in rows})
    stage_to_id = {stage: index for index, stage in enumerate(stages)}
    model_cfg = config["tiny_act"] if model_kind == "tiny_act" else config["world_model"]
    history_len = int(model_cfg["history_len"])
    chunk_len = int(config["tiny_act"]["chunk_len"]) if model_kind == "tiny_act" else 1
    kwargs = dict(
        episodes=episodes,
        model_kind=model_kind,
        history_len=history_len,
        chunk_len=chunk_len,
        stage_to_id=stage_to_id,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
    )
    train_ds = WindowDataset(episode_ids=train_ids, **kwargs)
    val_ds = WindowDataset(episode_ids=val_ids, **kwargs)
    workers = int(config["training"]["num_workers"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    metadata = {
        "train_episode_ids": train_ids,
        "val_episode_ids": val_ids,
        "parent_overlap": sorted(
            {episodes[ep][0].parent for ep in train_ids} & {episodes[ep][0].parent for ep in val_ids}
        ),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "stage_to_id": stage_to_id,
        "normalization": {
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
            "action_mean": action_mean.tolist(),
            "action_std": action_std.tolist(),
        },
    }
    return train_loader, val_loader, metadata


def epoch_loss(
    model: nn.Module,
    loader: DataLoader,
    model_kind: str,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    accumulation: int,
    stage_weight: float,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_stage_correct = 0
    total_samples = 0
    if optimizer:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, 1):
        states = batch["states"].to(device, non_blocking=True)
        actions_in = batch["actions_in"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            if model_kind == "tiny_act":
                prediction = model(states, actions_in)
                loss = nn.functional.smooth_l1_loss(prediction, target)
                stage_correct = 0
            else:
                prediction, stage_logits = model(states, actions_in)
                stage = batch["stage"].to(device, non_blocking=True)
                loss = nn.functional.smooth_l1_loss(prediction, target) + stage_weight * nn.functional.cross_entropy(stage_logits, stage)
                stage_correct = int((stage_logits.argmax(dim=-1) == stage).sum().item())
            scaled_loss = loss / accumulation
        if training and optimizer and scaler:
            scaler.scale(scaled_loss).backward()
            if step % accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        count = states.shape[0]
        total_loss += float(loss.detach().cpu()) * count
        total_stage_correct += stage_correct
        total_samples += count
    return total_loss / max(total_samples, 1), total_stage_correct / max(total_samples, 1)


def train_attempt(args: argparse.Namespace, config: dict[str, Any], batch_size: int, accumulation: int) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    episodes = load_rows(Path(args.train_jsonl))
    first = next(iter(episodes.values()))[0]
    state_dim, action_dim = len(first.state), len(first.action)
    train_loader, val_loader, metadata = make_loaders(episodes, args.model, config, args.seed, batch_size)
    if metadata["parent_overlap"]:
        raise RuntimeError("parent episode leakage detected")
    device = torch.device("cuda")
    if args.model == "tiny_act":
        model: nn.Module = TinyACT(
            state_dim,
            action_dim,
            int(config["tiny_act"]["history_len"]),
            int(config["tiny_act"]["chunk_len"]),
            config["tiny_act"],
        )
    else:
        model = TemporalWorldModel(state_dim, action_dim, len(metadata["stage_to_id"]), config["world_model"])
    model.to(device)
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
            model, train_loader, args.model, device, amp, optimizer, scaler, accumulation, float(config["world_model"]["stage_loss_weight"])
        )
        with torch.no_grad():
            val_loss, val_stage_acc = epoch_loss(
                model, val_loader, args.model, device, amp, None, None, accumulation, float(config["world_model"]["stage_loss_weight"])
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_stage_accuracy": train_stage_acc if args.model == "world_model" else None,
                "val_stage_accuracy": val_stage_acc if args.model == "world_model" else None,
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    return {
        "model_state_dict": best_state,
        "model_config": {
            "kind": args.model,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "metadata": metadata,
        "history": history,
        "best_val_loss": best_val,
        "batch_size": batch_size,
        "gradient_accumulation": accumulation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    preflight = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or not (preflight.get("dataset") or {}).get("gate_pass"):
        raise SystemExit("real episode preflight gate did not pass")
    train_path = Path(args.train_jsonl).expanduser().resolve()
    if sha256_file(train_path) != (preflight.get("dataset") or {}).get("train_jsonl_sha256"):
        raise SystemExit("training JSONL changed after preflight")

    initial_batch = int(config["training"]["initial_batch_size"])
    minimum_batch = int(config["training"]["minimum_batch_size"])
    accumulation = int(config["training"]["gradient_accumulation"])
    attempts: list[dict[str, Any]] = []
    batch = initial_batch
    result: dict[str, Any] | None = None
    while batch >= minimum_batch:
        try:
            result = train_attempt(args, config, batch, accumulation)
            attempts.append({"batch_size": batch, "gradient_accumulation": accumulation, "status": "PASS"})
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
                    "schema_version": "xrd-shadow-training-failure-v1",
                    "created_at": utc_now(),
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "attempts": attempts,
                    "truthfulness": TRUTH,
                },
            )
            raise
    if result is None:
        write_json(
            out_dir / "failure.json",
            {
                "schema_version": "xrd-shadow-training-failure-v1",
                "created_at": utc_now(),
                "status": "FAILED_CUDA_OOM_AT_MINIMUM_BATCH",
                "attempts": attempts,
                "truthfulness": TRUTH,
            },
        )
        return 3

    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "format": f"xrd-shadow-{args.model}-v1",
            "created_at": utc_now(),
            "seed": args.seed,
            "truthfulness": TRUTH,
            "training_data_sha256": sha256_file(train_path),
            "model_config": result["model_config"],
            "metadata": result["metadata"],
            "model_state_dict": result["model_state_dict"],
        },
        checkpoint_path,
    )
    metrics = {
        "schema_version": "xrd-shadow-training-metrics-v1",
        "created_at": utc_now(),
        "status": "PASS",
        "model": args.model,
        "seed": args.seed,
        "truthfulness": TRUTH,
        "training_data_sha256": sha256_file(train_path),
        "preflight_sha256": sha256_file(Path(args.preflight)),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
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
        "history": result["history"],
    }
    write_json(out_dir / "metrics.json", metrics)
    print(json.dumps({"status": "PASS", "model": args.model, "seed": args.seed, "metrics": str(out_dir / "metrics.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
