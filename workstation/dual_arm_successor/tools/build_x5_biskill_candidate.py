#!/usr/bin/env python3
"""Build a fixture-only X5 CPU/BPU shadow candidate for the frozen demos."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workstation.dual_arm_successor.models.x5_biskill_tcn import (  # noqa: E402
    ModelConfig,
    X5BiSkillTCN,
)


TRUTH = {
    "status": "FIXTURE_REPLAY_NOT_REAL_POLICY",
    "provenance": "COMMAND_DERIVED_DIGITAL_TWIN",
    "measured_robot_telemetry": False,
    "synchronized_camera_actions": False,
    "real_robot_policy": False,
    "motion_authority": False,
    "execution_allowed": False,
    "actuator_commands_issued": 0,
}
JOINT_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_episodes(path: Path) -> dict[str, list[dict[str, Any]]]:
    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if any(row.get(key) != expected for key, expected in (
                ("status", TRUTH["status"]),
                ("provenance_state", TRUTH["provenance"]),
                ("motion_authority", False),
                ("execution_allowed", False),
                ("actuator_commands_issued", 0),
            )):
                raise ValueError("fixture truth boundary mismatch")
            episodes[row["episode_id"]].append(row)
    for rows in episodes.values():
        rows.sort(key=lambda item: float(item["timestamp"]))
    return dict(episodes)


def fit_stats(episodes: dict[str, list[dict[str, Any]]], ids: list[str]) -> dict[str, list[float]]:
    states = np.concatenate([
        np.asarray([row["observation_state"] for row in episodes[episode_id]], dtype=np.float32)
        for episode_id in ids
    ])
    actions = np.concatenate([
        np.asarray([row["action"] for row in episodes[episode_id]], dtype=np.float32)
        for episode_id in ids
    ])
    delta = actions - states
    result: dict[str, list[float]] = {}
    for name, values in (("state", states), ("action", actions), ("delta", delta)):
        result[f"{name}_mean"] = values.mean(axis=0).astype(float).tolist()
        result[f"{name}_std"] = np.maximum(values.std(axis=0), 1e-4).astype(float).tolist()
    return result


def episode_features(rows: list[dict[str, Any]], stats: dict[str, list[float]]) -> np.ndarray:
    states = np.asarray([row["observation_state"] for row in rows], dtype=np.float32)
    actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
    delta = actions - states
    parts = []
    for name, values in (("state", states), ("action", actions), ("delta", delta)):
        mean = np.asarray(stats[f"{name}_mean"], dtype=np.float32)
        std = np.asarray(stats[f"{name}_std"], dtype=np.float32)
        parts.append((values - mean) / std)
    task = np.zeros((len(rows), 2), dtype=np.float32)
    task[:, 0] = [row["task"] == "single_arm_visual_redundancy" for row in rows]
    task[:, 1] = 1.0 - task[:, 0]
    activity = np.stack([
        np.clip(np.abs(delta[:, :6]).mean(axis=1) / 5.0, 0.0, 1.0),
        np.clip(np.abs(delta[:, 7:]).mean(axis=1) / 5.0, 0.0, 1.0),
    ], axis=1).astype(np.float32)
    gripper = np.clip(states[:, 6:7], 0.0, 1.0)
    progress = np.linspace(0.0, 1.0, len(rows), dtype=np.float32)[:, None]
    phase = np.concatenate([np.sin(2 * np.pi * progress), np.cos(2 * np.pi * progress)], axis=1)
    bias = np.ones((len(rows), 1), dtype=np.float32)
    features = np.concatenate([*parts, task, activity, gripper, progress, phase, bias], axis=1)
    if features.shape[1] != 48 or not np.all(np.isfinite(features)):
        raise ValueError(f"invalid feature tensor {features.shape}")
    return features


def make_arrays(
    episodes: dict[str, list[dict[str, Any]]],
    ids: list[str],
    stats: dict[str, list[float]],
    stage_to_id: dict[str, int],
    window: int,
    horizon: int,
) -> dict[str, np.ndarray]:
    buckets: dict[str, list[Any]] = defaultdict(list)
    action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
    action_std = np.asarray(stats["action_std"], dtype=np.float32)
    for episode_id in ids:
        rows = episodes[episode_id]
        features = episode_features(rows, stats)
        actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
        for index in range(window - 1, len(rows) - horizon):
            stage = rows[index]["stage"]
            future_stage = rows[index + horizon]["stage"]
            buckets["features"].append(features[index - window + 1:index + 1].T[:, :, None])
            buckets["stage"].append(stage_to_id[stage])
            buckets["next_stage"].append(stage_to_id[future_stage])
            buckets["sync"].append(float(stage.startswith("OVERLAP_")))
            buckets["done"].append(float(stage in {"SINGLE_DONE", "DUAL_DONE"}))
            target = actions[index + 1:index + horizon + 1]
            buckets["action"].append((target - action_mean) / action_std)
            buckets["current_action"].append(actions[index])
    return {
        "features": np.ascontiguousarray(np.stack(buckets["features"]), dtype=np.float32),
        "stage": np.asarray(buckets["stage"], dtype=np.int64),
        "next_stage": np.asarray(buckets["next_stage"], dtype=np.int64),
        "sync": np.asarray(buckets["sync"], dtype=np.float32)[:, None],
        "done": np.asarray(buckets["done"], dtype=np.float32)[:, None],
        "action": np.asarray(buckets["action"], dtype=np.float32),
        "current_action": np.asarray(buckets["current_action"], dtype=np.float32),
    }


def loader(values: dict[str, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(*(
        torch.from_numpy(values[key])
        for key in ("features", "stage", "next_stage", "sync", "done", "action")
    ))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def class_weights(labels: np.ndarray, count: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=count).astype(np.float64)
    weights = np.sqrt(np.maximum(counts.sum(), 1.0) / np.maximum(counts, 1.0))
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def train(
    model: X5BiSkillTCN,
    train_values: dict[str, np.ndarray],
    val_values: dict[str, np.ndarray],
    epochs: int,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    stage_weight = class_weights(train_values["stage"], model.config.stage_count)
    next_weight = class_weights(train_values["next_stage"], model.config.stage_count)
    train_loader = loader(train_values, batch_size, True)
    val_loader = loader(val_values, batch_size, False)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for features, stage, next_stage, sync, done, action in train_loader:
            outputs = model(features)
            corrupted = features + torch.randn_like(features) * 2.5
            ood_corrupt = model(corrupted)[4]
            loss = (
                F.cross_entropy(outputs[0], stage, weight=stage_weight)
                + 0.6 * F.cross_entropy(outputs[1], next_stage, weight=next_weight)
                + 0.25 * F.binary_cross_entropy_with_logits(outputs[2], sync)
                + 0.15 * F.binary_cross_entropy_with_logits(outputs[3], done)
                + 0.20 * F.binary_cross_entropy_with_logits(outputs[4], torch.zeros_like(outputs[4]))
                + 0.20 * F.binary_cross_entropy_with_logits(ood_corrupt, torch.ones_like(ood_corrupt))
                + 2.0 * F.smooth_l1_loss(outputs[5], action)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.item()) * features.shape[0]
            seen += features.shape[0]
        model.eval()
        val_total = 0.0
        val_seen = 0
        with torch.no_grad():
            for features, stage, next_stage, sync, done, action in val_loader:
                outputs = model(features)
                loss = (
                    F.cross_entropy(outputs[0], stage)
                    + 0.6 * F.cross_entropy(outputs[1], next_stage)
                    + 0.25 * F.binary_cross_entropy_with_logits(outputs[2], sync)
                    + 0.15 * F.binary_cross_entropy_with_logits(outputs[3], done)
                    + 2.0 * F.smooth_l1_loss(outputs[5], action)
                )
                val_total += float(loss.item()) * features.shape[0]
                val_seen += features.shape[0]
        train_loss = total / seen
        val_loss = val_total / val_seen
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("student training produced no checkpoint")
    return best_state, history


def evaluate(
    model: X5BiSkillTCN,
    values: dict[str, np.ndarray],
    stats: dict[str, list[float]],
) -> dict[str, float | int]:
    model.eval()
    with torch.no_grad():
        features = torch.from_numpy(values["features"])
        outputs = []
        for start in range(0, len(features), 512):
            batch = model(features[start:start + 512])
            outputs.append([part.cpu().numpy() for part in batch])
    merged = [np.concatenate([row[index] for row in outputs], axis=0) for index in range(6)]
    action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
    action_std = np.asarray(stats["action_std"], dtype=np.float32)
    learned = merged[5] * action_std + action_mean
    target = values["action"] * action_std + action_mean
    baseline = np.repeat(values["current_action"][:, None, :], learned.shape[1], axis=1)
    joint = np.asarray(JOINT_INDICES)
    learned_mae = float(np.abs(learned[:, :, joint] - target[:, :, joint]).mean())
    baseline_mae = float(np.abs(baseline[:, :, joint] - target[:, :, joint]).mean())
    corrupt = torch.from_numpy(values["features"][: min(1024, len(values["features"]))])
    with torch.no_grad():
        normal_ood = torch.sigmoid(model(corrupt)[4]).numpy()
        corrupt_ood = torch.sigmoid(model(corrupt + torch.randn_like(corrupt) * 2.5)[4]).numpy()
    return {
        "validation_windows": int(len(values["features"])),
        "stage_accuracy": float((merged[0].argmax(axis=1) == values["stage"]).mean()),
        "next_stage_accuracy": float((merged[1].argmax(axis=1) == values["next_stage"]).mean()),
        "sync_accuracy": float(((merged[2] >= 0) == (values["sync"] >= 0.5)).mean()),
        "done_accuracy": float(((merged[3] >= 0) == (values["done"] >= 0.5)).mean()),
        "ood_pair_accuracy": float(((normal_ood < 0.5).mean() + (corrupt_ood >= 0.5).mean()) / 2),
        "action_joint_mae_deg": learned_mae,
        "persistence_joint_mae_deg": baseline_mae,
        "action_improvement_fraction": float((baseline_mae - learned_mae) / baseline_mae),
        "action_gripper_mae": float(np.abs(learned[:, :, 6] - target[:, :, 6]).mean()),
    }


def export_student(model: X5BiSkillTCN, path: Path) -> float:
    model.eval()
    sample = torch.zeros(1, 48, model.config.window, 1, dtype=torch.float32)
    torch.onnx.export(
        model,
        (sample,),
        path,
        input_names=["features"],
        output_names=["stage_logits", "next_skill_logits", "sync_logit", "success_logit", "ood_logit", "action_chunk"],
        dynamic_axes=None,
        opset_version=11,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_outputs = [part.numpy() for part in model(sample)]
    ort_outputs = session.run(None, {"features": sample.numpy()})
    return max(float(np.max(np.abs(left - right))) for left, right in zip(torch_outputs, ort_outputs))


def teacher_fixture(
    episodes: dict[str, list[dict[str, Any]]],
    val_ids: list[str],
    teacher_root: Path,
    output: Path,
) -> dict[str, Any]:
    windows = []
    for episode_id in val_ids:
        rows = episodes[episode_id]
        states = np.asarray([row["observation_state"] for row in rows], dtype=np.float32)
        actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
        for index in range(7, len(rows) - 1, max(1, len(rows) // 4)):
            windows.append((states[index - 7:index + 1], actions[index - 7:index + 1]))
    windows = windows[:24]
    raw_states = np.stack([item[0] for item in windows]).astype(np.float32)
    raw_actions = np.stack([item[1] for item in windows]).astype(np.float32)
    payload: dict[str, np.ndarray] = {}
    records = {}
    for kind in ("tiny_act", "world_model"):
        checkpoint_path = teacher_root / kind / "seed_20260732" / "checkpoint.pt"
        onnx_path = teacher_root / kind / "seed_20260732" / "model.onnx"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        normalization = checkpoint["metadata"]["normalization"]
        state_mean = np.asarray(normalization["state_mean"], dtype=np.float32)
        state_std = np.asarray(normalization["state_std"], dtype=np.float32)
        action_mean = np.asarray(normalization["action_mean"], dtype=np.float32)
        action_std = np.asarray(normalization["action_std"], dtype=np.float32)
        states = (raw_states - state_mean) / state_std
        actions = (raw_actions - action_mean) / action_std
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        outputs = session.run(None, {"states": states, "actions_in": actions})
        payload[f"{kind}_states"] = states
        payload[f"{kind}_actions_in"] = actions
        for index, value in enumerate(outputs):
            payload[f"{kind}_output_{index}"] = np.asarray(value, dtype=np.float32)
        records[kind] = {
            "onnx_sha256": sha256_file(onnx_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "outputs": [list(value.shape) for value in outputs],
        }
    np.savez_compressed(output, **payload)
    return {"samples": len(windows), "sha256": sha256_file(output), "models": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--teacher-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing candidate: {output}")
    output.mkdir(parents=True)
    episodes = load_episodes(args.fixture.resolve())
    teacher_metrics = json.loads(args.teacher_metrics.read_text(encoding="utf-8"))
    train_ids = teacher_metrics["split"]["train_episode_ids"]
    val_ids = teacher_metrics["split"]["val_episode_ids"]
    stage_to_id = teacher_metrics["stage_to_id"]
    if set(train_ids) & set(val_ids):
        raise ValueError("episode split leakage")
    stats = fit_stats(episodes, train_ids)
    config = ModelConfig(hidden_channels=32, stage_count=len(stage_to_id))
    train_values = make_arrays(episodes, train_ids, stats, stage_to_id, config.window, config.action_horizon)
    val_values = make_arrays(episodes, val_ids, stats, stage_to_id, config.window, config.action_horizon)
    model = X5BiSkillTCN(config)
    best_state, history = train(model, train_values, val_values, args.epochs, args.batch_size)
    model.load_state_dict(best_state)
    metrics = evaluate(model, val_values, stats)
    checkpoint_path = output / "x5_biskill_tcn_fixture.pt"
    torch.save({
        "schema_version": "xrd-x5-biskill-fixture-checkpoint-v1",
        "created_at": utc_now(),
        "truthfulness": TRUTH,
        "seed": args.seed,
        "model_config": config.to_dict(),
        "feature_normalization": stats,
        "stage_to_id": stage_to_id,
        "train_episode_ids": train_ids,
        "val_episode_ids": val_ids,
        "model_state_dict": best_state,
    }, checkpoint_path)
    onnx_path = output / "x5_biskill_tcn_fixture.onnx"
    parity = export_student(model, onnx_path)
    sample_indices = np.linspace(0, len(val_values["features"]) - 1, 32, dtype=int)
    calibration = val_values["features"][sample_indices]
    calibration_dir = output / "calibration_data"
    calibration_dir.mkdir()
    calibration_records = []
    for index, sample in enumerate(calibration):
        path = calibration_dir / f"calib_{index:03d}.bin"
        path.write_bytes(np.ascontiguousarray(sample, dtype="<f4").tobytes())
        calibration_records.append({"file": path.name, "sha256": sha256_file(path)})
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    board_features = val_values["features"][sample_indices[:16]]
    per_sample_outputs = [
        session.run(None, {"features": board_features[index:index + 1]})
        for index in range(len(board_features))
    ]
    board_outputs = [
        np.concatenate([sample[output_index] for sample in per_sample_outputs], axis=0)
        for output_index in range(len(per_sample_outputs[0]))
    ]
    board_payload = {"features": board_features}
    for index, value in enumerate(board_outputs):
        board_payload[f"student_output_{index}"] = np.asarray(value, dtype=np.float32)
    board_fixture_path = output / "student_board_fixture.npz"
    np.savez_compressed(board_fixture_path, **board_payload)
    teacher_fixture_path = output / "teacher_board_fixture.npz"
    teacher_record = teacher_fixture(episodes, val_ids, args.teacher_root.resolve(), teacher_fixture_path)
    for kind in ("tiny_act", "world_model"):
        shutil.copy2(args.teacher_root / kind / "seed_20260732" / "model.onnx", output / f"{kind}_seed_20260732.onnx")
    accepted = (
        metrics["stage_accuracy"] >= 0.90
        and metrics["next_stage_accuracy"] >= 0.90
        and metrics["sync_accuracy"] >= 0.95
    )
    receipt = {
        "schema_version": "xrd-dual-arm-x5-shadow-candidate-v1",
        "created_at": utc_now(),
        "status": (
            "PC_STUDENT_ACCEPTED_STAGE_SKILL_ONLY_BPU_COMPILE_PENDING"
            if accepted
            else "PC_STUDENT_REJECTED_METRIC_GATE"
        ),
        "truthfulness": TRUTH,
        "training": {
            "seed": args.seed,
            "device": "cpu",
            "epochs": args.epochs,
            "history": history,
            "train_windows": int(len(train_values["features"])),
            "validation_windows": int(len(val_values["features"])),
            "episode_split_overlap": [],
        },
        "metrics": metrics,
        "capability_decisions": {
            "stage_classification": "PROMOTED_SHADOW_ONLY",
            "next_skill_classification": "PROMOTED_SHADOW_ONLY",
            "dual_arm_sync_classification": "PROMOTED_SHADOW_ONLY",
            "action_chunk": (
                "PROMOTED_SHADOW_ONLY"
                if metrics["action_improvement_fraction"] >= 0.0
                else "REJECTED_NOT_BETTER_THAN_PERSISTENCE"
            ),
            "action_chunk_consumed_by_runtime": False,
        },
        "student": {
            "model": "X5BiSkillTCN",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "contract": model.contract(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_sha256": sha256_file(onnx_path),
            "onnx_pytorch_max_abs_diff": parity,
            "mapper_target": "bayes-e",
        },
        "fixtures": {
            "source_sha256": sha256_file(args.fixture),
            "student_board_fixture_sha256": sha256_file(board_fixture_path),
            "teacher_board_fixture": teacher_record,
            "calibration": calibration_records,
        },
        "claim_boundary": (
            "All metrics are command-derived fixture replay only. The student and teachers are passive shadow advisors, "
            "not real-robot policies and never issue actuator commands."
        ),
    }
    write_json(output / "candidate_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "output": str(output), "metrics": metrics}, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
