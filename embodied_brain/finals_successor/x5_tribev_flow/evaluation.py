#!/usr/bin/env python3
"""Deterministic offline evaluation for the X5-TriBEV-Flow candidate.

This module is deliberately isolated from ROS, serial devices, F407 control,
TF, Nav2, and ``/cmd_vel``. Synthetic/replay metrics produced here validate
only the offline model and data pipeline; they are not evidence of real-world
navigation validity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .dataset import (
    GRID_HEIGHT,
    GRID_WIDTH,
    HISTORY_FRAMES,
    TRAJECTORY_TOKENS,
    TRIBEV_CHANNEL_NAMES,
    EpisodeRef,
    flatten_tribev_history,
    load_episode,
)
from .metrics import (
    binary_occupancy_metrics,
    expected_calibration_error,
    flow_endpoint_error,
    trajectory_distribution_metrics,
)

REPORT_SCHEMA_VERSION = "x5-tribev-flow-offline-evaluation.v1"
ABLATION_NAMES = (
    "full",
    "no_lidar",
    "no_depth",
    "no_vision",
    "t0_only",
    "reverse_temporal",
    "no_fused_validity",
)
CLAIM_BOUNDARY = (
    "This is an isolated offline evaluation. Synthetic and replay results "
    "validate only dataset, model, baseline, ablation, calibration, and export "
    "methodology. They do not establish real-world perception, navigation, "
    "collision avoidance, actuator authority, or finals-demo performance."
)


class Predictor(Protocol):
    """Minimal predictor contract used by the evaluator and test doubles."""

    def __call__(self, model_input: np.ndarray) -> Any:
        """Return the four TinyOccFlow outputs for one NCHW input."""


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it fully into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_episode_set(refs: Sequence[EpisodeRef]) -> str:
    digest = hashlib.sha256()
    for ref in sorted(refs, key=lambda item: (item.episode_id, str(item.path))):
        digest.update(ref.episode_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(ref.path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def deterministic_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize JSON deterministically and reject non-finite values."""

    normalized = _json_safe(payload)
    return (
        json.dumps(
            normalized,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def write_deterministic_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically write a deterministic report and return its SHA-256."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = deterministic_json_bytes(payload)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "path": str(destination),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    return probabilities / max(float(probabilities.sum()), 1e-12)


def _resize_nchw(
    array: np.ndarray,
    output_shape: tuple[int, int],
    *,
    mode: str,
) -> np.ndarray:
    """Use the same interpolation definitions as the training evaluator."""

    try:
        import torch
        import torch.nn.functional as functional
    except (ImportError, OSError) as exc:
        raise RuntimeError("PyTorch is required for TinyOccFlow target resizing") from exc

    tensor = torch.from_numpy(np.ascontiguousarray(array)).to(dtype=torch.float32)
    options: dict[str, Any] = {"size": output_shape, "mode": mode}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        options["align_corners"] = False
    with torch.inference_mode():
        resized = functional.interpolate(tensor, **options)
    return resized.cpu().numpy()


def _normalise_prediction(raw: Any) -> dict[str, np.ndarray]:
    if isinstance(raw, Mapping):
        values = {
            "future_occupancy": raw["future_occupancy"],
            "flow": raw["flow"],
            "dynamic_uncertainty": raw["dynamic_uncertainty"],
            "trajectory_logits": raw["trajectory_logits"],
        }
    elif all(
        hasattr(raw, name)
        for name in (
            "future_occupancy",
            "flow",
            "dynamic_uncertainty",
            "trajectory_logits",
        )
    ):
        values = {
            name: getattr(raw, name)
            for name in (
                "future_occupancy",
                "flow",
                "dynamic_uncertainty",
                "trajectory_logits",
            )
        }
    elif isinstance(raw, (tuple, list)) and len(raw) == 4:
        values = dict(
            zip(
                (
                    "future_occupancy",
                    "flow",
                    "dynamic_uncertainty",
                    "trajectory_logits",
                ),
                raw,
                strict=True,
            )
        )
    else:
        raise TypeError("predictor must return a mapping, named output, or four-item tuple")

    normalized: dict[str, np.ndarray] = {}
    expected = {
        "future_occupancy": (3, 64, 64),
        "flow": (6, 32, 32),
        "dynamic_uncertainty": (6, 64, 64),
        "trajectory_logits": (9,),
    }
    for name, value in values.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == len(expected[name]) + 1 and array.shape[0] == 1:
            array = array[0]
        if array.shape != expected[name]:
            raise ValueError(f"{name}.shape={array.shape}, expected {expected[name]}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")
        normalized[name] = np.ascontiguousarray(array)
    return normalized


class TorchCheckpointPredictor:
    """Load the current public TinyOccFlow model from a standard checkpoint."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu") -> None:
        try:
            import torch
        except (ImportError, OSError) as exc:
            raise RuntimeError("PyTorch is required to load a TinyOccFlow checkpoint") from exc

        from .models import TinyOccFlowStudent

        self.path = Path(checkpoint_path).expanduser().resolve()
        self.sha256 = sha256_file(self.path)
        self.device = torch.device(device)
        checkpoint = torch.load(self.path, map_location=self.device, weights_only=False)
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint
            else checkpoint
        )
        if not isinstance(state_dict, Mapping):
            raise ValueError("checkpoint does not contain a state_dict mapping")
        self.model = TinyOccFlowStudent().to(self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.metadata = {
            key: checkpoint[key]
            for key in (
                "epoch",
                "seed",
                "validation_loss",
                "shadow_only",
                "cmd_vel_authority",
            )
            if isinstance(checkpoint, Mapping) and key in checkpoint
        }

    def __call__(self, model_input: np.ndarray) -> Any:
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(model_input)).to(
            device=self.device,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            return self.model(tensor)


def apply_ablation(
    tribev_input: np.ndarray,
    sensor_validity: np.ndarray,
    name: str,
) -> np.ndarray:
    """Apply one deterministic causal-input ablation before model flattening."""

    if name not in ABLATION_NAMES:
        raise ValueError(f"unknown ablation {name!r}")
    frames = np.asarray(tribev_input, dtype=np.float32).copy()
    validity = np.asarray(sensor_validity, dtype=np.uint8).copy()
    expected_frames = (HISTORY_FRAMES, len(TRIBEV_CHANNEL_NAMES), 64, 64)
    if frames.shape != expected_frames:
        raise ValueError(f"tribev_input.shape={frames.shape}, expected={expected_frames}")
    if validity.shape != (HISTORY_FRAMES, 3):
        raise ValueError("sensor_validity must have shape [5,3]")

    if name == "reverse_temporal":
        frames = frames[::-1].copy()
    elif name == "t0_only":
        frames[:-1] = 0.0
    elif name == "no_fused_validity":
        frames[:, 6:8] = 0.0
    elif name in {"no_lidar", "no_depth", "no_vision"}:
        sensor_index = {"no_lidar": 0, "no_depth": 1, "no_vision": 2}[name]
        channels = {
            "no_lidar": (0, 1),
            "no_depth": (2, 3, 4),
            "no_vision": (5,),
        }[name]
        frames[:, channels] = 0.0
        validity[:, sensor_index] = 0
        frames[:, 6] = np.mean(validity, axis=1, dtype=np.float32)[:, None, None]
        frames[:, 7] = np.maximum.reduce(
            (frames[:, 0], frames[:, 2], frames[:, 3], frames[:, 4], frames[:, 5])
        )
    return flatten_tribev_history(frames)[None]


def _episode_targets(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays = record["arrays"]
    flow = arrays["future_flow_m"].reshape(1, 6, GRID_HEIGHT, GRID_WIDTH)
    dynamic = arrays["dynamic_mask"][None]
    return {
        "occupancy": arrays["future_occupancy"].astype(np.float32),
        "flow": _resize_nchw(flow, (32, 32), mode="bilinear")[0],
        "flow_mask": _resize_nchw(dynamic, (32, 32), mode="nearest")[0] > 0.5,
        "dynamic": arrays["dynamic_mask"].astype(np.float32),
        "uncertainty": arrays["uncertainty_target"].astype(np.float32),
        "trajectory": arrays["trajectory_soft_labels"].astype(np.float32),
    }


def _prediction_row(
    ref: EpisodeRef,
    record: Mapping[str, Any],
    prediction: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    targets = _episode_targets(record)
    occupancy_probability = 1.0 / (
        1.0 + np.exp(-np.clip(prediction["future_occupancy"], -40.0, 40.0))
    )
    score = float(np.quantile(np.abs(occupancy_probability - targets["occupancy"]), 0.95))
    validity = np.asarray(record["arrays"]["sensor_validity"], dtype=np.uint8)
    return {
        "episode_id": ref.episode_id,
        "scenario_id": ref.scenario_id,
        "source_kind": ref.source_kind,
        "has_dynamic": bool(np.any(targets["dynamic"] > 0.5)),
        "has_dropout": bool(np.any(validity == 0)),
        "prediction": prediction,
        "target": targets,
        "nonconformity_score": score,
    }


def _trajectory_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correctness: list[float] = []
    confidences: list[float] = []
    kls: list[float] = []
    by_token: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        logits = row["prediction"]["trajectory_logits"]
        target = row["target"]["trajectory"]
        metrics = trajectory_distribution_metrics(logits, target)
        hit = float(bool(metrics["top1_agreement"]))
        target_token = int(metrics["target_token"])
        correctness.append(hit)
        confidences.append(float(np.max(_softmax(logits))))
        kls.append(float(metrics["kl_target_to_prediction"]))
        by_token[target_token].append(hit)
    per_token = {
        str(token): {
            "episodes": len(hits),
            "top1_accuracy": float(np.mean(hits)),
        }
        for token, hits in sorted(by_token.items())
    }
    return {
        "top1_accuracy": float(np.mean(correctness)),
        "macro_top1_accuracy": float(
            np.mean([entry["top1_accuracy"] for entry in per_token.values()])
        ),
        "mean_kl_target_to_prediction": float(np.mean(kls)),
        "ece": expected_calibration_error(
            np.asarray(confidences),
            np.asarray(correctness),
        ),
        "represented_target_tokens": len(per_token),
        "per_target_token": per_token,
    }


def _aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"episodes": 0, "available": False}
    occupancy_prediction = np.stack(
        [row["prediction"]["future_occupancy"] for row in rows]
    )
    occupancy_target = np.stack([row["target"]["occupancy"] for row in rows])
    occupancy_horizons = []
    for horizon in range(occupancy_prediction.shape[1]):
        occupancy_horizons.append(
            {
                "horizon_index": horizon,
                **binary_occupancy_metrics(
                    occupancy_prediction[:, horizon],
                    occupancy_target[:, horizon],
                    from_logits=True,
                ),
            }
        )
    dynamic = binary_occupancy_metrics(
        np.stack([row["prediction"]["dynamic_uncertainty"][:3] for row in rows]),
        np.stack([row["target"]["dynamic"] for row in rows]),
        from_logits=True,
    )
    flow = flow_endpoint_error(
        np.stack([row["prediction"]["flow"] for row in rows]),
        np.stack([row["target"]["flow"] for row in rows]),
        valid_mask=np.stack([row["target"]["flow_mask"] for row in rows]),
    )
    uncertainty_probability = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                np.stack(
                    [row["prediction"]["dynamic_uncertainty"][3:] for row in rows]
                ),
                -40.0,
                40.0,
            )
        )
    )
    uncertainty_target = np.stack([row["target"]["uncertainty"] for row in rows])
    return {
        "episodes": len(rows),
        "available": True,
        "occupancy": {
            "horizons": occupancy_horizons,
            "mean_iou": float(np.mean([entry["iou"] for entry in occupancy_horizons])),
            "mean_f1": float(np.mean([entry["f1"] for entry in occupancy_horizons])),
        },
        "dynamic": dynamic,
        "flow": flow,
        "uncertainty_mae": float(
            np.mean(np.abs(uncertainty_probability - uncertainty_target))
        ),
        "trajectory": _trajectory_metrics(rows),
    }


def _scenario_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scenario_id"])].append(row)
    scenarios = {
        name: _aggregate_metrics(group)
        for name, group in sorted(grouped.items())
    }
    if not scenarios:
        return {"scenarios": {}, "macro": {"available": False}}
    macro = {
        "available": True,
        "scenario_count": len(scenarios),
        "occupancy_mean_iou": float(
            np.mean([entry["occupancy"]["mean_iou"] for entry in scenarios.values()])
        ),
        "occupancy_mean_f1": float(
            np.mean([entry["occupancy"]["mean_f1"] for entry in scenarios.values()])
        ),
        "dynamic_iou": float(
            np.mean([entry["dynamic"]["iou"] for entry in scenarios.values()])
        ),
        "dynamic_f1": float(
            np.mean([entry["dynamic"]["f1"] for entry in scenarios.values()])
        ),
        "flow_mean_epe": float(
            np.nanmean([entry["flow"]["mean_epe"] for entry in scenarios.values()])
        ),
        "flow_p95_epe": float(
            np.nanmean([entry["flow"]["p95_epe"] for entry in scenarios.values()])
        ),
        "trajectory_top1_accuracy": float(
            np.mean(
                [entry["trajectory"]["top1_accuracy"] for entry in scenarios.values()]
            )
        ),
        "trajectory_macro_top1_accuracy": float(
            np.mean(
                [
                    entry["trajectory"]["macro_top1_accuracy"]
                    for entry in scenarios.values()
                ]
            )
        ),
        "trajectory_mean_kl": float(
            np.mean(
                [
                    entry["trajectory"]["mean_kl_target_to_prediction"]
                    for entry in scenarios.values()
                ]
            )
        ),
    }
    return {"scenarios": scenarios, "macro": macro}


def _complete_metric_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scenario = _scenario_report(rows)
    return {
        "overall": _aggregate_metrics(rows),
        "by_scenario": scenario["scenarios"],
        "scenario_macro": scenario["macro"],
    }


def _baseline_occupancy_persistence(
    records: Sequence[tuple[EpisodeRef, Mapping[str, Any]]],
) -> dict[str, Any]:
    rows = []
    for ref, record in records:
        latest_fused = np.asarray(record["arrays"]["tribev_input"][-1, 7])
        prediction = np.repeat(latest_fused[None], 3, axis=0)
        target = np.asarray(record["arrays"]["future_occupancy"])
        rows.append(
            {
                "scenario_id": ref.scenario_id,
                "prediction": prediction,
                "target": target,
            }
        )

    def aggregate(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        prediction = np.stack([row["prediction"] for row in group])
        target = np.stack([row["target"] for row in group])
        horizons = [
            {
                "horizon_index": horizon,
                **binary_occupancy_metrics(
                    prediction[:, horizon],
                    target[:, horizon],
                    from_logits=False,
                ),
            }
            for horizon in range(3)
        ]
        return {
            "episodes": len(group),
            "mean_iou": float(np.mean([row["iou"] for row in horizons])),
            "mean_f1": float(np.mean([row["f1"] for row in horizons])),
            "horizons": horizons,
        }

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scenario_id"])].append(row)
    by_scenario = {
        name: aggregate(group)
        for name, group in sorted(grouped.items())
    }
    return {
        "definition": "repeat_latest_fused_occupancy_for_all_future_horizons",
        "overall": aggregate(rows),
        "by_scenario": by_scenario,
        "scenario_macro": {
            "mean_iou": float(
                np.mean([row["mean_iou"] for row in by_scenario.values()])
            ),
            "mean_f1": float(
                np.mean([row["mean_f1"] for row in by_scenario.values()])
            ),
        },
    }


def _baseline_zero_flow(
    records: Sequence[tuple[EpisodeRef, Mapping[str, Any]]],
) -> dict[str, Any]:
    rows = []
    for ref, record in records:
        targets = _episode_targets(record)
        rows.append(
            {
                "scenario_id": ref.scenario_id,
                "target": targets["flow"],
                "mask": targets["flow_mask"],
            }
        )

    def aggregate(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        target = np.stack([row["target"] for row in group])
        return flow_endpoint_error(
            np.zeros_like(target),
            target,
            valid_mask=np.stack([row["mask"] for row in group]),
        )

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scenario_id"])].append(row)
    by_scenario = {
        name: aggregate(group)
        for name, group in sorted(grouped.items())
    }
    return {
        "definition": "all_future_flow_vectors_are_zero",
        "overall": aggregate(rows),
        "by_scenario": by_scenario,
        "scenario_macro": {
            "mean_epe": float(
                np.nanmean([row["mean_epe"] for row in by_scenario.values()])
            ),
            "p95_epe": float(
                np.nanmean([row["p95_epe"] for row in by_scenario.values()])
            ),
        },
    }


def _trajectory_baseline_metrics(
    records: Sequence[tuple[EpisodeRef, Mapping[str, Any]]],
    probabilities: np.ndarray,
) -> dict[str, Any]:
    logits = np.log(np.maximum(probabilities, 1e-12)).astype(np.float32)
    rows = []
    for ref, record in records:
        target = np.asarray(record["arrays"]["trajectory_soft_labels"])
        metrics = trajectory_distribution_metrics(logits, target)
        rows.append(
            {
                "scenario_id": ref.scenario_id,
                "target_token": int(metrics["target_token"]),
                "hit": float(bool(metrics["top1_agreement"])),
                "kl": float(metrics["kl_target_to_prediction"]),
            }
        )

    def aggregate(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        by_token: dict[int, list[float]] = defaultdict(list)
        for row in group:
            by_token[int(row["target_token"])].append(float(row["hit"]))
        return {
            "episodes": len(group),
            "top1_accuracy": float(np.mean([row["hit"] for row in group])),
            "macro_top1_accuracy": float(
                np.mean([np.mean(hits) for hits in by_token.values()])
            ),
            "mean_kl_target_to_prediction": float(
                np.mean([row["kl"] for row in group])
            ),
            "represented_target_tokens": len(by_token),
        }

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scenario_id"])].append(row)
    by_scenario = {
        name: aggregate(group)
        for name, group in sorted(grouped.items())
    }
    return {
        "probabilities": probabilities,
        "overall": aggregate(rows),
        "by_scenario": by_scenario,
        "scenario_macro": {
            "top1_accuracy": float(
                np.mean([row["top1_accuracy"] for row in by_scenario.values()])
            ),
            "macro_top1_accuracy": float(
                np.mean(
                    [row["macro_top1_accuracy"] for row in by_scenario.values()]
                )
            ),
            "mean_kl_target_to_prediction": float(
                np.mean(
                    [
                        row["mean_kl_target_to_prediction"]
                        for row in by_scenario.values()
                    ]
                )
            ),
        },
    }


def _fit_empirical_prior(
    refs: Sequence[EpisodeRef],
) -> np.ndarray:
    labels = [
        np.asarray(load_episode(ref.path, validate=True)["arrays"]["trajectory_soft_labels"])
        for ref in refs
    ]
    if not labels:
        return np.full(TRAJECTORY_TOKENS, 1.0 / TRAJECTORY_TOKENS, dtype=np.float32)
    prior = np.mean(np.stack(labels), axis=0)
    return (prior / max(float(prior.sum()), 1e-12)).astype(np.float32)


def _extract_conformal_metadata(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    candidate: Any = payload.get("conformal", payload)
    if not isinstance(candidate, Mapping):
        return None
    q_hat = candidate.get("q_hat")
    alpha = candidate.get("alpha", 0.10)
    try:
        q_hat_value = float(q_hat)
        alpha_value = float(alpha)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(q_hat_value) or not 0.0 < alpha_value < 1.0:
        return None
    return {
        "q_hat": q_hat_value,
        "alpha": alpha_value,
        "calibration_episode_count": candidate.get("calibration_episode_count"),
        "exchangeability_scope": candidate.get("exchangeability_scope"),
    }


def load_conformal_metadata(path: str | Path | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load optional conformal metadata and its artifact hash."""

    if path is None:
        return None, None
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError("calibration metadata JSON root must be an object")
    metadata = _extract_conformal_metadata(payload)
    return metadata, {
        "path": str(source),
        "sha256": sha256_file(source),
        "recognized": metadata is not None,
    }


def evaluate_offline(
    refs: Sequence[EpisodeRef],
    predictor: Predictor,
    *,
    empirical_prior_refs: Sequence[EpisodeRef] | None = None,
    conformal_metadata: Mapping[str, Any] | None = None,
    artifact_hashes: Mapping[str, Any] | None = None,
    predictor_name: str = "tiny_occ_flow",
) -> dict[str, Any]:
    """Evaluate one predictor, all fixed ablations, and diagnostic baselines."""

    ordered_refs = sorted(refs, key=lambda ref: (ref.scenario_id, ref.episode_id))
    if not ordered_refs:
        raise ValueError("at least one evaluation episode is required")
    records = [
        (ref, load_episode(ref.path, validate=True))
        for ref in ordered_refs
    ]

    ablations: dict[str, Any] = {}
    full_rows: list[dict[str, Any]] = []
    for ablation in ABLATION_NAMES:
        rows: list[dict[str, Any]] = []
        for ref, record in records:
            arrays = record["arrays"]
            model_input = apply_ablation(
                arrays["tribev_input"],
                arrays["sensor_validity"],
                ablation,
            )
            prediction = _normalise_prediction(predictor(model_input))
            rows.append(_prediction_row(ref, record, prediction))
        ablations[ablation] = _complete_metric_report(rows)
        if ablation == "full":
            full_rows = rows

    fit_refs = (
        list(empirical_prior_refs)
        if empirical_prior_refs is not None
        else ordered_refs
    )
    prior_scope = (
        "independent_reference_split"
        if empirical_prior_refs is not None
        else "evaluation_labels_diagnostic_only"
    )
    empirical_prior = _fit_empirical_prior(fit_refs)
    straight = np.zeros(TRAJECTORY_TOKENS, dtype=np.float32)
    straight[TRAJECTORY_TOKENS // 2] = 1.0

    source_counts = Counter(ref.source_kind for ref in ordered_refs)
    subset_rows = {
        "dynamic": [row for row in full_rows if row["has_dynamic"]],
        "modality_dropout": [row for row in full_rows if row["has_dropout"]],
    }
    conformal = {
        "available": False,
        "reason": "no_recognized_calibration_metadata",
    }
    recognized = (
        _extract_conformal_metadata(conformal_metadata)
        if conformal_metadata is not None
        else None
    )
    if recognized is not None:
        scores = np.asarray(
            [row["nonconformity_score"] for row in full_rows],
            dtype=np.float64,
        )
        q_hat = float(recognized["q_hat"])
        coverage = float(np.mean(scores <= q_hat))
        target = 1.0 - float(recognized["alpha"])
        conformal = {
            "available": True,
            **recognized,
            "evaluation_episode_count": len(scores),
            "empirical_coverage": coverage,
            "nominal_coverage": target,
            "coverage_gap": coverage - target,
            "score_definition": (
                "per_episode_95th_percentile_absolute_future_occupancy_error"
            ),
        }

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mode": "isolated_offline_shadow_evaluation",
        "predictor": predictor_name,
        "episode_count": len(ordered_refs),
        "source_summary": dict(sorted(source_counts.items())),
        "dataset_validity": (
            "synthetic_only"
            if set(source_counts) == {"synthetic"}
            else "offline_mixed_or_non_synthetic_sources"
        ),
        "artifact_hashes": dict(artifact_hashes or {}),
        "evaluation_episode_set_sha256": _hash_episode_set(ordered_refs),
        "ablations": ablations,
        "baselines": {
            "occupancy_persistence": _baseline_occupancy_persistence(records),
            "zero_flow": _baseline_zero_flow(records),
            "empirical_prior_trajectory": {
                "fit_scope": prior_scope,
                **_trajectory_baseline_metrics(records, empirical_prior),
            },
            "always_straight_trajectory": {
                "straight_token": TRAJECTORY_TOKENS // 2,
                **_trajectory_baseline_metrics(records, straight),
            },
        },
        "subsets": {
            name: {
                "selection": (
                    "dynamic_mask_contains_positive_cell"
                    if name == "dynamic"
                    else "sensor_validity_contains_zero"
                ),
                **_complete_metric_report(rows),
            }
            for name, rows in subset_rows.items()
        },
        "conformal": conformal,
        "authority": {
            "shadow_only": True,
            "cmd_vel_authority": False,
            "ros_publishers": 0,
            "serial_or_f407_interfaces": 0,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return _json_safe(report)


def default_artifact_hashes(
    refs: Sequence[EpisodeRef],
    *,
    checkpoint_path: str | Path | None = None,
    calibration_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic hashes for the evaluator's direct artifacts."""

    hashes: dict[str, Any] = {
        "evaluation_module": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(__file__),
        },
        "episode_set": {
            "sha256": _hash_episode_set(refs),
            "episodes": len(refs),
        },
    }
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        hashes["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        }
    if calibration_artifact is not None:
        hashes["calibration_metadata"] = dict(calibration_artifact)
    if refs:
        common_root = Path(
            os.path.commonpath([str(ref.path.parent) for ref in refs])
        )
        manifest = common_root / "dataset_manifest.json"
        if manifest.is_file():
            hashes["dataset_manifest"] = {
                "path": str(manifest.resolve()),
                "sha256": sha256_file(manifest),
            }
    return hashes


__all__ = [
    "ABLATION_NAMES",
    "CLAIM_BOUNDARY",
    "Predictor",
    "REPORT_SCHEMA_VERSION",
    "TorchCheckpointPredictor",
    "apply_ablation",
    "default_artifact_hashes",
    "deterministic_json_bytes",
    "evaluate_offline",
    "load_conformal_metadata",
    "sha256_file",
    "write_deterministic_json",
]
