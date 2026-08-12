#!/usr/bin/env python3
"""CUDA smoke train and ONNX export for the fixed-shape X5 student.

Generated data validates only the software and export path.  The resulting
checkpoint is never a real robot policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

SUCCESSOR_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SUCCESSOR_ROOT / "models"))

from x5_biskill_tcn import ModelConfig, X5BiSkillTCN  # noqa: E402


STATUS = "SYNTHETIC_SMOKE_NOT_REAL_POLICY"
OUTPUT_NAMES = (
    "stage_logits",
    "next_skill_logits",
    "sync_logit",
    "success_logit",
    "ood_logit",
    "action_chunk",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_batch(
    config: ModelConfig,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    features = torch.randn(
        batch_size,
        config.input_channels,
        config.window,
        1,
        generator=generator,
    )
    phase_signal = features[:, 0, -1, 0]
    stage = ((phase_signal.abs() * 3).long() % config.stage_count).clamp_min(0)
    next_stage = (stage + 1) % config.stage_count
    sync = (features[:, 1:3].mean(dim=(1, 2, 3)) > 0).float().unsqueeze(1)
    success = (stage >= config.stage_count // 2).float().unsqueeze(1)
    ood = (features.abs().mean(dim=(1, 2, 3)) > 1.2).float().unsqueeze(1)
    base = features[:, : config.action_dim, -1, 0]
    horizon = torch.arange(config.action_horizon, dtype=torch.float32).view(1, -1, 1)
    action = base.unsqueeze(1) + 0.01 * horizon
    return (
        features.to(device),
        stage.to(device),
        next_stage.to(device),
        sync.to(device),
        success.to(device),
        ood.to(device),
        action.to(device),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for the 5090 synthetic smoke")
    device_name = torch.cuda.get_device_name(0)
    if "5090" not in device_name:
        raise SystemExit(f"RTX 5090 required, found {device_name}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    config = ModelConfig()
    model = X5BiSkillTCN(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history: list[float] = []
    model.train()
    for _ in range(args.steps):
        features, stage, next_stage, sync, success, ood, action = make_batch(
            config,
            args.batch_size,
            torch.device("cuda"),
            generator,
        )
        outputs = model(features)
        loss = (
            functional.cross_entropy(outputs[0], stage)
            + functional.cross_entropy(outputs[1], next_stage)
            + 0.2 * functional.binary_cross_entropy_with_logits(outputs[2], sync)
            + 0.2 * functional.binary_cross_entropy_with_logits(outputs[3], success)
            + 0.2 * functional.binary_cross_entropy_with_logits(outputs[4], ood)
            + functional.smooth_l1_loss(outputs[5], action)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "x5_biskill_tcn_synthetic_smoke.pt"
    torch.save(
        {
            "format": "xrd-x5-biskill-tcn-v1",
            "status": STATUS,
            "model_config": config.to_dict(),
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "seed": args.seed,
        },
        checkpoint,
    )
    model = model.cpu().eval()
    onnx_path = args.out_dir / "x5_biskill_tcn_synthetic_smoke.onnx"
    sample = torch.zeros(1, config.input_channels, config.window, 1, dtype=torch.float32)
    torch.onnx.export(
        model,
        (sample,),
        onnx_path,
        input_names=["features"],
        output_names=list(OUTPUT_NAMES),
        dynamic_axes=None,
        opset_version=11,
        do_constant_folding=True,
    )
    receipt = {
        "schema_version": "xrd-cloud5090-synthetic-smoke-v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": STATUS,
        "real_robot_policy": False,
        "real_episode_training": False,
        "synthetic_data_only": True,
        "gpu": device_name,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "loss_first": history[0],
        "loss_last": history[-1],
        "checkpoint_sha256": sha256_file(checkpoint),
        "onnx_sha256": sha256_file(onnx_path),
        "model_contract": model.contract(),
        "motion_authority": False,
        "execution_allowed": False,
        "actuator_commands_issued": 0,
    }
    receipt_path = args.out_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": STATUS, "receipt": str(receipt_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
